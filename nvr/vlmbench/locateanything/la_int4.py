"""INT4 weight-only quantization of NVIDIA LocateAnything-3B for a Jetson Orin Nano.

LocateAnything ships as BF16 PyTorch code only (7.7 GB; no TensorRT path, and its Parallel Box
Decoding is not something TensorRT-Edge-LLM's autoregressive runtime can run). So it stays in
PyTorch and the weights shrink instead: every Linear in the Qwen2.5-3B text tower and in MoonViT
becomes a group-128 asymmetric INT4 weight served by PyTorch's own tinygemm kernel
(aten._weight_int4pack_mm, sm80+). The tied lm_head gets its own INT4 copy so each decode step
reads 0.16 GB of head instead of the 0.62 GB embedding; the embedding itself becomes per-row INT8
(0.31 GB) for the lookup. MoonViT's fc1 (in_features 4304) fits no supported group size and stays
BF16. 7.7 GB of BF16 becomes 2.2 GB - still too much for the Orin beside the NVR once PyTorch's
CUDA context and the first inference's activations are added (vlmbench/README.md has the numbers).

  quantize (workstation, needs the BF16 checkpoint):   python la_int4.py quantize <model dir> <out.safetensors>
  load (Orin, never materializes the BF16 weights):     model = load_quantized(<model dir>, <out.safetensors>)
"""
import json
import sys

import torch
import torch.nn as nn

GROUP = 128
INNER_K_TILES = 8


def quantize_weight(w, group=GROUP):
    """[N, K] float -> (uint8 [N, K//2] two nibbles per byte, scales_and_zeros [K//group, N, 2] bf16).

    tinygemm's convention: w ~= (q - 8) * scale + zero, q in [0, 15], with zero = min + 8 * scale.
    """
    w = w.to(torch.float32)
    n, k = w.shape
    g = w.reshape(n, k // group, group)
    wmin, wmax = g.amin(dim=-1, keepdim=True), g.amax(dim=-1, keepdim=True)
    scale = (wmax - wmin).clamp(min=1e-6) / 15
    zero = wmin + scale * 8
    q = ((g - wmin) / scale).round().clamp(0, 15).to(torch.int32).reshape(n, k)
    packed = (q[:, ::2] << 4 | q[:, 1::2]).to(torch.uint8)
    sz = torch.cat([scale.reshape(n, k // group, 1), zero.reshape(n, k // group, 1)], dim=2)
    deq = ((q.reshape(n, k // group, group).float() - 8) * scale + zero).reshape(n, k)
    rel = ((deq - w).abs().mean() / w.abs().mean().clamp(min=1e-12)).item()
    return packed, sz.transpose(0, 1).contiguous().to(torch.bfloat16), rel


class Int4Linear(nn.Module):
    """Drop-in nn.Linear replacement running aten._weight_int4pack_mm on BF16 activations."""

    def __init__(self, in_features, out_features, bias, group=GROUP, device=None):
        super().__init__()
        self.in_features, self.out_features, self.group = in_features, out_features, group
        # The kernel tiles N in eights. Only the lm_head (152,681-token vocabulary) needs padding:
        # its extra rows are zero weights, and forward() slices their logits off again.
        self.n_padded = -(-out_features // 8) * 8
        self.register_buffer("qweight", torch.empty(self.n_padded, in_features // 2, dtype=torch.uint8, device=device))
        self.register_buffer("scales_and_zeros",
                             torch.empty(in_features // group, self.n_padded, 2, dtype=torch.bfloat16, device=device))
        self.register_buffer("bias", torch.empty(out_features, dtype=torch.bfloat16, device=device) if bias else None)
        self._packed = None

    @property
    def weight(self):
        # Callers only ever read .weight.dtype (flash-attn's upcast check); the INT4 data is not a weight.
        return torch.empty(0, dtype=torch.bfloat16, device=self.scales_and_zeros.device)

    def pack(self):
        """Tile the nibbles for the kernel, on the GPU it will run on, and drop the portable copy."""
        self._packed = torch.ops.aten._convert_weight_to_int4pack(self.qweight.cuda(), INNER_K_TILES)
        self.qweight = None
        self.scales_and_zeros = self.scales_and_zeros.cuda()

    def forward(self, x):
        shape = x.shape
        y = torch.ops.aten._weight_int4pack_mm(x.reshape(-1, shape[-1]).to(torch.bfloat16), self._packed,
                                                self.group, self.scales_and_zeros)
        if self.n_padded != self.out_features:
            y = y[:, :self.out_features]
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*shape[:-1], self.out_features).to(x.dtype)


class Int8Embedding(nn.Module):
    """The token embedding as per-row INT8: 0.31 GB instead of 0.62 GB, dequantized row by row on lookup.

    On the Orin the CPU and the GPU share one 8 GB pool, so the BF16 table was a quarter of the whole
    model, and loading it made a 0.6 GB transient copy that alone was enough to cross the memory cap.
    """

    def __init__(self, num, dim, device=None):
        super().__init__()
        self.register_buffer("qweight8", torch.empty(num, dim, dtype=torch.int8, device=device))
        self.register_buffer("scale8", torch.empty(num, dtype=torch.bfloat16, device=device))

    @property
    def weight(self):
        return torch.empty(0, dtype=torch.bfloat16, device=self.scale8.device)

    def forward(self, ids):
        return (self.qweight8[ids].to(torch.bfloat16) * self.scale8[ids].unsqueeze(-1))


def quantize_rows_int8(w):
    w = w.to(torch.float32)
    scale = w.abs().amax(dim=1).clamp(min=1e-8) / 127.0
    q = torch.round(w / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q, scale.to(torch.bfloat16)


def targets(model):
    """(parent, attr, linear) for every Linear to quantize: LM blocks, MoonViT blocks, the projector."""
    out = []
    for name, mod in model.named_modules():
        for attr, child in mod.named_children():
            if not isinstance(child, nn.Linear):
                continue
            full = f"{name}.{attr}" if name else attr
            if full.endswith("lm_head"):
                continue  # tied to the embedding; quantized separately as its own copy
            if child.in_features % GROUP or child.out_features % 8:
                continue  # MoonViT fc1 (K=4304) and anything else the kernel cannot tile
            out.append((mod, attr, child, full))
    return out


def quantize(model_dir, out_path):
    from transformers import AutoModel
    model = AutoModel.from_pretrained(model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True,
                                      attn_implementation="sdpa")
    state, rels, plan = {}, [], []
    for parent, attr, lin, full in targets(model):
        packed, sz, rel = quantize_weight(lin.weight.data)
        state[f"{full}.qweight"], state[f"{full}.scales_and_zeros"] = packed, sz
        if lin.bias is not None:
            state[f"{full}.bias"] = lin.bias.data.to(torch.bfloat16)
        rels.append(rel)
        plan.append(full)
    emb = model.language_model.get_input_embeddings().weight.data
    pad = -emb.shape[0] % 8
    packed, sz, rel = quantize_weight(torch.cat([emb, emb.new_zeros(pad, emb.shape[1])]) if pad else emb)
    state["language_model.lm_head.qweight"], state["language_model.lm_head.scales_and_zeros"] = packed, sz
    q8, s8 = quantize_rows_int8(emb)
    state["language_model.model.embed_tokens.qweight8"], state["language_model.model.embed_tokens.scale8"] = q8, s8
    rel8 = ((q8.float() * s8.float()[:, None] - emb.float()).abs().mean() / emb.float().abs().mean()).item()
    print(f"embedding INT8 per row: rel err {rel8 * 100:.2f}%")
    print(f"quantized {len(plan)} linears + lm_head; mean rel err {sum(rels) / len(rels) * 100:.2f}%, "
          f"lm_head {rel * 100:.2f}%")
    # Everything else exactly as it was (parameters are already BF16; persistent buffers keep their
    # own dtype - RoPE frequencies in BF16 would quietly cost box precision). Non-persistent buffers
    # are rebuilt by the model's own __init__ on load, so they are not saved.
    quantized = set(plan) | {"language_model.lm_head"}
    for name, t in model.state_dict().items():
        if name.rsplit(".", 1)[0] in quantized or name == "language_model.model.embed_tokens.weight":
            continue
        state[name] = t
    from safetensors.torch import save_file
    save_file({k: v.contiguous() for k, v in state.items()}, out_path,
              metadata={"format": "pt", "plan": json.dumps(plan), "group": str(GROUP)})
    size = sum(v.numel() * v.element_size() for v in state.values())
    print(f"wrote {out_path}: {len(state)} tensors, {size / 2**30:.2f} GiB")


def _release_heap():
    """Hand freed heap back to the OS. glibc keeps it otherwise, and on the Orin that memory is the
    same pool the GPU allocates from."""
    import ctypes
    import gc
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def load_quantized(model_dir, qpath, device="cuda"):
    """Build the model with no weights, swap in Int4Linear, then fill it straight from the file."""
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    config.text_config._attn_implementation = "sdpa"
    # Each of the 36 layers builds its own float32 RoPE cos/sin table for all 32,768 positions:
    # 1.1 GiB of identical copies, more than the whole INT4 text tower. Build them for 2,048 (a
    # prompt here is ~350 tokens; Qwen2RotaryEmbedding regrows its table if a longer one comes)
    # and share one table between all layers.
    config.text_config.max_position_embeddings = min(config.text_config.max_position_embeddings, 2048)
    cls = get_class_from_dynamic_module(config.auto_map["AutoModel"], model_dir)
    with init_empty_weights(include_buffers=False):
        model = cls(config)
    layers = model.language_model.model.layers
    for layer in layers[1:]:
        layer.self_attn.rotary_emb = layers[0].self_attn.rotary_emb
    _release_heap()
    # Read on the CPU and copy each tensor over, rather than safe_open(device="cuda"): on Jetson the
    # direct route goes through pinned staging buffers that PyTorch keeps cached, and every byte of
    # those is charged to this process's memory too (measured: 2.4 GB charged for 1.8 GB of tensors).
    with safe_open(qpath, "pt", device="cpu") as f:
        plan = json.loads(f.metadata()["plan"])
        index = dict(model.named_modules())
        if "language_model.model.embed_tokens.qweight8" in f.keys():
            old_emb = model.language_model.model.embed_tokens
            model.language_model.model.embed_tokens = Int8Embedding(old_emb.num_embeddings, old_emb.embedding_dim,
                                                                    device="meta")
        for full in plan + ["language_model.lm_head"]:
            parent_name, attr = full.rsplit(".", 1)
            old = getattr(index[parent_name], attr)
            new = Int4Linear(old.in_features, old.out_features, old.bias is not None, device="meta")
            setattr(index[parent_name], attr, new)
        state = {}
        for k in f.keys():
            t = f.get_tensor(k)
            state[k] = t.to(device)
            del t
    _release_heap()
    model.load_state_dict(state, strict=False, assign=True)
    del state
    model.to(device)
    for mod in model.modules():
        if isinstance(mod, Int4Linear):
            mod.pack()
    leftover = [n for n, p in model.named_parameters() if p.is_meta] + [n for n, b in model.named_buffers() if b is not None and b.is_meta]
    if leftover:
        raise RuntimeError(f"weights never loaded: {leftover[:8]}")
    torch.cuda.empty_cache()
    _release_heap()
    return model.eval()


if __name__ == "__main__":
    if sys.argv[1] == "quantize":
        quantize(sys.argv[2], sys.argv[3])
