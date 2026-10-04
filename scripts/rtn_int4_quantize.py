#!/usr/bin/env python3
"""Weight-only INT4 RTN quantization of a VLM's text tower (Cosmos3-Edge by default).

Produces a checkpoint in the ModelOpt W4A16 on-disk format that
tensorrt-edgellm-export consumes:
  <linear>.weight        uint8  [N//2, K]   two int4 nibbles/byte (even row low, odd row high)
  <linear>.weight_scale  fp32   [N, K//128] per-output-channel, per-group scale

Scale convention verified against modelopt INT4_BLOCKWISE_WEIGHT_ONLY_CFG:
  scale = amax/7 ; q = round(w/scale).clamp(-8, 7)   (matches to ~4e-9)

Only the language-model linears are quantized. embed_tokens, norms, the visual
tower and the projector stay FP16 (mirrors the reference recipe's exclude_modules).

--preset picks the key schema, so the peer VLMs benchmarked against Cosmos3-Edge get
exactly the same recipe. A preset with tie_embed names the embedding a tied lm_head
shares: that head is written out as its own quantized lm_head and the configs untied,
because Cosmos3-Edge's lm_head is quantized too and an FP16 head the size of a 150-250k
vocabulary would cost more memory and bandwidth than the whole INT4 decoder.
"""
import argparse
import json
import os
import re
import shutil

import torch
from safetensors.torch import load_file, save_file

GROUP = 128

# Text-tower linears per key schema. cosmos3 is the *original* nvidia schema.
_QWEN_LAYER = r"(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
PRESETS = {
    "cosmos3": {
        "quant_re": r"^layers\.\d+\.(self_attn\.(to_q|to_k|to_v|to_out)|mlp\.(up_proj|down_proj))\.weight$"
                    r"|^lm_head\.weight$",
        "exclude": ["embed_tokens", "model.projector*", "model.visual*", "norm"],
    },
    # Qwen3-VL-2B and Cosmos-Reason2-2B (a Qwen3-VL-2B post-train).
    "qwen3_vl": {
        "quant_re": rf"^model\.language_model\.layers\.\d+\.{_QWEN_LAYER}\.weight$|^lm_head\.weight$",
        "exclude": ["embed_tokens", "model.visual*", "norm"],
        "tie_embed": "model.language_model.embed_tokens.weight",
    },
    "internvl3_5": {
        "quant_re": rf"^language_model\.model\.layers\.\d+\.{_QWEN_LAYER}\.weight$|^language_model\.lm_head\.weight$",
        "exclude": ["embed_tokens", "vision_tower*", "multi_modal_projector*", "norm"],
    },
    # Hybrid Gated-DeltaNet + attention. in_proj_a / in_proj_b are 16 rows, below the
    # 64-row INT4 minimum, and the MTP head is not exported; all three stay FP16.
    "qwen3_5": {
        "quant_re": r"^model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj"
                    r"|linear_attn\.(in_proj_qkv|in_proj_z|out_proj))\.weight$|^lm_head\.weight$",
        "exclude": ["embed_tokens", "model.visual*", "mtp*", "norm"],
        "tie_embed": "model.language_model.embed_tokens.weight",
    },
}


def quantize_weight(w: torch.Tensor):
    """RTN-quantize [N, K] -> (packed uint8 [N//2, K], scale fp32 [N, K//GROUP])."""
    w32 = w.to(torch.float32)
    N, K = w32.shape
    assert N % 2 == 0, f"need even N for nibble packing, got {N}"
    assert K % GROUP == 0, f"need K %% {GROUP} == 0, got {K}"

    g = w32.reshape(N, K // GROUP, GROUP)
    amax = g.abs().amax(dim=-1, keepdim=True)
    amax = torch.clamp(amax, min=1e-8)

    # MSE-optimal clipping: search a per-group scale multiplier using only the
    # weights (no activations / no calibration data). Plain RTN uses alpha=1.0;
    # shrinking the range trades clipping error for finer resolution and is a
    # strict improvement on outlier-heavy LLM weights.
    best_err = None
    best_scale = None
    for alpha in torch.linspace(0.55, 1.0, 19).tolist():
        s_try = (amax * alpha) / 7.0
        q_try = torch.round(g / s_try).clamp(-8, 7)
        err = ((q_try * s_try - g) ** 2).sum(dim=-1, keepdim=True)
        if best_err is None:
            best_err, best_scale = err, s_try.clone()
        else:
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_scale = torch.where(better, s_try, best_scale)
    scale = best_scale
    q = torch.round(g / scale).clamp(-8, 7).to(torch.int8).reshape(N, K)

    # verify dequant round-trip before packing
    deq = (q.reshape(N, K // GROUP, GROUP).to(torch.float32)
           * scale).reshape(N, K)
    rel = ((deq - w32).abs().mean() / w32.abs().mean().clamp(min=1e-12)).item()

    qi = q.to(torch.int16) & 0xF
    packed = ((qi[1::2] << 4) | qi[0::2]).to(torch.uint8)

    # unpack must reproduce q exactly (mirrors loader.py _unpack_awq_prepacked)
    u16 = packed.to(torch.int16) & 0xFF
    back = torch.zeros(N, K, dtype=torch.int16)
    back[0::2] = u16 & 0xF
    back[1::2] = (u16 >> 4) & 0xF
    assert torch.equal(back, qi), "nibble pack/unpack round-trip failed"

    return packed, scale.squeeze(-1).to(torch.float32), rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="cosmos3")
    args = ap.parse_args()
    preset = PRESETS[args.preset]
    quant_re = re.compile(preset["quant_re"])
    tie_embed = preset.get("tie_embed")
    cfg = json.load(open(os.path.join(args.src, "config.json")))
    tied = bool(tie_embed) and (cfg.get("tie_word_embeddings")
                                or cfg.get("text_config", {}).get("tie_word_embeddings"))
    os.makedirs(args.dst, exist_ok=True)

    shards = sorted(f for f in os.listdir(args.src) if f.endswith(".safetensors"))
    print(f"shards: {shards}")

    n_quant = 0
    rels = []
    head_shard = None
    for shard in shards:
        tensors = load_file(os.path.join(args.src, shard))
        if tied and tie_embed in tensors:
            # The tied head becomes a real one, quantized like every other linear.
            tensors["lm_head.weight"] = tensors[tie_embed].clone()
            head_shard = shard
        out = {}
        for k, v in tensors.items():
            if quant_re.match(k):
                n, kk = v.shape
                if n % 64 or kk % 64:
                    # TensorRT-Edge-LLM would silently build such a linear in FP16; say so.
                    print(f"  SKIP {k}: {tuple(v.shape)} not 64-aligned, stays FP16")
                    out[k] = v
                    continue
                packed, scale, rel = quantize_weight(v)
                out[k] = packed
                out[k.replace(".weight", ".weight_scale")] = scale
                rels.append(rel)
                n_quant += 1
                if n_quant <= 3 or n_quant % 40 == 0 or k.endswith("lm_head.weight"):
                    print(f"  [{n_quant}] {k}: {tuple(v.shape)} -> packed "
                          f"{tuple(packed.shape)} scale {tuple(scale.shape)} relerr {rel*100:.2f}%")
            else:
                out[k] = v
        save_file(out, os.path.join(args.dst, shard), metadata={"format": "pt"})
        print(f"wrote {shard}  ({len(out)} tensors)")
    if tied and head_shard is None:
        raise SystemExit(f"tied embedding {tie_embed} not found in any shard")

    # copy non-weight files
    for f in os.listdir(args.src):
        s = os.path.join(args.src, f)
        if f.endswith(".safetensors"):
            continue
        d = os.path.join(args.dst, f)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)

    # index: add weight_scale entries alongside their weight
    idx_path = os.path.join(args.dst, "model.safetensors.index.json")
    if os.path.isfile(idx_path):
        idx = json.load(open(idx_path))
        wm = idx["weight_map"]
        if head_shard:
            wm["lm_head.weight"] = head_shard
        for k in list(wm):
            if quant_re.match(k):
                wm[k.replace(".weight", ".weight_scale")] = wm[k]
        json.dump(idx, open(idx_path, "w"), indent=2)
        print(f"index updated: {len(wm)} entries")

    quant_cfg = {
        "quant_algo": "W4A16_AWQ",
        "kv_cache_quant_algo": None,
        "group_size": GROUP,
        "has_zero_point": False,
        "pre_quant_scale": False,
        "exclude_modules": preset["exclude"],
    }
    json.dump({"quantization": quant_cfg},
              open(os.path.join(args.dst, "hf_quant_config.json"), "w"), indent=2)

    cfg_path = os.path.join(args.dst, "config.json")
    cfg["quantization_config"] = quant_cfg
    if tied:
        for c in (cfg, cfg.get("text_config")):
            if c is not None and "tie_word_embeddings" in c:
                c["tie_word_embeddings"] = False
    json.dump(cfg, open(cfg_path, "w"), indent=2)

    print(f"\nquantized {n_quant} linears; mean rel err {sum(rels)/len(rels)*100:.2f}%")
    print(f"output: {args.dst}")


if __name__ == "__main__":
    main()
