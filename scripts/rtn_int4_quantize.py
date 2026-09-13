#!/usr/bin/env python3
"""Weight-only INT4 RTN quantization of the Cosmos3-Edge text tower.

Produces a checkpoint in the ModelOpt W4A16 on-disk format that
tensorrt-edgellm-export consumes:
  <linear>.weight        uint8  [N//2, K]   two int4 nibbles/byte (even row low, odd row high)
  <linear>.weight_scale  fp32   [N, K//128] per-output-channel, per-group scale

Scale convention verified against modelopt INT4_BLOCKWISE_WEIGHT_ONLY_CFG:
  scale = amax/7 ; q = round(w/scale).clamp(-8, 7)   (matches to ~4e-9)

Only the language-model linears are quantized. embed_tokens, norms, the visual
tower and the projector stay FP16 (mirrors the reference recipe's exclude_modules).
"""
import argparse
import json
import os
import re
import shutil

import torch
from safetensors.torch import load_file, save_file

GROUP = 128

# Text-tower linears in the *original* nvidia key schema.
QUANT_RE = re.compile(
    r"^layers\.\d+\.(self_attn\.(to_q|to_k|to_v|to_out)|mlp\.(up_proj|down_proj))\.weight$"
    r"|^lm_head\.weight$"
)


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
    args = ap.parse_args()
    os.makedirs(args.dst, exist_ok=True)

    shards = sorted(f for f in os.listdir(args.src) if f.endswith(".safetensors"))
    print(f"shards: {shards}")

    n_quant = 0
    rels = []
    for shard in shards:
        tensors = load_file(os.path.join(args.src, shard))
        out = {}
        for k, v in tensors.items():
            if QUANT_RE.match(k):
                packed, scale, rel = quantize_weight(v)
                out[k] = packed
                out[k.replace(".weight", ".weight_scale")] = scale
                rels.append(rel)
                n_quant += 1
                if n_quant <= 3 or n_quant % 40 == 0:
                    print(f"  [{n_quant}] {k}: {tuple(v.shape)} -> packed "
                          f"{tuple(packed.shape)} scale {tuple(scale.shape)} relerr {rel*100:.2f}%")
            else:
                out[k] = v
        save_file(out, os.path.join(args.dst, shard), metadata={"format": "pt"})
        print(f"wrote {shard}  ({len(out)} tensors)")

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
        for k in list(wm):
            if QUANT_RE.match(k):
                wm[k.replace(".weight", ".weight_scale")] = wm[k]
        json.dump(idx, open(idx_path, "w"), indent=2)
        print(f"index updated: {len(wm)} entries")

    quant_cfg = {
        "quant_algo": "W4A16_AWQ",
        "kv_cache_quant_algo": None,
        "group_size": GROUP,
        "has_zero_point": False,
        "pre_quant_scale": False,
        "exclude_modules": ["embed_tokens", "model.projector*", "model.visual*", "norm"],
    }
    json.dump({"quantization": quant_cfg},
              open(os.path.join(args.dst, "hf_quant_config.json"), "w"), indent=2)

    cfg_path = os.path.join(args.dst, "config.json")
    cfg = json.load(open(cfg_path))
    cfg["quantization_config"] = quant_cfg
    json.dump(cfg, open(cfg_path, "w"), indent=2)

    print(f"\nquantized {n_quant} linears; mean rel err {sum(rels)/len(rels)*100:.2f}%")
    print(f"output: {args.dst}")


if __name__ == "__main__":
    main()
