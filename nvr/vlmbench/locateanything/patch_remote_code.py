#!/usr/bin/env python3
"""Make a downloaded nvidia/LocateAnything-3B snapshot load for inference on transformers 4.57.1.

NVIDIA's model code is not redistributed here (its license is non-commercial); this applies the two
edits it needs to a copy you download yourself, in place and idempotently:

1. processing_locateanything.py imports lmdb, cv2 and decord at module level. They serve LMDB
   datasets and video only, never image inference, and decord has no aarch64 wheel for the Orin.
   transformers refuses to load remote code whose top-level imports are missing, so they become
   optional.
2. The attention-implementation override has to accept the extra keyword argument newer
   transformers passes (allow_all_kernels); harmless on 4.57.1.

  hf download nvidia/LocateAnything-3B --local-dir <dir>   (then copy files out of the symlinked
  python patch_remote_code.py <dir>                          blob store if hf left symlinks)
"""
import pathlib
import sys

root = pathlib.Path(sys.argv[1])

proc = root / "processing_locateanything.py"
s = proc.read_text()
old = "import lmdb\nimport cv2\nimport pickle\nimport decord\n"
if old in s:
    s = s.replace(old, "import pickle\n"
                       "try:\n    import lmdb\nexcept ImportError:\n    lmdb = None\n"
                       "try:\n    import cv2\nexcept ImportError:\n    cv2 = None\n"
                       "try:\n    import decord\nexcept ImportError:\n    decord = None\n")
    proc.write_text(s)
    print("patched", proc.name)

for name in ("modeling_locateanything.py", "modeling_qwen2.py"):
    f = root / name
    s = f.read_text()
    n = s.replace("def _check_and_adjust_attn_implementation(self, attn_implementation, is_init_check=False):",
                  "def _check_and_adjust_attn_implementation(self, attn_implementation, is_init_check=False, **kwargs):")
    n = n.replace("return super()._check_and_adjust_attn_implementation(attn_implementation, is_init_check)",
                  "return super()._check_and_adjust_attn_implementation(attn_implementation, is_init_check, **kwargs)")
    if n != s:
        f.write_text(n)
        print("patched", name)
