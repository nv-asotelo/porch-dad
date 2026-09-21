#!/usr/bin/env python3
"""Read-only checks for the pinned Cosmos3 tokenizer and generated sidecars.

The build preflight accepts only the known, repairable missing media/RoPE fields.
Serving requires an already prepared, normalized bundle for the exact profile.
No checkpoint, engine or existing sidecar is written by this helper.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

import repair_cosmos_chat_template as chat_repair
import repair_cosmos_runtime_config as runtime_repair


# Official tokenizer bytes recorded in results/model-download.json at the same pin.
TOKENIZER_SHA256 = "4dc692a99dca6d92a44e9529ffb53541eb79bb260eec7281bc51268c05d7311c"


def source_tokenizer(model):
    originals = chat_repair.validate_source(model)
    if runtime_repair.sha256((model / "tokenizer.json").read_bytes()) != TOKENIZER_SHA256:
        raise ValueError("Source tokenizer.json SHA256 differs from the verified official pin")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True, trust_remote_code=False)
    if tokenizer.chat_template != originals["chat_template.jinja"].decode():
        raise ValueError("Loaded tokenizer does not use the original pinned chat_template.jinja")
    for token_id, token in chat_repair.SPECIAL_TOKENS.values():
        if tokenizer.encode(token, add_special_tokens=False) != [token_id]:
            raise ValueError(f"Tokenizer does not encode {token} as the pinned special token {token_id}")
    return tokenizer


def single_user_prompt(data, content, thinking):
    """Reconstruct the pinned C++ formatter for the UI's single user message.

    image_url becomes image in the public Python request adapter. Content order,
    ASCII trimming, default system text and thinking overrides follow tokenizer.cpp.
    This is a pre-expansion prompt check; visual pad expansion happens at runtime.
    """
    def role_text(role, field):
        return (role.get(field + "_thinking") if thinking else "") or role.get(field, "")

    def text(role, value):
        return value.strip(" \t\n\r\f\v") if role.get("trim_content") or data.get("trim_content") else value

    result = data.get("prompt_prefix", data.get("global_prefix", ""))
    system = data.get("default_system_prompt", "")
    if system:
        role = data["roles"]["system"]
        result += role_text(role, "prefix") + text(role, system) + role_text(role, "suffix")
    role = data["roles"]["user"]
    result += role_text(role, "prefix")
    parts = [{"type": "text", "text": content}] if isinstance(content, str) else content
    for part in parts:
        kind = part["type"]
        result += (text(role, part["text"]) if kind == "text" else
                   data["content_types"]["image" if kind == "image_url" else kind]["format"])
    result += role_text(role, "suffix")
    result += (data.get("generation_prompt_thinking") if thinking else "") or data["generation_prompt"]
    return result


def validate_chat(data, tokenizer, *, require_normalized=False):
    normalized, additions, _ = chat_repair.plan_repair(data)
    # Retain role validation beyond the single-user UI's currently used roles.
    for name in ("system", "user", "assistant"):
        role = normalized["roles"][name]
        if f"<|im_start|>{name}\n" not in role["prefix"] or role["suffix"] != "<|im_end|>\n":
            raise ValueError(f"Invalid or generic Cosmos3 {name} role")
    image_url = {"type": "image_url", "image_url": {"url": "fixture://not-loaded"}}
    image = {"type": "image"}
    text = {"type": "text", "text": "  Describe this image.\n"}
    probes = [
        "  Text-only probe.\n", [text], [text, image_url],  # Actual UI order and schema.
        [image, text], [{"type": "video"}, text], [image, text, image_url],
    ]
    for content in probes:
        for thinking in (False, True):
            messages = [{"role": "user", "content": content}]
            options = dict(add_generation_prompt=True, enable_thinking=thinking, add_vision_id=False)
            expected = tokenizer.apply_chat_template(messages, tokenize=False, **options)
            actual = single_user_prompt(normalized, content, thinking)
            if actual != expected:
                raise ValueError(f"Processed prompt differs from pinned tokenizer Jinja (thinking={thinking}, content={content!r})")
            if tokenizer.encode(actual, add_special_tokens=False) != tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, **options):
                raise ValueError("Processed prompt token IDs differ from pinned tokenizer Jinja")
    if require_normalized and additions:
        raise ValueError("Cached media formats need normalization; run scripts/build_model_cache.py before serving")
    return additions


def validate_bundle(model, bundle, source, tokenizer, *, require_ready, is_ready):
    """Check only the selected profile; a build may create its missing artifacts."""
    if require_ready and (not bundle.is_dir() or not is_ready()):
        raise ValueError(f"No ready engine bundle for this exact profile: {bundle}. Run scripts/build_model_cache.py first")
    processed = bundle / "processed_chat_template.json"
    visual = bundle / "visual/config.json"
    changes = {}
    if processed.is_file():
        changes["media"] = validate_chat(json.loads(processed.read_text()), tokenizer,
                                         require_normalized=require_ready)
    elif require_ready:
        raise ValueError("Ready bundle lacks processed_chat_template.json; run scripts/build_model_cache.py")
    if visual.is_file():
        _, changes["rope"] = runtime_repair.plan_repair(source, json.loads(visual.read_text()))
        if require_ready and changes["rope"]:
            raise ValueError("Cached visual RoPE aliases need normalization; run scripts/build_model_cache.py before serving")
    elif require_ready:
        raise ValueError("Ready bundle lacks visual/config.json; run scripts/build_model_cache.py")
    return changes


def validate(model, upstream, cache, *, max_input_len, max_kv_capacity, preflight_only,
             build_options=None):
    model, upstream, cache = (Path(value).resolve() for value in (model, upstream, cache))
    tokenizer = source_tokenizer(model)
    source = json.loads((model / "config.json").read_text())
    processed = model / "processed_chat_template.json"
    if processed.is_file():
        validate_chat(json.loads(processed.read_text()), tokenizer)
    else:
        # Verify exactly what this pinned builder will generate, including its
        # known empty content_types, without adding a file to the checkpoint.
        helper_path = upstream / "experimental/builder/core/artifacts/chat_template.py"
        spec = importlib.util.spec_from_file_location("cosmos_chat_preflight", helper_path)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        with tempfile.TemporaryDirectory(prefix="cosmos-chat-preflight-") as temporary:
            helper.write_processed_chat_template(str(model), temporary)
            validate_chat(json.loads((Path(temporary) / "processed_chat_template.json").read_text()), tokenizer)
    sys.path.insert(0, str(upstream))
    from experimental.server.runtime import engine_build
    options = build_options or engine_build.BuildOptions(max_input_len=max_input_len,
                                       max_kv_cache_capacity=max_kv_capacity, max_batch_size=1)
    if (options.max_input_len != max_input_len or options.max_kv_cache_capacity != max_kv_capacity
            or options.max_batch_size != 1):
        raise ValueError("Explicit build options must match the preflight input/KV/batch profile")
    bundle = Path(engine_build.bundle_cache_path(str(model), str(cache), options))
    changes = validate_bundle(model, bundle, source, tokenizer, require_ready=not preflight_only,
                              is_ready=lambda: engine_build._is_ready(str(model), str(bundle), options))
    print("Pinned Cosmos3 Jinja and token IDs match 12 single-user text/media/thinking probes; "
          + ("build preflight accepts known repairs." if preflight_only else "selected serving bundle is ready and normalized."), flush=True)
    return {"bundle": str(bundle), "repairable_additions": changes}
