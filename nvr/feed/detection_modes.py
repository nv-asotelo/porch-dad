#!/usr/bin/env python3
"""Detection modes: a second, targeted look by the VLM after Frigate's first detection.

Frigate's detector knows COCO classes only, so every dog in a doggy daycare is "dog". A detection
mode asks the loaded VLM one specific question about that camera's frame - is there a Bernese
mountain dog, and where? - and the answer replaces the generic one-sentence description in that
camera's notification: a sentence that says where it is, with the model's box drawn on the frame.

Two steps, both in formats measured in vlmbench/ rather than invented here:
  1. the benchmark's yes/no question, asked with logprobs, so the gate is the model's own
     probability for "yes" instead of whatever words it chose;
  2. only on a yes, the grounding prompt in the loaded model's own dialect - bbox_2d for the Qwen
     family and Cosmos, box_2d for Gemma, <ref> for InternVL. LocateAnything answers both at once:
     its query is the phrase itself, and no box means no.
alert_policy's rule still holds: the prompts name the one thing the mode looks for and nothing
else.

Settings live in a small JSON file (STATE_PATH) that porch-feed, which runs the modes, writes and
frigate-notify reads, so the stock push for a camera's label stands down while a mode owns it.
"""
from __future__ import annotations

import io
import json
import math
import os
import re
import time
from pathlib import Path

import requests

STATE_PATH = Path(os.environ.get("DETECTION_MODES_STATE", "/home/orin/nvr/feed/detection_modes.json"))

MODES = {
    "bernese": {
        "name": "Bernese mountain dog",
        # Frigate labels that trigger the second look.
        "labels": ["dog"],
        # Engines whose verdicts may push or speak, matched against the shim's model id. Measured
        # on 35 frames of the doggy-daycare feed the Reachy watched on 2026-10-04: Qwen3-VL-2B said
        # yes on 4, three of them the Bernese with a tight box and one a curled-up husky at P 0.54;
        # Cosmos3-Edge v3 said yes on 24, boxing doodles and huskies. Any other engine's verdict is
        # still recorded, so it can be compared, but stays off the phone.
        "engines": ["qwen3-vl", "locateanything"],
        "min_confidence": 0.6,
        # The breed service's way (breed_service.py), used whenever Frigate is tracking dogs: score
        # each dog's box against the Swiss mountain dog family, the classes a blurry crop of a
        # Bernese is taken for. On the same daycare footage the Bernese scored 0.70-0.99 in all four
        # frames it was in and no other dog above 0.09, so 0.5 is the gate.
        "breeds": ["Bernese mountain dog", "EntleBucher", "Appenzeller", "Greater Swiss Mountain dog"],
        "breed_min_score": 0.5,
    },
}

# Per-mode settings. config.yaml's detection_modes: block seeds them; the command centre edits
# the JSON copy at runtime.
SETTINGS = {
    "cameras": [],          # where the mode is on
    "notify": True,         # push the verdict (with the boxed frame) instead of the stock description
    "speak": False,         # also say it on the Reachy Mini
    "cooldown_s": 300,      # per camera: a daycare full of dogs must not become continuous inference
    "min_confidence": 0.5,  # the model's own P(yes) needed to call it found
    # Another OpenAI-compatible engine for this mode, e.g. a second box serving LocateAnything;
    # empty means the shim's loaded model. One VLM fits beside the NVR, so this is the only way a
    # camera's mode gets an engine of its own. Set in config.yaml only: the runtime API cannot,
    # so the control plane can never be told to send camera frames somewhere new.
    "url": "",
}

NUM = re.compile(r"-?\d+(?:\.\d+)?")


# --------------------------------------------------------------------------- settings
def load_state(defaults: dict | None = None) -> dict:
    """Every mode's settings: SETTINGS and the mode's own, then config.yaml's, then the runtime JSON."""
    try:
        saved = json.loads(STATE_PATH.read_text())
    except (OSError, ValueError):
        saved = {}
    out = {}
    for mid, mode in MODES.items():
        s = dict(SETTINGS)
        s.update({k: v for k, v in mode.items() if k in SETTINGS})
        s.update((defaults or {}).get(mid) or {})
        s.update(saved.get(mid) or {})
        s["cameras"] = [str(c) for c in s.get("cameras") or []]
        # The engine URL comes from config.yaml alone, never from the runtime file (see SETTINGS).
        s["url"] = str(((defaults or {}).get(mid) or {}).get("url") or "")
        out[mid] = s
    return out


def save_state(state: dict) -> None:
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(STATE_PATH)


def owner(camera: str, label: str, defaults: dict | None = None) -> str | None:
    """The mode that has taken over this camera's label, if any."""
    for mid, s in load_state(defaults).items():
        if camera in s["cameras"] and label in MODES[mid]["labels"]:
            return mid
    return None


# --------------------------------------------------------------------------- the model's dialect
def family(model_id: str) -> str:
    m = (model_id or "").lower()
    if "locateanything" in m:
        return "locateanything"
    if "text only" in m or "nemotron" in m:
        return "text"
    if "gemma" in m:
        return "gemma"
    if "internvl" in m:
        return "internvl"
    return "qwen"   # Qwen3-VL, Qwen3.5, Cosmos3-Edge and Cosmos-Reason2 all answer bbox_2d


def presence_prompt(name: str) -> str:
    return f"Is there a {name} in this image? Answer with only yes or no."


def ground_prompt(name: str, fam: str) -> str:
    if fam == "gemma":
        return (f"Detect the {name} in this image. Reply with only a JSON object with \"box_2d\": "
                "[ymin, xmin, ymax, xmax], with coordinates from 0 to 1000.")
    if fam == "internvl":
        return ("Please provide the bounding box coordinate of the region this sentence describes: "
                f"<ref>the {name}</ref>")
    return (f"Locate the {name} in this image. Reply with only a JSON object with \"bbox_2d\": its "
            "bounding box [x1, y1, x2, y2], with coordinates from 0 to 1000.")


def parse_box(text: str) -> list[float] | None:
    """First box in an answer as [x1, y1, x2, y2] in 0-1 - same rules as vlmbench/score.py."""
    if not text:
        return None
    m = re.search(r'"?(bbox_2d|box_2d|bbox|box)"?\s*[:=]\s*\[+\s*([^\]]+)', text)
    key = m.group(1) if m else None
    nums = [float(x) for x in NUM.findall(m.group(2) if m else text)][:4]
    if len(nums) < 4:
        return None
    if key == "box_2d":
        nums = [nums[1], nums[0], nums[3], nums[2]]
    scale = 1000.0 if max(nums) > 1.5 else 1.0
    x1, y1, x2, y2 = (min(max(v / scale, 0.0), 1.0) for v in nums)
    if x2 <= x1 or y2 <= y1:
        return None
    return [round(v, 4) for v in (x1, y1, x2, y2)]


def p_yes(choice: dict) -> float | None:
    """The model's probability for "yes" at its first answer token, from top_logprobs."""
    content = ((choice.get("logprobs") or {}).get("content")) or []
    if not content:
        return None
    first = content[0]
    alts = first.get("top_logprobs") or [{"token": first.get("token"), "logprob": first.get("logprob")}]
    p = sum(math.exp(a["logprob"]) for a in alts
            if a.get("logprob") is not None and (a.get("token") or "").strip().lower() == "yes")
    return round(min(p, 1.0), 4)


# --------------------------------------------------------------------------- one look
def _chat(url: str, image: bytes, text: str, max_tokens: int, logprobs: bool, timeout: float) -> dict:
    import base64
    body = {
        "model": "detection-mode",
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(image).decode()}},
            {"type": "text", "text": text},
        ]}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if logprobs:
        body.update(logprobs=True, top_logprobs=5)
    r = requests.post(f"{url.rstrip('/')}/v1/chat/completions", json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]


def model_id(url: str) -> str:
    try:
        r = requests.get(f"{url.rstrip('/')}/v1/models", timeout=5)
        return ((r.json().get("data") or [{}])[0]).get("id") or ""
    except (requests.RequestException, ValueError):
        return ""


def check(url: str, image: bytes, mode: str, min_confidence: float = 0.5, timeout: float = 60) -> dict:
    """Ask the loaded model about one frame. Returns the verdict; never raises for a model error."""
    name = MODES[mode]["name"]
    mid = model_id(url)
    fam = family(mid)
    out = {"mode": mode, "engine": mid, "family": fam, "found": False, "box": None,
           "confidence": None, "answers": [], "error": None,
           "engine_trusted": any(e in mid.lower() for e in MODES[mode].get("engines") or [])}
    t0 = time.time()
    try:
        if fam == "text":
            out["error"] = "the loaded model reads text only"
        elif fam == "locateanything":
            # Its server takes a short phrase as the query and answers "summary\n[json boxes]".
            ans = _chat(url, image, name, 256, False, timeout)["message"]["content"]
            out["answers"].append(ans)
            m = re.search(r"\[.*\]", ans, re.S)
            boxes = [b.get("bbox_2d") for b in (json.loads(m.group(0)) if m else []) if b.get("bbox_2d")]
            out["box"] = parse_box(json.dumps({"bbox_2d": boxes[0]})) if boxes else None
            out["found"] = out["box"] is not None
        else:
            choice = _chat(url, image, presence_prompt(name), 4, True, timeout)
            ans = (choice["message"]["content"] or "").strip()
            out["answers"].append(ans)
            p = p_yes(choice)
            said_yes = ans.lower().startswith("yes")
            # A runtime without logprobs (the llama.cpp proxy) still answers; its word is all there is.
            out["confidence"] = p if p is not None else (1.0 if said_yes else 0.0)
            if out["confidence"] >= min_confidence:
                ans = _chat(url, image, ground_prompt(name, fam), 48, False, timeout)["message"]["content"]
                out["answers"].append(ans)
                out["box"] = parse_box(ans)
                out["found"] = True
    except (requests.RequestException, KeyError, IndexError, ValueError, TypeError) as e:
        out["error"] = f"{type(e).__name__}: {e}"
    out["ms"] = int((time.time() - t0) * 1000)
    return out


def check_breed(url: str, image: bytes, boxes: list[list[float]], mode: str, timeout: float = 30) -> dict:
    """Score the dogs Frigate is tracking (boxes in 0-1) with the breed service; the best box wins.

    Independent of the shim: this is how a mode gets a second engine on a board with room for one
    VLM. The verdict has the same shape as check()'s, with the family score as its confidence."""
    import base64
    m = MODES[mode]
    out = {"mode": mode, "engine": "ImageNet ViT-B/16 breed scores", "family": "breed", "found": False,
           "box": None, "confidence": None, "answers": [], "error": None, "engine_trusted": True,
           "dogs": len(boxes)}
    t0 = time.time()
    try:
        r = requests.post(f"{url.rstrip('/')}/score", timeout=timeout,
                          json={"image": base64.b64encode(image).decode(), "boxes": boxes, "classes": m["breeds"]})
        r.raise_for_status()
        j = r.json()
        out["answers"] = [f"{label} {p}" for label, p in j["top"]]
        if j["scores"]:
            k = max(range(len(j["scores"])), key=j["scores"].__getitem__)
            out["confidence"] = j["scores"][k]
            if out["confidence"] >= m.get("breed_min_score", 0.5):
                out["found"], out["box"] = True, [round(v, 4) for v in boxes[k]]
    except (requests.RequestException, KeyError, ValueError, TypeError) as e:
        out["error"] = f"{type(e).__name__}: {e}"
    out["ms"] = int((time.time() - t0) * 1000)
    return out


# --------------------------------------------------------------------------- what the user sees
def where(box: list[float] | None) -> str:
    if not box:
        return "somewhere in the frame"
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    h = "left" if cx < 0.36 else "right" if cx > 0.64 else ""
    v = "upper" if cy < 0.36 else "lower" if cy > 0.64 else ""
    if h and v:
        return f"in the {v} {h} of the frame"
    if h:
        return f"on the {h} of the frame"
    if v:
        return f"at the {'top' if v == 'upper' else 'bottom'} of the frame"
    return "in the middle of the frame"


def headline(mode: str, verdict: dict) -> str:
    conf = verdict.get("confidence")
    sure = f" ({conf:.0%} sure)" if conf is not None else ""
    return f"{MODES[mode]['name']} spotted {where(verdict.get('box'))}{sure}."


def annotate(image: bytes, box: list[float] | None, label: str) -> bytes:
    """The frame with the model's box and label drawn on it, as JPEG."""
    from PIL import Image, ImageDraw, ImageFont
    im = Image.open(io.BytesIO(image)).convert("RGB")
    if box:
        w, h = im.size
        d = ImageDraw.Draw(im)
        x1, y1, x2, y2 = box[0] * w, box[1] * h, box[2] * w, box[3] * h
        lw = max(3, w // 240)
        d.rectangle([x1, y1, x2, y2], outline=(255, 196, 0), width=lw)
        size = max(14, w // 48)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
        except OSError:
            font = ImageFont.load_default()
        tx, ty = x1, max(0, y1 - size - 2 * lw)
        tw = d.textlength(label, font=font)
        d.rectangle([tx, ty, tx + tw + 2 * lw, ty + size + 2 * lw], fill=(255, 196, 0))
        d.text((tx + lw, ty + lw // 2), label, fill=(0, 0, 0), font=font)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    return buf.getvalue()


def push(cfg: dict, title: str, body: str, img: bytes | None, click: str = "") -> None:
    """Send through the phone provider frigate-notify uses (its config.yaml), image attached."""
    provider = (cfg.get("provider") or "ntfy").lower()
    if provider == "ntfy":
        c = cfg["ntfy"]
        headers = {"Title": title, "Priority": "default", "Tags": "dog"}
        if click:
            headers["Click"] = click
        if c.get("token"):
            headers["Authorization"] = f"Bearer {c['token']}"
        topic = f"{c['server'].rstrip('/')}/{c['topic']}"
        if img:
            headers.update(Filename="snapshot.jpg", Message=body)
            requests.put(topic, data=img, headers=headers, timeout=30).raise_for_status()
        else:
            requests.post(topic, data=body.encode("utf-8"), headers=headers, timeout=30).raise_for_status()
    elif provider == "pushover":
        c = cfg["pushover"]
        files = {"attachment": ("snapshot.jpg", img, "image/jpeg")} if img else None
        requests.post("https://api.pushover.net/1/messages.json", timeout=30, files=files,
                      data={"token": c["api_token"], "user": c["user_key"], "title": title,
                            "message": body, "url": click}).raise_for_status()
    else:
        raise ValueError(f"unknown provider {provider!r}")
