#!/usr/bin/env python3
"""Reachy Mini camera -> Cosmos3-Edge, native (no Live VLM WebUI, no browser).

Flow
    reachy_mini.ReachyMini().media.get_frame() -> resize -> JPEG -> base64
        -> POST to the Cosmos3-Edge shim's /v1/chat/completions (same endpoint the WebUI talks to)
        -> print the description to stdout

This is the "native" half of the Reachy Mini integration: deploy/06-reachy-mini-sdk.md's
first phase points the SDK's camera at Live VLM WebUI in a browser; this script instead talks
to the shim directly over HTTP, the same way nvr/bridge/porch_dad.py does for Frigate/Ring
clips, so the loop can run headless on whatever machine has the Reachy Mini SDK installed.

The shim is single-flight (deploy/05-serve-and-webui.md §2.4), so interval_seconds should stay
comfortably above one round trip or requests will queue up behind each other.
"""
from __future__ import annotations

import base64
import sys
import time
from pathlib import Path

import cv2
import requests
import yaml
from reachy_mini import ReachyMini

CFG_PATH = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).with_name("config.yaml"))
CFG = yaml.safe_load(Path(CFG_PATH).read_text())

SHIM = CFG["cosmos3_url"].rstrip("/")
FRAME_WIDTH = int(CFG.get("frame_width", 640))
INTERVAL = float(CFG.get("interval_seconds", 2.0))
MAX_TOKENS = int(CFG.get("max_tokens", 256))
TEMPERATURE = float(CFG.get("temperature", 0.0))
CONNECTION_MODE = CFG.get("connection_mode") or None
SYSTEM_PROMPT = CFG["system_prompt"]
USER_PROMPT = CFG["user_prompt"]


def encode_frame(frame) -> bytes:
    """frame is (height, width, 3) uint8 RGB, per the Reachy Mini SDK's mini.media.get_frame()."""
    h, w = frame.shape[:2]
    if w > FRAME_WIDTH:
        new_h = int(h * FRAME_WIDTH / w)
        frame = cv2.resize(frame, (FRAME_WIDTH, new_h), interpolation=cv2.INTER_AREA)
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def describe(jpeg: bytes) -> tuple[str, int]:
    payload = {
        "model": "cosmos3-edge",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}},
                {"type": "text", "text": USER_PROMPT},
            ]},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }
    t0 = time.time()
    r = requests.post(f"{SHIM}/v1/chat/completions", json=payload, timeout=300)
    ms = int((time.time() - t0) * 1000)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip(), ms


def main() -> None:
    kwargs = {"media_backend": "default"}
    if CONNECTION_MODE:
        kwargs["connection_mode"] = CONNECTION_MODE
    print(f"[cosmos-bridge] shim={SHIM} frame_width={FRAME_WIDTH} interval={INTERVAL}s", flush=True)
    with ReachyMini(**kwargs) as mini:
        while True:
            t_loop = time.time()
            try:
                frame = mini.media.get_frame()
                jpeg = encode_frame(frame)
                text, ms = describe(jpeg)
                print(f"[cosmos-bridge] {ms}ms :: {text}", flush=True)
            except Exception as e:
                print(f"[cosmos-bridge] frame skipped: {type(e).__name__}: {e}", flush=True)
            time.sleep(max(0.0, INTERVAL - (time.time() - t_loop)))


if __name__ == "__main__":
    main()
