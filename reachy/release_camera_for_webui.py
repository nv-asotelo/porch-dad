#!/usr/bin/env python3
"""Hold the Reachy Mini daemon off the camera so a browser (Live VLM WebUI) can open it directly.

By default the reachy_mini daemon owns the camera device. media_backend="no_media" tells the
daemon to release camera (and audio) hardware for as long as this process's `with` block is open;
the hardware is re-acquired automatically when it exits (see the Reachy Mini SDK docs, "Media
Backend Options" / "Disabling Media"). While this script is running, the camera shows up as a
normal OS video device that a browser tab pointed at Live VLM WebUI (systemd/live-vlm-webui.service,
port 8090) can select like any other webcam — no bridging code needed, because the WebUI already
talks to the Cosmos3-Edge shim; only the camera source changes.

Run this on the machine physically connected to Reachy Mini and open the browser on that same
machine (deploy/05-serve-and-webui.md notes browsers generally restrict camera access to secure
contexts — HTTPS or localhost — so cross-machine HTTP access to the camera picker may be blocked
by the browser, not by anything here).
"""
import signal
import sys

from reachy_mini import ReachyMini


def main() -> None:
    print("[release-camera] releasing camera/audio hardware for direct OS access "
          "(Ctrl+C to give it back to the daemon) ...", flush=True)
    with ReachyMini(media_backend="no_media"):
        print("[release-camera] released. Open Live VLM WebUI and select the Reachy Mini camera.",
              flush=True)
        signal.pause()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
