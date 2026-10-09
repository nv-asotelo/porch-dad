#!/usr/bin/env python3
"""Serve the VITURE Luma Ultra's camera to everything on the Jetson from one capture, the way
nvr/reachy/reachy_mjpeg_bridge.py serves the Reachy Mini's:

    /mjpeg      multipart MJPEG, for browser previews (Live Vision relays it under /luma/)
    /still.jpg  the newest frame, 503 when there is no fresh one
    /healthz    state, and WHY it is in that state

The glasses' camera is an ordinary UVC webcam - a Sonix "USB 2.0 Camera", 0c45:636b - behind the
glasses' own USB hub, beside VITURE's 35ca devices. It sends MJPEG, which is passed through as it
comes, not re-encoded. Its /dev/videoN depends on plug order, so the bridge finds it by that
neighbourhood (--device names one outright).

The camera runs only while something asks for it: a page polling /healthz, an open /mjpeg, a
/still.jpg. It stops --idle seconds after the last request. With the glasses unplugged the bridge
says so, and it picks the camera up again when they come back.

Needs only Python's standard library and gst-launch-1.0 (GStreamer's v4l2src and multipartmux).

  python3 luma_camera_bridge.py [--listen 127.0.0.1] [--port 8103] [--size 1280x720] [--fps 15]
"""
import argparse
import collections
import json
import logging
import re
import select
import signal
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG = logging.getLogger("luma-bridge")

VITURE_VENDOR = "35ca"
BOUNDARY = "lumaframe"
# still.jpg and has_frame only vouch for a frame this recent.
FRESH_FOR = 5.0
# A pipeline that delivers nothing for this long is stuck: it is stopped and started again.
FIRST_FRAME_TIMEOUT = 10.0
STALL_AFTER = 5.0
# Between attempts while the camera is missing or failing.
RETRY_S = 2.0
GST_ERROR = re.compile(r"ERROR: from element \S+: (.*)")   # the element path has colons too
NOT_PLUGGED = ("the VITURE glasses' camera is not on USB: plug the glasses into the Orin's USB-C "
               "port, and replug them if nothing appears within a few seconds")


def _read(path):
    try:
        return path.read_text().strip()
    except OSError:
        return None


def find_camera(sysfs=Path("/sys/class/video4linux"), vendor=VITURE_VENDOR):
    """The glasses' camera, as /dev/videoN: the capture node (index 0) of a USB video device whose
    hub also holds a device of `vendor`. None while the glasses are not plugged in."""
    def number(node):
        digits = re.sub(r"\D", "", node.name)
        return int(digits) if digits else 0
    for node in sorted(sysfs.glob("video*"), key=number):
        if _read(node / "index") != "0":
            continue                                   # a metadata node, not the picture
        try:
            interface = (node / "device").resolve(strict=True)   # .../1-1/1-1.2/1-1.2:1.0
        except OSError:
            continue
        device, hub = interface.parent, interface.parent.parent  # .../1-1/1-1.2 and .../1-1
        try:
            siblings = [p for p in hub.iterdir() if re.fullmatch(r"\d+-[\d.]+", p.name) and p != device]
        except OSError:
            continue
        if any(_read(sibling / "idVendor") == vendor for sibling in siblings):
            return f"/dev/{node.name}"
    return None


def gst_failure(lines):
    """gst-launch-1.0's own reason: the message of its last "ERROR: from element <path>: <message>"
    line, with the "system error: ..." detail when it gives one - not the generic trailers it adds
    after ("Failed to set pipeline to PAUSED."). Falls back to the last line."""
    lines = list(lines)
    for i in range(len(lines) - 1, -1, -1):
        match = GST_ERROR.match(lines[i])
        if match:
            detail = next((line for line in lines[i + 1:i + 4] if line.startswith("system error:")), "")
            return f"{match.group(1).strip()}{f' ({detail})' if detail else ''}"
    return lines[-1] if lines else ""


def read_parts(stream, boundary=BOUNDARY):
    """The JPEG frames of a multipartmux stream, by each part's Content-Length."""
    marker = f"--{boundary}".encode()
    while True:
        line = stream.readline()
        if not line:
            return
        if line.strip() != marker:
            continue
        length = None
        while True:
            header = stream.readline()
            if not header:
                return
            if not header.strip():
                break
            name, _, value = header.decode("latin-1").partition(":")
            if name.strip().lower() == "content-length" and value.strip().isdigit():
                length = int(value.strip())
        if length is None:
            continue
        data = stream.read(length)
        if len(data) < length:
            return
        yield data


class Camera:
    """The one capture, its newest frame, and why it is in the state it is in."""

    def __init__(self, device, size, fps, idle):
        self.device, self.size, self.fps, self.idle = device, size, fps, idle
        self.cond = threading.Condition()
        self.frame, self.seq, self.frame_at = None, 0, None
        self.state, self.reason, self.current = "idle", "starts when something asks for it", None
        # -inf, not 0: monotonic time starts near boot, and a bridge started then must not run the
        # camera before anything has asked.
        self.wanted_at, self.clients = float("-inf"), 0
        self.stopping, self.process = False, None

    # -------------------------------------------------------------- demand
    def want(self):
        with self.cond:
            self.wanted_at = time.monotonic()
            if self.state == "idle":           # say so at once: the capture thread wakes within a second
                self.state, self.reason = "starting", "a request woke the camera"
            self.cond.notify_all()

    def wanted(self):
        return self.clients > 0 or time.monotonic() - self.wanted_at < self.idle

    def set_state(self, state, reason):
        with self.cond:
            if (state, reason) != (self.state, self.reason):
                _LOG.info("%s: %s", state, reason)
            self.state, self.reason = state, reason
            self.cond.notify_all()

    # -------------------------------------------------------------- frames
    def publish(self, data):
        with self.cond:
            self.frame, self.seq, self.frame_at = data, self.seq + 1, time.monotonic()
            if self.state != "live":
                self.state, self.reason = "live", ""
                _LOG.info("live: %s at %s, %d fps", self.current, self.size, self.fps)
            self.cond.notify_all()

    def fresh(self):
        return self.frame_at is not None and time.monotonic() - self.frame_at < FRESH_FOR

    def latest(self):
        with self.cond:
            return (self.frame, self.seq) if self.fresh() else (None, self.seq)

    # -------------------------------------------------------------- capture
    def run(self):
        while not self.stopping:
            with self.cond:
                while not self.wanted() and not self.stopping:
                    if self.state != "idle":
                        self.state = "idle"
                        self.reason = f"nothing has asked for the camera for {self.idle:.0f} s; it starts on the next request"
                        _LOG.info("idle")
                    self.cond.wait(timeout=1.0)
            if self.stopping:
                return
            device = self.device or find_camera()
            if not device:
                self.set_state("no_camera", NOT_PLUGGED)
                time.sleep(RETRY_S)
                continue
            self.capture(device)
            if self.wanted() and not self.stopping:
                time.sleep(RETRY_S)

    def capture(self, device):
        """One GStreamer pipeline, until it ends, stalls or is no longer wanted."""
        width, height = self.size.split("x")
        command = ["gst-launch-1.0", "-q", "v4l2src", f"device={device}", "!",
                   f"image/jpeg,width={width},height={height},framerate={self.fps}/1", "!",
                   "multipartmux", f"boundary={BOUNDARY}", "!", "fdsink", "fd=1"]
        self.current = device
        self.set_state("starting", f"opening {device} at {self.size}, {self.fps} fps")
        try:
            process = self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError as exc:
            self.set_state("error", f"cannot run gst-launch-1.0: {exc}")
            return
        errors = collections.deque(maxlen=20)

        def drain_errors():
            for line in process.stderr:
                text = line.decode("utf-8", "replace").strip()
                if text:
                    errors.append(text)

        def read_frames():
            for data in read_parts(process.stdout):
                self.publish(data)

        threads = [threading.Thread(target=drain_errors, daemon=True), threading.Thread(target=read_frames, daemon=True)]
        for thread in threads:
            thread.start()
        started, why = time.monotonic(), None
        while process.poll() is None:
            time.sleep(0.25)
            now = time.monotonic()
            if self.stopping or not self.wanted():
                why = "idle"
            elif self.frame_at is None or self.frame_at < started:
                if now - started > FIRST_FRAME_TIMEOUT:
                    why = f"{device} gave no frame in {FIRST_FRAME_TIMEOUT:.0f} s"
            elif now - self.frame_at > STALL_AFTER:
                why = f"{device} stopped sending frames"
            if why:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                break
        for thread in threads:
            thread.join(timeout=2)
        # Closed here, or every restart of the pipeline would leave two descriptors behind.
        for pipe in (process.stdout, process.stderr):
            try:
                pipe.close()
            except OSError:
                pass
        if why == "idle" or self.stopping:
            return
        if errors:
            _LOG.warning("gst-launch-1.0 said: %s", " | ".join(errors))
        detail = why or gst_failure(errors) or f"gst-launch-1.0 exited with code {process.returncode}"
        if not (self.device or find_camera()):
            self.set_state("no_camera", NOT_PLUGGED)
        else:
            self.set_state("error", detail[:300])


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "luma-camera-bridge"

    def log_message(self, *args):
        pass

    def send_body(self, code, content_type, body):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        camera = self.server.camera
        route = self.path.partition("?")[0]
        if route == "/healthz":
            camera.want()
            with camera.cond:
                age = None if camera.frame_at is None else round(time.monotonic() - camera.frame_at, 1)
                body = {"state": camera.state, "reason": camera.reason,
                        "live": camera.state == "live" and camera.fresh(), "has_frame": camera.fresh(),
                        "frames": camera.seq, "last_frame_age_s": age, "device": camera.current,
                        "size": camera.size, "fps": camera.fps, "mjpeg_clients": camera.clients}
            self.send_body(200, "application/json", json.dumps(body).encode())
        elif route == "/still.jpg":
            camera.want()
            frame, _ = camera.latest()
            if frame is None:
                self.send_body(503, "text/plain; charset=utf-8",
                               f"no fresh frame ({camera.state}: {camera.reason or 'starting'})".encode())
            else:
                self.send_body(200, "image/jpeg", frame)
        elif route == "/mjpeg":
            self.mjpeg(camera)
        else:
            self.send_body(404, "text/plain; charset=utf-8", b"not found")

    def viewer_gone(self):
        """The viewer sends nothing after its GET, so a readable socket means it has hung up."""
        try:
            readable, _, _ = select.select([self.connection], [], [], 0)
            return bool(readable) and self.connection.recv(1, socket.MSG_PEEK) == b""
        except OSError:
            return True

    def mjpeg(self, camera):
        with camera.cond:
            camera.clients += 1
            camera.wanted_at = time.monotonic()
            camera.cond.notify_all()
        self.close_connection = True
        try:
            self.send_response(200)
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last = 0
            while True:
                with camera.cond:
                    camera.cond.wait_for(lambda: camera.seq != last or camera.stopping, timeout=1.0)
                    if camera.stopping:
                        return
                    frame, seq, fresh = camera.frame, camera.seq, camera.fresh()
                if seq == last:
                    # Nothing new - the camera unplugged, or stalled. A viewer that left meanwhile
                    # must not stay counted, or the camera would never go idle.
                    if self.viewer_gone():
                        return
                    continue           # the relay ends a stream silent for 20 s
                last = seq
                if frame is None or not fresh:
                    continue           # the last frame of an earlier capture, not this one
                self.wfile.write(f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(frame)}\r\n\r\n".encode()
                                 + frame + b"\r\n")
                self.wfile.flush()
        except OSError:
            pass                       # the viewer went away
        finally:
            with camera.cond:
                camera.clients -= 1
                camera.wanted_at = time.monotonic()
                camera.cond.notify_all()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--listen", default="127.0.0.1", help="bind address; loopback keeps it on the Jetson")
    parser.add_argument("--port", type=int, default=8103)
    parser.add_argument("--device", default=None, help="a /dev/videoN to use instead of finding the glasses' camera")
    parser.add_argument("--size", default="1280x720", help="an MJPEG size the camera offers, WxH (up to 1920x1080)")
    parser.add_argument("--fps", type=int, default=15, choices=(5, 10, 15, 20, 25, 30))
    parser.add_argument("--idle", type=float, default=30.0, help="seconds without a request before the camera stops")
    args = parser.parse_args()
    if not re.fullmatch(r"\d{2,4}x\d{2,4}", args.size):
        parser.error("--size must look like 1280x720")
    camera = Camera(args.device, args.size, args.fps, args.idle)
    server = ThreadingHTTPServer((args.listen, args.port), Handler)
    server.daemon_threads = True
    server.camera = camera
    threading.Thread(target=camera.run, name="capture", daemon=True).start()

    def on_term(*_):
        raise KeyboardInterrupt        # systemd's stop: the same clean exit as Ctrl-C
    signal.signal(signal.SIGTERM, on_term)
    _LOG.info("serving on http://%s:%d (camera %s)", args.listen, args.port, args.device or "found by its VITURE neighbours")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        # The pipeline holds the camera: it must not outlive the bridge.
        camera.stopping = True
        with camera.cond:
            camera.cond.notify_all()
        if camera.process and camera.process.poll() is None:
            camera.process.terminate()
            try:
                camera.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                camera.process.kill()
        _LOG.info("stopped")


if __name__ == "__main__":
    main()
