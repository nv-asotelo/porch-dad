"""The VITURE Luma Ultra camera bridge without the glasses: a fake sysfs tree, a fake gst-launch-1.0,
and its HTTP routes on loopback."""
import http.client
import importlib.util
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("luma_camera_bridge", HERE.parent / "luma_camera_bridge.py")
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)

FAKE_GST = """#!/usr/bin/env python3
import os, sys, time
mode, out = os.environ.get("FAKE_GST_MODE", "three"), sys.stdout.buffer
def frame(i):
    data = b"\\xff\\xd8" + bytes([i % 256]) * 100 + b"\\xff\\xd9"
    out.write(b"--lumaframe\\r\\nContent-Type: image/jpeg\\r\\nContent-Length: %d\\r\\n\\r\\n" % len(data) + data + b"\\r\\n")
    out.flush()
if mode == "three":
    for i in range(3):
        frame(i); time.sleep(0.05)
    # gst-launch-1.0's own shape: the cause, its debug lines, then generic trailers.
    sys.stderr.write("ERROR: from element /GstPipeline:pipeline0/GstV4l2Src:v4l2src0: Could not read from resource.\\n"
                     "Additional debug info:\\n../sys/v4l2/gstv4l2src.c(1220): gst_v4l2src_create ():\\n"
                     "system error: No such device\\n"
                     "ERROR: pipeline doesn't want to preroll.\\nFailed to set pipeline to PAUSED.\\n")
    sys.exit(1)
elif mode == "stream":
    i = 0
    while True:
        frame(i); i += 1; time.sleep(0.05)
else:
    time.sleep(60)
"""


def make_sysfs(root, viture=True):
    """Two cameras: the glasses' Sonix camera behind their hub (with a VITURE device beside it when
    `viture`), and a plain webcam elsewhere. Returns the video4linux class directory."""
    hub = root / "devices/usb1/1-1"
    for name, vendor in [("1-1.2", "0c45"), ("1-1.1", "35ca" if viture else "0bda")]:
        (hub / name).mkdir(parents=True)
        (hub / name / "idVendor").write_text(vendor + "\n")
    (hub / "1-1.2/1-1.2:1.0").mkdir()
    (root / "devices/usb1/1-2/1-2:1.0").mkdir(parents=True)
    (root / "devices/usb1/1-2/idVendor").write_text("046d\n")
    classes = root / "class/video4linux"
    for node, index, interface in [("video0", "0", "devices/usb1/1-2/1-2:1.0"),
                                   ("video2", "0", "devices/usb1/1-1/1-1.2/1-1.2:1.0"),
                                   ("video3", "1", "devices/usb1/1-1/1-1.2/1-1.2:1.0")]:
        (classes / node).mkdir(parents=True)
        (classes / node / "index").write_text(index + "\n")
        (classes / node / "device").symlink_to(root / interface)
    return classes


class FindCameraTest(unittest.TestCase):
    def test_the_glasses_camera_is_the_one_beside_a_viture_device(self):
        with tempfile.TemporaryDirectory() as temp:
            # video0 is another webcam, video3 the glasses' metadata node: neither is picked.
            self.assertEqual(bridge.find_camera(make_sysfs(Path(temp))), "/dev/video2")
        with tempfile.TemporaryDirectory() as temp:
            self.assertIsNone(bridge.find_camera(make_sysfs(Path(temp), viture=False)))
        with tempfile.TemporaryDirectory() as temp:
            self.assertIsNone(bridge.find_camera(Path(temp) / "missing"))


class ReadPartsTest(unittest.TestCase):
    def test_frames_are_cut_by_their_content_length(self):
        import io
        frames = [b"\xff\xd8one\xff\xd9", b"\xff\xd8--lumaframe inside\xff\xd9"]
        stream = b"".join(b"--lumaframe\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(f) + f + b"\r\n"
                          for f in frames)
        self.assertEqual(list(bridge.read_parts(io.BytesIO(stream))), frames)
        # A part cut short ends the stream rather than yielding half a picture.
        self.assertEqual(list(bridge.read_parts(io.BytesIO(stream[:-10]))), frames[:1])


class FakePipeline(unittest.TestCase):
    """A gst-launch-1.0 on PATH that writes frames the way multipartmux does."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        fake = Path(temp.name) / "gst-launch-1.0"
        fake.write_text(FAKE_GST)
        fake.chmod(0o755)
        path = mock.patch.dict(os.environ, {"PATH": f"{temp.name}:{os.environ['PATH']}"})
        path.start()
        self.addCleanup(path.stop)
        for name, value in [("FIRST_FRAME_TIMEOUT", 1.0), ("STALL_AFTER", 1.0), ("RETRY_S", 0.2)]:
            patcher = mock.patch.object(bridge, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def mode(self, mode):
        patcher = mock.patch.dict(os.environ, {"FAKE_GST_MODE": mode})
        patcher.start()
        self.addCleanup(patcher.stop)


class CaptureTest(FakePipeline):
    def test_a_pipeline_that_fails_says_why(self):
        self.mode("three")
        camera = bridge.Camera("/dev/video9", "1280x720", 15, idle=30)
        camera.want()
        camera.capture("/dev/video9")
        self.assertEqual((camera.seq, camera.state), (3, "error"))
        # The cause, not the trailer gst adds after it.
        self.assertEqual(camera.reason, "Could not read from resource. (system error: No such device)")
        self.assertEqual(camera.process.poll(), 1)

    def test_a_silent_pipeline_is_stopped(self):
        self.mode("silent")
        camera = bridge.Camera("/dev/video9", "1280x720", 15, idle=30)
        camera.want()
        started = time.monotonic()
        camera.capture("/dev/video9")
        self.assertLess(time.monotonic() - started, 5)
        self.assertEqual(camera.state, "error")
        self.assertIn("gave no frame", camera.reason)
        self.assertIsNotNone(camera.process.poll())        # terminated, not left holding the camera

    def test_the_camera_runs_while_wanted_and_stops_when_not(self):
        self.mode("stream")
        camera = bridge.Camera("/dev/video9", "1280x720", 15, idle=0.6)
        camera.want()
        thread = threading.Thread(target=camera.run, daemon=True)
        thread.start()
        self.addCleanup(lambda: setattr(camera, "stopping", True))
        deadline = time.monotonic() + 5
        while camera.state != "live" and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertEqual(camera.state, "live")
        self.assertIsNotNone(camera.latest()[0])
        # Nobody asks any more: idle, and the pipeline is gone.
        deadline = time.monotonic() + 5
        while camera.state != "idle" and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertEqual(camera.state, "idle")
        self.assertIsNotNone(camera.process.poll())
        # Asked again, it comes back.
        camera.want()
        deadline = time.monotonic() + 5
        while camera.state != "live" and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertEqual(camera.state, "live")

    def test_unplugged_glasses_are_reported_as_such(self):
        camera = bridge.Camera(None, "1280x720", 15, idle=30)
        camera.want()
        with mock.patch.object(bridge, "find_camera", return_value=None):
            thread = threading.Thread(target=camera.run, daemon=True)
            thread.start()
            deadline = time.monotonic() + 3
            while camera.state != "no_camera" and time.monotonic() < deadline:
                time.sleep(0.05)
            camera.stopping = True
        self.assertEqual(camera.state, "no_camera")
        self.assertIn("USB-C", camera.reason)


class RoutesTest(unittest.TestCase):
    def setUp(self):
        self.camera = bridge.Camera("/dev/video9", "1280x720", 15, idle=30)
        self.server = bridge.ThreadingHTTPServer(("127.0.0.1", 0), bridge.Handler)
        self.server.daemon_threads = True
        self.server.camera = self.camera
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.addCleanup(lambda: setattr(self.camera, "stopping", True))

    def get(self, path):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=5)
        self.addCleanup(connection.close)
        connection.request("GET", path)
        return connection.getresponse()

    def test_health_still_and_a_stream(self):
        # Nothing has asked yet, however soon after boot: the camera is off.
        self.assertFalse(self.camera.wanted())
        health = json.loads(self.get("/healthz").read())
        # Asking is what starts the camera, and the answer already says so.
        self.assertEqual((health["state"], health["live"], health["has_frame"]), ("starting", False, False))
        self.assertTrue(self.camera.wanted())
        still = self.get("/still.jpg")
        self.assertEqual(still.status, 503)
        self.assertIn(b"no fresh frame", still.read())
        frame = b"\xff\xd8frame\xff\xd9"
        self.camera.publish(frame)
        health = json.loads(self.get("/healthz").read())
        self.assertEqual((health["state"], health["live"], health["frames"]), ("live", True, 1))
        still = self.get("/still.jpg")
        self.assertEqual((still.status, still.getheader("Content-Type"), still.read()), (200, "image/jpeg", frame))
        stream = self.get("/mjpeg")
        self.assertEqual(stream.getheader("Content-Type"), "multipart/x-mixed-replace; boundary=lumaframe")
        parts = bridge.read_parts(stream)
        self.assertEqual(next(parts), frame)                       # a new viewer gets the newest at once
        self.camera.publish(b"\xff\xd8next\xff\xd9")
        self.assertEqual(next(parts), b"\xff\xd8next\xff\xd9")
        self.assertEqual(self.camera.clients, 1)
        # A frame older than FRESH_FOR is not vouched for.
        self.camera.frame_at -= bridge.FRESH_FOR + 1
        self.assertEqual(self.get("/still.jpg").status, 503)
        self.assertEqual(self.get("/elsewhere").status, 404)

    def test_a_viewer_who_leaves_during_an_outage_is_not_counted(self):
        # No frames at all (glasses unplugged): the stream writes nothing, yet a viewer who closes
        # it must stop counting, or the camera would never go idle.
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=5)
        connection.request("GET", "/mjpeg")
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        deadline = time.monotonic() + 3
        while self.camera.clients != 1 and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertEqual(self.camera.clients, 1)
        response.close()
        connection.close()
        deadline = time.monotonic() + 5
        while self.camera.clients and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertEqual(self.camera.clients, 0)


if __name__ == "__main__":
    unittest.main()
