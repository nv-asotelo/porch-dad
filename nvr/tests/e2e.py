"""End-to-end happy-path test: stub Frigate serves a real clip for any event id."""
import http.server, socketserver, threading, sys, time, json
sys.path.insert(0, '/home/orin/nvr/bridge')

CLIP = open('/tmp/testclip.mp4','rb').read()

class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.endswith('clip.mp4'):
            self.send_response(200); self.send_header('Content-Type','video/mp4')
            self.send_header('Content-Length', str(len(CLIP))); self.end_headers()
            self.wfile.write(CLIP)
        else:
            self.send_response(404); self.end_headers()
    def log_message(self, *a): pass

srv = socketserver.TCPServer(("127.0.0.1", 5999), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
time.sleep(0.5)

import porch_dad as pd
pd.FRIGATE = "http://127.0.0.1:5999"   # point the real handler at the stub
pd.init_db()

published = []
def publish(topic, payload): published.append((topic, payload))

after = {"id": f"e2e-{int(time.time())}", "camera": "Front Porch", "label": "person",
         "start_time": time.time()-12, "end_time": time.time(),
         "has_clip": True, "has_snapshot": True}
t0 = time.time()
pd.handle_event(after, publish)
print(f"handle_event took {time.time()-t0:.1f}s")

import sqlite3
c = sqlite3.connect(pd.DB_PATH); c.row_factory = sqlite3.Row
r = c.execute("SELECT * FROM events WHERE id=?", (after["id"],)).fetchone()
if r:
    print(f"DB row     : category={r['category']} frames={r['num_frames']} latency={r['latency_ms']}ms")
    print(f"description: {r['description'][:260]!r}")
else:
    print("DB row     : MISSING")
print(f"MQTT published: {len(published)} message(s)")
if published: print(f"  topic={published[0][0]} payload={published[0][1][:150]}")
srv.shutdown()
