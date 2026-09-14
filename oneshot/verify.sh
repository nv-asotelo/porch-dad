#!/usr/bin/env bash
# Phase 8 gate. Prove the deployment works AND that the measurement is honest.
#
# Two traps are checked before any timing is taken, because both silently produce good-looking
# numbers that mean nothing:
#   1. the encoder embedding cache - the runtime caches vision embeddings keyed on frame content,
#      so benchmarking one image repeatedly measures the cache and skips ~248 ms of ViT
#   2. a live WebUI browser tab - it drives ~1 request/sec on its own, so an open tab turns this
#      into a contention measurement
set -uo pipefail

SHIM=${SHIM:-http://127.0.0.1:8000}
WEBUI_PORT=${WEBUI_PORT:-8090}
RUNS=${RUNS:-20}
fail=0
ok()  { printf '  \033[32mPASS\033[0m  %s\n' "$1"; }
bad() { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; fail=1; }
warn(){ printf '  \033[33mWARN\033[0m  %s\n' "$1"; }

echo "== Phase 8 verification =="

# --- services -----------------------------------------------------------------------------------
for s in cosmos3-edge-shim live-vlm-webui; do
  st=$(systemctl is-active "$s.service" 2>/dev/null)
  [ "$st" = active ] && ok "$s: active" || bad "$s: $st"
done

curl -sf "$SHIM/v1/models" 2>/dev/null | grep -q "Cosmos3-Edge" \
  && ok "shim advertises nvidia/Cosmos3-Edge" \
  || bad "shim did not advertise the model at $SHIM/v1/models"

code=$(curl -sk -o /dev/null -w '%{http_code}' "https://localhost:$WEBUI_PORT/" 2>/dev/null)
[ "$code" = 200 ] && ok "WebUI serving on :$WEBUI_PORT" || bad "WebUI returned HTTP $code on :$WEBUI_PORT"

# --- trap 2: is anything else already driving the GPU? -------------------------------------------
gpu_path=$(ls /sys/devices/platform/bus@0/*.gpu/load /sys/devices/platform/*.gpu/load 2>/dev/null | head -1)
if [ -n "$gpu_path" ]; then
  busy=0
  for _ in 1 2 3; do
    [ "$(( $(cat "$gpu_path") / 10 ))" -gt 5 ] && busy=1
    sleep 1
  done
  [ "$busy" -eq 0 ] && ok "GPU idle before measurement (no competing client)" \
                    || warn "GPU already busy — a WebUI tab is probably streaming. Close it, or
         these numbers measure contention, not the model."
fi

# --- timing, with trap 1 defeated by varying the input -------------------------------------------
echo "  running $RUNS timed requests (input varied to defeat the embedding cache)..."
python3 - "$SHIM" "$RUNS" <<'PY'
import base64, io, json, statistics, sys, time, urllib.request

shim, runs = sys.argv[1], int(sys.argv[2])

def frame(i):
    # A distinct JPEG per request. Identical frames hit the runtime's encoder embedding cache,
    # which skips ~248 ms of ViT and makes the model look ~2x faster than it is.
    try:
        from PIL import Image
    except ImportError:
        print("  WARN  Pillow missing - cannot vary input; cache trap NOT defeated")
        return None
    im = Image.new("RGB", (640, 360), (i * 7 % 256, i * 13 % 256, i * 29 % 256))
    b = io.BytesIO(); im.save(b, "JPEG")
    return base64.b64encode(b.getvalue()).decode()

lat, gen = [], []
for i in range(runs):
    b64 = frame(i)
    if b64 is None:
        sys.exit(0)
    body = json.dumps({
        "model": "nvidia/Cosmos3-Edge", "max_tokens": 512, "temperature": 0.0,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
            {"type": "text", "text": "Describe only what is visible in this image, in one short sentence."}]}],
    }).encode()
    req = urllib.request.Request(shim + "/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        r = json.load(urllib.request.urlopen(req, timeout=180))
    except Exception as e:
        print(f"  FAIL  request {i}: {type(e).__name__}: {e}"); sys.exit(1)
    lat.append((time.perf_counter() - t0) * 1000)
    u = r.get("usage") or {}
    if u.get("completion_tokens"):
        gen.append(u["completion_tokens"])

lat.sort()
best, p50 = lat[0], statistics.median(lat)
print(f"  best {best:.0f} ms | median {p50:.0f} ms | p90 {lat[int(len(lat)*.9)-1]:.0f} ms")

# The expected band is wide on purpose: latency tracks caption LENGTH (elapsed ~= 200 + 13.5*tokens),
# so a run that happens to generate long captions is slower without anything being wrong.
if gen:
    marginal = (statistics.fmean(lat) - 200) / max(statistics.fmean(gen), 1)
    print(f"  generated {statistics.median(gen):.0f} tokens median -> ~{marginal:.1f} ms/token")
    print("  PASS  marginal decode in range" if 10 <= marginal <= 20 else
          f"  WARN  marginal decode {marginal:.1f} ms/token, expected ~13.5")
print("  PASS  best latency in range" if 180 <= best <= 400 else
      f"  WARN  best latency {best:.0f} ms, expected ~245 ms")
PY
[ $? -ne 0 ] && fail=1

# --- resources ------------------------------------------------------------------------------------
read -r used total avail < <(free -m | awk '/^Mem:/ {print $3, $2, $7}')
printf '  RAM %.1f / %.1f GB used, %.1f GB available\n' \
  "$(echo "$used/1024" | bc -l)" "$(echo "$total/1024" | bc -l)" "$(echo "$avail/1024" | bc -l)"
[ "$used" -lt 6000 ] && ok "RAM within expected envelope (~4.8 GB)" \
                     || warn "RAM ${used} MB — higher than the ~4.8 GB measured with only shim + WebUI running"

echo
[ "$fail" -eq 0 ] && echo "verification PASSED" || echo "verification FAILED"
exit "$fail"
