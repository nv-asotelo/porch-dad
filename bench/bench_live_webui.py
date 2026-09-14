#!/usr/bin/env python3
"""Cosmos3-Edge inference on a fully quiesced Orin Nano Super - Live VLM WebUI only.

v2 over the first pass:
  * runs only after the load average has actually settled, because the first pass started while
    the 15-minute average was still 6.56 and that showed up as a bimodal latency distribution
  * samples GPU load and the shim's own RSS, not just system-wide CPU and RAM
  * reports the full distribution, so "best" and "typical" are both visible rather than a single
    number standing in for both
"""
import base64
import json
import os
import re
import statistics
import subprocess
import threading
import time

import requests

SHIM = "http://127.0.0.1:8000/v1/chat/completions"
PROMPT = "Describe only what is visible in this image, in one short sentence."
WARMUP = 5
RUNS = 40

GPU_LOAD_PATHS = [
    "/sys/devices/platform/bus@0/17000000.gpu/load",
    "/sys/devices/platform/17000000.gpu/load",
    "/sys/devices/gpu.0/load",
]


def find_gpu_path():
    for p in GPU_LOAD_PATHS:
        if os.path.exists(p):
            return p
    return None


GPU_PATH = find_gpu_path()


def mem_mb():
    with open("/proc/meminfo") as fh:
        m = {k.strip(): int(v.split()[0]) // 1024 for k, v in
             (l.split(":", 1) for l in fh if ":" in l)}
    return m["MemTotal"], m["MemTotal"] - m["MemAvailable"], m["MemAvailable"]


def shim_rss_mb():
    try:
        out = subprocess.run(["pgrep", "-f", "cosmos3_shim"], capture_output=True, text=True).stdout
        tot = 0
        for pid in out.split():
            with open(f"/proc/{pid}/status") as fh:
                for line in fh:
                    if line.startswith("VmRSS:"):
                        tot += int(line.split()[1]) // 1024
        return tot
    except Exception:
        return None


class Sampler(threading.Thread):
    def __init__(self, interval=0.2):
        super().__init__(daemon=True)
        self.interval, self.stop_flag = interval, False
        self.cpu, self.gpu, self.used = [], [], []

    @staticmethod
    def _cpu():
        with open("/proc/stat") as fh:
            f = [int(x) for x in fh.readline().split()[1:]]
        return sum(f), f[3] + f[4]

    def run(self):
        pt, pi = self._cpu()
        while not self.stop_flag:
            time.sleep(self.interval)
            t, i = self._cpu()
            dt, di = t - pt, i - pi
            if dt > 0:
                self.cpu.append(100.0 * (dt - di) / dt)
            pt, pi = t, i
            if GPU_PATH:
                try:
                    self.gpu.append(int(open(GPU_PATH).read().strip()) / 10.0)
                except Exception:
                    pass
            self.used.append(mem_mb()[1])


def wait_settled(threshold=0.60, timeout=420):
    """Block until the 1-minute load average drops below threshold."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        la = os.getloadavg()
        if la[0] < threshold:
            print(f"settled: load {la[0]:.2f} {la[1]:.2f} {la[2]:.2f}", flush=True)
            return la
        time.sleep(15)
    la = os.getloadavg()
    print(f"proceeding un-settled: load {la[0]:.2f} {la[1]:.2f} {la[2]:.2f}", flush=True)
    return la


img_dir = "/home/orin/rerun_out"
imgs = sorted(f for f in os.listdir(img_dir) if f.endswith(".jpg"))
img = open(os.path.join(img_dir, imgs[0]), "rb").read()
b64 = base64.b64encode(img).decode()
body = {"model": "nvidia/Cosmos3-Edge", "max_tokens": 512, "temperature": 0.0,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
            {"type": "text", "text": PROMPT}]}]}

la = wait_settled()
engine = os.path.realpath("/home/orin/tensorrt-edgellm-workspace/active-engine")
total, used_idle, avail_idle = mem_mb()
rss_idle = shim_rss_mb()
nvp = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True).stdout.strip()
print(f"engine: {engine}", flush=True)
print(f"power:  {nvp.replace(chr(10),' | ')}", flush=True)
print(f"idle:   {used_idle} MB used / {total} MB, shim RSS {rss_idle} MB, GPU path {GPU_PATH}",
      flush=True)

for _ in range(WARMUP):
    requests.post(SHIM, json=body, timeout=180)

s = Sampler()
s.start()
lat, ctoks, ptoks = [], [], []
for i in range(RUNS):
    t0 = time.perf_counter()
    r = requests.post(SHIM, json=body, timeout=180)
    ms = (time.perf_counter() - t0) * 1000
    if r.status_code != 200:
        continue
    j = r.json()
    u = j.get("usage") or {}
    lat.append(ms)
    if u.get("completion_tokens"):
        ctoks.append(u["completion_tokens"])
    if u.get("prompt_tokens"):
        ptoks.append(u["prompt_tokens"])
s.stop_flag = True
s.join(timeout=3)

total, used_load, avail_load = mem_mb()
rss_load = shim_rss_mb()
sl = sorted(lat)
med_c = statistics.median(ctoks) if ctoks else 0


def pct(v, q):
    return round(v[max(0, min(len(v) - 1, int(len(v) * q) - 1))], 1)


res = {
    "engine": engine,
    "power_mode": nvp.split(":")[-1].strip().split()[0] if ":" in nvp else nvp,
    "load_at_start": [round(x, 2) for x in la],
    "runs": len(sl),
    "best_ms": round(sl[0], 1),
    "p50_ms": round(statistics.median(sl), 1),
    "mean_ms": round(statistics.fmean(sl), 1),
    "p90_ms": pct(sl, 0.90),
    "worst_ms": round(sl[-1], 1),
    "stdev_ms": round(statistics.pstdev(sl), 1),
    "prompt_tokens": ptoks[0] if ptoks else None,
    "completion_tokens_median": med_c,
    "best_tok_s": round(med_c / (sl[0] / 1000), 1) if med_c else None,
    "typical_tok_s": round(med_c / (statistics.median(sl) / 1000), 1) if med_c else None,
    "cpu_peak_pct": round(max(s.cpu), 1) if s.cpu else None,
    "cpu_p50_pct": round(statistics.median(s.cpu), 1) if s.cpu else None,
    "gpu_peak_pct": round(max(s.gpu), 1) if s.gpu else None,
    "gpu_p50_pct": round(statistics.median(s.gpu), 1) if s.gpu else None,
    "ram_total_mb": total,
    "ram_idle_used_mb": used_idle,
    "ram_peak_used_mb": max(s.used) if s.used else used_load,
    "ram_available_mb": avail_load,
    "shim_rss_idle_mb": rss_idle,
    "shim_rss_load_mb": rss_load,
}
print(json.dumps(res, indent=1), flush=True)
json.dump(res, open("/home/orin/bench_quiesced.json", "w"), indent=1)
