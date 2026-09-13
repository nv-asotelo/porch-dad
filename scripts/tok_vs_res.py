import base64, io, time, requests, random
from PIL import Image
URL = "http://127.0.0.1:8000/v1/chat/completions"

def img_b64(seed, w, h):
    random.seed(seed)
    im = Image.new("RGB", (w, h)); px = im.load()
    for _ in range(300):
        x, y = random.randrange(w), random.randrange(h)
        for dx in range(16):
            for dy in range(16):
                if x+dx < w and y+dy < h:
                    px[x+dx, y+dy] = (random.randrange(256),)*3
    b = io.BytesIO(); im.save(b, format="JPEG", quality=88)
    return base64.b64encode(b.getvalue()).decode()

def call(w, h, seed):
    c = [{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+img_b64(seed,w,h)}},
         {"type":"text","text":"Hi"}]
    t=time.time()
    r=requests.post(URL,json={"model":"cosmos3-edge","messages":[{"role":"user","content":c}],
                              "max_tokens":1},timeout=180)
    r.raise_for_status()
    ms=(time.time()-t)*1000
    return r.json()["usage"]["prompt_tokens"], ms

# text-only baseline
t=time.time()
r=requests.post(URL,json={"model":"cosmos3-edge","messages":[{"role":"user","content":"Hi"}],
                          "max_tokens":1},timeout=60); r.raise_for_status()
base=r.json()["usage"]["prompt_tokens"]
print(f"text-only prompt_tokens = {base}\n")
print(f"{'res':>10} {'prompt_tok':>11} {'img_tok':>8} {'best_ms':>8}")
for (w,h) in [(320,240),(448,336),(640,480),(896,672),(1280,960)]:
    best=1e9; pt=None
    for i in range(4):
        p,ms = call(w,h,hash((w,h,i))%99999)
        pt=p; best=min(best,ms)
    print(f"{w}x{h:<5} {pt:>11} {pt-base:>8} {best:>8.0f}")
