import base64, io, time, requests, random
from PIL import Image

URL = "http://127.0.0.1:8000/v1/chat/completions"

def img_b64(seed, w=640, h=480):
    # unique image each call -> defeats the encoder embedding cache
    random.seed(seed)
    im = Image.new("RGB", (w, h))
    px = im.load()
    for _ in range(400):
        x, y = random.randrange(w), random.randrange(h)
        for dx in range(20):
            for dy in range(20):
                if x+dx < w and y+dy < h:
                    px[x+dx, y+dy] = (random.randrange(256), random.randrange(256), random.randrange(256))
    b = io.BytesIO(); im.save(b, format="JPEG", quality=90)
    return base64.b64encode(b.getvalue()).decode()

def call(with_image, seed, max_tok):
    content = [{"type": "text", "text": "Describe."}]
    if with_image:
        content.insert(0, {"type": "image_url",
                           "image_url": {"url": "data:image/jpeg;base64," + img_b64(seed)}})
    t = time.time()
    r = requests.post(URL, json={"model": "cosmos3-edge",
                                 "messages": [{"role": "user", "content": content}],
                                 "max_tokens": max_tok}, timeout=120)
    r.raise_for_status()
    return (time.time() - t) * 1000

for lbl, wi in (("warm", True), ("warm", False)):
    call(wi, 999, 4)

img1, txt1 = [], []
for i in range(6):
    img1.append(call(True,  1000 + i, 1))
    txt1.append(call(False, 2000 + i, 1))

def med(v): return sorted(v)[len(v)//2]
mi, mt = med(img1), med(txt1)
print(f"image + prefill + 1 tok : {mi:7.1f} ms   (all: {[f'{x:.0f}' for x in img1]})")
print(f"text-only prefill + 1tok: {mt:7.1f} ms   (all: {[f'{x:.0f}' for x in txt1]})")
print(f"--> ViT encode + image prefill delta: {mi-mt:7.1f} ms")
