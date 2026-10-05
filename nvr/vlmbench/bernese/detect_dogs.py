"""Candidate dog boxes for labelling: RT-DETR v2 (COCO) on the full frame plus 2x2 upscaled tiles,
merged with NMS. Writes candidates.json {frame: [{"id", "box" (0-1 xyxy), "score"}]} and numbered
overlays for the labellers."""
import json, os, sys, glob
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from torchvision.ops import nms

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
name = "PekingU/rtdetr_v2_r50vd"
proc = AutoImageProcessor.from_pretrained(name)
model = AutoModelForObjectDetection.from_pretrained(name).to("cuda").eval()
dog = [i for i, l in model.config.id2label.items() if l == "dog"][0]

def detect(im, thr=0.25):
    inp = proc(images=im, return_tensors="pt").to("cuda")
    with torch.no_grad():
        o = model(**inp)
    r = proc.post_process_object_detection(o, target_sizes=[(im.height, im.width)], threshold=thr)[0]
    keep = r["labels"] == dog
    return r["boxes"][keep].cpu(), r["scores"][keep].cpu()

font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
cands = {}
for f in sorted(glob.glob(os.path.join(src, "*.jpg"))):
    im = Image.open(f).convert("RGB"); W, H = im.size
    boxes, scores = [], []
    b, s = detect(im); boxes.append(b); scores.append(s)
    for tx in (0, 1):
        for ty in (0, 1):
            x0, y0 = tx * W // 2 - (W // 8 if tx else 0), ty * H // 2 - (H // 8 if ty else 0)
            tile = im.crop((x0, y0, x0 + W // 2 + W // 8, y0 + H // 2 + H // 8))
            up = tile.resize((tile.width * 2, tile.height * 2), Image.LANCZOS)
            b, s = detect(up)
            if len(b):
                b = b / 2 + torch.tensor([x0, y0, x0, y0])
            boxes.append(b); scores.append(s)
    B, S = torch.cat(boxes), torch.cat(scores)
    keep = nms(B, S, 0.5).tolist() if len(B) else []
    # A tile sees part of a dog as a dog: drop a box that sits inside a kept, larger one.
    def inside(a, b):
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        area = (a[2] - a[0]) * (a[3] - a[1])
        return area > 0 and ix * iy / area > 0.8
    keep = sorted(keep, key=lambda i: -float((B[i][2] - B[i][0]) * (B[i][3] - B[i][1])))
    final = []
    for i in keep:
        if not any(inside(B[i].tolist(), B[j].tolist()) for j in final):
            final.append(i)
    keep = sorted(final, key=lambda i: (float(B[i][0]), float(B[i][1])))
    items = []
    for k, i in enumerate(keep):
        x1, y1, x2, y2 = B[i].tolist()
        items.append({"id": k + 1, "box": [round(x1 / W, 4), round(y1 / H, 4), round(x2 / W, 4), round(y2 / H, 4)],
                      "score": round(float(S[i]), 3)})
    cands[os.path.basename(f)] = items
    ov = im.copy(); d = ImageDraw.Draw(ov)
    for it in items:
        x1, y1, x2, y2 = it["box"][0] * W, it["box"][1] * H, it["box"][2] * W, it["box"][3] * H
        d.rectangle([x1, y1, x2, y2], outline=(0, 255, 255), width=2)
        d.rectangle([x1, max(0, y1 - 17), x1 + 22, max(0, y1 - 17) + 17], fill=(0, 255, 255))
        d.text((x1 + 3, max(0, y1 - 17)), str(it["id"]), fill=(0, 0, 0), font=font)
    ov.save(os.path.join(out, os.path.basename(f)), quality=92)
    # Zoomed crops of every candidate, so a labeller can tell a Bernese from a husky at this size.
    tiles = []
    for it in items:
        x1, y1, x2, y2 = it["box"][0] * W, it["box"][1] * H, it["box"][2] * W, it["box"][3] * H
        m = 0.3 * max(x2 - x1, y2 - y1)
        c = im.crop((int(x1 - m), int(y1 - m), int(x2 + m), int(y2 + m))).resize((220, 220), Image.LANCZOS)
        dd = ImageDraw.Draw(c); dd.rectangle([0, 0, 30, 20], fill=(0, 255, 255)); dd.text((4, 2), str(it["id"]), fill=(0, 0, 0), font=font)
        tiles.append(c)
    if tiles:
        cols = 5
        sheet = Image.new("RGB", (cols * 220, ((len(tiles) + cols - 1) // cols) * 220), "black")
        for k, t in enumerate(tiles):
            sheet.paste(t, ((k % cols) * 220, (k // cols) * 220))
        sheet.save(os.path.join(out, os.path.basename(f)[:-4] + "-dogs.jpg"), quality=90)
    print(os.path.basename(f), len(items), flush=True)
json.dump(cands, open(os.path.join(out, "candidates.json"), "w"), indent=1)
