"""Frigate's own CPU detector (/cpu_model.tflite) on the eval frames, the way Frigate sees them:
the frame at the reachy_mini detect size (640x360), cut into 320x320 regions (model size, no
upscale), dog detections at min_score 0.5; a track needs a top score of 0.7."""
import glob, json, os
import numpy as np
from PIL import Image
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from ai_edge_litert.interpreter import Interpreter
it = Interpreter(model_path="/cpu_model.tflite", num_threads=2)
it.allocate_tensors()
inp = it.get_input_details()[0]
outs = it.get_output_details()
res = {}
for f in sorted(glob.glob("/tmp/frigate-probe/*.jpg")):
    im = Image.open(f).convert("RGB").resize((640, 360))
    a = np.asarray(im)
    dets = []
    for x0 in (0, 160, 320):
        for y0 in (0, 40):
            reg = a[y0:y0 + 320, x0:x0 + 320]
            it.set_tensor(inp["index"], reg[None].astype(np.uint8))
            it.invoke()
            boxes, classes, scores = (it.get_tensor(outs[i]["index"])[0] for i in range(3))
            for b, c, s in zip(boxes, classes, scores):
                if int(c) == 17 and s >= 0.5:      # labelmap 17 = dog
                    y1, x1, y2, x2 = b
                    dets.append([round((x0 + x1 * 320) / 640, 4), round((y0 + y1 * 320) / 360, 4),
                                 round((x0 + x2 * 320) / 640, 4), round((y0 + y2 * 320) / 360, 4), round(float(s), 3)])
    res[os.path.basename(f)] = dets
json.dump(res, open("/tmp/frigate-probe/dogs.json", "w"))
print(len(res), "frames;", sum(1 for d in res.values() if any(x[4] >= 0.7 for x in d)), "with a dog at >= 0.7;",
      sum(1 for d in res.values() if d), "with any dog >= 0.5")
