import base64, json, sys, requests
URL="http://127.0.0.1:8000/v1/chat/completions"
img=base64.b64encode(open("/home/orin/bench_frame.jpg","rb").read()).decode()
PROMPTS=[
 "Describe this scene in one sentence.",
 "List every distinct object you can see, comma separated.",
 "What text or small details are visible? Be specific.",
]
tag=sys.argv[1]
for p in PROMPTS:
    r=requests.post(URL,json={"model":"cosmos3-edge","messages":[{"role":"user","content":[
        {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+img}},
        {"type":"text","text":p}]}],"max_tokens":80,"temperature":0.0},timeout=180)
    r.raise_for_status()
    out=r.json()["choices"][0]["message"]["content"].strip().replace("\n"," ")
    print(f"[{tag}] Q: {p}\n[{tag}] A: {out}\n")
