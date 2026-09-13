import base64,io,random,time,requests
from PIL import Image
URL="http://127.0.0.1:8000/v1/chat/completions"
def img(seed,w=640,h=480):
    random.seed(seed); im=Image.new("RGB",(w,h)); px=im.load()
    for _ in range(200):
        x,y=random.randrange(w),random.randrange(h)
        for dx in range(24):
            for dy in range(24):
                if x+dx<w and y+dy<h: px[x+dx,y+dy]=(random.randrange(256),)*3
    b=io.BytesIO(); im.save(b,format="JPEG",quality=85)
    return base64.b64encode(b.getvalue()).decode()

for n in (1,2,3,4):
    c=[{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+img(i)}} for i in range(n)]
    c.append({"type":"text","text":f"These are {n} sampled frames. Describe briefly."})
    t=time.time()
    try:
        r=requests.post(URL,json={"model":"cosmos3-edge","messages":[{"role":"user","content":c}],
                                  "max_tokens":48,"temperature":0.0},timeout=180)
        ms=(time.time()-t)*1000
        if r.status_code!=200:
            print(f"n={n}: HTTP {r.status_code} {r.text[:160]}"); continue
        j=r.json(); u=j.get("usage") or {}
        print(f"n={n}: OK {ms:7.0f}ms prompt_tok={u.get('prompt_tokens')} gen={u.get('completion_tokens')} :: {j['choices'][0]['message']['content'][:70]!r}")
    except Exception as e:
        print(f"n={n}: EXC {type(e).__name__} {str(e)[:120]}")
