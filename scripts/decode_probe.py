import base64,io,random,sys,time,requests
from PIL import Image
URL="http://127.0.0.1:8000/v1/chat/completions"
def img(seed,w=640,h=480):
    random.seed(seed); im=Image.new("RGB",(w,h)); px=im.load()
    for _ in range(300):
        x,y=random.randrange(w),random.randrange(h)
        for dx in range(16):
            for dy in range(16):
                if x+dx<w and y+dy<h: px[x+dx,y+dy]=(random.randrange(256),)*3
    b=io.BytesIO(); im.save(b,format="JPEG",quality=88)
    return base64.b64encode(b.getvalue()).decode()
tag=sys.argv[1]; rows=[]
for i,mt in enumerate([16,32,48,64]*3):
    c=[{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+img(hash((tag,i))%99999)}},
       {"type":"text","text":"Describe this scene in detail."}]
    t=time.time()
    r=requests.post(URL,json={"model":"cosmos3-edge","messages":[{"role":"user","content":c}],
                              "max_tokens":mt,"temperature":0.0},timeout=180); r.raise_for_status()
    ms=(time.time()-t)*1000; u=r.json().get("usage") or {}
    g=u.get("completion_tokens",0)
    if g: rows.append((g,ms))
xs=[r[0] for r in rows]; ys=[r[1] for r in rows]; n=len(rows)
mx=sum(xs)/n; my=sum(ys)/n; den=sum((x-mx)**2 for x in xs)
m=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/den; b=my-m*mx
ss=sum((y-my)**2 for y in ys); sr=sum((y-(m*x+b))**2 for x,y in zip(xs,ys))
print(f"[{tag}] n={n} gen_tok {min(xs)}-{max(xs)}  marginal={m:.2f} ms/tok  fixed={b:.0f} ms  R2={1-sr/ss:.3f}")
