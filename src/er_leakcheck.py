import os,json,hashlib,collections
import numpy as np
from sklearn.metrics import f1_score
ROOT=os.environ.get("ER_ROOT",os.path.expanduser("~/prml_proj/er"));DATA=f"{ROOT}/data"
rows=[json.loads(l) for l in open(f"{DATA}/pairs_sample.jsonl")]
sp=json.load(open(f"{DATA}/split.json"));tr=np.array(sp["train"]);te=np.array(sp["test"])
F=np.load(f"{DATA}/feats.npz",allow_pickle=True);y=F["y"]
P=np.load(f"{DATA}/matcher_probs.npz");pc=P["rf_cal"]
pm=(pc>=0.5).astype(int)
out={}
# 1. entity-id overlap between train and test pairs
train_ids=set()
for i in tr:
    train_ids.add(rows[i]["left"]["id"]);train_ids.add(rows[i]["right"]["id"])
seen=np.array([int(rows[i]["left"]["id"] in train_ids)+int(rows[i]["right"]["id"] in train_ids) for i in te])
llm={}
for tag in ["Qwen2.5-7B-Instruct","Qwen2.5-14B-Instruct-AWQ"]:
    f=f"{DATA}/llm_{tag}_main.jsonl"
    if os.path.exists(f):
        d={}
        for line in open(f):
            r=json.loads(line);d[r["idx"]]=r["pred"]
        llm[tag]=np.array([d.get(i,-1) for i in te])
print("test pairs by #entities already seen in some TRAIN pair:")
for k,name in [(0,"neither seen"),(1,"one seen"),(2,"both seen")]:
    m=seen==k
    d={"n":int(m.sum()),"frac":float(m.mean()),"pos_rate":float(y[te][m].mean()),"rf_f1":float(f1_score(y[te][m],pm[te][m])),"rf_err":float((pm[te][m]!=y[te][m]).mean())}
    for tag,v in llm.items():
        mm=m&(v>=0);d[f"llm_err_{tag}"]=float((v[mm]!=y[te][mm]).mean())
    out[name]=d
    print(f"  {name:13s} n={d['n']:5d} ({d['frac']*100:4.1f}%) pos={d['pos_rate']:.2f} RF F1={d['rf_f1']:.4f} err={d['rf_err']:.4f}"+"".join(f" | {t.split('-')[1]} LLM err={d[f'llm_err_{t}']:.4f}" for t in llm))
# 2. exact-content duplicates: same rendered pair text (order-insensitive) in train and test
def h(e):
    return hashlib.md5(json.dumps({"c":e["caption"],"s":e["schema"],"p":e["props"]},sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def ph(r):
    a,b=h(r["left"]),h(r["right"]);return a+b if a<b else b+a
trh=collections.Counter(ph(rows[i]) for i in tr)
dup=np.array([ph(rows[i]) in trh for i in te])
out["content_dup_test_in_train"]={"n":int(dup.sum()),"frac":float(dup.mean())}
print(f"test pairs whose exact record contents also appear as a TRAIN pair: {dup.sum()} ({dup.mean()*100:.2f}%)")
if dup.sum()>0:
    m=dup;print(f"   on those: pos={y[te][m].mean():.2f} RF err={(pm[te][m]!=y[te][m]).mean():.4f}")
    m=~dup;print(f"   on the rest: RF F1={f1_score(y[te][m],pm[te][m]):.4f} err={(pm[te][m]!=y[te][m]).mean():.4f}")
# 3. identical left/right content within a pair (would be trivially positive)
same=np.array([h(rows[i]["left"])==h(rows[i]["right"]) for i in te])
out["identical_sides_in_test"]={"n":int(same.sum()),"pos_rate":float(y[te][same].mean()) if same.sum() else None}
print(f"test pairs where both records are byte-identical: {same.sum()} (pos rate {out['identical_sides_in_test']['pos_rate']})")
# 4. how many distinct entities / how concentrated (cluster hubs)
cnt=collections.Counter()
for r in rows:
    cnt[r["left"]["id"]]+=1;cnt[r["right"]["id"]]+=1
top=cnt.most_common(5)
out["entities"]={"distinct":len(cnt),"pairs":len(rows),"top5_degree":[c for _,c in top]}
print(f"distinct entity ids in sample: {len(cnt)} over {len(rows)} pairs; max degree {top[0][1]}; mean degree {2*len(rows)/len(cnt):.2f}")
# 5. label-conditional view of the overlap: is 'seen' itself predictive?
for lab in [0,1]:
    m=y[te]==lab;print(f"  label={lab}: frac both-seen={(seen[m]==2).mean():.3f} one-seen={(seen[m]==1).mean():.3f} neither={(seen[m]==0).mean():.3f}")
json.dump(out,open(f"{DATA}/leakcheck_results.json","w"),indent=1)
print("LEAK_DONE")
