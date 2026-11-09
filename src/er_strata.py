import os,json,hashlib
import numpy as np
from sklearn.metrics import f1_score,roc_auc_score
ROOT=os.environ.get("ER_ROOT",os.path.expanduser("~/prml_proj/er"));DATA=f"{ROOT}/data"
rows=[json.loads(l) for l in open(f"{DATA}/pairs_sample.jsonl")]
sp=json.load(open(f"{DATA}/split.json"));te=np.array(sp["test"])
F=np.load(f"{DATA}/feats.npz",allow_pickle=True);X=F["X"];y=F["y"];names=list(F["names"])
P=np.load(f"{DATA}/matcher_probs.npz");pc=P["rf_cal"];pm=(pc>=0.5).astype(int)
llm={}
for tag in ["Qwen2.5-7B-Instruct","Qwen2.5-14B-Instruct-AWQ","Qwen2.5-7B-Instruct_4shot"]:
    f=f"{DATA}/llm_{tag}_main.jsonl"
    if os.path.exists(f):
        d={}
        for line in open(f):
            r=json.loads(line);d[r["idx"]]=r["pred"]
        llm[tag]=np.array([d.get(i,-1) for i in te])
def h(e):
    return hashlib.md5(json.dumps({"c":e["caption"],"s":e["schema"],"p":e["props"]},sort_keys=True,ensure_ascii=False).encode()).hexdigest()
same=np.array([h(rows[i]["left"])==h(rows[i]["right"]) for i in te])
ex=X[te,names.index("exact_name")]==1
jw=X[te,names.index("jw_max")]
yt=y[te];pt=pm[te];ct=pc[te]
out={}
def rep(mask,name):
    d={"n":int(mask.sum()),"pos_rate":float(yt[mask].mean()),"rf_f1":float(f1_score(yt[mask],pt[mask])),"rf_err":float((pt[mask]!=yt[mask]).mean()),"rf_auc":float(roc_auc_score(yt[mask],ct[mask])) if len(set(yt[mask]))>1 else None}
    s=f"{name:34s} n={d['n']:5d} pos={d['pos_rate']:.2f} RF F1={d['rf_f1']:.4f} err={d['rf_err']:.4f}"
    for tag,v in llm.items():
        m2=mask&(v>=0)
        if m2.sum():
            d[f"llm_f1_{tag}"]=float(f1_score(yt[m2],v[m2]));d[f"llm_err_{tag}"]=float((v[m2]!=yt[m2]).mean())
            s+=f" | {tag.replace('Qwen2.5-','').replace('-Instruct','')} F1={d[f'llm_f1_{tag}']:.4f} err={d[f'llm_err_{tag}']:.4f}"
    print(s,flush=True);out[name]=d
rep(np.ones(len(te),bool),"all test")
rep(~same,"excluding byte-identical pairs")
rep(ex,"exact normalised name shared")
rep(~ex,"NO exact name shared (hard)")
rep((~ex)&(jw>=0.9),"no exact name, jw>=0.9")
rep((~ex)&(jw<0.9),"no exact name, jw<0.9")
rep(same,"byte-identical pairs")
# deferral on the hard subset: confidence gate, human expert
conf=np.minimum(ct,1-ct)
for frac in [0.05,0.10,0.20]:
    m=~ex;k=int(frac*m.sum());order=np.argsort(-conf[m]);keep=np.ones(m.sum(),bool);keep[order[:k]]=False
    f_keep=f1_score(yt[m][keep],pt[m][keep]);
    yy=yt[m].copy();pp=pt[m].copy();pp[~keep]=yy[~keep]
    print(f"hard subset, defer {frac:.0%} to human: F1 {f1_score(yt[m],pt[m]):.4f} -> {f1_score(yy,pp):.4f} (kept-part F1 {f_keep:.4f})",flush=True)
    out[f"hard_defer_{frac}"]={"f1_after":float(f1_score(yy,pp)),"f1_kept":float(f_keep)}
json.dump(out,open(f"{DATA}/strata_results.json","w"),indent=1)
print("STRATA_DONE")
