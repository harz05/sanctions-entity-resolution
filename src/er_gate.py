import os,json,sys
import numpy as np
from sklearn.metrics import f1_score,accuracy_score,roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=os.environ.get("ER_ROOT",os.path.expanduser("~/prml_proj/er"));DATA=f"{ROOT}/data";FIG=f"{ROOT}/figs";os.makedirs(FIG,exist_ok=True)
MODEL=sys.argv[1] if len(sys.argv)>1 else "rf"
LLMTAG=sys.argv[2] if len(sys.argv)>2 else "Qwen2.5-7B-Instruct"
F=np.load(f"{DATA}/feats.npz",allow_pickle=True);X=F["X"];y=F["y"];names=list(F["names"])
DROP=["n_names_l","n_names_r","nprops_l","nprops_r","nprops_min"] if os.environ.get("NOCOUNTS","1")=="1" else []
_keep=[i for i,n in enumerate(names) if n not in DROP];X=X[:,_keep];names=[names[i] for i in _keep]
sp=json.load(open(f"{DATA}/split.json"));tr=np.array(sp["train"]);va=np.array(sp["val"]);te=np.array(sp["test"])
P=np.load(f"{DATA}/matcher_probs.npz");pm=P[MODEL+"_cal"];praw=P[MODEL]
mres=json.load(open(f"{DATA}/matcher_results.json"));thr=0.5
llm={}
for line in open(f"{DATA}/llm_{LLMTAG}_main.jsonl"):
    r=json.loads(line);llm[r["idx"]]=r
G=np.array([i for i in tr[:12000] if i in llm]);T=np.array([i for i in te if i in llm])
print("gate-train",len(G),"test",len(T),flush=True)

def preds(idx):
    return (pm[idx]>=thr).astype(int),np.array([llm[i]["pred"] for i in idx]),np.array([llm[i]["p_yes"] for i in idx]),y[idx]

def gate_feats(idx):
    p=pm[idx];u=np.minimum(p,1-p)
    return np.column_stack([X[idx],p,u,praw[idx]])

ym_G,yl_G,pl_G,yt_G=preds(G);ym_T,yl_T,pl_T,yt_T=preds(T)
mw_G=(ym_G!=yt_G).astype(int);gain_G=((yl_G==yt_G)&(ym_G!=yt_G)).astype(int)
mw_T=(ym_T!=yt_T).astype(int);gain_T=((yl_T==yt_T)&(ym_T!=yt_T)).astype(int)
print(f"matcher F1 test={f1_score(yt_T,ym_T):.4f} acc={accuracy_score(yt_T,ym_T):.4f} | LLM F1 test={f1_score(yt_T,yl_T):.4f} acc={accuracy_score(yt_T,yl_T):.4f}",flush=True)
both_r=((ym_T==yt_T)&(yl_T==yt_T)).mean();m_only=((ym_T==yt_T)&(yl_T!=yt_T)).mean();l_only=gain_T.mean();both_w=((ym_T!=yt_T)&(yl_T!=yt_T)).mean()
print(f"complementarity: both right={both_r:.4f} matcher-only right={m_only:.4f} LLM-only right={l_only:.4f} both wrong={both_w:.4f}",flush=True)

gates={}
XG=gate_feats(G);XT=gate_feats(T)
for tgt_name,tgt in [("err",mw_G),("gain",gain_G)]:
    if tgt.sum()<5:
        continue
    hgb=HistGradientBoostingClassifier(max_iter=300,learning_rate=0.05,class_weight="balanced",random_state=0).fit(XG,tgt)
    lr=make_pipeline(StandardScaler(),LogisticRegression(C=0.5,class_weight="balanced",max_iter=5000)).fit(XG,tgt)
    gates[f"hgb_{tgt_name}"]=hgb.predict_proba(XT)[:,1]
    gates[f"lr_{tgt_name}"]=lr.predict_proba(XT)[:,1]
    ref=mw_T if tgt_name=="err" else gain_T
    for k in [f"hgb_{tgt_name}",f"lr_{tgt_name}"]:
        print(f"gate {k}: AUC vs {tgt_name} on test = {roc_auc_score(ref,gates[k]):.4f}",flush=True)
conf=np.minimum(pm[T],1-pm[T])
print(f"gate conf: AUC vs err={roc_auc_score(mw_T,conf):.4f} vs gain={roc_auc_score(gain_T,conf):.4f}" if gain_T.sum()>0 else "",flush=True)

fracs=[0,0.01,0.02,0.03,0.05,0.075,0.1,0.15,0.2,0.3,0.4,0.5,0.75,1.0]
def curve(score,expert):
    out=[]
    order=np.argsort(-score)
    for f in fracs:
        k=int(round(f*len(T)));d=np.zeros(len(T),bool);d[order[:k]]=True
        yhat=ym_T.copy()
        yhat[d]=yl_T[d] if expert=="llm" else yt_T[d]
        out.append({"frac":f,"f1":float(f1_score(yt_T,yhat)),"acc":float(accuracy_score(yt_T,yhat)),"errors":int((yhat!=yt_T).sum())})
    return out
def random_curve(expert,reps=20):
    rng=np.random.default_rng(0);acc=[]
    for _ in range(reps):
        acc.append(curve(rng.random(len(T)),expert))
    return [{"frac":fracs[i],"f1":float(np.mean([c[i]["f1"] for c in acc])),"acc":float(np.mean([c[i]["acc"] for c in acc])),"errors":float(np.mean([c[i]["errors"] for c in acc]))} for i in range(len(fracs))]
res={"matcher":MODEL,"llm":LLMTAG,"n_test":len(T),"matcher_f1":float(f1_score(yt_T,ym_T)),"llm_f1":float(f1_score(yt_T,yl_T)),"complementarity":{"both_right":float(both_r),"matcher_only":float(m_only),"llm_only":float(l_only),"both_wrong":float(both_w)},"curves":{}}
for expert in ["llm","human"]:
    c={"confidence":curve(conf,expert),"random":random_curve(expert),"oracle":curve((gain_T if expert=="llm" else mw_T).astype(float)+conf*1e-3,expert)}
    for k,v in gates.items():
        c[k]=curve(v,expert)
    res["curves"][expert]=c
json.dump(res,open(f"{DATA}/gate_results_{LLMTAG}.json","w"),indent=1)
for expert in ["llm","human"]:
    print(f"== expert={expert}")
    hdr="frac    "+"  ".join(f"{k[:11]:>11s}" for k in res["curves"][expert]);print(hdr)
    for i,f in enumerate(fracs):
        print(f"{f:5.3f}   "+"  ".join(f"{res['curves'][expert][k][i]['f1']*100:11.2f}" for k in res["curves"][expert]))
fig,axes=plt.subplots(1,2,figsize=(11,4.2))
for ax,expert,title in zip(axes,["llm","human"],[f"Expert = LLM ({LLMTAG})","Expert = human review (always right)"]):
    for k,v in res["curves"][expert].items():
        ax.plot([c["frac"]*100 for c in v],[c["f1"]*100 for c in v],marker="o",ms=3,label=k)
    ax.set_xlabel("pairs deferred to expert (%)");ax.set_ylabel("F1 (%)");ax.set_title(title);ax.grid(alpha=.3);ax.legend(fontsize=7)
plt.tight_layout();plt.savefig(f"{FIG}/deferral_curves_{LLMTAG}.png",dpi=160);plt.savefig(f"{FIG}/deferral_curves_{LLMTAG}.pdf")
k=int(0.1*len(T));d=np.zeros(len(T),bool);d[np.argsort(-conf)[:k]]=True
prof={}
for col in ["cross_script","date_missing","kind_org","exact_name","id_both","geo_conflict","date_both"]:
    j=names.index(col);prof[col]={"deferred":float(X[T][d,j].mean()),"kept":float(X[T][~d,j].mean())}
for col in ["jw_max","tsr_max"]:
    j=names.index(col);prof[col]={"deferred":float(X[T][d,j].mean()),"kept":float(X[T][~d,j].mean())}
prof["pos_rate"]={"deferred":float(yt_T[d].mean()),"kept":float(yt_T[~d].mean())}
prof["matcher_err_rate"]={"deferred":float(mw_T[d].mean()),"kept":float(mw_T[~d].mean())}
prof["llm_err_rate"]={"deferred":float((yl_T!=yt_T)[d].mean()),"kept":float((yl_T!=yt_T)[~d].mean())}
json.dump(prof,open(f"{DATA}/deferred_profile_10pct_{LLMTAG}.json","w"),indent=1)
print("profile of 10% deferred (confidence gate):");[print(f"  {k:18s} deferred={v['deferred']:.3f} kept={v['kept']:.3f}") for k,v in prof.items()]
print("GATE_DONE")
