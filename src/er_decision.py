"""Decision-theory extras.
  (B) combining the matcher and the LLM instead of deferring (averaging, a logistic stacker, the LLM probability as a 30th feature)
  (C) a noisy human expert: the human deferral curve when the reviewer is wrong with probability eps
  (D) asymmetric costs: the Bayes reject region when a missed match costs more than a false match (Bishop 1.5.3)
Writes data/decision_results.json and figs/noisy_human.png, figs/cost_region.png.
"""
import os,json,sys
import numpy as np
from sklearn.metrics import f1_score,accuracy_score,roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=os.environ.get("ER_ROOT",os.path.expanduser("~/prml_proj/er"));DATA=f"{ROOT}/data";FIG=f"{ROOT}/figs";os.makedirs(FIG,exist_ok=True)
LLMTAG=sys.argv[1] if len(sys.argv)>1 else "Qwen2.5-7B-Instruct"
F=np.load(f"{DATA}/feats.npz",allow_pickle=True);X=F["X"];y=F["y"];names=list(F["names"])
DROP=["n_names_l","n_names_r","nprops_l","nprops_r","nprops_min"] if os.environ.get("NOCOUNTS","1")=="1" else []
keep=[i for i,n in enumerate(names) if n not in DROP];X=X[:,keep];names=[names[i] for i in keep]
sp=json.load(open(f"{DATA}/split.json"));tr=np.array(sp["train"]);va=np.array(sp["val"]);te=np.array(sp["test"])
P=np.load(f"{DATA}/matcher_probs.npz");pm=P["rf_cal"]
llm={}
for line in open(f"{DATA}/llm_{LLMTAG}_main.jsonl"):
    r=json.loads(line);llm[r["idx"]]=r
G=np.array([i for i in tr[:12000] if i in llm]);T=np.array([i for i in te if i in llm])
pl=np.zeros(len(y));pl[:]=np.nan
for i,r in llm.items():
    pl[i]=r["p_yes"]
yl=np.full(len(y),-1)
for i,r in llm.items():
    yl[i]=r["pred"]
res={"llm":LLMTAG}
def f1(idx,pred): return float(f1_score(y[idx],pred))
def logit(p): p=np.clip(p,1e-4,1-1e-4);return np.log(p/(1-p))
base=f1(T,(pm[T]>=0.5).astype(int))
print(f"matcher alone test F1={base*100:.2f} | LLM alone F1={f1(T,yl[T])*100:.2f} | AUC matcher={roc_auc_score(y[T],pm[T]):.4f} AUC LLM p_yes={roc_auc_score(y[T],pl[T]):.4f}",flush=True)

# (B) combination
comb={"matcher":base,"llm":f1(T,yl[T])}
comb["average"]=f1(T,((pm[T]+pl[T])/2>=0.5).astype(int))
ws=np.linspace(0,1,101);best=max(ws,key=lambda w:f1(G,((w*pm[G]+(1-w)*pl[G])>=0.5).astype(int)))
comb["weighted_w"]=float(best);comb["weighted"]=f1(T,((best*pm[T]+(1-best)*pl[T])>=0.5).astype(int))
st=LogisticRegression(C=1.0,max_iter=2000).fit(np.column_stack([logit(pm[G]),logit(pl[G])]),y[G])
comb["stacker_lr"]=f1(T,st.predict(np.column_stack([logit(pm[T]),logit(pl[T])])));comb["stacker_coef"]=[float(c) for c in st.coef_[0]]
rf_kw=dict(n_estimators=400,min_samples_leaf=2,class_weight="balanced_subsample",n_jobs=-1,random_state=0)
rfG=RandomForestClassifier(**rf_kw).fit(X[G],y[G]);comb["rf_12k_29feats"]=f1(T,rfG.predict(X[T]))
rfL=RandomForestClassifier(**rf_kw).fit(np.column_stack([X[G],pl[G]]),y[G]);comb["rf_12k_29feats+llm"]=f1(T,rfL.predict(np.column_stack([X[T],pl[T]])))
imp=rfL.feature_importances_;comb["llm_feature_importance"]=float(imp[-1]);comb["llm_feature_rank"]=int((imp>imp[-1]).sum()+1)
res["combination"]=comb
print("combination (test F1):",{k:(round(v*100,2) if isinstance(v,float) and k not in("weighted_w","llm_feature_importance") else v) for k,v in comb.items()},flush=True)

# (C) noisy human
conf=np.minimum(pm[T],1-pm[T]);order=np.argsort(-conf);ym=(pm[T]>=0.5).astype(int);yt=y[T]
fracs=[0,0.01,0.02,0.05,0.1,0.2];epss=[0,0.02,0.05,0.1,0.2];rng=np.random.default_rng(0)
noisy={}
for eps in epss:
    row=[]
    for f in fracs:
        k=int(round(f*len(T)));d=order[:k];vals=[]
        for _ in range(20 if eps>0 else 1):
            yh=ym.copy();flip=rng.random(k)<eps;yh[d]=np.where(flip,1-yt[d],yt[d]);vals.append(f1_score(yt,yh))
        row.append({"frac":f,"f1":float(np.mean(vals))})
    noisy[str(eps)]=row
k5=int(0.05*len(T));d5=order[:k5];err_rate_deferred=float((ym[d5]!=yt[d5]).mean())
noisy["matcher_error_rate_on_deferred_5pct"]=err_rate_deferred
res["noisy_human"]=noisy
print("noisy human, F1 at 5% deferred:",{e:round([r for r in noisy[e] if r['frac']==0.05][0]['f1']*100,2) for e in map(str,epss)},"| matcher error rate inside the deferred 5%:",round(err_rate_deferred,3),flush=True)
plt.figure(figsize=(5.2,3.4))
for eps in epss:
    plt.plot([r["frac"]*100 for r in noisy[str(eps)]],[r["f1"]*100 for r in noisy[str(eps)]],marker="o",ms=3,label=f"human error {int(eps*100)}%")
plt.xlabel("pairs deferred to human (%)");plt.ylabel("system F1 (%)");plt.grid(alpha=.3);plt.legend(fontsize=8);plt.tight_layout();plt.savefig(f"{FIG}/noisy_human.png",dpi=160);plt.savefig(f"{FIG}/noisy_human.pdf")

# (D) asymmetric costs with a reject option. Costs: false match c_fp=1, missed match c_fn, deferral c_d (human assumed right).
cost={};c_fp=1.0
for c_fn in [1,3,10]:
    for c_d in [0.05,0.1,0.2,0.5]:
        lo=c_d/c_fn;hi=1-c_d/c_fp           # defer when p*c_fn>c_d and (1-p)*c_fp>c_d
        p=pm[T];say_yes=(1-p)*c_fp<=np.minimum(p*c_fn,c_d);say_no=(p*c_fn<np.minimum((1-p)*c_fp,c_d));defer=~(say_yes|say_no)
        fp=int((say_yes&(yt==0)).sum());fn=int((say_no&(yt==1)).sum());nd=int(defer.sum())
        total=c_fp*fp+c_fn*fn+c_d*nd
        # best single threshold without reject (same costs)
        ths=np.linspace(0.01,0.99,99);costs=[c_fp*((p>=t)&(yt==0)).sum()+c_fn*((p<t)&(yt==1)).sum() for t in ths];j=int(np.argmin(costs))
        cost[f"cfn={c_fn},cd={c_d}"]={"region":[float(lo),float(hi)] if lo<hi else None,"defer_frac":nd/len(T),"fp":fp,"fn":fn,"cost_per_1000":total/len(T)*1000,"no_reject_best_thr":float(ths[j]),"no_reject_cost_per_1000":costs[j]/len(T)*1000}
res["asymmetric_cost"]=cost
print("asymmetric costs (cost per 1000 pairs, with reject vs best single threshold):",flush=True)
for k,v in cost.items():
    print(f"  {k:16s} region={None if v['region'] is None else [round(x,3) for x in v['region']]} defer={v['defer_frac']*100:5.1f}% FP={v['fp']:3d} FN={v['fn']:3d} cost={v['cost_per_1000']:6.1f} | no-reject thr={v['no_reject_best_thr']:.2f} cost={v['no_reject_cost_per_1000']:6.1f}",flush=True)
fig,ax=plt.subplots(figsize=(5.2,3.2))
ratios=np.array([1,2,3,5,10,20]);c_d=0.1
ax.fill_between(ratios,c_d/ratios,1-c_d,alpha=.25,label=f"reject region (c_d={c_d})")
ax.plot(ratios,c_d/ratios,color="k",lw=1);ax.plot(ratios,[1-c_d]*len(ratios),color="k",lw=1)
ax.set_xscale("log");ax.set_xlabel("cost of a missed match / cost of a false match");ax.set_ylabel("posterior P(match)");ax.set_ylim(0,1);ax.grid(alpha=.3);ax.legend(fontsize=8)
plt.tight_layout();plt.savefig(f"{FIG}/cost_region.png",dpi=160);plt.savefig(f"{FIG}/cost_region.pdf")
json.dump(res,open(f"{DATA}/decision_results.json","w"),indent=1)
print("DECISION_DONE")
