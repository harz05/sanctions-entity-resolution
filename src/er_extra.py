import os,json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score,roc_auc_score,precision_score,recall_score
from sklearn.calibration import calibration_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=os.environ.get("ER_ROOT",os.path.expanduser("~/prml_proj/er"));DATA=f"{ROOT}/data";FIG=f"{ROOT}/figs";os.makedirs(FIG,exist_ok=True)
F=np.load(f"{DATA}/feats.npz",allow_pickle=True);X=F["X"];y=F["y"];names=list(F["names"])
DROP=["n_names_l","n_names_r","nprops_l","nprops_r","nprops_min"]
_keep=[i for i,n in enumerate(names) if n not in DROP];X=X[:,_keep];names=[names[i] for i in _keep]
sp=json.load(open(f"{DATA}/split.json"));tr=np.array(sp["train"]);va=np.array(sp["val"]);te=np.array(sp["test"])
out={}
# 1. rule baselines (name similarity thresholds), threshold picked on val
rules={}
for feat in ["jw_max","tsr_max","tok_jac_max"]:
    j=names.index(feat);best=(0,0)
    for t in np.linspace(0.5,1.0,51):
        f=f1_score(y[va],(X[va,j]>=t).astype(int))
        if f>best[0]:
            best=(f,t)
    pt=(X[te,j]>=best[1]).astype(int)
    rules[feat]={"thr":float(best[1]),"test_f1":float(f1_score(y[te],pt)),"precision":float(precision_score(y[te],pt)),"recall":float(recall_score(y[te],pt))}
    print(f"rule {feat}>={best[1]:.2f}: test F1={rules[feat]['test_f1']:.4f} P={rules[feat]['precision']:.3f} R={rules[feat]['recall']:.3f}",flush=True)
jw=names.index("jw_max");ym=names.index("year_match");db=names.index("date_both");ic=names.index("id_conflict")
rule2=((X[te,jw]>=0.9)&~((X[te,db]==1)&(X[te,ym]==0))&(X[te,ic]==0)).astype(int)
rules["jw>=0.9 & no date/id conflict"]={"test_f1":float(f1_score(y[te],rule2)),"precision":float(precision_score(y[te],rule2)),"recall":float(recall_score(y[te],rule2))}
print("rule jw>=0.9 & no conflicts:",rules["jw>=0.9 & no date/id conflict"],flush=True)
out["rules"]=rules
# 2. learning curve
sizes=[50,100,200,500,1000,2000,5000,10000,20000,36000]
rng=np.random.default_rng(0);lc={"rf":[],"logreg":[]}
for n in sizes:
    f_rf=[];f_lr=[]
    for rep in range(3 if n<5000 else 1):
        idx=rng.choice(tr,n,replace=False)
        if len(set(y[idx]))<2:
            continue
        rf=RandomForestClassifier(n_estimators=300,min_samples_leaf=2,class_weight="balanced_subsample",n_jobs=-1,random_state=rep).fit(X[idx],y[idx])
        f_rf.append(f1_score(y[te],rf.predict(X[te])))
        lr=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,class_weight="balanced",max_iter=5000)).fit(X[idx],y[idx])
        f_lr.append(f1_score(y[te],lr.predict(X[te])))
    lc["rf"].append({"n":n,"f1":float(np.mean(f_rf)),"std":float(np.std(f_rf))});lc["logreg"].append({"n":n,"f1":float(np.mean(f_lr)),"std":float(np.std(f_lr))})
    print(f"n={n:6d} RF F1={np.mean(f_rf):.4f}  LR F1={np.mean(f_lr):.4f}",flush=True)
out["learning_curve"]=lc
plt.figure(figsize=(5.2,3.6))
for k,lab in [("rf","random forest"),("logreg","logistic regression")]:
    plt.errorbar([c["n"] for c in lc[k]],[c["f1"]*100 for c in lc[k]],yerr=[c["std"]*100 for c in lc[k]],marker="o",ms=3,label=lab)
plt.axhline(95.9,color="gray",ls="--",lw=1,label="Qwen2.5-7B zero-shot (approx.)")
plt.xscale("log");plt.xlabel("labelled training pairs");plt.ylabel("test F1 (%)");plt.grid(alpha=.3);plt.legend(fontsize=8);plt.tight_layout();plt.savefig(f"{FIG}/learning_curve.png",dpi=160);plt.savefig(f"{FIG}/learning_curve.pdf")
# 3. calibration + feature importance
P=np.load(f"{DATA}/matcher_probs.npz");praw=P["rf"][te];pcal=P["rf_cal"][te]
plt.figure(figsize=(4.2,3.8))
for p,lab in [(praw,"RF raw"),(pcal,"RF + isotonic (val)")]:
    fr,mp=calibration_curve(y[te],p,n_bins=10,strategy="quantile");plt.plot(mp,fr,marker="o",ms=3,label=lab)
plt.plot([0,1],[0,1],"k--",lw=1);plt.xlabel("predicted P(match)");plt.ylabel("observed match rate");plt.legend(fontsize=8);plt.grid(alpha=.3);plt.tight_layout();plt.savefig(f"{FIG}/calibration.png",dpi=160);plt.savefig(f"{FIG}/calibration.pdf")
mres=json.load(open(f"{DATA}/matcher_results.json"));imp=mres["rf_importance"][:15]
plt.figure(figsize=(5.2,4));plt.barh([a for a,b in imp][::-1],[b for a,b in imp][::-1]);plt.xlabel("RF feature importance");plt.tight_layout();plt.savefig(f"{FIG}/importance.png",dpi=160);plt.savefig(f"{FIG}/importance.pdf")
# 4. per-type breakdown on test (matcher, and LLM if available)
pm=(pcal>=0.5).astype(int);yt=y[te]
llmf=f"{DATA}/llm_Qwen2.5-7B-Instruct_main.jsonl";ll={}
if os.path.exists(llmf):
    for line in open(llmf):
        r=json.loads(line);ll[r["idx"]]=r["pred"]
yl=np.array([ll.get(i,-1) for i in te]);has=yl>=0
def grp(mask,name):
    d={"n":int(mask.sum()),"pos_rate":float(yt[mask].mean()),"matcher_f1":float(f1_score(yt[mask],pm[mask])) if mask.sum()>0 else None,"matcher_err":float((pm[mask]!=yt[mask]).mean())}
    m2=mask&has
    if m2.sum()>0:
        d["llm_f1"]=float(f1_score(yt[m2],yl[m2]));d["llm_err"]=float((yl[m2]!=yt[m2]).mean())
    print(f"{name:28s} n={d['n']:5d} pos={d['pos_rate']:.2f} matcher err={d['matcher_err']:.4f}"+(f" llm err={d['llm_err']:.4f}" if 'llm_err' in d else ""),flush=True)
    return d
kp=names.index("kind_person");ko=names.index("kind_org");cs=names.index("cross_script");dm=names.index("date_missing");ib=names.index("id_both")
bt={}
bt["person-person"]=grp(X[te,kp]==1,"person-person");bt["org-org"]=grp(X[te,ko]==1,"org-org");bt["cross-script"]=grp(X[te,cs]==1,"cross-script");bt["same-script"]=grp(X[te,cs]==0,"same-script")
bt["no dates on a side"]=grp(X[te,dm]==1,"no dates on a side");bt["dates on both"]=grp(X[te,dm]==0,"dates on both");bt["ids on both"]=grp(X[te,ib]==1,"ids on both")
out["breakdown"]=bt
json.dump(out,open(f"{DATA}/extra_results.json","w"),indent=1)
print("EXTRA_DONE")
