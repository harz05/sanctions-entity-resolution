"""Sampling-luck check: evaluate the matcher (trained on our 36k train split only) on EVERY other
analyst-judged entity pair in the raw file that is not in our 60k sample (~376k pairs)."""
import os,json,gzip,time,collections
import numpy as np
import er_lib as E
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score,roc_auc_score,precision_score,recall_score,accuracy_score
ROOT=os.environ.get("ER_ROOT",os.path.expanduser("~/prml_proj/er"));DATA=f"{ROOT}/data"
DROP=["n_names_l","n_names_r","nprops_l","nprops_r","nprops_min"]
rows=E.load_rows();sp=json.load(open(f"{DATA}/split.json"))
sampled=set()
for r in rows:
    a,b=r["left"]["id"],r["right"]["id"];sampled.add((a,b) if a<b else (b,a))
print("sampled pairs",len(sampled),flush=True)
# stream the raw file with exactly the same filter/dedup as er_lib.sample, keep everything not sampled
t0=time.time();seen=set();X=[];y=[];schema=[];cnt=collections.Counter()
with gzip.open(E.RAW,"rt") as f:
    for line in f:
        r=json.loads(line);j=r.get("judgement")
        if j not in("positive","negative"):
            continue
        a,b=r["left"]["id"],r["right"]["id"]
        if a==b or r["left"].get("schema") not in E.SCHEMAS or r["right"].get("schema") not in E.SCHEMAS:
            continue
        key=(a,b) if a<b else (b,a)
        if key in seen:
            continue
        seen.add(key)
        if key in sampled:
            cnt["in_sample"]+=1;continue
        l,rr=E.strip(r["left"]),E.strip(r["right"])
        X.append(E.pair_feats(l,rr));y.append(int(j=="positive"));schema.append((l["schema"],rr["schema"]))
        cnt[j]+=1
        if len(y)%50000==0:
            print(len(y),"pairs featurised",round(time.time()-t0),"s",flush=True)
X=np.array(X,dtype=np.float32);y=np.array(y)
print("held-out population:",dict(cnt),"pos rate",y.mean().round(4),"time",round(time.time()-t0),"s",flush=True)
names=list(E.FEATS);keep=[i for i,n in enumerate(names) if n not in DROP];X=X[:,keep];names=[names[i] for i in keep]
F=np.load(f"{DATA}/feats.npz",allow_pickle=True);Xs=F["X"][:,keep];ys=F["y"];tr=np.array(sp["train"]);va=np.array(sp["val"]);te=np.array(sp["test"])
rf=RandomForestClassifier(n_estimators=400,min_samples_leaf=2,class_weight="balanced_subsample",n_jobs=-1,random_state=0).fit(Xs[tr],ys[tr])
iso=IsotonicRegression(out_of_bounds="clip").fit(rf.predict_proba(Xs[va])[:,1],ys[va])
p=iso.predict(rf.predict_proba(X)[:,1]);pt=iso.predict(rf.predict_proba(Xs[te])[:,1])
out={"n_heldout":int(len(y)),"pos_rate_heldout":float(y.mean())}
def rep(name,yy,pp,thr=0.5):
    d={"f1":float(f1_score(yy,pp>=thr)),"precision":float(precision_score(yy,pp>=thr)),"recall":float(recall_score(yy,pp>=thr)),"acc":float(accuracy_score(yy,pp>=thr)),"auc":float(roc_auc_score(yy,pp))}
    print(f"{name:45s} n={len(yy):6d} F1={d['f1']:.4f} P={d['precision']:.4f} R={d['recall']:.4f} acc={d['acc']:.4f} AUC={d['auc']:.4f}",flush=True);return d
out["rf_test18k"]=rep("RF (train 36k) on our 18k test split",ys[te],pt)
out["rf_heldout"]=rep("RF (train 36k) on ALL other pairs (held-out pop.)",y,p)
# bootstrap CI on held-out F1
rng=np.random.default_rng(0);bs=[]
for _ in range(200):
    idx=rng.integers(0,len(y),len(y));bs.append(f1_score(y[idx],p[idx]>=0.5))
out["rf_heldout_f1_ci95"]=[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))];print("held-out F1 95% bootstrap CI",np.round(out["rf_heldout_f1_ci95"],4),flush=True)
# rule baselines + LR on the same held-out population, for scale
j=names.index("tsr_max");out["rule_tsr99_heldout"]=rep("rule token-set>=0.99 on held-out pop.",y,(X[:,j]>=0.99).astype(float))
j=names.index("jw_max");out["rule_jw95_heldout"]=rep("rule JW>=0.95 on held-out pop.",y,(X[:,j]>=0.95).astype(float))
lr=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,class_weight="balanced",max_iter=5000)).fit(Xs[tr],ys[tr])
out["lr_heldout"]=rep("logistic regression (train 36k) on held-out pop.",y,lr.predict_proba(X)[:,1])
# by schema pair type
kind=np.array([("person" if a=="Person" else "org")+"-"+("person" if b=="Person" else "org") for a,b in schema])
out["by_type"]={}
for k in sorted(set(kind)):
    m=kind==k
    if m.sum()>100 and len(set(y[m]))>1:
        out["by_type"][k]=rep(f"  RF held-out, {k}",y[m],p[m])
# human deferral on the held-out population (confidence gate)
conf=np.minimum(p,1-p);order=np.argsort(-conf);base=(p>=0.5).astype(int);out["heldout_deferral"]={}
for b in [0.01,0.02,0.05,0.10]:
    k=int(b*len(y));pred=base.copy();pred[order[:k]]=y[order[:k]];out["heldout_deferral"][str(b)]=float(f1_score(y,pred))
    print(f"held-out human deferral {b:.0%}: F1 {f1_score(y,base):.4f} -> {f1_score(y,pred):.4f}",flush=True)
json.dump(out,open(f"{DATA}/fullpop_results.json","w"),indent=1)
print("FULLPOP_DONE")
