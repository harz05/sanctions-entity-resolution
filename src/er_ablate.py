import os,json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score,roc_auc_score
DATA=os.path.join(os.environ.get("ER_ROOT",os.path.expanduser("~/prml_proj/er")),"data")
F=np.load(f"{DATA}/feats.npz",allow_pickle=True);X=F["X"];y=F["y"];names=list(F["names"])
sp=json.load(open(f"{DATA}/split.json"));tr=np.array(sp["train"]);te=np.array(sp["test"])
def idx(cols):
    return [names.index(c) for c in cols]
NAME=["jw_max","tsr_max","tok_jac_max","ratio_max","exact_name","first_last_match"]
COUNTS=["n_names_l","n_names_r","nprops_l","nprops_r","nprops_min"]
ATTR=["kind_person","kind_org","kind_mismatch","date_both","date_match","year_match","year_diff","date_missing","geo_both","geo_jac","geo_any","geo_conflict","gender_both","gender_mismatch","id_both","id_match","id_conflict","addr_both","addr_jac","bplace_match","pos_jac","cross_script"]
TOPIC=["topics_jac"]
groups={"all":names,"no_counts":NAME+ATTR+TOPIC,"no_counts_no_topics":NAME+ATTR,"names_only":NAME,"names+dates":NAME+["date_both","date_match","year_match","year_diff","date_missing"],"attrs_only(no names)":ATTR+TOPIC+COUNTS,"counts_only":COUNTS,"no_exact_name":[c for c in names if c!="exact_name"]}
out={}
for g,cols in groups.items():
    c=idx(cols)
    rf=RandomForestClassifier(n_estimators=300,min_samples_leaf=2,class_weight="balanced_subsample",n_jobs=-1,random_state=0).fit(X[tr][:,c],y[tr])
    p=rf.predict_proba(X[te][:,c])[:,1]
    lr=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,class_weight="balanced",max_iter=5000)).fit(X[tr][:,c],y[tr])
    pl=lr.predict_proba(X[te][:,c])[:,1]
    out[g]={"rf_f1":float(f1_score(y[te],p>=0.5)),"rf_auc":float(roc_auc_score(y[te],p)),"lr_f1":float(f1_score(y[te],pl>=0.5)),"lr_auc":float(roc_auc_score(y[te],pl)),"n_feats":len(c)}
    print(f"{g:22s} nf={len(c):2d} RF F1={out[g]['rf_f1']:.4f} AUC={out[g]['rf_auc']:.4f} | LR F1={out[g]['lr_f1']:.4f} AUC={out[g]['lr_auc']:.4f}",flush=True)
pos=y==1;neg=y==0
def rate(col,mask):
    return float(X[mask][:,names.index(col)].mean())
stats={}
for col in ["exact_name","cross_script","date_both","date_match","year_match","date_missing","id_both","id_match","id_conflict","geo_conflict","gender_mismatch","kind_mismatch","geo_any"]:
    stats[col]={"pos":rate(col,pos),"neg":rate(col,neg)}
    print(f"{col:16s} pos={stats[col]['pos']:.3f} neg={stats[col]['neg']:.3f}")
for col in ["jw_max","tsr_max","n_names_l","n_names_r","nprops_l","nprops_r"]:
    a=X[pos][:,names.index(col)];b=X[neg][:,names.index(col)]
    stats[col]={"pos_mean":float(a.mean()),"neg_mean":float(b.mean()),"pos_med":float(np.median(a)),"neg_med":float(np.median(b))}
    print(f"{col:16s} pos mean={a.mean():.3f} med={np.median(a):.3f} | neg mean={b.mean():.3f} med={np.median(b):.3f}")
json.dump({"ablation":out,"stats":stats},open(f"{DATA}/ablation.json","w"),indent=1)
print("ABL_DONE")
