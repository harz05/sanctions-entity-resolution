import os,json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,roc_auc_score
ROOT=os.environ.get("ER_ROOT",os.path.expanduser("~/prml_proj/er"));DATA=f"{ROOT}/data"
F=np.load(f"{DATA}/feats.npz",allow_pickle=True);X=F["X"];y=F["y"];names=list(F["names"])
DROP=["n_names_l","n_names_r","nprops_l","nprops_r","nprops_min"]
_keep=[i for i,n in enumerate(names) if n not in DROP];X=X[:,_keep];names=[names[i] for i in _keep]
sp=json.load(open(f"{DATA}/split.json"));tr=np.array(sp["train"]);te=np.array(sp["test"])
kp=names.index("kind_person");ko=names.index("kind_org");cs=names.index("cross_script")
groups={"person":lambda idx:X[idx,kp]==1,"org":lambda idx:X[idx,ko]==1,"same_script":lambda idx:X[idx,cs]==0,"cross_script":lambda idx:X[idx,cs]==1}
def fit(idx):
    return RandomForestClassifier(n_estimators=400,min_samples_leaf=2,class_weight="balanced_subsample",n_jobs=-1,random_state=0).fit(X[idx],y[idx])
res={}
full=fit(tr)
for src,dst in [("person","org"),("org","person"),("same_script","cross_script"),("person","person"),("org","org"),("cross_script","cross_script")]:
    trs=tr[groups[src](tr)];tes=te[groups[dst](te)]
    m=fit(trs)
    p=m.predict_proba(X[tes])[:,1];pf=full.predict_proba(X[tes])[:,1]
    res[f"{src}->{dst}"]={"n_train":int(len(trs)),"n_test":int(len(tes)),"f1":float(f1_score(y[tes],p>=0.5)),"auc":float(roc_auc_score(y[tes],p)),"f1_fullmodel":float(f1_score(y[tes],pf>=0.5))}
    print(f"train {src:12s} ({len(trs):5d}) -> test {dst:12s} ({len(tes):5d}): F1={res[f'{src}->{dst}']['f1']:.4f} AUC={res[f'{src}->{dst}']['auc']:.4f} | model trained on everything: F1={res[f'{src}->{dst}']['f1_fullmodel']:.4f}",flush=True)
json.dump(res,open(f"{DATA}/xtype_results.json","w"),indent=1)
print("XTYPE_DONE")
