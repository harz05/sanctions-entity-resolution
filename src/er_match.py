import os,json,time,sys
import numpy as np
from sklearn.model_selection import StratifiedKFold,cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score,accuracy_score,brier_score_loss
from sklearn.isotonic import IsotonicRegression

DATA=os.path.join(os.environ.get("ER_ROOT",os.path.expanduser("~/prml_proj/er")),"data")
F=np.load(f"{DATA}/feats.npz",allow_pickle=True);X=F["X"];y=F["y"];names=list(F["names"])
DROP=["n_names_l","n_names_r","nprops_l","nprops_r","nprops_min"] if os.environ.get("NOCOUNTS","1")=="1" else []
_keep=[i for i,n in enumerate(names) if n not in DROP];X=X[:,_keep];names=[names[i] for i in _keep]
sp=json.load(open(f"{DATA}/split.json"));tr=np.array(sp["train"]);va=np.array(sp["val"]);te=np.array(sp["test"])
print("X",X.shape,"train/val/test",len(tr),len(va),len(te),"pos rate",y.mean().round(3),flush=True)

def models():
    return {
    "logreg":make_pipeline(StandardScaler(),LogisticRegression(C=1.0,class_weight="balanced",max_iter=5000)),
    "svm_rbf":make_pipeline(StandardScaler(),SVC(C=3.0,gamma="scale",class_weight="balanced",probability=True,random_state=0)),
    "knn":make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=25,weights="distance")),
    "dtree":DecisionTreeClassifier(max_depth=8,min_samples_leaf=20,class_weight="balanced",random_state=0),
    "rf":RandomForestClassifier(n_estimators=400,min_samples_leaf=2,class_weight="balanced_subsample",n_jobs=-1,random_state=0),
    "adaboost":AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2),n_estimators=300,learning_rate=0.5,random_state=0),
    "hgb":HistGradientBoostingClassifier(max_iter=400,learning_rate=0.06,max_leaf_nodes=31,class_weight="balanced",random_state=0),
    }

def metrics(yt,p,thr=0.5):
    yp=(p>=thr).astype(int)
    return {"f1":f1_score(yt,yp),"precision":precision_score(yt,yp),"recall":recall_score(yt,yp),"acc":accuracy_score(yt,yp),"auc":roc_auc_score(yt,p),"brier":brier_score_loss(yt,p)}

def best_thr(yt,p):
    ths=np.linspace(0.05,0.95,181)
    f=[f1_score(yt,(p>=t).astype(int)) for t in ths]
    return float(ths[int(np.argmax(f))])

res={};probs={}
skf=StratifiedKFold(5,shuffle=True,random_state=0)
for name,m in models().items():
    t0=time.time()
    Xtr,ytr=X[tr],y[tr]
    if name=="svm_rbf":
        sub=np.random.default_rng(0).choice(len(tr),min(12000,len(tr)),replace=False)
        oof=np.full(len(tr),np.nan)
        oof_sub=cross_val_predict(m,Xtr[sub],ytr[sub],cv=skf,method="predict_proba")[:,1]
        m.fit(Xtr[sub],ytr[sub])
        full=m.predict_proba(Xtr)[:,1]
        oof=full.copy();oof[sub]=oof_sub
    else:
        oof=cross_val_predict(m,Xtr,ytr,cv=skf,method="predict_proba")[:,1]
        m.fit(Xtr,ytr)
    pva=m.predict_proba(X[va])[:,1];pte=m.predict_proba(X[te])[:,1]
    thr=best_thr(y[va],pva)
    iso=IsotonicRegression(out_of_bounds="clip").fit(pva,y[va])
    r={"cv_oof":metrics(ytr,oof),"val":metrics(y[va],pva),"test@0.5":metrics(y[te],pte),"thr_val":thr,"test@thr":metrics(y[te],pte,thr),"test_cal":metrics(y[te],iso.predict(pte)),"fit_time_s":round(time.time()-t0,1)}
    res[name]=r
    P=np.full(len(y),np.nan);P[tr]=oof;P[va]=pva;P[te]=pte
    Pc=np.full(len(y),np.nan);Pc[tr]=iso.predict(oof);Pc[va]=iso.predict(pva);Pc[te]=iso.predict(pte)
    probs[name]=P;probs[name+"_cal"]=Pc
    if name=="rf":
        imp=sorted(zip(names,m.feature_importances_),key=lambda t:-t[1])
        res["rf_importance"]=[(a,float(b)) for a,b in imp]
    if name=="logreg":
        coef=m.named_steps["logisticregression"].coef_[0]
        res["logreg_coef"]=sorted([(a,float(b)) for a,b in zip(names,coef)],key=lambda t:-abs(t[1]))
    print(f"{name:10s} cvF1={r['cv_oof']['f1']:.4f} cvAUC={r['cv_oof']['auc']:.4f} | test F1@0.5={r['test@0.5']['f1']:.4f} F1@thr={r['test@thr']['f1']:.4f} (thr={thr:.2f}) AUC={r['test@0.5']['auc']:.4f} P={r['test@thr']['precision']:.3f} R={r['test@thr']['recall']:.3f} | {r['fit_time_s']}s",flush=True)
np.savez(f"{DATA}/matcher_probs.npz",**probs)
json.dump(res,open(f"{DATA}/matcher_results.json","w"),indent=1)
print("MATCH_DONE",flush=True)
