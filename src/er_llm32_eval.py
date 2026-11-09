"""Evaluate the 32B expert runs (er_llm32.py): LLM-alone score, complementarity, deferral curves for every expert,
behaviour on the deferred pairs and on the hard core. Writes data/llm32_results.json."""
import os,json
import numpy as np
from sklearn.metrics import f1_score,accuracy_score

ROOT=os.environ.get("ER_ROOT",os.path.expanduser("~/prml_proj/er"));DATA=f"{ROOT}/data"
F=np.load(f"{DATA}/feats.npz",allow_pickle=True);y=F["y"]
sp=json.load(open(f"{DATA}/split.json"));te=np.array(sp["test"])
pm=np.load(f"{DATA}/matcher_probs.npz")["rf_cal"];ym=(pm>=0.5).astype(int)
def load(fn):
    d={}
    for line in open(fn):
        r=json.loads(line)
        if r["pred"] is not None:
            d[r["idx"]]=int(r["pred"])
    return d
experts={"Qwen2.5-7B zero-shot":load(f"{DATA}/llm_Qwen2.5-7B-Instruct_main.jsonl"),
         "Qwen2.5-14B zero-shot":load(f"{DATA}/llm_Qwen2.5-14B-Instruct-AWQ_main.jsonl"),
         "Qwen2.5-7B 4-shot":load(f"{DATA}/llm_Qwen2.5-7B-Instruct_4shot_main.jsonl"),
         "Qwen3-32B zero-shot":load(f"{DATA}/llm_Qwen3-32B-AWQ_main.jsonl"),
         "Qwen3-32B chain of thought":load(f"{DATA}/llm32_cot.jsonl"),
         "Qwen3-32B thinking":load(f"{DATA}/llm32_think.jsonl")}
conf=np.minimum(pm[te],1-pm[te]);order=te[np.argsort(-conf)]
yt=y[te];base_pred=ym[te];base_f1=f1_score(yt,base_pred);base_err=int((base_pred!=yt).sum())
res={"matcher_f1":float(base_f1),"matcher_errors":base_err,"experts":{}}
fracs=[0,0.01,0.02,0.03,0.05,0.075,0.1,0.15,0.2]
core=[i for i in te if ym[i]!=y[i] and all(i in experts[k] and experts[k][i]!=y[i] for k in ["Qwen2.5-7B zero-shot","Qwen2.5-14B zero-shot","Qwen2.5-7B 4-shot"])]
wrong=[i for i in te if ym[i]!=y[i]]
res["core_n"]=len(core);res["matcher_wrong_n"]=len(wrong)
print(f"matcher test F1={base_f1*100:.2f} errors={base_err} | hard core (wrong for matcher + 7B + 14B + 4shot) = {len(core)}")
for name,d in experts.items():
    e={}
    full=all(i in d for i in te)
    if full:
        pred=np.array([d[i] for i in te]);e["alone_f1"]=float(f1_score(yt,pred));e["alone_acc"]=float(accuracy_score(yt,pred))
        e["complementarity"]={"both_right":float(((base_pred==yt)&(pred==yt)).mean()),"matcher_only":float(((base_pred==yt)&(pred!=yt)).mean()),"llm_only":float(((base_pred!=yt)&(pred==yt)).mean()),"both_wrong":float(((base_pred!=yt)&(pred!=yt)).mean())}
    curve=[]
    for f in fracs:
        k=int(round(f*len(te)));dset=order[:k]
        if not all(i in d for i in dset):
            break
        yhat=ym.copy()
        for i in dset:
            yhat[i]=d[i]
        curve.append({"frac":f,"f1":float(f1_score(yt,yhat[te])),"errors":int((yhat[te]!=yt).sum())})
    e["curve"]=curve
    k=int(round(0.05*len(te)));dset=order[:k]
    if all(i in d for i in dset):
        mw=np.array([ym[i]!=y[i] for i in dset]);lp=np.array([d[i] for i in dset]);ly=np.array([y[i] for i in dset])
        e["deferred5"]={"n":int(k),"matcher_errors":int(mw.sum()),"fixed":int(((lp==ly)&mw).sum()),"broken":int(((lp!=ly)&(~mw)).sum()),"expert_error_rate":float((lp!=ly).mean())}
    cov=[i for i in core if i in d];e["core_answered"]=len(cov);e["core_fixed"]=int(sum(d[i]==y[i] for i in cov))
    wv=[i for i in wrong if i in d];e["matcher_wrong_answered"]=len(wv);e["matcher_wrong_fixed"]=int(sum(d[i]==y[i] for i in wv))
    res["experts"][name]=e
    c5=[c for c in curve if abs(c["frac"]-0.05)<1e-9]
    print(f"{name:28s} alone F1={e.get('alone_f1',float('nan'))*100:6.2f} | 5% deferred: F1={c5[0]['f1']*100 if c5 else float('nan'):6.2f} errors={c5[0]['errors'] if c5 else '-'} fixed={e.get('deferred5',{}).get('fixed','-')} broken={e.get('deferred5',{}).get('broken','-')} expert err on deferred={e.get('deferred5',{}).get('expert_error_rate',float('nan')):.3f} | core fixed {e['core_fixed']}/{e['core_answered']} | matcher-wrong fixed {e['matcher_wrong_fixed']}/{e['matcher_wrong_answered']}")
    if curve:
        print("   curve:"," ".join(f"{c['frac']*100:g}%:{c['f1']*100:.2f}" for c in curve))
# human reference
hcurve=[]
for f in fracs:
    k=int(round(f*len(te)));dset=order[:k];yhat=ym.copy();yhat[dset]=y[dset];hcurve.append({"frac":f,"f1":float(f1_score(yt,yhat[te])),"errors":int((yhat[te]!=yt).sum())})
res["human_curve"]=hcurve
json.dump(res,open(f"{DATA}/llm32_results.json","w"),indent=1)
print("LLM32_EVAL_DONE")
