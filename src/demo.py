"""Live demo for the viva: learned deferral for sanctions entity resolution on OpenSanctions Pairs.

Run from ~/prml_proj/er with the project environment, e.g.
    P=/scratch/hchauhan_prml/env/bin/python
    $P demo.py pair --random            # one test pair end to end (features -> p_match -> gate -> expert)
    $P demo.py pair --pid 9212          # a specific pair
    $P demo.py pair --hard              # a random pair from the hard core (wrong for every system)
    $P demo.py train                    # retrain the random forest live (~10 s) and report test F1 / AUC
    $P demo.py curve                    # deferral curve (human / LLM expert) computed live from the probabilities
    $P demo.py core --n 5               # show pairs that matcher AND all LLM configs get wrong
    $P demo.py ask --pid 9212           # live zero-shot LLM answer with vLLM (loads Qwen2.5-7B, ~1-2 min)
All subcommands except `ask` run on CPU in seconds and use only files already in data/.
"""
import os,sys,json,argparse,time
import numpy as np
import er_lib as E
from sklearn.metrics import f1_score,roc_auc_score,accuracy_score

ROOT=os.environ.get("ER_ROOT",os.path.expanduser("~/prml_proj/er"));DATA=f"{ROOT}/data"
DROP=["n_names_l","n_names_r","nprops_l","nprops_r","nprops_min"]
LLM_TAGS=["Qwen2.5-7B-Instruct","Qwen2.5-14B-Instruct-AWQ","Qwen2.5-7B-Instruct_4shot"]

def hr(t=""):
    print("\n"+"="*8+" "+t+" "+"="*max(0,70-len(t)))

def load_all():
    rows=E.load_rows();sp=json.load(open(f"{DATA}/split.json"))
    F=np.load(f"{DATA}/feats.npz",allow_pickle=True);X=F["X"];y=F["y"];names=list(F["names"])
    keep=[i for i,n in enumerate(names) if n not in DROP];X=X[:,keep];names=[names[i] for i in keep]
    P=np.load(f"{DATA}/matcher_probs.npz")
    pid2idx={r["pid"]:i for i,r in enumerate(rows)}
    llm={}
    for t in LLM_TAGS:
        f=f"{DATA}/llm_{t}_main.jsonl"
        if os.path.exists(f):
            d={}
            for line in open(f):
                r=json.loads(line);d[r["idx"]]=(r["pred"],r.get("p_yes"))
            llm[t]=d
    return rows,sp,X,y,names,P,pid2idx,llm

def show_pair(i,rows,X,y,names,P,llm,budget=0.05,te=None):
    r=rows[i]
    hr(f"pair pid={r['pid']}  (analyst label: {'SAME ENTITY' if y[i]==1 else 'DIFFERENT'})")
    print("--- record A ---");print(E.render(r["left"]));print("--- record B ---");print(E.render(r["right"]))
    hr("29 pairwise features")
    for k in range(0,len(names),3):
        print("  ".join(f"{names[j]:>16s}={X[i,j]:6.3f}" for j in range(k,min(k+3,len(names)))))
    praw=float(P["rf"][i]);pcal=float(P["rf_cal"][i]);conf=min(pcal,1-pcal)
    hr("matcher")
    print(f"random forest P(match) raw = {praw:.3f}   isotonic-calibrated = {pcal:.3f}   -> decision: {'MATCH' if pcal>=0.5 else 'NO MATCH'}")
    print(f"confidence score min(p,1-p) = {conf:.3f}   (0 = certain, 0.5 = coin flip)")
    if te is not None:
        c_te=np.minimum(P["rf_cal"][te],1-P["rf_cal"][te]);thr=np.quantile(c_te,1-budget)
        decision="DEFER to expert" if conf>thr else "ACCEPT matcher decision"
        print(f"gate at a {budget:.0%} review budget: threshold tau={thr:.3f} -> {decision}")
    hr("experts")
    for t,d in llm.items():
        if i in d:
            pred,py=d[i];print(f"{t:28s}: {'yes' if pred==1 else 'no ':3s}  P(yes)={py if py is None else round(py,3)}  {'correct' if pred==y[i] else 'WRONG'}")
    print(f"{'human analyst (ground truth)':28s}: {'yes' if y[i]==1 else 'no'}")
    print(f"matcher {'correct' if (pcal>=0.5)==(y[i]==1) else 'WRONG'}")

def cmd_pair(a):
    rows,sp,X,y,names,P,pid2idx,llm=load_all();te=np.array(sp["test"]);rng=np.random.default_rng(a.seed)
    if a.pid is not None:
        i=pid2idx[a.pid]
    elif a.hard:
        core=[]
        pm=(P["rf_cal"]>=0.5).astype(int)
        for i in te:
            if pm[i]!=y[i] and all(i in d and d[i][0]!=y[i] for d in llm.values()):
                core.append(i)
        i=int(rng.choice(core));print(f"[{len(core)} hard-core pairs in test; showing one]")
    elif a.uncertain:
        c=np.minimum(P["rf_cal"][te],1-P["rf_cal"][te]);i=int(te[np.argsort(-c)[rng.integers(0,int(0.02*len(te)))]])
    else:
        i=int(rng.choice(te))
    show_pair(i,rows,X,y,names,P,llm,a.budget,te)

def cmd_train(a):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.isotonic import IsotonicRegression
    rows,sp,X,y,names,P,pid2idx,llm=load_all();tr,va,te=[np.array(sp[k]) for k in ("train","val","test")]
    hr(f"training random forest on {len(tr)} pairs x {X.shape[1]} features")
    t0=time.time();rf=RandomForestClassifier(n_estimators=400,min_samples_leaf=2,class_weight="balanced_subsample",n_jobs=-1,random_state=0).fit(X[tr],y[tr]);dt=time.time()-t0
    pv=rf.predict_proba(X[va])[:,1];pt=rf.predict_proba(X[te])[:,1]
    iso=IsotonicRegression(out_of_bounds="clip").fit(pv,y[va]);ptc=iso.predict(pt)
    print(f"fit time {dt:.1f}s | test F1@0.5 = {f1_score(y[te],pt>=0.5):.4f}  AUC = {roc_auc_score(y[te],pt):.4f}  acc = {accuracy_score(y[te],pt>=0.5):.4f}")
    print(f"after isotonic calibration on val: F1 = {f1_score(y[te],ptc>=0.5):.4f}  Brier {np.mean((pt-y[te])**2):.4f} -> {np.mean((ptc-y[te])**2):.4f}")
    imp=sorted(zip(names,rf.feature_importances_),key=lambda t:-t[1])[:8]
    print("top features:",", ".join(f"{n} {v:.3f}" for n,v in imp))
    if llm:
        t=LLM_TAGS[0];d=llm[t];m=np.array([i in d for i in te]);yl=np.array([d[i][0] if i in d else -1 for i in te])
        print(f"for reference, {t} zero-shot on the same test pairs: F1 = {f1_score(y[te][m],yl[m]):.4f}")

def deferral(y,p,order,fracs,expert=None):
    out=[]
    base=(p>=0.5).astype(int)
    for b in fracs:
        k=int(round(b*len(y)));pred=base.copy()
        idx=order[:k]
        pred[idx]=y[idx] if expert is None else expert[idx]
        out.append((b,f1_score(y,pred)*100,int((pred!=y).sum())))
    return out

def cmd_curve(a):
    rows,sp,X,y,names,P,pid2idx,llm=load_all();te=np.array(sp["test"]);yt=y[te];p=P["rf_cal"][te]
    conf=np.minimum(p,1-p);order=np.argsort(-conf);rng=np.random.default_rng(0);rand=rng.permutation(len(te))
    wrong=np.where((p>=0.5).astype(int)!=yt)[0];oracle=np.concatenate([wrong,np.setdiff1d(np.arange(len(te)),wrong)])
    fracs=[0,0.01,0.02,0.05,0.10,0.20]
    hr("expert = human analyst (assumed correct); F1 % / remaining errors")
    print(f"{'budget':>8s} {'confidence':>14s} {'random':>14s} {'oracle':>14s}")
    for (b,f1,e),(_,fr,er),(_,fo,eo) in zip(deferral(yt,p,order,fracs),deferral(yt,p,rand,fracs),deferral(yt,p,oracle,fracs)):
        print(f"{b:8.0%} {f1:8.2f} ({e:3d}) {fr:8.2f} ({er:3.0f}) {fo:8.2f} ({eo:3d})")
    for t,d in llm.items():
        if all(i in d for i in te):
            ex=np.array([d[i][0] for i in te])
            hr(f"expert = {t}; F1 % / remaining errors (LLM alone: {f1_score(yt,ex)*100:.2f})")
            print(f"{'budget':>8s} {'confidence':>14s} {'random':>14s}")
            for (b,f1,e),(_,fr,er) in zip(deferral(yt,p,order,fracs+[1.0],ex),deferral(yt,p,rand,fracs+[1.0],ex)):
                print(f"{b:8.0%} {f1:8.2f} ({e:3d}) {fr:8.2f} ({er:3.0f})")
    if a.plot:
        import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
        fr=np.linspace(0,0.3,31);h=deferral(yt,p,order,fr);plt.figure(figsize=(5,3.2))
        plt.plot([b*100 for b,_,_ in h],[f for _,f,_ in h],"o-",ms=3,label="confidence gate, human expert")
        for t,d in llm.items():
            if all(i in d for i in te):
                ex=np.array([d[i][0] for i in te]);l=deferral(yt,p,order,fr,ex);plt.plot([b*100 for b,_,_ in l],[f for _,f,_ in l],"s--",ms=3,label=f"confidence gate, {t}")
        plt.xlabel("pairs deferred (%)");plt.ylabel("system F1 (%)");plt.grid(alpha=.3);plt.legend(fontsize=7);plt.tight_layout();os.makedirs(f"{ROOT}/figs",exist_ok=True);plt.savefig(f"{ROOT}/figs/demo_curve.png",dpi=150);print("saved figs/demo_curve.png")

def cmd_core(a):
    rows,sp,X,y,names,P,pid2idx,llm=load_all();te=np.array(sp["test"]);pm=(P["rf_cal"]>=0.5).astype(int)
    core=[i for i in te if pm[i]!=y[i] and all(i in d and d[i][0]!=y[i] for d in llm.values())]
    hr(f"{len(core)} test pairs wrong for the matcher AND {len(llm)} LLM configs")
    rng=np.random.default_rng(a.seed)
    for i in rng.choice(core,min(a.n,len(core)),replace=False):
        r=rows[i];print(f"\n--- pid={r['pid']} analyst={'same' if y[i] else 'different'} matcher P(match)={P['rf_cal'][i]:.3f} ---")
        print(E.render(r["left"]));print("   vs");print(E.render(r["right"]))

def cmd_ask(a):
    from vllm import LLM,SamplingParams
    rows,sp,X,y,names,P,pid2idx,llm=load_all();i=pid2idx[a.pid];r=rows[i]
    hr(f"live LLM: {a.model} on pid={a.pid}")
    print(E.render(r["left"]));print("   vs");print(E.render(r["right"]))
    m=LLM(model=a.model,gpu_memory_utilization=0.85,max_model_len=4096,seed=0);tok=m.get_tokenizer()
    prompt=tok.apply_chat_template([{"role":"user","content":E.PROMPT.format(a=E.render(r["left"]),b=E.render(r["right"]))}],tokenize=False,add_generation_prompt=True)
    o=m.generate([prompt],SamplingParams(temperature=0,max_tokens=3,logprobs=10))[0].outputs[0]
    py=pn=0.0
    for tid,lp in o.logprobs[0].items():
        t=(lp.decoded_token or "").strip().lower()
        if t in("yes","y"):py+=np.exp(lp.logprob)
        elif t in("no","n"):pn+=np.exp(lp.logprob)
    print(f"\nLLM answer: {o.text.strip()!r}   P(yes)={py/(py+pn) if py+pn>0 else float('nan'):.3f}   analyst label: {'yes' if y[i] else 'no'}   matcher P(match)={P['rf_cal'][i]:.3f}")

if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter);sub=ap.add_subparsers(dest="cmd",required=True)
    s=sub.add_parser("pair");s.add_argument("--pid",type=int);s.add_argument("--random",action="store_true");s.add_argument("--hard",action="store_true");s.add_argument("--uncertain",action="store_true");s.add_argument("--budget",type=float,default=0.05);s.add_argument("--seed",type=int,default=None);s.set_defaults(f=cmd_pair)
    s=sub.add_parser("train");s.set_defaults(f=cmd_train)
    s=sub.add_parser("curve");s.add_argument("--plot",action="store_true");s.set_defaults(f=cmd_curve)
    s=sub.add_parser("core");s.add_argument("--n",type=int,default=5);s.add_argument("--seed",type=int,default=None);s.set_defaults(f=cmd_core)
    s=sub.add_parser("ask");s.add_argument("--pid",type=int,required=True);s.add_argument("--model",default="Qwen/Qwen2.5-7B-Instruct");s.set_defaults(f=cmd_ask)
    a=ap.parse_args();a.f(a)
