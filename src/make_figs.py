import json,os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__));REPO=os.path.dirname(HERE)
RES=f"{HERE}/results_all.json" if os.path.exists(f"{HERE}/results_all.json") else f"{REPO}/results/results_all.json"
FIG=os.environ.get("ER_FIGS",f"{HERE}/figs" if os.path.exists(f"{HERE}/results_all.json") else f"{REPO}/figs");os.makedirs(FIG,exist_ok=True)
D=json.load(open(RES))
plt.rcParams.update({"font.family":"STIXGeneral","mathtext.fontset":"stix","font.size":8,"axes.labelsize":8,"axes.titlesize":8.5,"legend.fontsize":7,"xtick.labelsize":7,"ytick.labelsize":7,"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":0.25,"grid.linewidth":0.5,"lines.linewidth":1.4,"axes.linewidth":0.6,"xtick.major.width":0.6,"ytick.major.width":0.6,"pdf.fonttype":42})
BLUE,ORANGE,AQUA,YELLOW,VIOLET,RED="#2a78d6","#eb6834","#1baf7a","#eda100","#6250d6","#e34948"
GRAY="#7a7873"
def save(fig,name):
    fig.savefig(f"{FIG}/{name}.pdf",bbox_inches="tight",pad_inches=0.02);fig.savefig(f"{FIG}/{name}.png",dpi=200,bbox_inches="tight",pad_inches=0.02);plt.close(fig)
def xy(curve,key="f1",xmax=1.0):
    pts=[(c["frac"]*100,c[key]*100) for c in curve if c["frac"]<=xmax+1e-9]
    return [p[0] for p in pts],[p[1] for p in pts]

# ---------- Fig 1: deferral curves, two panels (double column) ----------
fig,axes=plt.subplots(1,2,figsize=(7.0,2.5))
ax=axes[0];H=D["gate7"]["curves"]["human"]
for key,lab,col,ls,mk in [("oracle","oracle gate","#333333",":","" ),("confidence","confidence gate $\\min(p,1-p)$",BLUE,"-","o"),("hgb_err","learned gate (GBT)",ORANGE,"--","s"),("random","random gate",GRAY,"-.","")]:
    x,y=xy(H[key],xmax=0.30);ax.plot(x,y,ls,color=col,marker=mk,ms=3,label=lab)
ax.set_xlabel("pairs deferred to human review (%)");ax.set_ylabel("system F1 (%)");ax.set_title("(a) expert = human analyst")
ax.set_ylim(98.6,100.05);ax.set_xlim(-0.5,30.5);ax.legend(loc="lower right",frameon=False)
ax.annotate("98.72 (no deferral)",xy=(0,98.715),xytext=(3.5,98.63),fontsize=6.5,color="#333",arrowprops=dict(arrowstyle="-",lw=0.5,color="#333"))
ax.annotate("5% deferred: 99.63",xy=(5,99.632),xytext=(9,99.3),fontsize=6.5,color=BLUE,arrowprops=dict(arrowstyle="-",lw=0.5,color=BLUE))
ax=axes[1]
for key,lab,col,ls,mk,G in [("oracle","oracle gate, 7B","#333333",":","",D["gate7"]),("confidence","confidence gate, Qwen2.5-7B",BLUE,"-","o",D["gate7"]),("confidence","confidence gate, Qwen2.5-14B",VIOLET,"-","^",D["gate14"]),("hgb_gain","learned gate (GBT), 7B",ORANGE,"--","s",D["gate7"]),("random","random gate, 7B",GRAY,"-.","",D["gate7"])]:
    x,y=xy(G["curves"]["llm"][key],xmax=1.0);ax.plot(x,y,ls,color=col,marker=mk,ms=3,label=lab)
ax.axhline(98.715,color="#333",lw=0.6,ls="-",alpha=0.4,xmax=0.55)
ax.text(1.5,98.55,"matcher alone: 98.72",fontsize=6.5,color="#333")
ax.set_xlabel("pairs deferred to LLM (%)");ax.set_title("(b) expert = zero-shot LLM");ax.set_ylim(95.5,99.6);ax.set_xlim(-1,101)
ax.legend(loc="upper right",frameon=False)
ax.annotate("LLM alone (7B): 95.83",xy=(100,95.83),xytext=(60,95.62),fontsize=6.5,color="#333")
fig.tight_layout(w_pad=1.5);save(fig,"fig_deferral")

# ---------- Fig 1b: zoom on small budgets for LLM expert (slides) ----------
fig,ax=plt.subplots(figsize=(4.6,2.4))
for key,lab,col,ls,mk,G in [("oracle","oracle (7B)","#333333",":","",D["gate7"]),("confidence","confidence, 7B",BLUE,"-","o",D["gate7"]),("confidence","confidence, 14B",VIOLET,"-","^",D["gate14"]),("hgb_gain","learned GBT gate, 7B",ORANGE,"--","s",D["gate7"])]:
    x,y=xy(G["curves"]["llm"][key],xmax=0.2);ax.plot(x,y,ls,color=col,marker=mk,ms=3,label=lab)
x,y=xy(D["gate7"]["curves"]["human"]["confidence"],xmax=0.2);ax.plot(x,y,"-",color=AQUA,marker="D",ms=3,label="confidence, human")
ax.set_xlabel("pairs deferred (%)");ax.set_ylabel("system F1 (%)");ax.set_ylim(97.9,100.1);ax.legend(frameon=False,loc="center left",bbox_to_anchor=(1.0,0.5),ncol=1)
fig.tight_layout();save(fig,"fig_deferral_zoom")

# ---------- Fig 2: learning curve ----------
fig,ax=plt.subplots(figsize=(3.4,2.3))
lc=D["extra"]["learning_curve"]
for k,lab,col,mk in [("rf","random forest (29 feats)",BLUE,"o"),("logreg","logistic regression",ORANGE,"s")]:
    n=[c["n"] for c in lc[k]];f=[c["f1"]*100 for c in lc[k]];s=[c["std"]*100 for c in lc[k]]
    ax.errorbar(n,f,yerr=s,color=col,marker=mk,ms=3,capsize=1.5,lw=1.2,label=lab)
ax.axhline(D["llm_runs"]["Qwen2.5-14B-Instruct-AWQ"]["test_f1"]*100,color=VIOLET,ls="--",lw=1,label="Qwen2.5-14B zero-shot")
ax.axhline(D["llm_runs"]["Qwen2.5-7B-Instruct"]["test_f1"]*100,color=GRAY,ls="--",lw=1,label="Qwen2.5-7B zero-shot")
ax.axhline(D["extra"]["rules"]["tsr_max"]["test_f1"]*100,color=RED,ls=":",lw=1,label="best single-threshold rule")
ax.set_xscale("log");ax.set_xlabel("labelled training pairs");ax.set_ylabel("test F1 (%)");ax.set_ylim(92.3,99.2);ax.legend(frameon=False,loc="lower right",fontsize=6.2,ncol=2,columnspacing=0.8,handlelength=1.8)
fig.tight_layout();save(fig,"fig_learning_curve")

# ---------- Fig 3: difficulty strata ----------
fig,ax=plt.subplots(figsize=(3.4,2.3))
S=D["strata"];groups=[("all test","all\n(n=18000)"),("exact normalised name shared","exact name\nshared (10172)"),("no exact name, jw>=0.9","no exact name,\nJW$\\geq$0.9 (1625)"),("no exact name, jw<0.9","no exact name,\nJW$<$0.9 (6203)")]
w=0.26;xs=np.arange(len(groups))
for i,(key,lab,col) in enumerate([("rf_f1","RF matcher",BLUE),("llm7_f1","Qwen2.5-7B 0-shot",ORANGE),("llm14_f1","Qwen2.5-14B 0-shot",VIOLET)]):
    v=[S[g][key]*100 for g,_ in groups];b=ax.bar(xs+(i-1)*w,v,w*0.92,color=col,label=lab,linewidth=0)
    for xx,vv in zip(xs+(i-1)*w,v):
        ax.text(xx,vv+0.4,f"{vv:.1f}",ha="center",va="bottom",fontsize=5.6,color="#222")
ax.set_xticks(xs);ax.set_xticklabels([l for _,l in groups],fontsize=6.3);ax.set_ylabel("F1 (%)");ax.set_ylim(78,101);ax.legend(frameon=False,loc="lower center",bbox_to_anchor=(0.5,1.0),ncol=3,fontsize=6.3,columnspacing=1.0)
ax.grid(axis="x",visible=False)
fig.tight_layout();save(fig,"fig_strata")

# ---------- Fig 4: feature importance ----------
fig,ax=plt.subplots(figsize=(3.4,2.4))
imp=D["matcher"]["rf_importance"][:12]
pretty={"tsr_max":"token-set ratio (max)","tok_jac_max":"token Jaccard (max)","ratio_max":"Levenshtein ratio (max)","jw_max":"Jaro–Winkler (max)","exact_name":"exact normalised name","id_conflict":"identifier conflict","year_diff":"birth/inc. year gap","topics_jac":"topics Jaccard","date_match":"full date match","year_match":"year match","first_last_match":"first/last name overlap","addr_both":"address on both"}
ax.barh([pretty.get(a,a) for a,b in imp][::-1],[b for a,b in imp][::-1],color=BLUE,height=0.7,linewidth=0)
ax.set_xlabel("random-forest impurity importance");ax.grid(axis="y",visible=False)
fig.tight_layout();save(fig,"fig_importance")

# ---------- Fig 5: hard core profile ----------
fig,ax=plt.subplots(figsize=(3.4,2.2))
P=D["core"]["profile"];feats=[("date_missing","no date on\na side"),("date_both","dates on\nboth"),("id_both","IDs on\nboth"),("kind_org","org–org\npair"),("cross_script","cross-\nscript"),("exact_name","exact name\nshared"),("geo_conflict","country\nconflict")]
xs=np.arange(len(feats));w=0.38
ax.bar(xs-w/2,[P[f][0]*100 for f,_ in feats],w,color=RED,label="hard core (89 pairs, wrong for all 4 systems)",linewidth=0)
ax.bar(xs+w/2,[P[f][1]*100 for f,_ in feats],w,color=GRAY,label="rest of test set",linewidth=0)
ax.set_xticks(xs);ax.set_xticklabels([l for _,l in feats],fontsize=6.2);ax.set_ylabel("share of pairs (%)");ax.legend(frameon=False,fontsize=6.2,loc="upper right");ax.grid(axis="x",visible=False);ax.set_ylim(0,100)
fig.tight_layout();save(fig,"fig_core")

# ---------- Fig 6: complementarity stacked bars (slides + paper) ----------
fig,ax=plt.subplots(figsize=(3.4,1.6))
rows=[("Qwen2.5-7B, 0-shot",D["gate7"]["complementarity"]),("Qwen2.5-14B, 0-shot",D["gate14"]["complementarity"]),("Qwen2.5-7B, 4-shot",D["gatefs"]["complementarity"])]
cats=[("matcher_only","matcher right, LLM wrong",BLUE),("llm_only","LLM right, matcher wrong",ORANGE),("both_wrong","both wrong",RED)]
ys=np.arange(len(rows))[::-1]
for yy,(lab,c) in zip(ys,rows):
    left=0
    for key,cl,col in cats:
        v=c[key]*100;ax.barh(yy,v,left=left,color=col,height=0.6,linewidth=0,label=cl if yy==ys[0] else None)
        ax.text(left+v/2,yy,f"{v:.2f}",ha="center",va="center",fontsize=6,color="white" if key!="llm_only" else "#222")
        left+=v
ax.set_yticks(ys);ax.set_yticklabels([r[0] for r in rows],fontsize=6.5);ax.set_xlabel("share of 18,000 test pairs (%); remaining pairs both right");ax.legend(frameon=False,fontsize=6,loc="lower center",bbox_to_anchor=(0.5,1.0),ncol=3,columnspacing=0.8);ax.grid(axis="y",visible=False)
fig.tight_layout();save(fig,"fig_complementarity")

# ---------- Fig 7: per-type error rates ----------
fig,ax=plt.subplots(figsize=(3.4,2.2))
B=D["extra"]["breakdown"];order=[("person-person","person–\nperson"),("org-org","org–\norg"),("same-script","same\nscript"),("cross-script","cross\nscript"),("dates on both","dates\non both"),("no dates on a side","no date\non a side"),("ids on both","IDs\non both")]
xs=np.arange(len(order));w=0.38
ax.bar(xs-w/2,[B[k]["matcher_err"]*100 for k,_ in order],w,color=BLUE,label="RF matcher",linewidth=0)
ax.bar(xs+w/2,[B[k]["llm_err"]*100 for k,_ in order],w,color=ORANGE,label="Qwen2.5-7B zero-shot",linewidth=0)
ax.set_xticks(xs);ax.set_xticklabels([l for _,l in order],fontsize=6.2);ax.set_ylabel("error rate (%)");ax.legend(frameon=False,fontsize=6.3);ax.grid(axis="x",visible=False)
fig.tight_layout();save(fig,"fig_pertype")

# ---------- Fig 8: model comparison bar (slides) ----------
fig,ax=plt.subplots(figsize=(3.6,2.0))
M=D["matcher"];names=[("logreg","Log.\nreg."),("dtree","Decision\ntree"),("adaboost","Ada-\nBoost"),("svm_rbf","SVM\n(RBF)"),("knn","kNN"),("hgb","Grad.\nboost"),("rf","Random\nforest")]
vals=[M[k]["test@0.5"]["f1"]*100 for k,_ in names]
ax.bar(range(len(names)),vals,color=BLUE,width=0.65,linewidth=0)
for i,v in enumerate(vals):
    ax.text(i,v+0.05,f"{v:.2f}",ha="center",va="bottom",fontsize=6)
ax.axhline(D["extra"]["rules"]["tsr_max"]["test_f1"]*100,color=RED,ls=":",lw=1,label="best single-threshold rule (94.6)")
ax.axhline(95.83,color=GRAY,ls="--",lw=1,label="Qwen2.5-7B zero-shot (95.8)")
ax.axhline(96.58,color=VIOLET,ls="--",lw=1,label="Qwen2.5-14B zero-shot (96.6)")
ax.legend(frameon=False,fontsize=6,loc="lower center",bbox_to_anchor=(0.5,1.0),ncol=3,columnspacing=0.8,handlelength=1.6)
ax.set_xticks(range(len(names)));ax.set_xticklabels([l for _,l in names],fontsize=6.3);ax.set_ylabel("test F1 (%)");ax.set_ylim(94,99.6);ax.grid(axis="x",visible=False)
fig.tight_layout();save(fig,"fig_matchers")

# ---------- Fig 1c: human panel alone (slides) ----------
fig,ax=plt.subplots(figsize=(4.2,2.7))
H=D["gate7"]["curves"]["human"]
for key,lab,col,ls,mk in [("oracle","oracle gate","#333333",":","" ),("confidence","confidence gate $\\min(p,1-p)$",BLUE,"-","o"),("hgb_err","learned gate (GBT)",ORANGE,"--","s"),("random","random gate",GRAY,"-.","")]:
    x,y=xy(H[key],xmax=0.30);ax.plot(x,y,ls,color=col,marker=mk,ms=3.5,label=lab)
ax.set_xlabel("pairs deferred to human review (%)");ax.set_ylabel("system F1 (%)");ax.set_ylim(98.6,100.05);ax.set_xlim(-0.5,30.5);ax.legend(loc="lower right",frameon=False,fontsize=7.5)
ax.annotate("5% deferred: 99.63",xy=(5,99.632),xytext=(9,99.3),fontsize=7,color=BLUE,arrowprops=dict(arrowstyle="-",lw=0.5,color=BLUE))
ax.annotate("no deferral: 98.72",xy=(0,98.715),xytext=(3.5,98.63),fontsize=7,color="#333",arrowprops=dict(arrowstyle="-",lw=0.5,color="#333"))
fig.tight_layout();save(fig,"fig_deferral_human")

# ---------- Fig 9/10: decision-theory extras (noisy human, asymmetric costs) ----------
if "decision" in D:
    N=D["decision"]["noisy_human"]
    fig,ax=plt.subplots(figsize=(4.2,2.6))
    for eps,col,mk in [("0",BLUE,"o"),("0.02",AQUA,"s"),("0.05",YELLOW,"D"),("0.1",ORANGE,"^"),("0.2",RED,"v")]:
        x,y=xy(N[eps],xmax=0.2);ax.plot(x,y,"-",color=col,marker=mk,ms=3.2,label=f"reviewer wrong {int(round(float(eps)*100))}% of the time")
    ax.axhline(98.715,color="#333",lw=0.6,alpha=0.5);ax.text(12.5,98.52,"matcher alone 98.72",fontsize=6.5,color="#333")
    ax.set_xlabel("pairs deferred to human review (%)");ax.set_ylabel("system F1 (%)");ax.set_ylim(96.6,100.1);ax.set_xlim(-0.5,20.5);ax.legend(frameon=False,fontsize=6.5,loc="lower left")
    fig.tight_layout();save(fig,"fig_noisy_human")
    C=D["decision"]["asymmetric_cost"]
    fig,ax=plt.subplots(figsize=(4.2,2.6))
    for cfn,col,mk in [(1,BLUE,"o"),(3,ORANGE,"s"),(10,RED,"^")]:
        xs=[];ys=[]
        for cd in [0.5,0.2,0.1,0.05]:
            v=C[f"cfn={cfn},cd={cd}"];xs.append(v["defer_frac"]*100);ys.append(v["cost_per_1000"])
            ax.annotate(f"$c_d$={cd}",xy=(xs[-1],ys[-1]),xytext=(3,3),textcoords="offset points",fontsize=5.5,color=col)
        ax.plot(xs,ys,"-",color=col,marker=mk,ms=3.5,label=f"missed match costs {cfn}$\\times$ a false match")
        ax.axhline(C[f"cfn={cfn},cd=0.1"]["no_reject_cost_per_1000"],color=col,ls=":",lw=1)
    ax.text(11.5,C["cfn=10,cd=0.1"]["no_reject_cost_per_1000"]+0.6,"dotted: best single threshold, no reject option",fontsize=6,color="#333",ha="right")
    ax.set_xlabel("pairs deferred to human review (%)");ax.set_ylabel("cost per 1000 pairs");ax.set_xlim(-0.5,12.5);ax.set_ylim(0,42);ax.legend(frameon=False,fontsize=6.5,loc="upper right",bbox_to_anchor=(1.0,0.86))
    fig.tight_layout();save(fig,"fig_cost")

# ---------- Fig 11: scaling the expert (7B, 14B, 32B zero-shot / CoT / thinking vs human) ----------
if "llm32" in D:
    L=D["llm32"];E=L["experts"]
    fig,axes=plt.subplots(1,2,figsize=(7.0,2.5),gridspec_kw={"width_ratios":[1.25,1]})
    ax=axes[0]
    x,y=xy(L["human_curve"],xmax=0.1);ax.plot(x,y,"-",color=AQUA,marker="D",ms=3,label="human analyst")
    for name,col,mk,ls in [("Qwen2.5-7B zero-shot",GRAY,"o","-"),("Qwen2.5-14B zero-shot",VIOLET,"^","-"),("Qwen3-32B zero-shot",BLUE,"s","-"),("Qwen3-32B chain of thought",ORANGE,"v","--"),("Qwen3-32B thinking",RED,"P","--")]:
        x,y=xy(E[name]["curve"],xmax=0.1);ax.plot(x,y,ls,color=col,marker=mk,ms=3,label=name)
    ax.set_xlabel("pairs deferred (%)");ax.set_ylabel("system F1 (%)");ax.set_ylim(98.55,100.0);ax.set_xlim(-0.3,10.3);ax.legend(frameon=False,fontsize=6,loc="upper left")
    ax=axes[1]
    names=[("Qwen2.5-7B zero-shot","7B\n0-shot"),("Qwen2.5-14B zero-shot","14B\n0-shot"),("Qwen3-32B zero-shot","32B\n0-shot"),("Qwen3-32B chain of thought","32B\nCoT"),("Qwen3-32B thinking","32B\nthinking")]
    vals=[E[n]["deferred5"]["expert_error_rate"]*100 for n,_ in names]
    ax.bar(range(len(names)),vals,color=[GRAY,VIOLET,BLUE,ORANGE,RED],width=0.62,linewidth=0)
    for i,v in enumerate(vals):
        ax.text(i,v+0.4,f"{v:.1f}",ha="center",va="bottom",fontsize=6)
    m=E["Qwen2.5-7B zero-shot"]["deferred5"];mrate=m["matcher_errors"]/m["n"]*100
    ax.axhline(mrate,color="#333",ls="--",lw=0.9);ax.text(4.3,mrate+0.5,f"matcher itself: {mrate:.1f}",fontsize=6,color="#333",ha="right")
    ax.set_xticks(range(len(names)));ax.set_xticklabels([l for _,l in names],fontsize=6.3);ax.set_ylabel("error rate on the 5% deferred pairs (%)");ax.set_ylim(0,29);ax.grid(axis="x",visible=False)
    fig.tight_layout(w_pad=1.5);save(fig,"fig_experts")
print("figs written:",sorted(os.listdir(FIG)))
