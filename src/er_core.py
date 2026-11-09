import os,json,itertools,collections
import numpy as np
import er_lib as E
ROOT=os.environ.get("ER_ROOT",os.path.expanduser("~/prml_proj/er"));DATA=f"{ROOT}/data"
rows=E.load_rows();sp=json.load(open(f"{DATA}/split.json"));te=np.array(sp["test"])
F=np.load(f"{DATA}/feats.npz",allow_pickle=True);X=F["X"];y=F["y"];names=list(F["names"])
P=np.load(f"{DATA}/matcher_probs.npz");pc=P["rf_cal"];pm=(pc>=0.5).astype(int)
tags=["Qwen2.5-7B-Instruct","Qwen2.5-14B-Instruct-AWQ","Qwen2.5-7B-Instruct_4shot"]
L={}
for t in tags:
    d={}
    for line in open(f"{DATA}/llm_{t}_main.jsonl"):
        r=json.loads(line);d[r["idx"]]=r["pred"]
    L[t]=np.array([d[i] for i in te])
yt=y[te];mw=pm[te]!=yt
both={t:set(te[mw&(L[t]!=yt)]) for t in tags}
llmonly={t:set(te[(~mw)&(L[t]!=yt)]) for t in tags}
out={"matcher_wrong":int(mw.sum())}
print("matcher wrong on test:",mw.sum())
for t in tags:
    print(f"{t:28s} both-wrong={len(both[t])} llm-wrong-total={(L[t]!=yt).sum()}")
    out[t]={"both_wrong":len(both[t]),"llm_wrong":int((L[t]!=yt).sum())}
for a,b in itertools.combinations(tags,2):
    j=len(both[a]&both[b])/len(both[a]|both[b])
    print(f"both-wrong overlap {a.split('-')[1]} vs {b.split('-')[1].replace('Instruct_','')}: inter={len(both[a]&both[b])} jaccard={j:.3f}")
    out[f"jaccard_{a}_{b}"]=j
core=both[tags[0]]&both[tags[1]]&both[tags[2]]
print("pairs wrong for matcher AND all three LLM configs:",len(core))
out["core_all"]=len(core)
# characterise the core: label distribution + feature means vs rest of test
ci=np.array(sorted(core));rest=np.array([i for i in te if i not in core])
print("core label mean (1=analyst said match):",y[ci].mean(),"| matcher said match on core:",pm[ci].mean())
for f in ["exact_name","jw_max","date_both","date_match","year_match","id_both","id_match","id_conflict","geo_conflict","cross_script","kind_org","date_missing"]:
    j=names.index(f);print(f"  {f:14s} core={X[ci,j].mean():.3f} rest={X[rest,j].mean():.3f}")
    out[f"core_{f}"]=float(X[ci,j].mean());out[f"rest_{f}"]=float(X[rest,j].mean())
# dump 12 examples (pid, label, matcher prob, rendered)
with open(f"{DATA}/core_examples.txt","w") as fh:
    rng=np.random.default_rng(0)
    for i in rng.choice(ci,min(12,len(ci)),replace=False):
        fh.write(f"===== pid={rows[i]['pid']} analyst_label={y[i]} matcher_p={pc[i]:.3f} llm7={L[tags[0]][list(te).index(i)]}\n{E.render(rows[i]['left'])}\n--- vs ---\n{E.render(rows[i]['right'])}\n\n")
json.dump(out,open(f"{DATA}/core_results.json","w"),indent=1)
print("CORE_DONE")
