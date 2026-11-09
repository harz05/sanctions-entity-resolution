"""Collect the per-script result files in data/ into one results_all.json (consumed by make_figs.py and the write-up)."""
import os,json
ROOT=os.environ.get("ER_ROOT",os.path.expanduser("~/prml_proj/er"));DATA=f"{ROOT}/data"
HERE=os.path.dirname(os.path.abspath(__file__))
def J(name):
    p=f"{DATA}/{name}"
    return json.load(open(p)) if os.path.exists(p) else None
out={}
g7=J("gate_results_Qwen2.5-7B-Instruct.json");g14=J("gate_results_Qwen2.5-14B-Instruct-AWQ.json");gfs=J("gate_results_Qwen2.5-7B-Instruct_4shot.json")
out["gate7"]=g7;out["gate14"]=g14;out["gatefs"]=gfs
out["matcher"]=J("matcher_results.json")
out["extra"]=J("extra_results.json")
out["xtype"]=J("xtype_results.json")
st=J("strata_results.json")
if st:
    ren={"Qwen2.5-7B-Instruct":"llm7","Qwen2.5-14B-Instruct-AWQ":"llm14","Qwen2.5-7B-Instruct_4shot":"llm7fs"}
    S={}
    for k,v in st.items():
        if not isinstance(v,dict):
            S[k]=v;continue
        d={}
        for kk,vv in v.items():
            m=[t for t in ren if kk.endswith("_"+t)]
            d[(kk[:-len(m[0])-1].replace("llm_",ren[m[0]]+"_")) if m else kk]=vv
        S[k]=d
    out["strata"]=S
co=J("core_results.json")
if co:
    feats=sorted({k[5:] for k in co if k.startswith("core_") and k!="core_all"})
    out["core"]={"matcher_wrong":co["matcher_wrong"],"core_all":co["core_all"],"profile":{f:[co[f"core_{f}"],co[f"rest_{f}"]] for f in feats},
        **{f"both_wrong_{t}":co[n]["both_wrong"] for t,n in [("7b","Qwen2.5-7B-Instruct"),("14b","Qwen2.5-14B-Instruct-AWQ"),("7bfs","Qwen2.5-7B-Instruct_4shot")] if n in co},
        **{k:v for k,v in co.items() if k.startswith("jaccard")}}
out["leak"]=J("leakcheck_results.json")
ab=J("ablation.json")
if ab:
    out["ablation"]=ab.get("ablation",ab);out["class_stats"]=ab.get("stats")
out["llm_runs"]={}
for tag,g in [("Qwen2.5-7B-Instruct",g7),("Qwen2.5-14B-Instruct-AWQ",g14),("Qwen2.5-7B-Instruct_4shot",gfs)]:
    if g:
        out["llm_runs"][tag]={"test_f1":g["llm_f1"]}
out["fullpop"]=J("fullpop_results.json")
out["decision"]=J("decision_results.json")
out["llm32"]=J("llm32_results.json")
out["data"]=J("sample_stats.json")
dest=f"{HERE}/results_all.json" if os.path.exists(f"{HERE}/make_figs.py") and not os.path.isdir(f"{os.path.dirname(HERE)}/results") else f"{os.path.dirname(HERE)}/results/results_all.json"
json.dump(out,open(dest,"w"),indent=1)
print("wrote",dest,"sections:",[k for k,v in out.items() if v is not None])
