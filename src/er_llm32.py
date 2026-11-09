"""Scaling the expert: a 32B model (Qwen3-32B-AWQ) as the deferral expert.

  python er_llm32.py zs      zero-shot one-word answers on the 18k test + 12k gate-training pairs (same protocol as the 7B/14B runs)
  python er_llm32.py cot     short chain of thought, then "Answer: yes/no", on the 10% least-confident test pairs plus every test pair the matcher gets wrong
  python er_llm32.py think   native thinking mode on the 5% least-confident test pairs plus every test pair the matcher gets wrong
Outputs data/llm_<tag>_main.jsonl (zs) and data/llm32_<mode>.jsonl (cot, think).
"""
import os,sys,json,re,time
import numpy as np
import er_lib as E
from sklearn.metrics import f1_score,accuracy_score

MODEL=os.environ.get("ER_LLM32","Qwen/Qwen3-32B-AWQ");TAG=MODEL.split("/")[-1]
MODE=sys.argv[1] if len(sys.argv)>1 else "zs"
DATA=E.DATA
COT_SUFFIX="Think step by step in at most five short sentences, comparing names, dates, places and identifiers. Then give the final answer on its own last line as exactly 'Answer: yes' or 'Answer: no'."

def subsets():
    sp=json.load(open(f"{DATA}/split.json"));te=np.array(sp["test"])
    F=np.load(f"{DATA}/feats.npz",allow_pickle=True);y=F["y"]
    P=np.load(f"{DATA}/matcher_probs.npz");p=P["rf_cal"]
    conf=np.minimum(p[te],1-p[te]);order=te[np.argsort(-conf)]
    wrong=te[((p[te]>=0.5).astype(int)!=y[te])]
    return sp,te,order,wrong

def parse_answer(text):
    t=text.lower()
    m=re.findall(r"answer\s*[:\-]?\s*\**\s*(yes|no)\b",t)
    if m:
        return 1 if m[-1]=="yes" else 0
    m=re.findall(r"\b(yes|no)\b",t)
    if m:
        return 1 if m[-1]=="yes" else 0
    return None

def main():
    from vllm import LLM,SamplingParams
    rows=E.load_rows();sp,te,order,wrong=subsets()
    if MODE=="zs":
        sel=list(sp["test"])+list(sp["train"][:12000]);think=False;suffix=None;maxtok=3;maxlen=6144
        s=SamplingParams(temperature=0,max_tokens=maxtok,logprobs=10)
    elif MODE=="cot":
        sel=sorted(set(order[:int(0.10*len(te))].tolist())|set(wrong.tolist()));think=False;suffix=COT_SUFFIX;maxlen=6144
        s=SamplingParams(temperature=0,max_tokens=600)
    elif MODE=="think":
        sel=sorted(set(order[:int(0.05*len(te))].tolist())|set(wrong.tolist()));think=True;suffix=None;maxlen=8192
        s=SamplingParams(temperature=0.6,top_p=0.95,top_k=20,max_tokens=6144,seed=0)
    else:
        raise SystemExit("mode must be zs, cot or think")
    print(f"[llm32] {MODEL} mode={MODE} n={len(sel)}",flush=True)
    llm=LLM(model=MODEL,gpu_memory_utilization=0.88,max_model_len=maxlen,seed=0)
    tok=llm.get_tokenizer()
    def prompt(i):
        txt=E.PROMPT.format(a=E.render(rows[i]["left"]),b=E.render(rows[i]["right"]))
        if suffix:
            txt=txt.replace("Answer with exactly one word: yes or no.",suffix)
        return tok.apply_chat_template([{"role":"user","content":txt}],tokenize=False,add_generation_prompt=True,enable_thinking=think)
    prompts=[prompt(i) for i in sel]
    t0=time.time();res=llm.generate(prompts,s);dt=time.time()-t0
    out=[];ntrunc=0;gen_tok=0
    for i,r in zip(sel,res):
        o=r.outputs[0];gen_tok+=len(o.token_ids)
        if MODE=="zs":
            txt=o.text.strip().lower();py=pn=0.0
            if o.logprobs:
                for tid,lp in o.logprobs[0].items():
                    t=(lp.decoded_token or "").strip().lower()
                    if t in("yes","y"):
                        py+=np.exp(lp.logprob)
                    elif t in("no","n"):
                        pn+=np.exp(lp.logprob)
            pred=1 if txt.startswith("yes") else (0 if txt.startswith("no") else (1 if py>pn else 0))
            out.append({"pid":int(rows[i]["pid"]),"idx":int(i),"pred":pred,"p_yes":float(py/(py+pn)) if (py+pn)>0 else float(pred),"raw":o.text,"ntok_in":len(r.prompt_token_ids)})
        else:
            text=o.text;ans_part=text.split("</think>")[-1] if "</think>" in text else text
            pred=parse_answer(ans_part)
            if pred is None and MODE=="think":
                pred=parse_answer(text[-300:])
            truncated=(o.finish_reason=="length")
            ntrunc+=int(truncated)
            out.append({"pid":int(rows[i]["pid"]),"idx":int(i),"pred":pred,"truncated":truncated,"ntok_out":len(o.token_ids),"raw":text[-1500:],"ntok_in":len(r.prompt_token_ids)})
    fn=f"{DATA}/llm_{TAG}_main.jsonl" if MODE=="zs" else f"{DATA}/llm32_{MODE}.jsonl"
    with open(fn,"w") as f:
        for x in out:
            f.write(json.dumps(x,ensure_ascii=False)+"\n")
    y=np.array([rows[x["idx"]]["label"] for x in out]);pred=np.array([-1 if x["pred"] is None else x["pred"] for x in out]);ok=pred>=0
    print(f"[llm32] {MODE} n={len(out)} answered={ok.sum()} truncated={ntrunc} acc={accuracy_score(y[ok],pred[ok]):.4f} f1={f1_score(y[ok],pred[ok]):.4f} time={dt:.0f}s gen_tok={gen_tok} out_tok/pair={gen_tok/len(out):.0f}",flush=True)
    if MODE!="zs":
        P=np.load(f"{DATA}/matcher_probs.npz");pm=(P["rf_cal"]>=0.5).astype(int)
        idx=np.array([x["idx"] for x in out]);mw=pm[idx]!=y
        fixed=((pred==y)&mw).sum();print(f"[llm32] {MODE} matcher-wrong pairs in subset={mw.sum()} fixed by LLM={fixed} LLM wrong where matcher right={((pred!=y)&(~mw)&ok).sum()}",flush=True)
    print(f"LLM32_{MODE.upper()}_DONE",flush=True)

if __name__=="__main__":
    main()
