import os,sys,json,gzip,re,random,argparse,unicodedata,collections,time
import numpy as np

ROOT=os.environ.get("ER_ROOT",os.path.expanduser("~/prml_proj/er"))
DATA=f"{ROOT}/data"
RAW=os.environ.get("ER_RAW","/scratch/hchauhan_prml/pairs-20251209.json.gz")
os.makedirs(DATA,exist_ok=True)

NAME_KEYS=["name","alias","weakAlias","previousName"]
KEEP=["name","alias","weakAlias","previousName","firstName","lastName","middleName","secondName","fatherName","birthDate","birthPlace","birthCountry","deathDate","gender","nationality","citizenship","country","jurisdiction","mainCountry","flag","address","position","title","idNumber","passportNumber","registrationNumber","taxNumber","imoNumber","mmsi","callSign","innCode","ogrnCode","leiCode","vatCode","swiftBic","okpoCode","kppCode","dunsCode","wikidataId","incorporationDate","dissolutionDate","legalForm","email","phone","website","topics"]
SCHEMAS={"Person","Organization","Company","LegalEntity","PublicBody","Vessel","Airplane","Asset","Security","CryptoWallet"}
IDK=["idNumber","passportNumber","registrationNumber","taxNumber","imoNumber","mmsi","callSign","innCode","ogrnCode","leiCode","vatCode","swiftBic","okpoCode","kppCode","dunsCode","wikidataId"]
GEO=["nationality","citizenship","country","birthCountry","jurisdiction","mainCountry","flag"]

def strip(e):
    p=e.get("properties",{}) or {}
    return {"id":e["id"],"caption":e.get("caption",""),"schema":e.get("schema",""),"props":{k:p[k] for k in KEEP if k in p and p[k]}}

def sample(n_pos,n_neg,seed=0):
    rng=random.Random(seed)
    res={"positive":[],"negative":[]};seen=set();cnt=collections.Counter();tot=0
    with gzip.open(RAW,"rt") as f:
        for line in f:
            r=json.loads(line);tot+=1
            j=r.get("judgement")
            if j not in res:
                continue
            a,b=r["left"]["id"],r["right"]["id"]
            if a==b:
                continue
            if r["left"].get("schema") not in SCHEMAS or r["right"].get("schema") not in SCHEMAS:
                cnt["skipped_schema"]+=1
                continue
            key=(a,b) if a<b else (b,a)
            if key in seen:
                continue
            seen.add(key);cnt[j]+=1
            cap=n_pos if j=="positive" else n_neg
            row={"label":int(j=="positive"),"left":strip(r["left"]),"right":strip(r["right"])}
            k=cnt[j]
            if k<=cap:
                res[j].append(row)
            else:
                i=rng.randrange(k)
                if i<cap:
                    res[j][i]=row
    rows=res["positive"]+res["negative"]
    rng.shuffle(rows)
    for i,r in enumerate(rows):
        r["pid"]=i
    with open(f"{DATA}/pairs_sample.jsonl","w") as f:
        for r in rows:
            f.write(json.dumps(r,ensure_ascii=False)+"\n")
    json.dump({"total_lines":tot,"unique_pairs":dict(cnt),"sampled":{k:len(v) for k,v in res.items()}},open(f"{DATA}/sample_stats.json","w"),indent=1)
    print("lines",tot,"unique",dict(cnt),"sampled",{k:len(v) for k,v in res.items()},flush=True)

def load_rows():
    return [json.loads(l) for l in open(f"{DATA}/pairs_sample.jsonl")]

def script_of(s):
    c=collections.Counter()
    for ch in s:
        if ch.isalpha():
            try:
                nm=unicodedata.name(ch).split()[0]
            except ValueError:
                nm="OTHER"
            c[nm]+=1
    return c.most_common(1)[0][0] if c else "NONE"

def norm(s):
    from unidecode import unidecode
    s=unidecode(s).lower()
    s=re.sub(r"[^a-z0-9 ]+"," ",s)
    return re.sub(r"\s+"," ",s).strip()

def names(e,cap=12):
    out=[]
    if e["caption"]:
        out.append(e["caption"])
    for k in NAME_KEYS:
        out+=e["props"].get(k,[])
    seen=set();res=[]
    for n in out:
        nn=norm(n)
        if nn and nn not in seen:
            seen.add(nn);res.append(nn)
    return res[:cap]

def years(e):
    ys=set()
    for d in e["props"].get("birthDate",[])+e["props"].get("incorporationDate",[]):
        m=re.match(r"(\d{4})",d)
        if m:
            ys.add(int(m.group(1)))
    return ys

def fulldates(e):
    return set(d for d in e["props"].get("birthDate",[]) if re.match(r"\d{4}-\d{2}-\d{2}$",d))

def geo(e):
    g=set()
    for k in GEO:
        g.update(x.lower() for x in e["props"].get(k,[]))
    return g

def ids(e):
    s=set()
    for k in IDK:
        s.update(re.sub(r"[^a-z0-9]","",x.lower()) for x in e["props"].get(k,[]))
    return s

def kind(schema):
    return "person" if schema=="Person" else ("vessel" if schema=="Vessel" else "org")

FEATS=["jw_max","tsr_max","tok_jac_max","ratio_max","exact_name","first_last_match","n_names_l","n_names_r","cross_script","kind_person","kind_org","kind_mismatch","date_both","date_match","year_match","year_diff","date_missing","geo_both","geo_jac","geo_any","geo_conflict","gender_both","gender_mismatch","id_both","id_match","id_conflict","addr_both","addr_jac","nprops_l","nprops_r","nprops_min","topics_jac","pos_jac","bplace_match"]

def pair_feats(l,r):
    from rapidfuzz import fuzz
    from rapidfuzz.distance import JaroWinkler
    nl,nr=names(l),names(r)
    jw=tsr=tj=ra=0.0
    for a in nl:
        ta=set(a.split())
        for b in nr:
            jw=max(jw,JaroWinkler.similarity(a,b))
            tsr=max(tsr,fuzz.token_set_ratio(a,b)/100)
            ra=max(ra,fuzz.ratio(a,b)/100)
            tb=set(b.split())
            if ta and tb:
                tj=max(tj,len(ta&tb)/len(ta|tb))
    exact=float(bool(set(nl)&set(nr)))
    fl_l={norm(x) for x in l["props"].get("firstName",[])}|{norm(x) for x in l["props"].get("lastName",[])}
    fl_r={norm(x) for x in r["props"].get("firstName",[])}|{norm(x) for x in r["props"].get("lastName",[])}
    flm=len(fl_l&fl_r)/max(1,len(fl_l|fl_r)) if fl_l and fl_r else 0.0
    cs=float(script_of(l["caption"])!=script_of(r["caption"]))
    kl,kr=kind(l["schema"]),kind(r["schema"])
    yl,yr=years(l),years(r);dl,dr=fulldates(l),fulldates(r)
    date_both=float(bool(yl and yr))
    date_match=float(bool(dl&dr))
    year_match=float(bool(yl&yr))
    ydiff=min(abs(a-b) for a in yl for b in yr) if yl and yr else -1
    gl,gr=geo(l),geo(r)
    gjac=len(gl&gr)/len(gl|gr) if gl and gr else 0.0
    gel=set(x.lower() for x in l["props"].get("gender",[]));ger=set(x.lower() for x in r["props"].get("gender",[]))
    il,ir=ids(l),ids(r)
    al=[norm(x) for x in l["props"].get("address",[])];ar=[norm(x) for x in r["props"].get("address",[])]
    ajac=0.0
    for a in al[:5]:
        for b in ar[:5]:
            ta,tb=set(a.split()),set(b.split())
            if ta and tb:
                ajac=max(ajac,len(ta&tb)/len(ta|tb))
    tl=set(l["props"].get("topics",[]));tr=set(r["props"].get("topics",[]))
    pl={norm(x) for x in l["props"].get("position",[])};pr={norm(x) for x in r["props"].get("position",[])}
    bl={norm(x) for x in l["props"].get("birthPlace",[])};br={norm(x) for x in r["props"].get("birthPlace",[])}
    f=[jw,tsr,tj,ra,exact,flm,np.log1p(len(nl)),np.log1p(len(nr)),cs,float(kl=="person" and kr=="person"),float(kl=="org" and kr=="org"),float(kl!=kr),date_both,date_match,year_match,ydiff,float(not(yl and yr)),float(bool(gl and gr)),gjac,float(bool(gl&gr)),float(bool(gl and gr) and not (gl&gr)),float(bool(gel and ger)),float(bool(gel and ger) and not (gel&ger)),float(bool(il and ir)),float(bool(il&ir)),float(bool(il and ir) and not (il&ir)),float(bool(al and ar)),ajac,np.log1p(len(l["props"])),np.log1p(len(r["props"])),np.log1p(min(len(l["props"]),len(r["props"]))),len(tl&tr)/len(tl|tr) if (tl or tr) else 0.0,len(pl&pr)/len(pl|pr) if (pl or pr) else 0.0,float(bool(bl&br))]
    return f

def features():
    rows=load_rows()
    t0=time.time()
    X=np.array([pair_feats(r["left"],r["right"]) for r in rows],dtype=np.float32)
    y=np.array([r["label"] for r in rows])
    pid=np.array([r["pid"] for r in rows])
    np.savez(f"{DATA}/feats.npz",X=X,y=y,pid=pid,names=np.array(FEATS))
    print("features",X.shape,"pos",y.mean(),"time",round(time.time()-t0,1),flush=True)

def split(seed=0):
    rows=load_rows()
    rng=np.random.default_rng(seed)
    pid=np.array([r["pid"] for r in rows]);y=np.array([r["label"] for r in rows])
    idx=rng.permutation(len(rows))
    n=len(rows);ntr=int(0.6*n);nva=int(0.1*n)
    sp={"train":idx[:ntr].tolist(),"val":idx[ntr:ntr+nva].tolist(),"test":idx[ntr+nva:].tolist()}
    json.dump(sp,open(f"{DATA}/split.json","w"))
    print({k:len(v) for k,v in sp.items()},{k:float(y[v].mean()) for k,v in sp.items()},flush=True)

def render(e,maxn=6):
    p=e["props"]
    lines=[f"type: {e['schema']}",f"name: {e['caption']}"]
    al=[x for k in ["name","alias","weakAlias","previousName"] for x in p.get(k,[]) if x!=e["caption"]]
    seen=set();al2=[]
    for a in al:
        if a not in seen:
            seen.add(a);al2.append(a)
    if al2:
        lines.append("other names: "+"; ".join(al2[:maxn]))
    for k,lab in [("birthDate","birth date"),("incorporationDate","incorporation date"),("dissolutionDate","dissolution date"),("gender","gender"),("nationality","nationality"),("citizenship","citizenship"),("country","country"),("jurisdiction","jurisdiction"),("flag","flag"),("birthPlace","birth place"),("address","address"),("position","position"),("legalForm","legal form"),("idNumber","id number"),("passportNumber","passport"),("registrationNumber","registration no"),("taxNumber","tax no"),("innCode","INN"),("ogrnCode","OGRN"),("leiCode","LEI"),("vatCode","VAT"),("imoNumber","IMO"),("mmsi","MMSI"),("callSign","call sign"),("wikidataId","wikidata")]:
        if p.get(k):
            lines.append(f"{lab}: "+"; ".join(p[k][:4]))
    return "\n".join(lines)

PROMPT="You are an expert analyst doing entity resolution on sanctions and watchlist data. Decide whether the two records below refer to the same real-world entity (same person, company or vessel). Names may be transliterated differently, abbreviated or only partially listed, and the two sources may record different subsets of fields. Minor inconsistencies in dates or identifiers are often data-entry errors and do not by themselves rule out a match. Different people can share a name, so use dates, places and identifiers to disambiguate.\n\nRecord A:\n{a}\n\nRecord B:\n{b}\n\nAnswer with exactly one word: yes or no."

def fewshot_block(rows,sp,k):
    rng=random.Random(1);pool=sp["train"][-4000:]
    pos=[i for i in pool if rows[i]["label"]==1];neg=[i for i in pool if rows[i]["label"]==0]
    ex=rng.sample(pos,k//2)+rng.sample(neg,k-k//2);rng.shuffle(ex)
    parts=["Here are some solved examples:"]
    for j,i in enumerate(ex):
        parts.append(f"Example {j+1}\nRecord A:\n{render(rows[i]['left'])}\n\nRecord B:\n{render(rows[i]['right'])}\n\nAnswer: {'yes' if rows[i]['label']==1 else 'no'}")
    return "\n\n".join(parts)+"\n\nNow the real question.\n\n"

def llm(model,which,maxn=None,shots=0):
    from vllm import LLM,SamplingParams
    rows=load_rows();sp=json.load(open(f"{DATA}/split.json"))
    sel=sp["test"]+sp["train"][:12000] if which=="main" else sp[which]
    if maxn:
        sel=sel[:maxn]
    llm=LLM(model=model,gpu_memory_utilization=0.85,max_model_len=6144,seed=0)
    tok=llm.get_tokenizer()
    fs=fewshot_block(rows,sp,shots) if shots else ""
    prompts=[tok.apply_chat_template([{"role":"user","content":fs+PROMPT.format(a=render(rows[i]["left"]),b=render(rows[i]["right"]))}],tokenize=False,add_generation_prompt=True) for i in sel]
    s=SamplingParams(temperature=0,max_tokens=3,logprobs=10)
    t0=time.time();res=llm.generate(prompts,s);dt=time.time()-t0
    out=[]
    for i,r in zip(sel,res):
        o=r.outputs[0];txt=o.text.strip().lower()
        py=pn=0.0
        if o.logprobs:
            for tid,lp in o.logprobs[0].items():
                t=(lp.decoded_token or "").strip().lower()
                if t in("yes","y"):
                    py+=np.exp(lp.logprob)
                elif t in("no","n"):
                    pn+=np.exp(lp.logprob)
        pred=1 if txt.startswith("yes") else (0 if txt.startswith("no") else (1 if py>pn else 0))
        out.append({"pid":int(rows[i]["pid"]),"idx":int(i),"pred":pred,"p_yes":float(py/(py+pn)) if (py+pn)>0 else float(pred),"raw":o.text,"ntok_in":len(r.prompt_token_ids)})
    tag=model.split("/")[-1]+(f"_{shots}shot" if shots else "")
    with open(f"{DATA}/llm_{tag}_{which}.jsonl","w") as f:
        for x in out:
            f.write(json.dumps(x)+"\n")
    y=np.array([rows[x["idx"]]["label"] for x in out]);p=np.array([x["pred"] for x in out])
    from sklearn.metrics import f1_score,accuracy_score
    print(f"[llm] {model} {which} n={len(out)} acc={accuracy_score(y,p):.4f} f1={f1_score(y,p):.4f} time={dt:.1f}s avg_in_tok={np.mean([x['ntok_in'] for x in out]):.0f}",flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("cmd");ap.add_argument("--npos",type=int,default=39000);ap.add_argument("--nneg",type=int,default=21000);ap.add_argument("--model");ap.add_argument("--which",default="main");ap.add_argument("--maxn",type=int);ap.add_argument("--shots",type=int,default=0)
    a=ap.parse_args()
    if a.cmd=="sample":
        sample(a.npos,a.nneg)
    elif a.cmd=="features":
        features()
    elif a.cmd=="split":
        split()
    elif a.cmd=="llm":
        llm(a.model,a.which,a.maxn,a.shots)
