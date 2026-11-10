import json
cells=[]
def md(s): cells.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
def code(s): cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.strip("\n")})

import os as _os
_SRCDIR=_os.path.dirname(_os.path.abspath(__file__))
def _lines(f,a,b):
    return "\n".join(open(f"{_SRCDIR}/{f}").read().split("\n")[a-1:b])
def codeview(title,parts):
    """A markdown cell showing the exact source lines that the following stage runs."""
    body=f"**The code that runs here.** {title}\n\n"
    for f,a,b,note in parts:
        body+=f"`src/{f}`, lines {a} to {b}"+(f" ({note})" if note else "")+":\n\n```python\n"+_lines(f,a,b)+"\n```\n\n"
    md(body)

md("""
# Learned deferral for sanctions entity resolution: classical matchers, LLMs and human review

**Harsh Chauhan, IIT Dharwad. PRML course project, November 2026.**

This notebook runs the whole project end to end and keeps the outputs under each cell. The heavy lifting lives in the
`er_*.py` scripts (`src/` in the repository, or next to this notebook on the ml.cern.ch server); they are also what the live demo uses. The notebook calls them stage by stage.

| stage | script | runtime |
|---|---|---|
| data: schema filter, 60k sample, split, 34 pair features | `er_lib.py sample / split / features` | ~3 min (already done, outputs in `data/`) |
| seven matchers, CV, threshold, isotonic calibration | `er_match.py` | ~1 min |
| rule baselines, learning curve, calibration curve, importance, per-type errors | `er_extra.py` | ~2 min |
| LLM scoring with vLLM (Qwen2.5-7B / 14B-AWQ / 7B 4-shot) | `er_lib.py llm ...` | 6 + 14 + 7 min on an H100 slice, **loaded from `data/llm_*.jsonl`, not re-run here** |
| complementarity, gates, deferral curves | `er_gate.py` | seconds |
| hard core, ablation, cross-type, leakage, difficulty strata | `er_core.py er_ablate.py er_xtype.py er_leakcheck.py er_strata.py` | ~3 min |
| sampling-luck check on all 376k other pairs | `er_fullpop.py` | ~2.5 min |
| noisy reviewer, asymmetric costs, combining matcher and LLM | `er_decision.py` | ~1 min |
| scaling the expert: Qwen3-32B zero-shot / chain of thought / thinking | `er_llm32.py zs|cot|think` | 32 + 3.5 + 7.5 min on the H100 slice, **loaded from `data/`, not re-run here** |
| evaluation of all experts on the deferred pairs | `er_llm32_eval.py` | seconds |

Paths: `ER_ROOT` (project folder, default `~/prml_proj/er`) and `ER_RAW` (the public pairs file). Data: OpenSanctions Pairs,
`pairs-20251209.json.gz`, CC BY-NC 4.0.
""")
code("""
import os,sys,json,warnings,time
warnings.filterwarnings("ignore")
ROOT=os.environ.get("ER_ROOT",os.path.expanduser("~/prml_proj/er"));DATA=f"{ROOT}/data";FIG=f"{ROOT}/figs"
os.environ["ER_ROOT"]=ROOT;os.environ["NOCOUNTS"]="1"     # main model drops the 5 cluster-size count features (see ablation)
NB=os.getcwd();SRC=next(d for d in [ROOT,f"{NB}/../src",f"{NB}/src",NB] if os.path.exists(f"{d}/er_lib.py"))   # scripts: flat server layout or repo src/
os.chdir(ROOT);sys.path.insert(0,SRC)
import numpy as np,sklearn,matplotlib
from IPython.display import Image,display
import er_lib as E
def show(name,width=760): display(Image(filename=f"{FIG}/{name}",width=width))
print("python",sys.version.split()[0],"| numpy",np.__version__,"| scikit-learn",sklearn.__version__,"| matplotlib",matplotlib.__version__)
print("ROOT =",ROOT,"| scripts from",SRC)
""")

md("""
## 1. Data and the leakage pitfall

The raw file has 755,540 lines. 280,487 of them are auto-generated *relational stubs* (schemas `Occupancy`, `Ownership`, ...):
both sides byte-identical and always labelled positive. A first run that included them reported a meaningless 99.4 F1; reading the
LLM's "mistakes" (`Occupancy` vs `Occupancy`) exposed it. `er_lib.sample` keeps only the ten entity schemas below, drops provenance
fields that could leak the label (`datasets`, `referents`, `first_seen`, `target`), de-duplicates unordered pairs and draws
39k positive + 21k negative pairs with a fixed seed. Split 36k / 6k / 18k.
""")
codeview('Cleaning and sampling in `er_lib.py`. `SCHEMAS` is the list of entity types kept (the relational stubs fail this test), `strip` keeps only the `KEEP` properties (the provenance fields are not in it), `sample` streams the raw file once with reservoir sampling per class, `split` is a seeded 60 / 10 / 30 permutation.',[('er_lib.py', 9, 17, 'vocabulary and strip'), ('er_lib.py', 19, 55, 'sample'), ('er_lib.py', 169, 177, 'split')])
codeview('The 34 pair features in `er_lib.py` (five count features are dropped later, leaving 29). Names are normalised, every name on side A is compared with every name on side B and the best score is kept.',[('er_lib.py', 60, 116, 'normalisation, name lists, helper extractors, feature names'), ('er_lib.py', 118, 158, 'pair_feats, one call per pair')])
code("""
print("entity schemas kept:",sorted(E.SCHEMAS))
print(json.dumps(json.load(open(f"{DATA}/sample_stats.json")),indent=1))
sp=json.load(open(f"{DATA}/split.json"));print({k:len(v) for k,v in sp.items()})
F=np.load(f"{DATA}/feats.npz",allow_pickle=True);print("feature matrix",F["X"].shape,"positive rate",F["y"].mean().round(4))
print("features:",", ".join(list(F["names"])))
rows=E.load_rows();r=rows[sp["test"][7]]
print("\\nexample test pair (analyst label:",r["label"],")");print(E.render(r["left"]));print("--- vs ---");print(E.render(r["right"]))
""")

md("""
## 2. Matchers: seven classifiers, 5-fold CV, validation threshold, isotonic calibration

29 pairwise features (names, script, type, dates, geography/gender, identifiers, address/topics/position/birthplace).
Model selection by 5-fold stratified CV on the training split; threshold tuned on validation; test touched once.
Isotonic regression on the validation split turns the random-forest vote fraction into a calibrated posterior.
""")
codeview('The seven classifiers in `er_match.py`. `models()` holds every hyperparameter. The loop does 5-fold stratified cross-validation for out-of-fold probabilities, fits on the full training split (the SVM on a 12k subsample for speed), picks the F1-best threshold on validation, and fits the isotonic calibration on validation.',[('er_match.py', 14, 19, 'load features, drop the count features when NOCOUNTS=1'), ('er_match.py', 21, 39, 'the models, the metric block, the threshold search'), ('er_match.py', 41, 63, 'cross-validation, fit, threshold, isotonic calibration')])
code("""
%run $SRC/er_match.py
""")
code("""
m=json.load(open(f"{DATA}/matcher_results.json"))
print(f"{'model':10s} {'CV F1':>7s} {'test F1@thr':>12s} {'cal F1':>7s} {'P':>6s} {'R':>6s} {'AUC':>7s} {'Brier':>7s} {'fit s':>6s}")
for k in ["logreg","dtree","adaboost","svm_rbf","knn","hgb","rf"]:
    d=m[k];print(f"{k:10s} {d['cv_oof']['f1']*100:7.2f} {d['test@thr']['f1']*100:12.2f} {d['test_cal']['f1']*100:7.2f} {d['test_cal']['precision']*100:6.1f} {d['test_cal']['recall']*100:6.1f} {d['test_cal']['auc']:7.4f} {d['test_cal']['brier']:7.4f} {d['fit_time_s']:6.1f}")
print("\\ntop RF features:",", ".join(f"{a} {b:.3f}" for a,b in m["rf_importance"][:10]))
""")

md("""
## 3. Rule baselines, label efficiency, calibration, feature importance, per-type errors
""")
codeview('Rule baselines and the learning curve in `er_extra.py`. Each rule is one feature and one threshold picked on validation. The learning curve trains the forest and logistic regression on random subsets of the training split of increasing size and scores on the full test split.',[('er_extra.py', 19, 33, 'rule baselines'), ('er_extra.py', 35, 49, 'learning curve')])
code("""
%run $SRC/er_extra.py
""")
code("""
show("learning_curve.png",620);show("calibration.png",480);show("importance.png",620)
""")

md("""
## 4. The LLM expert

Qwen2.5-7B-Instruct (bf16), Qwen2.5-14B-Instruct-AWQ (4-bit) and a 4-shot 7B variant, served with vLLM, temperature 0,
one-word `yes`/`no` answer, P(yes) from the first-token logprobs. Scoring 30k pairs (18k test + 12k gate-training pairs) took
376 s (7B), 850 s (14B) and 403 s (4-shot) on an H100 MIG slice. Commands (not re-run here):

```bash
python er_lib.py llm --model Qwen/Qwen2.5-7B-Instruct --which main
python er_lib.py llm --model Qwen/Qwen2.5-14B-Instruct-AWQ --which main
python er_lib.py llm --model Qwen/Qwen2.5-7B-Instruct --which main --shots 4
```
""")
codeview('The LLM as a matcher in `er_lib.py`. `render` turns a record into labelled lines, `PROMPT` is the exact prompt, `llm` scores a list of pairs with vLLM at temperature 0 and reads P(yes) from the first-token log probabilities. This stage runs on the GPU (`python er_lib.py llm --model ... --which main`, 6 to 14 minutes per model) and the notebook loads its saved answers from `data/llm_*.jsonl`.',[('er_lib.py', 179, 194, 'render and the prompt'), ('er_lib.py', 196, 203, 'the four-shot block'), ('er_lib.py', 205, 236, 'scoring with vLLM')])
code("""
print(E.PROMPT.format(a="<record A>",b="<record B>"))
from sklearn.metrics import f1_score,accuracy_score
y=F["y"];te=np.array(sp["test"])
for tag in ["Qwen2.5-7B-Instruct","Qwen2.5-14B-Instruct-AWQ","Qwen2.5-7B-Instruct_4shot"]:
    d={}
    for line in open(f"{DATA}/llm_{tag}_main.jsonl"):
        x=json.loads(line);d[x["idx"]]=x
    pred=np.array([d[i]["pred"] for i in te]);ntok=np.mean([d[i]["ntok_in"] for i in te])
    print(f"{tag:28s} test F1={f1_score(y[te],pred)*100:.2f} acc={accuracy_score(y[te],pred)*100:.2f} avg prompt tokens={ntok:.0f}")
""")

md("""
## 5. Deferral: when to hand a pair to an expert (complementarity, gates, deferral curves)

Reject option (Chow 1970): defer when $\\min(p,1-p)>\\lambda$; in budget form, rank test pairs by confidence and defer the top
fraction. Gates compared: confidence, learned error gate $\\hat P(\\text{matcher wrong}\\mid x,p)$, learned gain gate
$\\hat P(\\text{expert right}\\wedge\\text{matcher wrong}\\mid x,p)$ (learning to defer), random and oracle. Two experts: human
(assumed correct) and the LLM. Gates are trained on 12k training pairs with LLM answers, disjoint from calibration data and test.
""")
codeview("Gates and deferral curves in `er_gate.py`. The learned gates see the 29 features plus p, min(p,1-p) and the raw score, and are trained on 12k training pairs to predict the matcher's errors (`err`) or the pairs the LLM can fix (`gain`). `curve` is the deferral mechanic: rank by a score, replace the matcher's answer on the top fraction by the expert's answer, recompute F1.",[('er_gate.py', 19, 36, 'inputs, gate features, the two targets'), ('er_gate.py', 41, 54, 'the learned gates and the confidence gate'), ('er_gate.py', 56, 76, 'deferral curves for the confidence, random, oracle and learned gates, human and LLM experts')])
code("""
%run $SRC/er_gate.py rf Qwen2.5-7B-Instruct
""")
code("""
%run $SRC/er_gate.py rf Qwen2.5-14B-Instruct-AWQ
""")
code("""
%run $SRC/er_gate.py rf Qwen2.5-7B-Instruct_4shot
""")
code("""
show("deferral_curves_Qwen2.5-7B-Instruct.png",900);show("deferral_curves_Qwen2.5-14B-Instruct-AWQ.png",900)
""")

md("""
## 6. The hard core: pairs that every system gets wrong
""")
codeview('The hard core in `er_core.py`: test pairs that the matcher and all three LLM runs get wrong.',[('er_core.py', 8, 31, '')])
code("""
%run $SRC/er_core.py
print(open(f"{DATA}/core_examples.txt").read()[:3500])
""")

md("""
## 7. Robustness: feature ablation, cross-type generalisation, leakage checks, difficulty strata
""")
codeview('Feature-group ablation in `er_ablate.py`: a forest and a logistic regression per feature group. `all` against `no_counts` is the count-feature leakage check.',[('er_ablate.py', 13, 26, '')])
code("""
%run $SRC/er_ablate.py
""")
code("""
%run $SRC/er_xtype.py
""")
code("""
%run $SRC/er_leakcheck.py
""")
code("""
%run $SRC/er_strata.py
""")

md("""
## 8. Not a lucky sample: the forest on all 375,842 other analyst-judged pairs

The matcher trained on the 36k training split is applied to every other entity pair in the raw file (none of them seen in
training, validation or test). Needs the raw file at `ER_RAW`.
""")
codeview('The sampling-luck check in `er_fullpop.py`: the raw file is streamed again with the same filter, every pair not in the 60k sample is featurised, and the forest trained on the 36k training split (with the isotonic map from the 6k validation split) scores all of them. The confidence interval is a 200-draw bootstrap.',[('er_fullpop.py', 15, 46, 'stream, featurise, score'), ('er_fullpop.py', 53, 57, 'bootstrap confidence interval')])
code("""
%run $SRC/er_fullpop.py
""")

md("""
## 9. Decision theory extras: an imperfect reviewer, asymmetric costs, combining instead of deferring

The human expert above is perfect by construction. Here the reviewed labels are flipped with probability eps (20 draws), the Bayes
reject region is recomputed for unequal costs of a missed match, a false match and a review (defer when c_d/c_FN < p < 1 - c_d/c_FP),
and the LLM is combined with the matcher (weighted average, logistic stacker, LLM probability as a 30th feature) instead of replacing it on deferred pairs.
""")
codeview("Decision-theory extras in `er_decision.py`. (B) combines the matcher and the LLM instead of deferring. (C) flips the reviewer's answer with probability eps on the deferred pairs. (D) applies the three-way Bayes rule with costs c_fp = 1, c_fn in {1, 3, 10}, c_d in {0.05, 0.1, 0.2, 0.5} and compares the total cost with the best single threshold without a reject option.",[('er_decision.py', 39, 50, 'combination'), ('er_decision.py', 53, 67, 'noisy reviewer'), ('er_decision.py', 74, 85, 'asymmetric costs, the reject region is lo < p < hi')])
code("""
%run $SRC/er_decision.py Qwen2.5-7B-Instruct
""")
code("""
show("noisy_human.png",560);show("cost_region.png",560)
""")

md("""
## 10. Scaling the expert: a 32B model with reasoning

Qwen3-32B-AWQ was run three ways (commands below, not re-run here; answers are loaded from `data/`): zero-shot one-word answers on the
same 30k pairs as the smaller models, a short chain of thought ("think step by step in at most five sentences, then answer") on the 10%
least confident test pairs, and native thinking mode (650 output tokens per pair) on the 5% least confident test pairs.

```bash
python er_llm32.py zs      # 1934 s
python er_llm32.py cot     # 205 s, 1845 pairs
python er_llm32.py think   # 450 s, 987 pairs
```
""")
codeview('The 32B expert in `er_llm32.py` and its evaluation in `er_llm32_eval.py`. The three modes differ in which pairs they score and how the model is sampled. The GPU run is `bash run32.sh` (32 + 4 + 8 minutes); the notebook loads its saved answers from `data/`.',[('er_llm32.py', 13, 16, 'model and the chain-of-thought instruction'), ('er_llm32.py', 26, 47, 'answer parsing and the three modes'), ('er_llm32_eval.py', 18, 28, 'the six expert files, confidence ranking, the hard core'), ('er_llm32_eval.py', 32, 53, 'per expert: alone, complementarity, deferral curve, the 5 percent breakdown, core recovery')])
code("""
%run $SRC/er_llm32_eval.py
""")

md("""
## 11. Paper figures

`collect_results.py` gathers the result files written by the runs above into `results_all.json`; `make_figs.py` regenerates every figure of the paper and slides from it.
""")
code("""
%run $SRC/collect_results.py
%run $SRC/make_figs.py
show("fig_deferral.png",900);show("fig_experts.png",900);show("fig_strata.png",620);show("fig_core.png",620);show("fig_complementarity.png",620);show("fig_noisy_human.png",520);show("fig_cost.png",520)
""")

md("""
## 12. Headline numbers

* Random forest over 29 pair features: **98.7 F1** (AUC 0.997) on 18k test pairs; **98.5** on all 376k other pairs (95% CI 98.4 to 98.5);
  zero-shot Qwen2.5-7B 95.8, 14B 96.6, Qwen3-32B 96.8, best single-threshold rule 94.6, published production scorer 91.3. 100 labelled pairs beat the 7B LLM.
* Human review as expert, confidence gate: 5% of pairs reviewed removes 71% of errors (98.7 to 99.6 F1); learned gates do not beat it;
  a reviewer who is wrong 10% of the time still reaches 99.25, and the gate adapts to asymmetric costs (missed match 10x: 10.7% deferred, cost per 1000 pairs 37.9 to 15.6).
* LLM as expert: at most +0.25 F1 from 7B zero-shot to 32B with thinking, and the same +0.2 when combined with the matcher instead.
  On the deferred pairs every LLM is wrong 18 to 20% of the time against the matcher's own 24%, and the 89-pair hard core (same-name,
  attribute-poor pairs judged *different* by the analysts, 42% of them confidently wrong) stays hard: the thinking model recovers 9.
* Two dataset artefacts found and removed (relational stubs; cluster-size count features).

Live demo: `python demo.py pair --random`, `demo.py train`, `demo.py curve`, `demo.py core`, `demo.py ask --pid <id>` (see `DEMO.md`).
""")

nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python (prml-er)","language":"python","name":"prml-er"},"language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}
for i,c in enumerate(nb["cells"]):
    c["id"]=f"cell-{i:02d}"
json.dump(nb,open("sanctions_entity_resolution.ipynb","w"),indent=1)
print("cells",len(cells))
