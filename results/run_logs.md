# Raw logs pulled from ~/prml_proj/er on ml.cern.ch (2026-11-09)

## match.log (NOCOUNTS=1, 29 feats)
X (60000, 29) train/val/test 36000 6000 18000 pos rate 0.65
logreg     cvF1=0.9648 cvAUC=0.9892 | test F1@0.5=0.9659 F1@thr=0.9714 (thr=0.28) AUC=0.9891 P=0.963 R=0.980 | 1.9s
svm_rbf    cvF1=0.9853 cvAUC=0.9960 | test F1@0.5=0.9851 F1@thr=0.9846 (thr=0.68) AUC=0.9953 P=0.983 R=0.986 | 13.5s
knn        cvF1=0.9854 cvAUC=0.9951 | test F1@0.5=0.9860 F1@thr=0.9860 (thr=0.50) AUC=0.9948 P=0.983 R=0.989 | 1.5s
dtree      cvF1=0.9808 cvAUC=0.9926 | test F1@0.5=0.9804 F1@thr=0.9812 (thr=0.30) AUC=0.9925 P=0.976 R=0.987 | 0.7s
rf         cvF1=0.9869 cvAUC=0.9978 | test F1@0.5=0.9868 F1@thr=0.9871 (thr=0.48) AUC=0.9974 P=0.983 R=0.991 | 7.7s
adaboost   cvF1=0.9796 cvAUC=0.9954 | test F1@0.5=0.9790 F1@thr=0.9796 (thr=0.51) AUC=0.9947 P=0.979 R=0.980 | 38.5s
hgb        cvF1=0.9865 cvAUC=0.9975 | test F1@0.5=0.9851 F1@thr=0.9857 (thr=0.37) AUC=0.9972 P=0.981 R=0.990 | 5.4s

## ablate.log (34-feature run)
all                    nf=34 RF F1=0.9899 AUC=0.9983 | LR F1=0.9716 AUC=0.9919
no_counts              nf=29 RF F1=0.9869 AUC=0.9974 | LR F1=0.9659 AUC=0.9891
no_counts_no_topics    nf=28 RF F1=0.9855 AUC=0.9965 | LR F1=0.9652 AUC=0.9887
names_only             nf= 6 RF F1=0.9698 AUC=0.9777 | LR F1=0.9433 AUC=0.9559
names+dates            nf=11 RF F1=0.9730 AUC=0.9867 | LR F1=0.9497 AUC=0.9733
attrs_only(no names)   nf=28 RF F1=0.9742 AUC=0.9936 | LR F1=0.9389 AUC=0.9726
counts_only            nf= 5 RF F1=0.8568 AUC=0.8918 | LR F1=0.7778 AUC=0.7934
no_exact_name          nf=33 RF F1=0.9900 AUC=0.9983 | LR F1=0.9713 AUC=0.9915
exact_name       pos=0.842 neg=0.040
cross_script     pos=0.100 neg=0.016
date_both        pos=0.570 neg=0.543
date_match       pos=0.462 neg=0.013
year_match       pos=0.552 neg=0.071
date_missing     pos=0.430 neg=0.457
id_both          pos=0.329 neg=0.593
id_match         pos=0.283 neg=0.018
id_conflict      pos=0.047 neg=0.576
geo_conflict     pos=0.008 neg=0.280
gender_mismatch  pos=0.000 neg=0.011
kind_mismatch    pos=0.016 neg=0.003
geo_any          pos=0.793 neg=0.665
jw_max           pos mean=0.967 med=1.000 | neg mean=0.772 med=0.756
tsr_max          pos mean=0.977 med=1.000 | neg mean=0.697 med=0.667
n_names_l        pos mean=1.008 med=0.693 | neg mean=1.628 med=1.609
n_names_r        pos mean=1.361 med=1.099 | neg mean=1.771 med=1.792
nprops_l         pos mean=2.000 med=2.079 | neg mean=2.303 med=2.398
nprops_r         pos mean=2.262 med=2.303 | neg mean=2.389 med=2.485

## extra.log
rule jw_max>=0.95: test F1=0.9140 P=0.954 R=0.878
rule tsr_max>=0.99: test F1=0.9465 P=0.951 R=0.942
rule tok_jac_max>=0.63: test F1=0.9405 P=0.943 R=0.938
rule jw>=0.9 & no conflicts: test_f1 0.9056 precision 0.9751 recall 0.8454
n=    50 RF F1=0.9550  LR F1=0.9398
n=   100 RF F1=0.9639  LR F1=0.9527
n=   200 RF F1=0.9700  LR F1=0.9620
n=   500 RF F1=0.9757  LR F1=0.9587
n=  1000 RF F1=0.9780  LR F1=0.9631
n=  2000 RF F1=0.9807  LR F1=0.9636
n=  5000 RF F1=0.9826  LR F1=0.9594
n= 10000 RF F1=0.9853  LR F1=0.9662
n= 20000 RF F1=0.9861  LR F1=0.9614
n= 36000 RF F1=0.9868  LR F1=0.9659
person-person                n=11081 pos=0.74 matcher err=0.0106 llm err=0.0308
org-org                      n= 6380 pos=0.50 matcher err=0.0288 llm err=0.0901
cross-script                 n= 1267 pos=0.92 matcher err=0.0292 llm err=0.1523
same-script                  n=16733 pos=0.63 matcher err=0.0160 llm err=0.0458
no dates on a side           n= 7959 pos=0.64 matcher err=0.0318 llm err=0.0783
dates on both                n=10041 pos=0.67 matcher err=0.0051 llm err=0.0335
ids on both                  n= 7603 pos=0.52 matcher err=0.0099 llm err=0.0477

## xtype.log
train person       (22092) -> test org          ( 6380): F1=0.9003 AUC=0.9599 | model trained on everything: F1=0.9706
train org          (12881) -> test person       (11081): F1=0.9781 AUC=0.9660 | model trained on everything: F1=0.9927
train same_script  (33460) -> test cross_script ( 1267): F1=0.9708 AUC=0.9717 | model trained on everything: F1=0.9828
train person       (22092) -> test person       (11081): F1=0.9924 AUC=0.9984 | model trained on everything: F1=0.9927
train org          (12881) -> test org          ( 6380): F1=0.9713 AUC=0.9937 | model trained on everything: F1=0.9706
train cross_script ( 2540) -> test cross_script ( 1267): F1=0.9748 AUC=0.9808 | model trained on everything: F1=0.9828

## llm.log / llm14.log
[llm] Qwen/Qwen2.5-7B-Instruct val n=600 acc=0.9500 f1=0.9589 time=7.3s avg_in_tok=416
[llm] Qwen/Qwen2.5-7B-Instruct main n=30000 acc=0.9468 f1=0.9582 time=375.8s avg_in_tok=417
[llm] Qwen/Qwen2.5-14B-Instruct-AWQ main n=30000 acc=0.9558 f1=0.9656 time=849.7s avg_in_tok=417

## gate7.log (rf, 7B)
matcher F1 test=0.9872 acc=0.9831 | LLM F1 test=0.9583 acc=0.9467
complementarity: both right=0.9364 matcher-only right=0.0467 LLM-only right=0.0103 both wrong=0.0066
gate conf: AUC vs err=0.9452 vs gain=0.9555 | hgb_err .9316 | lr_err .9238 | hgb_gain .8825 | lr_gain .9256
== expert=llm (F1 %)
frac     confidence       random       oracle      hgb_err       lr_err     hgb_gain      lr_gain
0.000         98.72        98.72        98.72        98.72        98.72        98.72        98.72
0.010         98.76          .          99.48          .            .            .            .
0.030         98.80          .          99.22        98.94                       98.95
0.050         98.85          .          99.05
0.075         98.73        98.51        98.87        98.81        98.80        98.68        98.78
0.100         98.63        98.43        98.72        98.68        98.76        98.34        98.73
0.150         98.28        98.29        98.32        98.31        98.61        97.97        98.44
0.200         97.81        98.15        97.88        97.93        98.49        97.52        98.13
0.300         97.58        97.87        97.59        97.46        98.07        97.45        97.77
0.400         96.79        97.57        96.85        97.20        97.20        97.37        97.49
0.500         96.16        97.28        96.16        97.15        96.61        97.29        97.09
0.750         95.97        96.57        95.98        97.01        96.06        96.78        96.38
1.000         95.83        95.83        95.83        95.83        95.83        95.83        95.83
== expert=human
frac     confidence       random       oracle      hgb_err       lr_err     hgb_gain      lr_gain
0.000         98.72        98.72        98.72        98.72        98.72        98.72        98.72
0.010         98.99        98.73        99.47        99.00        98.98        98.98        98.96
0.020         99.15        98.74       100.00        99.23        99.17        99.21        99.17
0.030         99.31        98.76       100.00        99.37        99.35        99.42        99.36
0.050         99.63        98.78       100.00        99.51        99.60        99.53        99.59
0.075         99.74        98.82       100.00        99.67        99.69        99.63        99.67
0.100         99.81        98.85       100.00        99.72        99.73        99.68        99.72
0.150         99.86        98.91       100.00        99.82        99.80        99.74        99.78
0.200         99.89        98.98       100.00        99.88        99.84        99.74        99.83
0.300         99.95        99.11       100.00        99.96        99.91        99.74        99.90
0.400         99.98        99.23       100.00        99.97        99.94        99.75        99.94
0.500         99.99        99.36       100.00        99.98        99.97        99.76        99.96
0.750         99.99        99.68       100.00        99.98        99.98        99.78        99.97
1.000        100.00       100.00       100.00       100.00       100.00       100.00       100.00
profile of 10% deferred (confidence gate): cross_script .119 vs kept .065 | date_missing .839/.398 | kind_org .552/.333 | exact_name .518/.570 | id_both .312/.435 | geo_conflict .044/.111 | date_both .161/.602 | jw_max .920/.897 | tsr_max .939/.875 | pos_rate .701/.650 | matcher_err .144/.003 | llm_err .154/.042

## gate14.log (rf, 14B-AWQ)
gate-train 12000 test 18000
matcher F1 test=0.9872 acc=0.9831 | LLM F1 test=0.9658 acc=0.9560
complementarity: both right=0.9455 matcher-only right=0.0376 LLM-only right=0.0105 both wrong=0.0064
gate hgb_err: AUC vs err on test = 0.9316
gate lr_err: AUC vs err on test = 0.9238
gate hgb_gain: AUC vs gain on test = 0.8699
gate lr_gain: AUC vs gain on test = 0.9193
gate conf: AUC vs err=0.9452 vs gain=0.9586
== expert=llm
frac     confidence       random       oracle      hgb_err       lr_err     hgb_gain      lr_gain
0.000         98.72        98.72        98.72        98.72        98.72        98.72        98.72
0.010         98.77        98.70        99.48        98.78        98.76        98.83        98.75
0.020         98.82        98.68        99.37        98.89        98.79        98.94        98.81
0.030         98.87        98.66        99.31        98.97        98.92        98.98        98.93
0.050         98.94        98.62        99.17        98.95        98.93        98.91        98.92
0.075         98.87        98.57        98.97        98.89        98.92        98.80        98.86
0.100         98.83        98.51        98.90        98.76        98.91        98.63        98.85
0.150         98.52        98.41        98.56        98.49        98.82        98.51        98.80
0.200         98.18        98.30        98.25        98.16        98.76        98.03        98.67
0.300         98.00        98.09        98.02        97.86        98.40        98.01        98.36
0.400         97.25        97.87        97.25        97.69        97.56        97.99        98.11
0.500         96.65        97.65        96.65        97.67        97.03        97.84        97.84
0.750         96.61        97.11        96.61        97.66        96.65        97.17        97.18
1.000         96.58        96.58        96.58        96.58        96.58        96.58        96.58
== expert=human  (identical to gate7 human table)
profile of 10% deferred: same as 7B except llm_err_rate deferred=0.129 kept=0.035

## leak.log
test pairs by #entities already seen in some TRAIN pair:
  neither seen  n= 9578 (53.2%) pos=0.92 RF F1=0.9909 err=0.0167 | 7B LLM err=0.0772 | 14B LLM err=0.0659
  one seen      n= 4876 (27.1%) pos=0.61 RF F1=0.9823 err=0.0217 | 7B LLM err=0.0416 | 14B LLM err=0.0306
  both seen     n= 3546 (19.7%) pos=0.01 RF F1=0.6346 err=0.0107 | 7B LLM err=0.0048 | 14B LLM err=0.0034
test pairs whose exact record contents also appear as a TRAIN pair: 40 (0.22%)
   on those: pos=0.88 RF err=0.0750
   on the rest: RF F1=0.9872 err=0.0168
test pairs where both records are byte-identical: 160 (pos rate 0.95625)
distinct entity ids in sample: 86380 over 60000 pairs; max degree 22; mean degree 1.39
  label=0: frac both-seen=0.565 one-seen=0.306 neither=0.129
  label=1: frac both-seen=0.003 one-seen=0.253 neither=0.744

## strata.log
all test                           n=18000 pos=0.65 RF F1=0.9872 err=0.0169 | 7B F1=0.9583 err=0.0533 | 14B-AWQ F1=0.9658 err=0.0440
excluding byte-identical pairs     n=17840 pos=0.65 RF F1=0.9873 err=0.0166 | 7B F1=0.9580 err=0.0534 | 14B-AWQ F1=0.9657 err=0.0440
exact normalised name shared       n=10172 pos=0.98 RF F1=0.9920 err=0.0156 | 7B F1=0.9717 err=0.0543 | 14B-AWQ F1=0.9790 err=0.0406
NO exact name shared (hard)        n= 7828 pos=0.24 RF F1=0.9611 err=0.0185 | 7B F1=0.8827 err=0.0520 | 14B-AWQ F1=0.8925 err=0.0484
no exact name, jw>=0.9             n= 1625 pos=0.43 RF F1=0.9570 err=0.0369 | 7B F1=0.9473 err=0.0443 | 14B-AWQ F1=0.9628 err=0.0320
no exact name, jw<0.9              n= 6203 pos=0.19 RF F1=0.9636 err=0.0137 | 7B F1=0.8409 err=0.0540 | 14B-AWQ F1=0.8464 err=0.0527
byte-identical pairs               n=  160 pos=0.96 RF F1=0.9776 err=0.0437 | 7B F1=0.9776 err=0.0437 | 14B-AWQ F1=0.9776 err=0.0437
hard subset, defer 5% to human: F1 0.9611 -> 0.9863 (kept-part F1 0.9845)
hard subset, defer 10% to human: F1 0.9611 -> 0.9952 (kept-part F1 0.9937)
hard subset, defer 20% to human: F1 0.9611 -> 0.9976 (kept-part F1 0.9951)

## sample_stats.json
total_lines 755540; unique_pairs positive 264462 negative 171380 skipped_schema 280487; sampled positive 39000 negative 21000

## decision.log (er_decision.py Qwen2.5-7B-Instruct)
```
matcher alone test F1=98.72 | LLM alone F1=95.83 | AUC matcher=0.9974 AUC LLM p_yes=0.9870
combination (test F1): {'matcher': 98.72, 'llm': 95.83, 'average': 97.04, 'weighted_w': 0.64, 'weighted': 98.89, 'stacker_lr': 98.9, 'stacker_coef': [0.8295727137783768, 0.14318532519440766], 'rf_12k_29feats': 98.52, 'rf_12k_29feats+llm': 98.86, 'llm_feature_importance': 0.27031472149082497, 'llm_feature_rank': 1}
noisy human, F1 at 5% deferred: {'0': 99.63, '0.02': 99.55, '0.05': 99.45, '0.1': 99.25, '0.2': 98.87} | matcher error rate inside the deferred 5%: 0.241
asymmetric costs (cost per 1000 pairs, with reject vs best single threshold):
  cfn=1,cd=0.05    region=[0.05, 0.95] defer=  6.0% FP= 54 FN= 24 cost=   7.3 | no-reject thr=0.49 cost=  16.8
  cfn=1,cd=0.1     region=[0.1, 0.9] defer=  5.3% FP= 58 FN= 24 cost=   9.9 | no-reject thr=0.49 cost=  16.8
  cfn=1,cd=0.2     region=[0.2, 0.8] defer=  4.4% FP= 61 FN= 43 cost=  14.6 | no-reject thr=0.49 cost=  16.8
  cfn=1,cd=0.5     region=None defer=  0.0% FP=196 FN=108 cost=  16.9 | no-reject thr=0.49 cost=  16.8
  cfn=3,cd=0.05    region=[0.017, 0.95] defer=  8.5% FP= 54 FN=  9 cost=   8.7 | no-reject thr=0.22 cost=  26.2
  cfn=3,cd=0.1     region=[0.033, 0.9] defer=  6.1% FP= 58 FN= 15 cost=  11.8 | no-reject thr=0.22 cost=  26.2
  cfn=3,cd=0.2     region=[0.067, 0.8] defer=  5.1% FP= 61 FN= 24 cost=  17.5 | no-reject thr=0.22 cost=  26.2
  cfn=3,cd=0.5     region=[0.167, 0.5] defer=  1.2% FP=196 FN= 43 cost=  24.0 | no-reject thr=0.22 cost=  26.2
  cfn=10,cd=0.05   region=[0.005, 0.95] defer= 11.3% FP= 54 FN=  3 cost=  10.3 | no-reject thr=0.09 cost=  37.9
  cfn=10,cd=0.1    region=[0.01, 0.9] defer= 10.7% FP= 58 FN=  3 cost=  15.6 | no-reject thr=0.09 cost=  37.9
  cfn=10,cd=0.2    region=[0.02, 0.8] defer=  5.8% FP= 61 FN= 15 cost=  23.4 | no-reject thr=0.09 cost=  37.9
  cfn=10,cd=0.5    region=[0.05, 0.5] defer=  1.8% FP=196 FN= 24 cost=  33.5 | no-reject thr=0.09 cost=  37.9
DECISION_DONE
```

## Qwen3-32B runs (er_llm32.py zs / cot / think)
```
[llm32] Qwen/Qwen3-32B-AWQ mode=zs n=30000
[llm32] zs n=30000 answered=30000 truncated=0 acc=0.9579 f1=0.9674 time=1934s gen_tok=60000 out_tok/pair=2
[llm32] Qwen/Qwen3-32B-AWQ mode=cot n=1845
[llm32] cot n=1845 answered=1845 truncated=0 acc=0.8450 f1=0.8913 time=205s gen_tok=258762 out_tok/pair=140
[llm32] cot matcher-wrong pairs in subset=304 fixed by LLM=182 LLM wrong where matcher right=164
[llm32] Qwen/Qwen3-32B-AWQ mode=think n=987
[llm32] think n=987 answered=987 truncated=0 acc=0.7923 f1=0.8318 time=450s gen_tok=638475 out_tok/pair=647
[llm32] think matcher-wrong pairs in subset=304 fixed by LLM=181 LLM wrong where matcher right=82
```

## llm32_eval.log (er_llm32_eval.py)
```
matcher test F1=98.72 errors=304 | hard core (wrong for matcher + 7B + 14B + 4shot) = 89
Qwen2.5-7B zero-shot         alone F1= 95.83 | 5% deferred: F1= 98.85 errors=271 fixed=145 broken=112 expert err on deferred=0.204 | core fixed 0/89 | matcher-wrong fixed 186/304
   curve: 0%:98.72 1%:98.76 2%:98.78 3%:98.80 5%:98.85 7.5%:98.73 10%:98.63 15%:98.28 20%:97.81
Qwen2.5-14B zero-shot        alone F1= 96.58 | 5% deferred: F1= 98.94 errors=250 fixed=147 broken=93 expert err on deferred=0.181 | core fixed 0/89 | matcher-wrong fixed 189/304
   curve: 0%:98.72 1%:98.77 2%:98.82 3%:98.87 5%:98.94 7.5%:98.87 10%:98.83 15%:98.52 20%:98.18
Qwen2.5-7B 4-shot            alone F1= 94.45 | 5% deferred: F1= 98.69 errors=308 fixed=146 broken=150 expert err on deferred=0.246 | core fixed 0/89 | matcher-wrong fixed 185/304
   curve: 0%:98.72 1%:98.77 2%:98.72 3%:98.68 5%:98.69 7.5%:98.51 10%:98.30 15%:97.72 20%:97.10
Qwen3-32B zero-shot          alone F1= 96.77 | 5% deferred: F1= 98.86 errors=270 fixed=118 broken=84 expert err on deferred=0.203 | core fixed 4/89 | matcher-wrong fixed 158/304
   curve: 0%:98.72 1%:98.75 2%:98.79 3%:98.80 5%:98.86 7.5%:98.78 10%:98.74 15%:98.48 20%:98.20
Qwen3-32B chain of thought   alone F1=   nan | 5% deferred: F1= 98.93 errors=253 fixed=143 broken=92 expert err on deferred=0.184 | core fixed 4/89 | matcher-wrong fixed 182/304
   curve: 0%:98.72 1%:98.77 2%:98.84 3%:98.87 5%:98.93 7.5%:98.80 10%:98.72
Qwen3-32B thinking           alone F1=   nan | 5% deferred: F1= 98.97 errors=245 fixed=141 broken=82 expert err on deferred=0.176 | core fixed 9/89 | matcher-wrong fixed 181/304
   curve: 0%:98.72 1%:98.78 2%:98.87 3%:98.90 5%:98.97
LLM32_EVAL_DONE
```
