#!/bin/bash
# runs the three 32B expert experiments one after another; logs in llm32_*.log
cd ~/prml_proj/er
export HF_HOME=/scratch/hchauhan_prml/hf NOCOUNTS=1
P=/scratch/hchauhan_prml/env/bin/python
for m in zs cot think; do
  $P er_llm32.py $m > llm32_$m.log 2>&1
  tail -3 llm32_$m.log | grep -E "llm32|DONE" 
done
echo RUN32_ALL_DONE
