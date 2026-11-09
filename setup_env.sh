#!/bin/bash
# Recreate the (ephemeral) /scratch environment for the project on ml.cern.ch. Safe to re-run.
# Usage: bash setup_env.sh [--llm]      (--llm also installs vLLM and downloads Qwen2.5-7B-Instruct)
set -e
S=${SCRATCH_DIR:-/scratch/hchauhan_prml}
mkdir -p $S/hf $S/tmp
if [ ! -x $S/env/bin/python ]; then python3.13 -m venv $S/env; fi
$S/env/bin/pip install -q -U pip
$S/env/bin/pip install -q numpy scikit-learn rapidfuzz unidecode matplotlib ipykernel
echo "CORE_ENV_READY"
if [ ! -s $S/pairs-20251209.json.gz ]; then curl -sL -o $S/pairs-20251209.json.gz https://data.opensanctions.org/contrib/training/pairs-20251209.json.gz; fi
echo "DATA_READY $(ls -la $S/pairs-20251209.json.gz | awk '{print $5}') bytes"
if [ "$1" == "--llm" ]; then $S/env/bin/pip install -q vllm && echo "VLLM_READY"; HF_HOME=$S/hf $S/env/bin/python -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen2.5-7B-Instruct'); print('MODEL_READY')"; fi
echo "SETUP_DONE"
