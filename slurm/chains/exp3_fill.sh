#!/usr/bin/env bash
# exp3 composition sweep: debate the questions that never completed (run through slurm/run_chain.slurm via CHAIN_SCRIPT).
# MALLM substituted every failed five-agent debate with an already-debated question (see exp3_fill_missing.py), so
# the Llama file has 382 records per mix but only 337-377 distinct questions. Serve with a context window that fits
# five-agent debates and run only the missing question x mix cells with the original exp3 configuration:
#   MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct VLLM_MAX_MODEL_LEN=32768 VLLM_MAX_NUM_BATCHED_TOKENS=2048 \
#     VLLM_GPU_MEMORY_UTILIZATION=0.95 CHAIN_SCRIPT=slurm/chains/exp3_fill.sh \
#     sbatch --export=ALL --gpus=2 --time=20:00:00 --job-name=exp3_fill_llama slurm/run_chain.slurm
# Memory: Llama-3.3-70B bf16 at TP=2 takes 65.7 GiB per GPU. On 94 GB H100s that leaves 15.7 GiB of KV cache; on an
# 80 GB A100 at the default 0.90 only 4.3 GiB are left (vLLM: "5.0 GiB KV cache is needed for a 32768-token request,
# available 4.26 GiB"), which is why the sweep was served with an 8k context that long five-agent prompts exceed. 0.95 gives ~8 GiB per GPU (~54k KV tokens
# on two A100s, 1.6x a 32k request); 2,048 batched tokens is what vLLM 0.19 picks itself when memory is tight and
# keeps the profiling peak small (chunked prefill is exact, so results are unaffected).
# The plan (out/<model>/exp3_fill/plan.json) is built offline beforehand from the raw sweep file or, when the
# transcripts are not on this machine, from the committed rescored CSV (plan --from_rescored). The merge into
# out/<model>/exp3_results_winogrande.json runs here only if that raw file is present; otherwise `make exp3_fill_merge`
# once it is.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
FILL_DIR="out/${MODEL_DIR}/exp3_fill"
RAW="out/${MODEL_DIR}/exp3_results_winogrande.json"
PLAN="${FILL_DIR}/plan.json"
echo "[INFO] exp3 fill: model=${MODEL_NAME} VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-<unset>} plan=${PLAN} raw=${RAW}"
if [ "${VLLM_MAX_MODEL_LEN:-8192}" -lt 16384 ]; then
    echo "[ERROR] VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-<unset>}: the missing debates failed on the 8,192-token context; submit with VLLM_MAX_MODEL_LEN=32768"
    FAILED+=("context window"); finish
fi
if [ ! -f "${PLAN}" ]; then
    if [ -f "${RAW}" ]; then
        step "plan from raw sweep file" python exp3_fill_missing.py plan --model_name "${MODEL_NAME}" --source "${RAW}"
    else
        echo "[ERROR] neither ${PLAN} nor ${RAW} exists; build the plan offline first (exp3_fill_missing.py plan --from_rescored ...)"
        FAILED+=("plan"); finish
    fi
fi
step "fill missing five-agent debates (all mixes, up to 3 passes)" python exp3_fill_missing.py run "${COMMON[@]}" \
    --plan "${PLAN}" --continue --max_passes 3
if [ -f "${RAW}" ]; then
    step "merge: de-duplicate + insert fill-in debates" python exp3_fill_missing.py merge --model_name "${MODEL_NAME}" --source "${RAW}"
else
    skip "merge" "${RAW} not on this machine; run 'make exp3_fill_merge' after copying it to out/${MODEL_DIR}/"
fi
finish
