#!/usr/bin/env bash
# Submit the three per-model experiment chains with Slurm dependencies.
#   bash slurm/submit_chains.sh            # submit (from the repository root or anywhere)
#   bash slurm/submit_chains.sh --dry-run  # print the sbatch commands only
# Jobs (all via slurm/run_chain.slurm = the exp2.slurm serving machinery + reasoning parser + CHAIN_SCRIPT hook):
#   phase 1 (parallel): gptoss_phase1, glm_phase1, llama_phase1
#   glm_judge            after gptoss_phase1 (both judges edit MINT_dataset_v1.2 in place -> serialized)
#   phase 2 (parallel):  gptoss_phase2, glm_phase2, llama_phase2  after all phase-1 jobs + glm_judge
# Dependencies use 'afterany' so a partially failed phase-1 job does not block phase 2; every step is
# idempotent (--continue / skip-if-exists), so re-running this script fills in whatever failed.
set -euo pipefail
cd "$(dirname "$0")/.."
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
# site-specific settings (account, image, cache paths): slurm/site.env, see slurm/site.env.example
if [ -f slurm/site.env ]; then set -a; source slurm/site.env; set +a; fi

# 0) offline prerequisite: filler_information in MINT v1.2 (exp1 'filler' condition; copied into the control sets)
if ! grep -q '"filler_information"' MINT_dataset_v1.2/winogrande_misinformed.json; then
    if [ "${DRY}" = "1" ]; then
        echo "[DRY] would run: python scripts/generate_filler.py (needs Python >= 3.7)"
    else
        PYBIN="${PYBIN:-}"
        if [ -z "${PYBIN}" ]; then
            # login node: /usr/bin/python3 is 3.6 -> use the python module
            source /etc/profile.d/modules.sh 2>/dev/null || true
            module load python/3.11.9 2>/dev/null || true
            PYBIN="$(command -v python3.11 || command -v python3)"
        fi
        "${PYBIN}" -c 'import sys; assert sys.version_info >= (3, 7), sys.version' || { echo "need Python >= 3.7 (set PYBIN=...)"; exit 2; }
        echo "[INFO] adding filler_information to MINT_dataset_v1.2 (${PYBIN} scripts/generate_filler.py)"
        "${PYBIN}" scripts/generate_filler.py
    fi
fi
for m in openai_gpt-oss-120b zai-org_GLM-4.7-Flash meta-llama_Llama-3.3-70B-Instruct; do
    [ -f "out/${m}/exp2_results_winogrande.json" ] || echo "[WARN] out/${m}/exp2_results_*.json missing: the message ablation will be skipped for ${m}; the verifier falls back to the exposure transcripts."
done

sb() {  # sb <jobname> <MODEL_NAME> <chain script> [extra sbatch args...] -> prints job id
    local name="$1" model="$2" script="$3"; shift 3
    if [ "${DRY}" = "1" ]; then
        echo "MODEL_NAME=${model} CHAIN_SCRIPT=${script} sbatch --parsable --export=ALL --job-name=${name} $* slurm/run_chain.slurm" >&2
        echo "DRY_${name}"
    else
        MODEL_NAME="${model}" CHAIN_SCRIPT="${script}" sbatch --parsable --export=ALL --job-name="${name}" "$@" slurm/run_chain.slurm | cut -d';' -f1
    fi
}
GPTOSS=openai/gpt-oss-120b; GLM=zai-org/GLM-4.7-Flash; LLAMA=meta-llama/Llama-3.3-70B-Instruct

P1_GPTOSS=$(sb chain_gptoss_p1 "${GPTOSS}" slurm/chains/gptoss_phase1.sh)
P1_GLM=$(sb    chain_glm_p1    "${GLM}"    slurm/chains/glm_phase1.sh)
P1_LLAMA=$(sb  chain_llama_p1  "${LLAMA}"  slurm/chains/llama_phase1.sh --gpus=2)
J_GLMJUDGE=$(sb chain_glm_judge "${GLM}"   slurm/chains/glm_judge.sh --dependency="afterany:${P1_GPTOSS}")
DEP="afterany:${P1_GPTOSS}:${P1_GLM}:${P1_LLAMA}:${J_GLMJUDGE}"
P2_GPTOSS=$(sb chain_gptoss_p2 "${GPTOSS}" slurm/chains/gptoss_phase2.sh --dependency="${DEP}")
P2_GLM=$(sb    chain_glm_p2    "${GLM}"    slurm/chains/glm_phase2.sh    --dependency="${DEP}")
P2_LLAMA=$(sb  chain_llama_p2  "${LLAMA}"  slurm/chains/llama_phase2.sh  --gpus=2 --dependency="${DEP}")

cat <<SUMMARY

submitted:
  phase 1   gptoss=${P1_GPTOSS}  glm=${P1_GLM}  llama=${P1_LLAMA}
  glm judge ${J_GLMJUDGE}  (after gptoss phase 1)
  phase 2   gptoss=${P2_GPTOSS}  glm=${P2_GLM}  llama=${P2_LLAMA}  (after all of the above)
logs: slurm-<jobid>.out and out/<model>/runs/job_<jobid>/slurm.log ; step summary at the end of each log.
afterwards (offline, once the transcripts of all runs, including exp2/exp3/exp1 from slurm/*.slurm, are in out/<model>/): make all
SUMMARY
[ "${DRY}" = "1" ] || squeue -u "${USER}" -o "%.10i %.22j %.8T %.12l %.20E" | head -20
