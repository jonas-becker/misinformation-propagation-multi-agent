#!/usr/bin/env bash
# Additional runs (run through slurm/run_chain.slurm via CHAIN_SCRIPT).
# Select the parts with FOLLOWUP_PARTS (comma list), default: all of them:
#   exp1seeds  single-agent seeds 1,2 (baseline,false_fact) with --reprompt_unparsed
#              (overwrites the two conditions in out/<model>/seeds/seed<s>/)
#   diag       the diagonal generator x consumer cells (GLM on GLM passages, GPT-OSS on GPT-OSS passages)
#   wino0      WinoGrande debates k=1,2,3 seed 0 + mSC + source-skepticism mitigation k=1,3 + verifier, all with
#              --mint_filter all (all 382 questions; the analyses restrict them to the verified questions;
#              overwrites the WinoGrande conditions of seed 0)
#   wino1, wino2  same for seeds 1 and 2 (k=1,3 + mSC), separate files -> may run in parallel jobs
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
PARTS="${FOLLOWUP_PARTS:-exp1seeds,diag,wino0,wino1,wino2}"
has() { [[ ",${PARTS}," == *",$1,"* ]]; }
echo "[INFO] followup parts: ${PARTS} (model ${MODEL_NAME})"

if has exp1seeds; then
    for s in 1 2; do
        step "exp1 baseline,false_fact seed${s} (reprompt_unparsed, overwrite)" \
            python exp1.py "${COMMON[@]}" --conditions baseline,false_fact --seed "${s}" --reprompt_unparsed
    done
fi

if has diag; then
    case "${MODEL_NAME}" in
        zai-org/GLM-*)     exp1_on_generator glm    "${GEN_GLM}" ;;
        openai/gpt-oss*)   exp1_on_generator gptoss "${GEN_GPTOSS}" ;;
        *)                 skip "diagonal cell" "Llama x Llama passages is the main exp1 run" ;;
    esac
fi

wino_seed() {  # wino_seed <seed>: full-sample WinoGrande debates + mSC (no --continue: replaces existing conditions)
    local s="$1" ks="1,3"
    [ "${s}" = "0" ] && ks="1,2,3"
    step "exposure winogrande k=${ks} seed${s} mint_filter=all" \
        python exp_exposure.py "${COMMON[@]}" --datasets winogrande --k "${ks}" --mint_filter all --seed "${s}"
    step "msc winogrande k=${ks} seed${s} mint_filter=all" \
        python exp_msc.py "${COMMON[@]}" --datasets winogrande --k "${ks}" --mint_filter all --seed "${s}"
    if [ "${s}" = "0" ]; then
        step "exposure source_skepticism winogrande k=1,3 seed0 mint_filter=all" \
            python exp_exposure.py "${COMMON[@]}" --datasets winogrande --k 1,3 --mitigation source_skepticism --mint_filter all --seed 0
        step "verifier mitigation winogrande (exposure seed0 transcripts)" \
            python exp_mitigation_verifier.py "${COMMON[@]}" --datasets winogrande --source exposure
    fi
}
for s in 0 1 2; do
    if has "wino${s}"; then wino_seed "${s}"; fi
done
finish
