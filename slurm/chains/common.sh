#!/usr/bin/env bash
# Shared helpers for the per-model experiment chains (run through slurm/run_chain.slurm via CHAIN_SCRIPT).
# slurm/run_chain.slurm exports MODEL_NAME, ENDPOINT_URL and REPO_ROOT and activates the experiment venv.
set -uo pipefail
cd "${REPO_ROOT:?REPO_ROOT not set (run through slurm/run_chain.slurm)}"
: "${MODEL_NAME:?}" "${ENDPOINT_URL:?}"

COMMON=(--model_name "${MODEL_NAME}" --endpoint_url "${ENDPOINT_URL}" --no_autostart)
MODEL_DIR="${MODEL_NAME//\//_}"                       # out/<MODEL_DIR>/ (matches scripts/common.py MODEL_DIRS)
GEN_GLM="MINT_dataset_v1.2/generators/glm"
GEN_GPTOSS="MINT_dataset_v1.2/generators/gptoss"
RT_DIR="controls/relevant_true"
PARA_DIR="controls/paraphrase_gptoss"
# thinking tokens count against the cap. Measured: gpt-oss ~400 completion tokens per judge call;
# GLM-4.7-Flash ~1.5k (winogrande/ethics) but ~5.4k on cwq (2/5 rows truncated at 8192). GLM decodes at only
# 30-40 tok/s per request, so 16384 made single rows take >7 min, tripping the 600 s client timeout (retry storm).
# 12288 tokens ~ 6 min worst case; the client timeout for the judge is raised to 30 min accordingly.
case "${MODEL_NAME}" in zai-org/GLM-*) _judge_default=12288; _judge_timeout=1800 ;; *) _judge_default=2048; _judge_timeout=600 ;; esac
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-${_judge_default}}"
JUDGE_TIMEOUT="${JUDGE_TIMEOUT:-${_judge_timeout}}"
FAILED=()
T0=$(date +%s)

step() {  # step <name> <command...>: run, log, never abort the chain
    local name="$1"; shift
    local t=$(date +%s)
    echo; echo "[STEP $(date -Is)] ${name}"; echo "    $*"
    if "$@"; then
        echo "[OK   $(date -Is)] ${name} ($(( ($(date +%s) - t) / 60 )) min)"
    else
        echo "[FAIL $(date -Is)] ${name} (exit $?)"
        FAILED+=("${name}")
    fi
}

skip() { echo; echo "[SKIP $(date -Is)] $1: $2"; }

have_set() {  # have_set <dir>: all three item-level files present
    [ -f "$1/winogrande_misinformed.json" ] && [ -f "$1/ethics_commonsense_misinformed.json" ] && [ -f "$1/complex_web_questions_misinformed.json" ]
}
have_exp2() { [ -f "out/${MODEL_DIR}/exp2_results_winogrande.json" ]; }
have_exposure() { ls "out/${MODEL_DIR}/exposure/exposure_results_winogrande_seed0.json" >/dev/null 2>&1; }

finish() {
    echo; echo "================ chain summary (${MODEL_NAME}, $(( ($(date +%s) - T0) / 60 )) min) ================"
    if [ "${#FAILED[@]}" -eq 0 ]; then
        echo "all steps OK"; exit 0
    fi
    printf 'FAILED: %s\n' "${FAILED[@]}"
    exit 1
}

# ---- reusable step groups -------------------------------------------------------------------
exposure_and_msc() {  # exposure curve k=0..3 (seed 0) and seeds 1,2 for k=1,3; same for mSC
    step "exposure k=0,1,2,3 seed0"  python exp_exposure.py "${COMMON[@]}" --k 0,1,2,3 --mint_filter verified --seed 0 --continue
    step "msc k=0,1,2,3 seed0"       python exp_msc.py      "${COMMON[@]}" --k 0,1,2,3 --mint_filter verified --seed 0 --continue
    local s
    for s in 1 2; do
        step "exposure k=1,3 seed${s}" python exp_exposure.py "${COMMON[@]}" --k 1,3 --mint_filter verified --seed "${s}" --continue
        step "msc k=1,3 seed${s}"      python exp_msc.py      "${COMMON[@]}" --k 1,3 --mint_filter verified --seed "${s}" --continue
    done
}

exp1_main() {  # single agent, default conditions (seed 0) + seeds 1,2 for baseline/false_fact
    step "exp1 default conditions seed0" python exp1.py "${COMMON[@]}" --conditions default --seed 0 --reprompt_unparsed --continue
    local s
    for s in 1 2; do
        step "exp1 baseline,false_fact seed${s}" python exp1.py "${COMMON[@]}" --conditions baseline,false_fact --seed "${s}" --continue
    done
}

exp1_controls() {  # controls: relevant_true + filler (+ irrelevant_true) from the relevant-true control set (carries all v1.2 fields)
    if have_set "${RT_DIR}"; then
        step "exp1 controls (relevant_true, filler, irrelevant_true)" python exp1.py "${COMMON[@]}" --conditions controls --seed 0 \
            --reprompt_unparsed --data_dir "${RT_DIR}" --continue
    else
        skip "exp1 controls" "${RT_DIR} missing (Llama phase 1 generates it)"
    fi
}

exp1_on_generator() {  # generator x consumer: consumer on another generator's passages: exp1_on_generator <gen_key> <gen_dir>
    if have_set "$2"; then
        step "exp1 on $1 passages" python exp1.py "${COMMON[@]}" --conditions baseline,false_fact --seed 0 --reprompt_unparsed \
            --data_dir "$2" --output_dir "out/${MODEL_DIR}/generators/$1" --continue
    else
        skip "exp1 on $1 passages" "$2 missing"
    fi
}

relevant_true_debates() {  # relevant-true passages in debates: k=1,3 on WinoGrande
    if have_set "${RT_DIR}"; then
        step "exposure relevant_true k=1,3 winogrande" python exp_exposure.py "${COMMON[@]}" --k 1,3 --information_source relevant_true \
            --data_dir "${RT_DIR}" --datasets winogrande --seed 0 --continue
    else
        skip "exposure relevant_true" "${RT_DIR} missing"
    fi
}

message_ablation() {  # counterfactual message ablation: needs the full-sample exp2 transcripts in out/<model>/
    if ls out/results/ablation_run_"$1"_*.json >/dev/null 2>&1; then
        skip "message ablation" "out/results/ablation_run_$1_*.json exists (raw file is append-only; delete to re-run)"
    elif have_exp2; then
        step "message ablation (300 debates x 3 datasets)" python exp_message_ablation.py "${COMMON[@]}" --n_debates 300 --seed 0
    else
        skip "message ablation" "out/${MODEL_DIR}/exp2_results_*.json missing (run exp2.py first, see slurm/exp2.slurm)"
    fi
}

mitigations() {  # source-skepticism prompt (debates) + verifier agent (final stage re-run)
    step "exposure source_skepticism k=1,3 winogrande,cwq" python exp_exposure.py "${COMMON[@]}" --k 1,3 --mitigation source_skepticism \
        --datasets winogrande,cwq --mint_filter verified --seed 0 --continue
    if have_exp2; then
        step "verifier mitigation (exp2 transcripts)" python exp_mitigation_verifier.py "${COMMON[@]}" --datasets winogrande,cwq --source exp2
    elif have_exposure; then
        step "verifier mitigation (exposure transcripts)" python exp_mitigation_verifier.py "${COMMON[@]}" --datasets winogrande,cwq --source exposure
    else
        skip "verifier mitigation" "no exp2 or exposure transcripts for ${MODEL_DIR}"
    fi
}

optional_all_categories() {  # every passage category in debates (expensive, ~6x the false_fact exposure run); CHAIN_ALL_CATEGORIES=1
    if [ "${CHAIN_ALL_CATEGORIES:-0}" = "1" ]; then
        step "exposure all categories k=1,3" python exp_exposure.py "${COMMON[@]}" --k 1,3 --categories all --mint_filter verified --seed 0 --continue
    fi
}

judge() {  # judge <data_dir> [extra validate_mint args]: idempotent (already-judged rows are skipped)
    local dir="$1"; shift
    step "judge ${dir}" python scripts/validate_mint.py --judge_model "${MODEL_NAME}" --endpoint_url "${ENDPOINT_URL}" \
        --data_dir "${dir}" --max_tokens "${JUDGE_MAX_TOKENS}" --timeout "${JUDGE_TIMEOUT}" "$@"
}
