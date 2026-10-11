#!/usr/bin/env bash
# Re-judge the WinoGrande passages with the plausibility wording of Q1 (scripts/validate_mint.py --q1_criterion
# plausibility): the release flags ask whether the passage contradicts the gold completion, which is ill-posed for
# WinoGrande's fictional scenarios (27/382 verified on Q1 and Q2), so the release verifies WinoGrande on Q2 alone.
# The re-judge is a robustness check and does NOT touch the release files: it judges a copy of the WinoGrande files
# in MINT_dataset_v1.2/rejudge_q1_plausibility/ (created from the release on first use). Run once per judge model
# through slurm/run_chain.slurm via CHAIN_SCRIPT, serialized with --dependency (both judges edit the copy in
# place). Per-judge tables go to out/results/rejudge_winogrande_q1/<judge>/, the final --stats_only step writes
# out/results/mint_validation_stats_rejudge_q1.csv and mint_judge_agreement_rejudge_q1.csv from both judges' flags.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
RJ_DIR="MINT_dataset_v1.2/rejudge_q1_plausibility"
if [ ! -f "${RJ_DIR}/winogrande_passages.jsonl" ]; then
    mkdir -p "${RJ_DIR}"
    cp MINT_dataset_v1.2/winogrande_misinformed.json MINT_dataset_v1.2/winogrande_passages.jsonl "${RJ_DIR}/"
    echo "[INFO] copied the WinoGrande release files to ${RJ_DIR} (the release itself is not modified)"
fi
JK="$(python -c "from scripts.validate_mint import _judge_key; print(_judge_key('${MODEL_NAME}'))")"
judge "${RJ_DIR}" --datasets winogrande --overwrite --q1_criterion plausibility --results_dir "out/results/rejudge_winogrande_q1/${JK}"
step "re-judge tables (both judges)" python scripts/validate_mint.py --stats_only --datasets winogrande --data_dir "${RJ_DIR}" --out_tag rejudge_q1
finish
