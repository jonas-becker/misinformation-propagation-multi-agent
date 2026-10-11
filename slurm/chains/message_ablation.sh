#!/usr/bin/env bash
# Counterfactual message ablation on the full-sample exp2 transcripts (exp2.py, slurm/exp2.slurm). If they were produced
# on another machine, copy out/<model>/exp2_results_{winogrande,ethics,cwq}.json to the cluster first. Output: out/results/ablation_propagation_raw.jsonl (append-only, shared by all models) and
# out/results/ablation_run_<model>_<timestamp>.json; the analysis (--analyze_only) reads the raw file.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
if ! have_exp2; then
    echo "[FAIL] out/${MODEL_DIR}/exp2_results_*.json missing: run exp2.py (slurm/exp2.slurm) or copy its transcripts to out/${MODEL_DIR}/ first."
    FAILED+=("message ablation (no exp2 transcripts)")
    finish
fi
step "message ablation (300 debates x 3 datasets)" python exp_message_ablation.py "${COMMON[@]}" --n_debates 300 --seed 0
finish
