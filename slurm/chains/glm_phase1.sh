#!/usr/bin/env bash
# GLM-4.7-Flash, phase 1 (independent): GLM passage set, single agent. The GLM judge pass runs in glm_judge.sh
# AFTER the GPT-OSS judge pass, because both write MINT_dataset_v1.2/*_passages.jsonl in place.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
if have_set "${GEN_GLM}"; then skip "generate glm set" "exists"; else
    step "generate misinformation set (glm)" python scripts/generate_mint_passages.py --mode misinformation "${COMMON[@]}"; fi
exp1_main
finish
