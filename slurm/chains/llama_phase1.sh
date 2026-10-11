#!/usr/bin/env bash
# Llama-3.3-70B, phase 1 (independent): relevant-true control set (Llama generated MINT), single agent incl. controls.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
if have_set "${RT_DIR}"; then skip "generate relevant_true" "exists"; else
    step "generate relevant_true control set" python scripts/generate_mint_passages.py --mode relevant_true "${COMMON[@]}"; fi
exp1_main
exp1_controls
finish
