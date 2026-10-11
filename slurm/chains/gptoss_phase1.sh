#!/usr/bin/env bash
# GPT-OSS-120B, phase 1 (independent): judge pass on MINT, GPT-OSS passage set + paraphrase control, single agent.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
judge MINT_dataset_v1.2                                                                             # first judge (~1-2 h)
if have_set "${GEN_GPTOSS}"; then skip "generate gptoss set" "exists"; else
    step "generate misinformation set (gptoss)" python scripts/generate_mint_passages.py --mode misinformation "${COMMON[@]}"; fi
if have_set "${PARA_DIR}"; then skip "paraphrase control" "exists"; else
    step "paraphrase Llama passages (style control)" python scripts/generate_mint_passages.py --mode paraphrase "${COMMON[@]}"; fi
exp1_main                                                                                            # single agent (~1.5 h)
finish
