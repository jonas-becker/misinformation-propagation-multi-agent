#!/usr/bin/env bash
# GLM-4.7-Flash, phase 2 (after all phase-1 jobs and the GLM judge): main experiments.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
exp1_on_generator gptoss "${GEN_GPTOSS}"
exp1_controls
exposure_and_msc
relevant_true_debates
message_ablation "GLM-4.7-Flash"
mitigations
optional_all_categories
if have_set "${GEN_GPTOSS}"; then judge "${GEN_GPTOSS}"; fi   # cross-family judging of the GPT-OSS set, last
finish
