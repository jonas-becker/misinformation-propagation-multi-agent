#!/usr/bin/env bash
# Llama-3.3-70B, phase 2 (after all phase-1 jobs and the GLM judge): main experiments (2 GPUs).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
exp1_on_generator glm "${GEN_GLM}"
exp1_on_generator gptoss "${GEN_GPTOSS}"
if have_set "${PARA_DIR}"; then
    step "exp1 on paraphrased passages (style control)" python exp1.py "${COMMON[@]}" --conditions baseline,false_fact --seed 0 \
        --reprompt_unparsed --data_dir "${PARA_DIR}" --output_dir "out/${MODEL_DIR}/controls/paraphrase_gptoss" --continue
else skip "exp1 paraphrase control" "${PARA_DIR} missing"; fi
exposure_and_msc
relevant_true_debates
message_ablation "Llama-3.3-70B"
mitigations
optional_all_categories
finish
