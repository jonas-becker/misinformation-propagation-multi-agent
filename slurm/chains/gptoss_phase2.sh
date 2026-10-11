#!/usr/bin/env bash
# GPT-OSS-120B, phase 2 (after all phase-1 jobs and the GLM judge): main experiments.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
exp1_on_generator glm "${GEN_GLM}"          # generator x consumer (reads GLM set; judged only at the very end of this chain)
exp1_controls                                # controls, single agent
exposure_and_msc                             # exposure curve + mSC (largest block)
relevant_true_debates                        # relevant-true passages in debates
message_ablation "GPT-OSS-120B"              # message ablation (needs exp2 transcripts)
mitigations                                  # skepticism prompt + verifier agent
optional_all_categories
# cross-family judging last (writes the GLM set and the relevant-true set in place)
if have_set "${GEN_GLM}"; then judge "${GEN_GLM}"; fi
if have_set "${RT_DIR}"; then judge "${RT_DIR}" --passages_suffix _relevant_true_passages --mode relevant_true; fi
finish
