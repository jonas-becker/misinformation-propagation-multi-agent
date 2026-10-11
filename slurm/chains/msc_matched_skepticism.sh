#!/usr/bin/env bash
# Sampling-matched mSC and source-skepticism runs on ETHICS, without context and with relevant-true passages
# (run through slurm/run_chain.slurm via CHAIN_SCRIPT; Llama needs --gpus=2):
#   msc_matched   sampling-matched mSC. The debates send no sampling parameters, so Llama-3.3 debates run at its server
#                 default (T 0.6 / top-p 0.9) while mSC ran at T 1.0 / top-p 1.0 (paper, Sec. E.1). Re-run mSC at
#                 T 0.6 / top-p 0.9 into out/<model>/msc_matched/ (same conditions, seeds and question sets as
#                 out/<model>/msc/). Llama only: GLM-4.7 and GPT-OSS debates already run at the vLLM default 1.0 / 1.0 = mSC.
#   ethics_skept  source-skepticism prompt on ETHICS (k = 1, 3; the 114 verified questions) + verifier agent on ETHICS.
#   k0_skept, rt_skept   cost of the skepticism prompt when the context is absent or true: skepticism debates without any
#                 passage (k = 0) and with relevant-true passages (k = 1, 3) on all tasks. Question sets: the judge-verified
#                 subsets of the mitigation table (configs/subsets/verified_<dataset>.txt: CWQ 98, ETHICS 114, WinoGrande 182),
#                 passed with --mint_ids_file because the relevant-true control set carries no validity flags. Controls:
#                 the k = 0 debates and the WinoGrande relevant-true debates (all 382); relevant-true debates without the
#                 prompt are run here for CWQ and ETHICS on the same subsets.
# Select parts with MSC_SKEPT_PARTS (comma list), default all: msc_matched,ethics_skept,k0_skept,rt_skept
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
PARTS="${MSC_SKEPT_PARTS:-msc_matched,ethics_skept,k0_skept,rt_skept}"
has() { [[ ",${PARTS}," == *",$1,"* ]]; }
SUBSET="configs/subsets/verified_{dataset}.txt"
[ -f configs/subsets/verified_winogrande.txt ] || step "verified subsets (offline)" python scripts/make_verified_subsets.py
echo "[INFO] parts: ${PARTS} (model ${MODEL_NAME})"

if has msc_matched; then
    case "${MODEL_NAME}" in
        meta-llama/Llama-3.3-70B-Instruct)
            MSC_M=(python exp_msc.py "${COMMON[@]}" --temperature 0.6 --top_p 0.9 --output_dir "out/${MODEL_DIR}/msc_matched" --continue)
            step "msc matched (T0.6/p0.9) k=0,1,2,3 seed0 cwq,ethics verified" "${MSC_M[@]}" --datasets cwq,ethics --k 0,1,2,3 --mint_filter verified --seed 0
            step "msc matched (T0.6/p0.9) k=0,1,2,3 seed0 winogrande all"      "${MSC_M[@]}" --datasets winogrande --k 0,1,2,3 --mint_filter all --seed 0
            for s in 1 2; do
                step "msc matched (T0.6/p0.9) k=1,3 seed${s} cwq,ethics verified" "${MSC_M[@]}" --datasets cwq,ethics --k 1,3 --mint_filter verified --seed "${s}"
                step "msc matched (T0.6/p0.9) k=1,3 seed${s} winogrande all"      "${MSC_M[@]}" --datasets winogrande --k 1,3 --mint_filter all --seed "${s}"
            done ;;
        *) skip "msc matched" "only Llama-3.3 debates run at a non-default temperature (GLM/GPT-OSS: server default 1.0/1.0 = mSC)" ;;
    esac
fi

if has ethics_skept; then
    step "exposure source_skepticism k=1,3 ethics verified" python exp_exposure.py "${COMMON[@]}" --datasets ethics --k 1,3 \
        --mitigation source_skepticism --mint_filter verified --seed 0 --continue
    step "verifier mitigation ethics (exposure seed0 transcripts)" python exp_mitigation_verifier.py "${COMMON[@]}" --datasets ethics --source exposure
fi

if has k0_skept; then  # no context
    step "exposure source_skepticism k=0 cwq,ethics,winogrande (verified subsets)" python exp_exposure.py "${COMMON[@]}" \
        --datasets cwq,ethics,winogrande --k 0 --mitigation source_skepticism --mint_ids_file "${SUBSET}" --seed 0 --continue
fi

if has rt_skept; then  # true context
    if have_set "${RT_DIR}"; then
        step "exposure relevant_true k=1,3 cwq,ethics (verified subsets; control)" python exp_exposure.py "${COMMON[@]}" \
            --datasets cwq,ethics --k 1,3 --information_source relevant_true --data_dir "${RT_DIR}" --mint_ids_file "${SUBSET}" --seed 0 --continue
        step "exposure relevant_true + source_skepticism k=1,3 cwq,ethics,winogrande (verified subsets)" python exp_exposure.py "${COMMON[@]}" \
            --datasets cwq,ethics,winogrande --k 1,3 --information_source relevant_true --mitigation source_skepticism \
            --data_dir "${RT_DIR}" --mint_ids_file "${SUBSET}" --seed 0 --continue
    else
        skip "relevant_true skepticism" "${RT_DIR} missing"
    fi
fi
finish
