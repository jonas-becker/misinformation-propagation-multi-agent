# Running the experiments

All commands assume a vLLM OpenAI-compatible endpoint for the model named in `--model_name`
(the `slurm/*.slurm` scripts start one; otherwise pass `--no_autostart --endpoint_url http://host:port/v1`).
Run every new configuration first with `--max_items 50` (smoke test), then at full scale. Every runner
writes `run_metadata_*.json` next to its results. MALLM is expected as a sibling checkout (`../mallm`).

## Serving settings

* GLM-4.7-Flash: `--reasoning-parser glm45`, max_tokens 4096 for debates and 2048 for the single agent
  (without the parser the thinking text is streamed into the answer and truncated).
* GPT-OSS-120B: `--reasoning-parser openai_gptoss`, same token limits.
* Llama-3.3-70B: no reasoning parser; 1,024 tokens for debate turns, 512 for the single agent
  (the server applies the chat template).
* `model_server_config.vllm_reasoning_parser_default` returns "" by default; set `VLLM_REASONING_PARSER`
  in the environment for the reasoning models (the run metadata records it).
* Llama-3.3-70B in bf16 at tensor parallel 2 takes 65.7 GiB per GPU. On an 80 GB A100 the default
  `VLLM_GPU_MEMORY_UTILIZATION=0.90` leaves about 4.3 GiB of KV cache (roughly 27k tokens), which is why the
  five-agent sweep was served with an 8,192-token context; a 32,768-token request needs 5.0 GiB, so serve with
  `VLLM_GPU_MEMORY_UTILIZATION=0.95 VLLM_MAX_NUM_BATCHED_TOKENS=2048` (or 4 GPUs) when longer prompts are needed.

## MINT

```bash
python scripts/build_mint_v1_2.py                     # release files from v1.1 + v1 labels
python scripts/generate_filler.py                     # adds filler_information to MINT_dataset_v1.2

# judge validation (run once per judge model; ~10k passages x 2 judges)
python scripts/validate_mint.py --judge_model openai/gpt-oss-120b --endpoint_url $EP_GPTOSS
python scripts/validate_mint.py --judge_model zai-org/GLM-4.7-Flash --endpoint_url $EP_GLM
python scripts/validate_mint.py --stats_only          # tables + inter-judge kappa

# cross-generator passages (same pipeline, generator model swapped) and controls
python scripts/generate_mint_passages.py --mode misinformation --model_name zai-org/GLM-4.7-Flash --endpoint_url $EP_GLM --no_autostart
python scripts/generate_mint_passages.py --mode misinformation --model_name openai/gpt-oss-120b --endpoint_url $EP_GPTOSS --no_autostart
python scripts/generate_mint_passages.py --mode paraphrase --model_name openai/gpt-oss-120b --endpoint_url $EP_GPTOSS --no_autostart
python scripts/generate_mint_passages.py --mode relevant_true --model_name meta-llama/Llama-3.3-70B-Instruct --endpoint_url $EP_LLAMA --no_autostart
# validate the new passage sets with judges from other families than the generator
python scripts/validate_mint.py --data_dir MINT_dataset_v1.2/generators/glm --judge_model openai/gpt-oss-120b ...
python scripts/validate_mint.py --data_dir MINT_dataset_v1.2/generators/gptoss --judge_model zai-org/GLM-4.7-Flash ...
python scripts/validate_mint.py --data_dir controls/relevant_true --passages_suffix _relevant_true_passages --mode relevant_true --judge_model openai/gpt-oss-120b ...
```

## Main runs (per model)

```bash
M=openai/gpt-oss-120b; EP=$EP_GPTOSS
# exposure curve: k = 0..3 of 3 agents, both protocols come from the same transcripts
python exp_exposure.py --model_name $M --endpoint_url $EP --no_autostart --k 0,1,2,3 --mint_filter verified --seed 0
python exp_msc.py      --model_name $M --endpoint_url $EP --no_autostart --k 0,1,2,3 --mint_filter verified --seed 0
# seeds 1 and 2 for k = 1 and 3 (debate and mSC)
python exp_exposure.py --model_name $M --endpoint_url $EP --no_autostart --k 1,3 --mint_filter verified --seed 1
python exp_exposure.py --model_name $M --endpoint_url $EP --no_autostart --k 1,3 --mint_filter verified --seed 2
python exp_msc.py      --model_name $M --endpoint_url $EP --no_autostart --k 1,3 --mint_filter verified --seed 1
python exp_msc.py      --model_name $M --endpoint_url $EP --no_autostart --k 1,3 --mint_filter verified --seed 2
# WinoGrande: Q1 is ill-posed for its fictional scenarios, so run k >= 1 on all questions (--mint_filter all);
# the analyses restrict them to the questions verified on Q2
# single agent: seed 0 covers all conditions, seeds 1 and 2 repeat baseline and false_fact
python exp1.py --model_name $M --endpoint_url $EP --no_autostart --conditions all --seed 0 --reprompt_unparsed
python exp1.py --model_name $M --endpoint_url $EP --no_autostart --conditions baseline,false_fact --seed 1 --reprompt_unparsed
python exp1.py --model_name $M --endpoint_url $EP --no_autostart --conditions baseline,false_fact --seed 2 --reprompt_unparsed
# generator x consumer matrix: every consumer on the other generators' passages
python exp1.py --model_name $M --endpoint_url $EP --no_autostart --conditions baseline,false_fact --data_dir MINT_dataset_v1.2/generators/glm    --output_dir out/<model_dir>/generators/glm
python exp1.py --model_name $M --endpoint_url $EP --no_autostart --conditions baseline,false_fact --data_dir MINT_dataset_v1.2/generators/gptoss --output_dir out/<model_dir>/generators/gptoss
# paraphrase style control (Llama consumer only)
python exp1.py --model_name meta-llama/Llama-3.3-70B-Instruct ... --conditions baseline,false_fact --data_dir controls/paraphrase_gptoss --output_dir out/meta-llama_Llama-3.3-70B-Instruct/controls/paraphrase_gptoss
# relevant-true passages in debates (k = 1 and 3)
python exp_exposure.py --model_name $M --endpoint_url $EP --no_autostart --k 1,3 --information_source relevant_true --data_dir controls/relevant_true
# Llama mSC at the debate's sampling parameters (sensitivity check)
python exp_msc.py --model_name meta-llama/Llama-3.3-70B-Instruct ... --k 0,1,2,3 --temperature 0.6 --top_p 0.9 --output_dir out/meta-llama_Llama-3.3-70B-Instruct/msc_matched
```

## Mitigations and propagation

```bash
# source-skepticism instruction: false passages (k = 1, 3), relevant-true passages (k = 1, 3), no context (k = 0)
python exp_exposure.py --model_name $M --endpoint_url $EP --no_autostart --k 1,3 --mitigation source_skepticism --mint_filter verified
python exp_exposure.py --model_name $M --endpoint_url $EP --no_autostart --k 1,3 --mitigation source_skepticism --information_source relevant_true --data_dir controls/relevant_true
python exp_exposure.py --model_name $M --endpoint_url $EP --no_autostart --k 0 --mitigation source_skepticism
# verifier agent on the stored seed-0 k = 1 debates
python exp_mitigation_verifier.py --model_name $M --endpoint_url $EP --no_autostart
# counterfactual message ablation (one turn regenerated per debate; raw outputs appended to a jsonl, analysed with --analyze_only)
python exp_message_ablation.py --model_name $M --endpoint_url $EP --no_autostart --n_debates 300
```

## Full-sample run set

`exp2.py` (1-of-3 debates, all nine passage categories, misinformed agent always last), `exp3.py`
(five-agent WinoGrande composition sweep), `exp2_ablation.py` (15-sample self-consistency) and `exp1a.py`
(single agent with the irrelevant-true passages of MINT v1.1) run on all questions of the sampled subsets;
the `slurm/*.slurm` scripts launch them. Their outputs are re-scored with the unified extractor by
`scripts/rescore_results.py`.

## Cluster launcher and chains (`slurm/`)

`slurm/run_chain.slurm` starts the vLLM server for `MODEL_NAME` (reasoning parser for GLM / GPT-OSS, a
per-job port, the offline tiktoken cache) and then runs the chain script named in `CHAIN_SCRIPT` with
`MODEL_NAME`, `ENDPOINT_URL` and `REPO_ROOT` exported. The chains in `slurm/chains/` call the runners with
`--no_autostart`; every step is idempotent (`--continue`, skip-if-exists), a failing step is logged and the
chain continues. Each step writes one `[OK ...]`, `[FAIL ...]` or `[SKIP ...]` line with its wall-clock time
into `out/<model>/runs/job_<id>/slurm.log`.

```bash
cp slurm/site.env.example slurm/site.env       # Slurm account, container image, cache paths (gitignored)
bash slurm/submit_chains.sh --dry-run           # the three per-model chains and their dependencies
bash slurm/submit_chains.sh                     # phase 1 -> GLM judge -> phase 2
# one chain, e.g. the sampling-matched mSC and skepticism runs for Llama (2 GPUs):
MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct CHAIN_SCRIPT=slurm/chains/msc_matched_skepticism.sh \
    sbatch --export=ALL --gpus=2 slurm/run_chain.slurm
# export the variables in the shell instead of --export=ALL,NAME=... (sbatch splits --export on commas)
```

Chains: `gptoss_phase1`, `glm_phase1`, `llama_phase1` (passage sets, GPT-OSS judge pass, single agent);
`glm_judge` (second judge, after the GPT-OSS pass because both edit the release files in place);
`*_phase2` (exposure curve, mSC, seeds, controls, message ablation, mitigations, cross-family judging);
`followup` (single-agent seeds with re-prompting, diagonal generator cells, WinoGrande full-sample debates);
`msc_matched_skepticism` (sampling-matched mSC, skepticism on ETHICS, without context and with relevant-true passages);
`message_ablation`, `rejudge_winogrande`, `exp3_fill`; `verify_*` are short GPU checks of the serving setup.
Serving knobs: `VLLM_MAX_MODEL_LEN`, `VLLM_MAX_NUM_BATCHED_TOKENS`, `VLLM_GPU_MEMORY_UTILIZATION`,
`VLLM_REASONING_PARSER`; `SMOKE_ONLY=1` stops after the server smoke test.

Compute nodes without internet: `slurm/job_helpers.sh` picks a free port per job (two jobs on one node
do not collide), checks that the endpoint serves this job's model, stops the server process group
without hanging, and primes `TIKTOKEN_CACHE_DIR` from the offline vocab files so that host-side `tiktoken`
(`model_server_config.offline_tiktoken_encoding`, used by `scripts/validate_mint.py` and `scripts/costs.py`)
never tries to download. `bash slurm/test_job_helpers.sh` unit-tests the helpers on a login node.

## Question subsets (`--mint_ids_file`)

`exp_exposure.py` and `exp_msc.py` accept `--mint_ids_file configs/subsets/verified_{dataset}.txt`, which
restricts every condition (including k = 0) to the listed MINT ids. The lists are written by
`scripts/make_verified_subsets.py` (`make subsets`) with the verification rule of the analyses
(`scripts/common.py::verified_questions`; both judges Q1 and Q2 for CWQ and ETHICS, both judges Q2 for
WinoGrande). They are needed for passage sets without judge flags (the relevant-true control set) and for
runs that should cover exactly the verified questions at k = 0. The tracked files are the ones the
`msc_matched_skepticism` runs used: CWQ 98, ETHICS 114 and WinoGrande 182 questions, the latter from the re-judged
WinoGrande flags (next section; 148 of them are in the release's 183-question subset).

## WinoGrande re-judge with the plausibility wording of Q1

The release flags ask whether a passage contradicts the gold answer (Q1). WinoGrande's fictional scenarios
have nothing to contradict (27 of 382 questions pass Q1 and Q2), so the release verifies WinoGrande on Q2
alone (`configs/experiment_config.json` `verification_criterion`). As a robustness check both judges were
run again on a copy of the WinoGrande files with Q1 asking whether the passage makes the intended wrong
option more plausible than the gold completion (`scripts/validate_mint.py --q1_criterion plausibility`):

```bash
MODEL_NAME=openai/gpt-oss-120b  CHAIN_SCRIPT=slurm/chains/rejudge_winogrande.sh sbatch --export=ALL slurm/run_chain.slurm
MODEL_NAME=zai-org/GLM-4.7-Flash CHAIN_SCRIPT=slurm/chains/rejudge_winogrande.sh sbatch --export=ALL --dependency=afterany:<job> slurm/run_chain.slurm
```

The re-judged files live in `MINT_dataset_v1.2/rejudge_q1_plausibility/` (116 questions pass Q1 and Q2, 182
pass Q2), the tables in `out/results/rejudge_winogrande_q1/<judge>/` and
`out/results/mint_validation_stats_rejudge_q1.csv`. The release files are not modified.

## Five-agent composition sweep: debates that never ran

MALLM replaces a failed discussion by re-running the first entries of its input list
(`mallm/scheduler.py`, "Resampling from the dataset as a substitute"). `exp3.py` did not set
`Config.num_samples`, so Llama's five-agent WinoGrande sweep, served with an 8,192-token context that
five-agent memory-paradigm prompts exceed, ended with 382 records per mix but only 337-377 distinct
questions (176 question x mix cells never ran; GLM-4.7 and GPT-OSS were complete). `make rescore`
reports such duplicates (`n_duplicate_item` in `accuracy_summary.csv` plus a warning).
`exp3_fill_missing.py` completes the sweep without touching the original debates:

```bash
# 1. plan: the missing question x mix cells, from the raw sweep file or from a rescored mad_final.csv
make exp3_fill_plan [EXP3_FILL_CSV=out/results/rescored/mad_final.csv]
# 2. GPU: the original exp3 configuration (memory paradigm, simple_voting, 5 agents, 5 turns, 1,024 tokens, plain
#    false fact, identical per-question slot draw, seed 0) against a 32,768-token server; Config.num_samples
#    disables the substitution, failures are retried in up to three passes -> out/<model>/exp3_fill/
MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct VLLM_MAX_MODEL_LEN=32768 VLLM_MAX_NUM_BATCHED_TOKENS=2048 \
  VLLM_GPU_MEMORY_UTILIZATION=0.95 CHAIN_SCRIPT=slurm/chains/exp3_fill.sh \
  sbatch --export=ALL --gpus=2 --time=20:00:00 slurm/run_chain.slurm
# 3. offline: keep the first debate of every question, drop the substitutes, insert the fill-in debates
#    (backup next to the file, report in out/<model>/exp3_fill/merge_report_*.json), then rescore
make exp3_fill_merge && make rescore fig_composition
```

For Llama all 176 missing debates completed in one pass; every mix has one debate per question
(n = 382, 183 on the verified subset).

## Sampling-matched mSC and further mitigation runs (`slurm/chains/msc_matched_skepticism.sh`)

* **Sampling-matched mSC (Llama only).** The debates send no sampling parameters, so Llama-3.3 debates run at
  its server default (T 0.6 / top-p 0.9) while mSC sampled at 1.0 / 1.0. `exp_msc.py --temperature 0.6 --top_p 0.9
  --output_dir out/<model>/msc_matched` repeats every mSC condition; the rescorer labels these rows
  `exp_msc_matched` and `scripts/msc_matched_sensitivity.py` (part of `make fig_exposure`) compares them.
* **Source-skepticism prompt on ETHICS** (k = 1, 3, verified questions) plus the verifier agent on the ETHICS
  k = 1 transcripts.
* **True-context cost of the prompt.** Skepticism debates with the relevant-true passage (k = 1, 3) and without
  any passage (k = 0) on all tasks, plus relevant-true debates without the prompt on CWQ and ETHICS, all on the
  question subsets above (`--mint_ids_file`); analysed by `scripts/mitigation_effects.py` (`make fig_mitigation`).

## Analysis

`make all` (see the Makefile) re-scores every stored output under `out/<model>/` and regenerates all
figures and tables; `make rescore` alone takes 15-40 minutes, `scripts/propagation_metrics.py` about
25 minutes (`--tables_only` rewrites its tables from the CSV). The re-score of the full-sample debate transcripts
needs more than 60 GB of RAM (12 GB of GLM-4.7 transcripts); on the cluster run it as a CPU job, e.g.
`sbatch -p scc-cpu --mem=120G --wrap "make rescore"`. `make subsets`, `make exp3_fill_plan` and
`make exp3_fill_merge` are the offline halves of the sections above.
