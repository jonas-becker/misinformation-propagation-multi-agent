# Misinformation Propagation in Benign Multi-Agent Systems

Code, data (MINT) and analysis for the paper. Three consumer models (Llama-3.3-70B-Instruct,
GLM-4.7-Flash, GPT-OSS-120B), three tasks (ComplexWebQuestions, ETHICS commonsense, WinoGrande),
single agent vs. multi-agent debate (MALLM) with k of n agents exposed to a MINT passage, and an
exposure-matched self-consistency baseline (mSC).

## Reproduce the figures and tables

Everything in the paper is computed offline from the stored model outputs; no GPU is needed.

```bash
pip install -r requirements.txt
make all             # re-scores the stored outputs and regenerates every figure and table
make fig_exposure    # or a single target; the Makefile lists them
```

Figures are written to `out/figures/`, tables to `out/tables/*.tex`, the numbers behind them to
`out/results/*.csv`, and the appendix material (model settings, experimental details, run inventory,
prompts) to `out/appendix/`. `make rescore` re-parses every stored output under `out/<model>/` into
`out/results/rescored/*.csv`; all analysis scripts read those CSVs. The stored model outputs and the
generated results are distributed separately; this repository holds the code, the data and the prompts.

The appendix tables are generated from `configs/experiment_config.json`, the single source of truth
for the experimental settings that the runners also read (the `*_short` fields hold the wording of the
compact tables).

**LaTeX tables.** Every `.tex` table is written by `scripts/common.py::write_latex_table` and can be
`\input` unchanged into the ACL template; the main file only needs `\usepackage{booktabs}`. Text is
escaped outside inline math, unicode symbols are converted, wide tables are emitted as `table*`, and
tables longer than a page are split into continued floats that share one label.

**Figures.** All figures share one style (`scripts/figure_style.py`, applied through
`scripts/figure_common.py`): serif fonts, the project palette, framed legend above the panels, PDF and
PNG at 300 dpi cropped to the drawn ink. Figures meant for one ACL column are drawn at column size
(3.3 in wide, 8-9 pt fonts).

## Layout

| Path | Purpose |
|---|---|
| `exp1.py` | single agent: baseline, false_fact, strategy_*, irrelevant_misinformed, irrelevant_true, relevant_true, filler |
| `exp_exposure.py` | multi-agent debate with k of n exposed agents (passage sharing, information source, source-skepticism mitigation) |
| `exp_msc.py` | misinformed self-consistency: same exposure and aggregation as the voting debate, no interaction |
| `exp_mitigation_verifier.py` | verifier-agent mitigation, re-running the final stage of stored debates |
| `exp_message_ablation.py` | counterfactual message ablation on stored debates (`results/ablation_propagation_raw.jsonl` holds the raw outputs) |
| `exp2.py`, `exp3.py`, `exp2_ablation.py`, `exp1a.py` | the 1-of-3 debates on all passage categories, the five-agent composition sweep, the 15-sample self-consistency baseline and the irrelevant-true single-agent run (full-sample run set) |
| `exp3_fill_missing.py` | completes the five-agent sweep: re-runs the question x mix cells that MALLM silently substituted (Llama, 8k context) and de-duplicates the sweep file |
| `mint_data.py` | canonical loading of MINT items and gold answers (see `MINT_dataset_v1.2/ERRATUM.md`) |
| `answer_extraction.py` | unified answer extractor with status `parsed / ambiguous / unparsed` |
| `runner_common.py`, `exposure_common.py`, `shared_utils.py`, `model_server_config.py` | shared runner infrastructure (chat client, condition keys, prompts, model serving defaults) |
| `mint_prompts.py`, `prompts/` | MINT generation prompts and the mitigation prompts |
| `scripts/` | analysis and figure scripts (`rescore_results`, `exposure_curve`, `decomposition`, `propagation_metrics`, `adoption_figure`, `control_2x2`, `relevant_true_debates`, `generator_consumer_matrix`, `category_regression`, `category_figure`, `mitigation_effects`, `composition_sweep_figure`, `msc_matched_sensitivity`, `verified_subset_robustness`, `costs`, `make_details_table`, `teaser_figure`; `stats` holds the bootstrap and permutation helpers) |
| `scripts/` (MINT) | `download_datasets` (MINT v1 generation), `extend_mint_dataset_v1_1`, `build_mint_v1_2`, `validate_mint`, `generate_mint_passages`, `generate_filler`, `sample_mint_for_annotation`, `evaluate_annotation`, `make_verified_subsets` |
| `slurm/` | SLURM launchers that start a vLLM server and run the experiments on a cluster; `run_chain.slurm` + `chains/*.sh` run whole experiment chains per model (`submit_chains.sh`), `job_helpers.sh` handles ports, process groups and the offline tiktoken cache (`docs/running_experiments.md`) |
| `configs/` | `experiment_config.json` (experimental settings), `dataset_config.json` (sampling of the task datasets), `subsets/verified_<dataset>.txt` (judge-verified question ids for `--mint_ids_file` runs) |
| `docs/` | how to run the experiments, MINT datasheet, taxonomy dimensions |
| `MINT_dataset_v1.2/` | MINT release: item-level JSON per dataset, `<dataset>_passages.jsonl` (one row per passage with judge flags), cross-generator sets in `generators/`, the WinoGrande re-judge with the plausibility wording of Q1 in `rejudge_q1_plausibility/` |
| `MINT-dataset_v1.1/`, `MINT_dataset_v1/` | previous releases (v1.1 carries the WinoGrande label error) and the human annotation of v1.1 |
| `controls/` | relevant-true passages (`relevant_true/`) and the GPT-OSS paraphrases of the Llama passages (`paraphrase_gptoss/`) |
| `out/` | stored model outputs (`out/<model>/`) and the generated `figures/`, `tables/`, `appendix/` and `out/results/` (CSVs); only `out/appendix/prompts.tex` is hand-written and tracked |

## Running the experiments

All runners talk to an OpenAI-compatible endpoint (vLLM). The `slurm/*.slurm` scripts start a
server and run the experiments; the runners also accept `--endpoint_url ... --no_autostart`.
`docs/running_experiments.md` lists every command in dependency order, the smoke-test flags
(`--max_items 50`) and the serving settings of the reasoning models (reasoning parser, 4,096 tokens).

MALLM is expected as a sibling checkout: `../mallm` (https://github.com/Multi-Agent-LLMs/mallm).

```bash
python exp1.py --model_name openai/gpt-oss-120b --endpoint_url http://host:port/v1 --no_autostart --conditions all
python exp_exposure.py --model_name openai/gpt-oss-120b --endpoint_url http://host:port/v1 --no_autostart --k 0,1,2,3
python exp_msc.py --model_name openai/gpt-oss-120b --endpoint_url http://host:port/v1 --no_autostart --k 0,1,2,3
```

Every run writes `run_metadata_*.json` (arguments, git commit, token counts) next to its results, under
`out/<model>/{exposure,msc,msc_matched,mitigation,seeds,generators,controls}`.

## MINT

`python scripts/build_mint_v1_2.py` rebuilds the release from v1.1 and the v1 labels;
`python scripts/validate_mint.py` adds the judge flags (`is_false`, `supports_target`, `explicit_cue`,
`n_tokens`, `verified`); `python scripts/generate_mint_passages.py` produces the cross-generator sets and
the controls with the same prompts and another generator. Datasheet: `docs/mint_datasheet.md`.

## Scoring and analysis conventions

* An answer counts as correct only if it is parsed unambiguously and equals the gold option; unparsed or
  ambiguous answers count as incorrect and their rates are reported per configuration. Free-form (CWQ)
  answers are correct if a gold answer or alias occurs (normalized, whole-word) in the final-answer segment
  when a marker was requested, otherwise in the whole response.
* CIs are paired bootstraps over questions (10,000 resamples); tests are paired permutation tests; the
  debate effects of the main table are Holm-adjusted (`scripts/stats.py`).
* **Verified subset.** CWQ and ETHICS debates and mSC runs with k >= 1 use the questions whose plain false
  fact both judges confirm on Q1 (contradicts the gold answer) and Q2 (supports the intended wrong answer):
  98 / 114 questions. On WinoGrande Q1 is ill-posed, so its k >= 1 runs cover all 382 questions and every
  analysis restricts them to the 183 questions verified on Q2; the 27 Q1-and-Q2 questions are a sensitivity
  set (`exposure_curve_strict.tex`). The rule per dataset is `mint_generation.verification_criterion` in
  `configs/experiment_config.json`, applied through `scripts.common.verified_questions`.
* **mSC aggregation.** `mSC` (protocol `vote`): majority over the three samples, first sample when there is
  no majority. `mSC-strict` (`vote-strict`): a question without a majority counts as wrong, like `SC15`.
* **Run sets.** `verified`: the exposure runs on the verified questions (plain false fact, three seeds at
  k = 1 and 3). `full-sample`: the earlier 1-of-3 debates on all passage categories and all questions
  (misinformed agent always last) and the five-agent sweep; both are re-scored with the same extractor.
* **Intended wrong answer** (propagation metrics): on the binary datasets the other option, on CWQ an
  answer matching the judges' recorded implied answer for that passage (normalised whole-word containment);
  answers to passages without a recorded target are excluded from adoption denominators.
* `scripts/validate_mint.py --stats_only` rewrites the passage files of the given `--data_dir`
  (re-derived `verified` flag); use a copy to keep release files byte-identical.
* `scripts/propagation_metrics.py --tables_only`, `scripts/rescore_results.py --tables_only` and
  `scripts/costs.py --tables_only` rewrite their tables from the CSVs without reading the transcripts.
