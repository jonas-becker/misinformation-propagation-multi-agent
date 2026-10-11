# Reproduce every table and figure of the paper from the stored experiment outputs (no GPU needed).
# `make all` runs the full offline pipeline; the individual targets are listed below.
PY ?= python
BOOT ?= 10000
RELEVANT_TRUE_DIR ?= controls/relevant_true

.PHONY: all rescore fig_exposure fig_propagation fig_control fig_category fig_decomposition fig_mitigation fig_composition tables validation details mint subsets costs exp3_fill_plan exp3_fill_merge clean

all: rescore fig_exposure fig_propagation fig_control fig_category fig_decomposition fig_mitigation fig_composition tables validation costs details

rescore:            ## re-parse all stored outputs under out/ with the unified extractor -> out/results/rescored/*.csv
	$(PY) scripts/rescore_results.py

fig_exposure:       ## exposure curves, debate effect, matched-sampling check, teaser figure
	$(PY) scripts/exposure_curve.py --n_boot $(BOOT)
	-$(PY) scripts/msc_matched_sensitivity.py --n_boot $(BOOT)
	-$(PY) scripts/teaser_figure.py

fig_propagation:    ## adoption rates, transitions, share of the wrong answer per turn, counterfactual message ablation
	$(PY) scripts/propagation_metrics.py
	$(PY) scripts/adoption_figure.py
	-$(PY) exp_message_ablation.py --analyze_only --n_boot $(BOOT) --raw_out results/ablation_propagation_raw.jsonl

fig_control:        ## relevance x truth controls, length matching, debates with relevant-true passages
	$(PY) scripts/control_2x2.py --n_boot $(BOOT) --relevant_true_dir $(RELEVANT_TRUE_DIR)
	-$(PY) scripts/relevant_true_debates.py --n_boot $(BOOT)

fig_category:       ## accuracy by passage category (single agent and debate)
	$(PY) scripts/category_figure.py

fig_decomposition:  ## ensembling, interaction and misinformation cost
	$(PY) scripts/decomposition.py --n_boot $(BOOT)

fig_mitigation:     ## source-skepticism instruction (false, relevant-true and no context) and verifier agent vs. controls
	$(PY) scripts/mitigation_effects.py --n_boot $(BOOT)

fig_composition:    ## five-agent composition sweep and self-correction
	$(PY) scripts/composition_sweep_figure.py --n_boot $(BOOT)

tables:             ## generator x consumer matrix, category regression
	-$(PY) scripts/generator_consumer_matrix.py --n_boot $(BOOT)
	-$(PY) scripts/category_regression.py

validation:         ## judge-validation tables (MINT v1.2, cross-generator sets, relevant-true controls), verified-subset checks, human agreement
	-$(PY) scripts/validate_mint.py --stats_only
	-$(PY) scripts/validate_mint.py --stats_only --data_dir MINT_dataset_v1.2/generators/glm --out_tag gen_glm
	-$(PY) scripts/validate_mint.py --stats_only --data_dir MINT_dataset_v1.2/generators/gptoss --out_tag gen_gptoss
	-$(PY) scripts/validate_mint.py --stats_only --mode relevant_true --passages_suffix _relevant_true_passages --data_dir $(RELEVANT_TRUE_DIR)
	-$(PY) scripts/verified_subset_robustness.py --n_boot $(BOOT)
	-$(PY) scripts/evaluate_annotation.py --json-out MINT-dataset_v1.1/annotation_stats.json

details:            ## appendix tables (model settings, experimental details, run inventory) -> out/appendix/, from configs/experiment_config.json
	$(PY) scripts/make_details_table.py --full

mint:               ## rebuild the MINT v1.2 release files (labels, per-passage jsonl, filler)
	$(PY) scripts/build_mint_v1_2.py
	$(PY) scripts/generate_filler.py

subsets:            ## judge-verified question ids (configs/subsets/verified_<dataset>.txt) for --mint_ids_file runs; WinoGrande from the re-judged flags, as the runs used
	$(PY) scripts/make_verified_subsets.py --datasets cwq,ethics
	$(PY) scripts/make_verified_subsets.py --datasets winogrande --data_dir MINT_dataset_v1.2/rejudge_q1_plausibility

EXP3_FILL_MODEL ?= meta-llama/Llama-3.3-70B-Instruct
exp3_fill_plan:     ## exp3: questions without a debate per mix (raw out/<model>/exp3_results_winogrande.json, or EXP3_FILL_CSV=<rescored mad_final.csv>)
	$(PY) exp3_fill_missing.py plan --model_name $(EXP3_FILL_MODEL) $(if $(EXP3_FILL_CSV),--from_rescored $(EXP3_FILL_CSV),)

exp3_fill_merge:    ## exp3: de-duplicate the raw sweep file and insert the fill-in debates (after slurm/chains/exp3_fill.sh); then `make rescore`
	$(PY) exp3_fill_missing.py merge --model_name $(EXP3_FILL_MODEL)

costs:              ## token / GPU-hour estimates (pass TOKENS_PER_GPU_HOUR=...)
	$(PY) scripts/costs.py $(if $(TOKENS_PER_GPU_HOUR),--tokens_per_gpu_hour $(TOKENS_PER_GPU_HOUR),)

clean:
	rm -rf out/results/rescored out/results/*.csv out/tables/*.tex out/appendix/*.csv out/appendix/*_details*.tex out/appendix/model_settings.tex out/appendix/run_inventory.tex out/figures/fig_*
