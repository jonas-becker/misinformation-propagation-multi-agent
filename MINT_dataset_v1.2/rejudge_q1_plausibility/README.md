# WinoGrande re-judge with the plausibility wording of Q1

Copies of `../winogrande_misinformed.json` and `../winogrande_passages.jsonl` whose judge flags come from a second
judging pass (both judges GPT-OSS-120B and GLM-4.7-Flash) in which Q1 asked whether the passage makes
the intended wrong option *more plausible* than the gold completion instead of whether it *contradicts* the gold
answer (`scripts/validate_mint.py --q1_criterion plausibility`, run by `slurm/chains/rejudge_winogrande.sh`). The
release files keep the contradiction wording; WinoGrande is verified on Q2 alone there (both judges Q1 and Q2:
27 questions; Q2 alone: 183). Under the plausibility wording 116 questions pass Q1 and Q2 and 182 pass Q2 alone, 148
of them in common with the release subset. Every flag records its wording in `q1_criterion`.

The runs of `slurm/chains/msc_matched_skepticism.sh` (sampling-matched mSC, source-skepticism debates without context and with
relevant-true passages) restricted their question sets with `--mint_ids_file configs/subsets/verified_{dataset}.txt`,
which were generated from these flags (`scripts/make_verified_subsets.py --data_dir MINT_dataset_v1.2/rejudge_q1_plausibility`).
