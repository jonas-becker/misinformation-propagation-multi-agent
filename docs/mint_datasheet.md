# MINT v1.2 datasheet

**Name.** MINT (Misinformation with INTent) v1.2 - passages designed to mislead an answerer on three QA benchmarks.

**Composition.** 1,142 questions (WinoGrande 382, ETHICS commonsense 379, ComplexWebQuestions 381; uniform random samples sized by the finite-population 95%/5% formula) x 9 passage categories = 10,278 passages. Categories: plain *false fact* and eight rhetorical categories (clickbait, hoax, rumor, satire, propaganda, framing, conspiracy, unconstrained). Every passage row carries the question, options, 0-based `gold_index`/`gold_text` (CWQ: canonical answers + aliases), the intended wrong answer (binary tasks), token length, a lexical explicit-cue flag, and the judge flags below. The datasheet dimension table (intent to deceive / epistemic status / rhetorical style) is `docs/taxonomy_dimensions.csv`; satire and unconstrained are style controls, not intents.

**Generation.** Llama-3.3-70B-Instruct (HF transformers, temperature 0.7, top-p 0.9, repetition penalty 1.1). Stage 1 produces one believable false fact per question given the question and the gold answer; stage 2 rewrites it in each rhetorical category with the category description from Zhou & Zafarani-style taxonomies (prompts: `mint_prompts.py`, identical to `download_datasets.py`). Sentences announcing the text as misinformation are removed. Cross-generator subsets (GLM-4.7-Flash, GPT-OSS-120B) use the same pipeline (`scripts/generate_mint_passages.py --mode misinformation`).

**Controls.** `irrelevant_true_information`: length-matched Wikipedia passage about another topic (v1.1). `relevant_true_by_strategy`: passages supporting the gold answer in the same nine styles (`--mode relevant_true`, length-matched per category). `filler_information`: content-free, entity-free, length-matched text (`scripts/generate_filler.py`).

**Validation.** Two LLM judges from other model families (GPT-OSS-120B, GLM-4.7-Flash) answer per passage: Q1 contradicts the gold answer? Q2 supports the intended wrong answer? Q3 states it verbatim? (`scripts/validate_mint.py`). *MINT-verified* = both judges answer yes to Q1 and Q2. Filter rates per dataset and category: `results/mint_validation_stats.csv`; inter-judge agreement: `results/mint_judge_agreement.csv`. Human validation: stratified sample of 40 passages per dataset x category (1,080 items), three annotators, codebook `docs/annotation_codebook.md`; per-criterion Krippendorff's alpha and judge precision/recall: `results/human_validation_agreement.csv`, `results/judge_vs_human.csv`.

**Versions.** v1.2 corrects the WinoGrande `answer` field of v1.1 and stores explicit gold indices for every multiple-choice item (see `MINT_dataset_v1.2/ERRATUM.md`); all experiments are scored against v1.2 labels.

**Intended use / misuse.** Research on robustness of LLM systems to misleading context. Passages are synthetic and false by construction; they must not be redistributed as factual text. **License:** TODO (dataset license; underlying benchmarks: WinoGrande CC-BY, ETHICS MIT, ComplexWebQuestions - check terms).

