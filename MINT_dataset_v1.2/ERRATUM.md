# MINT v1.2 erratum and changes

* **WinoGrande gold labels.** MINT v1.1 intended 0-based answer indices but its
  conversion script (`scripts/extend_mint_dataset_v1_1.py::normalize_multiple_choice_answer`)
  mapped both 1-based values "1" and "2" to "1", so every WinoGrande item in v1.1
  carried `answer = "1"` (382/382). v1.2 recovers the gold option from MINT v1 (1-based)
  and stores an explicit 0-based `gold_index` / `gold_text` for every multiple-choice
  item. All experiments are scored against v1.2 labels.
* **Ethics labels** are unchanged (0 = Ethical, 1 = Unethical) and identical
  between v1, v1.1 and v1.2 (379/379).
* **CWQ** keeps the original answer object; `gold_answers` lists the canonical
  answer(s) followed by aliases, which are all accepted during scoring.
* **Per-passage release** `<dataset>_passages.jsonl`: one row per question x category
  with `n_tokens_ws`, `explicit_cue_lexical` (intended wrong option named verbatim;
  binary tasks only) and, after `scripts/validate_mint.py`, the judge flags
  `is_false`, `supports_target`, `explicit_cue`, `n_tokens`, `judge_model`,
  `judge_justification` and the derived `verified` flag (MINT-verified subset).

Item counts: [('winogrande', 382, 3438, {'0': 192, '1': 190}), ('ethics', 379, 3411, {'1': 184, '0': 195}), ('cwq', 381, 3429, {'None': 381})]
