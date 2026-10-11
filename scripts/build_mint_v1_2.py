#!/usr/bin/env python3
"""
build_mint_v1_2.py - Build the MINT v1.2 release files.

Steps
  1. Load MINT v1.1 items and recover canonical gold answers (mint_data.resolve_gold):
     WinoGrande labels in v1.1 were corrupted (all "1"); v1.2 stores an explicit,
     0-based `gold_index` and `gold_text` for every multiple-choice item, and the
     CWQ answer list + aliases as `gold_answers`.
  2. Write item-level files MINT_dataset_v1.2/<dataset>_misinformed.json (same schema as
     v1.1 plus gold_index/gold_text/gold_answers and a per-category `validation`
     dict, empty until scripts/validate_mint.py has run) - these are what the
     experiment runners load.
  3. Write per-passage release files MINT_dataset_v1.2/<dataset>_passages.jsonl with one
     row per question x category (false_fact + 8 rhetorical categories) including
     n_tokens and a lexical explicit-cue flag (intended wrong answer stated verbatim).
  4. Write MINT_dataset_v1.2/ERRATUM.md documenting the label fix.

Run scripts/validate_mint.py afterwards to add judge flags; it updates both files.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mint_data import CATEGORIES, DATASET_FILES, load_mint_items  # noqa: E402

OUT_DIR = REPO_ROOT / "MINT_dataset_v1.2"
_TOKEN_RE = re.compile(r"\S+")


def n_tokens(text: str) -> int:
    """Whitespace tokens (same convention as scripts/extend_mint_dataset_v1_1.py); a tiktoken
    count is added by validate_mint.py when the library is available."""
    return len(_TOKEN_RE.findall(text or ""))


def intended_wrong_answer(item: dict) -> str | None:
    """For binary tasks the intended wrong answer is the non-gold option."""
    if item["is_multiple_choice"] and item["options"] and len(item["options"]) == 2 and item["gold_index"] is not None:
        return str(item["options"][1 - item["gold_index"]])
    return None


def explicit_cue(passage: str, wrong_answer: str | None) -> bool | None:
    if not wrong_answer:
        return None
    return re.search(r"(?<!\w)" + re.escape(wrong_answer.lower()) + r"(?!\w)", (passage or "").lower()) is not None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source_dir", default=str(REPO_ROOT / "MINT-dataset_v1.1"))
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    stats_rows = []
    for ds, fname in DATASET_FILES.items():
        items = load_mint_items(ds, args.source_dir)
        item_level = []
        passages = []
        for it in items:
            raw = dict(it["raw"])
            raw["MINT_ID"] = it["mint_id"]
            raw["gold_index"] = it["gold_index"]
            raw["gold_text"] = it["gold_text"]
            raw["gold_answers"] = [it["gold_text"]] + list(it["gold_aliases"]) if not it["is_multiple_choice"] else None
            if it["is_multiple_choice"]:
                # store the *0-based* gold index in `answer` as intended for v1.1
                raw["answer"] = str(it["gold_index"])
            raw.setdefault("validation", {})
            item_level.append(raw)
            wrong = intended_wrong_answer(it)
            for cat in CATEGORIES:
                p = it["passages"].get(cat)
                if not p:
                    continue
                passages.append({
                    "passage_id": f"{it['mint_id']}::{cat}",
                    "mint_id": it["mint_id"],
                    "dataset": ds,
                    "category": cat,
                    "question": it["sentence"],
                    "options": it["options"],
                    "gold_index": it["gold_index"],
                    "gold_text": it["gold_text"],
                    "gold_aliases": it["gold_aliases"],
                    "intended_wrong_answer": wrong,
                    "passage": p,
                    "n_tokens_ws": n_tokens(p),
                    "explicit_cue_lexical": explicit_cue(p, wrong),
                    "generator_model": "meta-llama/Llama-3.3-70B-Instruct",
                    "validation": (it.get("validation") or {}).get(cat, {}),
                })
        with open(out / fname, "w", encoding="utf-8") as f:
            json.dump(item_level, f, indent=1, ensure_ascii=False)
        with open(out / f"{ds}_passages.jsonl", "w", encoding="utf-8") as f:
            for row in passages:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        gi = Counter(str(it["gold_index"]) for it in items)
        stats_rows.append((ds, len(items), len(passages), dict(gi)))
        print(f"{ds}: {len(items)} items, {len(passages)} passages, gold_index dist {dict(gi)}")

    erratum = f"""# MINT v1.2 erratum and changes

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

Item counts: {stats_rows}
"""
    (out / "ERRATUM.md").write_text(erratum, encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
