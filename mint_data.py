#!/usr/bin/env python3
"""
mint_data.py - Canonical loading of MINT items and gold answers.

Why this module exists:
  * MINT v1 stores WinoGrande answers 1-based ("1"/"2"), the `answer` field of v1.1 is
    unreliable for WinoGrande (see MINT_dataset_v1.2/ERRATUM.md) and v1.2 stores an
    explicit `gold_index`; the gold option is resolved here once (explicit `gold_index`,
    else the MINT v1 lookup).
  * All experiment and analysis code must obtain gold answers through
    `resolve_gold()` / `load_mint_items()` so that indexing conventions are
    applied in exactly one place.

Item schema returned by load_mint_items():
  {
    "mint_id": str, "dataset": str ("winogrande"|"ethics"|"cwq"),
    "sentence": str, "options": list[str] | None, "is_multiple_choice": bool,
    "gold_index": int | None, "gold_text": str, "gold_aliases": list[str],
    "false_fact": str, "misinformation_by_strategy": dict[str, str],
    "irrelevant_true_information": str | None,
    "passages": dict[category -> passage]  # "false_fact" + 8 strategies
    "validation": dict[category -> flags]  (present after validate_mint.py)
    "raw": original item
  }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parent

DATASET_FILES = {
    "winogrande": "winogrande_misinformed.json",
    "ethics": "ethics_commonsense_misinformed.json",
    "cwq": "complex_web_questions_misinformed.json",
}
DATASET_ALIASES = {
    "winogrande": "winogrande",
    "wino": "winogrande",
    "ethics": "ethics",
    "ethics_commonsense": "ethics",
    "cwq": "cwq",
    "complex_web_questions": "cwq",
    "complex": "cwq",
    "web_questions": "cwq",
}
STRATEGIES = ["clickbait", "conspiracy", "framing", "hoax", "other", "propaganda", "rumor", "satire"]
CATEGORIES = ["false_fact"] + STRATEGIES

# Search order for item-level dataset files.
_DATA_DIR_CANDIDATES = [
    REPO_ROOT / "MINT_dataset_v1.2",
    REPO_ROOT / "data",
    REPO_ROOT / "MINT-dataset_v1.1",
]
_V1_DIR = REPO_ROOT / "MINT_dataset_v1"


def canonical_dataset(name: str) -> str:
    key = (name or "").strip().lower().replace("-", "_")
    if key not in DATASET_ALIASES:
        raise ValueError(f"Unknown dataset '{name}'. Known: {sorted(set(DATASET_ALIASES.values()))}")
    return DATASET_ALIASES[key]


def resolve_dataset_path(dataset: str, data_dir: Optional[str] = None) -> Path:
    """Locate the item-level JSON for a dataset.

    Search order: explicit data_dir, MINT_dataset_v1.2, data/, MINT-dataset_v1.1.
    """
    ds = canonical_dataset(dataset)
    fname = DATASET_FILES[ds]
    candidates: List[Path] = []
    if data_dir:
        candidates.append(Path(data_dir) / fname)
    candidates += [d / fname for d in _DATA_DIR_CANDIDATES]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(f"No dataset file for '{ds}' in {[str(c) for c in candidates]}")


def _load_json(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _v1_gold_lookup(dataset: str) -> Dict[str, int]:
    """sentence -> 0-based gold index from MINT v1 (1-based for winogrande)."""
    ds = canonical_dataset(dataset)
    p = _V1_DIR / DATASET_FILES[ds]
    out: Dict[str, int] = {}
    if not p.is_file():
        return out
    for it in _load_json(p):
        ans = it.get("answer")
        opts = it.get("options") or []
        if not isinstance(opts, list) or not opts:
            continue
        try:
            idx = int(str(ans).strip())
        except Exception:
            continue
        if ds == "winogrande":
            idx = idx - 1  # v1 is 1-based
        if 0 <= idx < len(opts):
            out[it["sentence"]] = idx
    return out


def resolve_gold(item: dict, dataset: str, v1_lookup: Optional[Dict[str, int]] = None) -> tuple:
    """Return (gold_index, gold_text, gold_aliases, is_multiple_choice).

    Precedence: explicit `gold_index` field (MINT_dataset_v1.2) > MINT v1 lookup (winogrande)
    > dataset-specific interpretation of `answer`.
    """
    ds = canonical_dataset(dataset)
    options = item.get("options")
    if ds == "cwq" or not (isinstance(options, list) and options and isinstance(options[0], str)):
        ans = item.get("answer")
        answers: List[str] = []
        aliases: List[str] = []
        if isinstance(ans, dict):
            answers = [str(a) for a in (ans.get("answer") or []) if isinstance(a, str)]
            for al in ans.get("aliases") or []:
                if isinstance(al, list):
                    aliases += [str(a) for a in al if isinstance(a, str)]
                elif isinstance(al, str):
                    aliases.append(al)
        elif isinstance(item.get("answers"), dict):
            answers = [str(a) for a in (item["answers"].get("answer") or []) if isinstance(a, str)]
        elif isinstance(ans, str):
            answers = [ans]
        gold_text = answers[0] if answers else ""
        return None, gold_text, [a for a in answers[1:] + aliases if a], False

    if "gold_index" in item and item["gold_index"] is not None:
        gi = int(item["gold_index"])
        return gi, str(options[gi]), [], True

    if ds == "winogrande":
        lookup = v1_lookup if v1_lookup is not None else _v1_gold_lookup(ds)
        if item.get("sentence") in lookup:
            gi = lookup[item["sentence"]]
            return gi, str(options[gi]), [], True
        # Fall back to the *1-based* convention of the original WinoGrande data.
        gi = int(str(item.get("answer")).strip()) - 1
        return gi, str(options[gi]), [], True

    # ethics (0-based: 0 = Ethical, 1 = Unethical)
    gi = int(str(item.get("answer")).strip())
    if not (0 <= gi < len(options)):
        raise ValueError(f"Ethics answer out of range: {item.get('answer')}")
    return gi, str(options[gi]), [], True


def load_mint_items(dataset: str, data_dir: Optional[str] = None, max_items: Optional[int] = None) -> List[dict]:
    ds = canonical_dataset(dataset)
    path = resolve_dataset_path(ds, data_dir)
    raw = _load_json(path)
    if max_items:
        raw = raw[:max_items]
    v1 = _v1_gold_lookup(ds) if ds == "winogrande" else {}
    items: List[dict] = []
    for i, it in enumerate(raw):
        gi, gtext, aliases, is_mc = resolve_gold(it, ds, v1)
        passages = {}
        ff = it.get("false_fact")
        if isinstance(ff, str) and ff.strip():
            passages["false_fact"] = ff
        strat = it.get("misinformation_by_strategy") or {}
        if isinstance(strat, dict):
            for k, v in strat.items():
                if isinstance(v, str) and v.strip():
                    passages[k] = v
        items.append(
            {
                "mint_id": it.get("MINT_ID") or f"MINT_ID_{ds.upper()}_{i + 1:06d}",
                "id": it.get("id"),
                "dataset": ds,
                "sentence": it.get("sentence") or it.get("question") or "",
                "options": it.get("options") if is_mc else None,
                "is_multiple_choice": is_mc,
                "gold_index": gi,
                "gold_text": gtext,
                "gold_aliases": aliases,
                "false_fact": ff,
                "misinformation_by_strategy": strat if isinstance(strat, dict) else {},
                "irrelevant_true_information": it.get("irrelevant_true_information"),
                "relevant_true_information": it.get("relevant_true_information"),
                "filler_information": it.get("filler_information"),
                "passages": passages,
                "validation": it.get("validation") or {},
                "raw": it,
            }
        )
    return items


def passage_is_verified(item: dict, category: str) -> Optional[bool]:
    """True/False if validation flags exist for the category, else None."""
    v = (item.get("validation") or {}).get(category)
    if not isinstance(v, dict):
        return None
    if "verified" in v:
        return bool(v["verified"])
    return None


def load_mint_id_subset(spec: Optional[str], dataset: str) -> Optional[set]:
    """Read a question subset (one MINT id per line, '#' comments allowed) for --mint_ids_file.

    `spec` may contain the placeholder {dataset}, e.g. configs/subsets/verified_{dataset}.txt
    (see scripts/make_verified_subsets.py). Returns None when no spec is given.
    """
    if not spec:
        return None
    path = Path(str(spec).format(dataset=canonical_dataset(dataset)))
    if not path.is_file():
        raise FileNotFoundError(f"--mint_ids_file: {path} not found")
    ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                ids.add(line)
    if not ids:
        raise ValueError(f"--mint_ids_file: {path} contains no ids")
    return ids


def filter_items_by_ids(items: List[dict], ids: Optional[set]) -> List[dict]:
    """Restrict items to a MINT-id subset (no-op when ids is None)."""
    if ids is None:
        return items
    return [it for it in items if it["mint_id"] in ids]


def filter_items_for_condition(items: List[dict], category: str, mint_filter: str = "all") -> List[dict]:
    """Apply the MINT-verified filter for a given passage category."""
    if mint_filter in ("all", "", None):
        return items
    if mint_filter != "verified":
        raise ValueError("mint_filter must be 'all' or 'verified'")
    kept = []
    for it in items:
        v = passage_is_verified(it, category)
        if v is None:
            raise RuntimeError(
                f"mint_filter=verified requested but item {it['mint_id']} has no validation flags for "
                f"'{category}'. Run scripts/validate_mint.py first."
            )
        if v:
            kept.append(it)
    return kept


def list_categories(items: Iterable[dict]) -> List[str]:
    cats: set = set()
    for it in items:
        cats.update(it.get("passages", {}).keys())
    ordered = [c for c in CATEGORIES if c in cats]
    return ordered + sorted(cats - set(ordered))


def format_mc_input(item: dict) -> str:
    """MALLM-style input string used by exp2/exp3 (kept identical for comparability)."""
    s = item["sentence"]
    opts = item.get("options") or []
    if len(opts) >= 2:
        s += "\nOptions:\nA) " + str(opts[0]) + "\nB) " + str(opts[1])
    return s


if __name__ == "__main__":
    import argparse
    from collections import Counter

    ap = argparse.ArgumentParser(description="Print MINT dataset summary with canonical gold answers.")
    ap.add_argument("--data_dir", default=None)
    args = ap.parse_args()
    for ds in DATASET_FILES:
        try:
            items = load_mint_items(ds, args.data_dir)
        except FileNotFoundError as e:
            print(f"{ds}: {e}")
            continue
        gi = Counter(str(it["gold_index"]) for it in items)
        print(
            f"{ds}: n={len(items)} file={resolve_dataset_path(ds, args.data_dir)} "
            f"gold_index dist={dict(gi)} categories={list_categories(items)}"
        )
