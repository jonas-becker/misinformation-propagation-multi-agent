#!/usr/bin/env python3
"""Evaluate MINT Label Studio annotation export: coverage stats and inter-annotator agreement.

Only tasks with at least 3 non-cancelled annotations are included (see --min-annotations).

Agreement metrics (standard in NLP / HCI annotation papers):
  - Percent agreement (unanimous and majority-vote)
  - Fleiss' kappa overall and per MINT subset (dataset) for each dimension:
    faithful_to_false_fact, faithful_to_claim_type
  - Krippendorff's alpha (nominal)
  - Cohen's kappa (mean pairwise)

Usage (from repository root):
  python scripts/evaluate_annotation.py
  python scripts/evaluate_annotation.py --json-out MINT-dataset_v1.1/annotation_stats.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


def cohen_kappa_score(y1: Sequence[str], y2: Sequence[str]) -> float:
    """Unweighted Cohen's kappa of two label sequences (nan when chance agreement is 1, as in scikit-learn)."""
    n = len(y1)
    p_o = sum(1 for a, b in zip(y1, y2) if a == b) / n
    c1, c2 = Counter(y1), Counter(y2)
    p_e = sum(c1[lab] * c2[lab] for lab in set(c1) | set(c2)) / (n * n)
    if math.isclose(p_e, 1.0):
        return math.nan
    return (p_o - p_e) / (1.0 - p_e)

REPO_ROOT = Path(__file__).resolve().parents[1]
MINT_DATASET_DIR = REPO_ROOT / "MINT-dataset_v1.1"
DEFAULT_EXPORT = MINT_DATASET_DIR / "export_labelstudio.json"

LABEL_FIELDS = ("faithful_to_false_fact", "faithful_to_claim_type")
CHOICE_ORDER = ("Yes", "No", "Unclear")
FAITHFUL_LABEL = "Yes"
DEFAULT_MIN_ANNOTATIONS = 3


def resolve_repo_path(path: str | Path) -> Path:
    """Resolve relative paths against the repository root (where this script lives)."""
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (REPO_ROOT / p).resolve()


def annotation_count(raw_task: Mapping) -> int:
    return len([a for a in (raw_task.get("annotations") or []) if not a.get("was_cancelled")])


def filter_tasks_by_min_annotations(
    raw_tasks: Sequence[Mapping],
    min_annotations: int,
) -> Tuple[List[Mapping], Dict[str, int]]:
    kept = [t for t in raw_tasks if annotation_count(t) >= min_annotations]
    return kept, {
        "n_tasks_total": len(raw_tasks),
        "n_tasks_included": len(kept),
        "n_tasks_excluded": len(raw_tasks) - len(kept),
        "min_annotations_required": min_annotations,
    }


@dataclass(frozen=True)
class TaskLabels:
    task_id: int
    mint_id: str
    subset: str
    strategy: str
    claim_type: str
    labels_by_field: Dict[str, Dict[int, str]]  # field -> {annotator_id: choice}


def _labels_from_annotation(annotation: Mapping) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if annotation.get("was_cancelled"):
        return out
    for item in annotation.get("result") or []:
        if not isinstance(item, dict) or item.get("type") != "choices":
            continue
        field = item.get("from_name")
        if field not in LABEL_FIELDS:
            continue
        value = item.get("value")
        if not isinstance(value, dict):
            continue
        choices = value.get("choices")
        if isinstance(choices, list) and choices:
            out[str(field)] = str(choices[0])
    return out


def parse_export(tasks: Sequence[Mapping]) -> List[TaskLabels]:
    parsed: List[TaskLabels] = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        data = task.get("data") or {}
        by_field: Dict[str, Dict[int, str]] = {f: {} for f in LABEL_FIELDS}
        for ann in task.get("annotations") or []:
            if not isinstance(ann, dict) or ann.get("was_cancelled"):
                continue
            annotator = ann.get("completed_by")
            if not isinstance(annotator, int):
                continue
            field_labels = _labels_from_annotation(ann)
            for field, choice in field_labels.items():
                by_field[field][annotator] = choice
        parsed.append(
            TaskLabels(
                task_id=int(task.get("id", -1)),
                mint_id=str(data.get("MINT_ID", "")),
                subset=str(data.get("subset", "")),
                strategy=str(data.get("strategy") or "false_fact"),
                claim_type=str(data.get("claim_type", "")),
                labels_by_field=by_field,
            )
        )
    return parsed


def _label_matrix(
    tasks: Sequence[TaskLabels],
    field: str,
    *,
    min_raters: int = DEFAULT_MIN_ANNOTATIONS,
) -> Tuple[List[List[Optional[str]]], List[TaskLabels]]:
    """Rows = items; columns = raters (sorted by annotator id)."""
    matrix: List[List[Optional[str]]] = []
    kept: List[TaskLabels] = []
    for task in tasks:
        ratings = task.labels_by_field.get(field, {})
        if len(ratings) < min_raters:
            continue
        row = [ratings[a] for a in sorted(ratings)]
        matrix.append(row)
        kept.append(task)
    return matrix, kept


def strict_majority_label(labels: Sequence[str]) -> Optional[str]:
    """Return the label with a strict majority (> half of votes), or None if tied."""
    if not labels:
        return None
    counts = Counter(labels)
    best_label, best_count = counts.most_common(1)[0]
    if best_count > len(labels) / 2:
        return best_label
    return None


def majority_vote_faithful_stats(
    tasks: Sequence[TaskLabels],
    *,
    min_raters: int = DEFAULT_MIN_ANNOTATIONS,
) -> Dict[str, Dict[str, object]]:
    """Per-sample strict majority vote; 'faithful' = majority label is Yes."""
    out: Dict[str, Dict[str, object]] = {}
    for field in LABEL_FIELDS:
        n_samples = 0
        n_majority_yes = n_majority_no = n_majority_unclear = n_no_majority = 0
        for task in tasks:
            ratings = task.labels_by_field.get(field, {})
            if len(ratings) < min_raters:
                continue
            n_samples += 1
            majority = strict_majority_label(list(ratings.values()))
            if majority is None:
                n_no_majority += 1
            elif majority == FAITHFUL_LABEL:
                n_majority_yes += 1
            elif majority == "No":
                n_majority_no += 1
            elif majority == "Unclear":
                n_majority_unclear += 1

        out[field] = {
            "n_samples": n_samples,
            "n_majority_yes": n_majority_yes,
            "n_majority_no": n_majority_no,
            "n_majority_unclear": n_majority_unclear,
            "n_no_majority": n_no_majority,
            "percent_faithful_majority_yes": (
                n_majority_yes / n_samples if n_samples else math.nan
            ),
            "percent_not_faithful_majority_no": (
                n_majority_no / n_samples if n_samples else math.nan
            ),
            "percent_unclear_majority": (
                n_majority_unclear / n_samples if n_samples else math.nan
            ),
            "percent_no_majority": (
                n_no_majority / n_samples if n_samples else math.nan
            ),
        }
    return out


def percent_agreement(matrix: Sequence[Sequence[Optional[str]]]) -> Dict[str, float]:
    if not matrix:
        return {"unanimous": math.nan, "majority": math.nan, "n_items": 0.0}
    unanimous = majority = 0
    for row in matrix:
        labels = [x for x in row if x is not None]
        if not labels:
            continue
        counts = Counter(labels)
        top = counts.most_common(1)[0][1]
        if top == len(labels):
            unanimous += 1
        if top > len(labels) / 2:
            majority += 1
    n = len(matrix)
    return {
        "unanimous": unanimous / n,
        "majority": majority / n,
        "n_items": float(n),
    }


def fleiss_kappa(matrix: Sequence[Sequence[Optional[str]]], categories: Sequence[str]) -> float:
    """Fleiss' kappa for a fixed number of raters per item (no missing labels in a row)."""
    if not matrix:
        return math.nan
    n_raters = len(matrix[0])
    if n_raters < 2 or any(len(row) != n_raters for row in matrix):
        return math.nan
    cat_index = {c: i for i, c in enumerate(categories)}
    k = len(categories)
    n_items = len(matrix)

    counts_per_item = []
    for row in matrix:
        if any(v is None for v in row):
            return math.nan
        counts = [0] * k
        for label in row:
            if label not in cat_index:
                return math.nan
            counts[cat_index[label]] += 1
        counts_per_item.append(counts)

    p_per_item = []
    for counts in counts_per_item:
        n = sum(counts)
        if n < 2:
            return math.nan
        pe_i = sum(c * c for c in counts) - n
        p_per_item.append(pe_i / (n * (n - 1)))
    p_bar = sum(p_per_item) / n_items

    total_assignments = n_items * n_raters
    p_j = [0.0] * k
    for counts in counts_per_item:
        for j, c in enumerate(counts):
            p_j[j] += c
    p_j = [x / total_assignments for x in p_j]
    p_e = sum(x * x for x in p_j)
    if math.isclose(1.0 - p_e, 0.0):
        return 1.0 if math.isclose(p_bar, 1.0) else 0.0
    return (p_bar - p_e) / (1.0 - p_e)


def krippendorff_alpha_nominal(matrix: Sequence[Sequence[Optional[str]]]) -> float:
    """Krippendorff's alpha for nominal data (variable raters, missing allowed)."""
    values = sorted({v for row in matrix for v in row if v is not None})
    if len(values) < 2:
        return math.nan
    val_to_i = {v: i for i, v in enumerate(values)}
    n_v = len(values)

    coincidence = [[0.0] * n_v for _ in range(n_v)]
    n_pairs = 0.0

    for row in matrix:
        present = [val_to_i[v] for v in row if v is not None]
        m = len(present)
        if m < 2:
            continue
        for i in range(m):
            for j in range(i + 1, m):
                a, b = present[i], present[j]
                coincidence[a][b] += 1.0
                coincidence[b][a] += 1.0
                n_pairs += 1.0

    if n_pairs == 0:
        return math.nan

    for i in range(n_v):
        coincidence[i][i] *= 2.0

    do = 0.0
    for i in range(n_v):
        for j in range(n_v):
            if i == j:
                continue
            do += coincidence[i][j]
    do /= 2.0 * n_pairs

    n_c = [sum(coincidence[i][j] for j in range(n_v)) for i in range(n_v)]
    total = sum(n_c)
    if total == 0:
        return math.nan
    de = 0.0
    for i in range(n_v):
        for j in range(n_v):
            if i != j:
                de += n_c[i] * n_c[j]
    de /= total * (total - 1.0)

    if math.isclose(de, 0.0):
        return 1.0 if math.isclose(do, 0.0) else 0.0
    return 1.0 - do / de


def mean_pairwise_cohen_kappa(matrix: Sequence[Sequence[Optional[str]]]) -> float:
    """Average Cohen's kappa over all annotator pairs (items with both labels)."""
    if not matrix or len(matrix[0]) < 2:
        return math.nan
    n_raters = len(matrix[0])
    scores: List[float] = []
    for i in range(n_raters):
        for j in range(i + 1, n_raters):
            y1, y2 = [], []
            for row in matrix:
                if row[i] is None or row[j] is None:
                    continue
                y1.append(row[i])
                y2.append(row[j])
            if len(y1) < 2:
                continue
            scores.append(float(cohen_kappa_score(y1, y2)))
    return statistics.mean(scores) if scores else math.nan


def agreement_for_field(
    tasks: Sequence[TaskLabels],
    field: str,
    categories: Sequence[str] = CHOICE_ORDER,
    *,
    min_raters: int = DEFAULT_MIN_ANNOTATIONS,
) -> Dict[str, object]:
    matrix, _ = _label_matrix(tasks, field, min_raters=min_raters)

    return {
        "n_items": len(matrix),
        "min_raters_per_field": min_raters,
        "percent_agreement": percent_agreement(matrix),
        "cohens_kappa_mean_pairwise": mean_pairwise_cohen_kappa(matrix),
        "fleiss_kappa": fleiss_kappa(matrix, categories),
        "krippendorff_alpha_nominal": krippendorff_alpha_nominal(matrix),
    }


def descriptive_stats(
    tasks: Sequence[TaskLabels],
    raw_tasks: Sequence[Mapping],
    filter_summary: Mapping[str, int],
) -> Dict[str, object]:
    annotator_counts: Counter = Counter()
    lead_times: List[float] = []
    label_dist: Dict[str, Counter] = {f: Counter() for f in LABEL_FIELDS}
    raters_per_task = Counter()
    by_subset = Counter()
    by_strategy = Counter()
    ls_agreements: List[float] = []

    for raw in raw_tasks:
        n = annotation_count(raw)
        raters_per_task[n] += 1
        if raw.get("agreement") is not None:
            ls_agreements.append(float(raw["agreement"]))
        for ann in raw.get("annotations") or []:
            if ann.get("was_cancelled"):
                continue
            if isinstance(ann.get("completed_by"), int):
                annotator_counts[ann["completed_by"]] += 1
            lt = ann.get("lead_time")
            if isinstance(lt, (int, float)) and lt > 0:
                lead_times.append(float(lt))

    for task in tasks:
        by_subset[task.subset] += 1
        by_strategy[task.strategy] += 1
        for field in LABEL_FIELDS:
            for choice in task.labels_by_field.get(field, {}).values():
                label_dist[field][choice] += 1

    states = Counter(str(t.get("state", "")) for t in raw_tasks)

    lead_summary = {}
    if lead_times:
        lead_summary = {
            "n_annotations": len(lead_times),
            "mean_seconds": statistics.mean(lead_times),
            "median_seconds": statistics.median(lead_times),
            "total_hours": sum(lead_times) / 3600.0,
        }

    return {
        "filter": dict(filter_summary),
        "n_tasks": len(tasks),
        "unique_mint_ids": len({t.mint_id for t in tasks}),
        "task_state_counts": dict(states),
        "annotators_per_task": dict(sorted(raters_per_task.items())),
        "annotator_submission_counts": dict(sorted(annotator_counts.items())),
        "items_per_subset": dict(sorted(by_subset.items())),
        "items_per_strategy": dict(sorted(by_strategy.items())),
        "label_distribution": {f: dict(label_dist[f]) for f in LABEL_FIELDS},
        "label_studio_reported_agreement_percent": {
            "mean": statistics.mean(ls_agreements) if ls_agreements else None,
            "min": min(ls_agreements) if ls_agreements else None,
            "max": max(ls_agreements) if ls_agreements else None,
            "n_tasks_with_score": len(ls_agreements),
        },
        "lead_time": lead_summary,
    }


def fleiss_kappa_by_group(
    tasks: Sequence[TaskLabels],
    field: str,
    key_fn,
    *,
    min_raters: int = DEFAULT_MIN_ANNOTATIONS,
    min_items: int = 1,
) -> Dict[str, Dict[str, object]]:
    """Fleiss' kappa (and n) per group key, e.g. MINT subset / dataset."""
    groups: Dict[str, List[TaskLabels]] = defaultdict(list)
    for task in tasks:
        groups[key_fn(task)].append(task)
    out: Dict[str, Dict[str, object]] = {}
    for name, group in sorted(groups.items()):
        matrix, _ = _label_matrix(group, field, min_raters=min_raters)
        if len(matrix) < min_items:
            continue
        out[name] = {
            "n_items": len(group),
            "n_items_with_labels": len(matrix),
            "fleiss_kappa": fleiss_kappa(matrix, CHOICE_ORDER),
        }
    return out


def fleiss_kappa_by_dataset(
    tasks: Sequence[TaskLabels],
    *,
    min_raters: int = DEFAULT_MIN_ANNOTATIONS,
) -> Dict[str, Dict[str, object]]:
    """Fleiss' kappa per MINT subset (dataset) for each assessed dimension."""
    return {
        field: fleiss_kappa_by_group(
            tasks, field, lambda t: t.subset, min_raters=min_raters
        )
        for field in LABEL_FIELDS
    }


def stratified_agreement(
    tasks: Sequence[TaskLabels],
    field: str,
    key_fn,
    *,
    min_raters: int = DEFAULT_MIN_ANNOTATIONS,
) -> Dict[str, Dict[str, object]]:
    groups: Dict[str, List[TaskLabels]] = defaultdict(list)
    for task in tasks:
        groups[key_fn(task)].append(task)
    out: Dict[str, Dict[str, object]] = {}
    for name, group in sorted(groups.items()):
        if len(group) < 5:
            continue
        matrix, _ = _label_matrix(group, field, min_raters=min_raters)
        if len(matrix) < 5:
            continue
        out[name] = {
            "n_items": len(group),
            "n_items_with_labels": len(matrix),
            "percent_agreement_unanimous": percent_agreement(matrix)["unanimous"],
            "fleiss_kappa": fleiss_kappa(matrix, CHOICE_ORDER),
            "krippendorff_alpha_nominal": krippendorff_alpha_nominal(matrix),
        }
    return out


def build_report(
    tasks: Sequence[TaskLabels],
    raw_tasks: Sequence[Mapping],
    filter_summary: Mapping[str, int],
    *,
    min_annotations: int = DEFAULT_MIN_ANNOTATIONS,
) -> Dict[str, object]:
    agreement = {
        field: {
            "overall": agreement_for_field(tasks, field, min_raters=min_annotations),
            "by_subset": stratified_agreement(tasks, field, lambda t: t.subset, min_raters=min_annotations),
            "by_strategy": stratified_agreement(tasks, field, lambda t: t.strategy, min_raters=min_annotations),
        }
        for field in LABEL_FIELDS
    }
    fleiss_by_dataset = fleiss_kappa_by_dataset(tasks, min_raters=min_annotations)
    return {
        "descriptive": descriptive_stats(tasks, raw_tasks, filter_summary),
        "majority_vote_faithful": majority_vote_faithful_stats(
            tasks, min_raters=min_annotations
        ),
        "fleiss_kappa_by_dataset": fleiss_by_dataset,
        "inter_annotator_agreement": agreement,
        "metrics_documentation": {
            "percent_faithful_majority_yes": "Share of samples where a strict majority of raters "
            f"selected '{FAITHFUL_LABEL}' (faithful).",
            "percent_agreement_unanimous": "Fraction of items where all raters chose the same label.",
            "percent_agreement_majority": "Fraction of items where a strict majority shares one label.",
            "fleiss_kappa": "Multi-rater chance-corrected agreement on items with >= min annotations.",
            "krippendorff_alpha_nominal": "Chance-corrected nominal agreement on the same cohort.",
            "cohens_kappa_mean_pairwise": "Mean of pairwise Cohen's kappa across rater pairs.",
            "interpretation": "Landis & Koch (1977): κ/α < 0 poor, 0.01–0.20 slight, 0.21–0.40 fair, "
            "0.41–0.60 moderate, 0.61–0.80 substantial, 0.81–1.00 almost perfect.",
        },
    }


def _fmt_pct(x: float) -> str:
    if math.isnan(x):
        return "n/a"
    return f"{100.0 * x:.1f}%"


def _fmt_kappa(x: float) -> str:
    if math.isnan(x):
        return "n/a"
    return f"{x:.3f}"


def print_report(report: Dict[str, object]) -> None:
    desc = report["descriptive"]
    filt = desc.get("filter") or {}
    print("=" * 72)
    print("MINT annotation evaluation")
    print("=" * 72)
    if filt:
        print(
            f"Cohort: tasks with >={filt.get('min_annotations_required', 3)} annotations "
            f"({filt.get('n_tasks_included', '?')}/{filt.get('n_tasks_total', '?')}; "
            f"excluded {filt.get('n_tasks_excluded', '?')})"
        )
    print(f"Tasks (claim rows): {desc['n_tasks']}")
    print(f"Unique MINT items: {desc['unique_mint_ids']}")
    print(f"Task states: {desc['task_state_counts']}")
    print(f"Annotators per task: {desc['annotators_per_task']}")
    print(f"Annotator submissions: {desc['annotator_submission_counts']}")
    print(f"Items per subset: {desc['items_per_subset']}")
    print(f"Items per strategy: {desc['items_per_strategy']}")
    print("\nLabel distributions:")
    for field, dist in desc["label_distribution"].items():
        print(f"  {field}: {dist}")

    mv = report.get("majority_vote_faithful") or {}
    if mv:
        print("\nFaithful (majority vote = Yes, per sample):")
        for field in LABEL_FIELDS:
            stats = mv.get(field) or {}
            n = int(stats.get("n_samples", 0))
            n_yes = int(stats.get("n_majority_yes", 0))
            pct = stats.get("percent_faithful_majority_yes", math.nan)
            print(f"  {field}: {_fmt_pct(pct)} ({n_yes}/{n})")
            n_no_majority = int(stats.get("n_no_majority", 0))
            if n_no_majority:
                print(f"    (no strict majority on {n_no_majority} samples)")

    lt = desc.get("lead_time") or {}
    if lt:
        print(
            f"\nLead time: n={lt['n_annotations']}, "
            f"median={lt['median_seconds']:.1f}s, mean={lt['mean_seconds']:.1f}s, "
            f"total~{lt['total_hours']:.1f}h"
        )

    ls = desc["label_studio_reported_agreement_percent"]
    if ls.get("n_tasks_with_score"):
        print(
            f"\nLabel Studio agreement field (reference): mean={ls['mean']:.2f}%, "
            f"range=[{ls['min']:.2f}, {ls['max']:.2f}]%"
        )

    print("\n" + "-" * 72)
    print("Inter-annotator agreement (for paper)")
    print("-" * 72)
    iaa = report["inter_annotator_agreement"]
    for field in LABEL_FIELDS:
        overall = iaa[field]["overall"]
        pa = overall["percent_agreement"]
        print(f"\n[{field}] (n={int(overall['n_items'])})")
        print(
            f"  unanimous={_fmt_pct(pa['unanimous'])}, "
            f"majority={_fmt_pct(pa['majority'])}, "
            f"Fleiss kappa={_fmt_kappa(overall['fleiss_kappa'])}, "
            f"mean pairwise Cohen kappa={_fmt_kappa(overall['cohens_kappa_mean_pairwise'])}, "
            f"Krippendorff alpha={_fmt_kappa(overall['krippendorff_alpha_nominal'])}"
        )

    fleiss_by_dataset = report.get("fleiss_kappa_by_dataset") or {}
    if fleiss_by_dataset:
        print("\n" + "-" * 72)
        print("Fleiss' kappa by dataset (MINT subset)")
        print("-" * 72)
        subsets = sorted(
            {
                subset
                for per_field in fleiss_by_dataset.values()
                for subset in per_field
            }
        )
        col_w = max(len(f) for f in LABEL_FIELDS) + 2
        header = f"{'dataset':<28}" + "".join(f"{f:>{col_w}}" for f in LABEL_FIELDS)
        print(header)
        for subset in subsets:
            cells = []
            for field in LABEL_FIELDS:
                stats = (fleiss_by_dataset.get(field) or {}).get(subset)
                if stats is None:
                    cells.append(f"{'n/a':>{col_w}}")
                else:
                    k = stats["fleiss_kappa"]
                    n = stats["n_items_with_labels"]
                    cells.append(f"{_fmt_kappa(k)} (n={n})".rjust(col_w))
            print(f"{subset:<28}" + "".join(cells))

    print("\nInterpretation (Landis & Koch): see metrics_documentation in JSON output.")
    print("=" * 72)


def main() -> None:
    default_export = DEFAULT_EXPORT.relative_to(REPO_ROOT).as_posix()
    parser = argparse.ArgumentParser(description="Evaluate MINT Label Studio annotations")
    parser.add_argument(
        "--export",
        type=str,
        default=default_export,
        help=f"Label Studio JSON export relative to repo root (default: {default_export})",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Optional path to write full report as JSON (relative to repo root)",
    )
    parser.add_argument(
        "--min-annotations",
        type=int,
        default=DEFAULT_MIN_ANNOTATIONS,
        help="Include only tasks with at least this many non-cancelled annotations (default: 3)",
    )
    args = parser.parse_args()

    export_path = resolve_repo_path(args.export)
    if not export_path.is_file():
        raise FileNotFoundError(f"Export not found: {export_path}")

    raw_tasks = json.loads(export_path.read_text(encoding="utf-8"))
    if not isinstance(raw_tasks, list):
        raise ValueError("Expected a JSON list of Label Studio tasks")

    filtered_raw, filter_summary = filter_tasks_by_min_annotations(
        raw_tasks, args.min_annotations
    )
    tasks = parse_export(filtered_raw)
    report = build_report(
        tasks,
        filtered_raw,
        filter_summary,
        min_annotations=args.min_annotations,
    )
    print_report(report)

    if args.json_out:
        out_path = resolve_repo_path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote JSON report to {out_path}")


if __name__ == "__main__":
    main()
