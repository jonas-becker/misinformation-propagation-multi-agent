#!/usr/bin/env python3
"""Sample MINT v1.1 subsets and export an annotation-ready Excel workbook and CSV.

Default behavior:
- Sample 50 random rows from each subset in MINT-dataset_v1.1.
- Aggregate into one Excel workbook and a matching UTF-8 CSV (same rows as the annotation sheet).
- Use long format: one row per misinformation strategy (8 rows per sampled item).
- Do not add per-annotator label columns; keep only task/content fields.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation


SUBSET_FILES = [
    "complex_web_questions_misinformed.json",
    "ethics_commonsense_misinformed.json",
    "winogrande_misinformed.json",
]


TASK_INSTRUCTIONS = {
    "complex_web_questions": "Answer the question.",
    "winogrande": "Fill the gap.",
    "ethics_commonsense": "Is the described situation ethical or unethical?",
}


def _to_cell_text(value: object) -> object:
    """Render non-scalar values predictably for Excel cells."""
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _answer_for_export(sample: Dict, subset_name: str) -> object:
    """Normalize answer cell value for export."""
    value = sample.get("answer")

    # For CWQ, keep only canonical answer content (exclude aliases).
    if subset_name == "complex_web_questions":
        if isinstance(value, dict):
            canonical = value.get("answer")
            if isinstance(canonical, list):
                return canonical[0] if canonical else None
            return canonical
        if isinstance(value, list):
            return value[0] if value else None
        return value

    # For multiple-choice subsets, prefer full answer text from options.
    options = sample.get("options")
    if isinstance(options, list) and options:
        # Numeric encodings (expected: 0-based indexing from dataset builder).
        if isinstance(value, int):
            if 0 <= value < len(options):
                return options[value]
        if isinstance(value, str):
            v = value.strip()
            if v.isdigit():
                idx = int(v)
                if 0 <= idx < len(options):
                    return options[idx]
            # Letter encodings (A/B/C...).
            if len(v) == 1 and v.upper().isalpha():
                idx = ord(v.upper()) - ord("A")
                if 0 <= idx < len(options):
                    return options[idx]

    return value


def _collect_strategy_names(samples: List[Dict]) -> List[str]:
    names = set()
    for sample in samples:
        by_strategy = sample.get("misinformation_by_strategy")
        if isinstance(by_strategy, dict):
            names.update(by_strategy.keys())
    return sorted(names)


def _sample_subset(path: Path, n: int, rng: random.Random) -> List[Dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    if len(data) <= n:
        return list(data)
    indices = rng.sample(range(len(data)), n)
    return [data[i] for i in indices]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample MINT subsets into annotation Excel workbook")
    parser.add_argument("--input_dir", type=str, default="MINT-dataset_v1.1")
    parser.add_argument("--output_xlsx", type=str, default="MINT-dataset_v1.1/MINT_annotation_sampled.xlsx")
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path for annotation CSV (default: same path as --output_xlsx with .csv suffix).",
    )
    parser.add_argument("--samples_per_subset", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--label_values",
        type=str,
        default="",
        help="Optional comma-separated dropdown values for annotation label columns.",
    )
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_path = Path(args.output_xlsx)
    csv_path = Path(args.output_csv) if args.output_csv else out_path.with_suffix(".csv")
    rng = random.Random(args.seed)

    all_sampled: List[Dict] = []
    for file_name in SUBSET_FILES:
        subset_path = in_dir / file_name
        if not subset_path.exists():
            raise FileNotFoundError(f"Missing subset file: {subset_path}")
        sampled = _sample_subset(subset_path, args.samples_per_subset, rng)
        for row in sampled:
            row["_subset"] = file_name.replace("_misinformed.json", "")
        all_sampled.extend(sampled)
        print(f"Sampled {len(sampled)} from {file_name}")

    strategy_names = _collect_strategy_names(all_sampled)
    rows: List[Dict] = []
    for sample in all_sampled:
        by_strategy = sample.get("misinformation_by_strategy") or {}
        shared = {
            "subset": sample.get("_subset"),
            "MINT_ID": sample.get("MINT_ID"),
            "task_instruction": TASK_INSTRUCTIONS.get(sample.get("_subset"), "Complete the task."),
            "sentence": _to_cell_text(sample.get("sentence")),
            "answer": _to_cell_text(_answer_for_export(sample, sample.get("_subset"))),
            "options": _to_cell_text(sample.get("options")),
            "false_fact": _to_cell_text(sample.get("false_fact")),
            "irrelevant_true_information": _to_cell_text(sample.get("irrelevant_true_information")),
        }

        # One row per misinformation strategy annotation
        for strategy in strategy_names:
            row = dict(shared)
            row["claim_type"] = "misinformation_strategy"
            row["strategy"] = strategy
            row["claim_text"] = _to_cell_text(by_strategy.get(strategy))
            rows.append(row)

    # Final global shuffle so annotators see a randomized mix of all subsets/claims.
    rng.shuffle(rows)

    df = pd.DataFrame(rows)

    instructions = [
        {
            "field": "row format",
            "description": "Each row is one misinformation strategy claim (8 rows per sampled item).",
        },
        {
            "field": "claim_type",
            "description": "false_fact or misinformation_strategy.",
        },
        {
            "field": "strategy",
            "description": "Filled only when claim_type=misinformation_strategy.",
        },
        {
            "field": "claim_text",
            "description": "Text that annotators should label in this row.",
        },
        {
            "field": "annotation columns",
            "description": "No annotator-specific label columns are included in this export.",
        },
    ]
    guide_df = pd.DataFrame(instructions)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="annotation", index=False)
        guide_df.to_excel(writer, sheet_name="instructions", index=False)
        _format_annotation_sheet(
            writer=writer,
            annotation_df=df,
            label_values_csv=args.label_values,
        )

    df.to_csv(csv_path, index=False, encoding="utf-8", na_rep="")

    print(f"Wrote {out_path} with {len(df)} rows.")
    print(f"Wrote {csv_path} with {len(df)} rows.")
    print(f"Strategies included: {strategy_names}")


def _format_annotation_sheet(writer: pd.ExcelWriter, annotation_df: pd.DataFrame, label_values_csv: str) -> None:
    """Apply spreadsheet formatting that imports cleanly into Google Sheets."""
    ws = writer.sheets["annotation"]
    max_row = len(annotation_df) + 1
    max_col = len(annotation_df.columns)

    # Freeze header and enable filters.
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions

    # Header style for readability.
    header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
    for c in range(1, max_col + 1):
        cell = ws.cell(row=1, column=c)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    # Column widths tuned for Sheets import.
    width_overrides = {
        "subset": 20,
        "MINT_ID": 36,
        "task_instruction": 42,
        "sentence": 65,
        "answer": 40,
        "options": 40,
        "false_fact": 65,
        "irrelevant_true_information": 65,
        "claim_type": 26,
        "strategy": 18,
        "claim_text": 80,
    }
    for idx, col_name in enumerate(annotation_df.columns, start=1):
        width = width_overrides.get(col_name, 18)
        ws.column_dimensions[ws.cell(row=1, column=idx).column_letter].width = width

    # Wrap long text columns and top-align rows.
    wrap_cols = {
        "task_instruction",
        "sentence",
        "answer",
        "options",
        "false_fact",
        "irrelevant_true_information",
        "claim_text",
    }
    for idx, col_name in enumerate(annotation_df.columns, start=1):
        wrap = col_name in wrap_cols
        for r in range(2, max_row + 1):
            ws.cell(row=r, column=idx).alignment = Alignment(
                wrap_text=wrap,
                vertical="top",
                horizontal="left",
            )

    # Dynamic row heights so wrapped text is fully visible + one extra line.
    wrap_width_by_col_idx: Dict[int, int] = {}
    for idx, col_name in enumerate(annotation_df.columns, start=1):
        if col_name in wrap_cols:
            wrap_width_by_col_idx[idx] = int(width_overrides.get(col_name, 18))

    points_per_line = 15
    for r in range(2, max_row + 1):
        needed_lines = 1
        for col_idx, col_width in wrap_width_by_col_idx.items():
            value = ws.cell(row=r, column=col_idx).value
            if value is None:
                continue
            text = str(value)
            if not text:
                continue

            logical_lines = text.splitlines() or [text]
            line_count = 0
            for logical in logical_lines:
                length = max(1, len(logical))
                line_count += max(1, math.ceil(length / max(1, col_width)))
            needed_lines = max(needed_lines, line_count)

        # +1 extra line for visual spacing between rows.
        ws.row_dimensions[r].height = (needed_lines + 1) * points_per_line

    # Optional dropdown validation for annotation label columns.
    # (No-op when no ann* label columns are present.)
    label_values = [v.strip() for v in (label_values_csv or "").split(",") if v.strip()]
    if label_values:
        formula = '"' + ",".join(label_values) + '"'
        dv = DataValidation(type="list", formula1=formula, allow_blank=True)
        ws.add_data_validation(dv)
        for idx, col_name in enumerate(annotation_df.columns, start=1):
            if col_name.startswith("ann") and ("_label_1" in col_name or "_label_2" in col_name):
                col_letter = ws.cell(row=1, column=idx).column_letter
                dv.add(f"{col_letter}2:{col_letter}{max_row}")


if __name__ == "__main__":
    main()

