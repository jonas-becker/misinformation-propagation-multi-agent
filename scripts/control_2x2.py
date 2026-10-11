#!/usr/bin/env python3
"""
control_2x2.py - Relevance x truth control analysis and length matching.

Cells (single agent, from out/results/rescored/single_agent.csv):
  relevant-false     exp1 false_fact                       (MINT passage of the question)
  irrelevant-false   exp1 irrelevant_misinformed           (false fact of another question)
  relevant-true      exp1 relevant_true                    (controls/relevant_true)
  irrelevant-true    exp1a irrelevant_true_information     (MINT v1.1 wiki passage)
  filler             exp1 filler                           (content-free, length matched)
  baseline           exp1 baseline
Effects (paired bootstrap over questions):
  falsity effect   = relevant-false - relevant-true
  relevance effect = relevant-true  - irrelevant-true
  content effect   = irrelevant-true - filler
  length effect    = filler - baseline
Also reports token-length statistics per category for relevant-true vs relevant-false
passages (MINT_dataset_v1.2 and controls/relevant_true) and flags categories whose means
differ by more than 10%.
Outputs: out/results/control_2x2.csv, out/results/length_matching_stats.csv, LaTeX tables,
out/figures/fig_control_2x2.pdf
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mint_data import CATEGORIES, DATASET_FILES  # noqa: E402
from scripts.common import DATASET_DISPLAY, DATASET_ORDER, FIGURES_DIR, MODEL_ORDER, RESULTS_DIR, TABLES_DIR, ensure_dirs, read_csv, write_csv, write_latex_table  # noqa: E402
from scripts.stats import bootstrap_mean_ci, paired_bootstrap_diff  # noqa: E402

CELLS = {
    "baseline": ("exp1", "baseline"),
    "relevant_false": ("exp1", "false_fact"),
    "irrelevant_false": ("exp1", "irrelevant_misinformed"),
    "relevant_true": ("exp1", "relevant_true"),
    "irrelevant_true": ("exp1", "irrelevant_true"),
    "filler": ("exp1", "filler"),
}
# the exp1a irrelevant-true run (MINT v1.1 wiki passages) is used only when the
# exp1 run has no irrelevant_true condition
FALLBACK_CELLS = {"irrelevant_true": ("exp1a", "irrelevant_true_information")}
EFFECTS = [("falsity effect (relevant-false - relevant-true)", "relevant_false", "relevant_true"),
           ("relevance effect (relevant-true - irrelevant-true)", "relevant_true", "irrelevant_true"),
           ("content effect (irrelevant-true - filler)", "irrelevant_true", "filler"),
           ("length/deliberation effect (filler - baseline)", "filler", "baseline"),
           ("irrelevant-true gain (irrelevant-true - baseline)", "irrelevant_true", "baseline"),
           ("relevant-false cost (relevant-false - baseline)", "relevant_false", "baseline"),
           ("irrelevant-false cost (irrelevant-false - baseline)", "irrelevant_false", "baseline")]
_TOK = re.compile(r"\S+")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rescored_dir", default=str(RESULTS_DIR / "rescored"))
    ap.add_argument("--n_boot", type=int, default=10_000)
    ap.add_argument("--relevant_true_dir", default=None,
                    help="passage set of the relevant-true control (default: controls/relevant_true, else data/controls/relevant_true)")
    ap.add_argument("--seed", type=int, default=0,
                    help="single-agent seed used for the paired cells (seed 0 is the run that covers all conditions)")
    args = ap.parse_args()
    ensure_dirs()
    if args.relevant_true_dir:
        rt_dir = Path(args.relevant_true_dir)
    else:
        rt_dir = next((p for p in (REPO_ROOT / "controls" / "relevant_true", REPO_ROOT / "data" / "controls" / "relevant_true")
                       if p.is_dir()), REPO_ROOT / "controls" / "relevant_true")
    sa = read_csv(Path(args.rescored_dir) / "single_agent.csv")

    def _seed(r: dict) -> int:
        try:
            return int(float(r.get("seed") or 0))
        except Exception:
            return 0

    pq: Dict[tuple, Dict[str, float]] = defaultdict(dict)
    for r in sa:
        if _seed(r) != args.seed:
            continue
        for cell, (exp, cond) in CELLS.items():
            if r["experiment"] == exp and r["condition"] == cond:
                pq[(r["model"], r["dataset"], cell)][r["mint_id"]] = float(str(r["is_correct"]).lower() in ("true", "1"))
    for cell, (exp, cond) in FALLBACK_CELLS.items():
        have = {(m, d) for (m, d, c) in pq if c == cell}
        for r in sa:
            if r["experiment"] == exp and r["condition"] == cond and (r["model"], r["dataset"]) not in have:
                pq[(r["model"], r["dataset"], cell)][r["mint_id"]] = float(str(r["is_correct"]).lower() in ("true", "1"))

    rows = []
    for model in MODEL_ORDER:
        for ds in DATASET_ORDER:
            for cell in CELLS:
                v = pq.get((model, ds, cell))
                if v:
                    ci = bootstrap_mean_ci(list(v.values()), n_boot=args.n_boot)
                    rows.append({"model": model, "dataset": ds, "kind": "cell", "name": cell, "value": ci["mean"],
                                 "ci_lo": ci["ci_lo"], "ci_hi": ci["ci_hi"], "n": ci["n"], "p_perm": None})
            for name, a, b in EFFECTS:
                va, vb = pq.get((model, ds, a)), pq.get((model, ds, b))
                if not va or not vb:
                    rows.append({"model": model, "dataset": ds, "kind": "effect", "name": name, "value": None, "ci_lo": None,
                                 "ci_hi": None, "n": 0, "p_perm": None})
                    continue
                qs = sorted(set(va) & set(vb))
                res = paired_bootstrap_diff([va[q] for q in qs], [vb[q] for q in qs], n_boot=args.n_boot)
                rows.append({"model": model, "dataset": ds, "kind": "effect", "name": name, "value": res["diff"], "ci_lo": res["ci_lo"],
                             "ci_hi": res["ci_hi"], "n": res["n"], "p_perm": res["p_perm"]})
    write_csv(RESULTS_DIR / "control_2x2.csv", rows)
    tex = [[r["model"], DATASET_DISPLAY.get(r["dataset"], r["dataset"]), r["name"],
            f"{r['value']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]" if r["value"] is not None else "NA (run pending)", r["n"]]
           for r in rows if r["kind"] == "effect"]
    write_latex_table(TABLES_DIR / "control_2x2.tex", ["Model", "Dataset", "Effect", "$\\Delta$ accuracy [95\\% CI]", "n"], tex,
                      caption="Relevance x truth controls (single agent). Paired bootstrap CIs over questions; NA marks cells "
                              "whose runs are pending (relevant-true and filler require the control passages and exp1 --conditions controls).",
                      label="tab:control_2x2", col_spec="lllrr")
    avail = sum(1 for r in rows if r["kind"] == "effect" and r["value"] is not None)
    print(f"{avail} effects available of {sum(1 for r in rows if r['kind'] == 'effect')}")

    # ---- length matching statistics
    len_rows = []
    for ds, fname in DATASET_FILES.items():
        mint_path = REPO_ROOT / "MINT_dataset_v1.2" / fname
        if not mint_path.is_file():
            continue
        mint = json.load(open(mint_path, encoding="utf-8"))
        rt = json.load(open(rt_dir / fname, encoding="utf-8")) if (rt_dir / fname).is_file() else None
        rt_by_id = {it.get("MINT_ID"): it for it in rt} if rt else {}
        for cat in CATEGORIES:
            f_len, t_len = [], []
            for it in mint:
                p = it.get("false_fact") if cat == "false_fact" else (it.get("misinformation_by_strategy") or {}).get(cat)
                if p:
                    f_len.append(len(_TOK.findall(p)))
                r = rt_by_id.get(it.get("MINT_ID"))
                if r:
                    tp = (r.get("relevant_true_by_strategy") or {}).get(cat)
                    if tp:
                        t_len.append(len(_TOK.findall(tp)))
            fm, fs = (np.mean(f_len), np.std(f_len)) if f_len else (None, None)
            tm, ts = (np.mean(t_len), np.std(t_len)) if t_len else (None, None)
            ratio = (tm / fm) if (fm and tm) else None
            len_rows.append({"dataset": ds, "category": cat, "n_false": len(f_len), "false_mean": fm, "false_sd": fs,
                             "n_true": len(t_len), "true_mean": tm, "true_sd": ts, "true_over_false": ratio,
                             "flag_gt_10pct": (abs(ratio - 1) > 0.10) if ratio else None})
        # irrelevant-true and filler lengths
        it_len = [it.get("irrelevant_true_information_tokens") for it in mint if it.get("irrelevant_true_information_tokens")]
        fl_len = [it.get("filler_information_tokens") for it in mint if it.get("filler_information_tokens")]
        avg = [it.get("avg_misinformation_tokens") for it in mint if it.get("avg_misinformation_tokens")]
        len_rows.append({"dataset": ds, "category": "irrelevant_true (wiki)", "n_false": len(avg), "false_mean": np.mean(avg) if avg else None,
                         "false_sd": np.std(avg) if avg else None, "n_true": len(it_len), "true_mean": np.mean(it_len) if it_len else None,
                         "true_sd": np.std(it_len) if it_len else None, "true_over_false": (np.mean(it_len) / np.mean(avg)) if it_len and avg else None,
                         "flag_gt_10pct": None})
        len_rows.append({"dataset": ds, "category": "filler", "n_false": len(avg), "false_mean": np.mean(avg) if avg else None,
                         "false_sd": np.std(avg) if avg else None, "n_true": len(fl_len), "true_mean": np.mean(fl_len) if fl_len else None,
                         "true_sd": np.std(fl_len) if fl_len else None, "true_over_false": (np.mean(fl_len) / np.mean(avg)) if fl_len and avg else None,
                         "flag_gt_10pct": None})
    write_csv(RESULTS_DIR / "length_matching_stats.csv", len_rows)
    tex_len = [[r["dataset"], r["category"], f"{r['false_mean']:.1f} $\\pm$ {r['false_sd']:.1f}" if r["false_mean"] else "--",
                f"{r['true_mean']:.1f} $\\pm$ {r['true_sd']:.1f}" if r["true_mean"] else "--",
                f"{r['true_over_false']:.2f}{' *' if r['flag_gt_10pct'] else ''}" if r["true_over_false"] else "--"] for r in len_rows]
    write_latex_table(TABLES_DIR / "length_matching_stats.tex", ["Dataset", "Category", "False (tokens)", "Control (tokens)", "Ratio"], tex_len,
                      caption="Token-length matching (whitespace tokens, mean $\\pm$ SD) between MINT false passages and the relevant-true, "
                              "irrelevant-true and filler controls. * marks categories whose means differ by more than 10\\%.",
                      label="tab:length_matching", col_spec="lllll")

    # ---- figure (2x2 + filler + baseline per model/dataset)
    cells_avail = [r for r in rows if r["kind"] == "cell"]
    if not cells_avail:
        return
    try:
        import matplotlib.pyplot as plt

        from scripts.figure_common import (MODEL_COLOR, apply_paper_style, dataset_label, finish_with_legend, model_label,
                                           panel_title, patch_handle, save_figure, style_axes)

        fs = apply_paper_style()
        # three groups, left to right from no context to the treatment: the uninformed reference, the four
        # controls (content-free, off-topic true, off-topic false, on-topic true) and the misinformation condition
        groups = [("reference", ["baseline"]),
                  ("controls", ["filler", "irrelevant_true", "irrelevant_false", "relevant_true"]),
                  ("misinformation", ["relevant_false"])]
        order = [c for _, cells in groups for c in cells]
        labels = {"baseline": "Uninformed", "relevant_true": "Relevant true", "irrelevant_true": "Irrelevant true",
                  "filler": "Filler", "irrelevant_false": "Irrelevant false", "relevant_false": "Relevant false"}
        models = [m for m in MODEL_ORDER if any(r["model"] == m for r in cells_avail)]
        datasets = [d for d in DATASET_ORDER if any(r["dataset"] == d for r in cells_avail)]
        fig, axes = plt.subplots(1, len(datasets), figsize=(15.5, 6.4), sharey=True, squeeze=False)
        w = 0.8 / max(1, len(models))
        for ax, ds in zip(axes[0], datasets):
            for mi, m in enumerate(models):
                for ci_, cell in enumerate(order):
                    r = next((x for x in cells_avail if x["model"] == m and x["dataset"] == ds and x["name"] == cell), None)
                    if not r:
                        continue
                    x = ci_ + mi * w - 0.4 + w / 2
                    ax.bar(x, r["value"], width=w * 0.92, color=MODEL_COLOR.get(m, "#999999"), edgecolor="white",
                           linewidth=0.5, zorder=2)
                    ax.errorbar(x, r["value"], yerr=[[r["value"] - r["ci_lo"]], [r["ci_hi"] - r["value"]]], color="#2e2e2e",
                                linewidth=0.9, capsize=2.5, zorder=3)
            # dashed separators between the groups (the bracket labels are added below, once the tick labels exist)
            start = 0
            for gi, (gname, cells) in enumerate(groups):
                if gi < len(groups) - 1:
                    ax.axvline(start + len(cells) - 0.5, color="#4a4a4a", linestyle="--", linewidth=1.1, zorder=1)
                start += len(cells)
            ax.set_xlim(-0.6, len(order) - 0.4)
            ax.set_xticks(range(len(order)))
            # anchor each rotated label at its tick so that neighbouring labels stay parallel and do not overlap
            ax.set_xticklabels([labels[c] for c in order], fontsize=fs["tick"], rotation=40, ha="right",
                               rotation_mode="anchor")
            panel_title(ax, dataset_label(ds), fs)
            ax.set_ylim(0, 1.0)
            style_axes(ax, "y", fs)
        axes[0][0].set_ylabel("Accuracy (single agent)", fontsize=fs["label"])
        # group names as brackets below the lowest tick label: the drop is measured in inches, so it does not
        # depend on the axes height that the legend layout sets afterwards
        from matplotlib.transforms import ScaledTranslation
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        drop = 0.0
        for ax in axes[0]:
            ax_bottom = ax.get_window_extent(renderer).y0
            lab_bottom = min(t.get_window_extent(renderer).y0 for t in ax.get_xticklabels() if t.get_text())
            drop = max(drop, (ax_bottom - lab_bottom) / fig.dpi)
        drop += 0.10
        for ax in axes[0]:
            tr_line = ax.get_xaxis_transform() + ScaledTranslation(0, -drop, fig.dpi_scale_trans)
            tr_text = ax.get_xaxis_transform() + ScaledTranslation(0, -drop - 0.06, fig.dpi_scale_trans)
            start = 0
            for gname, cells in groups:
                lo, hi = start - 0.5, start + len(cells) - 0.5
                ax.plot([lo + 0.1, hi - 0.1], [0, 0], transform=tr_line, color="#4a4a4a", linewidth=1.1, clip_on=False,
                        zorder=5)
                ax.text((lo + hi) / 2, 0, gname, transform=tr_text, ha="center", va="top", fontsize=fs["tick"] * 0.95,
                        color="#4a4a4a", style="italic", clip_on=False, zorder=5)
                start += len(cells)
        handles = [patch_handle(model_label(m), MODEL_COLOR.get(m, "#999999")) for m in models]
        finish_with_legend(fig, handles, [h.get_label() for h in handles], fs, ncol=len(models), top=0.90)
        save_figure(fig, FIGURES_DIR / "fig_control_2x2")
        print(f"figure -> {FIGURES_DIR / 'fig_control_2x2.pdf'}")
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] figure skipped: {e}")


if __name__ == "__main__":
    main()
