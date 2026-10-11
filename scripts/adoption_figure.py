#!/usr/bin/env python3
"""
adoption_figure.py - Receiver-conditioned adoption in the verified run set (the cells of the adoption table,
tab:adoption / Table 17 of the paper), drawn at ACL column width.

Top row: adoption rate from turn 1 to 2 per model, for the misinformed agent as source (filled) and for the
designated uninformed peer as source (open). Bottom row: excess adoption (misinformed - peer) at the turn
transitions 1->2 and 2->3. Same selection as the table: verified run set, plain false fact, 1-of-3 debates,
debates in which the misinformed agent held the intended wrong answer at t (injection success), all
questions of the run set, cells with fewer than 10 eligible receiver pairs omitted.

Input:  out/results/propagation_by_turn.csv (scripts/propagation_metrics.py)
Output: out/figures/fig_adoption.{pdf,png}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from scripts.common import DATASET_ORDER, FIGURES_DIR, MODEL_ORDER, RESULTS_DIR, ensure_dirs, read_csv  # noqa: E402
from scripts.figure_common import (COLUMN_FIG_WIDTH, MODEL_COLOR, dataset_label, finish_with_legend,  # noqa: E402
                                   layout_style, panel_title, save_figure, style_axes)

SHORT = {"Llama-3.3-70B": "Llama-3.3", "GLM-4.7-Flash": "GLM-4.7", "GPT-OSS-120B": "GPT-OSS"}
LEGEND = {"Llama-3.3-70B": "Llama-3.3", "GLM-4.7-Flash": "GLM-4.7", "GPT-OSS-120B": "GPT-OSS"}
METRICS = ("adoption_from_misinformed", "adoption_from_uninformed_peer", "excess_adoption_vs_peer")


def load_cells(path: Path, min_transitions: int) -> dict:
    """(model, dataset, turn_from, metric) -> (value, ci_lo, ci_hi) for the cells shown in the adoption table."""
    cells = {}
    for r in read_csv(path):
        if not (r["run_set"] == "verified" and r["information_source"] == "misinformation" and r["subset"] == "all"
                and r["k"] == "1" and r["n"] == "3" and r["restrict"] == "injection_success"
                and r["metric"] in METRICS and r["turn_from"] in ("1", "2")):
            continue
        cells[(r["model"], r["dataset"], int(r["turn_from"]), r["metric"])] = (
            float(r["value"]), float(r["ci_lo"]), float(r["ci_hi"]), int(float(r["n_transitions"] or 0)))
    # the table omits a model x dataset x turn cell when the misinformed source has fewer eligible receiver pairs
    keep = {(m, d, t) for (m, d, t, met), v in cells.items()
            if met == "adoption_from_misinformed" and v[3] >= min_transitions and v[0] == v[0]}
    return {k: v for k, v in cells.items() if k[:3] in keep}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(RESULTS_DIR / "propagation_by_turn.csv"))
    ap.add_argument("--min_transitions", type=int, default=10)
    args = ap.parse_args()
    ensure_dirs()
    cells = load_cells(Path(args.csv), args.min_transitions)
    models = [m for m in MODEL_ORDER if any(k[0] == m for k in cells)]
    datasets = [d for d in DATASET_ORDER if any(k[1] == d for k in cells)]
    print(f"{len({k[:3] for k in cells})} model x dataset x turn cells")

    L = layout_style()
    fs = L["fs"]
    ms, ew, elw = 4.5, 0.9, 0.9
    fig, axes = plt.subplots(2, len(datasets), figsize=(COLUMN_FIG_WIDTH, 3.6), sharey="row", squeeze=False)

    def point(ax, x, v, color, filled):
        val, lo, hi, _n = v
        ax.errorbar(x, val, yerr=[[val - lo], [hi - val]], fmt="none", ecolor=color, elinewidth=elw, capsize=0, zorder=2)
        ax.plot(x, val, marker="o", markersize=ms, linestyle="none", zorder=3, markeredgewidth=ew,
                markerfacecolor=color if filled else "white", markeredgecolor=color)

    for j, ds in enumerate(datasets):
        # top: adoption at turn 1 -> 2, misinformed source (filled) vs uninformed peer (open)
        ax = axes[0][j]
        for i, m in enumerate(models):
            col = MODEL_COLOR.get(m, "#999999")
            for metric, dx, filled in (("adoption_from_misinformed", -0.15, True),
                                       ("adoption_from_uninformed_peer", 0.15, False)):
                v = cells.get((m, ds, 1, metric))
                if v:
                    point(ax, i + dx, v, col, filled)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_xlim(-0.5, len(models) - 0.5)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([SHORT.get(m, m) for m in models], rotation=40, ha="right", rotation_mode="anchor",
                           fontsize=fs["tick"] - 0.5)
        panel_title(ax, dataset_label(ds), fs)
        style_axes(ax, "y", fs)
        # bottom: excess adoption (misinformed - peer) at 1 -> 2 and 2 -> 3
        ax = axes[1][j]
        w = 0.18
        for i, m in enumerate(models):
            col = MODEL_COLOR.get(m, "#999999")
            for ti, t in enumerate((1, 2)):
                v = cells.get((m, ds, t, "excess_adoption_vs_peer"))
                if v:
                    point(ax, ti + (i - (len(models) - 1) / 2) * w, v, col, True)
        ax.axhline(0, color="#2e2e2e", linewidth=0.7, linestyle="--", zorder=1)
        ax.set_xlim(-0.5, 1.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["1$\\!\\rightarrow\\!$2", "2$\\!\\rightarrow\\!$3"], fontsize=fs["tick"])
        style_axes(ax, "y", fs)
    axes[1][0].set_ylim(-0.75, 0.5)
    axes[1][0].set_yticks([-0.6, -0.3, 0.0, 0.3])
    axes[1][len(datasets) // 2].set_xlabel("Turn transition", fontsize=fs["label"])
    axes[0][0].set_ylabel("Adoption rate\n(turn 1$\\!\\rightarrow\\!$2)", fontsize=fs["label"])
    axes[1][0].set_ylabel("Excess adoption\n(misinformed $-$ peer)", fontsize=fs["label"])

    # legend: model colours in the first row, source markers in the second (matplotlib fills legends by column)
    mh = {m: Line2D([], [], marker="o", linestyle="none", markersize=ms, markerfacecolor=MODEL_COLOR.get(m, "#999999"),
                    markeredgecolor=MODEL_COLOR.get(m, "#999999"), label=LEGEND.get(m, m)) for m in models}
    filled = Line2D([], [], marker="o", linestyle="none", markersize=ms, markerfacecolor="#2e2e2e",
                    markeredgecolor="#2e2e2e", label="misinformed source")
    open_ = Line2D([], [], marker="o", linestyle="none", markersize=ms, markerfacecolor="white",
                   markeredgecolor="#2e2e2e", markeredgewidth=ew, label="uninformed source")
    order = [mh[models[0]], filled, mh[models[1]], open_, mh[models[2]]] if len(models) == 3 else list(mh.values())
    # framed legend above the panels, as in the other figures (scripts.figure_common)
    finish_with_legend(fig, order, [h.get_label() for h in order], fs, ncol=3, top=0.88,
                       handlelength=0.8, handletextpad=0.3, columnspacing=0.8)
    out = FIGURES_DIR / "fig_adoption"
    save_figure(fig, out)
    print(f"figure -> {out}.pdf")


if __name__ == "__main__":
    main()
