#!/usr/bin/env python3
"""
teaser_figure.py - Figure 1 of the paper: setting and main result, at ACL column width.

Top: schematic of the comparison. Three agents answer the same question; k of them (one in the drawing)
also read a false MINT passage. In the multi-agent debate (MAD) the agents see each other's answers for up
to five turns and the group answer is a vote or the last speaker's answer; in the exposure-matched baseline
(mSC) the same three agents, with the same k misinformed, answer once each without interaction and the
group answer is the majority.

Bottom: the debate effect, accuracy of MAD minus accuracy of mSC at the same k, per model and dataset for
k = 0..3 (paired over the verified questions; dots with bootstrap 95% CIs).

Input:  out/results/debate_effect.csv (scripts/exposure_curve.py)
Output: out/figures/fig_teaser_consensus.{pdf,png}   (--protocol voting -> fig_teaser)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

from scripts.common import DATASET_ORDER, FIGURES_DIR, MODEL_ORDER, RESULTS_DIR, ensure_dirs, read_csv  # noqa: E402
from scripts.figure_common import (COLUMN_FIG_WIDTH, EDGE, MODEL_COLOR, dataset_label, layout_style,  # noqa: E402
                                   model_label, panel_title, save_figure, style_axes)
from scripts.figure_style import get_palette  # noqa: E402

MISINFORMED = get_palette(6)[3]   # rose, as the misinformation colour of the single-agent figure
BOX_FILL = "#f2f2f2"
MUTED = "#6a6a6a"


def draw_schematic(ax, fs, box_ratio: float) -> None:
    """Two rows of three agents (one misinformed) feeding a decision box.

    Axis coordinates: 6.2 units high and `6.2 * box_ratio` wide (box_ratio = width / height of the axes box in
    inches), so the drawing keeps circles circular and fills the whole box."""
    x_max = 6.2 * box_ratio
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 6.2)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    r = 0.42
    xs = (0.95, 2.35, 3.75)

    def agents(y: float, misinformed_at: int = 1) -> None:
        for i, x in enumerate(xs):
            face = MISINFORMED if i == misinformed_at else "white"
            ax.add_patch(Circle((x, y), r, facecolor=face, edgecolor=EDGE, linewidth=0.8, zorder=3))
        # the false passage: a small document on the misinformed agent
        px, py = xs[misinformed_at] + r * 0.55, y + r * 0.35
        ax.add_patch(Rectangle((px, py), 0.42, 0.52, facecolor="white", edgecolor=EDGE, linewidth=0.6, zorder=4))
        for k in range(3):
            ax.plot([px + 0.08, px + 0.34], [py + 0.40 - 0.12 * k] * 2, color=EDGE, linewidth=0.5, zorder=5)

    def decision(y: float, text: str) -> None:
        ax.add_patch(FancyArrowPatch((4.55, y), (5.25, y), arrowstyle="-|>", mutation_scale=7, color=EDGE,
                                     linewidth=0.8, zorder=3))
        x0, x1 = 5.4, x_max - 0.25
        ax.add_patch(FancyBboxPatch((x0, y - 0.5), x1 - x0, 1.0, boxstyle="round,pad=0.02,rounding_size=0.18",
                                    facecolor=BOX_FILL, edgecolor=EDGE, linewidth=0.7, zorder=3))
        ax.text((x0 + x1) / 2, y, text, ha="center", va="center", fontsize=fs["tick"], zorder=4)

    # row 1: debate
    y1 = 4.65
    ax.text(0.5, 5.75, "Multi-agent debate (MAD)", fontsize=fs["panel_title"], fontweight="bold", va="center")
    agents(y1)
    for (xa, xb) in ((xs[0], xs[1]), (xs[1], xs[2])):
        ax.add_patch(FancyArrowPatch((xa + r * 0.6, y1 + r * 0.95), (xb - r * 0.6, y1 + r * 0.95), arrowstyle="-|>",
                                     connectionstyle="arc3,rad=-0.45", mutation_scale=5, color=EDGE, linewidth=0.6,
                                     zorder=2))
        ax.add_patch(FancyArrowPatch((xb - r * 0.6, y1 - r * 0.95), (xa + r * 0.6, y1 - r * 0.95), arrowstyle="-|>",
                                     connectionstyle="arc3,rad=-0.45", mutation_scale=5, color=EDGE, linewidth=0.6,
                                     zorder=2))
    ax.text(xs[1], y1 - r - 0.55, "up to 5 turns", ha="center", va="center", fontsize=fs["value"], style="italic",
            color=MUTED)
    decision(y1, "vote or last speaker")

    # row 2: exposure-matched baseline
    y2 = 2.0
    ax.text(0.5, 3.1, "Exposure-matched baseline (mSC)", fontsize=fs["panel_title"], fontweight="bold", va="center")
    agents(y2)
    ax.text(xs[1], y2 - r - 0.55, "no interaction", ha="center", va="center", fontsize=fs["value"], style="italic",
            color=MUTED)
    decision(y2, "majority of 3 samples")

    # key
    y3 = 0.45
    ax.add_patch(Circle((0.95, y3), r * 0.7, facecolor="white", edgecolor=EDGE, linewidth=0.8, zorder=3))
    ax.text(1.5, y3, "uninformed", va="center", fontsize=fs["value"] + 0.5)
    ax.add_patch(Circle((5.4, y3), r * 0.7, facecolor=MISINFORMED, edgecolor=EDGE, linewidth=0.8, zorder=3))
    ax.text(5.95, y3, "misinformed (false passage)", va="center", fontsize=fs["value"] + 0.5)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(RESULTS_DIR / "debate_effect.csv"))
    ap.add_argument("--protocol", choices=("voting", "consensus"), default="consensus")
    args = ap.parse_args()
    ensure_dirs()
    rows = [r for r in read_csv(Path(args.csv)) if r["baseline"] == "mSC" and r["protocol"] == args.protocol
            and str(r.get("n", "3")) == "3"]
    cells = {(r["model"], r["dataset"], int(r["k"])): (float(r["debate_effect"]), float(r["ci_lo"]), float(r["ci_hi"]))
             for r in rows}
    models = [m for m in MODEL_ORDER if any(k[0] == m for k in cells)]
    datasets = [d for d in DATASET_ORDER if any(k[1] == d for k in cells)]
    ks = sorted({k[2] for k in cells})

    L = layout_style()
    fs = L["fs"]
    fig_h = 3.7
    fig = plt.figure(figsize=(COLUMN_FIG_WIDTH, fig_h))
    # the schematic spans the full column (it has no y axis); the panels leave room for theirs
    gs_top = fig.add_gridspec(1, 1, left=0.02, right=0.99, top=0.995, bottom=0.56)
    gs_bot = fig.add_gridspec(1, len(datasets), left=0.17, right=0.985, top=0.415, bottom=0.132, wspace=0.12)
    ax_top = fig.add_subplot(gs_top[0, 0])
    box = ax_top.get_position()
    draw_schematic(ax_top, fs, box_ratio=(box.width * COLUMN_FIG_WIDTH) / (box.height * fig_h))

    axs = []
    ms, elw, step = 3.8, 0.9, 0.24
    for j, ds in enumerate(datasets):
        ax = fig.add_subplot(gs_bot[0, j], sharey=axs[0] if axs else None)
        axs.append(ax)
        for mi, m in enumerate(models):
            col = MODEL_COLOR.get(m, "#999999")
            pts = [(k,) + cells[(m, ds, k)] for k in ks if (m, ds, k) in cells]
            if not pts:
                continue
            x = [k + (mi - (len(models) - 1) / 2) * step for k, *_ in pts]
            y = [v for _, v, _, _ in pts]
            ax.plot(x, y, color=col, linewidth=0.9, alpha=0.6, zorder=2)
            for xi, (k, v, lo, hi) in zip(x, pts):
                ax.errorbar(xi, v, yerr=[[v - lo], [hi - v]], fmt="none", ecolor=col, elinewidth=elw, capsize=0, zorder=2)
                ax.plot(xi, v, marker="o", linestyle="none", markersize=ms, markeredgewidth=elw * 0.9, markeredgecolor=col,
                        markerfacecolor=col, zorder=3)
        ax.axhline(0, color=EDGE, linewidth=0.7, linestyle="--", zorder=1)
        ax.set_xlim(min(ks) - 0.6, max(ks) + 0.6)
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks], fontsize=fs["tick"])
        panel_title(ax, dataset_label(ds), fs)
        style_axes(ax, "y", fs)
        if j > 0:
            ax.tick_params(axis="y", labelleft=False)
    lo = min(c[1] for c in cells.values())
    hi = max(c[2] for c in cells.values())
    axs[0].set_ylim(lo - 0.04, hi + 0.04)
    axs[0].set_yticks([-0.2, 0.0, 0.2])
    axs[0].set_ylabel("Debate effect\n(acc. MAD $-$ acc. mSC)", fontsize=fs["label"])
    axs[len(axs) // 2].set_xlabel("Misinformed agents (of 3)", fontsize=fs["label"])

    # framed legend between the schematic and the panels: model colours, plus the open marker
    handles = [Line2D([], [], marker="o", linestyle="none", markersize=ms, color=MODEL_COLOR.get(m, "#999999"),
                      label=model_label(m)) for m in models]
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    y_top = max(ax.get_tightbbox(fig.canvas.get_renderer()).transformed(inv).y1 for ax in axs)
    fig.legend(handles, [h.get_label() for h in handles], loc="lower center", bbox_to_anchor=(0.5, y_top + 0.012),
               ncol=3, frameon=True, fontsize=fs["legend"], handlelength=1.0, handletextpad=0.4, columnspacing=1.0,
               labelspacing=0.3, borderpad=0.45)
    stem = "fig_teaser" if args.protocol == "voting" else f"fig_teaser_{args.protocol}"
    out = FIGURES_DIR / stem
    save_figure(fig, out)
    print(f"figure -> {out}.pdf ({len(cells)} cells, {args.protocol})")


if __name__ == "__main__":
    main()
