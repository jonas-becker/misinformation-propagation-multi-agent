#!/usr/bin/env python3
"""
figure_common.py - Shared styling of the figures:

  * `figure_style.setup_plot_style()` (seaborn-muted base, serif fonts, project colour
    cycle, font sizes scaled for 15-inch-wide figures);
  * project palette from `figure_style.get_palette` (baseline grey, misinformed gold,
    purple, blue);
  * bold titles, dashed light grid on the value axis, no top/right spines, framed legend
    above the axes, value labels coloured like their series (2 decimals);
  * one model per colour (Llama purple, GLM blue, GPT-OSS gold);
  * `save_figure` writes PDF + PNG at 300 dpi with a white background.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.figure_style import get_palette, setup_plot_style  # noqa: E402

# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------
MODEL_LABEL = {"Llama-3.3-70B": "Llama-3.3", "GLM-4.7-Flash": "GLM-4.7", "GPT-OSS-120B": "GPT-OSS-120B"}
DATASET_LABEL = {"cwq": "CWQ", "complex_web_questions": "CWQ", "ethics": "Ethics", "winogrande": "WinoGrande"}

_PAL4 = get_palette(4)   # purple, blue, grey, gold
_PAL6 = get_palette(6)   # purple, blue, sage, rose, grey, gold

# one colour per model, as in exp3 (purple / blue for the first two models, gold for the third)
MODEL_COLOR = {"Llama-3.3-70B": _PAL4[0], "GLM-4.7-Flash": _PAL4[1], "GPT-OSS-120B": _PAL4[3]}
# one colour per system of the exposure figures (baseline grey, misinformed gold, as in the exp1 relevance colours)
SYSTEM_COLOR = {
    "MAD voting": _PAL6[0],        # purple
    "MAD consensus": _PAL6[1],     # blue
    "mSC": _PAL6[5],               # gold
    "single agent": _PAL6[4],      # grey
    "single agent (debate prompt)": _PAL6[2],  # sage
}
EDGE = "#2e2e2e"
GRID = dict(alpha=0.25, linestyle="--", linewidth=0.7, zorder=0)


def apply_paper_style() -> Dict[str, float]:
    """Apply the project style and return the font sizes of the wide (15-inch) figures."""
    setup_plot_style()
    base = float(plt.rcParams.get("font.size", 12))
    return {
        "base": base,
        "title": max(21.0, base * 1.10),
        "panel_title": max(18.0, base * 0.95),
        "label": max(18.0, base * 0.95),
        "tick": max(15.0, base * 0.80),
        "legend": max(15.0, base * 0.80),
        "value": max(12.0, base * 0.62),
        "group": max(17.0, base * 0.90),
    }


COLUMN_FIG_WIDTH = 3.3   # drawn slightly wider than the 3.03 in ACL column; scaled to \columnwidth the fonts stay at about 8-9 pt


def apply_column_style() -> Dict[str, float]:
    """Project style with print-size fonts for a figure that sits in one ACL column.

    The paper style scales fonts for 15-inch-wide figures; a figure printed at 3 inches needs
    fonts of 8-10 pt at the drawn size, otherwise they shrink to 3-4 pt on the page."""
    apply_paper_style()
    plt.rcParams.update({"font.size": 8.0, "axes.titlesize": 9.0, "axes.labelsize": 9.0, "xtick.labelsize": 8.0,
                         "ytick.labelsize": 8.0, "legend.fontsize": 8.0, "figure.titlesize": 9.5,
                         "lines.linewidth": 1.4, "lines.markersize": 4.0, "axes.linewidth": 0.7})
    return {"base": 8.0, "title": 9.5, "panel_title": 9.0, "label": 9.0, "tick": 8.0, "legend": 8.0, "value": 7.0,
            "group": 8.5}


def layout_style() -> Dict[str, object]:
    """Fonts and artist sizes of a figure that sits in one ACL column (3.3 in, panels side by side)."""
    return {"fs": apply_column_style(), "linewidth": 1.4, "markersize": 4.0, "edgewidth": 0.5, "elinewidth": 0.7,
            "capsize": 1.5, "legend_ncol_max": 3}


def panel_axes(n: int, height: float, sharey: bool = True):
    """n panels side by side, `height` inches tall, at column width."""
    fig, axes = plt.subplots(1, n, figsize=(COLUMN_FIG_WIDTH, height), sharey=sharey, squeeze=False)
    return fig, list(axes[0])


def shared_labels(axs, xlabel: str, ylabel: str, fs: Dict[str, float]) -> None:
    """One x label under the middle panel, the y label on the first panel."""
    if xlabel:
        axs[len(axs) // 2].set_xlabel(xlabel, fontsize=fs["label"])
    axs[0].set_ylabel(ylabel, fontsize=fs["label"])


def model_label(model: str) -> str:
    return MODEL_LABEL.get(model, str(model))


def dataset_label(ds: str) -> str:
    return DATASET_LABEL.get(str(ds).lower(), str(ds))


def style_axes(ax, grid_axis: str = "y", fs: Optional[Dict[str, float]] = None) -> None:
    """Dashed light grid on the value axis, grid below the artists, no top/right spines."""
    ax.grid(True, axis=grid_axis, **GRID)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if fs:
        ax.tick_params(axis="both", labelsize=fs["tick"])


def panel_title(ax, text: str, fs: Dict[str, float]) -> None:
    ax.set_title(text, fontweight="bold", fontsize=fs["panel_title"], pad=8)


def annotate_value(ax, x: float, y: float, value: float, color: str, fs: Dict[str, float],
                   above: bool = True, fmt: str = "{:.2f}", offset: float = 9.0) -> None:
    """Value label attached to a point, coloured like its series (exp2/exp3 convention)."""
    ax.annotate(fmt.format(value), (x, y), xytext=(0, offset if above else -offset), textcoords="offset points",
                ha="center", va="bottom" if above else "top", fontsize=fs["value"], color=color, zorder=5)


def series_handle(label: str, color: str, marker: str = "o", linestyle: str = "-", linewidth: float = 2.8,
                  markersize: float = 8.0):
    return plt.Line2D([0], [0], color=color, marker=marker, linestyle=linestyle, linewidth=linewidth, markersize=markersize,
                      markeredgecolor=EDGE, markeredgewidth=0.6, label=label)


def patch_handle(label: str, color: str):
    from matplotlib.patches import Patch

    return Patch(facecolor=color, edgecolor=EDGE, linewidth=0.6, label=label)


def finish_with_legend(fig, handles: Iterable, labels: Iterable[str], fs: Dict[str, float], ncol: int,
                       top: float = 0.90, gap: float = 0.012, **legend_kw):
    """Framed legend at the top of the figure with the panels laid out directly beneath it (no figure
    title; multi-panel figures carry their description in the caption).

    The legend is drawn first, its height is measured, and tight_layout then uses exactly the remaining
    space, so there is no empty band between legend and panel titles. `top` is only a fallback when
    the legend extent cannot be measured. `legend_kw` overrides the spacing defaults (e.g. a tighter
    handlelength / columnspacing for marker-only legends at column width)."""
    kw = dict(handlelength=2.0, handletextpad=0.5, columnspacing=1.4, labelspacing=0.35, borderpad=0.5)
    kw.update(legend_kw)
    leg = fig.legend(list(handles), list(labels), loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=ncol,
                     frameon=True, fontsize=fs["legend"], **kw)
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inv = fig.transFigure.inverted()
        leg_h = leg.get_window_extent(renderer).transformed(inv).height
        # 1) lay the panels out below the space the legend needs
        fig.tight_layout(rect=(0.0, 0.0, 1.0, max(0.5, 1.0 - leg_h - gap)))
        fig.canvas.draw()
        # 2) drop the legend onto the highest panel edge (titles and tick labels included);
        #    savefig(bbox_inches="tight") trims whatever is left above it
        axes_top = max(ax.get_tightbbox(renderer).transformed(inv).y1 for ax in fig.axes)
        leg.set_bbox_to_anchor((0.5, axes_top + gap), transform=fig.transFigure)
    except Exception:  # noqa: BLE001
        fig.tight_layout(rect=(0.0, 0.0, 1.0, top))
    return leg


def _crop_pdf_to_ink(path: Path, pad_pt: float = 1.5) -> bool:
    """Shrink the PDF page to the drawn content. matplotlib's tight bbox keeps the full rotated line box of
    tick labels, which leaves up to 0.1 in of white below rotated labels; this crops to the ink itself.
    Needs pypdfium2 (optional); returns False and leaves the file unchanged when it is not installed."""
    try:
        import numpy as np
        import pypdfium2 as pdfium
    except Exception:  # noqa: BLE001
        return False
    try:
        pdf = pdfium.PdfDocument(str(path))
        page = pdf[0]
        w, h = page.get_size()
        scale = 4.0
        img = np.asarray(page.render(scale=scale).to_pil().convert("L"))
        ink = img < 250
        rows, cols = np.where(ink.any(axis=1))[0], np.where(ink.any(axis=0))[0]
        if len(rows) == 0:
            return False
        left, right = max(0.0, cols[0] / scale - pad_pt), min(w, (cols[-1] + 1) / scale + pad_pt)
        bottom, top = max(0.0, h - (rows[-1] + 1) / scale - pad_pt), min(h, h - rows[0] / scale + pad_pt)
        page.set_mediabox(left, bottom, right, top)
        page.set_cropbox(left, bottom, right, top)
        tmp = path.with_suffix(".crop.tmp.pdf")
        pdf.save(str(tmp))
        pdf.close()
        tmp.replace(path)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] could not crop {path.name} to its ink: {e}")
        return False


def _crop_png_to_ink(path: Path, pad_px: int = 6) -> None:
    from PIL import Image, ImageChops

    im = Image.open(path).convert("RGB")
    diff = ImageChops.difference(im, Image.new("RGB", im.size, (255, 255, 255)))
    box = diff.getbbox()
    if box:
        l, t, r, b = box
        im.crop((max(0, l - pad_px), max(0, t - pad_px), min(im.width, r + pad_px), min(im.height, b + pad_px))).save(path)


def save_figure(fig, path_stem: Path, formats: Sequence[str] = ("pdf", "png"), dpi: int = 300) -> List[Path]:
    """Write <stem>.pdf and <stem>.png (white background, no white margin)."""
    out = []
    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        p = path_stem.with_suffix(f".{ext}")
        try:
            # bbox_inches="tight" crops to the artists' boxes; the ink crop below removes what those boxes
            # overestimate (rotated tick labels), so the figure fills \columnwidth or \textwidth completely
            fig.savefig(p, dpi=dpi, bbox_inches="tight", pad_inches=0.02, facecolor="white")
            if ext == "pdf":
                _crop_pdf_to_ink(p)
            elif ext == "png":
                _crop_png_to_ink(p)
            out.append(p)
        except PermissionError:
            # typically the file is open in a PDF viewer; keep the other format and report
            print(f"[WARN] cannot overwrite {p} (file in use); close it and re-run to refresh this format")
    plt.close(fig)
    return out
