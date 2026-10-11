"""figure_style.py - the project palette and the matplotlib style shared by all figure scripts."""

from __future__ import annotations

from typing import Dict, List

PALETTES: Dict[int, List[str]] = {
    2: ["#5e4c5f", "#ffbb6f"],
    3: ["#5e4c5f", "#999999", "#ffbb6f"],
    4: ["#5e4c5f", "#6f8fa6", "#999999", "#ffbb6f"],
    5: ["#5e4c5f", "#6f8fa6", "#7fa37a", "#999999", "#ffbb6f"],
    6: ["#5e4c5f", "#6f8fa6", "#7fa37a", "#c27a7a", "#999999", "#ffbb6f"],
    8: ["#5e4c5f", "#6f8fa6", "#7fa37a", "#c27a7a", "#a88fbf", "#d6a0c4", "#999999", "#ffbb6f"],
}


def get_palette(n: int) -> List[str]:
    """Return a categorical palette of size n (2/3/4/5/6/8)."""
    if n not in PALETTES:
        raise ValueError(f"Unsupported palette size {n}. Supported: {sorted(PALETTES)}")
    return list(PALETTES[n])


def setup_plot_style(font_scale: float = 2.9) -> None:
    """Project-wide matplotlib style: seaborn-muted base, serif fonts, the project colour
    cycle and font sizes scaled for 15-inch-wide figures (scripts.figure_common rescales
    them for column-width figures)."""
    import matplotlib.pyplot as plt
    from matplotlib import cycler

    plt.style.use("seaborn-v0_8-muted")
    plt.rcParams["axes.prop_cycle"] = cycler(color=get_palette(8))
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Palatino", "TeX Gyre Pagella", "Times New Roman", "Times", "DejaVu Serif"]
    plt.rcParams["text.usetex"] = False
    plt.rcParams["mathtext.fontset"] = "cm"
    sizes = {"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
             "ytick.labelsize": 7, "legend.fontsize": 7, "figure.titlesize": 8}
    for key, value in sizes.items():
        plt.rcParams[key] = value * font_scale
    plt.rcParams["lines.linewidth"] = 1.0
    plt.rcParams["lines.markersize"] = 4
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
