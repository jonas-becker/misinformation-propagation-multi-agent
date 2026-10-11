#!/usr/bin/env python3
"""
category_figure.py - Accuracy by passage category with CIs, from the
re-scored results (single agent and debate voting), plus the accuracy drop relative to
the uninformed baseline. Outputs out/results/category_accuracy.csv,
out/tables/category_accuracy.tex, out/figures/fig_category_accuracy.pdf
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.common import CATEGORY_ORDER, DATASET_DISPLAY, DATASET_ORDER, FIGURES_DIR, MODEL_ORDER, RESULTS_DIR, TABLES_DIR, ensure_dirs, read_csv, write_csv, write_latex_table  # noqa: E402
from scripts.stats import bootstrap_mean_ci, paired_bootstrap_diff  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rescored_dir", default=str(RESULTS_DIR / "rescored"))
    ap.add_argument("--n_boot", type=int, default=5000)
    args = ap.parse_args()
    ensure_dirs()
    rd = Path(args.rescored_dir)
    sa = read_csv(rd / "single_agent.csv")
    mad = read_csv(rd / "mad_final.csv")
    pq = defaultdict(dict)
    for r in sa:
        # seed 0 is the single-agent run that covers every condition with identical settings
        if str(r.get("seed") or "0") not in ("0", "0.0"):
            continue
        if r["experiment"] == "exp1" and (r["relevance"] == "relevant" or r["condition"] == "baseline"):
            pq[("SA", r["model"], r["dataset"], r["category"])][r["mint_id"]] = float(str(r["is_correct"]).lower() in ("true", "1"))
    for r in mad:
        if r["experiment"] == "exp2" and r["protocol"] == "voting":
            pq[("MAD", r["model"], r["dataset"], r["category"])][r["mint_id"]] = float(str(r["is_correct"]).lower() in ("true", "1"))
    rows = []
    for (system, model, ds, cat), v in pq.items():
        ci = bootstrap_mean_ci(list(v.values()), n_boot=args.n_boot)
        base = pq.get(("SA", model, ds, "none"))
        drop = None
        if system == "SA" and base and cat != "none":
            qs = sorted(set(v) & set(base))
            res = paired_bootstrap_diff([base[q] for q in qs], [v[q] for q in qs], n_boot=args.n_boot)
            drop = (res["diff"], res["ci_lo"], res["ci_hi"])
        rows.append({"system": system, "model": model, "dataset": ds, "category": cat, "acc": ci["mean"], "ci_lo": ci["ci_lo"],
                     "ci_hi": ci["ci_hi"], "n": ci["n"], "drop_vs_baseline": drop[0] if drop else None,
                     "drop_ci_lo": drop[1] if drop else None, "drop_ci_hi": drop[2] if drop else None})
    rows.sort(key=lambda r: (r["system"], MODEL_ORDER.index(r["model"]) if r["model"] in MODEL_ORDER else 9, r["dataset"],
                             CATEGORY_ORDER.index(r["category"]) if r["category"] in CATEGORY_ORDER else -1))
    write_csv(RESULTS_DIR / "category_accuracy.csv", rows)
    tex = [[r["system"], r["model"], DATASET_DISPLAY.get(r["dataset"], r["dataset"]), r["category"],
            f"{r['acc']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]",
            f"{r['drop_vs_baseline']:+.3f} [{r['drop_ci_lo']:+.3f}, {r['drop_ci_hi']:+.3f}]" if r["drop_vs_baseline"] is not None else "--",
            r["n"]] for r in rows]
    write_latex_table(TABLES_DIR / "category_accuracy.tex", ["System", "Model", "Dataset", "Category", "Accuracy [95\\% CI]",
                                                             "Drop vs. baseline [95\\% CI]", "n"], tex,
                      caption="Accuracy by passage category (re-scored). SA = single agent (exp1), MAD = 1-of-3 debate, voting. "
                              "Bootstrap CIs over questions; drop = baseline $-$ condition, paired.", label="tab:category_accuracy",
                      col_spec="llllllr")
    try:
        import matplotlib.pyplot as plt

        from scripts.figure_common import (MODEL_COLOR, apply_paper_style, dataset_label, finish_with_legend, model_label,
                                           panel_title, patch_handle, save_figure, style_axes)

        fs = apply_paper_style()
        models = [m for m in MODEL_ORDER if any(r["model"] == m for r in rows)]
        datasets = [d for d in DATASET_ORDER if any(r["dataset"] == d for r in rows)]
        cats = ["none"] + CATEGORY_ORDER
        cat_label = {"none": "Uninformed", "false_fact": "Plain false fact", "clickbait": "Clickbait", "conspiracy": "Conspiracy",
                     "framing": "Framing", "hoax": "Hoax", "other": "Unconstrained", "propaganda": "Propaganda", "rumor": "Rumor",
                     "satire": "Satire"}
        fig, axes = plt.subplots(len(datasets), 2, figsize=(15.5, 4.0 * len(datasets) + 0.8), sharey=True, squeeze=False)
        w = 0.8 / max(1, len(models))
        for i, ds in enumerate(datasets):
            for j, system in enumerate(("SA", "MAD")):
                ax = axes[i][j]
                # only categories with data in this panel: the full-sample debates have no uninformed runs, so the debate
                # panels have no "Uninformed" slot (the k = 0 debates of the verified run set use other serving settings)
                panel_cats = [c for c in cats if any(x["system"] == system and x["dataset"] == ds and x["category"] == c
                                                     for x in rows)]
                for mi, m in enumerate(models):
                    for ci_, cat in enumerate(panel_cats):
                        r = next((x for x in rows if x["system"] == system and x["model"] == m and x["dataset"] == ds and x["category"] == cat), None)
                        if not r:
                            continue
                        x = ci_ + mi * w - 0.4 + w / 2
                        ax.bar(x, r["acc"], width=w * 0.92, color=MODEL_COLOR.get(m, "#999999"), edgecolor="white",
                               linewidth=0.4, zorder=2)
                        ax.errorbar(x, r["acc"], yerr=[[r["acc"] - r["ci_lo"]], [r["ci_hi"] - r["acc"]]], color="#2e2e2e",
                                    linewidth=0.8, capsize=2, zorder=3)
                ax.set_xticks(range(len(panel_cats)))
                ax.set_xticklabels([cat_label.get(c, c) for c in panel_cats], rotation=35, ha="right", fontsize=fs["tick"])
                ax.set_ylim(0, 1.0)
                style_axes(ax, "y", fs)
                if i == 0:
                    panel_title(ax, "Single agent" if system == "SA" else "Multi-agent debate (1 of 3 misinformed, voting)", fs)
                if j == 0:
                    ax.set_ylabel(f"{dataset_label(ds)}\nAccuracy", fontsize=fs["label"])
        handles = [patch_handle(model_label(m), MODEL_COLOR.get(m, "#999999")) for m in models]
        finish_with_legend(fig, handles, [h.get_label() for h in handles], fs, ncol=len(models), top=0.95)
        save_figure(fig, FIGURES_DIR / "fig_category_accuracy")
        print(f"figure -> {FIGURES_DIR / 'fig_category_accuracy.pdf'}")
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] figure skipped: {e}")
    print(f"wrote {len(rows)} rows")


if __name__ == "__main__":
    main()
