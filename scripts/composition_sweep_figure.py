#!/usr/bin/env python3
"""
composition_sweep_figure.py - Five-agent WinoGrande composition sweep (consensus vs. voting) and the
share of misinformed agents corrected by debate end.

Inputs (from scripts/rescore_results.py):
  out/results/rescored/mad_final.csv   exp3 rows: one row per debate x protocol (voting / consensus)
  out/results/rescored/mad_turns.csv   exp3 rows: one row per debate x agent x turn

Outputs:
  out/results/composition_sweep.csv, out/tables/composition_sweep.tex  (verified questions by default, --questions all
      for the full sample; out/results/composition_sweep_full_sample.{csv,tex} always hold the full sample)
      accuracy per model x configuration x protocol with bootstrap 95% CIs over questions
  out/results/self_correction.csv, out/tables/self_correction.tex
      P(misinformed agent holds the correct answer at its last turn | it held a wrong answer at turn 1),
      per model x configuration, cluster-bootstrap CIs over debates
  out/figures/fig_composition_sweep.pdf|png   (verified questions)
  out/figures/fig_composition_sweep_full_sample.pdf|png   (all questions)
  out/figures/fig_self_correction.pdf|png
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.common import (DATASET_DISPLAY, FIGURES_DIR, MODEL_ORDER, RESULTS_DIR, TABLES_DIR, ensure_dirs, read_csv,  # noqa: E402
                           verified_questions, write_csv, write_latex_table)
from scripts.stats import bootstrap_mean_ci  # noqa: E402

_MIX = re.compile(r"(\d+)_misinformed_(\d+)_informed")


def _b(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "1.0", "yes")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rescored_dir", default=str(RESULTS_DIR / "rescored"))
    ap.add_argument("--dataset", default="winogrande")
    ap.add_argument("--n_boot", type=int, default=10_000)
    ap.add_argument("--questions", choices=("verified", "all"), default="verified",
                    help="question set of the accuracy table and figure: the verified questions of the dataset (default; "
                         "the same rule as the three-agent debates) or all questions of the sample")
    ap.add_argument("--no_figure", action="store_true")
    args = ap.parse_args()
    ensure_dirs()
    rd = Path(args.rescored_dir)
    ds_name = DATASET_DISPLAY.get(args.dataset, args.dataset)
    verified = verified_questions(args.dataset)

    # ---- accuracy per configuration and protocol (bootstrap over questions)
    final = [r for r in read_csv(rd / "mad_final.csv") if r["experiment"] == "exp3" and r["dataset"] == args.dataset]
    per: Dict[tuple, Dict[str, float]] = defaultdict(dict)
    n_agents = 5
    for r in final:
        m = _MIX.match(r["condition"])
        if not m:
            continue
        k = int(m.group(1))
        n_agents = k + int(m.group(2))
        per[(r["model"], k, r["protocol"])][r["mint_id"]] = float(_b(r["is_correct"]))

    def accuracy_rows(restrict: bool) -> List[dict]:
        out = []
        for (model, k, proto), qmap in sorted(per.items(), key=lambda x: (MODEL_ORDER.index(x[0][0]) if x[0][0] in MODEL_ORDER else 9, x[0][1], x[0][2])):
            vals = [v for q, v in qmap.items() if not restrict or q in verified]
            if not vals:
                continue
            ci = bootstrap_mean_ci(vals, n_boot=args.n_boot)
            out.append({"model": model, "dataset": args.dataset, "questions": "verified" if restrict else "all",
                        "k_misinformed": k, "n_agents": n_agents, "protocol": proto,
                        "acc": ci["mean"], "ci_lo": ci["ci_lo"], "ci_hi": ci["ci_hi"], "n_questions": ci["n"]})
        return out

    def write_accuracy(rows: List[dict], stem: str, label: str, qtext: str) -> None:
        write_csv(RESULTS_DIR / f"{stem}.csv", rows)
        write_latex_table(TABLES_DIR / f"{stem}.tex", ["Model", "Misinformed / uninformed", "Protocol", "Accuracy [95\\% CI]", "n"],
                          [[r["model"], f"{r['k_misinformed']} / {r['n_agents'] - r['k_misinformed']}", r["protocol"],
                            f"{r['acc']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]", r["n_questions"]] for r in rows],
                          caption=f"Re-scored accuracy of the five-agent {ds_name} debates by number of misinformed agents and "
                                  f"decision protocol ({qtext}; seed 0, unified extractor, misinformed slots shuffled per "
                                  "question; n = questions with an extracted outcome; bootstrap 95\\% CIs over questions).",
                          label=label, col_spec="lllrr")

    acc_verified, acc_all = accuracy_rows(True), accuracy_rows(False)
    verified_text = f"the {len(verified)} verified questions"
    write_accuracy(acc_all, "composition_sweep_full_sample", "tab:composition_sweep_full_sample", "all questions of the sample")
    if args.questions == "verified":
        acc_rows = acc_verified
        write_accuracy(acc_rows, "composition_sweep", "tab:composition_sweep", verified_text)
    else:
        acc_rows = acc_all
        write_accuracy(acc_rows, "composition_sweep", "tab:composition_sweep", "all questions of the sample")

    # ---- self-correction: misinformed agent wrong at turn 1 -> correct at its last turn
    turns = [r for r in read_csv(rd / "mad_turns.csv") if r["experiment"] == "exp3" and r["dataset"] == args.dataset]
    by_agent: Dict[tuple, Dict[int, dict]] = defaultdict(dict)
    for r in turns:
        if not _b(r.get("is_misinformed")):
            continue
        try:
            t = int(float(r["turn"]))
        except Exception:
            continue
        by_agent[(r["model"], r["condition"], r["mint_id"], r["agent_index"])][t] = r
    outcomes: Dict[tuple, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))  # (model, k) -> debate -> outcomes
    for (model, cond, mint_id, _a), tmap in by_agent.items():
        m = _MIX.match(cond)
        if not m:
            continue
        k = int(m.group(1))
        first = tmap.get(min(tmap))
        last = tmap.get(max(tmap))
        if first is None or last is None or first["status"] != "parsed" or _b(first["is_correct"]):
            continue  # only agents that start with a (parsed) wrong answer
        outcomes[(model, k)][mint_id].append(1.0 if (last["status"] == "parsed" and _b(last["is_correct"])) else 0.0)
    sc_rows = []
    rng = np.random.default_rng(12345)
    for (model, k), dmap in sorted(outcomes.items(), key=lambda x: (MODEL_ORDER.index(x[0][0]) if x[0][0] in MODEL_ORDER else 9, x[0][1])):
        num = np.asarray([sum(v) for v in dmap.values()], float)
        den = np.asarray([len(v) for v in dmap.values()], float)
        est = num.sum() / den.sum()
        idx = rng.integers(0, len(num), size=(args.n_boot, len(num)))
        boots = num[idx].sum(axis=1) / np.maximum(den[idx].sum(axis=1), 1e-9)
        sc_rows.append({"model": model, "dataset": args.dataset, "k_misinformed": k, "n_agents": n_agents,
                        "p_adjusted": est, "ci_lo": float(np.quantile(boots, 0.025)), "ci_hi": float(np.quantile(boots, 0.975)),
                        "n_agents_started_wrong": int(den.sum()), "n_debates": len(dmap)})
    write_csv(RESULTS_DIR / "self_correction.csv", sc_rows)
    write_latex_table(TABLES_DIR / "self_correction.tex", ["Model", "Misinformed / uninformed", "P(corrected) [95\\% CI]", "n agents", "n debates"],
                      [[r["model"], f"{r['k_misinformed']} / {r['n_agents'] - r['k_misinformed']}",
                        f"{r['p_adjusted']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]", r["n_agents_started_wrong"], r["n_debates"]] for r in sc_rows],
                      caption=f"Probability that a misinformed agent that starts with a wrong answer holds the correct answer at its last "
                              f"turn (five-agent {DATASET_DISPLAY.get(args.dataset, args.dataset)} debates, re-scored; cluster-bootstrap 95\\% CIs over debates).",
                      label="tab:self_correction", col_spec="lllrr")
    print(f"wrote {len(acc_rows)} accuracy rows and {len(sc_rows)} self-correction rows")
    if args.no_figure or not acc_rows:
        return

    try:
        import matplotlib.pyplot as plt

        from scripts.figure_common import (MODEL_COLOR, annotate_value, apply_paper_style, finish_with_legend, layout_style,
                                           model_label, panel_axes, panel_title, save_figure, series_handle, shared_labels,
                                           style_axes)

        ks = sorted({r["k_misinformed"] for r in acc_rows})
        models = [m for m in MODEL_ORDER if any(r["model"] == m for r in acc_rows)]
        proto_style = {"consensus": ("-", "o", "Consensus"), "voting": ("--", "s", "Voting")}

        # one panel per model, consensus (solid) vs voting (dashed), one ACL column wide
        # fig_composition_sweep: verified questions (paper Figure 5); fig_composition_sweep_full_sample: all questions (Figure 12)
        def sweep_figure(sweep_rows: List[dict], stem: str, y_floor: float = 0.25) -> None:
            L = layout_style()
            fs = L["fs"]
            fig, axs = panel_axes(len(models), height=2.1)
            for ax, m in zip(axs, models):
                col = MODEL_COLOR.get(m, "#999999")
                for proto, (ls, mk, _lab) in proto_style.items():
                    rows = sorted([r for r in sweep_rows if r["model"] == m and r["protocol"] == proto], key=lambda r: r["k_misinformed"])
                    if not rows:
                        continue
                    x = [r["k_misinformed"] for r in rows]
                    y = [r["acc"] for r in rows]
                    yerr = [[r["acc"] - r["ci_lo"] for r in rows], [r["ci_hi"] - r["acc"] for r in rows]]
                    ax.errorbar(x, y, yerr=yerr, linestyle=ls, marker=mk, color=col, markersize=L["markersize"],
                                markeredgecolor="#2e2e2e", markeredgewidth=L["edgewidth"], linewidth=L["linewidth"],
                                elinewidth=L["elinewidth"], capsize=L["capsize"], alpha=0.95, zorder=3)
                panel_title(ax, model_label(m), fs)
                ax.set_xticks(ks)
                ax.set_xticklabels([str(k) for k in ks], fontsize=fs["tick"])
                ax.set_ylim(y_floor, 1.02)
                style_axes(ax, "y", fs)
            shared_labels(axs, f"Misinformed agents (of {n_agents})", "Accuracy", fs)
            handles = [series_handle(lab, "#4a4a4a", mk, ls, L["linewidth"], L["markersize"]) for (ls, mk, lab) in proto_style.values()]
            finish_with_legend(fig, handles, [h.get_label() for h in handles], fs, ncol=len(handles))
            out = FIGURES_DIR / stem
            save_figure(fig, out)
            print(f"figure -> {out}.pdf")

        sweep_figure(acc_rows, "fig_composition_sweep")
        sweep_figure(acc_all, "fig_composition_sweep_full_sample", y_floor=0.3)  # axis of the previous Figure 4

        # self-correction (paper Figure 9; full-width figure)
        fs = apply_paper_style()
        if sc_rows:
            fig, axes = plt.subplots(1, len(models), figsize=(15.5, 5.8), sharey=True, squeeze=False)
            xs = sorted({n_agents - r["k_misinformed"] for r in sc_rows})
            for ax, m in zip(axes[0], models):
                col = MODEL_COLOR.get(m, "#999999")
                rows = sorted([r for r in sc_rows if r["model"] == m], key=lambda r: -r["k_misinformed"])
                if not rows:
                    continue
                x = [n_agents - r["k_misinformed"] for r in rows]  # number of uninformed agents
                y = [r["p_adjusted"] for r in rows]
                yerr = [[r["p_adjusted"] - r["ci_lo"] for r in rows], [r["ci_hi"] - r["p_adjusted"] for r in rows]]
                ax.errorbar(x, y, yerr=yerr, linestyle="-", marker="o", color=col, markersize=8, markeredgecolor="#2e2e2e",
                            markeredgewidth=0.6, linewidth=2.8, elinewidth=1.0, capsize=3, alpha=0.95, zorder=3)
                for xi, yi, r in zip(x, y, rows):
                    annotate_value(ax, xi, r["ci_hi"], yi, col, fs, above=True, offset=6)
                    ax.annotate(f"n={r['n_agents_started_wrong']}", (xi, 0.0), xytext=(0, 4), textcoords="offset points",
                                ha="center", va="bottom", fontsize=fs["value"] * 0.85, color="#8a8a8a", zorder=5)
                ax.axvline(n_agents / 2.0, color="#8a8a8a", linewidth=1.0, linestyle=":", zorder=1)
                panel_title(ax, model_label(m), fs)
                ax.set_xticks(xs)
                ax.set_xticklabels([f"{n_agents - u}/{u}" for u in xs], fontsize=fs["tick"])
                ax.set_xlabel("Misinformed / uninformed agents", fontsize=fs["label"])
                ax.set_ylim(0.0, 0.6)
                style_axes(ax, "y", fs)
            axes[0][0].set_ylabel("P(corrected | started wrong)", fontsize=fs["label"])
            handles = [series_handle("misinformed agent, wrong at turn 1, correct at its last turn", "#4a4a4a", "o", "-")]
            finish_with_legend(fig, handles, [h.get_label() for h in handles], fs, ncol=1)
            save_figure(fig, FIGURES_DIR / "fig_self_correction")
            print(f"figure -> {FIGURES_DIR / 'fig_self_correction.pdf'}")
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] figure skipped: {e}")


if __name__ == "__main__":
    main()
