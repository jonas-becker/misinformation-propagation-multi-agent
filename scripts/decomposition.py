#!/usr/bin/env python3
"""
decomposition.py - Decompose the multi-agent benefit into ensembling, interaction
and misinformation cost.

Per model x dataset (category passages, default false_fact):
  ensembling_gain    = SC(uninformed, n) - SA(uninformed)          [mSC k=0 vs exp1 baseline]
  interaction_gain   = MAD(0/n)          - SC(uninformed, n)       [exp_exposure k=0 vs mSC k=0]
  misinfo_cost_MAD(k) = MAD(k/n) - MAD(0/n)
  misinfo_cost_SC(k)  = mSC(k)  - SC(uninformed, n)
  net(k)             = MAD(k/n) - SA(uninformed)  (= ensembling + interaction + misinfo_cost_MAD(k))
All differences are paired over questions with bootstrap CIs. Missing cells are
reported as NA (e.g. before the exposure runs exist, only exp2 k=1 is available).

Also compares the misinformation cost with the effect of relevant-true / filler
passages when those single-agent controls exist, to check whether passages
add incidental task context.

Outputs: out/results/decomposition.csv, out/tables/decomposition.tex,
         out/figures/fig_decomposition.pdf (bridge chart per model x dataset:
         single sample -> +ensembling -> +interaction -> +misinformation cost = debate n/n)
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.common import (DATASET_DISPLAY, DATASET_ORDER, FIGURES_DIR, MODEL_ORDER, RESULTS_DIR, TABLES_DIR, ensure_dirs,  # noqa: E402
                            read_csv, verified_counts_text, verified_questions, write_csv, write_latex_table)
from scripts.exposure_curve import collect  # noqa: E402
from scripts.stats import paired_bootstrap_diff  # noqa: E402


def per_question(long_rows: List[dict], system: str, protocol: Optional[str], k: int, n: int) -> Dict[tuple, Dict[str, float]]:
    out: Dict[tuple, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in long_rows:
        if r["system"] != system or r["k"] != k or r["n"] != n:
            continue
        if protocol and r["protocol"] != protocol:
            continue
        out[(r["model"], r["dataset"])][r["mint_id"]].append(float(r["is_correct"]))
    return {key: {q: float(np.mean(v)) for q, v in qm.items()} for key, qm in out.items()}


def diff(a: Optional[Dict[str, float]], b: Optional[Dict[str, float]], n_boot: int) -> Optional[dict]:
    if not a or not b:
        return None
    qs = sorted(set(a) & set(b))
    if len(qs) < 10:
        return None
    return paired_bootstrap_diff([a[q] for q in qs], [b[q] for q in qs], n_boot=n_boot)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--category", default="false_fact")
    ap.add_argument("--n_agents", type=int, default=3)
    ap.add_argument("--protocol", default="voting")
    ap.add_argument("--rescored_dir", default=str(RESULTS_DIR / "rescored"))
    ap.add_argument("--n_boot", type=int, default=10_000)
    ap.add_argument("--no_figure", action="store_true")
    args = ap.parse_args()
    ensure_dirs()
    rd = Path(args.rescored_dir)
    sa = read_csv(rd / "single_agent.csv") if (rd / "single_agent.csv").is_file() else []
    mad = read_csv(rd / "mad_final.csv") if (rd / "mad_final.csv").is_file() else []
    sc = read_csv(rd / "self_consistency.csv") if (rd / "self_consistency.csv").is_file() else []
    mss = read_csv(rd / "msc_samples.csv") if (rd / "msc_samples.csv").is_file() else []
    long = collect(args.category, sa, mad, sc, msc_sample_rows=mss)
    n = args.n_agents

    # every exposed cell (k >= 1 debates, mSC samples with a passage) is restricted to the verified questions
    # of its dataset (scripts.common.verification_criterion); uninformed cells keep the full sample
    def _verified(cells: Dict[tuple, Dict[str, float]]) -> Dict[tuple, Dict[str, float]]:
        out = {}
        for key, qm in cells.items():
            V = verified_questions(key[1], args.category)
            sub = {q: v for q, v in qm.items() if q in V}
            if sub:
                out[key] = sub
        return out

    SA0 = per_question(long, "SA", "single", 0, 1)
    SA1 = _verified(per_question(long, "SA", "single", 1, 1))
    # single agent under the debate prompt (individual exp_msc samples): the prompt-matched reference
    SAdp0 = per_question(long, "SAdp", "single", 0, 1)
    SAdp1 = _verified(per_question(long, "SAdp", "single", 1, 1))
    # SC0 = mSC with no passage (tie -> first sample, the paper's rule); SC0_strict counts ties as wrong
    SC0 = per_question(long, "mSC", "vote", 0, n)
    SC0_strict = per_question(long, "mSC-strict", "vote-strict", 0, n)
    MAD = {k: (per_question(long, "MAD", args.protocol, k, n) if k == 0 else _verified(per_question(long, "MAD", args.protocol, k, n)))
           for k in range(0, n + 1)}
    MSC = {k: (SC0 if k == 0 else _verified(per_question(long, "mSC", "vote", k, n))) for k in range(0, n + 1)}
    MSC_strict = {k: (SC0_strict if k == 0 else _verified(per_question(long, "mSC-strict", "vote-strict", k, n)))
                  for k in range(0, n + 1)}

    # single-agent controls for the incidental-context check
    ctrl: Dict[str, Dict[tuple, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for r in sa:
        # seed 0 of the exp1 run; irrelevant_true from the same run (the exp1a
        # irrelevant_true_information run is used only where the exp1 condition is missing)
        if r["experiment"] == "exp1" and str(r.get("seed") or "0") in ("0", "0.0") and \
                r["condition"] in ("relevant_true", "filler", "irrelevant_true"):
            name = "irrelevant_true_information" if r["condition"] == "irrelevant_true" else r["condition"]
            ctrl[name][(r["model"], r["dataset"])][r["mint_id"]] = float(str(r["is_correct"]).lower() in ("true", "1"))
    have_it = {key for key, v in ctrl["irrelevant_true_information"].items() if v}
    for r in sa:
        if r["experiment"] == "exp1a" and r["condition"] == "irrelevant_true_information" and \
                (r["model"], r["dataset"]) not in have_it:
            ctrl[r["condition"]][(r["model"], r["dataset"])][r["mint_id"]] = float(str(r["is_correct"]).lower() in ("true", "1"))

    rows = []
    keys = sorted(set(SA0) | set(MAD[0]) | set(MAD[1]), key=lambda x: (MODEL_ORDER.index(x[0]) if x[0] in MODEL_ORDER else 9, x[1]))
    for key in keys:
        model, ds = key

        def add(component: str, a, b, k=None):
            res = diff(a, b, args.n_boot)
            rows.append({"model": model, "dataset": ds, "component": component, "k": k, "n": n, "protocol": args.protocol,
                         "value": res["diff"] if res else None, "ci_lo": res["ci_lo"] if res else None,
                         "ci_hi": res["ci_hi"] if res else None, "p_perm": res["p_perm"] if res else None,
                         "n_questions": res["n"] if res else 0, "acc_a": res["mean_a"] if res else None,
                         "acc_b": res["mean_b"] if res else None})

        # full-sample components (k = 0 cells): SA0 = single agent with its own prompt, SAdp0 = single sample
        # under the debate prompt, SC0 = mSC without passage (tie -> first sample), *_strict = ties count as wrong
        add("ensembling_gain (SC0 - SAdp0)", SC0.get(key), SAdp0.get(key))
        add("ensembling_gain_strict (SC0-strict - SAdp0)", SC0_strict.get(key), SAdp0.get(key))
        add("ensembling_gain_own_prompt (SC0 - SA0)", SC0.get(key), SA0.get(key))
        add("interaction_gain (MAD0 - SC0)", MAD[0].get(key), SC0.get(key))
        add("interaction_gain_strict (MAD0 - SC0-strict)", MAD[0].get(key), SC0_strict.get(key))
        # additive chain on ONE question set (the verified questions of the k = n runs):
        #   single sample -> +ensembling -> +interaction -> +misinformation cost = fully exposed debate
        Q = set(MAD[n].get(key, {}))

        def _sub(d):
            return {q: v for q, v in (d or {}).items() if q in Q} if Q else None

        add("ensembling_gain_matched (SC0 - SAdp0, verified questions)", _sub(SC0.get(key)), _sub(SAdp0.get(key)))
        add("interaction_gain_matched (MAD0 - SC0, verified questions)", _sub(MAD[0].get(key)), _sub(SC0.get(key)))
        add(f"net_matched (MAD{n} - SAdp0, verified questions)", MAD[n].get(key), _sub(SAdp0.get(key)), n)
        for k in range(1, n + 1):
            add(f"misinfo_cost_MAD (MAD{k} - MAD0)", MAD[k].get(key), MAD[0].get(key), k)
            add(f"misinfo_cost_SC (mSC{k} - SC0)", MSC[k].get(key), SC0.get(key), k)
            add(f"misinfo_cost_SC_strict (mSC-strict{k} - SC0-strict)", MSC_strict[k].get(key), SC0_strict.get(key), k)
            add(f"net (MAD{k} - SA0)", MAD[k].get(key), SA0.get(key), k)
            add(f"net_dp (MAD{k} - SAdp0)", MAD[k].get(key), SAdp0.get(key), k)
            add(f"debate_effect (MAD{k} - mSC{k})", MAD[k].get(key), MSC[k].get(key), k)
            add(f"debate_effect_strict (MAD{k} - mSC-strict{k})", MAD[k].get(key), MSC_strict[k].get(key), k)
        add("SA misinfo cost (SA1 - SA0)", SA1.get(key), SA0.get(key), 1)
        add("SA misinfo cost dp (SAdp1 - SAdp0)", SAdp1.get(key), SAdp0.get(key), 1)
        add("fully exposed MAD vs single sample dp (MAD3 - SAdp1)", MAD[n].get(key), SAdp1.get(key), n)
        for cond in ("relevant_true", "irrelevant_true_information", "filler"):
            add(f"SA context effect ({cond} - SA0)", ctrl[cond].get(key), SA0.get(key))

    write_csv(RESULTS_DIR / "decomposition.csv", rows)
    tex = [[r["model"], DATASET_DISPLAY.get(r["dataset"], r["dataset"]), r["component"],
            f"{r['value']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]" if r["value"] is not None else "NA (run pending)",
            r["n_questions"]] for r in rows]
    write_latex_table(TABLES_DIR / "decomposition.tex", ["Model", "Dataset", "Component", "$\\Delta$ accuracy [95\\% CI]", "n"], tex,
                      caption=f"Decomposition of the multi-agent benefit ({args.protocol}, {args.category} passages, n={n}). "
                              "SA0 = uninformed single agent (own prompt), SAdp0/SAdp1 = single sample under the debate prompt "
                              "without/with the passage, SC0 = mSC without passage, MADk / mSCk = debate / mSC with k exposed agents "
                              "or samples; mSC falls back to the first sample when there is no majority, the *\\_strict variants count "
                              "such questions as wrong. Exposed cells use the verified questions "
                              f"({verified_counts_text(args.category)}); rows without exposed cells use the full samples unless marked "
                              "'verified questions'. Paired bootstrap CIs over "
                              "questions.", label="tab:decomposition", col_spec="lllrr")
    print(f"wrote {len(rows)} decomposition rows; available: {sum(1 for r in rows if r['value'] is not None)}")

    if args.no_figure:
        return
    avail = [r for r in rows if r["value"] is not None and r["component"].split(" ")[0] in
             ("ensembling_gain_matched", "interaction_gain_matched", "net_matched")]
    if not avail:
        print("no complete decomposition yet; figure skipped")
        return
    try:
        import matplotlib.pyplot as plt

        from scripts.figure_style import get_palette
        from scripts.figure_common import (EDGE, SYSTEM_COLOR, apply_paper_style, dataset_label, finish_with_legend,
                                           model_label, panel_title, patch_handle, save_figure, style_axes)

        fs = apply_paper_style()
        pal6 = get_palette(6)  # purple, blue, sage, rose, grey, gold
        col_start, col_end = SYSTEM_COLOR["single agent"], SYSTEM_COLOR["MAD voting"]
        col_gain, col_loss = pal6[2], pal6[3]
        label_col = "#4a4a4a"

        def pick(m, ds, prefix, k=None):
            cand = [r for r in rows if r["value"] is not None and r["model"] == m and r["dataset"] == ds
                    and r["component"].startswith(prefix) and (k is None or r["k"] == k)]
            return cand[0] if cand else None

        # bridge chart per model x dataset on ONE question set (verified questions):
        #   single uninformed sample -> +ensembling -> +interaction -> +misinformation cost = debate with n/n misinformed
        cells = {}
        for m in MODEL_ORDER:
            for ds in DATASET_ORDER:
                ens = pick(m, ds, "ensembling_gain_matched")
                inter = pick(m, ds, "interaction_gain_matched")
                cost = pick(m, ds, "misinfo_cost_MAD", n)
                net = pick(m, ds, "net_matched", n)
                if not all((ens, inter, cost, net)):
                    continue
                start, end = ens["acc_b"], net["acc_a"]
                if abs(start + ens["value"] + inter["value"] + cost["value"] - end) > 1e-6:
                    print(f"[WARN] {m}/{ds}: components do not add up to the fully exposed debate accuracy")
                cells[(m, ds)] = {"start": start, "end": end, "steps": [ens, inter, cost], "n": net["n_questions"]}
        models = [m for m in MODEL_ORDER if any(k[0] == m for k in cells)]
        datasets = [d for d in DATASET_ORDER if any(k[1] == d for k in cells)]
        if not models or not datasets:
            print("no complete decomposition yet; figure skipped")
            return

        fig, axes = plt.subplots(len(models), len(datasets), figsize=(15.5, 3.9 * len(models) + 0.8),
                                 sharex=True, sharey=True, squeeze=False)
        width = 0.62
        step_names = ["Single sample", "+ Ensembling", "+ Interaction", "+ Misinformation cost",
                      f"= Debate, {n}/{n} misinformed"]

        def label(ax, x, y, text, above=True):
            ax.annotate(text, (x, y), xytext=(0, 3 if above else -3), textcoords="offset points", ha="center",
                        va="bottom" if above else "top", fontsize=fs["value"], color=label_col, zorder=5, linespacing=1.1)

        for ri, m in enumerate(models):
            for ci_, ds in enumerate(datasets):
                ax = axes[ri][ci_]
                style_axes(ax, "y", fs)
                if ri == 0:
                    ns = sorted({cells[(mm, ds)]["n"] for mm in models if (mm, ds) in cells})
                    panel_title(ax, f"{dataset_label(ds)} (n = {ns[0]})" if len(ns) == 1 else dataset_label(ds), fs)
                if ci_ == 0:
                    ax.set_ylabel("Accuracy", fontsize=fs["label"])
                    ax.text(-0.27, 0.5, model_label(m), transform=ax.transAxes, rotation=90, ha="center", va="center",
                            fontweight="bold", fontsize=fs["panel_title"])
                cell = cells.get((m, ds))
                if cell is None:
                    ax.text(0.5, 0.5, "run pending", transform=ax.transAxes, ha="center", va="center",
                            fontsize=fs["tick"], color=label_col)
                    continue
                level = cell["start"]
                ax.bar(0, level, width=width, color=col_start, edgecolor="white", linewidth=0.5, zorder=2)
                label(ax, 0, level, f"{level:.2f}")
                for j, r in enumerate(cell["steps"], start=1):
                    d = r["value"]
                    new = level + d
                    # running total carried over from the previous bar
                    ax.plot([j - 1 + width / 2, j - width / 2], [level, level], color=EDGE, linewidth=0.9,
                            linestyle="--", zorder=3)
                    ax.bar(j, d, bottom=level, width=width, color=col_gain if d >= 0 else col_loss, edgecolor="white",
                           linewidth=0.5, zorder=2)
                    ax.errorbar(j, new, yerr=[[d - r["ci_lo"]], [r["ci_hi"] - d]], color=EDGE, linewidth=0.9,
                                capsize=2.5, zorder=3)
                    if d >= 0:
                        label(ax, j, max(new, level + r["ci_hi"]), f"{d:+.2f}", above=True)
                    else:
                        label(ax, j, min(new, level + r["ci_lo"]), f"{d:+.2f}", above=False)
                    level = new
                ax.plot([3 + width / 2, 4 - width / 2], [level, level], color=EDGE, linewidth=0.9, linestyle="--",
                        zorder=3)
                ax.bar(4, cell["end"], width=width, color=col_end, edgecolor="white", linewidth=0.5, zorder=2)
                label(ax, 4, cell["end"], f"{cell['end']:.2f}\nnet {cell['end'] - cell['start']:+.2f}")
                ax.set_xticks(range(5))
        for ax in axes[-1]:
            ax.set_xticklabels(step_names, fontsize=fs["tick"], rotation=30, ha="right", rotation_mode="anchor")
        for row in axes:
            for ax in row:
                ax.set_ylim(0, 1.18)
                ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        handles = [patch_handle("Single uninformed sample", col_start), patch_handle("Gain", col_gain),
                   patch_handle("Loss", col_loss), patch_handle(f"Debate with {n}/{n} misinformed agents", col_end)]
        finish_with_legend(fig, handles, [h.get_label() for h in handles], fs, ncol=4, top=0.94)
        save_figure(fig, FIGURES_DIR / "fig_decomposition")
        print(f"figure -> {FIGURES_DIR / 'fig_decomposition.pdf'}")
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] figure skipped: {e}")


if __name__ == "__main__":
    main()
