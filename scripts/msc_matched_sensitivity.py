#!/usr/bin/env python3
"""
msc_matched_sensitivity.py - The debate effect against an mSC baseline sampled like the debates.

exp_msc.py sampled its three answers at temperature 1.0 / top-p 1.0, whereas the MALLM debates run at the
server default of the model (Llama-3.3: 0.6 / 0.9; GLM-4.7-Flash and GPT-OSS-120B: 1.0 / 1.0). Only Llama's
debate-versus-mSC comparison is therefore confounded by sampling. The runs under out/<model>/msc_matched
(exp_msc.py --temperature 0.6 --top_p 0.9; seeds 0-2 at k = 1 and 3, seed 0 at k = 0 and 2; re-scored as
experiment exp_msc_matched) repeat mSC at the debate's parameters. On the verified questions of every
model x dataset with such runs this script reports
    mSC-matched(k)             accuracy of the re-sampled baseline (vote: first sample when there is no majority)
    sampling effect(k)         mSC-matched(k) - mSC(k)                 (paired over questions)
    debate effect(k)           MAD(k, protocol) - mSC-matched(k)       (paired over questions)
    MAD 3/3 - SAdp-matched 1/1 fully exposed debate vs. one fully exposed sample at the same temperature
next to the debate effect against the T 1.0 baseline of the main text.

Inputs:  out/results/rescored/{single_agent,mad_final,self_consistency,msc_samples}.csv
Outputs: out/results/msc_matched.csv (cells), out/results/debate_effect_matched.csv (paired differences),
         out/results/exposure_long_matched.csv (per-question rows of the matched systems, read by scripts/stats.py),
         out/tables/debate_effect_matched.tex
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.common import (  # noqa: E402
    DATASET_DISPLAY,
    DATASET_ORDER,
    MODEL_ORDER,
    RESULTS_DIR,
    TABLES_DIR,
    ensure_dirs,
    read_csv,
    verified_counts_text,
    write_csv,
    write_latex_table,
)
from scripts.exposure_curve import collect, verified_question_sets  # noqa: E402
from scripts.stats import bootstrap_mean_ci, holm_bonferroni, paired_bootstrap_diff  # noqa: E402

MATCHED_SUFFIX = "-matched"


def _per_question(long: List[dict], Q: Dict[tuple, set]) -> Dict[tuple, Dict[str, float]]:
    """(model, dataset, system, protocol, k, n) -> {mint_id: mean outcome over seeds}, restricted to the matched set."""
    cells: Dict[tuple, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in long:
        qs = Q.get((r["model"], r["dataset"]))
        if qs is not None and r["mint_id"] not in qs:
            continue
        key = (r["model"], r["dataset"], r["system"], r["protocol"], int(r["k"]), int(r["n"]))
        cells[key][r["mint_id"]].append(float(r["is_correct"]))
    return {key: {q: float(np.mean(v)) for q, v in qmap.items()} for key, qmap in cells.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--category", default="false_fact")
    ap.add_argument("--rescored_dir", default=str(RESULTS_DIR / "rescored"))
    ap.add_argument("--n_boot", type=int, default=10_000)
    args = ap.parse_args()
    ensure_dirs()
    rd = Path(args.rescored_dir)
    sa = read_csv(rd / "single_agent.csv")
    mad = read_csv(rd / "mad_final.csv")
    sc = read_csv(rd / "self_consistency.csv")
    mss = read_csv(rd / "msc_samples.csv") if (rd / "msc_samples.csv").is_file() else []
    sc_m = [dict(r, experiment="exp_msc") for r in sc if r["experiment"] == "exp_msc_matched"]
    mss_m = [dict(r, experiment="exp_msc") for r in mss if r["experiment"] == "exp_msc_matched"]
    if not sc_m:
        print("no exp_msc_matched rows in the rescored CSVs - nothing to do")
        return
    models = sorted({r["model"] for r in sc_m}, key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 9)
    # main systems (SA, MAD, mSC, mSC-strict, SAdp at T 1.0) and the matched mSC systems on the same question sets
    long_main = [r for r in collect(args.category, sa, mad, sc, msc_sample_rows=mss) if r["model"] in models]
    long_m = collect(args.category, [], [], sc_m, msc_sample_rows=mss_m)
    for r in long_m:
        r["system"] = r["system"] + MATCHED_SUFFIX
        r["source"] = "exp_msc_matched"
    write_csv(RESULTS_DIR / "exposure_long_matched.csv", long_m)
    qsets = verified_question_sets(long_main)
    Q = {key: s["matched"] for key, s in qsets.items()}
    pq = _per_question(long_main + long_m, Q)

    # ---- cells
    cell_rows = []
    seeds_of: Dict[tuple, set] = defaultdict(set)
    for r in long_main + long_m:
        seeds_of[(r["model"], r["dataset"], r["system"], r["protocol"], int(r["k"]), int(r["n"]))].add(int(float(r["seed"])))
    for key, vals in sorted(pq.items(), key=lambda kv: (MODEL_ORDER.index(kv[0][0]) if kv[0][0] in MODEL_ORDER else 9,
                                                        DATASET_ORDER.index(kv[0][1]) if kv[0][1] in DATASET_ORDER else 9,
                                                        kv[0][2], kv[0][3], kv[0][5], kv[0][4])):
        model, ds, system, proto, k, n = key
        if not system.endswith(MATCHED_SUFFIX) and system not in ("MAD", "mSC", "mSC-strict", "SAdp"):
            continue
        ci = bootstrap_mean_ci(list(vals.values()), n_boot=args.n_boot)
        cell_rows.append({"model": model, "dataset": ds, "system": system, "protocol": proto, "k": k, "n": n,
                          "acc": ci["mean"], "ci_lo": ci["ci_lo"], "ci_hi": ci["ci_hi"], "n_questions": ci["n"],
                          "n_seeds": len(seeds_of[key])})
    write_csv(RESULTS_DIR / "msc_matched.csv", cell_rows)

    # ---- paired differences
    def pair(a_key, b_key, name, extra):
        a, b = pq.get(a_key), pq.get(b_key)
        if not a or not b:
            return None
        qs = sorted(set(a) & set(b))
        if len(qs) < 10:
            return None
        res = paired_bootstrap_diff([a[q] for q in qs], [b[q] for q in qs], n_boot=args.n_boot)
        return {"comparison": name, **extra, "acc_a": res["mean_a"], "acc_b": res["mean_b"], "diff": res["diff"],
                "ci_lo": res["ci_lo"], "ci_hi": res["ci_hi"], "p_perm": res["p_perm"], "n_questions": res["n"]}

    eff_rows = []
    for model in models:
        for ds in DATASET_ORDER:
            for k in (0, 1, 2, 3):
                base = {"model": model, "dataset": ds, "k": k, "n": 3}
                r = pair((model, ds, "mSC-matched", "vote", k, 3), (model, ds, "mSC", "vote", k, 3),
                         "sampling effect: mSC-matched - mSC", {**base, "protocol": "vote"})
                if r:
                    eff_rows.append(r)
                r = pair((model, ds, "mSC-strict-matched", "vote-strict", k, 3), (model, ds, "mSC-strict", "vote-strict", k, 3),
                         "sampling effect: mSC-strict-matched - mSC-strict", {**base, "protocol": "vote-strict"})
                if r:
                    eff_rows.append(r)
                for proto in ("voting", "consensus"):
                    for base_sys in ("mSC-matched", "mSC"):
                        r = pair((model, ds, "MAD", proto, k, 3), (model, ds, base_sys, "vote", k, 3),
                                 f"debate effect: MAD - {base_sys}", {**base, "protocol": proto})
                        if r:
                            eff_rows.append(r)
                    if k == 3:
                        for sadp in ("SAdp-matched", "SAdp"):
                            r = pair((model, ds, "MAD", proto, 3, 3), (model, ds, sadp, "single", 1, 1),
                                     f"MAD 3/3 - {sadp} 1/1", {**base, "protocol": proto})
                            if r:
                                eff_rows.append(r)
            # one sample at the two temperatures
            for k in (0, 1):
                r = pair((model, ds, "SAdp-matched", "single", k, 1), (model, ds, "SAdp", "single", k, 1),
                         "sampling effect: SAdp-matched - SAdp", {"model": model, "dataset": ds, "k": k, "n": 1, "protocol": "single"})
                if r:
                    eff_rows.append(r)
    fam = [r for r in eff_rows if r["comparison"] == "debate effect: MAD - mSC-matched"]
    for r, p_adj in zip(fam, holm_bonferroni([r["p_perm"] for r in fam])):
        r["p_holm"] = p_adj
    write_csv(RESULTS_DIR / "debate_effect_matched.csv", eff_rows)

    # ---- LaTeX: one row per dataset x protocol x k with both baselines side by side
    def _find(model, ds, proto, k, comp):
        return next((r for r in eff_rows if r["model"] == model and r["dataset"] == ds and r["protocol"] == proto
                     and r["k"] == k and r["comparison"] == comp), None)

    def _ci(r, star=False):
        if r is None:
            return "--"
        mark = ""
        if star:
            mark = "$^{*}$" if r.get("p_holm", 1.0) < 0.05 else ("$^{\\dagger}$" if r["p_perm"] < 0.05 else "")
        return f"{r['diff']:+.2f} [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]{mark}"

    tex = []
    for model in models:
        for ds in DATASET_ORDER:
            for proto in ("voting", "consensus"):
                for k in (0, 1, 2, 3):
                    dm = _find(model, ds, proto, k, "debate effect: MAD - mSC-matched")
                    d1 = _find(model, ds, proto, k, "debate effect: MAD - mSC")
                    sm = _find(model, ds, "vote", k, "sampling effect: mSC-matched - mSC")
                    if dm is None and d1 is None:
                        continue
                    tex.append([DATASET_DISPLAY.get(ds, ds), proto, f"{k}/3", f"{(dm or d1)['acc_a']:.3f}",
                                f"{d1['acc_b']:.3f}" if d1 else "--", f"{dm['acc_b']:.3f}" if dm else "--",
                                _ci(sm), _ci(d1), _ci(dm, star=True), (dm or d1)["n_questions"]])
    write_latex_table(TABLES_DIR / "debate_effect_matched.tex",
                      ["Dataset", "Protocol", "Exp.", "MAD", "mSC T 1.0", "mSC T 0.6",
                       "Sampling effect [95\\% CI]", "Debate effect vs. T 1.0", "Debate effect vs. T 0.6", "n"], tex,
                      caption="Sensitivity of the debate effect to the sampling parameters of the mSC baseline ("
                              + ", ".join(models) + "). The debates run at the "
                              "model's server default (Llama-3.3: temperature 0.6, top-p 0.9), the mSC baseline of the main text at "
                              "temperature 1.0 / top-p 1.0; mSC (T 0.6) repeats mSC at the debate's parameters (seeds 0--2 at 1/3 and "
                              "3/3, seed 0 at 0/3 and 2/3). Sampling effect = mSC (T 0.6) $-$ mSC (T 1.0); debate effect = MAD $-$ mSC, "
                              "each paired over the verified questions (" + verified_counts_text(args.category) + ") with bootstrap "
                              "95\\% CIs. $^{*}$: Holm-adjusted $p < 0.05$ within the matched debate effects; $^{\\dagger}$: unadjusted "
                              "$p < 0.05$.",
                      label="tab:debate_effect_matched", col_spec="lllrrrlllr", font="scriptsize")
    print(f"wrote {len(cell_rows)} cells and {len(eff_rows)} paired differences for {', '.join(models)}")


if __name__ == "__main__":
    main()
