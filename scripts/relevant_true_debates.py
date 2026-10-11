#!/usr/bin/env python3
"""
relevant_true_debates.py - Debates in which 1 or 3 of 3 agents receive the RELEVANT-TRUE passage.

The exposure runner was run with --information_source relevant_true at k = 1 and 3 (seed 0) on WinoGrande
(all 382 questions) and on the verified questions of CWQ and ETHICS (98 / 114). This script
compares those debates with the uninformed debate (k = 0, same seed, same questions)
and with the single agent that reads the relevant-true passage, paired over questions:
    true-context effect(k) = acc(MAD, k relevant-true agents) - acc(MAD, 0/3)
and, for symmetry with the misinformation runs, the same difference for k misinformed agents on the same
questions (full sample, seed 0). If true context were diluted like false context, the k = 1 gain would be a
small fraction of the k = 3 gain.

Inputs:  out/results/rescored/mad_final.csv, out/results/rescored/single_agent.csv
Outputs: out/results/relevant_true_debates.csv, out/tables/relevant_true_debates.tex
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.common import DATASET_DISPLAY, MODEL_ORDER, RESULTS_DIR, TABLES_DIR, ensure_dirs, read_csv, write_csv, write_latex_table  # noqa: E402
from scripts.stats import bootstrap_mean_ci, paired_bootstrap_diff  # noqa: E402


def _b(v) -> float:
    return float(str(v).strip().lower() in ("true", "1", "1.0", "yes"))


def _pq(rows: List[dict]) -> Dict[str, float]:
    acc: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        acc[r["mint_id"]].append(_b(r["is_correct"]))
    return {q: sum(v) / len(v) for q, v in acc.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rescored_dir", default=str(RESULTS_DIR / "rescored"))
    ap.add_argument("--n_boot", type=int, default=10_000)
    args = ap.parse_args()
    ensure_dirs()
    rd = Path(args.rescored_dir)
    mad = [r for r in read_csv(rd / "mad_final.csv") if r["experiment"] == "exp_exposure" and str(r.get("seed")) == "0"
           and str(r.get("num_agents")) == "3" and (r.get("mitigation") or "none") == "none"]
    sa = [r for r in read_csv(rd / "single_agent.csv") if r["experiment"] == "exp1" and str(r.get("seed") or "0") in ("0", "0.0")]
    rows = []
    tex = []
    for model in MODEL_ORDER:
        for ds in sorted({r["dataset"] for r in mad if r.get("information_source") == "relevant_true"}):
            sub = [r for r in mad if r["model"] == model and r["dataset"] == ds]
            if not any(r.get("information_source") == "relevant_true" for r in sub):
                continue
            sa_sub = [r for r in sa if r["model"] == model and r["dataset"] == ds]
            sa_base, sa_rt = _pq([r for r in sa_sub if r["condition"] == "baseline"]), _pq([r for r in sa_sub if r["condition"] == "relevant_true"])
            for proto in ("voting", "consensus"):
                k0 = _pq([r for r in sub if str(r["k_misinformed"]) == "0" and r["protocol"] == proto])
                cells = {"MAD 0/3": k0}
                for k in ("1", "3"):
                    cells[f"MAD {k}/3 relevant-true"] = _pq([r for r in sub if str(r["k_misinformed"]) == k and r["protocol"] == proto
                                                             and r.get("information_source") == "relevant_true"])
                    cells[f"MAD {k}/3 misinformed"] = _pq([r for r in sub if str(r["k_misinformed"]) == k and r["protocol"] == proto
                                                           and r.get("information_source") == "misinformation"
                                                           and r.get("category") == "false_fact"])
                if proto == "voting":
                    cells["SA uninformed"], cells["SA relevant-true"] = sa_base, sa_rt
                for name, pq in cells.items():
                    if not pq:
                        continue
                    ci = bootstrap_mean_ci(list(pq.values()), n_boot=args.n_boot)
                    ref = k0 if name.startswith("MAD") and name != "MAD 0/3" else (sa_base if name == "SA relevant-true" else None)
                    diff = None
                    if ref:
                        qs = sorted(set(pq) & set(ref))
                        if len(qs) >= 10:
                            diff = paired_bootstrap_diff([pq[q] for q in qs], [ref[q] for q in qs], n_boot=args.n_boot)
                    rows.append({"model": model, "dataset": ds, "protocol": proto if name.startswith("MAD") else "single", "system": name,
                                 "acc": ci["mean"], "ci_lo": ci["ci_lo"], "ci_hi": ci["ci_hi"], "n": ci["n"],
                                 "diff_vs_uninformed": diff["diff"] if diff else None, "diff_ci_lo": diff["ci_lo"] if diff else None,
                                 "diff_ci_hi": diff["ci_hi"] if diff else None, "diff_p_perm": diff["p_perm"] if diff else None,
                                 "diff_n": diff["n"] if diff else None})
                    tex.append([model, DATASET_DISPLAY.get(ds, ds), proto if name.startswith("MAD") else "single", name,
                                f"{ci['mean']:.3f} [{ci['ci_lo']:.3f}, {ci['ci_hi']:.3f}]",
                                f"{diff['diff']:+.3f} [{diff['ci_lo']:+.3f}, {diff['ci_hi']:+.3f}]" if diff else "--", ci["n"]])
    write_csv(RESULTS_DIR / "relevant_true_debates.csv", rows)
    write_latex_table(TABLES_DIR / "relevant_true_debates.tex",
                      ["Model", "Dataset", "Protocol", "System", "Accuracy [95\\% CI]", "$\\Delta$ vs. uninformed [95\\% CI]", "n"], tex,
                      caption="Debates with relevant-true context (seed 0; CWQ and ETHICS on the verified questions, WinoGrande "
                              "on all questions). Accuracy of the uninformed "
                              "three-agent debate, of debates in which 1 or 3 agents receive the relevant-true passage, and, on the "
                              "same questions, of debates in which 1 or 3 agents receive the plain false fact; the single agent with and "
                              "without the relevant-true passage for reference. $\\Delta$ is paired over questions against the "
                              "uninformed debate (single agent: against the uninformed single agent); bootstrap 95\\% CIs. The uninformed "
                              "debate and the single agent were run on the full samples, so their n is larger.",
                      label="tab:relevant_true_debates", col_spec="llllllr")
    print(f"wrote {len(rows)} rows")


if __name__ == "__main__":
    main()
