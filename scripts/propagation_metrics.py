#!/usr/bin/env python3
"""
propagation_metrics.py - Causal-style propagation metrics from debate transcripts.

Measures that condition on the receiver's prior answer and correct for chance (instead of a
'persistence' metric that only asks whether the next agent repeats the answer):

  adoption(t->t+1)      P(receiver answer at t+1 == source answer at t | receiver differed from source at t)
                        source = misinformed agent; receivers = uninformed agents (k >= 1 debates)
  peer_adoption         same statistic with an UNINFORMED peer as source (within-debate control)
  baseline_k0           same statistic in k = 0 debates (no misinformation anywhere), one pseudo-source per debate
  excess_adoption       adoption - peer_adoption (paired cluster bootstrap over debates)
                        adoption - baseline_k0    (independent cluster bootstrap)
  dissent_rate          share of receiver-source pairs that disagree at t (how often the source is a dissenter)
  flips_per_debate      receiver switches to the source's answer per debate = dissent x adoption (what reaches the group)
  kappa_agreement       chance-corrected agreement between receiver(t+1) and source(t) (binary tasks)
  injection_success     P(misinformed agent's answer at t == intended wrong answer)
  wrong_share           mean share of agents holding the intended wrong answer at each turn
                        (misinformed vs uninformed agents)
  transitions           state transition matrices per turn (correct / wrong / unparsed)
  persistence           the original metric, kept for reference

Run sets. "verified": exp_exposure debates with the plain false fact on the MINT-verified questions
(scripts.common.verification_criterion; the WinoGrande runs cover all questions and are restricted
here), three seeds at k = 1 and 3. "full-sample": the exp2 debates (all nine passage
categories, all questions, misinformed agent always Participant 3), reported as such and with three
restrictions that diagnose the difference between the run sets: plain false fact only, verified
passage only, both. For the verified run set the adoption from the misinformed agent is also split
by the slot the agent speaks in (first / second / last). Relevant-true debates (WinoGrande, k = 1,
full sample) give the adoption of a correctly informed agent's answer as a control.

All statistics are reported per turn transition (1->2, 2->3, ...) and both for all debates and
restricted to debates where injection succeeded at t (the source holds the passage's answer).

Input: out/results/rescored/mad_turns.csv (from scripts/rescore_results.py). Output: out/results/propagation_by_turn.csv,
out/results/propagation_transitions.csv, out/results/injection_success.csv, out/results/wrong_answer_share_by_turn.csv,
out/tables/propagation_by_turn.tex (adoption vs. peer, both run sets), propagation_rates.tex (dissent,
flips per debate, k = 0 switch rate, kappa), propagation_fullsample_diagnostics.tex, propagation_relevant_true.tex
and the per-turn share figure fig_wrong_share_per_turn.pdf.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.common import (  # noqa: E402
    DATASET_DISPLAY,
    DATASET_ORDER,
    FIGURES_DIR,
    MODEL_ORDER,
    RESULTS_DIR,
    TABLES_DIR,
    ensure_dirs,
    read_csv,
    verified_counts_text,
    verified_questions,
    write_csv,
    write_latex_table,
)

N_BOOT = 10_000
SEED = 7
RUN_SET = {"exp_exposure": "verified", "exp2": "full-sample"}


def _b(v) -> Optional[bool]:
    if v in (None, "", "None", "nan"):
        return None
    return str(v).strip().lower() in ("true", "1", "1.0", "yes")


def _state(row: dict) -> str:
    if row["status"] != "parsed":
        return "unparsed"
    if _b(row["is_correct"]):
        return "correct"
    return "wrong"


def _value(row: dict):
    """Comparable answer value: option index (MC) or normalized text (free-form). The same rule is applied to
    every run set (all debates are re-scored with the same extractor), so agreement between agents
    means the same thing in the verified and the full-sample rows."""
    if row["status"] != "parsed":
        return None
    if row["predicted_index"] not in ("", None):
        return int(float(row["predicted_index"]))
    return (row["predicted_text"] or "").strip().lower() or None


def load_debates(paths: List[Path], experiments: Optional[List[str]] = None) -> Dict[tuple, dict]:
    """Group turn rows into debates: key -> {meta, answers[turn][agent_index] -> row, mis}.

    `mis` holds the exposed slots (the misinformed agents, or the correctly informed agent in relevant-true
    debates, which the runner flags in the same field)."""
    debates: Dict[tuple, dict] = {}
    for p in paths:
        for r in read_csv(p):
            if experiments and r["experiment"] not in experiments:
                continue
            if r["experiment"] == "exp_mitigation_verifier":
                continue
            src = r.get("information_source") or "misinformation"
            if r["experiment"] == "exp_exposure":
                if src not in ("misinformation", "none", "relevant_true"):
                    continue
                if (r.get("mitigation") or "none") != "none":  # the source-skepticism debates are a mitigation, not propagation
                    continue
            else:
                src = "misinformation"
            key = (r["experiment"], src, r["model"], r["dataset"], r["condition"], r["mint_id"])
            d = debates.setdefault(key, {"meta": {"experiment": r["experiment"], "source": src, "model": r["model"],
                                                   "dataset": r["dataset"], "condition": r["condition"],
                                                   "mint_id": r["mint_id"],
                                                   "category": r.get("category"),
                                                   "k": int(float(r["k_misinformed"])) if r["k_misinformed"] not in ("", None) else None,
                                                   "n": int(float(r["num_agents"])) if r["num_agents"] not in ("", None) else None},
                                          "answers": defaultdict(dict), "mis": set()})
            try:
                t = int(float(r["turn"]))
                a = int(float(r["agent_index"]))
            except Exception:
                continue
            d["answers"][t][a] = r
            if _b(r["is_misinformed"]):
                d["mis"].add(a)
    return debates


def ratio_boot(num: np.ndarray, den: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> Tuple[float, float, float]:
    """Cluster (debate-level) bootstrap of a ratio of sums."""
    tot = den.sum()
    if tot <= 0:
        return float("nan"), float("nan"), float("nan")
    est = num.sum() / tot
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(n_boot, len(num)))
    bn = num[idx].sum(axis=1)
    bd = den[idx].sum(axis=1)
    ok = bd > 0
    boots = bn[ok] / bd[ok]
    if len(boots) == 0:
        return est, float("nan"), float("nan")
    return est, float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def paired_ratio_diff_boot(num_a, den_a, num_b, den_b, n_boot: int = N_BOOT, seed: int = SEED):
    """Paired cluster bootstrap of (sum num_a/sum den_a) - (sum num_b/sum den_b) on the same debates."""
    if den_a.sum() <= 0 or den_b.sum() <= 0:
        return float("nan"), float("nan"), float("nan")
    est = num_a.sum() / den_a.sum() - num_b.sum() / den_b.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num_a), size=(n_boot, len(num_a)))
    a = num_a[idx].sum(axis=1) / np.maximum(den_a[idx].sum(axis=1), 1e-9)
    b = num_b[idx].sum(axis=1) / np.maximum(den_b[idx].sum(axis=1), 1e-9)
    boots = a - b
    return est, float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def indep_ratio_diff_boot(num_a, den_a, num_b, den_b, n_boot: int = N_BOOT, seed: int = SEED):
    if den_a.sum() <= 0 or den_b.sum() <= 0:
        return float("nan"), float("nan"), float("nan")
    est = num_a.sum() / den_a.sum() - num_b.sum() / den_b.sum()
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, len(num_a), size=(n_boot, len(num_a)))
    ib = rng.integers(0, len(num_b), size=(n_boot, len(num_b)))
    a = num_a[ia].sum(axis=1) / np.maximum(den_a[ia].sum(axis=1), 1e-9)
    b = num_b[ib].sum(axis=1) / np.maximum(den_b[ib].sum(axis=1), 1e-9)
    boots = a - b
    return est, float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


STRATA = ("all", "source_minority", "source_majority")


def _source_is_minority(d: dict, t: int, s: int, vs) -> Optional[bool]:
    """Is the source's answer held by fewer agents than the most common other answer at turn t?"""
    vals = [_value(r) for a, r in d["answers"].get(t, {}).items()]
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return None
    n_src = sum(1 for v in vals if v == vs)
    others = Counter(v for v in vals if v != vs)
    n_alt = max(others.values()) if others else 0
    if n_src == n_alt:
        return None  # tie: excluded from both strata
    return n_src < n_alt


def _source_holds_passage_answer(d: dict, rs: dict) -> bool:
    """Injection success for the source: it states the passage's answer (the intended wrong answer for
    misinformation, the gold answer for relevant-true passages)."""
    if d["meta"]["source"] == "relevant_true":
        return bool(_b(rs.get("is_correct")))
    return bool(_b(rs.get("is_intended_wrong")))


def adoption_counts(d: dict, t: int, sources: List[int], receivers: List[int], require_injection: bool) -> dict:
    """Adoption counts for transition t -> t+1.

    Returns {"num": {stratum: int}, "den": {stratum: int}, "agree": int, "pairs": int, "injection_ok": int}
    where num/den count receiver transitions in which the receiver disagreed with the
    source at t (den) and holds the source's t-answer at t+1 (num); pairs counts every receiver-source
    pair with parsed answers at t and t+1 (den / pairs = dissent rate). Strata split by whether the
    source's answer was in the minority among all agents at t.
    """
    ans_t = d["answers"].get(t, {})
    ans_t1 = d["answers"].get(t + 1, {})
    num = {s_: 0 for s_ in STRATA}
    den = {s_: 0 for s_ in STRATA}
    agree = pairs = 0
    inj_ok = 0
    for s in sources:
        rs = ans_t.get(s)
        if rs is None:
            continue
        vs = _value(rs)
        if vs is None:
            continue
        if require_injection:
            if not _source_holds_passage_answer(d, rs):
                continue
            inj_ok = 1
        minority = _source_is_minority(d, t, s, vs)
        for r in receivers:
            if r == s:
                continue
            rr_t = ans_t.get(r)
            rr_t1 = ans_t1.get(r)
            if rr_t is None or rr_t1 is None:
                continue
            v_t, v_t1 = _value(rr_t), _value(rr_t1)
            if v_t is None or v_t1 is None:
                continue
            pairs += 1
            agree += int(v_t1 == vs)
            if v_t != vs:
                den["all"] += 1
                num["all"] += int(v_t1 == vs)
                if minority is True:
                    den["source_minority"] += 1
                    num["source_minority"] += int(v_t1 == vs)
                elif minority is False:
                    den["source_majority"] += 1
                    num["source_majority"] += int(v_t1 == vs)
    return {"num": num, "den": den, "agree": agree, "pairs": pairs, "injection_ok": inj_ok}


def _stable_seed(*parts: object) -> int:
    """Process-independent seed (Python's hash() of strings changes with every interpreter start)."""
    return int(hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:8], 16)


def designated_peer(d: dict, uninformed: List[int]) -> Optional[int]:
    """One uninformed agent per debate acts as the control source (seeded by the question id)."""
    if not uninformed:
        return None
    rng = np.random.default_rng(_stable_seed("peer", d["meta"]["mint_id"]))
    return int(uninformed[rng.integers(0, len(uninformed))])


def designated_k0_source(d: dict, n: int, fixed_position: Optional[int]) -> int:
    """Pseudo-source in k=0 debates: the position the misinformed agent occupies in the
    k>=1 debates of the same group when that position is constant (exp2: 2), else random."""
    if fixed_position is not None and 0 <= fixed_position < n:
        return fixed_position
    rng = np.random.default_rng(_stable_seed("k0", d["meta"]["mint_id"]))
    return int(rng.integers(0, n))


def compute_adoption(meta: dict, ds_debates: List[dict], k0: List[dict], n_boot: int, subset: str = "all") -> List[dict]:
    """Adoption / peer / k0 / dissent / flips / kappa rows for one group of k >= 1 debates (one run set, model,
    dataset, k, n and optional restriction `subset`)."""
    exp, src, model, ds, k, n = meta["experiment"], meta["source"], meta["model"], meta["dataset"], meta["k"], meta["n"]
    turns = sorted({t for d in ds_debates for t in d["answers"]})
    max_t = max(turns) if turns else 0
    mis_positions = {p for d in ds_debates for p in d["mis"]}
    fixed_pos = next(iter(mis_positions)) if len(mis_positions) == 1 else None
    rows: List[dict] = []

    def row(t, restrict, metric, est, lo, hi, n_trans, n_deb):
        return {"run_set": RUN_SET.get(exp, exp), "experiment": exp, "information_source": src, "subset": subset, "model": model,
                "dataset": ds, "k": k, "n": n, "turn_from": t, "turn_to": t + 1, "restrict": restrict, "metric": metric,
                "value": est, "ci_lo": lo, "ci_hi": hi, "n_transitions": n_trans, "n_debates": n_deb}

    for t in range(1, max_t):
        for restrict in ("all", "injection_success"):
            req = restrict == "injection_success"
            A = {s_: ([], []) for s_ in STRATA}
            P = {s_: ([], []) for s_ in STRATA}
            agree_l, pairs_l, peer_pairs_l, kept = [], [], [], 0
            for d in ds_debates:
                mis = sorted(d["mis"])
                if not mis:
                    continue
                uninformed = [a for a in range(n) if a not in d["mis"]]
                ac = adoption_counts(d, t, mis, uninformed, req)
                if req and not ac["injection_ok"]:
                    continue
                kept += 1
                # within-debate control: ONE designated uninformed peer as source, other uninformed as receivers
                peer = designated_peer(d, uninformed)
                pc = adoption_counts(d, t, [peer], [r for r in uninformed if r != peer], False) if peer is not None \
                    else {"num": {s_: 0 for s_ in STRATA}, "den": {s_: 0 for s_ in STRATA}, "pairs": 0}
                for s_ in STRATA:
                    A[s_][0].append(ac["num"][s_]); A[s_][1].append(ac["den"][s_])
                    P[s_][0].append(pc["num"][s_]); P[s_][1].append(pc["den"][s_])
                agree_l.append(ac["agree"]); pairs_l.append(ac["pairs"]); peer_pairs_l.append(pc["pairs"])
            if not A["all"][1]:
                continue
            for s_ in STRATA:
                A_num, A_den = np.asarray(A[s_][0], float), np.asarray(A[s_][1], float)
                P_num, P_den = np.asarray(P[s_][0], float), np.asarray(P[s_][1], float)
                suffix = "" if s_ == "all" else f"__{s_}"
                est, lo, hi = ratio_boot(A_num, A_den, n_boot)
                rows.append(row(t, restrict, f"adoption_from_misinformed{suffix}", est, lo, hi, int(A_den.sum()), kept))
                est, lo, hi = ratio_boot(P_num, P_den, n_boot)
                rows.append(row(t, restrict, f"adoption_from_uninformed_peer{suffix}", est, lo, hi, int(P_den.sum()), kept))
                est, lo, hi = paired_ratio_diff_boot(A_num, A_den, P_num, P_den, n_boot)
                rows.append(row(t, restrict, f"excess_adoption_vs_peer{suffix}", est, lo, hi, int(A_den.sum()), kept))
            A_num, A_den = np.asarray(A["all"][0], float), np.asarray(A["all"][1], float)
            P_num, P_den = np.asarray(P["all"][0], float), np.asarray(P["all"][1], float)
            pairs_a, pairs_p = np.asarray(pairs_l, float), np.asarray(peer_pairs_l, float)
            ones = np.ones(len(A_num))
            # how often the source dissents (receiver differs from it at t) and how many receiver switches toward
            # the source's answer that produces per debate (= dissent x adoption, what actually reaches the group)
            est, lo, hi = ratio_boot(A_den, pairs_a, n_boot)
            rows.append(row(t, restrict, "dissent_rate_misinformed", est, lo, hi, int(pairs_a.sum()), kept))
            est, lo, hi = ratio_boot(P_den, pairs_p, n_boot)
            rows.append(row(t, restrict, "dissent_rate_uninformed_peer", est, lo, hi, int(pairs_p.sum()), kept))
            est, lo, hi = ratio_boot(A_num, ones, n_boot)
            rows.append(row(t, restrict, "flips_per_debate_misinformed", est, lo, hi, int(A_num.sum()), kept))
            est, lo, hi = ratio_boot(P_num, ones, n_boot)
            rows.append(row(t, restrict, "flips_per_debate_uninformed_peer", est, lo, hi, int(P_num.sum()), kept))
            # chance-corrected agreement (binary tasks): observed - expected from marginals
            if ds in ("winogrande", "ethics") and sum(pairs_l) > 0:
                obs = sum(agree_l) / sum(pairs_l)
                src_vals, rec_vals = [], []
                for d in ds_debates:
                    for s in d["mis"]:
                        rs = d["answers"].get(t, {}).get(s)
                        vs = _value(rs) if rs else None
                        if vs is None or (req and not _source_holds_passage_answer(d, rs)):
                            continue
                        for r in range(n):
                            if r in d["mis"]:
                                continue
                            rr = d["answers"].get(t + 1, {}).get(r)
                            rr0 = d["answers"].get(t, {}).get(r)
                            if rr is None or rr0 is None or _value(rr) is None or _value(rr0) is None:
                                continue
                            src_vals.append(vs); rec_vals.append(_value(rr))
                if src_vals:
                    ps = Counter(src_vals); pr = Counter(rec_vals); N = len(src_vals)
                    pe = sum((ps[v] / N) * (pr.get(v, 0) / N) for v in ps)
                    kappa = (obs - pe) / (1 - pe) if pe < 1 else float("nan")
                    rows.append(row(t, restrict, "kappa_agreement_receiver_t1_vs_misinformed_t", kappa, float("nan"), float("nan"), N, kept))
            # baseline from k=0 debates: one designated pseudo-source per debate (full sample of the dataset)
            if k0 and restrict == "all":
                B_num, B_den = [], []
                for d in k0:
                    s = designated_k0_source(d, n, fixed_pos)
                    bc = adoption_counts(d, t, [s], [r for r in range(n) if r != s], False)
                    B_num.append(bc["num"]["all"]); B_den.append(bc["den"]["all"])
                B_num, B_den = np.asarray(B_num, float), np.asarray(B_den, float)
                est, lo, hi = ratio_boot(B_num, B_den, n_boot)
                rows.append(row(t, "all", "baseline_switch_rate_k0", est, lo, hi, int(B_den.sum()), len(k0)))
                est, lo, hi = indep_ratio_diff_boot(A_num, A_den, B_num, B_den, n_boot)
                rows.append(row(t, "all", "excess_adoption_vs_k0", est, lo, hi, int(A_den.sum()), kept))
    return rows


def write_tables(out_rows: List[dict], min_transitions: int = 10) -> None:
    """LaTeX tables from the metric rows (also used by --tables_only, which reloads out/results/propagation_by_turn.csv)."""
    def _fmt(r_, signed=False):
        if r_ is None or any(x != x for x in (r_["value"], r_["ci_lo"], r_["ci_hi"])):
            return "--"
        f = "{:+.2f}" if signed else "{:.2f}"
        return f"{f.format(r_['value'])} [{f.format(r_['ci_lo'])}, {f.format(r_['ci_hi'])}]"

    def _get(rows, **kw):
        return next((r for r in rows if all(r.get(a) == b for a, b in kw.items())), None)

    def _order(rows_keys):
        return sorted(rows_keys, key=lambda x: (0 if x[0] == "verified" else 1, MODEL_ORDER.index(x[1]) if x[1] in MODEL_ORDER else 9,
                                                DATASET_ORDER.index(x[2]) if x[2] in DATASET_ORDER else 9, x[3]))

    # Table: adoption vs. uninformed peer, both run sets, 1-of-3 debates, injection success (the paper's Table 12)
    base = [r for r in out_rows if r["information_source"] == "misinformation" and r["subset"] == "all" and r["k"] == 1
            and r["n"] == 3 and r["restrict"] == "injection_success" and r["experiment"] in ("exp_exposure", "exp2")
            and r["run_set"] == "verified"]  # full-sample rows: see propagation_fullsample_diagnostics.tex
    table_rows = []
    for rs, model, ds, t in _order({(r["run_set"], r["model"], r["dataset"], r["turn_from"]) for r in base}):
        sel = [r for r in base if r["run_set"] == rs and r["model"] == model and r["dataset"] == ds and r["turn_from"] == t]
        a, p, e = _get(sel, metric="adoption_from_misinformed"), _get(sel, metric="adoption_from_uninformed_peer"), \
            _get(sel, metric="excess_adoption_vs_peer")
        if not (a and p and e) or a["n_transitions"] < min_transitions:
            continue
        table_rows.append([rs, model, DATASET_DISPLAY.get(ds, ds), f"{t}$\\to${t + 1}", _fmt(a), _fmt(p), _fmt(e, signed=True),
                           a["n_transitions"], a["n_debates"]])
    write_latex_table(TABLES_DIR / "propagation_by_turn.tex",
                      ["Run set", "Model", "Dataset", "Turn", "Adoption (misinf. source)", "Adoption (uninf. peer)",
                       "Excess", "n trans.", "n debates"], table_rows,
                      caption="Adoption rate: probability that an uninformed agent switches to the source's answer "
                              "from turn $t$ to $t{+}1$, given that it disagreed with the source at $t$; source = the "
                              "misinformed agent vs. one designated uninformed peer in the same debates (1-of-3 debates, "
                              "restricted to debates in which the misinformed agent held the intended wrong answer at $t$). "
                              "Verified run set: plain false-fact passages on the MINT-verified questions (" + verified_counts_text() +
                              "), three seeds pooled; the misinformed agent's slot is shuffled per question. The three-agent full-sample "
                              "debates, in which the misinformed agent always speaks last, are analysed in the diagnostics table. "
                              "95\\% cluster-bootstrap CIs over debates; transitions with fewer than "
                              f"{min_transitions} eligible receiver pairs are omitted.",
                      label="tab:adoption", col_spec="llllllrrr")

    # Table: dissent, flips per debate, k = 0 switch rate and kappa (verified run set, 1-of-3, injection success)
    ver = [r for r in out_rows if r["run_set"] == "verified" and r["information_source"] == "misinformation" and r["subset"] == "all"
           and r["k"] == 1 and r["n"] == 3]
    rate_rows = []
    for _rs, model, ds, t in _order({("verified", r["model"], r["dataset"], r["turn_from"]) for r in ver}):
        inj = [r for r in ver if r["model"] == model and r["dataset"] == ds and r["turn_from"] == t and r["restrict"] == "injection_success"]
        al = [r for r in ver if r["model"] == model and r["dataset"] == ds and r["turn_from"] == t and r["restrict"] == "all"]
        a = _get(inj, metric="adoption_from_misinformed")
        if not a or a["n_transitions"] < min_transitions or t > 2:
            continue
        dm, dp = _get(inj, metric="dissent_rate_misinformed"), _get(inj, metric="dissent_rate_uninformed_peer")
        fm, fp = _get(inj, metric="flips_per_debate_misinformed"), _get(inj, metric="flips_per_debate_uninformed_peer")
        k0r, kap = _get(al, metric="baseline_switch_rate_k0"), _get(inj, metric="kappa_agreement_receiver_t1_vs_misinformed_t")
        rate_rows.append([model, DATASET_DISPLAY.get(ds, ds), f"{t}$\\to${t + 1}",
                          f"{dm['value']:.2f} / {dp['value']:.2f}" if dm and dp and dp["value"] == dp["value"] else "--",
                          f"{fm['value']:.2f} / {fp['value']:.2f}" if fm and fp else "--",
                          _fmt(k0r), f"{kap['value']:.2f}" if kap and kap["value"] == kap["value"] else "--", a["n_debates"]])
    write_latex_table(TABLES_DIR / "propagation_rates.tex",
                      ["Model", "Dataset", "Turn", "Dissent (misinf. / peer)", "Flips per debate (misinf. / peer)",
                       "Switch rate in $k{=}0$ debates", "$\\kappa$", "n debates"], rate_rows,
                      caption="How misinformation reaches the group (verified run set, 1-of-3 debates, debates in which the misinformed "
                              "agent held the intended wrong answer at $t$). Dissent: share of receiver-source pairs in which the "
                              "receiver disagreed with the source at $t$, for the misinformed agent and for the designated uninformed "
                              "peer; flips per debate: receiver switches to the source's answer at $t{+}1$ per debate ($=$ dissent "
                              "$\\times$ adoption, Table~\\ref{tab:adoption}); switch rate in $k{=}0$ debates: adoption of one designated "
                              "pseudo-source's answer in debates without misinformation (full sample); $\\kappa$: chance-corrected "
                              "agreement between the receiver's answer at $t{+}1$ and the misinformed agent's answer at $t$ (binary tasks).",
                      label="tab:propagation_rates", col_spec="lllllllr")

    # Table: full-sample diagnostics (restrictions of the full-sample run set) and speaking-slot split of the verified set, 1->2
    diag = [r for r in out_rows if r["information_source"] == "misinformation" and r["k"] == 1 and r["n"] == 3
            and r["restrict"] == "injection_success" and r["turn_from"] == 1 and r["experiment"] in ("exp_exposure", "exp2")]
    subset_label = {("verified", "all"): "verified run set", ("verified", "slot1"): "verified, misinf. agent speaks first",
                    ("verified", "slot2"): "verified, speaks second", ("verified", "slot3"): "verified, speaks last",
                    ("full-sample", "all"): "full-sample run set (9 categories)", ("full-sample", "false_fact"): "full-sample, plain false fact",
                    ("full-sample", "verified_passage"): "full-sample, verified passages",
                    ("full-sample", "false_fact_verified"): "full-sample, false fact + verified"}
    diag_rows = []
    for (rs, subset), label in subset_label.items():
        for model in MODEL_ORDER:
            for ds in DATASET_ORDER:
                sel = [r for r in diag if r["run_set"] == rs and r["subset"] == subset and r["model"] == model and r["dataset"] == ds]
                a, p, e = _get(sel, metric="adoption_from_misinformed"), _get(sel, metric="adoption_from_uninformed_peer"), \
                    _get(sel, metric="excess_adoption_vs_peer")
                if not (a and p and e) or a["n_transitions"] < min_transitions:
                    continue
                diag_rows.append([label, model, DATASET_DISPLAY.get(ds, ds), _fmt(a), _fmt(p), _fmt(e, signed=True),
                                  a["n_transitions"], a["n_debates"]])
    write_latex_table(TABLES_DIR / "propagation_fullsample_diagnostics.tex",
                      ["Debates", "Model", "Dataset", "Adoption misinf.", "Adoption peer", "Excess", "n tr.", "n deb."], diag_rows,
                      caption="Why the two run sets differ (turn $1\\to2$, 1-of-3 debates, injection success). The verified run set is "
                              "split by the slot in which the misinformed agent speaks (its slot is shuffled per question); the "
                              "three-agent full-sample debates, in which the misinformed agent always speaks last, are restricted to the plain "
                              "false fact, to verified passages, and to both (the question set of the verified run set). 95\\% "
                              "cluster-bootstrap CIs over debates; transitions with fewer than "
                              f"{min_transitions} eligible receiver pairs are omitted.", label="tab:adoption_diagnostics",
                      col_spec="p{3.3cm}p{1.9cm}p{1.6cm}lllrr")

    # Table: relevant-true control (adoption of a correctly informed agent's answer), WinoGrande, 1-of-3
    rt = [r for r in out_rows if r["information_source"] == "relevant_true" and r["subset"] == "all" and r["k"] == 1 and r["n"] == 3]
    rt_rows = []
    for _rs, model, ds, t in _order({("verified", r["model"], r["dataset"], r["turn_from"]) for r in rt}):
        for restrict, lab in (("injection_success", "source holds gold"), ("all", "all")):
            sel = [r for r in rt if r["model"] == model and r["dataset"] == ds and r["turn_from"] == t and r["restrict"] == restrict]
            a, p, e = _get(sel, metric="adoption_from_misinformed"), _get(sel, metric="adoption_from_uninformed_peer"), \
                _get(sel, metric="excess_adoption_vs_peer")
            if not (a and p and e) or a["n_transitions"] < min_transitions:
                continue
            rt_rows.append([model, DATASET_DISPLAY.get(ds, ds), f"{t}$\\to${t + 1}", lab, _fmt(a), _fmt(p), _fmt(e, signed=True),
                            a["n_transitions"], a["n_debates"]])
    if rt_rows:
        write_latex_table(TABLES_DIR / "propagation_relevant_true.tex",
                          ["Model", "Dataset", "Turn", "Debates", "Adoption informed", "Adoption peer", "Excess", "n tr.", "n deb."],
                          rt_rows,
                          caption="Control: adoption of a correctly informed agent's answer. In 1-of-3 debates with the relevant-true "
                                  "passage (seed 0; CWQ and ETHICS on the verified questions, WinoGrande on the full sample) one agent "
                                  "receives the relevant-true passage; the table gives the probability that an "
                                  "uninformed agent that disagreed with it at $t$ holds its answer at $t{+}1$, against the designated "
                                  "uninformed peer, for debates in which the informed agent holds the gold answer at $t$ and for all "
                                  "debates. 95\\% cluster-bootstrap CIs over debates; transitions with fewer than "
                                  f"{min_transitions} eligible receiver pairs are omitted.", label="tab:adoption_relevant_true",
                          col_spec="llllp{1.6cm}lllrr")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--turns_csv", action="append", default=None, help="repeatable; default out/results/rescored/mad_turns.csv")
    ap.add_argument("--n_boot", type=int, default=N_BOOT)
    ap.add_argument("--no_figure", action="store_true")
    ap.add_argument("--tables_only", action="store_true", help="rewrite the LaTeX tables from out/results/propagation_by_turn.csv")
    args = ap.parse_args()
    ensure_dirs()
    if args.tables_only:
        rows = []
        for r in read_csv(RESULTS_DIR / "propagation_by_turn.csv"):
            for c in ("value", "ci_lo", "ci_hi"):
                r[c] = float(r[c]) if r[c] not in ("", None) else float("nan")
            for c in ("k", "n", "turn_from", "turn_to", "n_transitions", "n_debates"):
                r[c] = int(float(r[c])) if r[c] not in ("", None) else None
            rows.append(r)
        write_tables(rows)
        return
    paths = [Path(p) for p in (args.turns_csv or [str(RESULTS_DIR / "rescored" / "mad_turns.csv")])]
    debates = load_debates(paths)
    print(f"loaded {len(debates)} debates")

    # verified run set: k >= 1 misinformation debates of exp_exposure on the verified questions of each dataset
    # (CWQ/ETHICS runs already contain only these; the WinoGrande runs cover all questions and are restricted here)
    V = {ds: verified_questions(ds) for ds in DATASET_ORDER}
    n_before = len(debates)
    for key in list(debates):
        m = debates[key]["meta"]
        if m["experiment"] == "exp_exposure" and m["source"] == "misinformation" and (m["k"] or 0) > 0 \
                and m["mint_id"] not in V.get(m["dataset"], set()):
            del debates[key]
    print(f"verified filter: {n_before} -> {len(debates)} debates ({verified_counts_text()})")

    # group debates by (experiment, source, model, dataset, k, n)
    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for d in debates.values():
        m = d["meta"]
        groups[(m["experiment"], m["source"], m["model"], m["dataset"], m["k"], m["n"])].append(d)

    out_rows: List[dict] = []
    trans_rows: List[dict] = []
    inj_rows: List[dict] = []
    share_rows: List[dict] = []
    vcache: Dict[tuple, set] = {}

    def passage_verified(ds: str, mint_id: str, category: Optional[str]) -> bool:
        cat = category or "false_fact"
        if (ds, cat) not in vcache:
            vcache[(ds, cat)] = verified_questions(ds, cat)
        return mint_id in vcache[(ds, cat)]

    for (exp, src, model, ds, k, n), ds_debates in sorted(groups.items(), key=lambda x: [str(v) for v in x[0]]):
        turns = sorted({t for d in ds_debates for t in d["answers"]})
        max_t = max(turns) if turns else 0
        # baseline group (k=0) for the same experiment/model/dataset/n
        k0 = groups.get((exp, "none", model, ds, 0, n), []) or groups.get((exp, "misinformation", model, ds, 0, n), [])
        meta = {"experiment": exp, "source": src, "model": model, "dataset": ds, "k": k, "n": n}

        if k and k > 0:
            out_rows += compute_adoption(meta, ds_debates, k0, args.n_boot)
            if exp == "exp_exposure" and src == "misinformation" and k == 1:
                # by the slot the misinformed agent speaks in (first / second / last): position-of-speaking diagnostic
                for slot in range(n):
                    sub = [d for d in ds_debates if d["mis"] == {slot}]
                    if sub:
                        out_rows += compute_adoption(meta, sub, [], args.n_boot, subset=f"slot{slot + 1}")
            if exp == "exp2" and k == 1:
                # restrictions of the full-sample run set that align it with the verified run set
                sub_ff = [d for d in ds_debates if (d["meta"]["category"] or "") == "false_fact"]
                sub_v = [d for d in ds_debates if passage_verified(ds, d["meta"]["mint_id"], d["meta"]["category"])]
                sub_both = [d for d in sub_ff if passage_verified(ds, d["meta"]["mint_id"], "false_fact")]
                for name, sub in (("false_fact", sub_ff), ("verified_passage", sub_v), ("false_fact_verified", sub_both)):
                    if sub:
                        out_rows += compute_adoption(meta, sub, [], args.n_boot, subset=name)

        for t in range(1, max_t):
            # ---- transitions (all debates), separately for misinformed / uninformed agents
            for agent_type in ("misinformed", "uninformed"):
                cnt: Counter = Counter()
                for d in ds_debates:
                    for a in range(n):
                        is_mis = a in d["mis"]
                        if (agent_type == "misinformed") != is_mis:
                            continue
                        r0 = d["answers"].get(t, {}).get(a)
                        r1 = d["answers"].get(t + 1, {}).get(a)
                        if r0 is None or r1 is None:
                            continue
                        cnt[(_state(r0), _state(r1))] += 1
                tot_from: Counter = Counter()
                for (s0, s1), c in cnt.items():
                    tot_from[s0] += c
                for (s0, s1), c in sorted(cnt.items()):
                    trans_rows.append({"run_set": RUN_SET.get(exp, exp), "experiment": exp, "information_source": src, "model": model,
                                       "dataset": ds, "k": k, "n": n, "agent_type": agent_type, "turn_from": t, "turn_to": t + 1,
                                       "state_from": s0, "state_to": s1, "count": c, "prob": c / tot_from[s0]})
            # ---- persistence (original metric) for reference
            for agent_type in ("misinformed", "uninformed"):
                same = tot = 0
                for d in ds_debates:
                    for a in range(n):
                        if (agent_type == "misinformed") != (a in d["mis"]):
                            continue
                        r0 = d["answers"].get(t, {}).get(a)
                        r1 = d["answers"].get(t + 1, {}).get(a)
                        if r0 is None or r1 is None or _value(r0) is None or _value(r1) is None:
                            continue
                        tot += 1
                        same += int(_value(r0) == _value(r1))
                if tot:
                    out_rows.append({"run_set": RUN_SET.get(exp, exp), "experiment": exp, "information_source": src, "subset": "all",
                                     "model": model, "dataset": ds, "k": k, "n": n, "turn_from": t, "turn_to": t + 1,
                                     "restrict": "all", "metric": f"persistence_{agent_type}", "value": same / tot,
                                     "ci_lo": float("nan"), "ci_hi": float("nan"), "n_transitions": tot, "n_debates": len(ds_debates)})

        # ---- injection success per turn and category; wrong-answer share per turn (misinformation debates only)
        if src == "relevant_true":
            continue
        for t in turns:
            by_cat: Dict[str, List[int]] = defaultdict(list)
            for d in ds_debates:
                for a in d["mis"]:
                    r = d["answers"].get(t, {}).get(a)
                    if r is None:
                        continue
                    iw = _b(r.get("is_intended_wrong"))
                    if r["status"] == "parsed" and iw is None:
                        continue  # the passage has no recorded target answer: adoption undefined, not "no"
                    by_cat[d["meta"]["category"] or "all"].append(int(bool(iw)) if r["status"] == "parsed" else 0)
            for cat, vals in by_cat.items():
                inj_rows.append({"run_set": RUN_SET.get(exp, exp), "experiment": exp, "model": model, "dataset": ds, "k": k, "n": n,
                                 "category": cat, "turn": t, "injection_success_rate": float(np.mean(vals)), "n_agents": len(vals)})
            if by_cat:
                allv = [v for vals in by_cat.values() for v in vals]
                inj_rows.append({"run_set": RUN_SET.get(exp, exp), "experiment": exp, "model": model, "dataset": ds, "k": k, "n": n,
                                 "category": "ALL", "turn": t, "injection_success_rate": float(np.mean(allv)), "n_agents": len(allv)})
            for agent_type in ("misinformed", "uninformed", "all"):
                shares = []
                for d in ds_debates:
                    vals = []
                    for a in range(n):
                        if agent_type == "misinformed" and a not in d["mis"]:
                            continue
                        if agent_type == "uninformed" and a in d["mis"]:
                            continue
                        r = d["answers"].get(t, {}).get(a)
                        if r is None:
                            continue
                        iw = _b(r.get("is_intended_wrong"))
                        if r["status"] == "parsed" and iw is None:
                            continue  # no recorded target answer for this passage
                        vals.append(1 if (r["status"] == "parsed" and iw) else 0)
                    if vals:
                        shares.append(np.mean(vals))
                if shares:
                    arr = np.asarray(shares)
                    rng = np.random.default_rng(SEED)
                    boots = arr[rng.integers(0, len(arr), size=(args.n_boot, len(arr)))].mean(axis=1)
                    share_rows.append({"run_set": RUN_SET.get(exp, exp), "experiment": exp, "model": model, "dataset": ds, "k": k,
                                       "n": n, "agent_type": agent_type, "turn": t, "wrong_share": float(arr.mean()),
                                       "ci_lo": float(np.quantile(boots, 0.025)), "ci_hi": float(np.quantile(boots, 0.975)),
                                       "n_debates": len(arr)})

    write_csv(RESULTS_DIR / "propagation_by_turn.csv", out_rows)
    write_csv(RESULTS_DIR / "propagation_transitions.csv", trans_rows)
    write_csv(RESULTS_DIR / "injection_success.csv", inj_rows)
    write_csv(RESULTS_DIR / "wrong_answer_share_by_turn.csv", share_rows)

    write_tables(out_rows)

    if not args.no_figure and share_rows:
        try:
            from scripts.figure_common import (MODEL_COLOR, dataset_label, finish_with_legend, layout_style, model_label,
                                               panel_axes, panel_title, save_figure, series_handle, shared_labels, style_axes)

            # verified 1-of-3 debates (exp_exposure, k=1) when available, else the exp2 runs
            exp_rows = [r for r in share_rows if r["experiment"] == "exp_exposure" and r["k"] == 1 and r["n"] == 3]
            if not exp_rows:
                exp_rows = [r for r in share_rows if r["experiment"] == "exp2"]
            datasets = [d for d in ("cwq", "ethics", "winogrande") if any(r["dataset"] == d for r in exp_rows)]
            models = [m for m in MODEL_ORDER if any(r["model"] == m for r in exp_rows)]
            type_style = {"misinformed": ("-", "o", "misinformed agent"), "uninformed": ("--", "s", "uninformed agents")}
            L = layout_style()
            fs = L["fs"]
            fig, axs = panel_axes(len(datasets), height=2.1)
            for ax, ds in zip(axs, datasets):
                for m in models:
                    col = MODEL_COLOR.get(m, "#999999")
                    for agent_type, (ls, mk, _lab) in type_style.items():
                        rows = sorted([r for r in exp_rows if r["dataset"] == ds and r["model"] == m
                                       and r["agent_type"] == agent_type and r["turn"] <= 3], key=lambda r: r["turn"])
                        if not rows:
                            continue
                        x = [r["turn"] for r in rows]
                        y = [r["wrong_share"] for r in rows]
                        lo = [r["ci_lo"] for r in rows]
                        hi = [r["ci_hi"] for r in rows]
                        ax.plot(x, y, linestyle=ls, color=col, marker=mk, markersize=L["markersize"], markeredgecolor="#2e2e2e",
                                markeredgewidth=L["edgewidth"], linewidth=L["linewidth"], alpha=0.95, zorder=3)
                        ax.fill_between(x, lo, hi, color=col, alpha=0.12, linewidth=0, zorder=1)
                panel_title(ax, dataset_label(ds), fs)
                ax.set_xticks([1, 2, 3])
                ax.set_xlim(0.8, 3.2)
                ax.set_ylim(0, 0.9)
                ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
                style_axes(ax, "y", fs)
            shared_labels(axs, "Turn", "Share holding wrong answer", fs)
            handles = [series_handle(model_label(m), MODEL_COLOR.get(m, "#999999"), "o", "-", L["linewidth"], L["markersize"])
                       for m in models]
            # short legend entries at column width (the caption says "agent(s)"); the legend must not exceed the figure width
            handles += [series_handle(lab.split(" ")[0], "#4a4a4a", mk, ls, L["linewidth"], L["markersize"])
                        for (ls, mk, lab) in type_style.values()]
            finish_with_legend(fig, handles, [h.get_label() for h in handles], fs, ncol=min(len(handles), L["legend_ncol_max"]),
                               top=0.90)
            out = FIGURES_DIR / "fig_wrong_share_per_turn"
            save_figure(fig, out)
            print(f"figure -> {out}.pdf")
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] figure skipped: {e}")
    print(f"wrote {len(out_rows)} metric rows, {len(trans_rows)} transition rows, {len(inj_rows)} injection rows")


if __name__ == "__main__":
    main()
