#!/usr/bin/env python3
"""
rescore_results.py - Re-parse and re-score all stored experiment outputs.

Reads every results file under out/<model>/ (exp1, exp1a, exp2, exp2_ablation,
exp3), joins each record to the canonical MINT item (gold answers via
mint_data.resolve_gold), re-extracts
answers with answer_extraction (status: parsed / ambiguous / unparsed) and
writes long-format CSVs:

  out/results/rescored/single_agent.csv     one row per question x condition (exp1, exp1a)
  out/results/rescored/mad_final.csv        one row per debate x protocol (voting | consensus)
  out/results/rescored/mad_turns.csv        one row per question x agent x turn (exp2, exp3)
  out/results/rescored/self_consistency.csv one row per question x condition (exp2_ablation)
  out/results/rescored/accuracy_summary.csv stored vs rescored accuracy per configuration
  out/results/extraction_failure_rates.csv  + out/tables/extraction_failure_rates.tex

Handling rule (stated in the paper, Sec. 3.1): an answer counts as correct only
if it was parsed unambiguously and equals the gold option; unparsed/ambiguous
answers count as incorrect and their rates are reported. Sensitivity columns
(`is_correct_best_guess`, parsed-only accuracy) are provided as well.

Free-form (CWQ): correctness is containment of a gold answer/alias (normalized,
whole-word) in the 'Final Solution:' segment when a marker is present (MAD, SC),
otherwise in the whole response (single-agent exp1, whose prompt requests no
marker). Both lenient and strict variants are stored.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from answer_extraction import (  # noqa: E402
    STATUS_PARSED,
    extract_free_form_answer,
    extract_mc_answer,
    free_form_matches,
    normalize_text,
)
from mint_data import canonical_dataset, load_mint_items  # noqa: E402
from scripts.common import (  # noqa: E402
    DATASET_DISPLAY,
    DATASET_ORDER,
    MODEL_DIRS,
    MODEL_ORDER,
    RESULTS_DIR,
    TABLES_DIR,
    condition_to_relevance_category,
    ensure_dirs,
    load_json,
    model_display,
    read_csv,
    write_csv,
    write_latex_table,
)

RESCORED_DIR = RESULTS_DIR / "rescored"
# per-sample rows of exp_msc (filled by process_self_consistency, written to out/results/rescored/msc_samples.csv)
MSC_SAMPLE_ROWS: List[dict] = []
DATASET_FROM_FILENAME = {
    "complex_web_questions": "cwq",
    "cwq": "cwq",
    "ethics": "ethics",
    "ethics_commonsense": "ethics",
    "winogrande": "winogrande",
}


def _dataset_from_filename(name: str) -> Optional[str]:
    base = re.sub(r"\.json$", "", Path(name).name)
    m = re.match(r"(exp1a?|exp2_ablation|exp2|exp3|exposure|msc|mitigation_verifier)_results_(.+)$", base)
    if not m:
        return None
    rest = m.group(2)
    for key in sorted(DATASET_FROM_FILENAME, key=len, reverse=True):
        if rest == key or rest.startswith(key + "_"):
            return DATASET_FROM_FILENAME[key]
    return None


def _experiment_from_filename(name: str) -> Optional[str]:
    base = Path(name).name
    for exp, label in (("exp2_ablation", "exp2_ablation"), ("exp1a", "exp1a"), ("exp1", "exp1"), ("exp2", "exp2"),
                       ("exp3", "exp3"), ("exposure", "exp_exposure"), ("msc", "exp_msc"),
                       ("mitigation_verifier", "exp_mitigation_verifier")):
        if base.startswith(exp + "_results_"):
            return label
    return None


def _seed_from_filename(name: str) -> Optional[int]:
    m = re.search(r"_seed(-?\d+)\.json$", Path(name).name)
    return int(m.group(1)) if m else None


class ItemIndex:
    def __init__(self, data_dir: Optional[str]):
        self.data_dir = data_dir
        self._cache: Dict[str, Dict[str, dict]] = {}

    def get(self, dataset: str) -> Dict[str, dict]:
        ds = canonical_dataset(dataset)
        if ds not in self._cache:
            idx: Dict[str, dict] = {}
            for it in load_mint_items(ds, self.data_dir):
                idx[_norm_key(it["sentence"])] = it
            self._cache[ds] = idx
        return self._cache[ds]

    def lookup(self, dataset: str, sentence: str) -> Optional[dict]:
        if sentence is None:
            return None
        s = str(sentence)
        # exp2/exp3 inputs append "\nOptions:\nA) ...\nB) ..."
        s = s.split("\nOptions:")[0]
        return self.get(dataset).get(_norm_key(s))


def _norm_key(s: str) -> str:
    return " ".join(str(s or "").strip().split())


def _golds(item: dict) -> List[str]:
    return [item["gold_text"]] + list(item.get("gold_aliases") or [])


_NON_ANSWERS = {"", "none", "null", "n a", "na", "unknown", "not specified", "unclear", "not stated"}


def _implied_answers(item: dict, category: Optional[str]) -> List[str]:
    """Answers that the two judges recorded as implied by the passage of `category` (free-form datasets only).

    Falls back to the plain false fact when the category has no flags (k = 0 debates, controls), so the
    flag then measures how often an agent produces the passage's target answer spontaneously."""
    val = item.get("validation") or {}
    flags = val.get(category or "false_fact") or val.get("false_fact") or {}
    out: List[str] = []
    for judge in ("gpt_oss_120b", "glm_4_7_flash"):
        j = flags.get(judge) or {}
        a = j.get("implied_answer") if j.get("parsed") else None
        if isinstance(a, str) and normalize_text(a) not in _NON_ANSWERS and \
                normalize_text(a) not in {normalize_text(x) for x in out}:
            out.append(a)
    return out


def _matches_any(text: Optional[str], cands: List[str]) -> bool:
    """Whole-word containment in either direction after normalisation (at least three characters)."""
    p = normalize_text(text or "")
    if len(p) < 3:
        return False
    for c in cands:
        n = normalize_text(c)
        if len(n) < 3:
            continue
        if re.search(r"(?<!\w)" + re.escape(n) + r"(?!\w)", p) or re.search(r"(?<!\w)" + re.escape(p) + r"(?!\w)", n):
            return True
    return False


def _score_record(item: dict, text: Optional[str], marker_required: bool, category: Optional[str] = None) -> dict:
    """Common scoring for one answer string given a MINT item.

    `is_intended_wrong` = the answer is the wrong answer the passage argues for. On the binary datasets every
    parsed wrong answer is that answer (two options). On free-form CWQ the answer must match the answer the
    judges recorded as implied by the passage (`category`); None when no implied answer is on record."""
    if item["is_multiple_choice"]:
        ex = extract_mc_answer(text, item["options"])
        gi = item["gold_index"]
        return {
            "predicted_index": ex.index,
            "predicted_text": ex.text,
            "status": ex.status,
            "method": ex.method,
            "is_correct": bool(ex.status == STATUS_PARSED and ex.index == gi),
            "is_correct_best_guess": bool(ex.index is not None and ex.index == gi),
            "is_intended_wrong": (bool(ex.index is not None and ex.index != gi and len(item["options"]) == 2)
                                  if ex.status != "unparsed" else None),
            "is_correct_lenient": None,
            "is_correct_strict": None,
        }
    ex = extract_free_form_answer(text)
    golds = _golds(item)
    lenient = free_form_matches(text, golds) if ex.status != "unparsed" else False
    strict = free_form_matches(ex.text, golds) if ex.method == "marker" else None
    if marker_required:
        is_correct = bool(strict) if strict is not None else bool(lenient)
    else:
        is_correct = bool(lenient)
    implied = _implied_answers(item, category)
    if ex.status == "unparsed" or not implied:
        intended_wrong = None
    else:
        intended_wrong = bool((not is_correct) and _matches_any(ex.text, implied))
    return {
        "predicted_index": None,
        "predicted_text": (ex.text or "")[:300],
        "status": ex.status,
        "method": ex.method,
        "is_correct": is_correct,
        "is_correct_best_guess": is_correct,
        "is_intended_wrong": intended_wrong,
        "is_correct_lenient": bool(lenient),
        "is_correct_strict": strict,
    }


def _misinformed_positions(mallm_log: dict) -> List[int]:
    pos = []
    for i, p in enumerate(mallm_log.get("personas") or []):
        desc = str(p.get("personaDescription") or "").lower()
        if "following information" in desc or "extra information" in desc:
            pos.append(i)
    return pos


def _agent_positions(mallm_log: dict) -> Dict[str, int]:
    return {p.get("agentId"): i for i, p in enumerate(mallm_log.get("personas") or []) if p.get("agentId")}


def _parse_mix_key(cond: str):
    m = re.match(r"(\d+)_misinformed_(\d+)_informed", cond)
    if m:
        k = int(m.group(1))
        return k, k + int(m.group(2))
    return None, None


def process_single_agent(path: Path, exp: str, dataset: str, model_dir: str, index: ItemIndex,
                         rows: List[dict], summary: List[dict], status_counts: Counter) -> None:
    data = load_json(path)
    model = model_display(model_dir)
    for condition, payload in data.items():
        if not isinstance(payload, dict) or "results" not in payload:
            continue
        results = payload.get("results") or []
        cfg = payload.get("config") or {}
        # seed: recorded in the config of the exp1 runs; older runs (exp1a) have none -> 0.
        seed = cfg.get("seed")
        if seed is None:
            m_seed = re.search(r"seeds[\\/]seed(\d+)", str(path))
            seed = int(m_seed.group(1)) if m_seed else 0
        reprompt = cfg.get("reprompt_unparsed")
        n_ok = 0
        n_bg = 0
        n_parsed = 0
        n_total = 0
        n_missing = 0
        for r in results:
            item = index.lookup(dataset, r.get("sentence"))
            if item is None:
                n_missing += 1
                continue
            rel, cat = condition_to_relevance_category(condition, r.get("misinformation_strategy"))
            # runs with --reprompt_unparsed: the constrained follow-up answer replaces the
            # first (unparseable) response, exactly as in exp1.py; the first response stays in the file
            reprompted = bool(r.get("reprompted")) and bool(r.get("reprompt_response"))
            text = r.get("reprompt_response") if reprompted else r.get("response")
            sc = _score_record(item, text, marker_required=False, category=cat)
            sc_first = _score_record(item, r.get("response"), marker_required=False, category=cat) if reprompted else sc
            n_total += 1
            n_ok += int(sc["is_correct"])
            n_bg += int(sc["is_correct_best_guess"])
            n_parsed += int(sc["status"] == STATUS_PARSED)
            status_counts[(model, dataset, exp, "single_agent_response", condition, sc["status"])] += 1
            rows.append({
                "experiment": exp, "model": model, "model_dir": model_dir, "dataset": dataset,
                "condition": condition, "relevance": rel, "category": cat,
                "misinformation_strategy": r.get("misinformation_strategy"),
                "seed": seed, "reprompt_unparsed": reprompt,
                "mint_id": item["mint_id"], "gold_index": item["gold_index"], "gold_text": item["gold_text"],
                **sc,
                "reprompted": reprompted, "status_first_response": sc_first["status"],
                "original_is_correct": bool(r.get("is_correct")),
                "response_chars": len(str(r.get("response") or "")),
            })
        if n_total:
            summary.append({
                "experiment": exp, "model": model, "dataset": dataset, "condition": condition, "protocol": "single",
                "seed": seed, "n": n_total, "n_missing_item": n_missing,
                "original_accuracy": payload.get("accuracy"),
                "rescored_accuracy": n_ok / n_total,
                "rescored_accuracy_best_guess": n_bg / n_total,
                "rescored_accuracy_parsed_only": (n_ok / n_parsed) if n_parsed else None,
                "parsed_rate": n_parsed / n_total,
            })


def process_mad(path: Path, exp: str, dataset: str, model_dir: str, index: ItemIndex, final_rows: List[dict],
                turn_rows: List[dict], summary: List[dict], status_counts: Counter) -> None:
    data = load_json(path)
    model = model_display(model_dir)
    for condition, payload in data.items():
        if not isinstance(payload, dict) or "results" not in payload:
            continue
        results = payload.get("results") or []
        seed = _seed_from_filename(path.name)
        cond_info = None
        if exp == "exp3":
            k, n_agents = _parse_mix_key(condition)
            category = "false_fact"
        elif exp in ("exp_exposure", "exp_mitigation_verifier"):
            from exposure_common import parse_condition_key

            cond_info = parse_condition_key(condition) or {}
            k, n_agents, category = cond_info.get("k"), cond_info.get("n"), cond_info.get("category", condition)
        else:
            k, n_agents, category = 1, 3, condition
        acc = {"voting": [0, 0], "consensus": [0, 0]}
        n_missing = 0
        # MALLM substitutes a failed discussion by re-running the first input examples, so a stored
        # file can hold two debates of one question while another question never ran (Llama exp3,
        # see exp3_fill_missing.py). Both debates are kept here (n counts records); the summary
        # reports the duplicates so that n != distinct questions does not go unnoticed.
        seen_items: set = set()
        n_duplicate = 0
        for r in results:
            ml = r.get("mallm_log") or {}
            item = index.lookup(dataset, r.get("sentence") or (ml.get("input") or [""])[0])
            if item is None:
                n_missing += 1
                continue
            if item["mint_id"] in seen_items:
                n_duplicate += 1
            seen_items.add(item["mint_id"])
            positions = _agent_positions(ml)
            mis_pos = _misinformed_positions(ml)
            if k is None:
                k = len(mis_pos)
                n_agents = len(positions) or 3
            turns = int(ml.get("turns") or 0)
            base = {
                "experiment": exp, "model": model, "model_dir": model_dir, "dataset": dataset,
                "condition": condition, "category": category, "k_misinformed": k, "num_agents": n_agents,
                "seed": seed if seed is not None else (0 if exp == "exp3" else None),
                "information_source": (cond_info or {}).get("information_source", "misinformation"),
                "passage_sharing": (cond_info or {}).get("passage_sharing", "shared"),
                "mitigation": (cond_info or {}).get("mitigation", "none"),
                "mint_id": item["mint_id"], "gold_index": item["gold_index"], "gold_text": item["gold_text"],
                "turns": turns, "decision_success": ml.get("decisionSuccess"),
                "misinformed_positions": ";".join(str(p) for p in mis_pos),
            }
            # --- voting protocol: MALLM finalAnswer
            sc_v = _score_record(item, ml.get("finalAnswer") if ml else r.get("predicted_answer"), marker_required=True,
                                 category=category)
            status_counts[(model, dataset, exp, "mad_final_voting", condition, sc_v["status"])] += 1
            acc["voting"][0] += int(sc_v["is_correct"])
            acc["voting"][1] += 1
            final_rows.append({**base, "protocol": "voting", **sc_v, "original_is_correct": bool(r.get("is_correct")),
                               "consensus_fallback_used": None})

            # --- per-turn rows and consensus protocol (last speaker of the final turn)
            mem = ml.get("globalMemory") or []
            by_turn: Dict[int, List[dict]] = defaultdict(list)
            for msg in mem:
                try:
                    t = int(msg.get("turn"))
                except Exception:
                    continue
                by_turn[t].append(msg)
            for t in sorted(by_turn):
                msgs = sorted(by_turn[t], key=lambda m: int(m.get("message_id", 0)))
                for order, msg in enumerate(msgs):
                    aid = msg.get("agent_id")
                    pos = positions.get(aid)
                    sc_t = _score_record(item, msg.get("solution"), marker_required=False, category=category)
                    status_counts[(model, dataset, exp, "mad_turn_solution", condition, sc_t["status"])] += 1
                    turn_rows.append({
                        **{kk: base[kk] for kk in ("experiment", "model", "dataset", "condition", "category",
                                                   "k_misinformed", "num_agents", "seed", "information_source",
                                                   "mitigation", "mint_id", "gold_index")},
                        "turn": t, "message_id": msg.get("message_id"), "speaking_order": order,
                        "agent_index": pos, "agent_id": aid, "persona": msg.get("persona"),
                        "is_misinformed": (pos in mis_pos) if pos is not None else None,
                        "agreement": msg.get("agreement"),
                        "predicted_index": sc_t["predicted_index"], "predicted_text": sc_t["predicted_text"],
                        "status": sc_t["status"], "method": sc_t["method"], "is_correct": sc_t["is_correct"],
                        "is_intended_wrong": sc_t["is_intended_wrong"],
                    })
            if by_turn:
                last_t = max(by_turn)
                msgs = sorted(by_turn[last_t], key=lambda m: int(m.get("message_id", 0)))
                chosen = None
                fallback = False
                for msg in reversed(msgs):
                    sc_c = _score_record(item, msg.get("solution"), marker_required=False, category=category)
                    if sc_c["status"] == STATUS_PARSED:
                        chosen = sc_c
                        break
                    fallback = True
                if chosen is None:
                    chosen = _score_record(item, msgs[-1].get("solution"), marker_required=False, category=category)
                    fallback = False
                status_counts[(model, dataset, exp, "mad_final_consensus", condition, chosen["status"])] += 1
                acc["consensus"][0] += int(chosen["is_correct"])
                acc["consensus"][1] += 1
                final_rows.append({**base, "protocol": "consensus", **chosen, "original_is_correct": None,
                                   "consensus_fallback_used": fallback})
        if n_duplicate:
            print(f"[WARN] {model} {exp} {dataset} {condition}: {n_duplicate} duplicate debates "
                  f"({len(seen_items)} distinct questions in {len(seen_items) + n_duplicate} records); "
                  "see exp3_fill_missing.py")
        for proto, (c, n) in acc.items():
            if n:
                summary.append({
                    "experiment": exp, "model": model, "dataset": dataset, "condition": condition, "protocol": proto,
                    "n": n, "n_missing_item": n_missing, "n_duplicate_item": n_duplicate,
                    "original_accuracy": payload.get("accuracy") if proto == "voting" else None,
                    "rescored_accuracy": c / n, "rescored_accuracy_best_guess": None,
                    "rescored_accuracy_parsed_only": None, "parsed_rate": None,
                })


def process_self_consistency(path: Path, dataset: str, model_dir: str, index: ItemIndex, rows: List[dict],
                             summary: List[dict], status_counts: Counter, exp: str = "exp2_ablation") -> None:
    data = load_json(path)
    model = model_display(model_dir)
    seed = _seed_from_filename(path.name)
    for condition, payload in data.items():
        if not isinstance(payload, dict) or "results" not in payload:
            continue
        cond_info = {}
        if exp.startswith("exp_msc"):
            from exposure_common import parse_condition_key

            cond_info = parse_condition_key(condition) or {}
        n_ok = n_total = n_missing = 0
        for r in payload.get("results") or []:
            item = index.lookup(dataset, r.get("sentence"))
            if item is None:
                n_missing += 1
                continue
            sc_log = r.get("sc_log") or r.get("msc_log") or {}
            samples = sc_log.get("samples") or []
            if exp.startswith("exp_msc"):
                # exp_msc stores per-sample extraction results (raw responses live in the JSONL file)
                votes_m = []
                n_parsed_m = 0
                for s in samples:
                    status_counts[(model, dataset, exp, "sc_sample", condition, s.get("status") or "unparsed")] += 1
                    # per-sample rows: single-agent outcomes under the debate prompt format (exposure matched
                    # single agent: informed samples = 1/1 exposure, uninformed samples = 0/1)
                    if item["is_multiple_choice"]:
                        ok_s = bool(s.get("status") == "parsed" and s.get("predicted_index") is not None
                                    and int(s["predicted_index"]) == item["gold_index"])
                    else:
                        ok_s = bool(s.get("status") == "parsed" and s.get("is_correct"))
                    MSC_SAMPLE_ROWS.append({
                        "experiment": exp, "model": model, "model_dir": model_dir, "dataset": dataset, "condition": condition,
                        "category": cond_info.get("category", condition), "k_misinformed": cond_info.get("k"),
                        "num_samples_cfg": cond_info.get("n"), "seed": seed,
                        "information_source": cond_info.get("information_source"), "mitigation": cond_info.get("mitigation"),
                        "mint_id": item["mint_id"], "gold_index": item["gold_index"], "sample_idx": s.get("sample_idx"),
                        "informed": bool(s.get("informed")), "status": s.get("status"), "predicted_index": s.get("predicted_index"),
                        "is_correct": ok_s, "finish_reason": s.get("finish_reason"),
                        "prompt_tokens": s.get("prompt_tokens"), "completion_tokens": s.get("completion_tokens"),
                    })
                    if s.get("status") == "parsed":
                        n_parsed_m += 1
                        if item["is_multiple_choice"]:
                            votes_m.append(int(s["predicted_index"]))
                        else:
                            votes_m.append(normalize_text(s.get("predicted_text") or "") or None)
                votes_m = [v for v in votes_m if v is not None]
                winner_m, tie_m, share_m = None, False, 0.0
                if votes_m:
                    top = Counter(votes_m).most_common()
                    winner_m = top[0][0]
                    tie_m = len(top) > 1 and top[1][1] == top[0][1]
                    share_m = top[0][1] / len(votes_m)
                if item["is_multiple_choice"]:
                    ok_m = bool(winner_m is not None and not tie_m and winner_m == item["gold_index"])
                    pred_m = item["options"][winner_m] if winner_m is not None else None
                else:
                    ok_m = bool(winner_m is not None and not tie_m and free_form_matches(winner_m, _golds(item)))
                    pred_m = winner_m
                n_total += 1
                n_ok += int(ok_m)
                rows.append({
                    "experiment": exp, "model": model, "model_dir": model_dir, "dataset": dataset,
                    "condition": condition, "category": cond_info.get("category", condition),
                    "k_misinformed": cond_info.get("k"), "num_samples_cfg": cond_info.get("n"), "seed": seed,
                    "information_source": cond_info.get("information_source"), "mitigation": cond_info.get("mitigation"),
                    "mint_id": item["mint_id"], "gold_index": item["gold_index"], "gold_text": item["gold_text"],
                    "num_samples": len(samples), "num_parsed": n_parsed_m,
                    "majority_index": winner_m if item["is_multiple_choice"] else None, "majority_text": pred_m,
                    "majority_share": share_m, "tie": tie_m, "status": "parsed" if votes_m else "unparsed",
                    "is_correct": ok_m, "original_is_correct": bool(r.get("is_correct")),
                })
                continue
            votes: List = []
            n_parsed = 0
            for s in samples:
                resp = s.get("response")
                if resp is None:
                    # raw responses not stored: fall back to stored extracted answer
                    if item["is_multiple_choice"] and s.get("vote_key") is not None:
                        votes.append(int(s["vote_key"]))
                        n_parsed += 1
                        status_counts[(model, dataset, exp, "sc_sample", condition, "stored_vote")] += 1
                    continue
                if item["is_multiple_choice"]:
                    ex = extract_mc_answer(resp, item["options"])
                    status_counts[(model, dataset, exp, "sc_sample", condition, ex.status)] += 1
                    if ex.status == STATUS_PARSED:
                        votes.append(ex.index)
                        n_parsed += 1
                else:
                    ex = extract_free_form_answer(resp)
                    status_counts[(model, dataset, exp, "sc_sample", condition, ex.status)] += 1
                    if ex.status == STATUS_PARSED and ex.text:
                        votes.append(normalize_text(ex.text) or None)
                        n_parsed += 1
            votes = [v for v in votes if v is not None]
            winner = None
            tie = False
            share = 0.0
            if votes:
                cnt = Counter(votes)
                top = cnt.most_common()
                winner = top[0][0]
                tie = len(top) > 1 and top[1][1] == top[0][1]
                share = top[0][1] / len(votes)
            if item["is_multiple_choice"]:
                is_correct = bool(winner is not None and not tie and winner == item["gold_index"])
                pred_text = item["options"][winner] if winner is not None else None
            else:
                is_correct = bool(winner is not None and not tie and free_form_matches(winner, _golds(item)))
                pred_text = winner
            n_total += 1
            n_ok += int(is_correct)
            rows.append({
                "experiment": exp, "model": model, "model_dir": model_dir, "dataset": dataset,
                "condition": condition, "category": condition, "k_misinformed": 1, "num_samples_cfg": 15, "seed": seed,
                "information_source": "misinformation", "mitigation": "none", "mint_id": item["mint_id"],
                "gold_index": item["gold_index"], "gold_text": item["gold_text"],
                "num_samples": len(samples), "num_parsed": n_parsed, "majority_index": winner if item["is_multiple_choice"] else None,
                "majority_text": pred_text, "majority_share": share, "tie": tie, "status": "parsed" if votes else "unparsed",
                "is_correct": is_correct, "original_is_correct": bool(r.get("is_correct")),
            })
        if n_total:
            summary.append({
                "experiment": exp, "model": model, "dataset": dataset, "condition": condition, "protocol": "self_consistency",
                "n": n_total, "n_missing_item": n_missing, "original_accuracy": payload.get("accuracy"),
                "rescored_accuracy": n_ok / n_total, "rescored_accuracy_best_guess": None,
                "rescored_accuracy_parsed_only": None, "parsed_rate": None,
            })


# paper terms for the extraction-failure table (the CSV keeps the raw identifiers)
FAILURE_EXP_NAMES = {
    "exp1": "Single agent",
    "exp_msc": "mSC",
    "exp_msc_matched": "mSC, matched sampling",
    "exp_exposure": "Debate, verified run set",
    "exp_mitigation_verifier": "Verifier re-run",
    "exp2": "Debate, full-sample run set",
    "exp2_ablation": "SC15",
    "exp3": "Five-agent debate",
}
FAILURE_STAGE_NAMES = {
    "single_agent_response": "response",
    "mad_final_voting": "final answer, voting",
    "mad_final_consensus": "final answer, consensus",
    "mad_turn_solution": "per-turn answer",
    "sc_sample": "sample",
}
FAILURE_EXP_ORDER = ["exp1", "exp_msc", "exp_msc_matched", "exp_exposure", "exp_mitigation_verifier", "exp2", "exp2_ablation", "exp3"]
# exp1a (the single-agent irrelevant-true run on the v1.1 Wikipedia passages) is replaced by the irrelevant-true
# condition of exp1 in the paper; it stays in the CSV but not in the table
FAILURE_EXCLUDE = {"exp1a"}


def write_failure_table(fr_rows: List[dict]) -> None:
    rows = [r for r in fr_rows if r["condition"] == "ALL" and r["experiment"] not in FAILURE_EXCLUDE]
    model_rank = {m: i for i, m in enumerate(MODEL_ORDER)}
    ds_rank = {d: i for i, d in enumerate(DATASET_ORDER)}
    stage_rank = {k: i for i, k in enumerate(FAILURE_STAGE_NAMES)}
    rows.sort(key=lambda r: (model_rank.get(r["model"], 9), ds_rank.get(r["dataset"], 9),
                             FAILURE_EXP_ORDER.index(r["experiment"]) if r["experiment"] in FAILURE_EXP_ORDER else 99,
                             stage_rank.get(r["stage"], 9)))
    tex_rows = []
    for r in rows:
        fail = float(r["failure_rate"])
        flag = " *" if fail > 0.05 else ""
        stage = f"{FAILURE_EXP_NAMES.get(r['experiment'], r['experiment'])}: {FAILURE_STAGE_NAMES.get(r['stage'], r['stage'])}"
        tex_rows.append([r["model"], DATASET_DISPLAY.get(r["dataset"], r["dataset"]), stage, int(float(r["n"])),
                         f"{100 * float(r['unparsed_rate']):.1f}", f"{100 * float(r['ambiguous_rate']):.1f}", f"{100 * fail:.1f}{flag}"])
    write_latex_table(
        TABLES_DIR / "extraction_failure_rates.tex",
        ["Model", "Dataset", "Stage", "n", "Unparsed %", "Ambig. %", "Failure %"],
        tex_rows,
        caption="Answer-extraction failure rates per model, dataset and stage (all conditions pooled). Single agent: all "
                "context conditions at seed 0, uninformed and false fact at seeds 1 and 2. mSC: the samples of misinformed "
                "self-consistency. Debate, verified run set: the three-agent debates of the exposure levels, including the "
                "uninformed, relevant-true and source-skepticism debates. Verifier re-run: the regenerated final stage with and "
                "without the verifier report. Debate, full-sample run set: the three-agent debates with all nine passage "
                "categories. SC15: the samples of the 15-sample self-consistency. Five-agent debate: the composition sweep. "
                "Unparsed and ambiguous answers are counted as incorrect in all accuracies. Rows marked * exceed 5\\%.",
        label="tab:extraction_failures", col_spec="lllrrrr",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", default=str(REPO_ROOT / "out"))
    ap.add_argument("--data_dir", default=None, help="Dataset dir (default: MINT_dataset_v1.2 -> data -> MINT-dataset_v1.1)")
    ap.add_argument("--models", default=None, help="Comma-separated model dir names (default: all known)")
    ap.add_argument("--include_runs", action="store_true", help="Also scan out/<model>/runs/*/ (replicates)")
    ap.add_argument("--results_dir", default=None, help="Write outputs here instead of out/results/ (tests)")
    ap.add_argument("--tables_only", action="store_true",
                    help="rewrite the extraction-failure table from out/results/extraction_failure_rates.csv (no re-scoring)")
    args = ap.parse_args()
    ensure_dirs()
    global RESCORED_DIR, RESULTS_DIR, TABLES_DIR
    if args.results_dir:
        RESULTS_DIR = Path(args.results_dir)
        RESCORED_DIR = RESULTS_DIR / "rescored"
        TABLES_DIR = RESULTS_DIR / "tables"
    RESCORED_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    if args.tables_only:
        write_failure_table(read_csv(RESULTS_DIR / "extraction_failure_rates.csv"))
        print("rewrote extraction_failure_rates.tex")
        return

    out_dir = Path(args.out_dir)
    model_dirs = [m for m in (args.models.split(",") if args.models else MODEL_DIRS) if (out_dir / m).is_dir()]
    index = ItemIndex(args.data_dir)

    sa_rows: List[dict] = []
    final_rows: List[dict] = []
    turn_rows: List[dict] = []
    sc_rows: List[dict] = []
    summary: List[dict] = []
    status_counts: Counter = Counter()

    for mdir in model_dirs:
        paths = sorted((out_dir / mdir).glob("*_results_*.json"))
        for sub in ("exposure", "msc", "msc_matched", "mitigation", "seeds/*", "controls"):
            paths += sorted((out_dir / mdir).glob(f"{sub}/*_results_*.json"))
        if args.include_runs:
            paths += sorted((out_dir / mdir).glob("runs/*/*_results_*.json"))
        for p in paths:
            exp = _experiment_from_filename(p.name)
            if exp == "exp_msc" and p.parent.name == "msc_matched":
                # mSC re-sampled at the debate's server-default temperature / top-p (Llama-3.3)
                exp = "exp_msc_matched"
            ds = _dataset_from_filename(p.name)
            if exp is None or ds is None:
                print(f"[skip] {p}")
                continue
            print(f"[rescore] {mdir} {exp} {ds} <- {p.name}")
            if exp in ("exp1", "exp1a"):
                process_single_agent(p, exp, ds, mdir, index, sa_rows, summary, status_counts)
            elif exp in ("exp2", "exp3", "exp_exposure", "exp_mitigation_verifier"):
                process_mad(p, exp, ds, mdir, index, final_rows, turn_rows, summary, status_counts)
            elif exp == "exp2_ablation":
                process_self_consistency(p, ds, mdir, index, sc_rows, summary, status_counts)
            elif exp in ("exp_msc", "exp_msc_matched"):
                process_self_consistency(p, ds, mdir, index, sc_rows, summary, status_counts, exp=exp)

    write_csv(RESCORED_DIR / "single_agent.csv", sa_rows)
    write_csv(RESCORED_DIR / "mad_final.csv", final_rows)
    write_csv(RESCORED_DIR / "mad_turns.csv", turn_rows)
    write_csv(RESCORED_DIR / "self_consistency.csv", sc_rows)
    write_csv(RESCORED_DIR / "msc_samples.csv", MSC_SAMPLE_ROWS)
    write_csv(RESCORED_DIR / "accuracy_summary.csv", summary)

    # ---- extraction failure rates (per model x dataset x experiment x stage; all conditions pooled + per condition)
    agg: Dict[tuple, Counter] = defaultdict(Counter)
    per_cond: Dict[tuple, Counter] = defaultdict(Counter)
    for (model, ds, exp, stage, cond, status), n in status_counts.items():
        agg[(model, ds, exp, stage)][status] += n
        per_cond[(model, ds, exp, stage, cond)][status] += n
    fr_rows = []
    for key, cnt in sorted(agg.items()):
        n = sum(cnt.values())
        fr_rows.append({"model": key[0], "dataset": key[1], "experiment": key[2], "stage": key[3], "condition": "ALL",
                        "n": n, "parsed": cnt.get("parsed", 0) + cnt.get("stored_vote", 0),
                        "ambiguous": cnt.get("ambiguous", 0), "unparsed": cnt.get("unparsed", 0),
                        "unparsed_rate": cnt.get("unparsed", 0) / n, "ambiguous_rate": cnt.get("ambiguous", 0) / n,
                        "failure_rate": (cnt.get("unparsed", 0) + cnt.get("ambiguous", 0)) / n})
    for key, cnt in sorted(per_cond.items()):
        n = sum(cnt.values())
        fr_rows.append({"model": key[0], "dataset": key[1], "experiment": key[2], "stage": key[3], "condition": key[4],
                        "n": n, "parsed": cnt.get("parsed", 0) + cnt.get("stored_vote", 0),
                        "ambiguous": cnt.get("ambiguous", 0), "unparsed": cnt.get("unparsed", 0),
                        "unparsed_rate": cnt.get("unparsed", 0) / n, "ambiguous_rate": cnt.get("ambiguous", 0) / n,
                        "failure_rate": (cnt.get("unparsed", 0) + cnt.get("ambiguous", 0)) / n})
    write_csv(RESULTS_DIR / "extraction_failure_rates.csv", fr_rows)

    write_failure_table(fr_rows)
    print(f"single_agent rows={len(sa_rows)} mad_final rows={len(final_rows)} mad_turns rows={len(turn_rows)} "
          f"sc rows={len(sc_rows)}; wrote {RESCORED_DIR} and {RESULTS_DIR / 'extraction_failure_rates.csv'}")


if __name__ == "__main__":
    main()
