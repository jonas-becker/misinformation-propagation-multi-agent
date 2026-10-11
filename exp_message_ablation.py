#!/usr/bin/env python3
"""
exp_message_ablation.py - Counterfactual message ablation for causal propagation.

For a random subset of 1-of-3 debates in which the misinformed agent M held the
intended wrong answer at turn t, the turn t+1 response of each receiving
(uninformed) agent R is regenerated from the exact prompt R saw (reconstructed from
the stored MALLM `additional_args`: task instruction, input, current draft, persona
and discussion history) under three variants:
  actual    - the original prompt (regenerated, to control for sampling noise)
  removed   - M's messages deleted from the history; the current draft is recomputed
              from the remaining messages as MALLM does
  replaced  - M's messages replaced by an uninformed agent's message for the same
              question (from a k=0 run of the same model when available, otherwise the
              other uninformed agent's message of the same debate, marked as such)
Causal propagation = P(R wrong at t+1 | actual) - P(R wrong at t+1 | ablated), paired
over (debate, receiver) with bootstrap CIs. The first receiver after M (Participant 1
at turn 2, who has seen no other turn-2 message) gives the clean direct effect; later
receivers include spill-over through other agents and are reported separately.

Usage
  python exp_message_ablation.py --model_name openai/gpt-oss-120b --endpoint_url http://host:port/v1
  python exp_message_ablation.py --analyze_only                     # aggregate existing raw outputs
  python exp_message_ablation.py --mock --n_debates 5                # pipeline test
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from answer_extraction import extract_free_form_answer, extract_mc_answer, free_form_matches
from mint_data import canonical_dataset, load_mint_items
from model_server_config import normalize_model_name
from runner_common import ChatClient, JsonlWriter, TokenLedger, load_experiment_config, read_jsonl, resolve_endpoint, stable_seed, write_run_metadata
from scripts.common import MODEL_DIRS, RESULTS_DIR, TABLES_DIR, ensure_dirs, model_display, write_csv, write_latex_table
from scripts.stats import paired_bootstrap_diff

REPO_ROOT = Path(__file__).resolve().parent
IMPROVE_INSTRUCTION = ("Improve the current solution. If you agree with the current solution, answer with [AGREE], "
                       "else answer with [DISAGREE] and explain why and provide an improved solution.")
FINAL_EXTRACT_INSTRUCTION = ("Extract the final solution to the task from the provided text. Remove statements of agreement, "
                             "disagreement, and explanations. Do not modify the text. Do not output any text besides the solution. "
                             "If there is no solution provided, just copy the previous response.")


def filled_system_prompt(task_instruction: str, input_str: str, persona: str, persona_description: str,
                         current_draft: Optional[str], has_memory: bool) -> str:
    """Verbatim reconstruction of MALLM SimpleResponseGenerator.get_filled_template."""
    s = (f"You take part in a discussion to solve a task.\nTask: {task_instruction}\nInput: {input_str}\n"
         f"Your role: {persona} ({persona_description})")
    if current_draft:
        s += f"\nCurrent Solution: {current_draft}"
    if has_memory:
        s += "\nThis is the discussion to the current point: \n"
    return s


def merge_consecutive(messages: List[dict]) -> List[dict]:
    """MALLM Chat.merge_consecutive_messages."""
    if not messages:
        return []
    merged = []
    role, content = messages[0]["role"], ""
    for m in messages:
        if m["role"] == role:
            content += m["content"] + "\n\n"
        else:
            merged.append({"role": role, "content": content.strip()})
            role, content = m["role"], m["content"] + "\n\n"
    if content:
        merged.append({"role": role, "content": content.strip()})
    return merged


def recompute_draft(memory_msgs: List[dict]) -> Optional[str]:
    """MALLM get_memories: draft = solution of the latest 'draft' or disagreeing 'improve' message."""
    draft = None
    for m in sorted(memory_msgs, key=lambda x: int(x.get("message_id", 0))):
        if m.get("contribution") == "draft" or (m.get("contribution") == "improve" and m.get("agreement") is False):
            draft = m.get("solution")
    return draft


def build_variants(record: dict, receiver_msg: dict, mis_persona: str, mis_agent_id: str, replacement_text: Optional[str],
                   replacement_source: str) -> Dict[str, List[dict]]:
    aa = receiver_msg["additional_args"]
    history: List[dict] = list(aa.get("agent_memory") or [])
    prefix = f"{mis_persona}: "
    mis_entries = [h for h in history if h["role"] == "user" and str(h["content"]).startswith(prefix)]
    variants: Dict[str, List[dict]] = {}

    def assemble(hist: List[dict], draft: Optional[str]) -> List[dict]:
        msgs = [{"role": "system", "content": filled_system_prompt(aa["task_instruction"], aa["input_str"], aa["persona"],
                                                                    aa["persona_description"], draft, bool(hist))}]
        msgs += hist
        msgs.append({"role": "user", "content": IMPROVE_INSTRUCTION})
        return merge_consecutive(msgs)

    variants["actual"] = assemble(history, aa.get("current_draft"))

    # removed: drop M's messages; recompute the draft from the remaining visible messages
    visible_ids = set(receiver_msg.get("memory_ids") or [])
    gm = [m for m in record["globalMemory"] if int(m["message_id"]) in visible_ids and m["agent_id"] != mis_agent_id]
    hist_removed = [h for h in history if h not in mis_entries]
    variants["removed"] = assemble(hist_removed, recompute_draft(gm) if gm else None)

    # replaced: substitute M's message text (keep position and persona label)
    if replacement_text is not None:
        hist_replaced = []
        for h in history:
            if h in mis_entries:
                hist_replaced.append({"role": "user", "content": prefix + replacement_text})
            else:
                hist_replaced.append(h)
        gm_rep = [dict(m) for m in record["globalMemory"] if int(m["message_id"]) in visible_ids]
        for m in gm_rep:
            if m["agent_id"] == mis_agent_id:
                m["solution"] = None  # replacement carries no explicit solution; draft falls back to earlier proposals
                m["agreement"] = True
        variants["replaced"] = assemble(hist_replaced, recompute_draft(gm_rep) if gm_rep else None)
    return variants


def load_k0_messages(model_dir: str, dataset: str, index_by_input: Dict[str, str]) -> Dict[str, str]:
    """Question input -> an uninformed agent's turn-1 message from a k=0 run (exp3 0/5 or exposure k=0)."""
    out: Dict[str, str] = {}
    cands = [REPO_ROOT / "out" / model_dir / f"exp3_results_{dataset}.json"]
    cands += sorted((REPO_ROOT / "out" / model_dir / "exposure").glob(f"exposure_results_{dataset}_seed*.json"))
    for p in cands:
        if not p.is_file():
            continue
        data = json.load(open(p, encoding="utf-8"))
        for cond, payload in data.items():
            if not (cond.startswith("0_misinformed") or "|k=0|" in cond):
                continue
            for r in payload.get("results", []):
                ml = r.get("mallm_log") or {}
                inp = (ml.get("input") or [""])[0]
                for m in ml.get("globalMemory") or []:
                    if int(m.get("turn", 0)) == 1 and m.get("message"):
                        out.setdefault(inp, m["message"])
                        break
    return out


def select_debates(model_dir: str, dataset: str, category: str, turn: int, n_debates: int, seed: int) -> List[dict]:
    path = REPO_ROOT / "out" / model_dir / f"exp2_results_{dataset}.json"
    data = json.load(open(path, encoding="utf-8"))
    payload = data.get(category) or {}
    items = {it["sentence"]: it for it in load_mint_items(dataset)}
    picked = []
    for r in payload.get("results", []):
        ml = r.get("mallm_log") or {}
        sent = r["sentence"].split("\nOptions:")[0]
        it = items.get(sent)
        if it is None:
            continue
        personas = ml.get("personas") or []
        mis = [p for p in personas if "following information" in str(p.get("personaDescription", "")).lower()]
        if len(mis) != 1:
            continue
        mis_id = mis[0]["agentId"]
        m_turn = [m for m in ml.get("globalMemory", []) if m["agent_id"] == mis_id and int(m["turn"]) == turn]
        if not m_turn:
            continue
        sol = m_turn[-1].get("solution")
        if it["is_multiple_choice"]:
            ex = extract_mc_answer(sol, it["options"])
            injected = ex.status == "parsed" and ex.index is not None and ex.index != it["gold_index"]
        else:
            ex = extract_free_form_answer(sol)
            injected = ex.status == "parsed" and not free_form_matches(ex.text, [it["gold_text"]] + list(it["gold_aliases"]))
        if not injected:
            continue
        picked.append({"record": ml, "item": it, "mis_id": mis_id, "mis_persona": mis[0]["persona"]})
    rng = random.Random(seed)
    rng.shuffle(picked)
    return picked[:n_debates]


def score(item: dict, text: str) -> Tuple[Optional[bool], str]:
    if item["is_multiple_choice"]:
        ex = extract_mc_answer(text, item["options"])
        if ex.status != "parsed":
            return None, ex.status
        return ex.index != item["gold_index"], ex.status
    ex = extract_free_form_answer(text)
    if ex.status != "parsed":
        return None, ex.status
    return not free_form_matches(ex.text, [item["gold_text"]] + list(item["gold_aliases"])), ex.status


def analyze(raw_path: Path, n_boot: int) -> None:
    rows = read_jsonl(raw_path)
    by_key: Dict[tuple, Dict[str, Optional[bool]]] = defaultdict(dict)
    meta: Dict[tuple, dict] = {}
    for r in rows:
        key = (r["model"], r["dataset"], r["mint_id"], r["receiver_index"], r["turn"])
        by_key[key][r["variant"]] = r["is_wrong"]
        meta[key] = r
    out = []
    groups = defaultdict(list)
    for key, v in by_key.items():
        groups[(key[0], key[1], key[4], meta[key]["receiver_kind"])].append(v)
        groups[(key[0], key[1], key[4], "all")].append(v)
    for (model, ds, turn, kind), vs in sorted(groups.items()):
        for variant in ("removed", "replaced"):
            pairs = [(float(v["actual"]), float(v[variant])) for v in vs
                     if v.get("actual") is not None and v.get(variant) is not None]
            if len(pairs) < 5:
                continue
            a = [p[0] for p in pairs]
            b = [p[1] for p in pairs]
            res = paired_bootstrap_diff(a, b, n_boot=n_boot)
            obs = [float(v["observed"]) for v in vs if v.get("observed") is not None]
            out.append({"model": model, "dataset": ds, "turn": turn, "receiver_kind": kind, "ablation": variant,
                        "p_wrong_actual_regen": res["mean_a"], "p_wrong_ablated": res["mean_b"],
                        "causal_propagation": res["diff"], "ci_lo": res["ci_lo"], "ci_hi": res["ci_hi"], "p_perm": res["p_perm"],
                        "n_pairs": res["n"], "p_wrong_observed_original": float(np.mean(obs)) if obs else None})
    ensure_dirs()
    write_csv(RESULTS_DIR / "ablation_propagation.csv", out)
    tex = [[r["model"], r["dataset"], f"{r['turn']}$\\to${r['turn'] + 1}", r["receiver_kind"], r["ablation"],
            f"{r['p_wrong_actual_regen']:.3f}", f"{r['p_wrong_ablated']:.3f}",
            f"{r['causal_propagation']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]", r["n_pairs"]] for r in out]
    write_latex_table(TABLES_DIR / "ablation_propagation.tex",
                      ["Model", "Dataset", "Turn", "Receiver", "Ablation", "P(wrong) actual", "P(wrong) ablated", "Causal effect [95\\% CI]", "n"],
                      tex, caption="Counterfactual message ablation: probability that an uninformed agent gives a wrong answer at $t{+}1$ "
                                   "when its prompt contains the misinformed agent's message (regenerated) vs. when that message is removed or "
                                   "replaced by an uninformed message. 'first' = the first receiver after the misinformed agent (direct effect); "
                                   "'later' receivers include spill-over. Paired bootstrap CIs over (debate, receiver).",
                      label="tab:ablation_propagation", col_spec="lllllrrlr")
    print(f"wrote {len(out)} ablation rows -> {RESULTS_DIR / 'ablation_propagation.csv'}")


def main() -> None:
    cfg = load_experiment_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_name", "--model", dest="model_name", default="openai/gpt-oss-120b")
    ap.add_argument("--endpoint_url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--no_autostart", action="store_true")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--datasets", default="cwq,ethics,winogrande")
    ap.add_argument("--category", default="false_fact")
    ap.add_argument("--turn", type=int, default=1, help="regenerate turn+1 responses")
    ap.add_argument("--n_debates", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_tokens", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--concurrent_requests", type=int, default=64)
    ap.add_argument("--raw_out", default=str(RESULTS_DIR / "ablation_propagation_raw.jsonl"))
    ap.add_argument("--analyze_only", action="store_true")
    ap.add_argument("--n_boot", type=int, default=10_000)
    args = ap.parse_args()
    ensure_dirs()
    if args.analyze_only:
        analyze(Path(args.raw_out), args.n_boot)
        return

    model_name = normalize_model_name(args.model_name)
    model_dir = next((d for d, disp in MODEL_DIRS.items() if disp == model_display(model_name)), model_name.replace("/", "_"))
    endpoint = None if args.mock else resolve_endpoint(args.endpoint_url, model_name, args.no_autostart)
    ledger = TokenLedger()
    client = ChatClient(endpoint, model_name, mock=args.mock, ledger=ledger,
                        mock_answer_fn=lambda msgs: "[DISAGREE] I think the answer is B.\nFinal Solution: B")
    low = model_name.lower()
    max_tokens = args.max_tokens or (int(cfg["debate"]["max_tokens_reasoning_models"]) if ("glm" in low or "gpt-oss" in low)
                                     else int(cfg["debate"]["max_tokens"]))
    writer = JsonlWriter(Path(args.raw_out))

    for ds_raw in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        ds = canonical_dataset(ds_raw)
        debates = select_debates(model_dir, ds, args.category, args.turn, args.n_debates, args.seed)
        print(f"[{ds}] {len(debates)} debates with successful injection at turn {args.turn}")
        k0_msgs = load_k0_messages(model_dir, ds, {})
        jobs = []
        for d in debates:
            rec, it = d["record"], d["item"]
            positions = {p["agentId"]: i for i, p in enumerate(rec["personas"])}
            mis_pos = positions[d["mis_id"]]
            inp = (rec.get("input") or [""])[0]
            uninformed_msgs_t = [m for m in rec["globalMemory"] if int(m["turn"]) == args.turn and m["agent_id"] != d["mis_id"]]
            repl_text, repl_src = None, None
            if inp in k0_msgs:
                repl_text, repl_src = k0_msgs[inp], "k0_run"
            elif uninformed_msgs_t:
                repl_text, repl_src = uninformed_msgs_t[0]["message"], "same_debate_peer"
            receivers = [m for m in rec["globalMemory"] if int(m["turn"]) == args.turn + 1 and m["agent_id"] != d["mis_id"]]
            receivers.sort(key=lambda m: int(m["message_id"]))
            for ri, rmsg in enumerate(receivers):
                variants = build_variants(rec, rmsg, d["mis_persona"], d["mis_id"], repl_text, repl_src or "")
                observed_wrong, _ = score(it, rmsg.get("solution"))
                for variant, msgs in variants.items():
                    jobs.append((d, rmsg, positions[rmsg["agent_id"]], "first" if ri == 0 else "later", variant, msgs,
                                 observed_wrong, repl_src))

        def _one(job):
            d, rmsg, rpos, kind, variant, msgs, observed_wrong, repl_src = job
            res = client.chat(msgs, max_tokens=max_tokens, temperature=args.temperature if args.temperature is not None else 1.0,
                              seed=stable_seed(args.seed, d["item"]["mint_id"], rpos, variant), label=f"ablation_{ds}")
            is_wrong, status = score(d["item"], res.text)
            return {"model": model_display(model_name), "dataset": ds, "category": args.category, "mint_id": d["item"]["mint_id"],
                    "turn": args.turn, "receiver_index": rpos, "receiver_kind": kind, "variant": variant,
                    "replacement_source": repl_src if variant == "replaced" else None, "response": res.text,
                    "finish_reason": res.finish_reason, "status": status, "is_wrong": is_wrong, "observed": observed_wrong,
                    "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens}

        with ThreadPoolExecutor(max_workers=max(1, args.concurrent_requests)) as ex:
            futs = [ex.submit(_one, j) for j in jobs]
            done = 0
            for fut in as_completed(futs):
                writer.write(fut.result())
                done += 1
                if done % 200 == 0 or done == len(futs):
                    print(f"  {done}/{len(futs)}")
    write_run_metadata(RESULTS_DIR / f"ablation_run_{model_display(model_name)}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json",
                       args, {"tokens": ledger.as_dict()})
    analyze(Path(args.raw_out), args.n_boot)


if __name__ == "__main__":
    main()
