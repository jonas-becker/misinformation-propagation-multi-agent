#!/usr/bin/env python3
"""
exp_mitigation_verifier.py - Verifier-agent mitigation, evaluated on stored
debate transcripts by re-running only the final stage.

For each 1-of-3 debate (exp2 or exp_exposure results) the final stage is repeated:
  1. a verifier (same model) reads the task, input and the whole discussion and lists
     claims that are not supported by the task input (prompts/mitigation_verifier.txt);
  2. every agent produces its final answer with MALLM's final-answer prompt, extended
     by the verifier's note;
  3. every agent votes with MALLM's anonymous voting prompt; plurality wins (ties ->
     no decision, counted as incorrect, as in the re-scored main results).
A control re-run of steps 2-3 WITHOUT the verifier note is produced from the same
transcripts so that the mitigation effect is measured against a regenerated
baseline rather than the stored vote.

Outputs: out/<model>/mitigation/mitigation_verifier_results_<dataset>_seed<seed>.json
(exp2-compatible; condition keys use exposure_common.condition_key with mit=verifier /
mit=none_regen) and out/results/mitigation_verifier.csv (+ LaTeX) with paired CIs.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from answer_extraction import extract_free_form_answer, extract_mc_answer, free_form_matches
from exposure_common import condition_key
from mint_data import canonical_dataset, load_mint_items
from model_server_config import normalize_model_name
from runner_common import ChatClient, TokenLedger, load_experiment_config, resolve_endpoint, stable_seed, write_run_metadata
from scripts.common import MODEL_DIRS, ensure_dirs, model_display

REPO_ROOT = Path(__file__).resolve().parent
VERIFIER_PROMPT = (REPO_ROOT / "prompts" / "mitigation_verifier.txt").read_text(encoding="utf-8").strip()


def final_answer_prompt(task: str, input_str: str, previous: str, persona: str, persona_description: str,
                        verifier_note: Optional[str]) -> List[dict]:
    """MALLM ResponseGenerator.generate_final_answer_prompt (+ optional verifier note)."""
    msgs = [{"role": "system", "content": f"Your role: {persona} ({persona_description})"},
            {"role": "system", "content": "You are tasked with creating a final solution based on the given input and your previous response."},
            {"role": "user", "content": f"Task: {task}\nInput: {input_str}\nYour previous response: {previous}"}]
    if verifier_note:
        msgs.append({"role": "user", "content": "A verifier reviewed the discussion and reports the following about claims that "
                                                f"are not supported by the task input:\n{verifier_note}\nTake this into account."})
    msgs.append({"role": "user", "content": "Extract the final solution to the task from the provided text. Remove statements of "
                                            "agreement, disagreement, and explanations. Do not modify the text. Do not output any text "
                                            "besides the solution. If there is no solution provided, just copy the previous response."})
    return msgs


def voting_prompt(task: str, input_str: str, persona: str, persona_description: str, solutions: List[str],
                  verifier_note: Optional[str]) -> List[dict]:
    """MALLM ResponseGenerator.generate_voting_prompt (anonymous) (+ optional verifier note)."""
    msgs = [{"role": "system", "content": f"Your role: {persona} ({persona_description})"}]
    if verifier_note:
        msgs.append({"role": "user", "content": f"Verifier report on unsupported claims in the discussion:\n{verifier_note}"})
    msgs.append({"role": "user", "content": "You are tasked with voting for the best solution from the list provided below based on the given task.\n"
                                            f"Task: {task}\nQuestion: {input_str}\n\nHere are the possible solutions:"})
    for i, s in enumerate(solutions):
        msgs.append({"role": "user", "content": f"Solution {i}: {s}"})
    msgs.append({"role": "user", "content": "Based on the above solutions, please provide the number of the solution you are voting for. "
                                            "Answer only with the number."})
    return msgs


def parse_vote(text: str, n: int) -> Optional[int]:
    nums = re.findall(r"\d+", text or "")
    if not nums:
        return None
    v = int(nums[-1])
    return v if 0 <= v < n else None


def remove_duplicates(answers: List[str]) -> List[str]:
    """MALLM DecisionProtocol.remove_duplicate_answers."""
    seen_lower, uniq = [], []
    for a in answers:
        if a.lower() not in seen_lower:
            seen_lower.append(a.lower())
            uniq.append(a)
    out, prefixes = [], set()
    for a in uniq:
        p = a.split(")")[0] if ")" in a else a
        if p not in prefixes:
            out.append(a)
            prefixes.add(p)
    return out


def main() -> None:
    cfg = load_experiment_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_name", "--model", dest="model_name", default="openai/gpt-oss-120b")
    ap.add_argument("--endpoint_url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--no_autostart", action="store_true")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--datasets", default="winogrande,cwq")
    ap.add_argument("--category", default="false_fact")
    ap.add_argument("--source", choices=("exp2", "exposure"), default="exp2", help="which transcripts to re-run")
    ap.add_argument("--exposure_condition", default=None, help="condition key when --source exposure (default: k=1 false_fact)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_items", type=int, default=None)
    ap.add_argument("--max_tokens", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--concurrent_requests", type=int, default=64)
    ap.add_argument("--output_dir", default=None)
    args = ap.parse_args()
    ensure_dirs()

    model_name = normalize_model_name(args.model_name)
    disp = model_display(model_name)
    model_dir = next((d for d, dd in MODEL_DIRS.items() if dd == disp), model_name.replace("/", "_"))
    endpoint = None if args.mock else resolve_endpoint(args.endpoint_url, model_name, args.no_autostart)
    ledger = TokenLedger()

    def _mock(msgs: List[dict]) -> str:
        last = msgs[-1]["content"]
        if "number of the solution" in last:
            return "0"
        if "unsupported" in last.lower() or "verifier" in msgs[0]["content"].lower():
            return "Claim: 'Nelson was typically the driver' - not verifiable from the input; doubtful."
        return "A) William"

    client = ChatClient(endpoint, model_name, mock=args.mock, ledger=ledger, mock_answer_fn=_mock)
    low = model_name.lower()
    max_tokens = args.max_tokens or (int(cfg["debate"]["max_tokens_reasoning_models"]) if ("glm" in low or "gpt-oss" in low)
                                     else int(cfg["debate"]["max_tokens"]))
    temp = args.temperature if args.temperature is not None else 1.0
    out_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "out" / model_dir / "mitigation"
    out_dir.mkdir(parents=True, exist_ok=True)

    for ds_raw in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        ds = canonical_dataset(ds_raw)
        items = {it["sentence"]: it for it in load_mint_items(ds)}
        if args.source == "exp2":
            data = json.load(open(REPO_ROOT / "out" / model_dir / f"exp2_results_{ds}.json", encoding="utf-8"))
            records = [r["mallm_log"] for r in data[args.category]["results"]]
        else:
            paths = sorted((REPO_ROOT / "out" / model_dir / "exposure").glob(f"exposure_results_{ds}_seed*.json"))
            data = json.load(open(paths[0], encoding="utf-8"))
            cond = args.exposure_condition or condition_key(args.category, 1, 3, seed=args.seed)
            records = [r["mallm_log"] for r in data[cond]["results"]]
        if args.max_items:
            records = records[: args.max_items]
        print(f"[{ds}] re-running the final stage for {len(records)} debates (with and without verifier)")

        def _run_debate(rec: dict) -> Optional[dict]:
            inp = (rec.get("input") or [""])[0]
            it = items.get(inp.split("\nOptions:")[0])
            if it is None:
                return None
            task = rec.get("instruction") or ""
            personas = rec.get("personas") or []
            # last solution per agent (agreements list holds the final held solutions)
            last_sol = {a["agent_id"]: a["solution"] for a in (rec.get("agreements") or [])}
            history = "\n".join(f"{m['persona']}: {m['message']}" for m in rec.get("globalMemory") or [])
            mid = it["mint_id"]
            # verifier
            vres = client.chat([{"role": "system", "content": VERIFIER_PROMPT},
                                {"role": "user", "content": f"Task: {task}\nInput: {inp}\n\nDiscussion:\n{history}"}],
                               max_tokens=max_tokens, temperature=temp, seed=stable_seed(args.seed, mid, "verifier"), label=f"verifier_{ds}")
            note = vres.text.strip() or "No unsupported claims."
            out = {"mint_id": mid, "verifier_note": note}
            for variant, vnote in (("verifier", note), ("none_regen", None)):
                finals = []
                for i, p in enumerate(personas):
                    prev = last_sol.get(p["agentId"]) or ""
                    fr = client.chat(final_answer_prompt(task, inp, prev, p["persona"], p["personaDescription"], vnote),
                                     max_tokens=max_tokens, temperature=temp, seed=stable_seed(args.seed, mid, variant, "final", i),
                                     label=f"final_{ds}")
                    finals.append(fr.text.strip() or prev)
                uniq = remove_duplicates(finals)
                votes = []
                for i, p in enumerate(personas):
                    vr = client.chat(voting_prompt(task, inp, p["persona"], p["personaDescription"], uniq, vnote),
                                     max_tokens=max_tokens, temperature=temp, seed=stable_seed(args.seed, mid, variant, "vote", i), label=f"vote_{ds}")
                    v = parse_vote(vr.text, len(uniq))
                    if v is not None:
                        votes.append(v)
                winner, tie = None, False
                if votes:
                    mc = Counter(votes).most_common()
                    winner = mc[0][0]
                    tie = len(mc) > 1 and mc[1][1] == mc[0][1]
                final = uniq[winner] if (winner is not None and not tie) else ""
                if it["is_multiple_choice"]:
                    ex = extract_mc_answer(final, it["options"])
                    ok = ex.status == "parsed" and ex.index == it["gold_index"]
                else:
                    ex = extract_free_form_answer(final)
                    ok = ex.status == "parsed" and free_form_matches(ex.text, [it["gold_text"]] + list(it["gold_aliases"]))
                out[variant] = {"final_answers": finals, "unique": uniq, "votes": votes, "tie": tie, "final": final,
                                "is_correct": bool(ok), "status": ex.status}
            out["item"] = it
            out["record"] = rec
            return out

        results = []
        with ThreadPoolExecutor(max_workers=max(1, args.concurrent_requests)) as ex_:
            futs = [ex_.submit(_run_debate, r) for r in records]
            done = 0
            for fut in as_completed(futs):
                r = fut.result()
                if r:
                    results.append(r)
                done += 1
                if done % 50 == 0 or done == len(futs):
                    print(f"  {done}/{len(futs)}")

        agg_path = out_dir / f"mitigation_verifier_results_{ds}_seed{args.seed}.json"
        agg = {}
        for variant in ("verifier", "none_regen"):
            cond = condition_key(args.category, 1, 3, "misinformation", "shared", variant, args.seed)
            res_list = []
            for r in results:
                it = r["item"]
                v = r[variant]
                ml = dict(r["record"])
                ml["finalAnswer"] = v["final"]
                ml["verifier_note"] = r["verifier_note"] if variant == "verifier" else None
                ml["regenerated_final_stage"] = v
                res_list.append({"sentence": (ml.get("input") or [""])[0], "mint_id": it["mint_id"], "response": v["final"],
                                 "predicted_answer": v["final"], "correct_answer": it["gold_text"], "is_correct": v["is_correct"],
                                 "is_multiple_choice": it["is_multiple_choice"], "options": it["options"],
                                 "misinformation_strategy": args.category, "mallm_log": ml})
            c = sum(1 for x in res_list if x["is_correct"])
            agg[cond] = {"accuracy": c / len(res_list) if res_list else 0.0, "correct_count": c, "total_count": len(res_list),
                         "results": res_list, "config": {"dataset": ds, "variant": variant, "source": args.source, "seed": args.seed,
                                                         "model_name": model_name, "max_tokens": max_tokens, "temperature": args.temperature,
                                                         "timestamp_utc": datetime.now(timezone.utc).isoformat()}}
        with open(agg_path, "w", encoding="utf-8") as f:
            json.dump(agg, f)
    write_run_metadata(out_dir / f"run_metadata_verifier_seed{args.seed}.json", args, {"tokens": ledger.as_dict()})


if __name__ == "__main__":
    main()
