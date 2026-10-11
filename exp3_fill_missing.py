#!/usr/bin/env python3
"""
exp3_fill_missing.py - Complete the five-agent WinoGrande composition sweep (exp3) for the
questions that never produced a debate, and de-duplicate the sweep file.

Why. MALLM replaces a failed discussion by re-running the FIRST entries of its input list
(mallm/scheduler.py: "Resampling from the dataset as a substitute"). exp3.py does not set
Config.num_samples, so the substitutes are questions that were already debated: the record
count still comes out at 382 per mix, but some questions appear twice and the failed ones never
appear. Llama-3.3-70B was served with an 8,192-token context, which five-agent memory-paradigm
debates exceed, so 176 of its debates failed this way (0/5: 5, 1/5: 28, 2/5: 35, 3/5: 31,
4/5: 32, 5/5: 45); GLM-4.7 and GPT-OSS (65k / 32k context) have 382 distinct questions in
every mix.

Sub-commands
  plan   Which questions are missing per mix. Source: the raw sweep file
         (out/<model>/exp3_results_winogrande.json) or, when the transcripts are not on this
         machine, a rescored mad_final.csv (one row per debate x protocol with mint_id).
  run    Debate the missing questions with the ORIGINAL exp3 configuration (MALLM memory
         paradigm, simple_voting, 5 agents, max 5 turns, 1,024 tokens, plain false fact, the same
         per-question slot draw with seed 0) against a server whose context window fits
         five-agent debates (VLLM_MAX_MODEL_LEN=32768 for Llama). Config.num_samples is set so
         that MALLM cannot substitute a failure with an already-debated question; failures are
         retried in further passes instead. Output: out/<model>/exp3_fill/exp3_fill_results_winogrande.json
  merge  De-duplicate the raw sweep file (keep the first debate of every question, drop the
         substitution re-runs) and insert the fill-in debates, so that every mix has exactly one
         debate per question. The original file is backed up next to it and a JSON report lists
         every dropped and added record.

Typical use (Llama; the GPU part runs through slurm/run_chain.slurm, see slurm/chains/exp3_fill.sh):
  python exp3_fill_missing.py plan  --model_name meta-llama/Llama-3.3-70B-Instruct --from_rescored <mad_final.csv>
  python exp3_fill_missing.py run   --model_name meta-llama/Llama-3.3-70B-Instruct --endpoint_url $EP --no_autostart --continue
  python exp3_fill_missing.py merge --model_name meta-llama/Llama-3.3-70B-Instruct   # make exp3_fill_merge
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from answer_extraction import STATUS_PARSED, extract_mc_answer  # noqa: E402
from exp3 import _choose_misinformation, _safe_model_name, _stable_shuffle_informations  # noqa: E402
from mint_data import format_mc_input, load_mint_items  # noqa: E402
from model_server_config import normalize_model_name  # noqa: E402
from scripts.common import model_display  # noqa: E402

DATASET = "winogrande"
NUM_AGENTS = 5
SEED = 0
MAX_TURNS = 5
MAX_TOKENS = 1024  # MALLM Config default, which exp3.py relied on
CATEGORY = "false_fact"


def mix_key(k: int, n: int = NUM_AGENTS) -> str:
    return f"{k}_misinformed_{n - k}_informed"


def mix_k(key: str) -> Optional[int]:
    try:
        return int(key.split("_")[0])
    except Exception:
        return None


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm(s: Optional[str]) -> str:
    return " ".join(str(s or "").strip().split())


def record_sentence(rec: dict) -> str:
    """Question text of an exp3 record (the stored input carries '\\nOptions:\\nA) ...\\nB) ...')."""
    s = rec.get("sentence")
    if not s:
        ml = rec.get("mallm_log") or {}
        inputs = ml.get("input") or []
        s = inputs[0] if inputs else ""
    return _norm(str(s).split("\nOptions:")[0])


class Items:
    """MINT WinoGrande items indexed by mint_id, raw id and normalized sentence."""

    def __init__(self, data_dir: Optional[str] = None):
        self.items = load_mint_items(DATASET, data_dir)
        self.by_mint_id = {it["mint_id"]: it for it in self.items}
        self.by_raw_id = {str(it["raw"].get("id")): it for it in self.items if it["raw"].get("id")}
        self.by_sentence = {_norm(it["sentence"]): it for it in self.items}
        self.order = {it["mint_id"]: i for i, it in enumerate(self.items)}

    def for_record(self, rec: dict) -> Optional[dict]:
        it = self.by_sentence.get(record_sentence(rec))
        if it is None:
            ml = rec.get("mallm_log") or {}
            it = self.by_raw_id.get(str(ml.get("exampleId")))
        if it is None and rec.get("mint_id"):
            it = self.by_mint_id.get(rec["mint_id"])
        return it


def model_paths(model_name: str, out_root: Optional[str] = None) -> dict:
    model_dir = _safe_model_name(model_name)
    base = Path(out_root) if out_root else REPO_ROOT / "out"
    fill_dir = base / model_dir / "exp3_fill"
    return {
        "model_dir": model_dir,
        "raw": base / model_dir / f"exp3_results_{DATASET}.json",
        "fill_dir": fill_dir,
        "io_dir": fill_dir / "io",
        "plan": fill_dir / "plan.json",
        "fill": fill_dir / f"exp3_fill_results_{DATASET}.json",
    }


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------------------------
# plan
# ----------------------------------------------------------------------------------------------
def presence_from_raw(raw: dict, items: Items) -> Dict[str, dict]:
    """Per mix: first-occurrence question list, duplicates and unmatched records of a raw exp3 file."""
    out: Dict[str, dict] = {}
    for key, payload in raw.items():
        if mix_k(key) is None or not isinstance(payload, dict):
            continue
        seen: Dict[str, int] = {}
        dups: List[str] = []
        unmatched = 0
        for i, rec in enumerate(payload.get("results") or []):
            it = items.for_record(rec)
            if it is None:
                unmatched += 1
                continue
            mid = it["mint_id"]
            if mid in seen:
                dups.append(mid)
            else:
                seen[mid] = i
        out[key] = {"present": list(seen), "duplicates": dups, "n_records": len(payload.get("results") or []),
                    "n_unmatched": unmatched}
    return out


def presence_from_rescored(csv_path: Path, model_name: str) -> Dict[str, dict]:
    """Per mix: distinct mint_ids of the exp3 voting rows of this model in a rescored mad_final.csv."""
    disp = model_display(_safe_model_name(model_name))
    per_mix: Dict[str, List[str]] = defaultdict(list)
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("experiment") != "exp3" or r.get("protocol") != "voting" or r.get("dataset") != DATASET:
                continue
            if r.get("model") != disp:
                continue
            per_mix[r["condition"]].append(r["mint_id"])
    out: Dict[str, dict] = {}
    for key, ids in per_mix.items():
        seen: List[str] = []
        dups: List[str] = []
        for mid in ids:
            (dups if mid in seen else seen).append(mid)
        out[key] = {"present": seen, "duplicates": dups, "n_records": len(ids), "n_unmatched": 0}
    return out


def cmd_plan(args) -> int:
    model_name = normalize_model_name(args.model_name)
    paths = model_paths(model_name, args.out_root)
    items = Items(args.data_dir)
    if args.from_rescored:
        src = Path(args.from_rescored)
        presence = presence_from_rescored(src, model_name)
        source_kind = "rescored_csv"
    else:
        src = Path(args.source) if args.source else paths["raw"]
        if not src.is_file():
            print(f"[ERROR] raw exp3 file not found: {src} (pass --source or --from_rescored)")
            return 2
        presence = presence_from_raw(load_json(src), items)
        source_kind = "raw_exp3"
    if not presence:
        print(f"[ERROR] no exp3 {DATASET} records for {model_name} in {src}")
        return 2
    all_ids = [it["mint_id"] for it in items.items]
    mixes = {}
    total_missing = 0
    for k in range(0, args.num_agents + 1):
        key = mix_key(k, args.num_agents)
        p = presence.get(key)
        if p is None:
            print(f"[WARN] {key}: no records in the source; every question counts as missing")
            p = {"present": [], "duplicates": [], "n_records": 0, "n_unmatched": 0}
        present = set(p["present"])
        missing = [m for m in all_ids if m not in present]
        total_missing += len(missing)
        mixes[key] = {"k": k, "n_records": p["n_records"], "n_unique": len(present), "n_duplicates": len(p["duplicates"]),
                      "n_unmatched": p["n_unmatched"], "duplicate_mint_ids": sorted(set(p["duplicates"])),
                      "missing_mint_ids": missing}
        print(f"{key:28s} records={p['n_records']:4d} unique={len(present):4d} duplicated={len(set(p['duplicates'])):3d} "
              f"missing={len(missing):3d}")
    plan = {"model_name": model_name, "model_dir": paths["model_dir"], "dataset": DATASET, "num_agents": args.num_agents,
            "seed": args.seed, "category": CATEGORY, "n_items": len(all_ids), "source": str(src), "source_kind": source_kind,
            "created_utc": now_utc(), "total_missing": total_missing, "mixes": mixes}
    plan_path = Path(args.plan) if args.plan else paths["plan"]
    dump_json(plan_path, plan)
    print(f"[plan] {total_missing} missing debates over {len(mixes)} mixes -> {plan_path}")
    return 0


# ----------------------------------------------------------------------------------------------
# run
# ----------------------------------------------------------------------------------------------
def build_input(items: Items, mint_ids: List[str], k: int, n: int, seed: int) -> List[dict]:
    """MALLM InputExample list; identical to exp3._convert_winogrande_for_mix except that the gold
    option comes from mint_data (exp3's converter predates the v1.2 label format)."""
    out = []
    for mid in mint_ids:
        it = items.by_mint_id[mid]
        raw = it["raw"]
        misinfo = _choose_misinformation(raw)
        informations = [misinfo] * k + [None] * (n - k)
        informations = _stable_shuffle_informations(informations, str(raw.get("id") or mid), k, seed)
        out.append({
            "example_id": str(raw.get("id") or mid),
            "dataset_id": DATASET,
            "inputs": [format_mc_input(it)],
            "context": None,
            "references": [it["gold_text"]] if it["gold_text"] else [],
            "metadata": {"misinformed_count": k, "informed_count": n - k, "num_agents": n, "mint_id": mid,
                         "exp3_fill": True, "seed": seed},
            "informations": informations,
        })
    return out


def score_record(rec: dict, it: dict, extra: dict) -> dict:
    final = rec.get("finalAnswer")
    predicted = str(final) if final is not None else ""
    ex = extract_mc_answer(predicted, it["options"])
    is_correct = bool(ex.status == STATUS_PARSED and ex.index == it["gold_index"])
    return {
        "sentence": (rec.get("input") or [""])[0],
        "mint_id": it["mint_id"],
        "response": predicted,
        "predicted_answer": ex.text,
        "extraction_status": ex.status,
        "correct_answer": it["gold_text"],
        "is_correct": is_correct,
        "is_multiple_choice": True,
        "options": it["options"],
        "misinformation_strategy": CATEGORY,
        "mallm_log": rec,
        "exp3_fill": extra,
    }


def server_max_model_len(endpoint_url: str) -> Optional[int]:
    try:
        import requests

        r = requests.get(endpoint_url.rstrip("/") + "/models", timeout=10)
        for m in (r.json().get("data") or []):
            if m.get("max_model_len"):
                return int(m["max_model_len"])
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] could not read max_model_len from {endpoint_url}: {e}")
    return None


def aggregate(results: List[dict]) -> dict:
    correct = sum(1 for r in results if r.get("is_correct"))
    return {"accuracy": (correct / len(results)) if results else 0.0, "correct_count": correct,
            "total_count": len(results), "results": results}


def cmd_run(args) -> int:
    model_name = normalize_model_name(args.model_name)
    paths = model_paths(model_name, args.out_root)
    plan_path = Path(args.plan) if args.plan else paths["plan"]
    if not plan_path.is_file():
        print(f"[ERROR] plan not found: {plan_path} (run `exp3_fill_missing.py plan` first)")
        return 2
    plan = load_json(plan_path)
    if plan.get("model_name") != model_name:
        print(f"[ERROR] plan is for {plan.get('model_name')}, not {model_name}")
        return 2
    n = int(plan.get("num_agents", NUM_AGENTS))
    seed = int(plan.get("seed", SEED))
    items = Items(args.data_dir)

    import exp2 as exp2_base  # endpoint discovery / timeout patch, as the other runners

    exp2_base._ensure_mallm_on_path()
    if args.mock:
        import exp3 as exp3_base

        exp3_base._install_runtime_stubs()
        endpoint = "mock://local"
        ctx = None
    else:
        from runner_common import resolve_endpoint

        endpoint = resolve_endpoint(args.endpoint_url, model_name, args.no_autostart)
        exp2_base._patch_mallm_openai_timeout(timeout_seconds=int(os.environ.get("MALLM_OPENAI_TIMEOUT_SECONDS", "1800")),
                                              max_retries=5)
        ctx = server_max_model_len(endpoint)
        print(f"[INFO] server max_model_len={ctx}")
        if ctx is not None and ctx < args.min_context and not args.allow_short_context:
            print(f"[ERROR] the server context window ({ctx} tokens) is smaller than --min_context {args.min_context}; "
                  "the missing debates failed on exactly this limit. Serve with VLLM_MAX_MODEL_LEN>=16384 "
                  "(or pass --allow_short_context).")
            return 3
    from mallm import scheduler
    from mallm.utils.config import Config

    fill_path = Path(args.output) if args.output else paths["fill"]
    io_dir = paths["io_dir"]
    io_dir.mkdir(parents=True, exist_ok=True)
    existing = load_json(fill_path) if (args.continue_mode and fill_path.is_file()) else {}
    conc = args.concurrent_api_requests or int(os.environ.get("MALLM_CONCURRENT_API_REQUESTS", "250"))
    job_id = os.environ.get("SLURM_JOB_ID")
    print(f"[INFO] exp3 fill model={model_name} endpoint={endpoint} plan={plan_path} out={fill_path} "
          f"max_tokens={args.max_tokens} max_turns={args.max_turns} concurrency={conc}")

    still_missing: Dict[str, List[str]] = {}
    for key, info in plan["mixes"].items():
        k = int(info["k"])
        missing = list(info["missing_mint_ids"])
        if args.mixes and k not in args.mixes:
            continue
        prev = existing.get(key) or {}
        results: List[dict] = list(prev.get("results") or [])
        done = {r.get("mint_id") for r in results}
        todo = [m for m in missing if m not in done]
        if args.max_items:
            todo = todo[: args.max_items]
        passes: List[dict] = list((prev.get("config") or {}).get("passes") or [])
        print(f"[MIX] {key}: {len(missing)} missing, {len(done)} already filled, {len(todo)} to run")
        pass_no = len(passes)
        while todo and pass_no < args.max_passes:
            pass_no += 1
            mallm_in = build_input(items, todo, k, n, seed)
            in_path = io_dir / f"exp3_fill_input_{DATASET}_{key}_pass{pass_no}_{job_id or 'local'}.json"
            out_path = io_dir / f"exp3_fill_output_{DATASET}_{key}_pass{pass_no}_{job_id or 'local'}.json"
            dump_json(in_path, mallm_in)
            if out_path.exists():
                out_path.unlink()
            cfg = Config(
                input_json_file_path=str(in_path),
                output_json_file_path=str(out_path),
                task_instruction_prompt="",
                task_instruction_prompt_template=DATASET,
                endpoint_url=endpoint,
                model_name=model_name,
                discussion_paradigm="memory",
                decision_protocol="simple_voting",
                max_turns=args.max_turns,
                num_agents=n,
                max_tokens=args.max_tokens,
                agent_generator="informed",
                agent_generators_list=["informed"] * n,
                use_chain_of_thought=False,
                concurrent_api_requests=conc,
                shuffle_input_samples=False,
                # every input example is processed exactly once: nothing is left in MALLM's data list to
                # "substitute" a failed discussion with (that substitution produced the duplicates)
                num_samples=len(mallm_in),
            )
            print(f"[RUN] {key} pass {pass_no}: {len(mallm_in)} debates")
            t0 = datetime.now(timezone.utc)
            error = None
            try:
                scheduler.Scheduler(cfg).run()
            except Exception as e:  # noqa: BLE001 - MALLM raises once failures cannot be substituted
                error = f"{type(e).__name__}: {e}"
                print(f"[WARN] {key} pass {pass_no}: MALLM stopped with {error}; keeping the completed debates")
            recs = load_json(out_path) if out_path.is_file() else []
            got = 0
            for rec in recs:
                it = items.by_raw_id.get(str(rec.get("exampleId"))) or items.by_mint_id.get(
                    ((rec.get("metadata") or {}).get("mint_id")))
                if it is None or it["mint_id"] in done:
                    continue
                results.append(score_record(rec, it, {"pass": pass_no, "slurm_job_id": job_id, "server_max_model_len": ctx,
                                                       "timestamp_utc": now_utc()}))
                done.add(it["mint_id"])
                got += 1
            todo = [m for m in todo if m not in done]
            passes.append({"pass": pass_no, "n_requested": len(mallm_in), "n_completed": got, "failed_mint_ids": todo,
                           "error": error, "started_utc": t0.isoformat(), "finished_utc": now_utc(),
                           "input": str(in_path), "output": str(out_path)})
            print(f"[PASS] {key} pass {pass_no}: {got}/{len(mallm_in)} completed; {len(todo)} still missing")
            results.sort(key=lambda r: items.order.get(r.get("mint_id"), 10**9))
            agg = aggregate(results)
            agg["config"] = {
                "dataset": DATASET, "num_agents": n, "misinformed_count": k, "informed_count": n - k, "seed": seed,
                "category": CATEGORY, "model_name": model_name, "endpoint_url": endpoint, "max_tokens": args.max_tokens,
                "max_turns": args.max_turns, "concurrent_api_requests": conc, "server_max_model_len": ctx,
                "vllm_max_model_len_env": os.environ.get("VLLM_MAX_MODEL_LEN"), "slurm_job_id": job_id,
                "plan": str(plan_path), "n_missing_in_plan": len(missing), "passes": passes, "timestamp_utc": now_utc(),
            }
            existing[key] = agg
            dump_json(fill_path, existing)
        if todo:
            still_missing[key] = todo
    print(f"[DONE] fill results -> {fill_path}")
    if still_missing:
        for key, ids in still_missing.items():
            print(f"[WARN] {key}: {len(ids)} debates still missing after {args.max_passes} passes: {ids}")
        return 0 if args.allow_incomplete else 1
    return 0


# ----------------------------------------------------------------------------------------------
# merge
# ----------------------------------------------------------------------------------------------
def cmd_merge(args) -> int:
    model_name = normalize_model_name(args.model_name)
    paths = model_paths(model_name, args.out_root)
    src = Path(args.source) if args.source else paths["raw"]
    fill_path = Path(args.fill) if args.fill else paths["fill"]
    if not src.is_file():
        print(f"[ERROR] raw exp3 file not found: {src}")
        return 2
    items = Items(args.data_dir)
    raw = load_json(src)
    fill = load_json(fill_path) if fill_path.is_file() else {}
    if not fill and not args.dedupe_only:
        print(f"[ERROR] fill results not found: {fill_path} (run the fill first, or pass --dedupe_only)")
        return 2
    all_ids = [it["mint_id"] for it in items.items]
    report = {"model_name": model_name, "source": str(src), "fill": str(fill_path) if fill else None,
              "merged_utc": now_utc(), "mixes": {}}
    merged = dict(raw)
    incomplete = False
    for key in sorted(raw, key=lambda s: (mix_k(s) is None, mix_k(s) or 0)):
        payload = raw[key]
        if mix_k(key) is None or not isinstance(payload, dict):
            continue
        kept: List[dict] = []
        first: Dict[str, dict] = {}
        dropped: List[dict] = []
        unmatched = 0
        for i, rec in enumerate(payload.get("results") or []):
            it = items.for_record(rec)
            if it is None:
                unmatched += 1
                kept.append(rec)  # never silently drop a record we cannot attribute
                continue
            mid = it["mint_id"]
            ml = rec.get("mallm_log") or {}
            if mid in first:
                dropped.append({"mint_id": mid, "record_index": i, "exampleId": ml.get("exampleId"),
                                "clockSeconds": ml.get("clockSeconds"), "is_correct": rec.get("is_correct"),
                                "kept_clockSeconds": (first[mid].get("mallm_log") or {}).get("clockSeconds"),
                                "kept_is_correct": first[mid].get("is_correct")})
                continue
            first[mid] = rec
            kept.append(rec)
        fill_recs = {r.get("mint_id"): r for r in ((fill.get(key) or {}).get("results") or [])}
        added: List[str] = []
        missing_after: List[str] = []
        for mid in all_ids:
            if mid in first:
                continue
            fr = fill_recs.get(mid)
            if fr is None:
                missing_after.append(mid)
                continue
            kept.append(fr)
            added.append(mid)
        agg = aggregate(kept)
        cfg = dict(payload.get("config") or {})
        cfg["expected_total_count"] = len(all_ids)
        cfg["exp3_fill"] = {"deduplicated_records": len(dropped), "added_records": len(added), "unmatched_records": unmatched,
                            "still_missing": missing_after, "fill_file": str(fill_path) if fill else None,
                            "fill_config": (fill.get(key) or {}).get("config"), "merged_utc": report["merged_utc"],
                            "source_backup": None}
        agg["config"] = cfg
        merged[key] = agg
        report["mixes"][key] = {"n_records_before": len(payload.get("results") or []), "n_unique_before": len(first),
                                "dropped_duplicates": dropped, "added": added, "still_missing": missing_after,
                                "n_records_after": len(kept), "accuracy_before": payload.get("accuracy"),
                                "accuracy_after": agg["accuracy"]}
        print(f"{key:28s} before={len(payload.get('results') or [])} unique={len(first)} dropped={len(dropped)} "
              f"added={len(added)} after={len(kept)} still_missing={len(missing_after)} unmatched={unmatched}")
        incomplete = incomplete or bool(missing_after)
    if incomplete and not args.allow_incomplete:
        print("[ERROR] some questions have no debate even after the fill; nothing written (pass --allow_incomplete to write anyway)")
        return 1
    out = Path(args.out) if args.out else src
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    if out.resolve() == src.resolve() and not args.no_backup:
        backup = src.with_name(f"{src.stem}.pre_fill_{ts}.json")
        shutil.copy2(src, backup)
        for key in merged:
            if isinstance(merged[key], dict) and "exp3_fill" in (merged[key].get("config") or {}):
                merged[key]["config"]["exp3_fill"]["source_backup"] = str(backup)
        report["source_backup"] = str(backup)
        print(f"[backup] {backup}")
    dump_json(out, merged)
    rep_path = paths["fill_dir"] / f"merge_report_{ts}.json"
    dump_json(rep_path, report)
    print(f"[merge] {out}\n[report] {rep_path}")
    return 0


# ----------------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--model_name", "--model", dest="model_name", default="meta-llama/Llama-3.3-70B-Instruct")
        p.add_argument("--data_dir", default=None, help="MINT item directory (default: MINT_dataset_v1.2)")
        p.add_argument("--out_root", default=None, help="results root (default: out/)")

    p = sub.add_parser("plan", help="list the missing questions per mix")
    common(p)
    p.add_argument("--source", default=None, help="raw exp3 file (default out/<model>/exp3_results_winogrande.json)")
    p.add_argument("--from_rescored", default=None, help="rescored mad_final.csv instead of the raw file")
    p.add_argument("--plan", default=None, help="where to write the plan (default out/<model>/exp3_fill/plan.json)")
    p.add_argument("--num_agents", type=int, default=NUM_AGENTS)
    p.add_argument("--seed", type=int, default=SEED)

    p = sub.add_parser("run", help="debate the missing questions (needs a model endpoint)")
    common(p)
    p.add_argument("--endpoint_url", default="http://127.0.0.1:8080/v1")
    p.add_argument("--no_autostart", action="store_true")
    p.add_argument("--mock", action="store_true", help="MALLM mock endpoint (pipeline test)")
    p.add_argument("--plan", default=None)
    p.add_argument("--output", default=None, help="fill results file (default out/<model>/exp3_fill/exp3_fill_results_winogrande.json)")
    p.add_argument("--continue", dest="continue_mode", action="store_true", help="keep debates already in the fill file")
    p.add_argument("--mixes", type=lambda s: [int(x) for x in s.split(",") if x.strip()], default=None,
                   help="comma list of k values to run (default: all mixes of the plan)")
    p.add_argument("--max_passes", type=int, default=3, help="retry passes for debates that still fail")
    p.add_argument("--max_items", type=int, default=None, help="first N missing questions per mix (smoke tests)")
    p.add_argument("--max_tokens", type=int, default=MAX_TOKENS)
    p.add_argument("--max_turns", type=int, default=MAX_TURNS)
    p.add_argument("--concurrent_api_requests", type=int, default=None)
    p.add_argument("--min_context", type=int, default=16384, help="refuse servers with a smaller max_model_len")
    p.add_argument("--allow_short_context", action="store_true")
    p.add_argument("--allow_incomplete", action="store_true", help="exit 0 even if debates are still missing")

    p = sub.add_parser("merge", help="de-duplicate the raw file and insert the fill-in debates")
    common(p)
    p.add_argument("--source", default=None, help="raw exp3 file (default out/<model>/exp3_results_winogrande.json)")
    p.add_argument("--fill", default=None, help="fill results file")
    p.add_argument("--out", default=None, help="merged file (default: overwrite --source after a backup)")
    p.add_argument("--no_backup", action="store_true")
    p.add_argument("--dedupe_only", action="store_true", help="no fill file: only drop the duplicated debates")
    p.add_argument("--allow_incomplete", action="store_true", help="write even if questions remain without a debate")

    args = ap.parse_args()
    return {"plan": cmd_plan, "run": cmd_run, "merge": cmd_merge}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
