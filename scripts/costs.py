#!/usr/bin/env python3
"""
costs.py - Token and inference cost per protocol.

Sources
  * exact prompt/completion token counts where the runners record them
    (exp1 records, exp_msc samples, exp_message_ablation raw rows,
    run_metadata token ledgers);
  * for MALLM debates (exp2/exp3/exp_exposure) the transcripts contain the generated
    text but no usage counts, so completion tokens are estimated by tokenizing every
    stored message, final answer and vote with tiktoken (o200k_base; falls back to
    words x 1.3) and prompt tokens are reconstructed from the message context size
    (system prompt + visible history); these are marked as estimates.
Reports mean tokens per question and GPU-hour estimates per 100 questions for a
user-supplied throughput (--tokens_per_gpu_hour, default from the vLLM logs if given).

Outputs: out/results/costs.csv, out/tables/costs.tex
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.common import MODEL_DIRS, RESULTS_DIR, TABLES_DIR, ensure_dirs, write_csv, write_latex_table  # noqa: E402

try:
    from model_server_config import offline_tiktoken_encoding  # noqa: E402

    _ENC = offline_tiktoken_encoding("o200k_base")
    if _ENC is None:
        raise RuntimeError("offline tiktoken vocab not available")

    def ntok(s: str) -> int:
        return len(_ENC.encode(s or "", disallowed_special=()))
except Exception:  # noqa: BLE001
    _ENC = None

    def ntok(s: str) -> int:
        return int(len((s or "").split()) * 1.3)


def debate_tokens(ml: dict) -> Dict[str, float]:
    """Estimate prompt/completion tokens of one MALLM debate from its transcript."""
    comp = 0
    prompt = 0
    msgs = sorted(ml.get("globalMemory") or [], key=lambda m: int(m.get("message_id", 0)))
    for m in msgs:
        comp += ntok(m.get("message") or "")
        comp += ntok(m.get("solution") or "")  # the solution-extraction call output
        aa = m.get("additional_args") or {}
        ctx = ntok(aa.get("task_instruction") or "") + ntok(aa.get("input_str") or "") + ntok(aa.get("persona_description") or "") \
            + ntok(aa.get("current_draft") or "") + sum(ntok(h.get("content", "")) for h in (aa.get("agent_memory") or []))
        prompt += ctx + 60  # template overhead
        prompt += ctx + ntok(m.get("message") or "") + 60  # solution-extraction call sees the message again
    vet = ml.get("votesEachTurn") or {}
    for _t, v in vet.items():
        if not isinstance(v, dict):
            continue
        answers = v.get("answers") or []
        comp += sum(ntok(a) for a in answers) + 3 * len(answers)  # final answers + one-number votes
        prompt += len(answers) * (200 + sum(ntok(a) for a in answers))
    return {"prompt_tokens_est": prompt, "completion_tokens_est": comp, "n_generations": len(msgs) + 2 * len(vet) * 3}


# paper terms for the protocols of the cost table (the CSV keeps the raw labels)
PROTOCOL_NAMES = {"SA": "single agent", "SA (irrelevant-true)": "single agent (irrelevant true)", "mSC": "mSC (3 samples)",
                  "mSC (matched sampling)": "mSC (3 samples, T 0.6 / top-p 0.9)", "SC15": "SC15 (15 samples)", "MAD (exposure)": "debate, 3 agents (verified run set)",
                  "MAD (exp2, 3 agents)": "debate, 3 agents (full-sample)", "MAD (exp3, 5 agents)": "debate, 5 agents (sweep)"}
PROTOCOL_ORDER = list(PROTOCOL_NAMES)


def write_cost_table(rows: List[dict]) -> None:
    """Compact LaTeX table: mean per model x protocol (conditions pooled)."""
    agg2: Dict[tuple, List[dict]] = defaultdict(list)
    for r in rows:
        agg2[(r["model"], r["protocol"])].append(r)
    order = {m: i for i, m in enumerate(["Llama-3.3-70B", "GLM-4.7-Flash", "GPT-OSS-120B"])}
    tex = []
    for (model, proto), rs in sorted(agg2.items(), key=lambda x: (order.get(x[0][0], 9),
                                                                    PROTOCOL_ORDER.index(x[0][1]) if x[0][1] in PROTOCOL_ORDER else 99)):
        pt = sum(float(r["prompt_tokens_per_q"]) for r in rs) / len(rs)
        ct = sum(float(r["completion_tokens_per_q"]) for r in rs) / len(rs)
        g = sum(float(r["generations_per_q"]) for r in rs) / len(rs)
        exact = all(str(r["exact"]).strip().lower() in ("true", "1") for r in rs)
        tex.append([model, PROTOCOL_NAMES.get(proto, proto), f"{g:.1f}", f"{pt:,.0f}", f"{ct:,.0f}", f"{100 * (pt + ct) / 1000:,.0f}k",
                    "exact" if exact else "estimate"])
    write_latex_table(TABLES_DIR / "costs.tex", ["Model", "Protocol", "Generations / q", "Prompt tok / q", "Completion tok / q",
                                                 "Tokens / 100 q", "Source"], tex,
                      caption="Inference cost per question and protocol. Debate costs are estimated from the stored transcripts "
                              "(tiktoken o200k\\_base) because MALLM does not log usage; single-agent and self-consistency counts are "
                              "exact where recorded.",
                      label="tab:costs", col_spec="llrrrrl")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", default=str(REPO_ROOT / "out"))
    ap.add_argument("--tokens_per_gpu_hour", type=float, default=None,
                    help="measured generation throughput (tokens/GPU-hour) to convert tokens into GPU-hours")
    ap.add_argument("--max_debates", type=int, default=400, help="debates sampled per file for the estimate")
    ap.add_argument("--tables_only", action="store_true", help="rewrite costs.tex from out/results/costs.csv")
    args = ap.parse_args()
    ensure_dirs()
    if args.tables_only:
        from scripts.common import read_csv
        write_cost_table(read_csv(RESULTS_DIR / "costs.csv"))
        print("rewrote costs.tex")
        return
    out_dir = Path(args.out_dir)
    rows: List[dict] = []

    for mdir, model in MODEL_DIRS.items():
        base = out_dir / mdir
        if not base.is_dir():
            continue
        # exp1 (exact if recorded, else estimated from response text)
        for p in sorted(base.glob("exp1_results_*.json")) + sorted(base.glob("exp1a_results_*.json")):
            data = json.load(open(p, encoding="utf-8"))
            for cond, payload in data.items():
                res = payload.get("results") or []
                if not res:
                    continue
                exact = all(r.get("completion_tokens") is not None for r in res)
                pt = sum((r.get("prompt_tokens") or ntok(r.get("prompt") or "")) for r in res) / len(res)
                ct = sum((r.get("completion_tokens") or ntok(r.get("response") or "")) for r in res) / len(res)
                rows.append({"model": model, "protocol": "SA" if p.name.startswith("exp1_") else "SA (irrelevant-true)",
                             "file": p.name, "condition": cond, "n_questions": len(res), "prompt_tokens_per_q": pt,
                             "completion_tokens_per_q": ct, "generations_per_q": 1, "exact": exact})
        # self-consistency (15 samples): responses stored per sample
        for p in sorted(base.glob("exp2_ablation_results_*.json")):
            data = json.load(open(p, encoding="utf-8"))
            for cond, payload in data.items():
                res = payload.get("results") or []
                if not res:
                    continue
                ct = pt = 0.0
                for r in res:
                    samples = (r.get("sc_log") or {}).get("samples") or []
                    ct += sum(ntok(s.get("response") or "") for s in samples)
                    pt += len(samples) * (ntok(r.get("sentence") or "") + ntok((r.get("sc_log") or {}).get("information") or "") + 80)
                rows.append({"model": model, "protocol": "SC15", "file": p.name, "condition": cond, "n_questions": len(res),
                             "prompt_tokens_per_q": pt / len(res), "completion_tokens_per_q": ct / len(res), "generations_per_q": 15,
                             "exact": False})
        # mSC (exact from sample rows); msc_matched = the Llama re-run at the debate's sampling parameters
        msc_files = [(p, "mSC") for p in sorted((base / "msc").glob("msc_samples_*.jsonl"))] if (base / "msc").is_dir() else []
        if (base / "msc_matched").is_dir():
            msc_files += [(p, "mSC (matched sampling)") for p in sorted((base / "msc_matched").glob("msc_samples_*.jsonl"))]
        for p, proto_msc in msc_files:
            agg: Dict[str, List[dict]] = defaultdict(list)
            with open(p, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        agg[r["condition"]].append(r)
            for cond, rs in agg.items():
                qs = {r["mint_id"] for r in rs}
                rows.append({"model": model, "protocol": proto_msc, "file": p.name, "condition": cond, "n_questions": len(qs),
                             "prompt_tokens_per_q": sum(r.get("prompt_tokens") or 0 for r in rs) / max(1, len(qs)),
                             "completion_tokens_per_q": sum(r.get("completion_tokens") or 0 for r in rs) / max(1, len(qs)),
                             "generations_per_q": len(rs) / max(1, len(qs)), "exact": True})
        # debates (estimated from transcripts)
        debate_files = sorted(base.glob("exp2_results_*.json")) + sorted(base.glob("exp3_results_*.json"))
        if (base / "exposure").is_dir():
            debate_files += sorted((base / "exposure").glob("exposure_results_*.json"))
        for p in debate_files:
            data = json.load(open(p, encoding="utf-8"))
            for cond, payload in data.items():
                res = (payload.get("results") or [])[: args.max_debates]
                if not res:
                    continue
                est = [debate_tokens(r.get("mallm_log") or {}) for r in res]
                proto = "MAD (exp2, 3 agents)" if p.name.startswith("exp2") else ("MAD (exp3, 5 agents)" if p.name.startswith("exp3") else "MAD (exposure)")
                rows.append({"model": model, "protocol": proto, "file": p.name, "condition": cond, "n_questions": len(res),
                             "prompt_tokens_per_q": sum(e["prompt_tokens_est"] for e in est) / len(est),
                             "completion_tokens_per_q": sum(e["completion_tokens_est"] for e in est) / len(est),
                             "generations_per_q": sum(e["n_generations"] for e in est) / len(est), "exact": False})
    for r in rows:
        tot = r["prompt_tokens_per_q"] + r["completion_tokens_per_q"]
        r["total_tokens_per_100q"] = 100 * tot
        r["gpu_hours_per_100q"] = (100 * tot / args.tokens_per_gpu_hour) if args.tokens_per_gpu_hour else None
    write_csv(RESULTS_DIR / "costs.csv", rows)

    write_cost_table(rows)
    print(f"wrote {len(rows)} cost rows; tokenizer={'tiktoken' if _ENC else 'word-estimate'}")


if __name__ == "__main__":
    main()
