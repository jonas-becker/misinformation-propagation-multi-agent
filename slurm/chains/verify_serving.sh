#!/usr/bin/env bash
# GPU check of the serving setup (run through slurm/run_chain.slurm via CHAIN_SCRIPT; ~5 min after model load):
#   1. per-job port (not 8080) and the endpoint serves THIS job's model
#   2. host-side tiktoken loads offline in seconds
#   3. scripts/validate_mint.py judges 5 passages per dataset against the live model on a scratch copy
#      of MINT (real dataset and out/results untouched) and writes n_tokens + judge flags
#   4. the job must end right after this script (check from outside with sacct that the server cleanup does not hang)
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
V="out/_verify/job_${SLURM_JOB_ID:-local}"; rm -rf "${V}"; mkdir -p "${V}/MINT" "${V}/results"

check() {  # check <name> <command...>
    if "${@:2}"; then echo "[OK   $(date -Is)] $1"; else echo "[FAIL $(date -Is)] $1"; FAILED+=("$1"); fi
}
port="${ENDPOINT_URL##*:}"; port="${port%%/*}"
echo "endpoint=${ENDPOINT_URL} port=${port} TIKTOKEN_CACHE_DIR=${TIKTOKEN_CACHE_DIR:-<unset>}"
check "port is per-job (${port} != 8080)" test "${port}" != "8080"
check "endpoint lists ${MODEL_NAME}" python - <<'PY'
import os, requests
ids = [m["id"] for m in requests.get(os.environ["ENDPOINT_URL"].rstrip("/") + "/models", timeout=10).json()["data"]]
print("   served:", ids); raise SystemExit(0 if os.environ["MODEL_NAME"] in ids else 1)
PY
check "tiktoken o200k_base loads offline in < 10 s" timeout 60 python - <<'PY'
import socket, sys, time
socket.socket.connect = lambda self, *a: (_ for _ in ()).throw(OSError("network disabled for this test"))
sys.path.insert(0, ".")
from model_server_config import offline_tiktoken_encoding
t = time.time(); enc = offline_tiktoken_encoding("o200k_base"); dt = time.time() - t
print(f"   encoding={enc} in {dt:.2f}s"); raise SystemExit(0 if enc is not None and dt < 10 else 1)
PY
check "raw judge responses (diagnostic, 2 passages, non-streaming + streaming)" timeout 600 python - <<'PY'
import json, os, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from runner_common import ChatClient, THINK_END
from validate_mint import build_prompt, parse_judge_json
from openai import OpenAI
rows = [json.loads(l) for l in open("MINT_dataset_v1.2/cwq_passages.jsonl")][:2]
cl = ChatClient(os.environ["ENDPOINT_URL"], os.environ["MODEL_NAME"], mock=False)
raw = OpenAI(base_url=os.environ["ENDPOINT_URL"], api_key="-")
mt = int(os.environ.get("JUDGE_MAX_TOKENS", "2048"))
for r in rows:
    msgs = build_prompt(r, "misinformation")
    res = cl.chat(msgs, max_tokens=mt, temperature=0.0, label="diag")
    print(f"   [non-stream] finish={res.finish_reason} completion_tokens={res.completion_tokens} content_len={len(res.text)} "
          f"reasoning_len={len(res.reasoning_text or '')} think_end_in_reasoning={THINK_END in (res.reasoning_text or '')} "
          f"parsed={parse_judge_json(res.text, 'misinformation') is not None}")
    print("      content head:", repr(res.text[:200])); print("      content tail:", repr(res.text[-200:]))
    print("      reasoning tail:", repr((res.reasoning_text or '')[-200:]))
    c, rs = [], []
    for ch in raw.chat.completions.create(model=os.environ["MODEL_NAME"], messages=msgs, max_tokens=mt, temperature=0.0, stream=True):
        d = ch.choices[0].delta
        if getattr(d, "content", None): c.append(d.content)
        x = getattr(d, "reasoning_content", None) or getattr(d, "reasoning", None)
        if x: rs.append(x)
    c, rs = "".join(c), "".join(rs)
    print(f"   [stream]     content_len={len(c)} reasoning_len={len(rs)} think_end_in_content={THINK_END in c} think_end_in_reasoning={THINK_END in rs}")
    print("      content head:", repr(c[:200]))
PY
for ds in cwq ethics winogrande; do head -5 "MINT_dataset_v1.2/${ds}_passages.jsonl" > "${V}/MINT/${ds}_passages.jsonl"; done
cp MINT_dataset_v1.2/*_misinformed.json "${V}/MINT/"
t0=$(date +%s)
step "judge 5 passages x 3 datasets (live ${MODEL_NAME}, scratch copy)" timeout 900 python scripts/validate_mint.py \
    --judge_model "${MODEL_NAME}" --endpoint_url "${ENDPOINT_URL}" --data_dir "${V}/MINT" --max_tokens "${JUDGE_MAX_TOKENS}" \
    --max_passages 5 --results_dir "${V}/results"
echo "judge wall time: $(( $(date +%s) - t0 )) s"
check "judge wrote flags + n_tokens for every row" python - "${V}/MINT" <<'PY'
import json, sys, glob
bad = 0; n = 0; unparsed = 0
for f in glob.glob(sys.argv[1] + "/*_passages.jsonl"):
    for r in map(json.loads, open(f)):
        v = r.get("validation") or {}; judged = [k for k in v if k not in ("judges", "verified")]
        parsed = all(v[k].get("parsed") for k in judged)
        print(f"   {r['passage_id']}: judges={judged} parsed={parsed} n_tokens={r.get('n_tokens')} verified={v.get('verified')}")
        bad += (not judged) or (r.get("n_tokens") is None)
        n += 1; unparsed += (not parsed)
print(f"   unparsed {unparsed}/{n}"); raise SystemExit(1 if (bad or unparsed > n // 5) else 0)
PY
finish
