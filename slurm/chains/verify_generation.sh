#!/usr/bin/env bash
# GPU check: passage generation with a thinking model must produce non-empty passages
# (generate_mint_passages.py uses 4096/2048-token budgets for GLM / gpt-oss and retries on truncation).
# Also re-checks the judge on the generated scratch set. Runs through slurm/run_chain.slurm via CHAIN_SCRIPT.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
V="out/_verify/gen_job_${SLURM_JOB_ID:-local}"; rm -rf "${V}"; mkdir -p "${V}"
check() { if "${@:2}"; then echo "[OK   $(date -Is)] $1"; else echo "[FAIL $(date -Is)] $1"; FAILED+=("$1"); fi; }

step "generate misinformation set (2 items x 2 datasets, scratch)" timeout 1500 python scripts/generate_mint_passages.py \
    --mode misinformation "${COMMON[@]}" --max_items 2 --datasets winogrande,cwq --out_dir "${V}/gen"
check "all generated passages non-empty (5 strategies x 2 items x 2 datasets)" python - "${V}/gen" <<'PY'
import json, sys, glob
n = empty = 0
for f in sorted(glob.glob(sys.argv[1] + "/*_misinformed.json")):
    for it in json.load(open(f)):
        n += 1; empty += (not (it.get("false_fact") or "").strip())
        print(f"   {it['MINT_ID']} false_fact={repr((it.get('false_fact') or '')[:90])}")
        for s, p in it["misinformation_by_strategy"].items():
            n += 1; empty += (not (p or "").strip())
            print(f"      {s:11s} len={len(p or '')} {repr((p or '')[:70])}")
meta = glob.glob(sys.argv[1] + "/generation_run_*.json")
if meta:
    m = json.load(open(meta[0])); print("   tokens:", m.get("tokens"), "empty_generations:", m.get("empty_generations"))
print(f"   empty {empty}/{n}"); raise SystemExit(1 if empty else 0)
PY
if [[ "${MODEL_NAME}" == openai/gpt-oss* ]]; then
    step "paraphrase control (2 items, scratch)" timeout 900 python scripts/generate_mint_passages.py \
        --mode paraphrase "${COMMON[@]}" --max_items 2 --datasets winogrande --out_dir "${V}/para"
    check "paraphrases non-empty" python - "${V}/para" <<'PY'
import json, sys, glob
n = empty = 0
for it in json.load(open(glob.glob(sys.argv[1] + "/winogrande_misinformed.json")[0])):
    for s, p in it["misinformation_by_strategy"].items():
        n += 1; empty += (not (p or "").strip()); print(f"      {s:11s} len={len(p or '')} {repr((p or '')[:70])}")
print(f"   empty {empty}/{n}"); raise SystemExit(1 if empty else 0)
PY
fi
step "judge the generated scratch set" timeout 900 python scripts/validate_mint.py --judge_model "${MODEL_NAME}" \
    --endpoint_url "${ENDPOINT_URL}" --data_dir "${V}/gen" --max_tokens "${JUDGE_MAX_TOKENS}" --results_dir "${V}/results"
check "judge parsed >= 80% of generated passages" python - "${V}/gen" <<'PY'
import json, sys, glob
n = unparsed = 0
for f in glob.glob(sys.argv[1] + "/*_passages.jsonl"):
    for r in map(json.loads, open(f)):
        v = r.get("validation") or {}; ks = [k for k in v if k not in ("judges", "verified")]
        n += 1; unparsed += (not ks) or (not all(v[k].get("parsed") for k in ks))
print(f"   unparsed {unparsed}/{n}"); raise SystemExit(1 if n == 0 or unparsed > n // 5 else 0)
PY
finish
