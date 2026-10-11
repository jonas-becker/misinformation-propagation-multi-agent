#!/usr/bin/env bash
# Unit tests for slurm/job_helpers.sh (run on a login node): bash slurm/test_job_helpers.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export REPO_ROOT="$PWD"
source slurm/job_helpers.sh
fail=0; ok() { echo "  ok   $*"; }; bad() { echo "  FAIL $*"; fail=1; }

echo "[1] pick_free_port"
p1=$(pick_free_port 16223993); p2=$(pick_free_port 16223997); p3=$(pick_free_port 16223993)
[[ "$p1" =~ ^[0-9]+$ && "$p2" =~ ^[0-9]+$ ]] && ok "ports $p1 / $p2" || bad "non-numeric port: '$p1' '$p2'"
[ "$p1" != "$p2" ] && ok "different job ids -> different ports" || bad "same port for two job ids"
[ "$p1" = "$p3" ] && ok "deterministic for the same job id" || bad "not deterministic"
[ "$p1" != "8080" ] && [ "$p2" != "8080" ] && ok "not the fixed default 8080" || bad "8080"
# a port that is in use must be skipped
python3 -c "import socket,time; s=socket.socket(); s.bind(('127.0.0.1',$p1)); s.listen(1); time.sleep(4)" & hold=$!
sleep 1; p4=$(pick_free_port 16223993); [ "$p4" != "$p1" ] && ok "busy port $p1 skipped -> $p4" || bad "returned busy port"; wait $hold
# fallback path without python3
tb=$(mktemp -d); ln -s "$(command -v bash)" "$tb/bash"; p5=$(PATH="$tb" "$tb/bash" -c 'source slurm/job_helpers.sh; pick_free_port 12345'); rm -rf "$tb"; [[ "$p5" =~ ^[0-9]+$ ]] && ok "no-python fallback -> $p5" || bad "fallback: '$p5'"

echo "[2] launch_process_group + stop_process_group on a tree that ignores SIGTERM (as an Apptainer parent may)"
log=$(mktemp); launch_process_group "$log" bash -c 'trap "" TERM; (trap "" TERM; sleep 600) & (sleep 600) & sleep 600'
sleep 1; n_before=$(ps -o pid= -g "$LAUNCHED_PID" | wc -l)
[ "$n_before" -ge 3 ] && ok "group $LAUNCHED_PID has $n_before processes" || bad "group has only $n_before processes"
t0=$(date +%s); stop_process_group "$LAUNCHED_PID" 3; rc=$?; dt=$(( $(date +%s) - t0 ))
n_after=$(ps -o pid= -g "$LAUNCHED_PID" | wc -l)
[ "$n_after" -eq 0 ] && ok "no processes left (rc=$rc)" || bad "$n_after processes left"
[ "$dt" -le 15 ] && ok "returned in ${dt}s (grace 3s + SIGKILL)" || bad "took ${dt}s"
echo "[2b] a well-behaved group exits on SIGTERM without SIGKILL"
launch_process_group "$log" bash -c 'sleep 600 & wait'; sleep 1; t0=$(date +%s); stop_process_group "$LAUNCHED_PID" 30 2>"$log.err"; dt=$(( $(date +%s) - t0 ))
[ "$dt" -le 3 ] && ! grep -q SIGKILL "$log.err" && ok "stopped by SIGTERM in ${dt}s" || bad "TERM path: ${dt}s, $(cat "$log.err")"
echo "[2c] stop_process_group on an already-gone group returns immediately"
stop_process_group 4000000 5 && ok "no-op" || bad "returned error"

echo "[3] endpoint_serves_model"
port=$(pick_free_port 777)
python3 - "$port" <<'PY' & srv=$!
import sys, json
from http.server import BaseHTTPRequestHandler, HTTPServer
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"data": [{"id": "openai/gpt-oss-120b"}]}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)
    def log_message(self, *a): pass
HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
PY
sleep 1.5
endpoint_serves_model "http://127.0.0.1:$port/v1" "openai/gpt-oss-120b" >/dev/null && ok "served model accepted" || bad "served model rejected"
endpoint_serves_model "http://127.0.0.1:$port/v1" "zai-org/GLM-4.7-Flash" >/dev/null; [ $? -eq 1 ] && ok "other job's model rejected" || bad "foreign model accepted"
endpoint_serves_model "http://127.0.0.1:1/v1" "x" >/dev/null; [ $? -eq 2 ] && ok "unreachable endpoint -> 2" || bad "unreachable"
kill $srv 2>/dev/null; wait $srv 2>/dev/null

echo "[4] prime_tiktoken_cache_env"
export TIKTOKEN_CACHE_DIR="$(mktemp -d)"; prime_tiktoken_cache_env >/dev/null
[ -n "$(ls -A "$TIKTOKEN_CACHE_DIR")" ] && ok "cache primed: $(ls "$TIKTOKEN_CACHE_DIR" | wc -l) files in $TIKTOKEN_CACHE_DIR" || bad "cache empty"
rm -rf "$TIKTOKEN_CACHE_DIR" "$log" "$log.err"
[ $fail -eq 0 ] && echo "ALL HELPER TESTS PASSED" || { echo "HELPER TESTS FAILED"; exit 1; }
