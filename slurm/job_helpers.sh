#!/usr/bin/env bash
# Helpers sourced by slurm/run_chain.slurm (bash 4+). Kept in a separate file so they can be unit-tested
# on a login node: bash slurm/test_job_helpers.sh
#
# pick_free_port [seed]        -> prints a free TCP port on 127.0.0.1, derived from the Slurm job id so
#                                 that two jobs sharing a node never both default to 8080.
# launch_process_group <log> <cmd...>
#                              -> starts <cmd> in its own session/process group (setsid), stdout+stderr
#                                 to <log>; sets LAUNCHED_PID (also the process-group id).
# stop_process_group <pgid> [grace_seconds]
#                              -> SIGTERM the whole group, wait up to grace_seconds, then SIGKILL; never
#                                 blocks forever (a container parent may ignore TERM).
# endpoint_serves_model <endpoint_url> <model_name>
#                              -> 0 when GET <endpoint>/models lists <model_name>. Guards against talking to
#                                 another job's server on the same host/port.
# prime_tiktoken_cache_env     -> exports TIKTOKEN_CACHE_DIR primed from the offline vocab files so host-side
#                                 python never tries to download o200k_base (compute nodes have no internet).

pick_free_port() {
    local seed="${1:-${SLURM_JOB_ID:-$$}}"
    if ! command -v python3 >/dev/null 2>&1; then   # fallback without python: probe with /dev/tcp
        local port n
        for ((n = 0; n < 2000; n++)); do
            port=$(( 20000 + (seed % 20000 + n) % 20000 ))
            if ! (exec 3<>"/dev/tcp/127.0.0.1/${port}") 2>/dev/null; then echo "${port}"; return 0; fi
        done
        return 1
    fi
    python3 - "${seed}" <<'PY'
import socket, sys
seed = int(sys.argv[1]) if sys.argv[1].isdigit() else abs(hash(sys.argv[1]))
base = 20000 + seed % 20000            # 20000..39999, disjoint from the runners' 8080 default
for offset in range(0, 2000):
    port = 20000 + (base - 20000 + offset) % 20000
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            continue
    print(port)
    sys.exit(0)
sys.exit("no free port found")
PY
}

launch_process_group() {
    local log="$1"; shift
    setsid "$@" > "${log}" 2>&1 &
    LAUNCHED_PID=$!
}

_pg_alive() {  # any process left in process group $1?
    ps -o pid= -g "$1" 2>/dev/null | grep -q .
}

stop_process_group() {
    local pgid="$1" grace="${2:-60}" i
    [ -n "${pgid}" ] || return 0
    _pg_alive "${pgid}" || return 0
    kill -TERM -- "-${pgid}" 2>/dev/null || true
    for ((i = 0; i < grace; i++)); do
        _pg_alive "${pgid}" || return 0
        sleep 1
    done
    echo "[WARN] process group ${pgid} ignored SIGTERM for ${grace}s; sending SIGKILL" >&2
    kill -KILL -- "-${pgid}" 2>/dev/null || true
    for ((i = 0; i < 10; i++)); do
        _pg_alive "${pgid}" || return 0
        sleep 1
    done
    echo "[WARN] process group ${pgid} still has processes after SIGKILL: $(ps -o pid=,cmd= -g "${pgid}" | head -3)" >&2
    return 1
}

endpoint_serves_model() {
    local url="$1" model="$2"
    python3 - "${url}" "${model}" <<'PY'
import sys, requests
url, model = sys.argv[1].rstrip("/"), sys.argv[2]
try:
    r = requests.get(f"{url}/models", timeout=5)
    ids = [m.get("id") for m in r.json().get("data", [])] if r.status_code == 200 else []
except Exception as e:  # noqa: BLE001
    print(f"models query failed: {e}")
    sys.exit(2)
print(f"models served at {url}: {ids}")
sys.exit(0 if model in ids else 1)
PY
}

prime_tiktoken_cache_env() {
    local d
    d="$(python3 - <<'PY'
import os, sys
sys.path.insert(0, os.environ.get("REPO_ROOT", "."))
from model_server_config import prime_tiktoken_cache
print(prime_tiktoken_cache())
PY
)" || d=""
    if [ -n "${d}" ]; then
        export TIKTOKEN_CACHE_DIR="${d}"
        echo "[INFO] TIKTOKEN_CACHE_DIR=${d} (offline o200k_base/cl100k_base for host-side python)"
    else
        echo "[WARN] could not prime the tiktoken cache; host-side tiktoken.get_encoding() would hang offline"
    fi
}
