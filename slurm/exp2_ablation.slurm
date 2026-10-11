#!/bin/bash -l
#SBATCH --job-name=exp2_sc_abl
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --time=48:00:00
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=128G
# GLM-4.7-Flash BF16 weights need >40GB VRAM with vLLM 0.19; 40GB nodes OOM during load.
# Request 80GB GPUs for GLM-4.7-Flash; 40GB nodes OOM during model load.
#SBATCH --constraint=80gb_vram
#SBATCH --ntasks=1
#SBATCH --nodes=1
# exp2 ablation: Self-Consistency (Wang et al., 2022) instead of the MALLM multi-agent
# debate. Same model serving stack as exp2.slurm, but no MALLM dependency is needed.
# If jobs fail with "cannot create temp file for here-document: No space left on device",
# your /tmp (or Slurm spool) may be full. This script sets TMPDIR early (see init_ablation_tmpdir).
# Site-specific: request node-local scratch if your cluster supports it, e.g. #SBATCH --tmp=200G

set -euo pipefail

# Log effective RAM budget for this step (helps confirm #SBATCH --mem was applied).
log_slurm_memory_budget() {
    echo "============ Slurm / cgroup memory budget ============"
    echo "JobID=${SLURM_JOB_ID:-<unset>}  NodeList=${SLURM_JOB_NODELIST:-<unset>}"
    echo "SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-<unset>} (Slurm: MiB per node unless your site documents otherwise)"
    echo "SLURM_MEM_PER_CPU=${SLURM_MEM_PER_CPU:-<unset>}"
    echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-<unset>}  SLURM_CPUS_ON_NODE=${SLURM_CPUS_ON_NODE:-<unset>}"
    if [ -n "${SLURM_JOB_ID:-}" ] && command -v scontrol >/dev/null 2>&1; then
        echo "--- scontrol show job (memory / TRES) ---"
        scontrol show job "${SLURM_JOB_ID}" 2>/dev/null | grep -Ei 'JobId=|MinMem|Mem=|TRES=|AllocTRES=' || true
    fi
    # cgroup v2: memory.max for this process (max + swap cap; "max" means effectively unlimited cgroup-side)
    _cg2_rel=""
    if [ -r /proc/self/cgroup ]; then
        _cg2_rel="$(awk -F: '$1==0 {print $3; exit}' /proc/self/cgroup 2>/dev/null || true)"
    fi
    if [ -n "${_cg2_rel}" ] && [ -r "/sys/fs/cgroup${_cg2_rel}/memory.max" ]; then
        echo "cgroup v2 memory.max: $(tr -d '\n' <"/sys/fs/cgroup${_cg2_rel}/memory.max") bytes (max=max means no cgroup hard cap)"
    fi
    # cgroup v1 fallback
    _cg1_mem=""
    if [ -r /proc/self/cgroup ]; then
        _cg1_mem="$(awk -F: '$2=="memory" {print $3; exit}' /proc/self/cgroup 2>/dev/null || true)"
    fi
    if [ -n "${_cg1_mem}" ] && [ -r "/sys/fs/cgroup/memory${_cg1_mem}/memory.limit_in_bytes" ]; then
        echo "cgroup v1 memory.limit_in_bytes: $(cat "/sys/fs/cgroup/memory${_cg1_mem}/memory.limit_in_bytes" 2>/dev/null || true)"
    fi
    if command -v free >/dev/null 2>&1; then
        echo "--- free -h (whole node; job may be limited to cgroup above) ---"
        free -h || true
    fi
    echo "======================================================"
}
log_slurm_memory_budget

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}}"
if [ ! -f "${REPO_ROOT}/exp2_ablation.py" ]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

PROJECT_HOME="${PROJECT_HOME:-${HOME}}"
APPTAINER_BIN="${APPTAINER_BIN:-apptainer}"
VLLM_IMAGE="${VLLM_IMAGE:-/path/to/vllm-openai.sif}"
# After the endpoint smoke: optional parallel /chat/completions probe (default 64; set 250 to match full runs).
EXP2_ABLATION_CONCURRENT_PROBE="${EXP2_ABLATION_CONCURRENT_PROBE:-64}"
# exp2_ablation.py: cap misinformation conditions per dataset (unset = all). Use e.g. 1 for a fast full-pipeline check.
EXP2_ABLATION_MAX_CONDITIONS="${EXP2_ABLATION_MAX_CONDITIONS:-${EXP2_MAX_CONDITIONS:-}}"
_HF_CACHE_HOME_DEFAULT="${PROJECT_HOME}/.cache/huggingface"
HF_CACHE_VOLUME="${HF_CACHE_VOLUME:-${_HF_CACHE_HOME_DEFAULT}}"
LLAMA_70B_HF_CACHE="${LLAMA_70B_HF_CACHE:-${HF_HOME:-${HOME}/.cache/huggingface}}"
SHARED_HF_CACHE_VOLUME="${SHARED_HF_CACHE_VOLUME:-/shared/hf_cache}"
VLLM_CACHE_VOLUME="${VLLM_CACHE_VOLUME:-${PROJECT_HOME}/.cache/vllm}"
# Same venv as exp2 (its dependencies are a superset of what the ablation needs).
VENV_PATH="${VENV_PATH:-${PROJECT_HOME}/.venvs/misinformed_agents_exp2}"
VLLM_SERVER_BACKEND="${VLLM_SERVER_BACKEND:-}"
VLLM_SERVER_VENV_PATH="${VLLM_SERVER_VENV_PATH:-${PROJECT_HOME}/.venvs/vllm}"
GLM_PYTHON_OVERRIDES_DIR="${GLM_PYTHON_OVERRIDES_DIR:-${PROJECT_HOME}/glm47_pyoverrides}"
APPTAINER_CACHEDIR_PATH="${APPTAINER_CACHEDIR_PATH:-${PROJECT_HOME}/.cache/apptainer}"
APPTAINER_TMPDIR_PATH="${APPTAINER_TMPDIR_PATH:-${PROJECT_HOME}/.cache/apptainer-tmp}"
JOB_TMP_BASE="${JOB_TMP_BASE:-${PROJECT_HOME}/.cache/job-tmp}"

# shellcheck source=slurm/model_slurm_config.sh
source "${REPO_ROOT}/slurm/model_slurm_config.sh"

# Bash here-documents (<<'PY'), pip, and many tools write under TMPDIR (default /tmp).
# Compute nodes often have a small or full /tmp — use Slurm job scratch or JOB_TMP_BASE first.
init_ablation_tmpdir() {
    local _cand=""
    for _cand in "${SLURM_TMPDIR:-}" "${JOB_TMP_BASE}/slurm_tmp_${SLURM_JOB_ID:-$$}"; do
        if [ -n "${_cand}" ] && mkdir -p "${_cand}" 2>/dev/null; then
            if touch "${_cand}/.exp2_ablation_tmp_probe_$$" 2>/dev/null; then
                rm -f "${_cand}/.exp2_ablation_tmp_probe_$$" 2>/dev/null || true
                export TMPDIR="${_cand}"
                export TMP="${TMPDIR}"
                export TEMP="${TMPDIR}"
                echo "[INFO] TMPDIR=${TMPDIR} (bash here-docs, pip, etc.)"
                df -h "${TMPDIR}" 2>/dev/null || df -h . 2>/dev/null || true
                return 0
            fi
        fi
    done
    echo "[WARN] Could not set a writable TMPDIR under SLURM_TMPDIR or JOB_TMP_BASE; using system default (may be /tmp)."
}
init_ablation_tmpdir

MODEL_NAME="${MODEL_NAME:-zai-org/GLM-4.7-Flash}"
MODEL_NAME="$(normalize_model_name "${MODEL_NAME}")"
echo "Normalized MODEL_NAME=${MODEL_NAME}"
AUTO_FALLBACK_ON_40GB="${AUTO_FALLBACK_ON_40GB:-1}"
MODEL_FALLBACK_FOR_40GB="${MODEL_FALLBACK_FOR_40GB:-zai-org/GLM-4.5-Air-FP8}"
GPU_TOTAL_MEM_MIB_OVERRIDE="${GPU_TOTAL_MEM_MIB_OVERRIDE:-}"
DATASET="${DATASET:-cwq,winogrande,ethics}"

# Self-consistency parameters (see exp2_ablation.py; arXiv:2203.11171).
# Default num_samples=15 mirrors exp2's MALLM budget (num_agents=3 x max_turns=5).
EXP2_ABLATION_NUM_SAMPLES="${EXP2_ABLATION_NUM_SAMPLES:-15}"
EXP2_ABLATION_TEMPERATURE="${EXP2_ABLATION_TEMPERATURE:-0.8}"
EXP2_ABLATION_TOP_P="${EXP2_ABLATION_TOP_P:-0.95}"
EXP2_ABLATION_SEED="${EXP2_ABLATION_SEED:-42}"
# Per-completion cap; matches exp2's MALLM max_tokens default (1024).
EXP2_ABLATION_MAX_TOKENS="${EXP2_ABLATION_MAX_TOKENS:-1024}"

# Per-run output directory. Set EXP2_ABLATION_RUN_ID=rep1|rep2|rep3 for separate replicates
# (same model, compare variance across runs). Defaults to job_<SLURM_JOB_ID>.
safe_model_name_for_path() {
    local raw="${1:-}"
    local safe="${raw//\//_}"
    safe="$(printf '%s' "${safe}" | sed -E 's/[^A-Za-z0-9._-]+/_/g')"
    printf '%s\n' "${safe:-unknown_model}"
}

init_ablation_run_dir() {
    local model_safe
    model_safe="$(safe_model_name_for_path "${MODEL_NAME}")"
    EXP2_ABLATION_RUN_ID="${EXP2_ABLATION_RUN_ID:-job_${SLURM_JOB_ID:-$$}}"
    if [ -n "${EXP2_ABLATION_OUTPUT_DIR:-}" ]; then
        EXP2_ABLATION_RUN_DIR="${EXP2_ABLATION_OUTPUT_DIR}"
    elif [ -z "${EXP2_ABLATION_RUN_DIR:-}" ]; then
        EXP2_ABLATION_RUN_DIR="${REPO_ROOT}/out/${model_safe}/runs/${EXP2_ABLATION_RUN_ID}"
    fi
    mkdir -p "${EXP2_ABLATION_RUN_DIR}"
    export EXP2_ABLATION_OUTPUT_DIR="${EXP2_ABLATION_RUN_DIR}"
    echo "[INFO] EXP2 ablation run directory: ${EXP2_ABLATION_RUN_DIR}"
    {
        echo "EXPERIMENT=exp2_ablation_self_consistency"
        echo "EXP2_ABLATION_RUN_ID=${EXP2_ABLATION_RUN_ID}"
        echo "MODEL_NAME=${MODEL_NAME}"
        echo "DATASET=${DATASET}"
        echo "NUM_SAMPLES=${EXP2_ABLATION_NUM_SAMPLES}"
        echo "TEMPERATURE=${EXP2_ABLATION_TEMPERATURE}"
        echo "TOP_P=${EXP2_ABLATION_TOP_P}"
        echo "SEED=${EXP2_ABLATION_SEED}"
        echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
        echo "started_at=$(date -Is 2>/dev/null || date)"
    } > "${EXP2_ABLATION_RUN_DIR}/run_info.env"
    if [ "${EXP2_ABLATION_TEE_SLURM_LOG:-1}" = "1" ]; then
        _ablation_slurm_log="${EXP2_ABLATION_RUN_DIR}/slurm.log"
        _ablation_tee_bin=""
        if [ -x /usr/bin/tee ]; then
            _ablation_tee_bin="/usr/bin/tee"
        elif command -v tee >/dev/null 2>&1; then
            _ablation_tee_bin="$(command -v tee)"
        fi
        if [ -n "${_ablation_tee_bin}" ]; then
            # Use an absolute tee path: Slurm process-substitution subshells often have an empty PATH.
            exec > >("${_ablation_tee_bin}" -a "${_ablation_slurm_log}") 2>&1
            echo "[INFO] Mirroring Slurm output to ${_ablation_slurm_log}"
        else
            echo "[WARN] tee not found; skipping slurm.log mirroring (see slurm-${SLURM_JOB_ID:-}.out)"
        fi
    fi
}
init_ablation_run_dir

DEBUG="${DEBUG:-0}"
VERIFY_ONLY="${VERIFY_ONLY:-0}"
PORT="${PORT:-8080}"
ENDPOINT_URL="${ENDPOINT_URL:-http://127.0.0.1:${PORT}/v1}"
HF_TOKEN="${HF_TOKEN:-}"
VLLM_LOG="${VLLM_LOG:-${EXP2_ABLATION_RUN_DIR}/apptainer_vllm.log}"
# Default is chosen after MODEL_NAME is finalized (see below). Override with VLLM_MAX_MODEL_LEN=...
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-}"
VLLM_TRUST_REMOTE_CODE="${VLLM_TRUST_REMOTE_CODE:-}"
# GLM + vLLM (see https://docs.vllm.ai/projects/recipes/en/latest/GLM/GLM.html ). vLLM 0.19+ uses --no-enable-log-requests (not --disable-log-requests).
VLLM_TOOL_CALL_PARSER="${VLLM_TOOL_CALL_PARSER:-}"
VLLM_REASONING_PARSER="${VLLM_REASONING_PARSER:-}"
VLLM_ENABLE_AUTO_TOOL_CHOICE="${VLLM_ENABLE_AUTO_TOOL_CHOICE:-}"
VLLM_NO_ENABLE_LOG_REQUESTS="${VLLM_NO_ENABLE_LOG_REQUESTS:-}"
VLLM_NO_ENABLE_PREFIX_CACHING="${VLLM_NO_ENABLE_PREFIX_CACHING:-}"
# vLLM 0.19 images ship transformers 4.57 without glm4_moe_lite; bind a pip --target tree here (see exp2.slurm comments).
VLLM_GLM_TRANSFORMERS_SITE="${VLLM_GLM_TRANSFORMERS_SITE:-${HOME}/.cache/glm_vllm_transformers}"
PYTHON_BIN="${PYTHON_BIN:-}"
SETUP_ONLY="${SETUP_ONLY:-0}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"
ALLOW_RUNTIME_PIP="${ALLOW_RUNTIME_PIP:-0}"
USE_MOCK="${USE_MOCK:-0}"
ALLOW_DOCKER_PULL="${ALLOW_DOCKER_PULL:-0}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-}"
MALLM_OPENAI_TIMEOUT_SECONDS="${MALLM_OPENAI_TIMEOUT_SECONDS:-3600}"
MALLM_OPENAI_MAX_RETRIES="${MALLM_OPENAI_MAX_RETRIES:-5}"
GLM_SHIM_QUEUE_KEEPALIVE_SECONDS="${GLM_SHIM_QUEUE_KEEPALIVE_SECONDS:-5}"
GLM_SHIM_MAX_QUEUE_WAIT_SECONDS="${GLM_SHIM_MAX_QUEUE_WAIT_SECONDS:-600}"
GLM_SHIM_STREAMER_TIMEOUT_SECONDS="${GLM_SHIM_STREAMER_TIMEOUT_SECONDS:-120}"
MALLM_CONCURRENT_API_REQUESTS="${MALLM_CONCURRENT_API_REQUESTS:-250}"

if [ ! -d "${REPO_ROOT}" ]; then
    echo "Experiment repo does not exist: ${REPO_ROOT}"
    exit 2
fi

apply_model_hf_cache_path "${MODEL_NAME}"

mkdir -p "${REPO_ROOT}/out"
mkdir -p "${HF_CACHE_VOLUME}" "${HF_CACHE_VOLUME}/hub" "${HF_CACHE_VOLUME}/transformers"
mkdir -p "${JOB_TMP_BASE}"

if [ -n "${SHARED_MODEL_CACHE_SOURCE:-}" ]; then
    if [ ! -d "${SHARED_MODEL_CACHE_SOURCE}" ]; then
        echo "Shared model cache source not found: ${SHARED_MODEL_CACHE_SOURCE}"
        exit 2
    fi
    if [ ! -e "${LOCAL_MODEL_CACHE_TARGET}" ]; then
        ln -s "${SHARED_MODEL_CACHE_SOURCE}" "${LOCAL_MODEL_CACHE_TARGET}"
    fi
fi
mkdir -p "${VLLM_CACHE_VOLUME}"
mkdir -p "${APPTAINER_CACHEDIR_PATH}" "${APPTAINER_TMPDIR_PATH}"

# Fallback if init_ablation_tmpdir could not pick SLURM_TMPDIR / JOB_TMP_BASE subdir.
export TMPDIR="${TMPDIR:-${JOB_TMP_BASE}}"
export TMP="${TMP:-${TMPDIR}}"
export TEMP="${TEMP:-${TMPDIR}}"

cd "${REPO_ROOT}"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
else
    echo "nvidia-smi not found. This job is expected to run on a GPU partition."
    exit 2
fi

# GLM-4.7-Flash (BF16) can OOM on 40GB A100 nodes with vLLM 0.19. If we detect
# a 40GB card, auto-fallback to an FP8 GLM variant that fits better.
GPU_TOTAL_MEM_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | awk 'NR==1 {print int($1)}')"
if [ -n "${GPU_TOTAL_MEM_MIB_OVERRIDE}" ]; then
    GPU_TOTAL_MEM_MIB="${GPU_TOTAL_MEM_MIB_OVERRIDE}"
fi
if [ -z "${GPU_TOTAL_MEM_MIB}" ]; then
    GPU_TOTAL_MEM_MIB=0
fi
if [ "${AUTO_FALLBACK_ON_40GB}" = "1" ] && [ "${MODEL_NAME}" = "zai-org/GLM-4.7-Flash" ] && [ "${GPU_TOTAL_MEM_MIB}" -gt 0 ] && [ "${GPU_TOTAL_MEM_MIB}" -lt 70000 ]; then
    echo "Detected ${GPU_TOTAL_MEM_MIB} MiB GPU memory (< 70GB)."
    echo "Switching model from ${MODEL_NAME} to ${MODEL_FALLBACK_FOR_40GB} to avoid startup OOM on 40GB GPUs."
    MODEL_NAME="${MODEL_FALLBACK_FOR_40GB}"
fi
# Self-consistency prompts are short, but keep the exp2 defaults so serving behaves identically.
apply_model_vllm_max_model_len_default "${MODEL_NAME}"
echo "Using VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN} for model ${MODEL_NAME}"
ensure_hub_model_cached_or_exit "${MODEL_NAME}"
if command -v module >/dev/null 2>&1; then
    module load Apptainer >/dev/null 2>&1 || true
    module load apptainer >/dev/null 2>&1 || true
    module load Singularity >/dev/null 2>&1 || true
fi

# ---- Python interpreter & environment selection ------------------------------
# GPU nodes may be heterogeneous: some ship /usr/bin/python3.11 (the shared exp2
# venv works there), others only have python3 = 3.6. Strategy:
#   1. Use VENV_PATH when its interpreter actually runs on this node and the
#      required packages import.
#   2. Otherwise load a site Python module (e.g. python/3.11.9) and
#      use/create a dedicated ablation venv. Compute nodes have no internet, so
#      the venv installs offline from EXP2_ABLATION_WHEEL_DIR; populate it once
#      on a login node:
#        module load python/3.11.9
#        python3.11 -m venv /tmp/wheeltmp
#        /tmp/wheeltmp/bin/pip download --dest ~/.cache/exp2_ablation_wheels requests openai
# The shared exp2 venv is never deleted here: on nodes lacking its base
# interpreter it is merely skipped (it still works on the other nodes).
EXP2_ABLATION_VENV_PATH="${EXP2_ABLATION_VENV_PATH:-${PROJECT_HOME}/.venvs/misinformed_agents_exp2_ablation}"
EXP2_ABLATION_WHEEL_DIR="${EXP2_ABLATION_WHEEL_DIR:-${PROJECT_HOME}/.cache/exp2_ablation_wheels}"

_python_is_ge_39() {
    # True when $1 executes on this node and reports Python >= 3.9.
    "${1}" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 9) else 1)
PY
}

_venv_is_usable() {
    # True when the venv's interpreter runs here, is >= 3.9, and deps import.
    local venv_dir="${1}"
    [ -f "${venv_dir}/bin/activate" ] || return 1
    _python_is_ge_39 "${venv_dir}/bin/python" || return 1
    "${venv_dir}/bin/python" -c "import requests, openai" >/dev/null 2>&1
}

load_site_python_modules() {
    command -v module >/dev/null 2>&1 || return 0
    for m in python/3.11.9 python miniforge3 env/staging/2024.1 Python/3.11.10-GCCcore-13.3.0; do
        module load "${m}" >/dev/null 2>&1 || true
        if command -v python3.11 >/dev/null 2>&1 || command -v python3.10 >/dev/null 2>&1; then
            echo "[INFO] Loaded site module providing a modern Python: ${m}"
            return 0
        fi
    done
}

find_python_ge_39() {
    if [ -n "${PYTHON_BIN}" ]; then
        if _python_is_ge_39 "${PYTHON_BIN}"; then
            return 0
        fi
        echo "[WARN] PYTHON_BIN=${PYTHON_BIN} is not a working Python >= 3.9 on this node; auto-detecting."
        PYTHON_BIN=""
    fi
    local candidate=""
    for candidate in python3.13 python3.12 python3.11 python3.10 python3.9 python3; do
        if command -v "${candidate}" >/dev/null 2>&1 && _python_is_ge_39 "$(command -v "${candidate}")"; then
            PYTHON_BIN="$(command -v "${candidate}")"
            return 0
        fi
    done
    return 1
}

SELECTED_VENV=""
if _venv_is_usable "${VENV_PATH}"; then
    SELECTED_VENV="${VENV_PATH}"
    echo "[INFO] Using shared experiment venv: ${SELECTED_VENV}"
else
    echo "[INFO] Venv ${VENV_PATH} is not usable on this node (node=$(hostname); its base interpreter may not exist here)."
    if _venv_is_usable "${EXP2_ABLATION_VENV_PATH}"; then
        SELECTED_VENV="${EXP2_ABLATION_VENV_PATH}"
        echo "[INFO] Using ablation venv: ${SELECTED_VENV}"
    else
        if ! find_python_ge_39; then
            load_site_python_modules
        fi
        if ! find_python_ge_39; then
            echo "No working Python >= 3.9 found on this node (python3 is $(python3 --version 2>&1))."
            echo "A site module (e.g. 'module load python/3.11.9') should provide one; otherwise set PYTHON_BIN explicitly."
            exit 2
        fi
        echo "[INFO] Creating ablation venv at ${EXP2_ABLATION_VENV_PATH} with ${PYTHON_BIN} ($("${PYTHON_BIN}" --version 2>&1))"
        if [ "${EXP2_ABLATION_VENV_PATH}" = "${VENV_PATH}" ]; then
            echo "EXP2_ABLATION_VENV_PATH must differ from VENV_PATH (${VENV_PATH}); refusing to recreate the shared venv."
            exit 2
        fi
        rm -rf "${EXP2_ABLATION_VENV_PATH}"
        "${PYTHON_BIN}" -m venv "${EXP2_ABLATION_VENV_PATH}"
        if compgen -G "${EXP2_ABLATION_WHEEL_DIR}/*.whl" >/dev/null 2>&1; then
            echo "[INFO] Installing requests+openai offline from ${EXP2_ABLATION_WHEEL_DIR}"
            "${EXP2_ABLATION_VENV_PATH}/bin/python" -m pip install --quiet --no-index \
                --find-links "${EXP2_ABLATION_WHEEL_DIR}" requests openai
        elif [ "${ALLOW_RUNTIME_PIP}" = "1" ]; then
            echo "[INFO] Wheel cache empty; trying online pip (ALLOW_RUNTIME_PIP=1; needs internet, e.g. -C inet)"
            "${EXP2_ABLATION_VENV_PATH}/bin/python" -m pip install --quiet requests openai
        else
            echo "Cannot install dependencies: wheel cache ${EXP2_ABLATION_WHEEL_DIR} is empty and compute nodes have no internet."
            echo "Populate it once from a login node (see comment above EXP2_ABLATION_VENV_PATH in this script)."
            exit 2
        fi
        if ! _venv_is_usable "${EXP2_ABLATION_VENV_PATH}"; then
            echo "Ablation venv at ${EXP2_ABLATION_VENV_PATH} is still not usable after setup."
            exit 2
        fi
        SELECTED_VENV="${EXP2_ABLATION_VENV_PATH}"
    fi
fi

echo "Activating environment at ${SELECTED_VENV}"
source "${SELECTED_VENV}/bin/activate"
python --version

if [ "${VERIFY_ONLY}" = "1" ]; then
    echo "VERIFY_ONLY=1 -> exp2 ablation Slurm setup looks good."
    echo "Repo root: ${REPO_ROOT}"
    echo "Slurm account: ${SLURM_JOB_ACCOUNT:-unknown}"
    echo "Slurm partition: ${SLURM_JOB_PARTITION:-unknown}"
    python --version
    scontrol show job "${SLURM_JOB_ID:-}" || true
    exit 0
fi

export HF_HOME="${HF_CACHE_VOLUME}"
export HF_HUB_CACHE="${HF_CACHE_VOLUME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_CACHE_VOLUME}/hub"
export TRANSFORMERS_CACHE="${HF_CACHE_VOLUME}/transformers"

if is_hub_model_id "${MODEL_NAME}"; then
    if is_model_cached_in "${MODEL_NAME}" "${HF_CACHE_VOLUME}"; then
        if [ -z "${HF_HUB_OFFLINE}" ]; then
            HF_HUB_OFFLINE="1"
        fi
        if [ -z "${TRANSFORMERS_OFFLINE}" ]; then
            TRANSFORMERS_OFFLINE="1"
        fi
    fi
fi

if [ "${HF_HUB_OFFLINE}" = "1" ] && is_hub_model_id "${MODEL_NAME}"; then
    if ! is_model_cached_in "${MODEL_NAME}" "${HF_CACHE_VOLUME}"; then
        echo "HF_HUB_OFFLINE=1 but model is not cached: ${MODEL_NAME}"
        echo "Cache checked: ${HF_CACHE_VOLUME}/hub/models--${MODEL_NAME//\//--}/snapshots"
        exit 2
    fi
fi

if [ -n "${HF_HUB_OFFLINE}" ]; then
    export HF_HUB_OFFLINE
fi
if [ -n "${TRANSFORMERS_OFFLINE}" ]; then
    export TRANSFORMERS_OFFLINE
fi

# The ablation only needs an OpenAI-compatible client stack (no MALLM).
# Importing exp2_ablation also pulls in exp2, model_server_config, and shared_utils.
if python - <<'PY'
import importlib
import sys
sys.path.insert(0, ".")
modules = ["requests", "openai", "exp2_ablation"]
for name in modules:
    importlib.import_module(name)
print("Runtime dependencies for exp2_ablation import check passed")
PY
then
    echo "Using existing dependencies from ${SELECTED_VENV}"
else
    if [ "${ALLOW_RUNTIME_PIP}" != "1" ]; then
        echo "Dependencies missing and runtime pip install is disabled (ALLOW_RUNTIME_PIP=0)."
        echo "This cluster's compute nodes may not reach PyPI; prebuild venv on login node or rerun with ALLOW_RUNTIME_PIP=1."
        exit 2
    fi
    echo "Installing runtime dependencies inside job (ALLOW_RUNTIME_PIP=1)"
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install requests openai
    python - <<'PY'
import importlib
import sys
sys.path.insert(0, ".")
modules = ["requests", "openai", "exp2_ablation"]
for name in modules:
    importlib.import_module(name)
print("Runtime dependencies for exp2_ablation import check passed")
PY
fi

if [ "${SETUP_ONLY}" = "1" ]; then
    echo "SETUP_ONLY=1 -> dependency setup successful. Exiting before vLLM launch."
    exit 0
fi

run_mock_mode() {
    DEBUG_FLAG=""
    if [ "${DEBUG}" = "1" ]; then
        DEBUG_FLAG="--debug"
    fi
    CMD="python exp2_ablation.py --mock --dataset ${DATASET} --output_dir ${EXP2_ABLATION_RUN_DIR} ${DEBUG_FLAG}"
    echo "Running exp2_ablation in mock mode: ${CMD}"
    eval "${CMD}"
}

if [ "${USE_MOCK}" = "1" ]; then
    run_mock_mode
    exit 0
fi

export OPENAI_BASE_URL="${ENDPOINT_URL}"
export MALLM_OPENAI_TIMEOUT_SECONDS
export MALLM_OPENAI_MAX_RETRIES
export GLM_SHIM_QUEUE_KEEPALIVE_SECONDS
export GLM_SHIM_MAX_QUEUE_WAIT_SECONDS
export GLM_SHIM_STREAMER_TIMEOUT_SECONDS
export MALLM_CONCURRENT_API_REQUESTS
export EXP2_ABLATION_NUM_SAMPLES
export EXP2_ABLATION_TEMPERATURE
export EXP2_ABLATION_TOP_P
export EXP2_ABLATION_SEED
export EXP2_ABLATION_MAX_TOKENS
export EXP2_ABLATION_MAX_CONDITIONS

export APPTAINERENV_HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
export APPTAINERENV_HF_TOKEN="${HF_TOKEN}"
export APPTAINERENV_HF_HOME="/root/.cache/huggingface"
export APPTAINERENV_HF_HUB_CACHE="/root/.cache/huggingface/hub"
export APPTAINERENV_HUGGINGFACE_HUB_CACHE="/root/.cache/huggingface/hub"
export APPTAINERENV_TRANSFORMERS_CACHE="/root/.cache/huggingface/transformers"
export APPTAINERENV_HF_HUB_ENABLE_HF_TRANSFER="0"
export APPTAINERENV_HF_HUB_OFFLINE="${HF_HUB_OFFLINE}"
export APPTAINERENV_TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE}"
normalize_cuda_visible_devices() {
    # Slurm may set GPU variables to values like:
    #   SLURM_JOB_GPUS="0" or "0,1" or "gpu:0" or "gpu:0,gpu:1"
    # vLLM/pynvml expects CUDA_VISIBLE_DEVICES to be a comma-separated list of integers.
    local raw="${1:-}"
    raw="${raw//gpu:/}"
    raw="${raw//GPU:/}"
    raw="${raw// /}"
    raw="${raw//;/,}"
    if [[ -z "${raw}" ]]; then
        return 1
    fi
    if [[ "${raw}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        echo "${raw}"
        return 0
    fi
    return 1
}

CUDA_DEVICES_RAW="${CUDA_VISIBLE_DEVICES:-}"
if [ -z "${CUDA_DEVICES_RAW}" ]; then
    CUDA_DEVICES_RAW="${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-}}"
fi

if CUDA_DEVICES_NORM="$(normalize_cuda_visible_devices "${CUDA_DEVICES_RAW}")"; then
    :
else
    CUDA_DEVICES_NORM="0"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES_NORM}"
export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_DEVICES_NORM}"

# Avoid setting HOME explicitly; apptainer may refuse overriding it (harmless but noisy).
export APPTAINERENV_XDG_CACHE_HOME="/root/.cache"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR_PATH}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR_PATH}"

ALLOCATED_GPUS="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS:-1}}"
if [[ "${ALLOCATED_GPUS}" == *","* ]]; then
    ALLOCATED_GPUS="$(echo "${ALLOCATED_GPUS}" | awk -F',' '{print NF}')"
fi
if ! [[ "${ALLOCATED_GPUS}" =~ ^[0-9]+$ ]] || [ "${ALLOCATED_GPUS}" -lt 1 ]; then
    ALLOCATED_GPUS="1"
fi

ensure_model_gpu_allocation "${MODEL_NAME}"
apply_model_vllm_trust_remote_code_default "${MODEL_NAME}"
apply_model_vllm_gpu_mem_default "${MODEL_NAME}"
apply_model_vllm_glm_gptoss_defaults "${MODEL_NAME}"
apply_gpt_oss_tiktoken_env "${MODEL_NAME}"

# GLM transformers staging warning (gpt-oss does not need glm4_moe_lite).
if [[ "${MODEL_NAME}" == zai-org/GLM-* ]]; then
  if [ "${VLLM_SERVER_BACKEND:-apptainer}" = "apptainer" ] && [ ! -f "${VLLM_GLM_TRANSFORMERS_SITE}/transformers/__init__.py" ]; then
    echo "WARN: GLM + vLLM Apptainer: install a transformers 5.0 staging tree for glm4_moe_lite (bundled 4.57 is too old), e.g. on a login node with Apptainer:"
    echo "  T=\"\${HOME}/.cache/glm_vllm_transformers\"; rm -rf \"\${T:?}\"/*; mkdir -p \"\${T}\""
    echo "  apptainer exec --bind \"\${T}:/out\" \"\${VLLM_IMAGE}\" pip install --target /out \"transformers==5.0.0\" \"huggingface_hub==1.3.0\" tokenizers safetensors regex requests pyyaml filelock tqdm packaging fsspec"
    echo "  rm -rf \"\${T}/numpy\" \"\${T}/numpy-\"*.dist-info \"\${T}/numpy.libs\" 2>/dev/null || true"
    echo "Then set VLLM_GLM_TRANSFORMERS_SITE=\${T} (default) or leave unset. See https://docs.vllm.ai/projects/recipes/en/latest/GLM/GLM.html"
  fi
fi

if [ -z "${VLLM_SERVER_BACKEND}" ]; then
  case "${MODEL_NAME}" in
    zai-org/GLM-*)
      # vLLM supports GLM-4.X (see vLLM GLM recipes). Served with vLLM like the other models.
      VLLM_SERVER_BACKEND="apptainer"
      ;;
    *)
      VLLM_SERVER_BACKEND="apptainer"
      ;;
  esac
fi

if [ "${VLLM_SERVER_BACKEND}" = "venv" ]; then
  if [ ! -x "${VLLM_SERVER_VENV_PATH}/bin/python" ]; then
    echo "GLM vLLM environment is missing: ${VLLM_SERVER_VENV_PATH}"
    echo "Either create it or override VLLM_SERVER_BACKEND=apptainer and VLLM_IMAGE."
    exit 2
  fi
  if [[ "${MODEL_NAME}" == zai-org/GLM-* ]]; then
    echo "Validating GLM serving environment at ${VLLM_SERVER_VENV_PATH}"
    if ! "${VLLM_SERVER_VENV_PATH}/bin/python" - <<PY
from importlib.metadata import version
import huggingface_hub
import transformers
transformers_version = version("transformers")
huggingface_hub_version = version("huggingface_hub")
print("transformers", transformers_version)
print("huggingface_hub", huggingface_hub_version)
major = int(transformers_version.split(".", 1)[0])
hub_major = int(huggingface_hub_version.split(".", 1)[0])
if (major < 5 and ".dev" not in transformers_version) or hub_major < 1:
    raise SystemExit(1)
PY
    then
      echo "Updating GLM serving dependencies in ${VLLM_SERVER_VENV_PATH} for ${MODEL_NAME}"
      "${VLLM_SERVER_VENV_PATH}/bin/pip" install --upgrade huggingface_hub
      "${VLLM_SERVER_VENV_PATH}/bin/pip" install --upgrade --no-deps \
        "git+https://github.com/huggingface/transformers.git"
      "${VLLM_SERVER_VENV_PATH}/bin/python" - <<PY
from importlib.metadata import version
print("transformers", version("transformers"))
print("huggingface_hub", version("huggingface_hub"))
PY
    fi
  fi
  VLLM_CMD=(
    "${VLLM_SERVER_VENV_PATH}/bin/python" -m vllm.entrypoints.openai.api_server
    --model "${MODEL_NAME}"
    --port "${PORT}"
    --tensor-parallel-size "${ALLOCATED_GPUS}"
    --max-model-len "${VLLM_MAX_MODEL_LEN}"
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
  )
else
  if [ -z "${VLLM_IMAGE}" ]; then
    echo "VLLM_IMAGE is not set."
    echo "Set VLLM_IMAGE to a local readable .sif image path, e.g.:"
    echo "  sbatch --export=ALL,VLLM_IMAGE=/path/to/vllm-openai.sif exp2_ablation.slurm"
    exit 2
  fi
  if [ ! -e "${VLLM_IMAGE}" ]; then
    echo "VLLM_IMAGE path does not exist: ${VLLM_IMAGE}"
    exit 2
  fi
  if [[ "${VLLM_IMAGE}" == docker://* ]] && [ "${ALLOW_DOCKER_PULL}" != "1" ]; then
    echo "VLLM_IMAGE uses docker:// but compute nodes cannot pull from registries here."
    echo "Provide a local .sif image path via VLLM_IMAGE or set ALLOW_DOCKER_PULL=1 if network access is available."
    exit 2
  fi
  if ! command -v "${APPTAINER_BIN}" >/dev/null 2>&1; then
    if command -v apptainer >/dev/null 2>&1; then
      APPTAINER_BIN="apptainer"
    elif command -v singularity >/dev/null 2>&1; then
      APPTAINER_BIN="singularity"
    else
      echo "Apptainer/Singularity not found. Set APPTAINER_BIN or load the container runtime module."
      exit 2
    fi
  fi
  APPTAINER_BINDS=(
    "${HF_CACHE_VOLUME}:/root/.cache/huggingface"
    "${VLLM_CACHE_VOLUME}:/root/.cache/vllm"
  )
  # Symlinks under HF_HOME often point to shared Ceph; bind target so Apptainer can resolve them.
  if [ -d "${SHARED_HF_CACHE_VOLUME}" ]; then
    APPTAINER_BINDS+=("${SHARED_HF_CACHE_VOLUME}:${SHARED_HF_CACHE_VOLUME}")
  fi
  if [ -n "${GPT_OSS_TIKTOKEN_BIND:-}" ]; then
    APPTAINER_BINDS+=("${GPT_OSS_TIKTOKEN_BIND}")
  fi
  if [[ "${MODEL_NAME}" == zai-org/GLM-* ]] && [ -f "${VLLM_GLM_TRANSFORMERS_SITE}/transformers/__init__.py" ]; then
    APPTAINER_BINDS+=("${VLLM_GLM_TRANSFORMERS_SITE}:/glm_tf_override")
    export APPTAINERENV_PYTHONPATH="/glm_tf_override${APPTAINERENV_PYTHONPATH:+:${APPTAINERENV_PYTHONPATH}}"
    echo "Using GLM transformers staging at ${VLLM_GLM_TRANSFORMERS_SITE} (PYTHONPATH /glm_tf_override)."
  fi
  if [[ "${MODEL_NAME}" == zai-org/GLM-* ]]; then
    if [ -d "${GLM_PYTHON_OVERRIDES_DIR}" ]; then
      export APPTAINERENV_PYTHONPATH="/glm47_pyoverrides"
      APPTAINER_BINDS+=("${GLM_PYTHON_OVERRIDES_DIR}:/glm47_pyoverrides")
    else
      echo "GLM overrides not found at ${GLM_PYTHON_OVERRIDES_DIR}; continuing without overrides."
    fi
  fi
  VLLM_CMD=(
    "${APPTAINER_BIN}" run --cleanenv --nv
  )
  for bind_path in "${APPTAINER_BINDS[@]}"; do
    VLLM_CMD+=(--bind "${bind_path}")
  done
  VLLM_CMD+=(
    "${VLLM_IMAGE}"
    --model "${MODEL_NAME}"
    --port "${PORT}"
    --tensor-parallel-size "${ALLOCATED_GPUS}"
    --max-model-len "${VLLM_MAX_MODEL_LEN}"
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
  )
fi
# vLLM flags.
if [ "${VLLM_SERVER_BACKEND}" = "apptainer" ] || [ "${VLLM_SERVER_BACKEND}" = "venv" ]; then
  if [ -n "${VLLM_MAX_NUM_BATCHED_TOKENS}" ]; then
    VLLM_CMD+=(--max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}")
  fi
  if [ "${VLLM_TRUST_REMOTE_CODE}" = "1" ] || [ "${VLLM_TRUST_REMOTE_CODE}" = "true" ]; then
    VLLM_CMD+=(--trust-remote-code)
  fi
  if [ -n "${VLLM_TOOL_CALL_PARSER:-}" ]; then
    VLLM_CMD+=(--tool-call-parser "${VLLM_TOOL_CALL_PARSER}")
  fi
  if [ -n "${VLLM_REASONING_PARSER:-}" ]; then
    VLLM_CMD+=(--reasoning-parser "${VLLM_REASONING_PARSER}")
  fi
  if [ "${VLLM_ENABLE_AUTO_TOOL_CHOICE:-}" = "1" ] || [ "${VLLM_ENABLE_AUTO_TOOL_CHOICE:-}" = "true" ]; then
    VLLM_CMD+=(--enable-auto-tool-choice)
  fi
  if [ "${VLLM_NO_ENABLE_LOG_REQUESTS:-}" = "1" ] || [ "${VLLM_NO_ENABLE_LOG_REQUESTS:-}" = "true" ]; then
    VLLM_CMD+=(--no-enable-log-requests)
  fi
  if [ "${VLLM_NO_ENABLE_PREFIX_CACHING:-}" = "1" ] || [ "${VLLM_NO_ENABLE_PREFIX_CACHING:-}" = "true" ]; then
    VLLM_CMD+=(--no-enable-prefix-caching)
  fi
fi

if [ "${VLLM_SERVER_BACKEND}" = "venv" ] && [[ "${MODEL_NAME}" == zai-org/GLM-* ]] && [ -f "${VLLM_GLM_TRANSFORMERS_SITE}/transformers/__init__.py" ]; then
  export PYTHONPATH="${VLLM_GLM_TRANSFORMERS_SITE}${PYTHONPATH:+:${PYTHONPATH}}"
  echo "Prepended VLLM_GLM_TRANSFORMERS_SITE to PYTHONPATH for venv vLLM."
fi

echo "Launching LM with backend: ${VLLM_SERVER_BACKEND}"
"${VLLM_CMD[@]}" > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!

cleanup() {
    if kill -0 "${VLLM_PID}" 2>/dev/null; then
        kill "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "Waiting for LM endpoint: ${ENDPOINT_URL}"
READY=0
for _ in $(seq 1 180); do
    if python - <<PY
import requests
import sys
try:
    r = requests.get("${ENDPOINT_URL}/models", timeout=3)
    print(r.status_code)
    sys.exit(0 if r.status_code < 500 else 1)
except Exception:
    sys.exit(1)
PY
    then
        READY=1
        break
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "LM server exited before becoming ready. See log: ${VLLM_LOG}"
        exit 3
    fi
    sleep 10
done

if [ "${READY}" != "1" ]; then
    echo "LM server did not become ready in time. See log: ${VLLM_LOG}"
    exit 4
fi

echo "Running model endpoint smoke test"
python - <<PY
import requests
payload = {
    "model": "${MODEL_NAME}",
    "messages": [{"role": "user", "content": "Reply with OK only."}],
    "max_tokens": 8,
    "temperature": 0,
}
r = requests.post("${ENDPOINT_URL}/chat/completions", json=payload, timeout=300)
print("chat status", r.status_code)
print(r.text[:500])
r.raise_for_status()
PY

echo "Running self-consistency smoke test against the live endpoint (sampled paths + majority vote)"
python - <<PY
import sys
from collections import Counter

sys.path.insert(0, "${REPO_ROOT}")
from openai import OpenAI

from model_server_config import extract_openai_message_text

client = OpenAI(base_url="${ENDPOINT_URL}", api_key="-", max_retries=1)
num_paths = 3
seed_base = int("${EXP2_ABLATION_SEED}")
answers = []
for k in range(num_paths):
    kwargs = dict(
        model="${MODEL_NAME}",
        messages=[
            {
                "role": "system",
                "content": (
                    "Solve the provided task. Do not ask back questions. Clearly indicate "
                    "your final solution after the text 'Final Solution:'."
                ),
            },
            {"role": "user", "content": "Task: What is 2 + 2?\nInput: 2 + 2 = ?"},
        ],
        max_tokens=256,
        temperature=float("${EXP2_ABLATION_TEMPERATURE}"),
        top_p=float("${EXP2_ABLATION_TOP_P}"),
        timeout=600,
    )
    if seed_base >= 0:
        kwargs["seed"] = seed_base + k
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as e:
        if seed_base >= 0 and "seed" in str(e).lower():
            print("Endpoint rejected 'seed'; retrying smoke without seeds")
            kwargs.pop("seed", None)
            resp = client.chat.completions.create(**kwargs)
        else:
            raise
    text = extract_openai_message_text(resp.choices[0].message)
    if not text.strip():
        raise SystemExit(f"Self-consistency smoke: empty completion for path {k}")
    answers.append(text.strip().splitlines()[-1])

print("Sampled final lines:", answers)
print("Vote counts:", Counter(answers).most_common())
print("Self-consistency smoke test OK")
PY

if [ "${EXP2_ABLATION_CONCURRENT_PROBE}" != "0" ]; then
echo "Running concurrent chat probe (${EXP2_ABLATION_CONCURRENT_PROBE} parallel /chat/completions)"
python - <<PY
import concurrent.futures

import requests

n = int("${EXP2_ABLATION_CONCURRENT_PROBE}")
base = "${ENDPOINT_URL}".rstrip("/")
url = f"{base}/chat/completions"
model = "${MODEL_NAME}"

def one(i: int) -> None:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": f"Probe {i}: reply with the single digit 1 only."}],
        "max_tokens": 4,
        "temperature": 0,
    }
    r = requests.post(url, json=payload, timeout=900)
    if r.status_code >= 500:
        raise RuntimeError(f"status {r.status_code}: {r.text[:500]}")
    r.raise_for_status()

with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
    list(ex.map(one, range(n)))
print("Concurrent chat probe OK:", n)
PY
fi

if [ "${SMOKE_ONLY}" = "1" ]; then
    echo "SMOKE_ONLY=1 -> endpoint and self-consistency smoke checks passed. Skipping full exp2_ablation.py run."
    exit 0
fi

DEBUG_FLAG=""
if [ "${DEBUG}" = "1" ]; then
    DEBUG_FLAG="--debug"
fi

EXP2_ABLATION_CMD=(
    python exp2_ablation.py
    --endpoint_url "${ENDPOINT_URL}"
    --model_name "${MODEL_NAME}"
    --dataset "${DATASET}"
    --output_dir "${EXP2_ABLATION_RUN_DIR}"
    --num_samples "${EXP2_ABLATION_NUM_SAMPLES}"
    --temperature "${EXP2_ABLATION_TEMPERATURE}"
    --top_p "${EXP2_ABLATION_TOP_P}"
    --seed "${EXP2_ABLATION_SEED}"
    --max_tokens "${EXP2_ABLATION_MAX_TOKENS}"
    --concurrent_requests "${MALLM_CONCURRENT_API_REQUESTS}"
    --continue
)
if [ -n "${DEBUG_FLAG}" ]; then
    EXP2_ABLATION_CMD+=("${DEBUG_FLAG}")
fi
# Forward extra script args to exp2_ablation.py, e.g.:
#   sbatch exp2_ablation.slurm --num_samples 25
if [ "$#" -gt 0 ]; then
    EXP2_ABLATION_CMD+=("$@")
fi

echo "Running exp2 ablation command: ${EXP2_ABLATION_CMD[*]}"
"${EXP2_ABLATION_CMD[@]}"
echo "finished_at=$(date -Is 2>/dev/null || date)" >> "${EXP2_ABLATION_RUN_DIR}/run_info.env"
echo "[INFO] exp2 ablation complete. Results in ${EXP2_ABLATION_RUN_DIR}"
