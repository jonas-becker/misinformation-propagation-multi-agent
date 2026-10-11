"""
Shared model-name normalization and vLLM serving defaults for the experiment runners.

Aligned with upstream recipes:
  - GLM: https://docs.vllm.ai/projects/recipes/en/latest/GLM/GLM.html
  - GPT-OSS: https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html
"""

from __future__ import annotations

import os
from typing import Any, List


def normalize_model_name(name: str) -> str:
    """Map CLI/Slurm aliases to Hugging Face hub ids."""
    if not isinstance(name, str):
        return ""
    raw = name.strip()
    key = raw.lower().replace("_", "-")
    aliases = {
        "llama-3.3-70b-instruct": "meta-llama/Llama-3.3-70B-Instruct",
        "meta-llama/llama-3.3-70b-instruct": "meta-llama/Llama-3.3-70B-Instruct",
        "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
        "llama-3.2-3b-instruct": "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/llama-3.2-3b-instruct": "meta-llama/Llama-3.2-3B-Instruct",
        "gpt-oss-120b": "openai/gpt-oss-120b",
        "openai/gpt-oss-120b": "openai/gpt-oss-120b",
        "gpt-oss-20b": "openai/gpt-oss-20b",
        "openai/gpt-oss-20b": "openai/gpt-oss-20b",
        "glm-4.7-flash": "zai-org/GLM-4.7-Flash",
        "zai-org/glm-4.7-flash": "zai-org/GLM-4.7-Flash",
    }
    return aliases.get(key, raw)


def is_glm_model(model_name: str) -> bool:
    return (model_name or "").startswith("zai-org/GLM-")


def is_gpt_oss_model(model_name: str) -> bool:
    m = (model_name or "").lower()
    return m.startswith("openai/gpt-oss") or "gpt-oss" in m


def is_llama_70b_model(model_name: str) -> bool:
    m = (model_name or "").lower()
    return "llama-3.3-70b" in m or "llama-3.1-70b" in m or "llama3.3-70b" in m


def vllm_max_model_len_default(model_name: str) -> int:
    if is_glm_model(model_name):
        return 65536
    if is_gpt_oss_model(model_name):
        # Full context is 131072; use a practical default for KV cache on 80GB nodes.
        return 32768
    if is_llama_70b_model(model_name):
        return 8192
    return 8192


def vllm_gpu_memory_utilization_default(model_name: str) -> float:
    if is_glm_model(model_name) or is_llama_70b_model(model_name) or is_gpt_oss_model(model_name):
        return 0.90
    return 0.98


def vllm_trust_remote_code_default(model_name: str) -> bool:
    if is_glm_model(model_name) or (model_name or "").startswith("meta-llama/"):
        return True
    return False


def vllm_tool_call_parser_default(model_name: str) -> str:
    if is_gpt_oss_model(model_name):
        return "openai"
    if not is_glm_model(model_name):
        return ""
    m = model_name.lower()
    if "glm-4.5" in m or "glm4.5" in m:
        return "glm45"
    if "glm-4.6" in m or "glm4.6" in m or "glm-4.7" in m or "glm4.7" in m or "glm-5" in m or "glm5" in m:
        return "glm47"
    return ""


def vllm_reasoning_parser_default(model_name: str) -> str:
    # Do NOT default reasoning-parser for MALLM: it routes text into delta.reasoning /
    # reasoning_content and leaves delta.content empty. MALLM concatenates content first
    # but exp1a also reads reasoning_content as fallback.
    return ""


def vllm_enable_auto_tool_choice_default(model_name: str) -> bool:
    return is_glm_model(model_name) or is_gpt_oss_model(model_name)


def vllm_no_enable_log_requests_default(model_name: str) -> bool:
    return is_glm_model(model_name) or is_gpt_oss_model(model_name)


def vllm_no_enable_prefix_caching_default(model_name: str) -> bool:
    return is_glm_model(model_name)


def vllm_extra_args_for_model(model_name: str) -> List[str]:
    """
    Optional vLLM flags per model family. Env overrides:
      VLLM_TOOL_CALL_PARSER, VLLM_REASONING_PARSER,
      VLLM_ENABLE_AUTO_TOOL_CHOICE, VLLM_NO_ENABLE_LOG_REQUESTS,
      VLLM_NO_ENABLE_PREFIX_CACHING
    """
    extra: List[str] = []
    tcp = (os.environ.get("VLLM_TOOL_CALL_PARSER") or "").strip()
    if not tcp:
        tcp = vllm_tool_call_parser_default(model_name)
    if tcp:
        extra += ["--tool-call-parser", tcp]

    rp = (os.environ.get("VLLM_REASONING_PARSER") or "").strip()
    if not rp:
        rp = vllm_reasoning_parser_default(model_name)
    if rp:
        extra += ["--reasoning-parser", rp]

    eatc = (os.environ.get("VLLM_ENABLE_AUTO_TOOL_CHOICE") or "").strip().lower()
    if not eatc and vllm_enable_auto_tool_choice_default(model_name):
        eatc = "1"
    if eatc in ("1", "true", "yes"):
        extra.append("--enable-auto-tool-choice")

    nlr = (os.environ.get("VLLM_NO_ENABLE_LOG_REQUESTS") or "").strip().lower()
    if not nlr and vllm_no_enable_log_requests_default(model_name):
        nlr = "1"
    if nlr in ("1", "true", "yes"):
        extra.append("--no-enable-log-requests")

    npc = (os.environ.get("VLLM_NO_ENABLE_PREFIX_CACHING") or "").strip().lower()
    if not npc and vllm_no_enable_prefix_caching_default(model_name):
        npc = "1"
    if npc in ("1", "true", "yes"):
        extra.append("--no-enable-prefix-caching")

    return extra


def extract_openai_message_text(message: Any) -> str:
    """Best-effort assistant text from a chat completion message (gpt-oss aware)."""
    if message is None:
        return ""
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content.strip()
    for attr in ("reasoning_content", "reasoning"):
        alt = getattr(message, attr, None)
        if isinstance(alt, str) and alt.strip():
            return alt.strip()
    if content is not None:
        return str(content).strip()
    return ""


def gpt_oss_tiktoken_dir() -> str:
    """Host directory with o200k_base.tiktoken and cl100k_base.tiktoken for offline gpt-oss."""
    candidates = [
        (os.environ.get("GPT_OSS_TIKTOKEN_DIR") or "").strip(),
        (os.environ.get("SHARED_HF_CACHE_VOLUME") or "") and os.path.join(os.environ["SHARED_HF_CACHE_VOLUME"], "tiktoken_encodings"),
        os.path.expanduser("~/.cache/tiktoken_encodings"),
    ]
    required = ("o200k_base.tiktoken", "cl100k_base.tiktoken")
    for directory in candidates:
        if not directory:
            continue
        if all(os.path.isfile(os.path.join(directory, name)) for name in required):
            return directory
    return ""


_TIKTOKEN_BLOB_BASE = "https://openaipublic.blob.core.windows.net/encodings"
_TIKTOKEN_CACHE: dict = {}


def tiktoken_cache_dir() -> str:
    """Host-side cache directory for the Python `tiktoken` package (TIKTOKEN_CACHE_DIR).

    `tiktoken` looks vocab files up under sha1(<download url>) in TIKTOKEN_CACHE_DIR and only
    downloads when that file is missing. Compute nodes have no outbound internet, so a missing
    cache entry makes every `tiktoken.get_encoding()` hang until the TCP timeout (~2 min).
    """
    configured = (os.environ.get("TIKTOKEN_CACHE_DIR") or "").strip()
    return configured or os.path.expanduser("~/.cache/tiktoken_py_cache")


def prime_tiktoken_cache(names=("o200k_base", "cl100k_base")) -> str:
    """Copy the offline vocab files into the tiktoken cache under the names tiktoken expects.

    Returns the cache directory (and exports TIKTOKEN_CACHE_DIR) or "" when the offline vocab
    files are not available. Never touches the network.
    """
    import hashlib
    import shutil

    source_dir = gpt_oss_tiktoken_dir()
    if not source_dir:
        return ""
    cache_dir = tiktoken_cache_dir()
    try:
        os.makedirs(cache_dir, exist_ok=True)
        for name in names:
            src = os.path.join(source_dir, f"{name}.tiktoken")
            if not os.path.isfile(src):
                continue
            url = f"{_TIKTOKEN_BLOB_BASE}/{name}.tiktoken"
            dst = os.path.join(cache_dir, hashlib.sha1(url.encode()).hexdigest())
            if not os.path.isfile(dst) or os.path.getsize(dst) != os.path.getsize(src):
                tmp = f"{dst}.tmp.{os.getpid()}"
                shutil.copyfile(src, tmp)
                os.replace(tmp, dst)
    except OSError:
        return ""
    os.environ["TIKTOKEN_CACHE_DIR"] = cache_dir
    return cache_dir


def offline_tiktoken_encoding(name: str = "o200k_base"):
    """`tiktoken.get_encoding(name)` from the offline vocab files when they are present (cluster nodes
    without internet; no download is attempted), otherwise from tiktoken's own cache with one bounded
    download attempt (workstations). None if unavailable; the result is memoised.
    """
    if name in _TIKTOKEN_CACHE:
        return _TIKTOKEN_CACHE[name]
    enc = None
    try:
        import requests
        import tiktoken

        original_get = requests.get
        if prime_tiktoken_cache():
            def _patched_get(*_args, **_kwargs):
                raise RuntimeError("tiktoken download disabled (offline cache miss)")
        else:
            # no offline vocab directory: tiktoken's own cache, downloading once if needed; the bounded
            # timeout lets a node without internet fail fast (callers then count words) instead of hanging
            def _patched_get(*args, **kwargs):
                kwargs.setdefault("timeout", 15)
                return original_get(*args, **kwargs)
        requests.get = _patched_get  # tiktoken.load.read_file resolves requests.get at call time
        try:
            enc = tiktoken.get_encoding(name)
        finally:
            requests.get = original_get
    except Exception:  # noqa: BLE001 - callers fall back to whitespace counting
        enc = None
    _TIKTOKEN_CACHE[name] = enc
    return enc


def apply_gpt_oss_tiktoken_env(model_name: str) -> None:
    """Set TIKTOKEN_* env vars required for offline vLLM gpt-oss serving."""
    if not is_gpt_oss_model(model_name):
        return
    directory = gpt_oss_tiktoken_dir()
    if not directory:
        raise RuntimeError(
            "gpt-oss requires offline tiktoken vocab files. "
            "Run: bash slurm/download_gpt_oss_tiktoken.sh"
        )
    os.environ["TIKTOKEN_ENCODINGS_BASE"] = directory
    os.environ["TIKTOKEN_RS_CACHE_DIR"] = directory


def mallm_concurrency_default(_model_name: str) -> int:
    raw = (os.environ.get("MALLM_CONCURRENT_API_REQUESTS") or "").strip()
    if raw.isdigit():
        return max(1, int(raw))
    return 64
