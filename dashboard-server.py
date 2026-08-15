#!/usr/bin/env python3
import ast
import json
import os
import re
import select
import shlex
import sqlite3
import subprocess
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HOST = "0.0.0.0"
PORT = int(os.environ.get("DASHBOARD_PORT", "8080"))
REQUEST_LOG_DB = os.environ.get("REQUEST_LOG_DB", "/request-logs/requests.db")

# Columns returned by /api/request-log (subset/order for the UI table).
_REQUEST_LOG_COLUMNS = [
    "id", "ts", "ts_iso", "endpoint", "client_ip", "forwarded_for", "user_agent",
    "method", "path", "model", "stream", "status", "latency_ms",
    "req_bytes", "resp_bytes", "prompt_tokens", "completion_tokens",
    "total_tokens", "prompt_snippet", "error",
]


def read_request_log(params):
    """Read rows from the shared request-log SQLite DB (read-only).

    params: dict of lists (from parse_qs). Supported keys:
      limit (int, default 200, max 2000), endpoint ('8080'|'28082'|'all'),
      q (substring over ip/ua/model/path/snippet), since (epoch seconds float).
    Returns {"rows": [...], "available": bool, "error": str|None}.
    """
    def _one(key, default=None):
        vals = params.get(key)
        return vals[0] if vals else default

    if not os.path.exists(REQUEST_LOG_DB):
        return {"rows": [], "available": False, "error": None}

    try:
        limit = int(_one("limit", "200"))
    except (TypeError, ValueError):
        limit = 200
    limit = max(1, min(limit, 2000))

    endpoint = _one("endpoint", "all")
    q = _one("q", "").strip()
    since = _one("since")

    where = []
    args = []
    if endpoint and endpoint != "all":
        where.append("endpoint = ?")
        args.append(endpoint)
    if since:
        try:
            where.append("ts >= ?")
            args.append(float(since))
        except ValueError:
            pass
    # Exclude root '/' and health check endpoints by default unless explicitly queried
    if not q or (q.lower() not in ("/", "/health", "/health/")):
        where.append("IFNULL(path,'') NOT IN ('', '/', '/health', '/health/')")
    # Exclude low token count requests (< 5) by default unless querying
    if not q:
        where.append("(total_tokens IS NULL OR total_tokens >= 5)")
    if q:
        like = f"%{q}%"
        where.append(
            "(IFNULL(client_ip,'') LIKE ? OR IFNULL(user_agent,'') LIKE ? "
            "OR IFNULL(model,'') LIKE ? OR IFNULL(path,'') LIKE ? "
            "OR IFNULL(prompt_snippet,'') LIKE ?)"
        )
        args.extend([like, like, like, like, like])

    sql = f"SELECT {','.join(_REQUEST_LOG_COLUMNS)} FROM requests"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY id DESC LIMIT ?"
    args.append(limit)

    try:
        uri = f"file:{REQUEST_LOG_DB}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=5.0)
        try:
            conn.execute("PRAGMA query_only=ON")
            conn.row_factory = sqlite3.Row
            rows = [dict(r) for r in conn.execute(sql, args).fetchall()]
        finally:
            conn.close()
        return {"rows": rows, "available": True, "error": None}
    except Exception as e:  # noqa: BLE001
        return {"rows": [], "available": True, "error": str(e)}


def read_request_detail(params):
    """Read a single request row including the heavy request/response bodies."""
    vals = params.get("id")
    if not vals:
        return {"row": None, "error": "missing id"}
    try:
        row_id = int(vals[0])
    except (TypeError, ValueError):
        return {"row": None, "error": "bad id"}
    if not os.path.exists(REQUEST_LOG_DB):
        return {"row": None, "error": "no log db"}
    cols = _REQUEST_LOG_COLUMNS + ["request_body", "response_body"]
    try:
        uri = f"file:{REQUEST_LOG_DB}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=5.0)
        try:
            conn.execute("PRAGMA query_only=ON")
            conn.row_factory = sqlite3.Row
            r = conn.execute(
                f"SELECT {','.join(cols)} FROM requests WHERE id = ?", (row_id,)
            ).fetchone()
        finally:
            conn.close()
        return {"row": dict(r) if r else None, "error": None}
    except Exception as e:  # noqa: BLE001
        return {"row": None, "error": str(e)}


def _strip_inline_comment(value: str) -> str:
    quote = ""
    escaped = False
    out = []
    for ch in value:
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if ch == "\\":
            out.append(ch)
            escaped = True
            continue
        if quote:
            if ch == quote:
                quote = ""
            out.append(ch)
            continue
        if ch in ("'", '"'):
            quote = ch
            out.append(ch)
            continue
        if ch == "#":
            break
        out.append(ch)
    return "".join(out).strip()


def _parse_config_scalar(raw: str):
    value = _strip_inline_comment(raw)
    if value == "":
        return ""
    lower = value.lower()
    if lower in ("null", "~"):
        return None
    if lower == "true":
        return True
    if lower == "false":
        return False
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        try:
            return ast.literal_eval(value)
        except Exception:
            return value[1:-1]
    return value


def _load_yaml_kv_file(path: Path) -> dict:
    data = {}
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return data
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if raw_line[:1].isspace() or ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        key = key.strip()
        if not key:
            continue
        data[key] = _parse_config_scalar(value.strip())
    return data


def load_runtime_config() -> dict:
    config = {}
    config_path = Path(os.environ.get("DASHBOARD_CONFIG", "config.yaml"))
    local_path = Path(os.environ.get("DASHBOARD_CONFIG_LOCAL", "config.local.yaml"))
    for path in (config_path, local_path):
        if path.is_file():
            config.update(_load_yaml_kv_file(path))
    return config


RUNTIME_CONFIG = load_runtime_config()
DEFAULT_LLAMA_DIR = str(RUNTIME_CONFIG.get("llama_dir") or "/opt/llama.cpp")
LLAMA_DIR = Path(os.environ.get("LLAMA_DIR", DEFAULT_LLAMA_DIR))

DEFAULT_STACKS_DIR = str(RUNTIME_CONFIG.get("stacks_dir") or (LLAMA_DIR / "stacks"))
STACKS_DIR = Path(os.environ.get("STACKS_DIR", DEFAULT_STACKS_DIR))

MODELS_DIR = Path(
    os.environ.get("MODELS_DIR", str(RUNTIME_CONFIG.get("models_dir") or (LLAMA_DIR / "models")))
)

DEFAULT_SWITCH_SCRIPT = str(RUNTIME_CONFIG.get("switch_script") or (LLAMA_DIR / "switch-llm.sh"))
SWITCH_SCRIPT = Path(os.environ.get("SWITCH_SCRIPT", DEFAULT_SWITCH_SCRIPT))

LLAMA_WORKING_DIR_LABEL = os.environ.get(
    "LLAMA_WORKING_DIR_LABEL",
    str(RUNTIME_CONFIG.get("llama_working_dir_label") or "/opt/llama.cpp"),
)

# ── vLLM config ──────────────────────────────────────────────────────────────
DEFAULT_VLLM_DIR = str(RUNTIME_CONFIG.get("vllm_dir") or "/opt/vllm")
VLLM_DIR = Path(os.environ.get("VLLM_DIR", DEFAULT_VLLM_DIR))
VLLM_SERVER_SERVICE = os.environ.get(
    "VLLM_SERVER_SERVICE",
    str(RUNTIME_CONFIG.get("vllm_server_service") or "vllm-server"),
)
VLLM_WORKING_DIR_LABEL = os.environ.get(
    "VLLM_WORKING_DIR_LABEL",
    str(RUNTIME_CONFIG.get("vllm_working_dir_label") or str(VLLM_DIR)),
)
# ── BeeLlama config ──────────────────────────────────────────────────────────
DEFAULT_BEELLAMA_DIR = str(RUNTIME_CONFIG.get("beellama_dir") or "/opt/beellama.cpp")
BEELLAMA_DIR = Path(os.environ.get("BEELLAMA_DIR", DEFAULT_BEELLAMA_DIR))
BEELLAMA_WORKING_DIR_LABEL = os.environ.get(
    "BEELLAMA_WORKING_DIR_LABEL",
    str(RUNTIME_CONFIG.get("beellama_working_dir_label") or str(BEELLAMA_DIR)),
)
BEELLAMA_STACKS_DIR = Path(os.environ.get(
    "BEELLAMA_STACKS_DIR",
    str(RUNTIME_CONFIG.get("beellama_stacks_dir") or str(BEELLAMA_DIR / "stacks")),
))

SWITCH_READY_TIMEOUT_SEC = int(os.environ.get("SWITCH_READY_TIMEOUT_SEC", "900"))
SWITCH_POLL_SEC = float(os.environ.get("SWITCH_POLL_SEC", "5"))
SWITCH_STALE_SEC = int(
    os.environ.get("SWITCH_STALE_SEC", str(max(SWITCH_READY_TIMEOUT_SEC + 120, 1020)))
)
THROUGHPUT_LOG_TAIL_LINES = int(os.environ.get("THROUGHPUT_LOG_TAIL_LINES", "500"))
THROUGHPUT_CACHE_TTL_SEC = float(os.environ.get("THROUGHPUT_CACHE_TTL_SEC", "1"))
BENCHMARK_TIMEOUT_SEC = int(os.environ.get("BENCHMARK_TIMEOUT_SEC", "180"))
BENCHMARK_PROMPT_TOKENS = int(os.environ.get("BENCHMARK_PROMPT_TOKENS", "768"))
BENCHMARK_N_PREDICT = int(os.environ.get("BENCHMARK_N_PREDICT", "256"))
FULL_BENCHMARK_MAX_HISTORY = int(os.environ.get("BENCHMARK_HISTORY_LIMIT", "10"))
BENCHMARK_STALE_SEC = int(
    os.environ.get("BENCHMARK_STALE_SEC", str(max(BENCHMARK_TIMEOUT_SEC * 4, 600)))
)
FULL_BENCHMARK_SPECS = [
    {"name": "prefill_xl", "prompt_tokens": 4096, "n_predict": 32},
    {"name": "prefill_heavy", "prompt_tokens": 2048, "n_predict": 64},
    {"name": "prefill_medium", "prompt_tokens": 1024, "n_predict": 128},
    {"name": "mixed", "prompt_tokens": 1024, "n_predict": 256},
    {"name": "mixed_even", "prompt_tokens": 512, "n_predict": 512},
    {"name": "gen_heavy", "prompt_tokens": 256, "n_predict": 1024},
    {"name": "gen_xl", "prompt_tokens": 128, "n_predict": 2048},
]
FULL_BENCHMARK_REPEATS = 2
FULL_BENCHMARK_RUNS = len(FULL_BENCHMARK_SPECS) * FULL_BENCHMARK_REPEATS
LOG_MAX_EVENTS = int(os.environ.get("LOG_MAX_EVENTS", "100"))
LOG_WATCHER_INTERVAL_SEC = float(os.environ.get("LOG_WATCHER_INTERVAL_SEC", "2.0"))
HEARTBEAT_INTERVAL_SEC = float(os.environ.get("HEARTBEAT_INTERVAL_SEC", "30.0"))
RESTART_LOOP_THRESHOLD = 3
RESTART_LOOP_WINDOW_SEC = 300.0

# ---------------------------------------------------------------------------
# Dynamic model discovery from stacks/*.yml LLM_META headers
# ---------------------------------------------------------------------------
MODELS_CACHE_TTL_SEC = 30.0
_models_ts = 0.0
_models_data: dict = {}


def parse_llm_meta(path: Path) -> dict:
    meta = {}
    with open(path) as f:
        for line in f:
            if not line.startswith("# LLM_META "):
                if line.strip() and not line.startswith("#"):
                    break
                continue
            m = re.match(r'^# LLM_META (\w+)="([^"]*)"', line)
            if m:
                meta[m.group(1)] = m.group(2)
    return meta


def discover_models() -> dict:
    models = {}
    # Discover llama.cpp stacks
    if STACKS_DIR.is_dir():
        entries = []
        for path in STACKS_DIR.glob("*.yml"):
            meta = parse_llm_meta(path)
            if not meta.get("display_name"):
                continue
            rel = f"stacks/{path.name}"
            try:
                order = float(meta.get("sort_order", 999))
            except ValueError:
                order = 999
            entries.append((order, path.stem, rel, meta))
        entries.sort(key=lambda x: (x[0], x[1]))
        for _, stem, rel, meta in entries:
            models[stem] = {
                "label":          meta["display_name"],
                "compose":        rel,
                "server_service": meta.get("server_service", ""),
                "thinking":       meta.get("thinking", "false") == "true",
                "vision":           meta.get("vision", "false") == "true",
                "family":         meta.get("family", "other"),
                "ctx_size":       meta.get("ctx_size", ""),
                "quant":          meta.get("quant", ""),
                "params":         meta.get("params", ""),
                "internal_port":  int(meta.get("internal_port", "8080")),
            }

    # Add vLLM stacks — discover any *.yml in VLLM_DIR with LLM_META headers.
    # Falls back to a generic entry when no metadata is found.
    if VLLM_DIR.is_dir():
        vllm_entries = []
        for path in sorted(VLLM_DIR.glob("*.yml")):
            meta = parse_llm_meta(path)
            if not meta.get("display_name"):
                continue
            try:
                order = float(meta.get("sort_order", "999"))
            except ValueError:
                order = 999
            vllm_entries.append((order, path.stem, path, meta))
        vllm_entries.sort(key=lambda x: (x[0], x[1]))
        for _, stem, path, meta in vllm_entries:
            key = f"vllm-{stem}"
            models[key] = {
                "label":          meta["display_name"],
                "compose":        path.name,
                "compose_file":   str(path),
                "server_service": meta.get("server_service", VLLM_SERVER_SERVICE),
                "thinking":       meta.get("thinking", "false") == "true",
                "vision":         meta.get("vision", "false") == "true",
                "family":         meta.get("family", "vllm"),
                "ctx_size":       meta.get("ctx_size", ""),
                "quant":          meta.get("quant", ""),
                "params":         meta.get("params", ""),
                "vllm":           True,
            }
        if not vllm_entries:
            models["vllm"] = {
                "label":          "vLLM",
                "compose":        "docker-compose.yml",
                "compose_file":   str(VLLM_DIR / "docker-compose.yml"),
                "server_service": VLLM_SERVER_SERVICE,
                "thinking":       False,
                "vision":         False,
                "family":         "vllm",
                "ctx_size":       "",
                "quant":          "",
                "params":         "",
                "vllm":           True,
            }
    # Add BeeLlama stacks from beellama.cpp/stacks/*.yml
    if BEELLAMA_DIR.is_dir() and BEELLAMA_STACKS_DIR.is_dir():
        entries = []
        for path in BEELLAMA_STACKS_DIR.glob("*.yml"):
            meta = parse_llm_meta(path)
            if not meta.get("display_name"):
                continue
            rel = f"beellama-stacks/{path.name}"
            try:
                order = float(meta.get("sort_order", "999"))
            except ValueError:
                order = 999
            entries.append((order, path.stem, rel, meta))
        entries.sort(key=lambda x: (x[0], x[1]))
        for _, stem, rel, meta in entries:
            models[f"beellama-{stem}"] = {
                "label":          meta["display_name"],
                "compose":        rel,
                "server_service": meta.get("server_service", ""),
                "thinking":       meta.get("thinking", "false") == "true",
                "vision":         meta.get("vision", "false") == "true",
                "family":         meta.get("family", "other"),
                "ctx_size":       meta.get("ctx_size", ""),
                "quant":          meta.get("quant", ""),
                "params":         meta.get("params", ""),
                "beellama":       True,
                "internal_port":  int(meta.get("internal_port", "8085")),
            }

    return models


def get_models() -> tuple[dict, dict]:
    global _models_ts, _models_data
    if time.time() - _models_ts < MODELS_CACHE_TTL_SEC:
        return _models_data, {v["compose"]: k for k, v in _models_data.items()}
    _models_data = discover_models()
    _models_ts = time.time()
    return _models_data, {v["compose"]: k for k, v in _models_data.items()}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
STATE_LOCK = threading.Lock()
STATE = {
    "switch_in_progress": False,
    "last_requested_model": None,
    "last_started_at": None,
    "last_completed_at": None,
    "last_exit_code": None,
    "last_message": "",
    "last_output": "",
}
THROUGHPUT_CACHE_LOCK = threading.Lock()
THROUGHPUT_CACHE = {
    "active_key": None,
    "container": None,
    "checked_at": 0.0,
    "result": None,
}
LIVE_TPS_LOCK = threading.Lock()
LIVE_TPS_STATE = {
    "active_key": None,
    "container": None,
    "sampled_at": 0.0,
    "decoded_tokens": None,
}

# Live TPS cache for fast API responses
LIVE_TPS_CACHE_TTL_SEC = float(os.environ.get("LIVE_TPS_CACHE_TTL_SEC", "0.25"))
LIVE_TPS_CACHE_LOCK = threading.Lock()
LIVE_TPS_CACHE = {
    "active_key": None,
    "container": None,
    "checked_at": 0.0,
    "result": None,
}
INGEST_LIVE_STATE_LOCK = threading.Lock()
INGEST_LIVE_STATE = {
    "active_key": None,
    "ingest_tps": None,
    "ingest_start_ts": None,
    "ingest_start_tokens": None,
}
CONTEXT_STATE_LOCK = threading.Lock()
CONTEXT_STATE = {
    "active_key": None,
    "n_tokens": None,
    "n_ctx_slot": None,
    "n_prompt_tokens": None,
    "task_n_tokens": None,
    "last_good_n_past": None,
}

# ── vLLM Prometheus metrics cache ────────────────────────────────────────────
VLLM_METRICS_CACHE_TTL_SEC = float(os.environ.get("VLLM_METRICS_CACHE_TTL_SEC", "1.0"))
VLLM_METRICS_CACHE_LOCK = threading.Lock()
VLLM_METRICS_CACHE = {
    "container": None,
    "checked_at": 0.0,
    "raw": None,
    "generation_tokens_total": 0.0,
    "prompt_tokens_total": 0.0,
    "iteration_tokens_total_sum": 0.0,
    "iteration_tokens_total_count": 0.0,
    "request_queue_time_seconds_sum": 0.0,
    "request_queue_time_seconds_count": 0.0,
    "time_to_first_token_seconds_sum": 0.0,
    "time_to_first_token_seconds_count": 0.0,
    "request_gen_tokens_sum": 0.0,
    "request_gen_tokens_count": 0.0,
    "request_prompt_tokens_sum": 0.0,
    "request_prompt_tokens_count": 0.0,
}
VLLM_TPS_STATE_LOCK = threading.Lock()
VLLM_TPS_STATE = {
    "container": None,
    "sampled_at": 0.0,
    "generation_tokens": None,
}
VLLM_INGEST_STATE_LOCK = threading.Lock()
VLLM_INGEST_STATE = {
    "container": None,
    "sampled_at": 0.0,
    "prompt_tokens": None,
}

EVAL_TPS_PATTERN = re.compile(
    r"eval time\s*=.*?\(\s*[0-9.]+\s+ms per token,\s*([0-9.]+)\s+tokens per second\)"
)
INGEST_TPS_PATTERN = re.compile(
    r"prompt eval time\s*=\s*[0-9.]+\s*ms\s*/\s*([0-9]+)\s*tokens\s*\(.*?([0-9.]+)\s+tokens per second\)"
)
TIMING_TASK_PATTERN = re.compile(r"slot print_timing: id\s+\d+\s+\|\s+task\s+(\d+)\s+\|")
SLOT_N_TOKENS_PATTERN = re.compile(
    r"slot\s+(?:update_slots|release|load):\s+id\s+\d+\s+\|\s+task\s+\d+\s+\|\s+.*n_tokens\s*=\s*(\d+)"
)
SLOT_N_CTX_PATTERN = re.compile(r"n_ctx_slot\s*=\s*(\d+)")
SLOT_PROMPT_DONE_PATTERN = re.compile(
    r"slot update_slots:.*prompt processing done.*n_tokens\s*=\s*(\d+)"
)
SLOT_TASK_TOKENS_PATTERN = re.compile(
    r"slot update_slots:.*task\.n_tokens\s*=\s*(\d+)"
)
_LOG_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?\s+")
_LOG_PATTERNS = [
    ("error",   "oom",        re.compile(r"CUDA error: out of memory|cudaMalloc failed|not enough memory|CUDA_ERROR_OUT_OF_MEMORY", re.I)),
    ("error",   "model_load", re.compile(r"error loading model|failed to load model|unable to open model|llama_model_load: error|failed to mmap", re.I)),
    ("error",   "config",     re.compile(r"invalid argument|error: unknown argument|unrecognized option", re.I)),
    ("warning", "health",     re.compile(r"health check failed|unhealthy", re.I)),
    ("info",    "loading",    re.compile(r"llm_load_tensors|llama_new_context_with_model|print_info|model size\s*=|loaded meta data", re.I)),
    ("info",    "loading",    re.compile(r"\.\s+\d+\.?\d*\s*[%％]", re.I)),
    ("info",    "ready",      re.compile(r"HTTP server listening|server is listening", re.I)),
]
MODEL_STATS_LOCK = threading.Lock()
MODEL_STATS = {
    "active_key": None,
    "stats": None,
}
BENCHMARK_LOCK = threading.Lock()
BENCHMARK_STATE = {
    "in_progress": False,
    "profile": "balanced",
    "started_at": None,
    "completed_at": None,
    "last_result": None,
    "last_error": None,
    "history": [],
}
LOG_LOCK = threading.Lock()
LOG_STATE = {
    "events": [],
    "error_count": 0,
    "last_container": None,
    "last_restart_count": None,
    "restart_times": [],
    "watcher_alive": False,
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_iso_utc(ts: str | None) -> datetime | None:
    if not ts or not isinstance(ts, str):
        return None
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def _set_last_message(msg: str) -> None:
    with STATE_LOCK:
        STATE["last_message"] = msg


def _append_log_event(severity: str, category: str, message: str) -> None:
    event = {
        "ts": now_iso(),
        "_ts_f": time.time(),
        "severity": severity,
        "category": category,
        "message": message[:300],
    }
    with LOG_LOCK:
        LOG_STATE["events"].append(event)
        if len(LOG_STATE["events"]) > LOG_MAX_EVENTS:
            removed = LOG_STATE["events"].pop(0)
            if removed["severity"] == "error":
                LOG_STATE["error_count"] = max(0, LOG_STATE["error_count"] - 1)
        if severity == "error":
            LOG_STATE["error_count"] += 1


def _has_recent_event(category: str, within_sec: float = 60.0) -> bool:
    cutoff = time.time() - within_sec
    with LOG_LOCK:
        for evt in reversed(LOG_STATE["events"]):
            if evt["_ts_f"] < cutoff:
                break
            if evt["category"] == category:
                return True
    return False


def _human_readable_path(path: str, body: dict | None = None) -> str:
    if path == "/api/switch":
        model = (body or {}).get("model", "?")
        return f"Model switch to {model}"
    if path == "/api/stop":
        return "Stop all models"
    if path == "/api/restart":
        return "Restart active model"
    if path == "/api/benchmark":
        profile = (body or {}).get("profile", "balanced")
        return f"Run {profile} benchmark"
    if path == "/api/reset":
        return "Emergency reset state"
    if path in ("/api/status",):
        return "Status poll"
    if path in ("/", "/index.html"):
        return "Dashboard page"
    return path


def _build_heartbeat_summary(active_key: str, containers: list[dict]) -> str | None:
    models, _ = get_models()
    model = models.get(active_key, {})
    label = model.get("label", active_key)
    throughput = build_throughput_status(active_key, containers)
    live = build_live_throughput_status(active_key, containers)
    gpu_stats = get_gpu_stats()

    parts = []
    if live.get("tokens_per_second") not in (None, 0.0):
        parts.append(f"Gen: {live['tokens_per_second']:.1f} T/S")
    elif throughput.get("tokens_per_second") is not None:
        parts.append(f"Last Gen: {throughput['tokens_per_second']:.1f} T/S")

    ctx_n = live.get("n_ctx")
    decoded = live.get("decoded_tokens") or 0
    with CONTEXT_STATE_LOCK:
        base = CONTEXT_STATE.get("task_n_tokens") or CONTEXT_STATE.get("n_prompt_tokens") or CONTEXT_STATE.get("n_tokens")
    if ctx_n is not None and base is not None and base > 10:
        n_past = base + decoded
        parts.append(f"Ctx: {n_past}/{ctx_n}")

    if gpu_stats:
        g0 = gpu_stats[0]
        parts.append(f"GPU: {g0['util']:.0f}%  VRAM: {g0['mem_used']:.0f}/{g0['mem_total']:.0f}MB  {g0['temp']:.0f}°C")

    if not parts:
        return f"Heartbeat | {label} | idle"
    return f"Heartbeat | {label} | " + "  ".join(parts)


def _process_log_line(raw_line: str, active_key: str | None = None) -> None:
    m_prompt_done = None
    match = _LOG_TS_RE.search(raw_line)
    clean = _LOG_TS_RE.sub("", raw_line.strip(), count=1)

    m_n_tokens = SLOT_N_TOKENS_PATTERN.search(clean)
    if m_n_tokens:
        n_tokens = int(m_n_tokens.group(1))
        if n_tokens > 10:
            with CONTEXT_STATE_LOCK:
                CONTEXT_STATE["n_tokens"] = n_tokens
                if active_key:
                    CONTEXT_STATE["active_key"] = active_key

    m_n_ctx = SLOT_N_CTX_PATTERN.search(clean)
    if m_n_ctx:
        n_ctx = int(m_n_ctx.group(1))
        with CONTEXT_STATE_LOCK:
            CONTEXT_STATE["n_ctx_slot"] = n_ctx

    m_task_tokens = SLOT_TASK_TOKENS_PATTERN.search(clean)
    if m_task_tokens:
        task_tokens = int(m_task_tokens.group(1))
        with CONTEXT_STATE_LOCK:
            CONTEXT_STATE["task_n_tokens"] = task_tokens
        # Track start of prompt ingestion for live ingest TPS
        with INGEST_LIVE_STATE_LOCK:
            INGEST_LIVE_STATE["ingest_start_ts"] = time.time()
            INGEST_LIVE_STATE["ingest_start_tokens"] = task_tokens
            if active_key:
                INGEST_LIVE_STATE["active_key"] = active_key

    m_prompt_done = SLOT_PROMPT_DONE_PATTERN.search(clean)
    # When prompt processing finishes, compute ingest TPS
    if m_prompt_done:
        n_tokens = int(m_prompt_done.group(1))
        with CONTEXT_STATE_LOCK:
            CONTEXT_STATE["n_prompt_tokens"] = n_tokens
            if active_key:
                CONTEXT_STATE["active_key"] = active_key
        with INGEST_LIVE_STATE_LOCK:
            start_ts = INGEST_LIVE_STATE.get("ingest_start_ts")
            start_tokens = INGEST_LIVE_STATE.get("ingest_start_tokens")
            if start_ts is not None and start_tokens is not None and start_tokens > 0:
                dt = time.time() - start_ts
                if dt > 0.01 and n_tokens > start_tokens:
                    INGEST_LIVE_STATE["ingest_tps"] = (n_tokens - 0) / dt
                elif dt > 0.01:
                    INGEST_LIVE_STATE["ingest_tps"] = n_tokens / dt
            INGEST_LIVE_STATE["ingest_start_ts"] = None
            INGEST_LIVE_STATE["ingest_start_tokens"] = None

    for severity, category, pattern in _LOG_PATTERNS:
        if pattern.search(clean):
            _append_log_event(severity, category, clean)
            return


def _check_restart_count(container: str) -> None:
    code, out = run_command(["docker", "inspect", "--format={{.RestartCount}}", container])
    if code != 0:
        return
    try:
        count = int(out.strip())
    except ValueError:
        return
    now = time.time()
    recent = 0
    with LOG_LOCK:
        prev = LOG_STATE["last_restart_count"]
        LOG_STATE["last_restart_count"] = count
        if prev is None or count <= prev:
            return
        for _ in range(count - prev):
            LOG_STATE["restart_times"].append(now)
        cutoff = now - RESTART_LOOP_WINDOW_SEC
        LOG_STATE["restart_times"] = [t for t in LOG_STATE["restart_times"] if t >= cutoff]
        recent = len(LOG_STATE["restart_times"])
    if recent >= RESTART_LOOP_THRESHOLD and not _has_recent_event("restart_loop", 60):
        _append_log_event(
            "error", "restart_loop",
            f"Restart loop: {recent} restarts in {int(RESTART_LOOP_WINDOW_SEC)}s — check logs for root cause",
        )


def _ingest_log_tail(container: str, tail: int = 50, active_key: str | None = None) -> None:
    code, output = run_command(["docker", "logs", "--tail", str(tail), container])
    if code == 0:
        for line in output.splitlines():
            _process_log_line(line, active_key)


def run_command(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    combined = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, combined.strip()


def _run_log_watcher() -> None:
    """Daemon: streams docker logs and emits events into LOG_STATE.

    Only watches llama.cpp containers — vLLM metrics are collected via
    Prometheus /metrics endpoint instead of log parsing.
    """
    proc = None
    current_container: str | None = None
    _last_heartbeat_ts = 0.0
    _was_unhealthy = False
    while True:
        try:
            
            containers = list_llama_compose_containers()
            active_key = detect_active_model_key(containers)

            if not active_key:
                if proc is not None:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    proc = None
                    _append_log_event("info", "watcher", f"Detached from container {current_container} (model went away)")
                    current_container = None
                time.sleep(LOG_WATCHER_INTERVAL_SEC)
                continue

            target = find_model_server_container_name(active_key, containers)
            if not target:
                # For vLLM stacks, fall back to container name lookup by project label
                if _key_is_vllm(active_key):
                    target = find_vllm_server_container_name()

                if not target:
                    time.sleep(LOG_WATCHER_INTERVAL_SEC)
                    continue

            _check_restart_count(target)

            currently_unhealthy = False
            check_list = containers + list_vllm_compose_containers() if _key_is_vllm(active_key) else containers
            for c in check_list:
                if c["name"] == target and "(unhealthy)" in c.get("status", ""):
                    currently_unhealthy = True
            print(f"Log watcher checking target: {target}, current: {current_container}")
            if currently_unhealthy:
                if not _has_recent_event("health", 30):
                    _append_log_event(
                        "warning", "health",
                        f"Container {target} is unhealthy — health check failing",
                    )
            if _was_unhealthy and not currently_unhealthy:
                _append_log_event("info", "health", f"Container {target} health restored")
            _was_unhealthy = currently_unhealthy

            if target != current_container:
                if proc is not None:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    proc = None
                    if current_container:
                        _append_log_event("info", "watcher", f"Detached from container {current_container}")
                _ingest_log_tail(target, tail=50, active_key=active_key)
                proc = subprocess.Popen(
                    ["docker", "logs", "--follow", "--tail", "0", "--timestamps", target],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                _append_log_event("info", "watcher", f"Attached to container {target} (model: {active_key})")
                current_container = target
                with LOG_LOCK:
                    LOG_STATE["last_container"] = current_container
                    LOG_STATE["watcher_alive"] = True

            if proc is not None:
                deadline = time.monotonic() + LOG_WATCHER_INTERVAL_SEC
                count = 0
                while count < 200:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    rlist, _, _ = select.select([proc.stdout], [], [], min(remaining, 0.1))
                    if not rlist:
                        break
                    line = proc.stdout.readline()
                    if not line:
                        break
                    _process_log_line(line, active_key)
                    count += 1
                if proc.poll() is not None:
                    _append_log_event("warning", "watcher", f"Container {current_container} log stream ended (container stopped?)")
                    proc = None
                    current_container = None
            else:
                time.sleep(LOG_WATCHER_INTERVAL_SEC)

            now = time.time()
            if now - _last_heartbeat_ts >= HEARTBEAT_INTERVAL_SEC and active_key:
                hb = _build_heartbeat_summary(active_key, containers)
                if hb:
                    _append_log_event("info", "heartbeat", hb)
                    _last_heartbeat_ts = now

        except Exception as e:
            print(f"Log watcher error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(LOG_WATCHER_INTERVAL_SEC)


def list_llama_compose_containers() -> list[dict]:
    # Query for containers that have a working_dir label
    cmd = [
        "docker",
        "ps",
        "-a",
        "--filter",
        "label=com.docker.compose.project.working_dir",
        "--format",
        "{{.Label \"com.docker.compose.project.working_dir\"}}|{{.Label \"com.docker.compose.project.config_files\"}}|{{.Label \"com.docker.compose.service\"}}|{{.Status}}|{{.Names}}",
    ]
    code, output = run_command(cmd)
    if code != 0 or not output:
        return []

    _, compose_to_model = get_models()

    # We accept containers from both LLAMA_WORKING_DIR_LABEL and its stacks/ subfolder
    valid_dirs = {
        LLAMA_WORKING_DIR_LABEL,
        str(Path(LLAMA_WORKING_DIR_LABEL) / "stacks"),
    }
    valid_dirs_list = [d.rstrip("/") for d in valid_dirs]

    containers = []
    for line in output.splitlines():
        parts = line.split("|", 4)
        if len(parts) != 5:
            continue
        working_dir, config_files, service, status, name = parts

        # Check if this is a BeeLlama container (working_dir may be BEELLAMA_DIR or BEELLAMA_DIR/stacks)
        _beellama_valid_dirs = {BEELLAMA_WORKING_DIR_LABEL.rstrip("/"), str(BEELLAMA_DIR / "stacks").rstrip("/")}
        if BEELLAMA_DIR.is_dir() and working_dir.rstrip("/") in _beellama_valid_dirs:
            all_models, _ = get_models()
            for model_key, model in all_models.items():
                if model.get("beellama") and service.strip() == model["server_service"]:
                    containers.append({
                        "compose": model["compose"],
                        "service": service.strip(),
                        "status":  status.strip(),
                        "name":    name.strip(),
                        "beellama": True,
                    })
            continue

        # Check if this is a vLLM container
        if working_dir.rstrip("/") == VLLM_WORKING_DIR_LABEL.rstrip("/"):
            if service.strip().lower() == VLLM_SERVER_SERVICE.lower():
                containers.append(
                    {
                        "compose": "vllm",
                        "service": service.strip(),
                        "status": status.strip(),
                        "name": name.strip(),
                        "vllm": True,
                    }
                )
            continue

        # Only process containers from our expected project directories
        if working_dir.rstrip("/") not in valid_dirs_list:
            continue

        compose_name = ""
        for entry in config_files.split(","):
            full = entry.strip()
            prefix = LLAMA_WORKING_DIR_LABEL.rstrip("/") + "/"
            if full.startswith(prefix):
                rel = full[len(prefix):]
                if rel in compose_to_model:
                    compose_name = rel
                    break

        # If we couldn't match the compose file to a known model, it's not one of ours
        if not compose_name:
            continue

        containers.append(
            {
                "compose": compose_name,
                "service": service.strip(),
                "status": status.strip(),
                "name": name.strip(),
            }
        )
    return containers


def detect_active_model_key(containers: list[dict] | None = None) -> str | None:
    _, compose_to_model = get_models()
    records = containers if containers is not None else list_llama_compose_containers()
    for c in records:
        model_key = compose_to_model.get(c["compose"])
        if model_key and c["status"].startswith("Up"):
            return model_key
    # Fallback: check vLLM directly; resolve to specific stack key if possible
    vllm_records = list_vllm_compose_containers()
    for c in vllm_records:
        if c.get("vllm") and c["status"].startswith("Up"):
            return compose_to_model.get(c["compose"]) or "vllm"
    return None


def is_model_server_healthy(model_key: str, containers: list[dict] | None = None) -> bool:
    models, _ = get_models()
    model = models.get(model_key)
    if not model:
        return False
    records = containers if containers is not None else list_llama_compose_containers()
    target_compose = model["compose"]
    target_service = model["server_service"]

    # Special case: any vLLM stack
    if model.get("vllm"):
        return is_vllm_healthy(containers)

    for c in records:
        if c["compose"] != target_compose:
            continue
        if c["service"] != target_service:
            continue
        return "(healthy)" in c["status"]
    return False


def find_model_server_container_name(model_key: str, containers: list[dict]) -> str | None:
    models, _ = get_models()
    model = models.get(model_key)
    if not model:
        return None
    target_compose = model["compose"]
    target_service = model["server_service"]
    for c in containers:
        if c["compose"] == target_compose and c["service"] == target_service:
            return c["name"]
    return None

# ---------------------------------------------------------------------------
# vLLM container detection
# ---------------------------------------------------------------------------
def list_vllm_compose_containers() -> list[dict]:
    """Detect containers from the vLLM docker-compose project directory."""
    if not VLLM_DIR.is_dir():
        return []

    cmd = [
        "docker",
        "ps",
        "-a",
        "--filter",
        "label=com.docker.compose.project.working_dir",
        "--format",
        '{{.Label "com.docker.compose.project.working_dir"}}|{{.Label "com.docker.compose.project.config_files"}}|{{.Label "com.docker.compose.service"}}|{{.Status}}|{{.Names}}',
    ]
    code, output = run_command(cmd)
    if code != 0 or not output:
        return []

    valid_dirs = {VLLM_WORKING_DIR_LABEL, str(Path(VLLM_WORKING_DIR_LABEL) / "stacks")}
    valid_dirs_list = [d.rstrip("/") for d in valid_dirs]

    containers = []
    for line in output.splitlines():
        parts = line.split("|", 4)
        if len(parts) != 5:
            continue
        working_dir, config_files, service, status, name = parts
        if working_dir.rstrip("/") not in valid_dirs_list:
            continue
        # Check if this container belongs to the vllm server service
        if service.strip().lower() == VLLM_SERVER_SERVICE.lower():
            # Resolve which compose file is active from the Docker project label
            compose_basename = (
                Path(config_files.split(",")[0].strip()).name
                if config_files.strip()
                else "docker-compose.yml"
            )
            containers.append(
                {
                    "compose": compose_basename,
                    "config_files": config_files.strip(),
                    "service": service.strip(),
                    "status": status.strip(),
                    "name": name.strip(),
                    "vllm": True,
                }
            )
    return containers


def _key_is_vllm(key: str | None) -> bool:
    """True for any model key that belongs to a vLLM stack."""
    if not key:
        return False
    m, _ = get_models()
    return bool(m.get(key, {}).get("vllm"))


def detect_active_vllm_key(containers: list[dict] | None = None) -> bool:
    """Return True if a vLLM container is active and healthy."""
    records = list_vllm_compose_containers()
    for c in records:
        if c.get("vllm") and c["status"].startswith("Up"):
            return True
    return False


def find_vllm_server_container_name(containers: list[dict] | None = None) -> str | None:
    """Find the vLLM server container name."""
    records = list_vllm_compose_containers()
    for c in records:
        if c.get("vllm"):
            return c["name"]
    return None


def is_vllm_healthy(containers: list[dict] | None = None) -> bool:
    """Check if the vLLM container is healthy."""
    records = list_vllm_compose_containers()
    for c in records:
        if c.get("vllm"):
            return "(healthy)" in c.get("status", "")
    return False


# ---------------------------------------------------------------------------
# vLLM Prometheus metrics parsing
# ---------------------------------------------------------------------------
def _parse_prometheus_value(line: str) -> float | None:
    """Extract the numeric value from a Prometheus metric line."""
    line = line.strip()
    parts = line.split()
    if not parts:
        return None
    val = parts[-1]
    # Handle "inf", "-inf", etc.
    if val in ("+Inf", "-Inf", "Inf", "inf"):
        return None
    try:
        return float(val)
    except ValueError:
        return None


def _parse_prometheus_metrics(raw: str) -> dict:
    """Parse Prometheus /metrics output into a dict of metric_name -> value."""
    result = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        
        parts = line.split()
        if len(parts) < 2:
            continue
            
        # The last part is the value
        metric_full_name = parts[0]
        # The first part contains the metric name and potentially labels
        metric_name = metric_full_name.split('{')[0]
        
        val = _parse_prometheus_value(line)
        if val is not None:
            result[metric_name] = val
    return result


def parse_vllm_metrics(raw: str, active_key: str) -> dict:
    """Parse vLLM Prometheus metrics and compute TPS.

    Returns dict with generation_tokens, prompt_tokens, tps, ingest_tps, etc.
    """
    parsed = _parse_prometheus_metrics(raw)

    def get_metric(name: str) -> float:
        return float(parsed.get(f"vllm:{name}", parsed.get(name, 0)))

    gen_total = get_metric("generation_tokens_total")
    prompt_total = get_metric("prompt_tokens_total")
    iter_sum = get_metric("iteration_tokens_total_sum")
    iter_count = get_metric("iteration_tokens_total_count")

    # Queue time
    queue_sum = get_metric("request_queue_time_seconds_sum")
    queue_count = get_metric("request_queue_time_seconds_count")

    # Time to first token
    ttft_sum = get_metric("time_to_first_token_seconds_sum")
    ttft_count = get_metric("time_to_first_token_seconds_count")

    # Generation tokens per request
    gen_tokens_sum = get_metric("request_generation_tokens_sum")
    gen_tokens_count = get_metric("request_generation_tokens_count")

    # Prompt tokens per request
    prompt_tokens_sum = get_metric("request_prompt_tokens_sum")
    prompt_tokens_count = get_metric("request_prompt_tokens_count")

    # Cache usage
    gpu_cache_usage = get_metric("gpu_cache_usage_perc")

    # Compute live TPS (delta-based)
    now = time.time()
    with VLLM_TPS_STATE_LOCK:
        prev_container = VLLM_TPS_STATE.get("container")
        prev_sampled = VLLM_TPS_STATE.get("sampled_at", 0)
        prev_gen = VLLM_TPS_STATE.get("generation_tokens")

        live_tps = None
        if (prev_container and prev_gen is not None and
                prev_sampled > 0 and prev_gen <= gen_total):
            dt = now - prev_sampled
            if dt > 0:
                live_tps = (gen_total - prev_gen) / dt

        VLLM_TPS_STATE["container"] = active_key
        VLLM_TPS_STATE["sampled_at"] = now
        VLLM_TPS_STATE["generation_tokens"] = gen_total

    # Compute live ingest TPS (delta-based)
    with VLLM_INGEST_STATE_LOCK:
        prev_ingest_container = VLLM_INGEST_STATE.get("container")
        prev_ingest_sampled = VLLM_INGEST_STATE.get("sampled_at", 0)
        prev_ingest = VLLM_INGEST_STATE.get("prompt_tokens")

        live_ingest_tps = None
        if (prev_ingest_container and prev_ingest is not None and
                prev_ingest_sampled > 0 and prev_ingest <= prompt_total):
            dt = now - prev_ingest_sampled
            if dt > 0:
                live_ingest_tps = (prompt_total - prev_ingest) / dt

        VLLM_INGEST_STATE["container"] = active_key
        VLLM_INGEST_STATE["sampled_at"] = now
        VLLM_INGEST_STATE["prompt_tokens"] = prompt_total

    # Averages
    avg_gen_per_request = gen_tokens_sum / gen_tokens_count if gen_tokens_count > 0 else None
    avg_prompt_per_request = prompt_tokens_sum / prompt_tokens_count if prompt_tokens_count > 0 else None
    avg_queue_time = queue_sum / queue_count if queue_count > 0 else None
    avg_ttft = ttft_sum / ttft_count if ttft_count > 0 else None
    avg_iter_tokens = iter_sum / iter_count if iter_count > 0 else None

    return {
        "generation_tokens_total": gen_total,
        "prompt_tokens_total": prompt_total,
        "tps": live_tps,
        "ingest_tps": live_ingest_tps,
        "avg_gen_per_request": avg_gen_per_request,
        "avg_prompt_per_request": avg_prompt_per_request,
        "avg_queue_time_sec": avg_queue_time,
        "avg_ttft_sec": avg_ttft,
        "avg_iter_tokens": avg_iter_tokens,
        "total_requests_approx": max(iter_count, queue_count, gen_tokens_count),
        "source": "prometheus",
        "container": active_key,
        "updated_at": now_iso(),
        "gen_total": gen_total,
        "prompt_total": prompt_total,
        "gpu_cache_usage_perc": gpu_cache_usage,
        "completed_requests": gen_tokens_count,
    }


def parse_latest_completion(log_text: str) -> tuple[float | None, float | None, int | None, str | None, str | None, float | None, int | None]:
    latest_gen_tps = None
    latest_ingest_tps = None
    latest_prompt_tokens = None
    latest_task_id = None
    latest_eval_line = None
    latest_ingest_line = None
    latest_ts_f = None
    current_task_id = None
    for line in log_text.splitlines():
        ts_match = _LOG_TS_RE.search(line)
        line_ts_f = None
        if ts_match:
            try:
                line_ts_f = datetime.fromisoformat(ts_match.group(0).strip().replace("Z", "+00:00")).timestamp()
            except Exception:
                pass
            line = _LOG_TS_RE.sub("", line, count=1)

        task_match = TIMING_TASK_PATTERN.search(line)
        if task_match:
            current_task_id = int(task_match.group(1))

        ingest_match = INGEST_TPS_PATTERN.search(line)
        if ingest_match:
            n_tok = int(ingest_match.group(1))
            val = float(ingest_match.group(2))
            if n_tok > 10 and val < 20000:
                latest_ingest_tps = val
                latest_prompt_tokens = n_tok
                latest_ingest_line = line.strip()
                if line_ts_f: latest_ts_f = line_ts_f

        match = EVAL_TPS_PATTERN.search(line)
        if match:
            val = float(match.group(1))
            if val < 20000:
                latest_gen_tps = val
                latest_task_id = current_task_id
                latest_eval_line = line.strip()
                if line_ts_f: latest_ts_f = line_ts_f

    return latest_gen_tps, latest_ingest_tps, latest_task_id, latest_eval_line, latest_ingest_line, latest_ts_f, latest_prompt_tokens


def parse_live_tps_from_slots(slots_text: str, active_key: str, container: str) -> dict:
    try:
        slots = json.loads(slots_text)
    except Exception:
        return {
            "tokens_per_second": None,
            "n_ctx": None,
            "decoded_tokens": None,
            "state": "error",
            "detail": "Failed to parse /slots response",
        }

    if not isinstance(slots, list):
        return {
            "tokens_per_second": None,
            "n_ctx": None,
            "decoded_tokens": None,
            "state": "error",
            "detail": "Unexpected /slots format",
        }

    processing = False
    decoded_tokens = 0
    n_ctx = None
    for slot in slots:
        if not isinstance(slot, dict):
            continue
        slot_n_ctx = slot.get("n_ctx")
        if isinstance(slot_n_ctx, int) and (n_ctx is None or slot_n_ctx > n_ctx):
            n_ctx = slot_n_ctx
        if not slot.get("is_processing"):
            continue
        processing = True
        next_tokens = slot.get("next_token")
        if not isinstance(next_tokens, list):
            continue
        for token_state in next_tokens:
            if not isinstance(token_state, dict):
                continue
            n_decoded = token_state.get("n_decoded")
            if isinstance(n_decoded, int):
                decoded_tokens += n_decoded

    now = time.time()
    with LIVE_TPS_LOCK:
        if not processing:
            LIVE_TPS_STATE["active_key"] = active_key
            LIVE_TPS_STATE["container"] = container
            LIVE_TPS_STATE["sampled_at"] = now
            LIVE_TPS_STATE["decoded_tokens"] = None
            return {
                "tokens_per_second": 0.0,
                "n_ctx": n_ctx,
                "decoded_tokens": None,
                "state": "idle",
                "detail": "No active generation in /slots",
            }

        prev_valid = (
            LIVE_TPS_STATE["active_key"] == active_key
            and LIVE_TPS_STATE["container"] == container
            and LIVE_TPS_STATE["decoded_tokens"] is not None
            and LIVE_TPS_STATE["sampled_at"] > 0
        )
        live_tps = None
        if prev_valid:
            dt = now - float(LIVE_TPS_STATE["sampled_at"])
            prev_decoded = int(LIVE_TPS_STATE["decoded_tokens"])
            if dt > 0 and decoded_tokens >= prev_decoded:
                live_tps = (decoded_tokens - prev_decoded) / dt

        LIVE_TPS_STATE["active_key"] = active_key
        LIVE_TPS_STATE["container"] = container
        LIVE_TPS_STATE["sampled_at"] = now
        LIVE_TPS_STATE["decoded_tokens"] = decoded_tokens

    # Reset ingest state on idle (already handled above, also reset on model switch)
    with INGEST_LIVE_STATE_LOCK:
        if not processing:
            INGEST_LIVE_STATE["ingest_start_ts"] = None
            INGEST_LIVE_STATE["ingest_start_tokens"] = None

    if live_tps is None:
        return {
            "tokens_per_second": None,
            "n_ctx": n_ctx,
            "decoded_tokens": decoded_tokens,
            "state": "warming",
            "detail": "Collecting live TPS sample",
        }
    return {
        "tokens_per_second": live_tps,
        "n_ctx": n_ctx,
        "decoded_tokens": decoded_tokens,
        "state": "ok",
        "detail": "Live decode throughput from /slots",
    }


def fetch_live_tps(active_key: str | None, container: str) -> dict:
    port = _get_server_port(active_key) if active_key else 8080
    code, output = run_command(
        ["docker", "exec", container, "curl", "-fsS", f"http://127.0.0.1:{port}/slots"]
    )
    if code != 0:
        return {
            "tokens_per_second": None,
            "n_ctx": None,
            "decoded_tokens": None,
            "source": "slots",
            "updated_at": now_iso(),
            "container": container,
            "state": "error",
            "detail": "Failed to read /slots from llama.cpp",
        }

    parsed = parse_live_tps_from_slots(output, active_key, container)
    return {
        "tokens_per_second": parsed["tokens_per_second"],
        "n_ctx": parsed.get("n_ctx"),
        "decoded_tokens": parsed.get("decoded_tokens"),
        "source": "slots",
        "updated_at": now_iso(),
        "container": container,
        "state": parsed["state"],
        "detail": parsed["detail"],
    }


def build_live_throughput_status(active_key: str | None, containers: list[dict]) -> dict:
    if _key_is_vllm(active_key):
        return build_vllm_throughput_status()
    if not active_key:
        return {
            "tokens_per_second": None,
            "n_ctx": None,
            "decoded_tokens": None,
            "source": "slots",
            "updated_at": None,
            "container": None,
            "state": "unavailable",
            "detail": "No active model detected",
        }

    container = find_model_server_container_name(active_key, containers)
    if not container:
        return {
            "tokens_per_second": None,
            "n_ctx": None,
            "decoded_tokens": None,
            "source": "slots",
            "updated_at": None,
            "container": None,
            "state": "unavailable",
            "detail": "Active model server container not found",
        }

    now = time.time()
    with LIVE_TPS_CACHE_LOCK:
        cached = LIVE_TPS_CACHE["result"]
        if (
            cached is not None
            and LIVE_TPS_CACHE["active_key"] == active_key
            and LIVE_TPS_CACHE["container"] == container
            and (now - LIVE_TPS_CACHE["checked_at"]) < LIVE_TPS_CACHE_TTL_SEC
        ):
            return dict(cached)

    result = fetch_live_tps(active_key, container)
    with LIVE_TPS_CACHE_LOCK:
        LIVE_TPS_CACHE["active_key"] = active_key
        LIVE_TPS_CACHE["container"] = container
        LIVE_TPS_CACHE["checked_at"] = now
        LIVE_TPS_CACHE["result"] = dict(result)
    return result


def build_throughput_status(active_key: str | None, containers: list[dict]) -> dict:
    if _key_is_vllm(active_key):
        return build_vllm_throughput_status()
    if not active_key:
        return {
            "tokens_per_second": None,
            "ingest_tps": None,
            "completion_id": None,
            "completion_key": None,
            "source": "slots",
            "updated_at": None,
            "container": None,
            "state": "unavailable",
            "detail": "No active model detected",
        }

    container = find_model_server_container_name(active_key, containers)
    if not container:
        return {
            "tokens_per_second": None,
            "ingest_tps": None,
            "completion_id": None,
            "completion_key": None,
            "source": "slots",
            "updated_at": None,
            "container": None,
            "state": "unavailable",
            "detail": "Active model server container not found",
        }

    now = time.time()
    with THROUGHPUT_CACHE_LOCK:
        cached = THROUGHPUT_CACHE["result"]
        if (
            cached is not None
            and THROUGHPUT_CACHE["active_key"] == active_key
            and THROUGHPUT_CACHE["container"] == container
            and (now - THROUGHPUT_CACHE["checked_at"]) < THROUGHPUT_CACHE_TTL_SEC
        ):
            return dict(cached)

    code, output = run_command(["docker", "logs", "--timestamps", "--tail", str(THROUGHPUT_LOG_TAIL_LINES), container])
    if code != 0:
        result = {
            "tokens_per_second": None,
            "ingest_tps": None,
            "completion_id": None,
            "completion_key": None,
            "source": "logs",
            "updated_at": now_iso(),
            "container": container,
            "state": "error",
            "detail": "Failed to read llama.cpp logs",
        }
    else:
        gen_tps, ingest_tps, completion_id, completion_line, ingest_line, ts_f, prompt_tokens = parse_latest_completion(output)
        completion_key = (
            f"task:{completion_id}" if completion_id is not None else f"line:{completion_line}"
        )
        if gen_tps is None:
            result = {
                "tokens_per_second": None,
                "ingest_tps": None,
                "completion_id": None,
                "completion_key": None,
                "prompt_tokens": None,
                "ts_f": ts_f,
                "source": "logs",
                "updated_at": now_iso(),
                "container": container,
                "state": "no_data",
                "detail": "No completed generation timing in recent logs",
            }
        else:
            result = {
                "tokens_per_second": gen_tps,
                "ingest_tps": ingest_tps,
                "completion_id": completion_id,
                "completion_key": completion_key,
                "prompt_tokens": prompt_tokens,
                "ts_f": ts_f,
                "source": "logs",
                "updated_at": now_iso(),
                "container": container,
                "state": "ok",
                "detail": "Latest completed generation throughput",
            }

    with THROUGHPUT_CACHE_LOCK:
        THROUGHPUT_CACHE["active_key"] = active_key
        THROUGHPUT_CACHE["container"] = container
        THROUGHPUT_CACHE["checked_at"] = now
        THROUGHPUT_CACHE["result"] = dict(result)

    return result


def build_vllm_throughput_status() -> dict:
    """Fetch and parse vLLM Prometheus metrics, with caching."""
    container = find_vllm_server_container_name()
    if not container:
        return {
            "tokens_per_second": None,
            "ingest_tps": None,
            "source": "prometheus",
            "updated_at": None,
            "container": None,
            "state": "unavailable",
            "detail": "No vLLM container detected",
            "generation_tokens_total": 0,
            "prompt_tokens_total": 0,
            "avg_queue_time_sec": None,
            "avg_ttft_sec": None,
            "avg_iter_tokens": None,
            "avg_gen_per_request": None,
            "avg_prompt_per_request": None,
        }

    now = time.time()
    with VLLM_METRICS_CACHE_LOCK:
        cached = VLLM_METRICS_CACHE["raw"]
        if (
            cached is not None
            and VLLM_METRICS_CACHE["container"] == container
            and (now - VLLM_METRICS_CACHE["checked_at"]) < VLLM_METRICS_CACHE_TTL_SEC
        ):
            return _build_vllm_throughput_from_cache(container)

    code, output = run_command(
        ["docker", "exec", container, "curl", "-fsS", "--max-time", "5",
         "http://127.0.0.1:8080/metrics"]
    )
    if code != 0:
        result = {
            "tokens_per_second": None,
            "ingest_tps": None,
            "source": "prometheus",
            "updated_at": now_iso(),
            "container": container,
            "state": "error",
            "detail": "Failed to read vLLM metrics",
            "generation_tokens_total": 0,
            "prompt_tokens_total": 0,
            "avg_queue_time_sec": None,
            "avg_ttft_sec": None,
            "avg_iter_tokens": None,
            "avg_gen_per_request": None,
            "avg_prompt_per_request": None,
        }
    else:
        parsed = parse_vllm_metrics(output, container)
        result = {
            "tokens_per_second": parsed["tps"],
            "ingest_tps": parsed["ingest_tps"],
            "source": "prometheus",
            "updated_at": parsed["updated_at"],
            "container": container,
            "state": "ok" if parsed["tps"] is not None else "collecting",
            "detail": "vLLM Prometheus metrics",
            "generation_tokens_total": parsed["gen_total"],
            "prompt_tokens_total": parsed["prompt_total"],
            "avg_queue_time_sec": parsed["avg_queue_time_sec"],
            "avg_ttft_sec": parsed["avg_ttft_sec"],
            "avg_iter_tokens": parsed["avg_iter_tokens"],
            "avg_gen_per_request": parsed["avg_gen_per_request"],
            "avg_prompt_per_request": parsed["avg_prompt_per_request"],
            "total_requests_approx": parsed["total_requests_approx"],
            "completion_key": f"vllm-reqs:{parsed['completed_requests']}",
            "gpu_cache_usage_perc": parsed["gpu_cache_usage_perc"],
        }

    with VLLM_METRICS_CACHE_LOCK:
        VLLM_METRICS_CACHE["container"] = container
        VLLM_METRICS_CACHE["checked_at"] = now
        if code == 0:
            VLLM_METRICS_CACHE["raw"] = output
            VLLM_METRICS_CACHE["gpu_cache_usage_perc"] = parsed["gpu_cache_usage_perc"]
            VLLM_METRICS_CACHE["generation_tokens_total"] = parsed["gen_total"]
            VLLM_METRICS_CACHE["prompt_tokens_total"] = parsed["prompt_total"]
            VLLM_METRICS_CACHE["iteration_tokens_total_sum"] = parsed["avg_iter_tokens"] * parsed["total_requests_approx"] if parsed["avg_iter_tokens"] else 0
            # Note: I should probably just store the parsed dict or more fields if needed, 
            # but for now let's just make sure _build_vllm_throughput_from_cache works.
        else:
            VLLM_METRICS_CACHE["raw"] = None

    return result


def _build_vllm_throughput_from_cache(container: str) -> dict:
    """Build vLLM throughput result from cached metrics."""
    cache = VLLM_METRICS_CACHE
    gen_total = cache.get("generation_tokens_total", 0)
    prompt_total = cache.get("prompt_tokens_total", 0)
    iter_sum = cache.get("iteration_tokens_total_sum", 0)
    iter_count = cache.get("iteration_tokens_total_count", 0)
    queue_sum = cache.get("request_queue_time_seconds_sum", 0)
    queue_count = cache.get("request_queue_time_seconds_count", 0)
    ttft_sum = cache.get("time_to_first_token_seconds_sum", 0)
    ttft_count = cache.get("time_to_first_token_seconds_count", 0)
    gen_tokens_sum = cache.get("request_generation_tokens_sum", 0)
    gen_tokens_count = cache.get("request_generation_tokens_count", 0)
    prompt_tokens_sum = cache.get("request_prompt_tokens_sum", 0)
    prompt_tokens_count = cache.get("request_prompt_tokens_count", 0)

    avg_gen_per_request = gen_tokens_sum / gen_tokens_count if gen_tokens_count > 0 else None
    avg_prompt_per_request = prompt_tokens_sum / prompt_tokens_count if prompt_tokens_count > 0 else None
    avg_queue_time = queue_sum / queue_count if queue_count > 0 else None
    avg_ttft = ttft_sum / ttft_count if ttft_count > 0 else None
    avg_iter_tokens = iter_sum / iter_count if iter_count > 0 else None

    return {
        "tokens_per_second": None,  # TPS only available on live scrape
        "ingest_tps": None,
        "source": "prometheus",
        "updated_at": now_iso(),
        "container": container,
        "state": "cached",
        "detail": "Cached vLLM Prometheus metrics",
        "generation_tokens_total": gen_total,
        "prompt_tokens_total": prompt_total,
        "avg_queue_time_sec": avg_queue_time,
        "avg_ttft_sec": avg_ttft,
        "avg_iter_tokens": avg_iter_tokens,
        "avg_gen_per_request": avg_gen_per_request,
        "avg_prompt_per_request": avg_prompt_per_request,
        "total_requests_approx": max(iter_count, queue_count, gen_tokens_count),
        "completion_key": f"vllm-reqs:{gen_tokens_count}",
        "gpu_cache_usage_perc": cache.get("gpu_cache_usage_perc", 0),
    }


def build_model_stats(active_key: str | None, live: dict, completed: dict) -> dict:
    if not active_key:
        return {
            "model_key": None,
            "reset_at": None,
            "live_tps": 0.0,
            "live_state": "unavailable",
            "live_average_tps": None,
            "last_completed_tps": None,
            "last_ingest_tps": None,
            "last_live_ingest_tps": None,
            "best_ingest_tps": None,
            "last_completed_at": None,
            "last_completed_at_f": None,
            "completed_count": 0,
            "average_rate_tps": None,
        }

    now = now_iso()
    with MODEL_STATS_LOCK:
        if MODEL_STATS["active_key"] != active_key or MODEL_STATS["stats"] is None:
            with INGEST_LIVE_STATE_LOCK:
                INGEST_LIVE_STATE.update({"ingest_tps": 0.0, "ingest_start_ts": None, "ingest_start_tokens": None, "active_key": None})
            MODEL_STATS["active_key"] = active_key
            MODEL_STATS["stats"] = {
                "model_key": active_key,
                "reset_at": now,
                "live_tps": 0.0,
                "live_state": "idle",
                "live_sum": 0.0,
                "live_samples": 0,
                "live_average_tps": None,
                "last_completed_tps": None,
                "last_ingest_tps": None,
                "last_live_ingest_tps": None,
                "best_ingest_tps": None,
                "last_completed_at": None,
                "last_completed_at_f": None,
                "last_completed_completion_key": None,
                "completed_count": 0,
                "completed_sum_tps": 0.0,
                "average_rate_tps": None,
            }

        stats = MODEL_STATS["stats"]

        live_state = live.get("state", "unavailable")
        live_tps = live.get("tokens_per_second")
        if isinstance(live_tps, (int, float)):
            stats["live_tps"] = float(live_tps)
        else:
            stats["live_tps"] = 0.0 if live_state == "idle" else stats["live_tps"]
        stats["live_state"] = live_state

        if live_state == "ok" and isinstance(live_tps, (int, float)):
            stats["live_sum"] += float(live_tps)
            stats["live_samples"] += 1
            stats["live_average_tps"] = stats["live_sum"] / stats["live_samples"]

        # Check log-watcher-based live ingest from INGEST_LIVE_STATE
        with INGEST_LIVE_STATE_LOCK:
            ingest_live_tps = INGEST_LIVE_STATE.get("ingest_tps")
            ingest_active_key = INGEST_LIVE_STATE.get("active_key")
        if ingest_active_key == active_key and isinstance(ingest_live_tps, (int, float)) and ingest_live_tps > 0:
            stats["last_live_ingest_tps"] = float(ingest_live_tps)
            if stats["best_ingest_tps"] is None or float(ingest_live_tps) > stats["best_ingest_tps"]:
                stats["best_ingest_tps"] = float(ingest_live_tps)

        completion_tps = completed.get("tokens_per_second")
        completion_ingest = completed.get("ingest_tps")
        completion_key = completed.get("completion_key")
        should_add_completion = (
            completed.get("state") == "ok"
            and isinstance(completion_tps, (int, float))
            and completion_key is not None
            and completion_key != stats["last_completed_completion_key"]
        )
        if should_add_completion:
            stats["last_completed_tps"] = float(completion_tps)
            stats["last_completed_at"] = now
            stats["last_completed_at_f"] = completed.get("ts_f")
            stats["last_completed_completion_key"] = completion_key
            stats["completed_count"] += 1
            stats["completed_sum_tps"] += float(completion_tps)
            stats["average_rate_tps"] = stats["completed_sum_tps"] / stats["completed_count"]

            models, _ = get_models()
            model = models.get(active_key, {})
            label = model.get("label", active_key)
            quant = model.get("quant", "")

            n_prompt = completed.get("prompt_tokens")
            n_gen = completed.get("tokens_per_second")
            n_ingest = completed.get("ingest_tps")
            task_id = completed.get("completion_id")

            with CONTEXT_STATE_LOCK:
                n_ctx = CONTEXT_STATE.get("n_ctx_slot")
                n_past_from_log = CONTEXT_STATE.get("task_n_tokens") or CONTEXT_STATE.get("n_prompt_tokens") or CONTEXT_STATE.get("n_tokens")
            if n_ctx is None and isinstance(n_past_from_log, int) and n_past_from_log > 10:
                n_ctx = n_past_from_log

            parts = [label]
            if quant:
                parts.append(f"({quant})")
            parts.append(f"task:{task_id}" if task_id is not None else f"key:{completion_key}")
            if isinstance(n_prompt, int) and n_prompt > 0:
                parts.append(f"{n_prompt} prompt tok")
            if isinstance(n_gen, (int, float)):
                parts.append(f"{n_gen:.1f} gen T/S")
            if isinstance(n_ingest, (int, float)):
                parts.append(f"{n_ingest:.1f} ing T/S")
            if n_ctx is not None and n_past_from_log is not None and n_past_from_log > 0:
                parts.append(f"ctx {n_past_from_log}/{n_ctx}")
            parts.append(f"run #{stats['completed_count']}")

            _append_log_event("info", "completion", "Completed Run | " + "  ".join(parts))
        # Always update ingest from completions, even for dup keys
        if isinstance(completion_ingest, (int, float)) and completion_ingest > 0:
            stats["last_ingest_tps"] = float(completion_ingest)
            if stats["best_ingest_tps"] is None or float(completion_ingest) > stats["best_ingest_tps"]:
                stats["best_ingest_tps"] = float(completion_ingest)

        return {
            "model_key": stats["model_key"],
            "reset_at": stats["reset_at"],
            "live_tps": stats["live_tps"],
            "live_state": stats["live_state"],
            "live_average_tps": stats["live_average_tps"],
            "last_completed_tps": stats["last_completed_tps"],
            "last_ingest_tps": stats["last_ingest_tps"],
            "last_live_ingest_tps": stats["last_live_ingest_tps"],
            "best_ingest_tps": stats["best_ingest_tps"],
            "last_completed_at": stats["last_completed_at"],
            "last_completed_at_f": stats["last_completed_at_f"],
            "completed_count": stats["completed_count"],
            "average_rate_tps": stats["average_rate_tps"],
        }


def build_benchmark_prompt(estimated_tokens: int) -> str:
    # Keep request bodies compact to avoid CLI argument limits while still driving token load.
    repeats = max(32, estimated_tokens)
    return ("x " * repeats).strip()


def _pick_float(d: dict, keys: list[str]) -> float | None:
    for key in keys:
        v = d.get(key)
        if isinstance(v, (int, float)):
            vf = float(v)
            if vf > 0:
                return vf
    return None


def _get_server_port(active_key: str) -> int:
    """Return the internal (in-container) port the model server listens on."""
    models, _ = get_models()
    return models.get(active_key, {}).get("internal_port", 8080)


def run_single_benchmark(
    active_key: str,
    containers: list[dict],
    *,
    prompt_tokens: int,
    n_predict: int,
) -> tuple[dict | None, str | None]:
    container = find_model_server_container_name(active_key, containers)
    if not container:
        if _key_is_vllm(active_key):
            container = find_vllm_server_container_name()
        if not container:
            return None, "Active model server container not found"

    is_vllm = _key_is_vllm(active_key)

    port = _get_server_port(active_key)

    if is_vllm:
        # Fetch valid model ID from vLLM
        model_id = "qwen"
        code_model, out_model = run_command(["docker", "exec", container, "curl", "-s", f"http://127.0.0.1:{port}/v1/models"])
        if code_model == 0:
            try:
                models_data = json.loads(out_model)
                if "data" in models_data and len(models_data["data"]) > 0:
                    model_id = models_data["data"][0]["id"]
            except Exception:
                pass

        # vLLM OpenAI completions API
        payload = {
            "model": model_id,
            "prompt": build_benchmark_prompt(prompt_tokens),
            "max_tokens": n_predict,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True}
        }
        endpoint = f"http://127.0.0.1:{port}/v1/completions"
    else:
        # llama.cpp / BeeLlama completions API
        payload = {
            "prompt": build_benchmark_prompt(prompt_tokens),
            "n_predict": n_predict,
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": False,
            "cache_prompt": False,
        }
        endpoint = f"http://127.0.0.1:{port}/completion"

    cmd = [
        "docker",
        "exec",
        container,
        "curl",
        "-fsS",
        "--max-time",
        str(BENCHMARK_TIMEOUT_SEC),
        "-w", "\\n%{time_starttransfer}",
        "-X",
        "POST",
        endpoint,
        "-H",
        "Content-Type: application/json",
        "-d",
        json.dumps(payload),
    ]
    start_t = time.time()
    code, output = run_command(cmd)
    end_t = time.time()
    duration = end_t - start_t

    if code != 0:
        return None, f"Benchmark request failed (code {code}): {output[:200]}"

    # Extract TTFT from last line
    lines = output.strip().splitlines()
    ttft = 0.0
    if lines:
        try:
            ttft = float(lines[-1])
            output = "\n".join(lines[:-1]) # Remove TTFT line from JSON output
        except Exception:
            pass

    body = {}
    if not is_vllm:
        try:
            body = json.loads(output)
        except Exception:
            return None, f"Benchmark returned invalid JSON: {output[:100]}"

    if is_vllm:
        # vLLM streaming usage format (it's in the last data chunk)
        usage = None
        # body in this case is the list of lines if it was a stream? 
        # No, body = json.loads(output) likely failed because output is a stream.
        # Let's fix the JSON loading for vLLM streams.
        
        # Re-parse output for usage block
        usage_match = re.search(r'"usage":\s*({[^}]+})', output)
        if usage_match:
            try:
                usage = json.loads(usage_match.group(1))
            except Exception:
                pass
        
        if not usage:
            # Fallback to common choice-based usage if available
            return None, f"Benchmark usage missing from vLLM stream response. Output: {output[:100]}"
        
        prompt_n = usage.get("prompt_tokens", prompt_tokens)
        gen_n = usage.get("completion_tokens", n_predict)
        
        # Accurate speeds from TTFT
        # ttft here is time_starttransfer, which for a stream is when the first chunk (first token) arrived.
        # This is a very good measure of prefill time.
        prefill_tps = prompt_n / ttft if ttft > 0 else 0.0
        
        # Generation time is total wall duration minus prefill time
        gen_duration = duration - ttft
        gen_tps = gen_n / gen_duration if gen_duration > 0 else 0.0
        
        return (
            {
                "container": container,
                "prefill_tps": prefill_tps,
                "gen_tps": gen_tps,
                "prompt_tokens": prompt_n,
                "gen_tokens": gen_n,
                "is_vllm": True,
                "duration_sec": duration,
                "ttft_sec": ttft
            },
            None,
        )

    timings = body.get("timings")
    if not isinstance(timings, dict):
        return None, "Benchmark timings missing from llama.cpp response"

    prefill_tps = _pick_float(
        timings,
        ["prompt_per_second", "prompt_tokens_per_second", "prompt_tps"],
    )
    gen_tps = _pick_float(
        timings,
        ["predicted_per_second", "predicted_tokens_per_second", "eval_per_second", "eval_tps"],
    )

    if prefill_tps is None or gen_tps is None:
        return None, "Benchmark timings incomplete in llama.cpp response"

    return (
        {
            "container": container,
            "prefill_tps": prefill_tps,
            "gen_tps": gen_tps,
            "prompt_tokens": timings.get("prompt_n"),
            "gen_tokens": timings.get("predicted_n"),
            "prompt_ms": timings.get("prompt_ms"),
            "gen_ms": timings.get("predicted_ms"),
        },
        None,
    )


def run_benchmark_profile(
    active_key: str,
    containers: list[dict],
    profile: str,
    progress_cb=None,
) -> tuple[dict | None, str | None]:
    if profile == "balanced":
        result, error = run_single_benchmark(
            active_key,
            containers,
            prompt_tokens=BENCHMARK_PROMPT_TOKENS,
            n_predict=BENCHMARK_N_PREDICT,
        )
        if error:
            return None, error
        result["profile"] = "balanced"
        result["runs"] = [dict(result)]
        return result, None

    if profile == "full":
        specs = FULL_BENCHMARK_SPECS
        repeats = FULL_BENCHMARK_REPEATS
        runs = []
        for idx, spec in enumerate(specs, start=1):
            for rep in range(repeats):
                if callable(progress_cb):
                    progress_cb(f"full benchmark: pass {idx}/{len(specs)} ({spec['name']}, run {rep + 1}/{repeats})...")
                run, error = run_single_benchmark(
                    active_key,
                    containers,
                    prompt_tokens=spec["prompt_tokens"],
                    n_predict=spec["n_predict"],
                )
                if error:
                    return None, f"Full benchmark failed on {spec['name']} (run {rep + 1}/{repeats}): {error}"
                run["name"] = f"{spec['name']}#{rep + 1}"
                run["requested_prompt_tokens"] = spec["prompt_tokens"]
                run["requested_n_predict"] = spec["n_predict"]
                runs.append(run)

        prefill_avg = sum(float(r["prefill_tps"]) for r in runs) / len(runs)
        gen_avg = sum(float(r["gen_tps"]) for r in runs) / len(runs)
        return (
            {
                "profile": "full",
                "container": runs[0]["container"],
                "prefill_tps": prefill_avg,
                "gen_tps": gen_avg,
                "prompt_tokens": sum(int(r.get("prompt_tokens") or 0) for r in runs),
                "gen_tokens": sum(int(r.get("gen_tokens") or 0) for r in runs),
                "prompt_ms": sum(float(r.get("prompt_ms") or 0.0) for r in runs),
                "gen_ms": sum(float(r.get("gen_ms") or 0.0) for r in runs),
                "runs": runs,
            },
            None,
        )

    return None, "Unknown benchmark profile"


def start_switch(model_key: str) -> tuple[bool, str]:
    models, _ = get_models()
    model = models.get(model_key)
    if not model:
        return False, "Unknown model"

    # Handle vLLM switching (any stack with vllm=True)
    if model.get("vllm"):
        return start_vllm_switch(
            compose_file=model.get("compose_file") or str(VLLM_DIR / "docker-compose.yml"),
            model_key=model_key,
            label=model["label"],
        )

    # Handle BeeLlama switching
    if model.get("beellama"):
        return start_beellama_switch(model_key)

    current_containers = list_llama_compose_containers()
    current_active = detect_active_model_key(current_containers)
    if current_active == model_key and is_model_server_healthy(model_key, current_containers):
        return False, f"{model['label']} is already active and healthy"

    with STATE_LOCK:
        if STATE["switch_in_progress"]:
            return False, "A model switch is already in progress"
        STATE["switch_in_progress"] = True
        STATE["last_requested_model"] = model_key
        STATE["last_started_at"] = now_iso()
        STATE["last_message"] = f"Switching to {model['label']}..."
    _append_log_event("info", "switch", f"Switch requested: {model['label']}")

    def _worker() -> None:
        # Stop BeeLlama if running before starting a llama.cpp model
        if BEELLAMA_DIR.is_dir():
            _stop_beellama()
        compose_path = model["compose"]
        cmd = [
            "bash",
            "-lc",
            f"cd {shlex.quote(str(LLAMA_DIR))} && {shlex.quote(str(SWITCH_SCRIPT))} --stack {shlex.quote(compose_path)}",
        ]
        exit_code, output = run_command(cmd)
        if exit_code != 0:
            with STATE_LOCK:
                STATE["switch_in_progress"] = False
                STATE["last_completed_at"] = now_iso()
                STATE["last_exit_code"] = exit_code
                STATE["last_output"] = output[-4000:]
                STATE["last_message"] = f"Switch failed for {model['label']}"
            _append_log_event("error", "switch", f"Switch failed for {model['label']} (exit {exit_code})")
            return

        with STATE_LOCK:
            STATE["last_exit_code"] = exit_code
            STATE["last_output"] = output[-4000:]
            STATE["last_message"] = f"Loading {model['label']}..."

        # Poll until healthy, surfacing log events in last_message
        deadline = time.time() + SWITCH_READY_TIMEOUT_SEC
        while time.time() < deadline:
            local_containers = list_llama_compose_containers()
            if is_model_server_healthy(model_key, local_containers):
                with STATE_LOCK:
                    STATE["switch_in_progress"] = False
                    STATE["last_completed_at"] = now_iso()
                    STATE["last_message"] = f"Ready: {model['label']}"
                _append_log_event("info", "switch", f"Switch complete: {model['label']} is ready")
                return

            with LOG_LOCK:
                recent_events = list(LOG_STATE["events"])
            error_found = False
            for evt in reversed(recent_events):
                if evt["category"] in ("oom", "model_load", "config", "restart_loop"):
                    _set_last_message(f"Error loading {model['label']}: {evt['message'][:80]}")
                    error_found = True
                    break
            if not error_found:
                for evt in reversed(recent_events):
                    if evt["category"] == "loading":
                        pct = re.search(r"(\d+(?:\.\d+)?)\s*[%％]", evt["message"])
                        if pct:
                            _set_last_message(f"Loading {model['label']}: {pct.group(1)}%")
                        else:
                            _set_last_message(f"Loading {model['label']}...")
                        break

            time.sleep(SWITCH_POLL_SEC)

        with STATE_LOCK:
            STATE["switch_in_progress"] = False
            STATE["last_completed_at"] = now_iso()
            STATE["last_message"] = f"Load timeout for {model['label']}"
        _append_log_event("error", "switch", f"Switch timeout: {model['label']} did not become healthy within {SWITCH_READY_TIMEOUT_SEC}s")

    threading.Thread(target=_worker, daemon=True).start()
    return True, f"Switch request accepted: {model['label']}"


def start_vllm_switch(
    compose_file: str | None = None,
    model_key: str = "vllm",
    label: str = "vLLM",
) -> tuple[bool, str]:
    """Start/restart a vLLM stack (stop llama.cpp or a different vLLM stack first)."""
    if compose_file is None:
        compose_file = str(VLLM_DIR / "docker-compose.yml")

    current_containers = list_llama_compose_containers()
    current_active = detect_active_model_key(current_containers)
    vllm_containers = list_vllm_compose_containers()

    if current_active == model_key and is_vllm_healthy(vllm_containers):
        return False, f"{label} is already active and healthy"

    with STATE_LOCK:
        if STATE["switch_in_progress"]:
            return False, "A model switch is already in progress"
        STATE["switch_in_progress"] = True
        STATE["last_requested_model"] = model_key
        STATE["last_started_at"] = now_iso()
        STATE["last_message"] = f"Switching to {label}..."
    _append_log_event("info", "switch", f"Switch requested: {label}")

    def _vllm_worker() -> None:
        try:
            # Stop llama.cpp if it is active
            if current_active and not _key_is_vllm(current_active):
                _append_log_event("info", "switch", "Stopping current llama.cpp model...")
                cmd = [
                    "bash", "-lc",
                    f"cd {shlex.quote(str(LLAMA_DIR))} && {shlex.quote(str(SWITCH_SCRIPT))} --stop-all",
                ]
                run_command(cmd)
                time.sleep(2)

            # Stop a different vLLM stack if one is currently running
            elif _key_is_vllm(current_active) and current_active != model_key:
                _append_log_event("info", "switch", "Stopping current vLLM stack...")
                for c in vllm_containers:
                    cf = c.get("config_files", "")
                    if cf:
                        old_file = cf.split(",")[0].strip()
                        run_command(["bash", "-lc",
                            f"cd {shlex.quote(str(VLLM_DIR))} && "
                            f"docker compose -p vllm -f {shlex.quote(old_file)} down --remove-orphans"])
                        break
                time.sleep(2)

            # Start the requested vLLM stack
            compose_path = Path(compose_file)
            if not compose_path.exists():
                _append_log_event("error", "switch", f"vLLM compose file not found: {compose_file}")
                with STATE_LOCK:
                    STATE["switch_in_progress"] = False
                    STATE["last_completed_at"] = now_iso()
                    STATE["last_message"] = f"Compose file not found: {compose_path.name}"
                return

            _append_log_event("info", "switch", f"Starting {label}...")
            cmd = [
                "bash", "-lc",
                f"cd {shlex.quote(str(VLLM_DIR))} && "
                f"docker compose -p vllm -f {shlex.quote(str(compose_path))} up -d --remove-orphans",
            ]
            exit_code, output = run_command(cmd)

            if exit_code != 0:
                with STATE_LOCK:
                    STATE["switch_in_progress"] = False
                    STATE["last_completed_at"] = now_iso()
                    STATE["last_exit_code"] = exit_code
                    STATE["last_output"] = output[-4000:]
                    STATE["last_message"] = f"{label} start failed (exit {exit_code})"
                _append_log_event("error", "switch", f"{label} start failed (exit {exit_code})")
                return

            # Poll until healthy
            deadline = time.time() + SWITCH_READY_TIMEOUT_SEC
            while time.time() < deadline:
                time.sleep(SWITCH_POLL_SEC)
                local_vllm = list_vllm_compose_containers()
                if is_vllm_healthy(local_vllm):
                    with STATE_LOCK:
                        STATE["switch_in_progress"] = False
                        STATE["last_completed_at"] = now_iso()
                        STATE["last_message"] = f"Ready: {label}"
                    _append_log_event("info", "switch", f"Ready: {label}")
                    return

            with STATE_LOCK:
                STATE["switch_in_progress"] = False
                STATE["last_completed_at"] = now_iso()
                STATE["last_message"] = f"{label} load timeout"
            _append_log_event("error", "switch", f"{label} load timeout")

        except Exception as exc:
            _append_log_event("error", "switch", f"vLLM switch error: {exc}")
            with STATE_LOCK:
                STATE["switch_in_progress"] = False
                STATE["last_completed_at"] = now_iso()
                STATE["last_message"] = f"vLLM switch error: {exc}"

    threading.Thread(target=_vllm_worker, daemon=True).start()
    return True, f"Switch to {label} accepted"


def _get_beellama_compose_file(model: dict) -> Path:
    """Return the compose file for a beellama model.
    Stack files that contain 'services:' are used directly; stub-only files
    fall back to docker-compose.yml."""
    stack_rel = model.get("compose", "")
    if stack_rel.startswith("beellama-stacks/"):
        stack_file = BEELLAMA_STACKS_DIR / stack_rel[len("beellama-stacks/"):]
        try:
            if stack_file.exists() and "services:" in stack_file.read_text():
                return stack_file
        except Exception:
            pass
    return BEELLAMA_DIR / "docker-compose.yml"


def _stop_beellama() -> None:
    compose_file = BEELLAMA_DIR / "docker-compose.yml"
    if compose_file.exists():
        run_command(["bash", "-lc",
            f"docker compose -p beellamacpp -f {shlex.quote(str(compose_file))} down 2>/dev/null || true"])
    # Force-stop any remaining beellamacpp containers regardless of which compose file started them
    run_command(["bash", "-lc",
        "docker ps -q --filter label=com.docker.compose.project=beellamacpp "
        "| xargs -r docker stop 2>/dev/null || true; "
        "docker ps -aq --filter label=com.docker.compose.project=beellamacpp "
        "| xargs -r docker rm -f 2>/dev/null || true"])


def start_beellama_switch(model_key: str) -> tuple[bool, str]:
    """Start BeeLlama (stop llama.cpp first, then docker compose up -d)."""
    models, _ = get_models()
    model = models.get(model_key, {})
    current_containers = list_llama_compose_containers()
    current_active = detect_active_model_key(current_containers)
    if current_active == model_key and is_model_server_healthy(model_key, current_containers):
        return False, f"{model.get('label', 'BeeLlama')} is already active and healthy"

    with STATE_LOCK:
        if STATE["switch_in_progress"]:
            return False, "A model switch is already in progress"
        STATE["switch_in_progress"] = True
        STATE["last_requested_model"] = model_key
        STATE["last_started_at"] = now_iso()
        STATE["last_message"] = f"Switching to {model.get('label', 'BeeLlama')}..."
    _append_log_event("info", "switch", f"Switch requested: {model.get('label', 'BeeLlama')}")

    def _beellama_worker() -> None:
        try:
            compose_file = _get_beellama_compose_file(model)
            compose_file_q = shlex.quote(str(compose_file))

            # Pre-build while old service is still alive
            _append_log_event("info", "switch", "Building beellama image (cache speeds this up)...")
            build_code, build_output = run_command(["bash", "-lc",
                f"cd {shlex.quote(str(BEELLAMA_DIR))} && "
                f"docker compose -p beellamacpp --project-directory {shlex.quote(str(BEELLAMA_DIR))} -f {compose_file_q} build"])
            if build_code != 0:
                with STATE_LOCK:
                    STATE["switch_in_progress"] = False
                    STATE["last_completed_at"] = now_iso()
                    STATE["last_exit_code"] = build_code
                    STATE["last_output"] = build_output[-4000:]
                    STATE["last_message"] = f"BeeLlama build failed (exit {build_code})"
                _append_log_event("error", "switch", "BeeLlama build failed")
                return

            # Stop the current service now that the image is ready
            if current_active and not models.get(current_active, {}).get("beellama"):
                _append_log_event("info", "switch", "Stopping current llama.cpp model...")
                run_command(["bash", "-lc",
                    f"cd {shlex.quote(str(LLAMA_DIR))} && {shlex.quote(str(SWITCH_SCRIPT))} --stop-all"])
                time.sleep(2)
            elif current_active:
                _append_log_event("info", "switch", "Stopping current beellama model...")
                _stop_beellama()
                time.sleep(1)

            # Start new service — image already built, --no-build skips pull_policy: build
            _append_log_event("info", "switch", "Starting BeeLlama...")
            exit_code, output = run_command(["bash", "-lc",
                f"cd {shlex.quote(str(BEELLAMA_DIR))} && "
                f"docker compose -p beellamacpp --project-directory {shlex.quote(str(BEELLAMA_DIR))} -f {compose_file_q} up -d --no-build"])

            if exit_code != 0:
                with STATE_LOCK:
                    STATE["switch_in_progress"] = False
                    STATE["last_completed_at"] = now_iso()
                    STATE["last_exit_code"] = exit_code
                    STATE["last_output"] = output[-4000:]
                    STATE["last_message"] = f"BeeLlama start failed (exit {exit_code})"
                _append_log_event("error", "switch", "BeeLlama start failed")
                return

            with STATE_LOCK:
                STATE["last_exit_code"] = exit_code
                STATE["last_output"] = output[-4000:]
                STATE["last_message"] = f"Loading {model.get('label', 'BeeLlama')}..."

            deadline = time.time() + SWITCH_READY_TIMEOUT_SEC
            while time.time() < deadline:
                local_containers = list_llama_compose_containers()
                if is_model_server_healthy(model_key, local_containers):
                    with STATE_LOCK:
                        STATE["switch_in_progress"] = False
                        STATE["last_completed_at"] = now_iso()
                        STATE["last_message"] = f"Ready: {model.get('label', 'BeeLlama')}"
                    _append_log_event("info", "switch", f"Switch complete: {model.get('label', 'BeeLlama')} is ready")
                    return
                time.sleep(SWITCH_POLL_SEC)

            with STATE_LOCK:
                STATE["switch_in_progress"] = False
                STATE["last_completed_at"] = now_iso()
                STATE["last_message"] = f"Load timeout for {model.get('label', 'BeeLlama')}"
            _append_log_event("error", "switch", "BeeLlama load timeout")
        except Exception as e:
            with STATE_LOCK:
                STATE["switch_in_progress"] = False
                STATE["last_message"] = f"BeeLlama switch error: {e}"
            _append_log_event("error", "switch", f"BeeLlama switch error: {e}")

    threading.Thread(target=_beellama_worker, daemon=True).start()
    return True, f"Switch request accepted: {model.get('label', 'BeeLlama')}"


def get_gpu_stats() -> list[dict]:
    cmd = ["nvidia-smi", "--query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free,fan.speed,power.draw,power.limit,clocks.current.graphics,clocks.current.memory", "--format=csv,noheader,nounits"]
    code, output = run_command(cmd)
    if code != 0:
        return []

    gpus = []
    for line in output.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 13:
            try:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "temp": float(parts[2]),
                    "util": float(parts[3]),
                    "mem_util": float(parts[4]),
                    "mem_total": float(parts[5]),
                    "mem_used": float(parts[6]),
                    "mem_free": float(parts[7]),
                    "fan": float(parts[8]) if parts[8] != "[Not Supported]" else 0,
                    "power": float(parts[9]),
                    "power_limit": float(parts[10]),
                    "clock_gfx": float(parts[11]),
                    "clock_mem": float(parts[12]),
                })
            except ValueError:
                pass
    return gpus


def _get_proc_cpu_mem(pids: list[int]) -> dict[int, dict]:
    """Get CPU% and RAM (MB) for a list of PIDs via ps. Requires pid:host in Docker."""
    if not pids:
        return {}
    pid_args = ",".join(str(p) for p in pids)
    code, output = run_command(["ps", "-o", "pid=,pcpu=,rss=", "-p", pid_args])
    result: dict[int, dict] = {}
    if code == 0:
        for line in output.splitlines():
            parts = line.split()
            if len(parts) >= 3:
                try:
                    result[int(parts[0])] = {
                        "cpu_pct": float(parts[1]),
                        "ram_mb": int(parts[2]) / 1024.0,
                    }
                except ValueError:
                    pass
    return result


def _get_proc_gpu_pct(pids: set[int]) -> dict[int, float]:
    """Per-process GPU SM% via nvidia-smi pmon. Falls back to {} on error."""
    code, output = run_command(["nvidia-smi", "pmon", "-c", "1", "-s", "u"])
    result: dict[int, float] = {}
    if code != 0:
        return result
    for line in output.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 4:
            try:
                pid = int(parts[1])
                sm = parts[3]
                if pid in pids and sm not in ("-", "N/A"):
                    result[pid] = float(sm)
            except (ValueError, IndexError):
                pass
    return result


def get_gpu_processes() -> list[dict]:
    cmd = ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"]
    code, output = run_command(cmd)
    if code != 0:
        return []

    procs = []
    for line in output.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            try:
                procs.append({
                    "pid": int(parts[0]),
                    "name": parts[1].split("/")[-1],
                    "vram_mb": float(parts[2]),
                })
            except ValueError:
                pass

    if not procs:
        return []

    pids = [p["pid"] for p in procs]
    cpu_mem = _get_proc_cpu_mem(pids)
    gpu_pct = _get_proc_gpu_pct(set(pids))

    for p in procs:
        pid = p["pid"]
        cm = cpu_mem.get(pid, {})
        p["cpu_pct"] = cm.get("cpu_pct")
        p["ram_mb"] = cm.get("ram_mb")
        p["gpu_pct"] = gpu_pct.get(pid)

    return sorted(procs, key=lambda x: x["vram_mb"], reverse=True)


def build_status(handler: BaseHTTPRequestHandler | None = None) -> dict:
    models, _ = get_models()
    containers = list_llama_compose_containers()
    active_key = detect_active_model_key(containers)
    active_healthy = bool(active_key and is_model_server_healthy(active_key, containers))

    # Reconcile switch state with actual containers in case switch completed externally
    # (e.g. user ran switch-llm.sh manually) or state got stale.
    with STATE_LOCK:
        if STATE["switch_in_progress"]:
            requested_key = STATE.get("last_requested_model")
            requested = models.get(requested_key) if requested_key else None
            switched = False

            if requested:
                target_compose = requested["compose"]
                target_service = requested["server_service"]
                switched = any(
                    c["compose"] == target_compose
                    and c["service"] == target_service
                    and "(healthy)" in c["status"]
                    for c in containers
                )

            if switched:
                STATE["switch_in_progress"] = False
                STATE["last_completed_at"] = now_iso()
                if requested:
                    STATE["last_message"] = f"Switched to {requested['label']}"
            else:
                started_dt = parse_iso_utc(STATE.get("last_started_at"))
                if started_dt is not None:
                    age = (datetime.now(timezone.utc) - started_dt).total_seconds()
                    if age > SWITCH_STALE_SEC:
                        STATE["switch_in_progress"] = False
                        STATE["last_completed_at"] = now_iso()
                        STATE["last_message"] = (
                            f"Switch state auto-reset after {int(age)}s (target not healthy yet)"
                        )
                        _append_log_event(
                            "warning", "switch",
                            f"Switch state auto-reset after {int(age)}s for {requested_key}",
                        )

    with STATE_LOCK:
        snapshot = dict(STATE)

    active = None
    if active_key:
        m = models[active_key]
        active = {
            "key": active_key,
            "label": m["label"],
            "compose": m["compose"],
            "healthy": active_healthy,
             "thinking": m["thinking"],
             "vision": m["vision"],
             "family": m["family"],
            "ctx_size": m["ctx_size"],
            "quant": m["quant"],
            "params": m["params"],
        }
    throughput = build_throughput_status(active_key, containers)
    live_throughput = build_live_throughput_status(active_key, containers)
    model_stats = build_model_stats(active_key, live_throughput, throughput)

    context_info = {"n_ctx": None, "n_past": None}
    if _key_is_vllm(active_key) and throughput.get("gpu_cache_usage_perc") is not None:
        try:
            m = models.get(active_key) or {}
            ctx_size = int(m.get("ctx_size", 32768))
            context_info["n_ctx"] = ctx_size
            context_info["n_past"] = int(ctx_size * throughput["gpu_cache_usage_perc"])
        except (ValueError, KeyError):
            pass

    with CONTEXT_STATE_LOCK:
        ctx_state = dict(CONTEXT_STATE)
        # Reset stale context if model changed
        if ctx_state.get("active_key") and ctx_state["active_key"] != active_key:
            CONTEXT_STATE["n_tokens"] = None
            CONTEXT_STATE["n_prompt_tokens"] = None
            CONTEXT_STATE["task_n_tokens"] = None
            CONTEXT_STATE["last_good_n_past"] = None
            ctx_state.update({"n_tokens": None, "n_prompt_tokens": None, "task_n_tokens": None, "last_good_n_past": None})
    if live_throughput.get("n_ctx") is not None:
        context_info["n_ctx"] = live_throughput["n_ctx"]
    if live_throughput.get("n_past") is not None:
        context_info["n_past"] = live_throughput["n_past"]
    if context_info["n_ctx"] is None and ctx_state.get("n_ctx_slot") is not None:
        context_info["n_ctx"] = ctx_state["n_ctx_slot"]
    if context_info["n_ctx"] is None and active_key and active_key in models:
        ctx_str = models[active_key].get("ctx_size", "")
        try:
            context_info["n_ctx"] = int(ctx_str) if ctx_str else None
        except ValueError:
            pass
    if context_info["n_past"] is None:
        # Prefer task.n_tokens (arrives at start of processing, has full prompt size)
        # then n_prompt_tokens (from "prompt processing done")
        # then n_tokens (from slot release/progress)
        base_tokens = ctx_state.get("task_n_tokens") or ctx_state.get("n_prompt_tokens") or ctx_state.get("n_tokens")
        decoded = live_throughput.get("decoded_tokens") or 0
        if base_tokens is not None and ctx_state.get("active_key") == active_key:
            computed_n_past = base_tokens + decoded
            if computed_n_past > 10:
                context_info["n_past"] = computed_n_past
                with CONTEXT_STATE_LOCK:
                    CONTEXT_STATE["last_good_n_past"] = computed_n_past
    # If still None or tiny, use last known good value
    if (context_info["n_past"] is None or (isinstance(context_info["n_past"], int) and context_info["n_past"] < 10)):
        last_good = ctx_state.get("last_good_n_past")
        if last_good is not None and last_good >= 10:
            context_info["n_past"] = last_good

    # Safety valve: clear stale benchmark runs that never flipped state (e.g. worker crash).
    with BENCHMARK_LOCK:
        if BENCHMARK_STATE["in_progress"]:
            started_dt = parse_iso_utc(BENCHMARK_STATE.get("started_at"))
            if started_dt is not None:
                age = (datetime.now(timezone.utc) - started_dt).total_seconds()
                stale_sec = BENCHMARK_STALE_SEC
                if BENCHMARK_STATE.get("profile") == "full":
                    # 14 sequential runs can legitimately take much longer
                    stale_sec = max(stale_sec, FULL_BENCHMARK_RUNS * BENCHMARK_TIMEOUT_SEC + 60)
                if age > stale_sec:
                    BENCHMARK_STATE["in_progress"] = False
                    BENCHMARK_STATE["completed_at"] = now_iso()
                    BENCHMARK_STATE["last_error"] = (
                        f"Benchmark marked stale after {int(age)}s and auto-reset"
                    )
                    _append_log_event(
                        "warning", "benchmark",
                        f"Benchmark auto-reset after {int(age)}s (stale)",
                    )
                    BENCHMARK_STATE["history"].append(
                        {
                            "profile": BENCHMARK_STATE.get("profile", "balanced"),
                            "model_key": active_key,
                            "started_at": BENCHMARK_STATE.get("started_at"),
                            "completed_at": BENCHMARK_STATE.get("completed_at"),
                            "duration_sec": age,
                            "success": False,
                            "error": BENCHMARK_STATE["last_error"],
                        }
                    )
                    BENCHMARK_STATE["history"] = BENCHMARK_STATE["history"][-FULL_BENCHMARK_MAX_HISTORY:]

    with BENCHMARK_LOCK:
        benchmark = dict(BENCHMARK_STATE)

    status = {
        "active": active,
        "throughput": throughput,
        "live_throughput": live_throughput,
        "model_stats": model_stats,
        "context_info": context_info,
        "benchmark": benchmark,
        "switch_in_progress": snapshot["switch_in_progress"],
        "last_requested_model": snapshot["last_requested_model"],
        "last_started_at": snapshot["last_started_at"],
        "last_completed_at": snapshot["last_completed_at"],
        "last_exit_code": snapshot["last_exit_code"],
        "last_message": snapshot["last_message"],
        "gpu_stats": get_gpu_stats(),
        "gpu_procs": get_gpu_processes(),
        "models": [
            {
                "key": key,
                "label": m["label"],
                "compose": m["compose"],
                "server_service": m["server_service"],
                "thinking": m["thinking"],
                "vision": m["vision"],
                "family": m["family"],
                "ctx_size": m["ctx_size"],
                "quant": m["quant"],
                "params": m["params"],
            }
            for key, m in models.items()
        ],
        "vllm_dir": str(VLLM_DIR),
        "vllm_dir_exists": VLLM_DIR.is_dir(),
    }

    with LOG_LOCK:
        raw_events = list(LOG_STATE["events"])
        log_error_count = LOG_STATE["error_count"]
        log_watcher_ok = LOG_STATE["watcher_alive"]

    status["log_events"] = [{k: v for k, v in e.items() if k != "_ts_f"} for e in raw_events]
    status["log_error_count"] = log_error_count
    status["log_watcher_ok"] = log_watcher_ok

    if handler is not None:
        status["ttyd_url"] = "/ttyd/"
    return status


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LLM GPU Dashboard</title>
  <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect x='10' y='14' width='44' height='36' rx='9' fill='%230f1a2a' stroke='%235ec2ff' stroke-width='4'/%3E%3Ccircle cx='25' cy='31' r='5' fill='%235ec2ff'/%3E%3Ccircle cx='39' cy='31' r='5' fill='%235ec2ff'/%3E%3Crect x='24' y='41' width='16' height='3' rx='1.5' fill='%236fe58d'/%3E%3Crect x='30' y='6' width='4' height='8' rx='2' fill='%235ec2ff'/%3E%3Ccircle cx='32' cy='5' r='3' fill='%236fe58d'/%3E%3C/svg%3E" />
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {
      --bg: #030712;
      --sidebar-bg: #0a0f1a;
      --card: #111827;
      --card-active: #1e293b;
      --border: #1f2937;
      --border-strong: #374151;
      --text: #f9fafb;
      --text-muted: #6b7280;
      --text-dim: #9ca3af;
      --accent: #3b82f6;
      --success: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
      --sidebar-w: 260px;
      --header-h: 56px;
      --font-mono: 'JetBrains Mono', 'Fira Code', 'Roboto Mono', monospace;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    html, body { height: 100%; overflow: hidden; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      display: flex;
      flex-direction: column;
    }

    /* ── Header ── */
    .app-header {
      height: var(--header-h);
      min-height: var(--header-h);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 20px;
      background: var(--sidebar-bg);
      border-bottom: 1px solid var(--border);
      flex-shrink: 0;
      z-index: 10;
    }
    .brand { font-size: 16px; font-weight: 800; letter-spacing: 1px; color: var(--accent); }
    /* Mobile drawer toggle + backdrop (hidden on desktop; enabled in media query) */
    .menu-toggle { display: none; background: none; border: 1px solid var(--border-strong); color: var(--text); font-size: 18px; line-height: 1; width: 40px; height: 34px; border-radius: 8px; cursor: pointer; margin-right: 10px; flex-shrink: 0; align-items: center; justify-content: center; }
    .menu-toggle:active { background: rgba(255,255,255,0.08); }
    .sidebar-backdrop { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.6); z-index: 150; }
    .sidebar-backdrop.open { display: block; }
    .header-center { display: flex; align-items: center; gap: 10px; font-size: 13px; color: var(--text-dim); }
    .header-active-label { font-weight: 700; color: var(--text); }
    .status-badge {
      display: flex; align-items: center; gap: 7px;
      padding: 4px 12px;
      background: rgba(255,255,255,0.04);
      border: 1px solid var(--border-strong);
      border-radius: 999px;
      font-size: 12px; font-weight: 700; letter-spacing: 0.5px;
    }
    .status-dot { width: 8px; height: 8px; border-radius: 50%; }
    .dot-success { background: var(--success); box-shadow: 0 0 6px var(--success); }
    .dot-warning { background: var(--warning); animation: blink 1s infinite; }
    .dot-danger  { background: var(--danger); }
    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:.35} }

    /* ── App shell ── */
    .app-body {
      display: flex;
      flex: 1;
      overflow: hidden;
    }

    /* ── Sidebar ── */
    .sidebar {
      width: var(--sidebar-w);
      min-width: var(--sidebar-w);
      background: var(--sidebar-bg);
      border-right: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    .sidebar-nav {
      display: flex; flex-direction: column; gap: 2px;
      padding: 10px 10px 8px;
      border-bottom: 1px solid var(--border);
      flex-shrink: 0;
    }
    .nav-link {
      display: flex; align-items: center; gap: 10px;
      width: 100%; text-align: left;
      background: none; border: none; color: var(--text-dim);
      font: inherit; font-size: 13px; font-weight: 600;
      padding: 9px 10px; border-radius: 8px; cursor: pointer;
      transition: background 0.12s, color 0.12s;
    }
    .nav-link:hover { background: rgba(255,255,255,0.05); color: var(--text); }
    .nav-link:active { background: rgba(59,130,246,0.15); }
    .nav-ico { font-size: 15px; width: 18px; text-align: center; flex-shrink: 0; }
    .sidebar-scroll {
      flex: 1;
      overflow-y: auto;
      padding: 12px 0 8px;
    }
    .sidebar-scroll::-webkit-scrollbar { width: 4px; }
    .sidebar-scroll::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 2px; }

    .family-group { margin-bottom: 4px; }
    .family-label {
      font-size: 10px; font-weight: 700; letter-spacing: 1.5px;
      color: var(--text-muted);
      padding: 6px 16px 4px;
      text-transform: uppercase;
    }
    .model-row {
      display: flex;
      align-items: center;
      gap: 6px;
      padding: 8px 16px;
      cursor: pointer;
      transition: background 0.1s;
      border-left: 3px solid transparent;
      position: relative;
    }
    .model-row:hover { background: rgba(255,255,255,0.04); }
    .model-row.active {
      border-left-color: var(--success);
      background: rgba(16, 185, 129, 0.07);
    }
    .model-row.switching-target { opacity: 0.55; pointer-events: none; }
    .model-row-inner { flex: 1; min-width: 0; }
    .model-row-top { display: flex; align-items: center; gap: 6px; }
    .model-name { font-size: 13px; font-weight: 600; flex: 1; line-height: 1.3; }
    .model-quant { font-size: 10px; color: var(--text-muted); font-family: var(--font-mono); }
    .model-tags { display: flex; flex-wrap: wrap; gap: 3px; margin-top: 4px; }
    .tag-pill { font-size: 9px; font-weight: 700; padding: 1px 4px; border-radius: 3px; }
    .tag-ctx { background: rgba(59,130,246,0.12); color: #60a5fa; border: 1px solid rgba(59,130,246,0.25); }
    .tag-quant { background: rgba(168,85,247,0.12); color: #c084fc; border: 1px solid rgba(168,85,247,0.25); }
    .tag-params { background: rgba(20,184,166,0.12); color: #2dd4bf; border: 1px solid rgba(20,184,166,0.25); }
    .tag-cot { background: rgba(245,158,11,0.15); color: var(--warning); border: 1px solid rgba(245,158,11,0.3); }
    .tag-vision { background: rgba(59,130,246,0.15); color: var(--accent); border: 1px solid rgba(59,130,246,0.3); }
    .switching-spin { display: inline-block; width: 10px; height: 10px; border: 2px solid rgba(245,158,11,0.3); border-top-color: var(--warning); border-radius: 50%; animation: spin 0.6s linear infinite; flex-shrink: 0; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .badge-think {
      font-size: 9px; font-weight: 800; letter-spacing: 0.5px;
      background: rgba(245,158,11,0.15);
      color: var(--warning);
      border: 1px solid rgba(245,158,11,0.3);
      border-radius: 3px;
      padding: 1px 4px;
    }
    .sidebar-search-wrap { padding: 8px 12px 6px; flex-shrink: 0; border-bottom: 1px solid var(--border); }
    .sidebar-search {
      width: 100%; padding: 6px 10px; border-radius: 6px;
      background: rgba(255,255,255,0.06); border: 1px solid var(--border-strong);
      color: var(--text); font-size: 12px; outline: none;
    }
    .sidebar-search::placeholder { color: var(--text-muted); }
    .sidebar-search:focus { border-color: var(--accent); background: rgba(59,130,246,0.08); }

    .sidebar-footer {
      padding: 12px;
      border-top: 1px solid var(--border);
      flex-shrink: 0;
    }
    .btn-stop-all {
      width: 100%;
      padding: 8px;
      background: rgba(239,68,68,0.1);
      border: 1px solid rgba(239,68,68,0.3);
      color: var(--danger);
      border-radius: 6px;
      font-size: 12px; font-weight: 700; letter-spacing: 0.5px;
      cursor: pointer;
      transition: background 0.15s;
    }
    .btn-stop-all:hover { background: rgba(239,68,68,0.2); }

    .btn-restart {
      width: 100%;
      padding: 8px;
      background: rgba(245,158,11,0.1);
      border: 1px solid rgba(245,158,11,0.3);
      color: var(--warning);
      border-radius: 6px;
      font-size: 12px; font-weight: 700; letter-spacing: 0.5px;
      cursor: pointer;
      transition: background 0.15s;
      margin-bottom: 8px;
    }
    .btn-restart:hover { background: rgba(245,158,11,0.2); }
    .btn-restart:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }
    .btn-benchmark {
      width: 100%;
      padding: 8px;
      background: rgba(59,130,246,0.1);
      border: 1px solid rgba(59,130,246,0.35);
      color: var(--accent);
      border-radius: 6px;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.5px;
      cursor: pointer;
      transition: background 0.15s;
    }
    .btn-benchmark:hover { background: rgba(59,130,246,0.2); }
    .btn-benchmark:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }
    .btn-benchmark-full {
      width: 100%;
      padding: 8px;
      background: rgba(16,185,129,0.1);
      border: 1px solid rgba(16,185,129,0.35);
      color: var(--success);
      border-radius: 6px;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.5px;
      cursor: pointer;
      transition: background 0.15s;
    }
    .btn-benchmark-full:hover { background: rgba(16,185,129,0.2); }
    .btn-benchmark-full:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }
    .bench-btn-row {
      display: grid;
      grid-template-columns: 1fr;
      gap: 8px;
    }
    .bench-history {
      font-size: 11px;
      color: var(--text-dim);
      font-family: var(--font-mono);
      line-height: 1.4;
      display: flex;
      flex-direction: column;
      gap: 4px;
      max-height: 140px;
      overflow-y: auto;
      padding-right: 4px;
    }
    .bench-history-empty {
      color: var(--text-muted);
    }
    .bench-history-row {
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 4px 6px;
      background: rgba(255,255,255,0.02);
    }

    /* ── Main content ── */
    .main-content {
      flex: 1;
      overflow-y: auto;
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 18px;
    }
    .main-content::-webkit-scrollbar { width: 6px; }
    .main-content::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 3px; }

    /* ── Quick stats ── */
    .stats-row {
      display: grid;
      grid-template-columns: repeat(6, 1fr);
      gap: 14px;
    }
    .stat-card {
      background: var(--card);
      border: 1px solid var(--border-strong);
      border-radius: 12px;
      padding: 14px 16px;
    }
    .stat-label { font-size: 11px; font-weight: 700; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }
    .stat-value { font-size: 26px; font-weight: 700; font-family: var(--font-mono); line-height: 1; }
    .stat-sub { font-size: 11px; color: var(--text-muted); margin-top: 3px; }
    .stat-bar-bg { width: 100%; height: 4px; background: rgba(255,255,255,0.08); border-radius: 2px; overflow: hidden; margin-top: 8px; }
    .stat-bar-fill { height: 100%; background: var(--accent); transition: width 0.4s ease; border-radius: 2px; }

    /* ── Main grid ── */
    .content-grid {
      display: grid;
      grid-template-columns: 1fr 300px;
      gap: 18px;
      align-items: start;
    }
    .main-column {
      display: flex;
      flex-direction: column;
      gap: 18px;
    }
    .chart-panel {
      background: var(--card);
      border: 1px solid var(--border-strong);
      border-radius: 12px;
      padding: 18px;
      height: 300px;
    }
    .section-title {
      font-size: 11px; font-weight: 700; color: var(--text-muted);
      text-transform: uppercase; letter-spacing: 1px;
      margin-bottom: 14px;
    }
    .info-panel {
      background: var(--card);
      border: 1px solid var(--border-strong);
      border-radius: 12px;
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .info-item { display: flex; flex-direction: column; gap: 3px; }
    .info-pair { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .info-label { font-size: 10px; font-weight: 700; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; }
    .info-value { font-size: 13px; font-weight: 600; font-family: var(--font-mono); }
    .info-row { display: flex; justify-content: space-between; align-items: center; }
    .info-badges { display: flex; gap: 5px; flex-wrap: wrap; }
    .badge {
      font-size: 10px; font-weight: 700;
      padding: 2px 6px; border-radius: 4px;
    }
    .badge-ctx { background: rgba(59,130,246,0.15); color: var(--accent); border: 1px solid rgba(59,130,246,0.3); }
    .badge-think-panel { background: rgba(245,158,11,0.15); color: var(--warning); border: 1px solid rgba(245,158,11,0.3); }
    .badge-vision { background: rgba(59,130,246,0.15); color: var(--accent); border: 1px solid rgba(59,130,246,0.3); }

    /* ── Process table ── */
    .table-panel {
      background: var(--card);
      border: 1px solid var(--border-strong);
      border-radius: 12px;
      overflow: hidden;
    }
    .table-header { padding: 14px 18px 10px; border-bottom: 1px solid var(--border); }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th { text-align: left; padding: 10px 16px; background: rgba(255,255,255,0.02); color: var(--text-muted); font-weight: 700; font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; }
    td { padding: 10px 16px; border-bottom: 1px solid var(--border); font-family: var(--font-mono); }
    tr:last-child td { border-bottom: none; }

    /* ── Switch status ── */
    .switch-status-bar {
      font-size: 12px; color: var(--text-muted); font-family: var(--font-mono);
      padding: 6px 0; min-height: 22px;
    }

    /* ── Modal ── */
    .modal-overlay {
      position: fixed; inset: 0;
      background: rgba(0,0,0,0.85);
      backdrop-filter: blur(4px);
      display: flex; align-items: center; justify-content: center;
      z-index: 1000; opacity: 0; visibility: hidden; transition: 0.2s;
    }
    .modal-overlay.active { opacity: 1; visibility: visible; }
    .modal {
      background: var(--card);
      border: 1px solid var(--border-strong);
      border-radius: 16px;
      padding: 28px;
      max-width: 440px; width: 90%;
      text-align: center;
    }
    .modal h2 { font-size: 18px; margin-bottom: 10px; }
    .modal p { color: var(--text-dim); line-height: 1.6; font-size: 14px; }
    .modal-btn-row { display: flex; gap: 12px; justify-content: center; margin-top: 24px; }
    .btn { padding: 10px 22px; border-radius: 8px; font-weight: 700; cursor: pointer; border: none; font-size: 14px; }
    .btn-confirm { background: var(--accent); color: white; }
    .btn-cancel { background: transparent; border: 1px solid var(--border-strong); color: var(--text-dim); }

    @media (max-width: 1200px) {
      .stats-row { grid-template-columns: repeat(3, 1fr); }
      .content-grid { grid-template-columns: 1fr; }
      .main-column { order: 1; }
      .info-panel { order: 2; }
    }
    @media (max-width: 900px) {
      .stats-row { grid-template-columns: repeat(3, 1fr); }
    }
    /* ── Tablet / phone: sidebar becomes a slide-in drawer ── */
    @media (max-width: 768px) {
      .menu-toggle { display: flex; }
      .app-header { padding: 0 12px; }
      .brand { font-size: 14px; letter-spacing: 0.5px; }
      .header-center { display: none; }
      .sidebar {
        position: fixed; top: 0; left: 0; height: 100%;
        z-index: 200;
        width: 84vw; max-width: 320px; min-width: 0;
        transform: translateX(-100%);
        transition: transform 0.22s ease;
        box-shadow: 2px 0 24px rgba(0,0,0,0.5);
      }
      .sidebar.open { transform: translateX(0); }
      /* Position the Request Log (Activity Panel) below the GPU chart and details panel on mobile.
         Order: stats → switch status → GPU chart & details → Activity/Request Log. */
      .main-content { display: flex; flex-direction: column; }
      .main-content > * { flex: 0 0 auto; }   /* don't let flex shrink panels below content height */
      .stats-row { order: 0; }
      .switch-status-bar { order: 1; }
      .content-grid { order: 2; }
      .activity-panel { order: 3; margin-top: 0; }
    }
    /* ── Phone: compact layout ── */
    @media (max-width: 600px) {
      .stats-row { grid-template-columns: repeat(2, 1fr); gap: 8px; }
      .main-content { padding: 12px; }
      .stat-value { font-size: 20px; }
      .activity-head { padding: 0 10px; }
      .activity-tools { width: 100%; padding-bottom: 10px; }
      .reqlog-input[type=search] { flex: 1; min-width: 0; }
      /* Keep Time · Endpoint · Model · Status · (view); hide the rest — full data is in the tap-through detail modal */
      .reqlog-table th:nth-child(3), .reqlog-table td:nth-child(3),
      .reqlog-table th:nth-child(5), .reqlog-table td:nth-child(5),
      .reqlog-table th:nth-child(7), .reqlog-table td:nth-child(7),
      .reqlog-table th:nth-child(8), .reqlog-table td:nth-child(8),
      .reqlog-table th:nth-child(9), .reqlog-table td:nth-child(9) { display: none; }
      .reqlog-table tbody td { padding: 10px 10px; font-size: 12px; }
      .reqlog-view { opacity: 1; }
      /* Full-screen detail modal on phones */
      .reqdetail-modal { width: 100vw; height: 100dvh; max-width: 100vw; max-height: 100dvh; border-radius: 0; }
      .reqdetail-body { padding: 16px; }
      .rd-meta { gap: 8px 14px; }
      /* Confirmation modal sits above the drawer */
      .modal-overlay { z-index: 1000; }
    }

    /* ── Activity panel (Requests + Events, tabbed) ── */
    .activity-panel { background:var(--card); border:1px solid var(--border-strong); border-radius:12px; overflow:hidden; margin-top:16px; display:flex; flex-direction:column; }
    .activity-head { display:flex; align-items:center; justify-content:space-between; gap:12px; padding:0 14px; border-bottom:1px solid var(--border); flex-wrap:wrap; }
    .activity-tabs { display:flex; gap:2px; }
    .activity-tab { background:none; border:none; color:var(--text-muted); font:inherit; font-size:13px; font-weight:700; padding:14px 12px 12px; cursor:pointer; border-bottom:2px solid transparent; display:flex; align-items:center; gap:7px; transition:color .15s; }
    .activity-tab:hover { color:var(--text-dim); }
    .activity-tab.active { color:var(--text); border-bottom-color:var(--accent); }
    .tab-chip { background:rgba(255,255,255,0.08); color:var(--text-dim); font-size:10px; font-weight:700; padding:1px 7px; border-radius:999px; min-width:16px; text-align:center; }
    .tab-chip-err { background:rgba(239,68,68,0.2); color:var(--danger); display:none; }
    .activity-tools { display:flex; align-items:center; gap:8px; padding:7px 0; flex-wrap:wrap; }
    .reqlog-input { background:var(--bg); border:1px solid var(--border-strong); color:var(--text); border-radius:7px; padding:6px 9px; font-size:12px; font-family:inherit; }
    .reqlog-input[type=search] { min-width:220px; }
    .reqlog-input:focus { outline:none; border-color:var(--accent); background:rgba(59,130,246,0.06); }
    .reqlog-auto { color:var(--text-muted); font-size:11px; display:flex; align-items:center; gap:5px; cursor:pointer; user-select:none; }
    .reqlog-refresh { background:rgba(59,130,246,0.15); border:1px solid rgba(59,130,246,0.4); color:var(--accent); border-radius:7px; width:32px; height:30px; font-size:15px; font-weight:700; cursor:pointer; line-height:1; }
    .reqlog-refresh:hover { background:rgba(59,130,246,0.28); }
    .activity-body { min-height:120px; }

    /* Request Log table */
    .reqlog-scroll { max-height:60vh; overflow:auto; }
    .reqlog-scroll::-webkit-scrollbar { width:8px; height:8px; }
    .reqlog-scroll::-webkit-scrollbar-thumb { background:var(--border-strong); border-radius:4px; }
    .reqlog-table { width:100%; border-collapse:collapse; }
    .reqlog-table thead th { position:sticky; top:0; z-index:1; background:#0d1524; color:var(--text-muted); font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:.5px; text-align:left; padding:9px 12px; border-bottom:1px solid var(--border-strong); white-space:nowrap; }
    .reqlog-table tbody td { font-size:11.5px; padding:8px 12px; vertical-align:top; border-bottom:1px solid rgba(255,255,255,0.04); font-family:var(--font-mono); }
    .reqlog-row { cursor:pointer; transition:background .1s; }
    .reqlog-row:hover { background:rgba(59,130,246,0.07); }
    .reqlog-row:hover .reqlog-view { opacity:1; }
    .reqlog-time { color:var(--text-dim); white-space:nowrap; }
    .reqlog-ep { font-weight:800; font-size:10px; padding:2px 7px; border-radius:5px; white-space:nowrap; }
    .reqlog-ep-8080 { background:rgba(59,130,246,0.18); color:#93c5fd; }
    .reqlog-ep-28082 { background:rgba(16,185,129,0.18); color:#6ee7b7; }
    .reqlog-origin-ip { color:var(--text); }
    .reqlog-ua { color:var(--text-muted); font-size:10px; display:block; max-width:210px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .reqlog-model { color:var(--text-dim); }
    .reqlog-path { color:var(--text-dim); }
    .reqlog-snip { color:var(--text-muted); max-width:280px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .col-prompt { min-width:160px; }
    .status-pill { font-weight:800; font-size:10px; padding:2px 7px; border-radius:5px; }
    .status-ok { background:rgba(16,185,129,0.15); color:#6ee7b7; }
    .status-bad { background:rgba(239,68,68,0.15); color:#fca5a5; }
    .reqlog-view { opacity:.35; color:var(--accent); font-weight:700; font-size:11px; white-space:nowrap; }
    .reqlog-empty { text-align:center; color:var(--text-muted); padding:28px 16px; font-size:12.5px; }

    /* Events tab */
    .events-body { max-height:60vh; overflow-y:auto; padding:6px 0; font-family:var(--font-mono); font-size:12px; }
    .events-body::-webkit-scrollbar { width:8px; }
    .events-body::-webkit-scrollbar-thumb { background:var(--border-strong); border-radius:4px; }
    .event-row { display:flex; gap:10px; align-items:baseline; padding:5px 16px; border-bottom:1px solid rgba(255,255,255,0.03); }
    .event-row:last-child { border-bottom:none; }
    .event-ts { color:var(--text-muted); font-size:10px; white-space:nowrap; flex-shrink:0; }
    .event-sev { font-size:10px; font-weight:800; width:52px; flex-shrink:0; }
    .ev-error { color:var(--danger); } .ev-warning { color:var(--warning); } .ev-info { color:var(--success); }
    .event-msg { flex:1; color:var(--text-dim); word-break:break-word; }
    .events-empty-msg { text-align:center; color:var(--text-muted); padding:28px 16px; font-size:12.5px; }
    .hdr-err-badge { font-size:10px; font-weight:800; padding:2px 8px; border-radius:999px; background:rgba(239,68,68,0.2); color:var(--danger); border:1px solid rgba(239,68,68,0.4); display:none; margin-right:8px; }

    /* ── Request detail modal ── */
    .reqdetail-modal { max-width:940px; width:94vw; max-height:88vh; padding:0; text-align:left; display:flex; flex-direction:column; }
    .reqdetail-head { display:flex; align-items:center; justify-content:space-between; padding:16px 22px; border-bottom:1px solid var(--border); flex-shrink:0; }
    .reqdetail-head h2 { margin:0; font-size:16px; }
    .reqdetail-close { background:none; border:none; color:var(--text-muted); font-size:16px; cursor:pointer; padding:4px 8px; border-radius:6px; }
    .reqdetail-close:hover { background:rgba(255,255,255,0.06); color:var(--text); }
    .reqdetail-body { padding:18px 22px; overflow:auto; }
    .rd-meta { display:flex; flex-wrap:wrap; gap:8px 18px; padding-bottom:16px; margin-bottom:16px; border-bottom:1px solid var(--border); }
    .rd-meta .rd-m { display:flex; flex-direction:column; gap:2px; }
    .rd-meta .rd-k { font-size:9px; text-transform:uppercase; letter-spacing:.5px; color:var(--text-muted); font-weight:700; }
    .rd-meta .rd-v { font-size:12px; color:var(--text); font-family:var(--font-mono); }
    .rd-section-title { font-size:11px; font-weight:800; text-transform:uppercase; letter-spacing:.8px; color:var(--text-muted); margin:18px 0 10px; }
    .rd-section-title:first-child { margin-top:0; }
    .rd-params { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:12px; }
    .rd-param { background:var(--bg); border:1px solid var(--border); border-radius:6px; padding:3px 9px; font-size:11px; font-family:var(--font-mono); color:var(--text-dim); }
    .rd-param b { color:var(--text); font-weight:700; }
    .rd-msg { border:1px solid var(--border); border-radius:9px; margin-bottom:8px; overflow:hidden; }
    .rd-msg-role { font-size:10px; font-weight:800; text-transform:uppercase; letter-spacing:.5px; padding:6px 12px; }
    .rd-role-system { background:rgba(148,163,184,0.14); color:#cbd5e1; }
    .rd-role-user { background:rgba(59,130,246,0.16); color:#93c5fd; }
    .rd-role-assistant { background:rgba(16,185,129,0.14); color:#6ee7b7; }
    .rd-role-tool { background:rgba(245,158,11,0.14); color:#fcd34d; }
    .rd-msg-content { padding:10px 12px; font-size:12.5px; line-height:1.55; white-space:pre-wrap; word-break:break-word; color:var(--text); font-family:var(--font-mono); }
    .rd-reasoning { margin-top:8px; }
    .rd-reasoning summary { cursor:pointer; color:var(--warning); font-size:11px; font-weight:700; }
    .rd-reasoning .rd-msg-content { color:var(--text-muted); border-top:1px dashed var(--border); margin-top:6px; }
    .rd-img { font-size:11px; color:var(--text-muted); font-style:italic; padding:6px 12px; }
    .rd-raw { margin-top:14px; }
    .rd-raw summary { cursor:pointer; color:var(--text-muted); font-size:11px; font-weight:700; }
    .rd-raw pre { background:var(--bg); border:1px solid var(--border); border-radius:8px; padding:12px; overflow:auto; max-height:320px; font-size:11px; color:var(--text-dim); margin-top:8px; white-space:pre-wrap; word-break:break-word; }
    .rd-note { color:var(--text-muted); font-size:12px; font-style:italic; }
  </style>
</head>
<body>

  <!-- Header -->
  <div class="app-header">
    <button class="menu-toggle" id="menuToggle" onclick="toggleSidebar()" aria-label="Toggle model list">&#9776;</button>
    <div class="brand">GPU LLM DASHBOARD</div>
    <div class="header-center">
      <span id="headerActive" style="color: var(--text-muted)">Detecting...</span>
      &middot;
      <span id="headerUpdated" style="font-size: 12px; color: var(--text-muted)">--</span>
    </div>
    <span id="hdrErrBadge" class="hdr-err-badge"></span>
    <div id="statusBadge" class="status-badge">
      <div id="statusDot" class="status-dot"></div>
      <span id="statusText">DETECTING</span>
    </div>
  </div>

  <!-- App body -->
  <div class="app-body">

    <!-- Mobile drawer backdrop -->
    <div class="sidebar-backdrop" id="sidebarBackdrop" onclick="closeSidebar()"></div>

    <!-- Sidebar -->
    <div class="sidebar">
      <nav class="sidebar-nav">
        <button class="nav-link" onclick="goTo('overview')"><span class="nav-ico">&#128202;</span> Overview</button>
        <button class="nav-link" onclick="goTo('requests')"><span class="nav-ico">&#128220;</span> Request Log</button>
        <button class="nav-link" onclick="goTo('events')"><span class="nav-ico">&#128276;</span> Events</button>
      </nav>
      <div class="sidebar-search-wrap">
        <input type="search" class="sidebar-search" id="sidebarSearch" placeholder="Search models..." oninput="filterSidebar()" />
      </div>
      <div class="sidebar-scroll">
        <div id="sidebarModels"></div>
      </div>
      <div class="sidebar-footer">
        <button class="btn-restart" id="btnRestart" onclick="confirmRestart()">&#x21bb; Restart</button>
        <button class="btn-stop-all" onclick="confirmStopAll()">&#9632; Stop All</button>
      </div>
    </div>

    <!-- Main content -->
    <div class="main-content">

      <!-- Quick stats -->
      <div class="stats-row">
        <div class="stat-card">
          <div class="stat-label">Generation Speed</div>
          <div class="stat-value" id="val-tps">--</div>
          <div class="stat-sub">tokens / sec</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Ingest Speed</div>
          <div class="stat-value" id="val-ingest">--</div>
          <div class="stat-sub">tokens / sec</div>
        </div>
        <div class="stat-card" id="stat-ctx-card">
          <div class="stat-label" id="stat-ctx-label">Context</div>
          <div class="stat-value" id="val-ctx">-- / --</div>
          <div class="stat-bar-bg"><div id="bar-ctx" class="stat-bar-fill" style="width:0%"></div></div>
        </div>
        <div class="stat-card">
          <div class="stat-label">GPU Utilization</div>
          <div class="stat-value" id="val-util">--%</div>
          <div class="stat-bar-bg"><div id="bar-util" class="stat-bar-fill" style="width:0%"></div></div>
        </div>
        <div class="stat-card">
          <div class="stat-label">VRAM Usage</div>
          <div class="stat-value" id="val-vram">-- GB</div>
          <div class="stat-bar-bg"><div id="bar-vram" class="stat-bar-fill" style="width:0%"></div></div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Temperature</div>
          <div class="stat-value" id="val-temp">--&deg;C</div>
          <div class="stat-sub" id="val-fan">Fan: --%</div>
        </div>
      </div>

      <!-- Switch status bar -->
      <div class="switch-status-bar" id="switchStatus"></div>

      <!-- Chart + Info -->
      <div class="content-grid">
        <div class="main-column">
          <div class="chart-panel">
            <div class="section-title">Hardware Performance History (60s)</div>
            <div style="height: 230px;"><canvas id="historyChart"></canvas></div>
          </div>

          <!-- GPU Processes -->
          <div class="table-panel">
            <div class="table-header"><div class="section-title" style="margin:0">Active GPU Processes</div></div>
            <table>
              <thead><tr><th>PID</th><th>Application</th><th>VRAM</th><th>CPU%</th><th>RAM</th><th>GPU%</th><th>Status</th></tr></thead>
              <tbody id="procTable">
                <tr><td colspan="7" style="text-align:center; color:var(--text-muted)">Scanning processes...</td></tr>
              </tbody>
            </table>
          </div>
        </div>
        <div class="info-panel">
          <div class="section-title">Service Details</div>
          <div class="info-item">
            <div class="info-label">Active Model</div>
            <div class="info-value" id="info-model" style="color: var(--success); font-size: 12px; line-height: 1.4">NONE</div>
          </div>
          <div class="info-item">
            <div class="info-badges" id="info-badges"></div>
          </div>
          <div class="info-pair">
            <div class="info-item">
              <div class="info-label">Avg Rate</div>
              <div class="info-value" id="info-avg-rate">-- T/S</div>
            </div>
            <div class="info-item">
              <div class="info-label">Completed Runs</div>
              <div class="info-value" id="info-runs">0</div>
            </div>
          </div>
          <div class="info-pair">
            <div class="info-item">
              <div class="info-label">Clocks (GFX/MEM)</div>
              <div class="info-value" id="info-clocks" style="font-size:11px">-- / -- MHz</div>
            </div>
            <div class="info-item">
              <div class="info-label">Power Draw</div>
              <div class="info-value" id="info-power">--W / --W</div>
            </div>
          </div>
          <div class="info-item">
            <div class="bench-btn-row">
              <button class="btn-benchmark" id="btnBenchmark" onclick="triggerBenchmark('balanced')">RUN BENCHMARK</button>
              <button class="btn-benchmark-full" id="btnBenchmarkFull" onclick="triggerBenchmark('full')">RUN FULL BENCHMARK</button>
            </div>
          </div>
          <div class="info-pair">
            <div class="info-item">
              <div class="info-label">Bench Prefill</div>
              <div class="info-value" id="info-bench-prefill">-- T/S</div>
            </div>
            <div class="info-item">
              <div class="info-label">Bench Gen</div>
              <div class="info-value" id="info-bench-gen">-- T/S</div>
            </div>
          </div>
          <div class="info-item">
            <div class="info-label">Last Run</div>
            <div class="info-value" id="info-bench-last" style="font-size:11px">--</div>
          </div>
          <div class="info-item">
            <div class="info-label">Benchmark History (Last 10)</div>
            <div class="bench-history" id="info-bench-history">
              <div class="bench-history-empty">No benchmark runs yet.</div>
            </div>
          </div>
        </div>
      </div>

      <!-- Activity: Request Log + Events -->
      <div class="activity-panel" id="activityPanel">
        <div class="activity-head">
          <div class="activity-tabs">
            <button class="activity-tab active" id="tabbtn-reqlog" onclick="switchActivityTab('reqlog')">
              Requests<span class="tab-chip" id="reqlogCount"></span>
            </button>
            <button class="activity-tab" id="tabbtn-events" onclick="switchActivityTab('events')">
              Events<span class="tab-chip tab-chip-err" id="eventsBadge"></span>
            </button>
          </div>
          <div class="activity-tools" id="reqlogTools">
            <select id="reqlogEndpoint" class="reqlog-input" onchange="refreshRequestLog(true)">
              <option value="all">All endpoints</option>
              <option value="8080">8080 · llama.cpp</option>
              <option value="28082">28082 · minimal proxy</option>
            </select>
            <input type="search" id="reqlogSearch" class="reqlog-input" placeholder="Filter IP, UA, model, path…" oninput="onReqlogSearch()" />
            <label class="reqlog-auto" title="Auto-refresh"><input type="checkbox" id="reqlogAuto" checked onchange="refreshRequestLog(true)" /> Live</label>
            <button class="reqlog-refresh" title="Refresh now" onclick="refreshRequestLog(true)">&#8635;</button>
          </div>
        </div>

        <!-- Request Log tab -->
        <div class="activity-body" id="tab-reqlog">
          <div class="reqlog-scroll">
            <table class="reqlog-table">
              <thead><tr>
                <th>Time</th><th>Endpoint</th><th>Origin</th><th>Model</th>
                <th>Path</th><th>Status</th><th>Latency</th><th>Tokens</th><th class="col-prompt">Prompt</th><th></th>
              </tr></thead>
              <tbody id="reqlogBody">
                <tr><td colspan="10" class="reqlog-empty">Loading request log…</td></tr>
              </tbody>
            </table>
          </div>
        </div>

        <!-- Events tab -->
        <div class="activity-body" id="tab-events" hidden>
          <div class="events-body" id="eventsBody">
            <div class="events-empty-msg">No events yet.</div>
          </div>
        </div>
      </div>

    </div><!-- /.main-content -->
  </div><!-- /.app-body -->

  <!-- Confirmation modal -->
  <div id="confirmModal" class="modal-overlay">
    <div class="modal">
      <h2 id="modalTitle">Switch Model?</h2>
      <p id="modalText"></p>
      <div class="modal-btn-row">
        <button class="btn btn-cancel" onclick="closeModal()">CANCEL</button>
        <button class="btn btn-confirm" id="btnConfirmSwitch">PROCEED</button>
      </div>
    </div>
  </div>

  <!-- Request detail modal -->
  <div id="reqDetailModal" class="modal-overlay" onclick="if(event.target===this)closeRequestDetail()">
    <div class="modal reqdetail-modal">
      <div class="reqdetail-head">
        <h2 id="reqDetailTitle">Request detail</h2>
        <button class="reqdetail-close" onclick="closeRequestDetail()" title="Close">&#10005;</button>
      </div>
      <div class="reqdetail-body" id="reqDetailBody">Loading…</div>
    </div>
  </div>

  <script>
    let chart = null;
    let isRefreshing = false;
    let pendingAction = null;
    let lastUpdated = Date.now();
    let lastData = null;
    let eventsPanelOpen = true;
    let eventsLastCount = 0;
    let searchQuery = '';
    const history = { util: Array(60).fill(0), vram: Array(60).fill(0), tps: Array(60).fill(0) };

    // Family display order for sidebar grouping
    // NOTE: a model whose LLM_META family="..." is missing from FAMILY_ORDER is silently
    // dropped from the sidebar — the render loop below iterates this list, not the API
    // response. Adding a stack with a new family value REQUIRES adding it here too.
    const FAMILY_ORDER = ['vllm','muse-glimmer','gemma','gptoss','glm','mistral','nemotron','qwen','deepseek','other'];
    const FAMILY_LABELS = {
      vllm: 'vLLM', qwen: 'QWEN', nemotron: 'NEMOTRON', gptoss: 'GPT-OSS',
      glm: 'GLM', mistral: 'MISTRAL', gemma: 'GEMMA', deepseek: 'DEEPSEEK',
      'muse-glimmer': 'MUSE GLIMMER', other: 'OTHER'
    };
    // Persisted collapse state: key = family, value = true if collapsed
    let FAMILY_COLLAPSED = {};
    try { FAMILY_COLLAPSED = JSON.parse(localStorage.getItem('familyCollapsed') || '{}'); } catch(e) {}

    function toggleFamily(fam) {
      FAMILY_COLLAPSED[fam] = !FAMILY_COLLAPSED[fam];
      try { localStorage.setItem('familyCollapsed', JSON.stringify(FAMILY_COLLAPSED)); } catch(e) {}
      if (lastData) buildSidebar(lastData);
    }

    function initChart() {
      const ctx = document.getElementById('historyChart').getContext('2d');
      chart = new Chart(ctx, {
        type: 'line',
        data: {
          labels: Array(60).fill(''),
          datasets: [
            { label: 'GPU Util %', data: history.util, borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.08)', fill: true, tension: 0.4, pointRadius: 0, borderWidth: 2, yAxisID: 'y' },
            { label: 'VRAM %',     data: history.vram, borderColor: '#10b981', backgroundColor: 'rgba(16,185,129,0.08)', fill: true, tension: 0.4, pointRadius: 0, borderWidth: 2, yAxisID: 'y' },
            { label: 'Gen Speed (T/S)', data: history.tps, borderColor: '#f59e0b', backgroundColor: 'rgba(245,158,11,0.08)', fill: true, tension: 0.4, pointRadius: 0, borderWidth: 2, yAxisID: 'y1' }
          ]
        },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          scales: {
            y: {
              beginAtZero: true, max: 100,
              grid: { color: 'rgba(255,255,255,0.04)' },
              ticks: { color: '#6b7280', font: { size: 11 } }
            },
            y1: {
              beginAtZero: true,
              position: 'right',
              grid: { drawOnChartArea: false },
              ticks: { color: '#f59e0b', font: { size: 11 } },
              title: { display: true, text: 'Tokens/Sec', color: '#f59e0b' }
            },
            x: { display: false }
          },
          plugins: { legend: { labels: { color: '#9ca3af', boxWidth: 10, font: { size: 11 } } } }
        }
      });
    }

    function fmtCtx(val) {
      const n = parseInt(val);
      if (!n) return String(val);
      if (n >= 1024) return Math.round(n/1024) + 'k ctx';
      return n + ' ctx';
    }

    function buildSidebar(data) {
      const q = searchQuery.toLowerCase().trim();
      const groups = {};
      for (const m of (data.models || [])) {
        if (q && !m.label.toLowerCase().includes(q) &&
            !(m.family||'').toLowerCase().includes(q) &&
            !(m.quant||'').toLowerCase().includes(q)) continue;
        const f = m.family || 'other';
        (groups[f] = groups[f] || []).push(m);
      }
      const switching = data.switch_in_progress;
      const switchTarget = data.last_requested_model;
      let html = '';
      // Any family present in the data but absent from FAMILY_ORDER is appended here,
      // so a new LLM_META family="..." can never silently vanish from the sidebar.
      const renderOrder = FAMILY_ORDER.concat(
        Object.keys(groups).filter(f => !FAMILY_ORDER.includes(f)).sort()
      );
      for (const fam of renderOrder) {
        if (!groups[fam]) continue;
        const collapsed = FAMILY_COLLAPSED[fam];
        const chevron = collapsed ? '&#9658;' : '&#9660;';
        html += `<div class="family-group">
          <div class="family-label" onclick="toggleFamily('${fam}')" style="cursor:pointer;user-select:none;">${chevron}&nbsp;${FAMILY_LABELS[fam] || fam.toUpperCase()}</div>`;
        if (!collapsed) {
        for (const m of groups[fam]) {
          const active = data.active?.key === m.key;
          const isTarget = switching && switchTarget === m.key;
          const cls = 'model-row' + (active ? ' active' : '') + (isTarget ? ' switching-target' : '');
          const spin = isTarget ? '<span class="switching-spin"></span>' : '';
          let tags = '';
          if (m.params) tags += `<span class="tag-pill tag-params">${escHtml(m.params)}</span>`;
          if (m.ctx_size) tags += `<span class="tag-pill tag-ctx">${fmtCtx(m.ctx_size)}</span>`;
          if (m.quant) tags += `<span class="tag-pill tag-quant">${escHtml(m.quant)}</span>`;
          if (m.thinking) tags += `<span class="tag-pill tag-cot">COT</span>`;
          if (m.vision) tags += `<span class="tag-pill tag-vision">VISION</span>`;
          html += `<div class="${cls}" onclick="confirmSwitch('${m.key}','${escHtml(m.label)}')">
            <div class="model-row-inner">
              <div class="model-row-top">${spin}<span class="model-name">${escHtml(m.label)}</span></div>
              ${tags ? `<div class="model-tags">${tags}</div>` : ''}
            </div>
          </div>`;
        }
        }
        html += '</div>';
      }
      document.getElementById('sidebarModels').innerHTML = html;
    }

    function filterSidebar() {
      searchQuery = document.getElementById('sidebarSearch').value;
      if (lastData) buildSidebar(lastData);
    }

    function escHtml(s) {
      return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
    }

    function fmtLocalTs(s) {
      if (!s) return '--';
      const d = new Date(s);
      if (Number.isNaN(d.getTime())) return s;
      return d.toLocaleString();
    }

    function setHeaderStatus(data) {
      const dot = document.getElementById('statusDot');
      const txt = document.getElementById('statusText');
      const active = document.getElementById('headerActive');
      const upd = document.getElementById('headerUpdated');
      const btnRestart = document.getElementById('btnRestart');

      const secs = Math.round((Date.now() - lastUpdated) / 1000);
      upd.textContent = `Updated ${secs}s ago`;

      if (btnRestart) {
        btnRestart.disabled = data.switch_in_progress || !data.active?.key;
      }

      if (data.switch_in_progress) {
        dot.className = 'status-dot dot-warning';
        txt.textContent = 'SWITCHING';
        active.textContent = 'Switching...';
      } else if (data.active?.healthy) {
        dot.className = 'status-dot dot-success';
        txt.textContent = 'HEALTHY';
        active.innerHTML = `<span style="color:var(--text-dim)">Active:</span> <span style="color:var(--text); font-weight:700">${escHtml(data.active.label)}</span>`;
      } else if (data.active) {
        dot.className = 'status-dot dot-warning';
        txt.textContent = 'STARTING';
        active.innerHTML = `<span style="color:var(--text-dim)">Loading:</span> <span style="color:var(--text)">${escHtml(data.active.label)}</span>`;
      } else {
        dot.className = 'status-dot dot-danger';
        txt.textContent = 'OFFLINE';
        active.textContent = 'No model active';
      }
    }

    function updateDashboard(data) {
      const gpu = data.gpu_stats?.[0] || {};
      const bench = data.benchmark || {};

      // Unified metrics from standard model_stats
      const stats = data.model_stats || {};
      const live = data.live_throughput || {};
      const ctxInfo = data.context_info || {};

      // Generation Speed
      const liveTps = stats.live_tps || 0;
      const lastTps = stats.last_completed_tps || stats.live_average_tps || 0;
      const genDisplay = liveTps > 0 ? liveTps.toFixed(1) : (lastTps > 0 ? lastTps.toFixed(1) : '0.0');

      // Ingest Speed
      let ingestVal = '0.0';
      const lastIngest = stats.last_ingest_tps;
      const liveIngest = stats.last_live_ingest_tps;
      const bestIngest = stats.best_ingest_tps;
      const isActivelyIngesting = live?.state !== 'idle' && liveIngest != null && liveIngest > 0;
      if (lastIngest != null && lastIngest > 0) {
        ingestVal = lastIngest.toFixed(1);
      } else if (isActivelyIngesting) {
        ingestVal = liveIngest.toFixed(1);
      } else if (bestIngest != null && bestIngest > 0) {
        ingestVal = bestIngest.toFixed(1);
      } else if (liveIngest != null && liveIngest > 0) {
        ingestVal = liveIngest.toFixed(1);
      }

      document.getElementById('val-tps').textContent = genDisplay;
      document.getElementById('val-ingest').textContent = ingestVal;
      document.getElementById('val-util').textContent = gpu.util != null ? `${gpu.util}%` : '--%';
      document.getElementById('bar-util').style.width = `${gpu.util || 0}%`;
      const vramGB = ((gpu.mem_used || 0) / 1024).toFixed(1);
      document.getElementById('val-vram').textContent = `${vramGB} GB`;
      const vramPct = gpu.mem_total > 0 ? (gpu.mem_used / gpu.mem_total) * 100 : 0;
      document.getElementById('bar-vram').style.width = `${vramPct.toFixed(1)}%`;
      document.getElementById('val-temp').textContent = gpu.temp != null ? `${gpu.temp}°C` : '--°C';
      document.getElementById('val-fan').textContent = `Fan: ${gpu.fan || 0}%`;

      // Context card
      const nCtx = ctxInfo.n_ctx;
      const nPast = ctxInfo.n_past;
      const ctxEl = document.getElementById('val-ctx');
      const ctxBar = document.getElementById('bar-ctx');
      if (nCtx != null) {
        const ctxMax = nCtx >= 1024 ? `${Math.round(nCtx/1024)}k` : `${nCtx}`;
        if (nPast != null && nPast > 0) {
          const ctxUsed = nPast >= 1024 ? `${(nPast/1024).toFixed(1)}k` : `${nPast}`;
          ctxEl.textContent = `${ctxUsed} / ${ctxMax}`;
          const pct = Math.min(100, (nPast / nCtx) * 100);
          ctxBar.style.width = `${pct.toFixed(1)}%`;
          if (pct > 90) ctxBar.style.background = 'var(--danger)';
          else if (pct > 70) ctxBar.style.background = 'var(--warning)';
          else ctxBar.style.background = 'var(--accent)';
        } else {
          ctxEl.textContent = `0 / ${ctxMax}`;
          ctxBar.style.width = '0%';
          ctxBar.style.background = 'var(--accent)';
        }
      } else {
        ctxEl.textContent = '-- / --';
        ctxBar.style.width = '0%';
      }

      // History chart
      const tpsVal = stats.live_tps || 0;
      history.util.push(gpu.util || 0); history.util.shift();
      history.vram.push(vramPct); history.vram.shift();
      history.tps.push(tpsVal); history.tps.shift();
      chart.update();

      // Service details
      const infoModel = document.getElementById('info-model');
      const infoBadges = document.getElementById('info-badges');
      const infoAvgRate = document.getElementById('info-avg-rate');
      const infoRuns = document.getElementById('info-runs');

      if (data.active) {
        infoModel.textContent = data.active.label;
        let badges = '';
        if (data.active.params) badges += `<span class="badge badge-ctx">${escHtml(data.active.params)}</span>`;
        if (data.active.ctx_size) {
          const ctx = parseInt(data.active.ctx_size);
          const ctxLabel = ctx >= 1024 ? `${Math.round(ctx/1024)}k ctx` : `${ctx} ctx`;
          badges += `<span class="badge badge-ctx">${ctxLabel}</span>`;
        }
        if (data.active.thinking) badges += `<span class="badge badge-think-panel">THINKING</span>`;
        if (data.active.vision) badges += `<span class="badge badge-vision">VISION</span>`;
        infoBadges.innerHTML = badges;
      } else {
        infoModel.textContent = 'NONE';
        infoBadges.innerHTML = '';
      }

      infoAvgRate.textContent = stats.average_rate_tps ? `${stats.average_rate_tps.toFixed(2)} T/S` : '-- T/S';
      infoRuns.textContent = stats.completed_count || 0;

      const clocksEl = document.getElementById('info-clocks');
      const powerEl = document.getElementById('info-power');
      clocksEl.textContent = `${gpu.clock_gfx || '--'} / ${gpu.clock_mem || '--'} MHz`;
      powerEl.textContent = `${gpu.power ? gpu.power.toFixed(0) : '--'}W / ${gpu.power_limit ? gpu.power_limit.toFixed(0) : '--'}W`;

      const btnBenchmark = document.getElementById('btnBenchmark');
      const btnBenchmarkFull = document.getElementById('btnBenchmarkFull');
      const infoBenchPrefill = document.getElementById('info-bench-prefill');
      const infoBenchGen = document.getElementById('info-bench-gen');
      const infoBenchLast = document.getElementById('info-bench-last');
      const infoBenchHistory = document.getElementById('info-bench-history');

      const benchEnabled = true;
      const canRun = (!!data.active?.healthy) && !data.switch_in_progress && !bench.in_progress;
      if (btnBenchmark) {
        btnBenchmark.disabled = !canRun;
        btnBenchmark.textContent = (bench.in_progress && bench.profile === 'balanced') ? 'BENCHMARKING...' : 'RUN BENCHMARK';
      }
      if (btnBenchmarkFull) {
        btnBenchmarkFull.disabled = !canRun;
        btnBenchmarkFull.textContent = (bench.in_progress && bench.profile === 'full') ? 'BENCHMARKING...' : 'RUN FULL BENCHMARK';
      }

      const benchRes = bench.last_result || {};
      infoBenchPrefill.textContent = benchRes.prefill_tps ? `${benchRes.prefill_tps.toFixed(2)} T/S` : '-- T/S';
      infoBenchGen.textContent = benchRes.gen_tps ? `${benchRes.gen_tps.toFixed(2)} T/S` : '-- T/S';
      if (bench.in_progress && bench.started_at) {
        infoBenchLast.textContent = `${(bench.profile || 'balanced').toUpperCase()} running since ${fmtLocalTs(bench.started_at)}`;
      } else if (bench.last_error) {
        infoBenchLast.textContent = `Failed: ${bench.last_error}`;
      } else if (benchRes.completed_at) {
        infoBenchLast.textContent = `${(benchRes.profile || 'balanced').toUpperCase()} at ${fmtLocalTs(benchRes.completed_at)}`;
      } else {
        infoBenchLast.textContent = '--';
      }

      const historyEl = document.getElementById('info-bench-history');
      const rows = (bench.history || []).slice().reverse();
      if (!rows.length) {
        historyEl.innerHTML = '<div class="bench-history-empty">No benchmark runs yet.</div>';
      } else {
        historyEl.innerHTML = rows.map((r) => {
          const when = fmtLocalTs(r.completed_at || r.started_at);
          const profile = (r.profile || 'balanced').toUpperCase();
          if (!r.success) {
            return `<div class="bench-history-row">${profile} ${escHtml(when)}<br/>FAIL: ${escHtml(r.error || 'Unknown error')}</div>`;
          }
          const p = Number(r.prefill_tps || 0).toFixed(2);
          const g = Number(r.gen_tps || 0).toFixed(2);
          return `<div class="bench-history-row">${profile} ${escHtml(when)}<br/>P ${p} T/S | G ${g} T/S</div>`;
        }).join('');
      }

      // Process table
      let procs = '<tr><td colspan="7" style="text-align:center; color:var(--text-muted)">No active compute processes</td></tr>';
      if (data.gpu_procs?.length) {
        procs = data.gpu_procs.map(p => {
          const vram = p.vram_mb != null ? (p.vram_mb/1024).toFixed(1)+' GB' : '--';
          const cpu  = p.cpu_pct  != null ? p.cpu_pct.toFixed(1)+'%' : '--';
          const ram  = p.ram_mb   != null ? (p.ram_mb/1024).toFixed(1)+' GB' : '--';
          const gpu  = p.gpu_pct  != null ? p.gpu_pct.toFixed(0)+'%' : '--';
          return `<tr><td>${p.pid}</td><td style="color:var(--accent);font-weight:700">${escHtml(p.name)}</td><td>${vram}</td><td>${cpu}</td><td>${ram}</td><td>${gpu}</td><td><span style="color:var(--success)">&#9679;</span> RUNNING</td></tr>`;
        }).join('');
      }
      document.getElementById('procTable').innerHTML = procs;

      // Switch status
      document.getElementById('switchStatus').textContent = data.last_message || '';

      // Sidebar
      buildSidebar(data);

      // Events & Logs
      updateEventsPanel(data);
    }

    function confirmSwitch(key, label) {
      const isActive = (lastData?.active?.key === key && lastData?.active?.healthy) ||
                       (key === 'vllm' && lastData?.active_vllm?.healthy);
      if (isActive) {
        const el = document.getElementById('switchStatus');
        el.textContent = `${label} is already active and healthy.`;
        setTimeout(() => { el.textContent = lastData?.last_message || ''; }, 3000);
        return;
      }
      if (lastData?.switch_in_progress) return;
      closeSidebar();
      pendingAction = { type: 'switch', key };
      document.getElementById('modalTitle').textContent = 'Switch Model?';
      if (key === 'vllm') {
        document.getElementById('modalText').textContent =
          'Switch to vLLM? The current LLM (llama.cpp) will be stopped and vLLM loaded into VRAM.';
      } else {
        document.getElementById('modalText').textContent =
          `Switch to ${label}? The current LLM will be stopped and the new model loaded into VRAM.`;
      }
      document.getElementById('btnConfirmSwitch').textContent = 'PROCEED';
      document.getElementById('btnConfirmSwitch').style.background = 'var(--accent)';
      document.getElementById('confirmModal').classList.add('active');
    }

    function confirmStopAll() {
      closeSidebar();
      pendingAction = { type: 'stop' };
      document.getElementById('modalTitle').textContent = 'Stop All Models?';
      document.getElementById('modalText').textContent =
        'Stop all running LLM containers? GPU will be released.';
      document.getElementById('btnConfirmSwitch').textContent = 'STOP ALL';
      document.getElementById('btnConfirmSwitch').style.background = 'var(--danger)';
      document.getElementById('confirmModal').classList.add('active');
    }

    function confirmRestart() {
      if (!lastData?.active?.key) {
        const el = document.getElementById('switchStatus');
        el.textContent = 'No active model to restart.';
        setTimeout(() => { el.textContent = lastData?.last_message || ''; }, 3000);
        return;
      }
      if (lastData?.switch_in_progress) return;
      closeSidebar();
      pendingAction = { type: 'restart' };
      document.getElementById('modalTitle').textContent = 'Restart Model?';
      document.getElementById('modalText').textContent =
        `Restart ${lastData.active.label}? The model will be stopped and reloaded.`;
      document.getElementById('btnConfirmSwitch').textContent = 'RESTART';
      document.getElementById('btnConfirmSwitch').style.background = 'var(--warning)';
      document.getElementById('confirmModal').classList.add('active');
    }

    async function triggerBenchmark(profile) {
      const p = (profile || 'balanced').toLowerCase();
      if (lastData?.switch_in_progress || !lastData?.active?.healthy || lastData?.benchmark?.in_progress) {
        return;
      }
      try {
        document.getElementById('switchStatus').textContent = `Starting ${p} benchmark...`;
        const r = await fetch('/api/benchmark', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ profile: p })
        });
        let d = {};
        try {
          d = await r.json();
        } catch (_) {}
        if (!r.ok) {
          const msg = d?.error || `Benchmark request failed (${r.status})`;
          document.getElementById('switchStatus').textContent = msg;
          return;
        }
        document.getElementById('switchStatus').textContent = d?.message || `${p} benchmark started`;
        refresh();
      } catch (e) {
        document.getElementById('switchStatus').textContent = 'Benchmark error: ' + e.message;
      }
    }

    function closeModal() {
      document.getElementById('confirmModal').classList.remove('active');
      pendingAction = null;
    }

    document.getElementById('btnConfirmSwitch').onclick = async () => {
      const action = pendingAction;
      closeModal();
      if (!action) return;
      try {
        document.getElementById('switchStatus').textContent = 'Sending request...';
        if (action.type === 'switch') {
          await fetch('/api/switch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: action.key })
          });
        } else if (action.type === 'stop') {
          await fetch('/api/stop', { method: 'POST' });
        } else if (action.type === 'restart') {
          await fetch('/api/restart', { method: 'POST' });
        }
        refresh();
      } catch (e) {
        document.getElementById('switchStatus').textContent = 'Error: ' + e.message;
      }
    };

    // ---- Mobile sidebar drawer ----
    function openSidebar() {
      document.querySelector('.sidebar').classList.add('open');
      document.getElementById('sidebarBackdrop').classList.add('open');
    }
    function closeSidebar() {
      document.querySelector('.sidebar').classList.remove('open');
      document.getElementById('sidebarBackdrop').classList.remove('open');
    }
    function toggleSidebar() {
      if (document.querySelector('.sidebar').classList.contains('open')) closeSidebar();
      else openSidebar();
    }
    window.addEventListener('resize', () => { if (window.innerWidth > 768) closeSidebar(); });

    // Quick navigation from the drawer/sidebar to a section.
    // Scroll the .main-content container explicitly (scrollIntoView is unreliable
    // inside a nested overflow container).
    function scrollMainTo(el) {
      const main = document.querySelector('.main-content');
      const top = el.getBoundingClientRect().top - main.getBoundingClientRect().top + main.scrollTop - 8;
      main.scrollTo({ top: Math.max(0, top), behavior: 'smooth' });
    }
    function goTo(section) {
      closeSidebar();
      setTimeout(() => {
        if (section === 'overview') {
          document.querySelector('.main-content').scrollTo({ top: 0, behavior: 'smooth' });
        } else if (section === 'requests') {
          switchActivityTab('reqlog');
          scrollMainTo(document.getElementById('activityPanel'));
        } else if (section === 'events') {
          switchActivityTab('events');
          scrollMainTo(document.getElementById('activityPanel'));
        }
      }, 60);
    }

    let activeActivityTab = 'reqlog';
    function switchActivityTab(tab) {
      activeActivityTab = tab;
      document.getElementById('tabbtn-reqlog').classList.toggle('active', tab === 'reqlog');
      document.getElementById('tabbtn-events').classList.toggle('active', tab === 'events');
      document.getElementById('tab-reqlog').hidden = tab !== 'reqlog';
      document.getElementById('tab-events').hidden = tab !== 'events';
      // The filter toolbar only applies to the request log.
      document.getElementById('reqlogTools').style.visibility = tab === 'reqlog' ? 'visible' : 'hidden';
      if (tab === 'reqlog') refreshRequestLog(true);
    }

    function updateEventsPanel(data) {
      const events = data.log_events || [];
      const errCnt = data.log_error_count || 0;
      const hb = document.getElementById('hdrErrBadge');
      if (hb) { hb.style.display = errCnt > 0 ? '' : 'none'; hb.textContent = `${errCnt} ERROR${errCnt !== 1 ? 'S' : ''}`; }
      const pb = document.getElementById('eventsBadge');
      if (pb) { pb.style.display = errCnt > 0 ? '' : 'none'; pb.textContent = `${errCnt} error${errCnt !== 1 ? 's' : ''}`; }
      const body = document.getElementById('eventsBody');
      if (!events.length) {
        body.innerHTML = '<div class="events-empty-msg">No events yet.</div>';
        eventsLastCount = 0; return;
      }
      const sorted = [...events].reverse();
      body.innerHTML = sorted.map(e => {
        const ts = fmtLocalTs(e.ts).replace(/^.*?,\s*/, '');
        return `<div class="event-row"><span class="event-ts">${escHtml(ts)}</span><span class="event-sev ev-${e.severity}">${e.severity.toUpperCase()}</span><span class="event-msg">${escHtml(e.message)}</span></div>`;
      }).join('');
      if (events.length > eventsLastCount) body.scrollTop = 0;
      eventsLastCount = events.length;
    }

    async function refresh() {
      if (isRefreshing) return;
      isRefreshing = true;
      try {
        const r = await fetch('/api/status', { cache: 'no-store' });
        const d = await r.json();
        lastUpdated = Date.now();
        lastData = d;
        setHeaderStatus(d);
        updateDashboard(d);
      } catch (e) {
        console.error('Refresh error:', e);
      } finally {
        isRefreshing = false;
      }
    }

    // Update "X seconds ago" without a full API call
    setInterval(() => {
      const secs = Math.round((Date.now() - lastUpdated) / 1000);
      const el = document.getElementById('headerUpdated');
      if (el) el.textContent = `Updated ${secs}s ago`;
    }, 1000);

    // ---- Request Log ----
    let reqlogSearchTimer = null;
    let reqlogBusy = false;

    function onReqlogSearch() {
      clearTimeout(reqlogSearchTimer);
      reqlogSearchTimer = setTimeout(() => refreshRequestLog(true), 300);
    }

    function fmtLatency(ms) {
      if (ms == null) return '--';
      if (ms >= 1000) return (ms / 1000).toFixed(1) + 's';
      return Math.round(ms) + 'ms';
    }

    function reqlogTokens(r) {
      if (r.prompt_tokens == null && r.completion_tokens == null) return '--';
      const p = r.prompt_tokens == null ? '?' : r.prompt_tokens;
      const c = r.completion_tokens == null ? '?' : r.completion_tokens;
      return p + '/' + c;
    }

    async function refreshRequestLog(force) {
      const auto = document.getElementById('reqlogAuto');
      if (!force && (!auto || !auto.checked)) return;
      // Don't churn the list out from under an open detail modal.
      if (!force && document.getElementById('reqDetailModal').classList.contains('active')) return;
      if (reqlogBusy) return;
      reqlogBusy = true;
      try {
        const ep = document.getElementById('reqlogEndpoint').value;
        const q = document.getElementById('reqlogSearch').value.trim();
        const params = new URLSearchParams({ limit: '300', endpoint: ep });
        if (q) params.set('q', q);
        const r = await fetch('/api/request-log?' + params.toString(), { cache: 'no-store' });
        const d = await r.json();
        const body = document.getElementById('reqlogBody');
        const countEl = document.getElementById('reqlogCount');
        if (!d.available) {
          body.innerHTML = '<tr><td colspan="10" class="reqlog-empty">No request log yet — start a model stack, then send a request to port 8080 or 28082.</td></tr>';
          countEl.textContent = '';
          return;
        }
        if (d.error) {
          body.innerHTML = '<tr><td colspan="10" class="reqlog-empty">Error reading log: ' + escHtml(d.error) + '</td></tr>';
          countEl.textContent = '';
          return;
        }
        const rows = d.rows || [];
        countEl.textContent = rows.length ? String(rows.length) : '';
        if (!rows.length) {
          body.innerHTML = '<tr><td colspan="10" class="reqlog-empty">No matching requests.</td></tr>';
          return;
        }
        body.innerHTML = rows.map(r => {
          const t = fmtLocalTs(r.ts_iso).replace(/^.*?,\s*/, '');
          const epCls = r.endpoint === '8080' ? 'reqlog-ep-8080' : 'reqlog-ep-28082';
          const ua = r.user_agent ? '<span class="reqlog-ua" title="' + escHtml(r.user_agent) + '">' + escHtml(r.user_agent) + '</span>' : '';
          const origin = '<span class="reqlog-origin-ip">' + escHtml(r.client_ip || '--') + '</span>' + ua;
          const statusCls = (r.status && r.status < 400) ? 'status-ok' : 'status-bad';
          const snip = r.prompt_snippet ? '<span class="reqlog-snip" title="' + escHtml(r.prompt_snippet) + '">' + escHtml(r.prompt_snippet) + '</span>' : '';
          const streamMark = r.stream ? ' <span title="streamed" style="color:var(--accent)">↯</span>' : '';
          return '<tr class="reqlog-row" onclick="openRequestDetail(' + r.id + ')">'
            + '<td class="reqlog-time">' + escHtml(t) + '</td>'
            + '<td><span class="reqlog-ep ' + epCls + '">' + escHtml(r.endpoint || '?') + '</span></td>'
            + '<td>' + origin + '</td>'
            + '<td class="reqlog-model">' + escHtml(r.model || '--') + '</td>'
            + '<td class="reqlog-path">' + escHtml(r.path || '--') + streamMark + '</td>'
            + '<td><span class="status-pill ' + statusCls + '">' + escHtml(r.status == null ? '--' : r.status) + '</span></td>'
            + '<td>' + escHtml(fmtLatency(r.latency_ms)) + '</td>'
            + '<td>' + escHtml(reqlogTokens(r)) + '</td>'
            + '<td>' + snip + '</td>'
            + '<td><span class="reqlog-view">View →</span></td>'
            + '</tr>';
        }).join('');
      } catch (e) {
        console.error('Request log error:', e);
      } finally {
        reqlogBusy = false;
      }
    }

    // ---- Request detail modal ----
    function closeRequestDetail() {
      document.getElementById('reqDetailModal').classList.remove('active');
    }

    function rdContentToHtml(content) {
      // content may be a string or an array of OpenAI content parts.
      if (typeof content === 'string') return escHtml(content);
      if (Array.isArray(content)) {
        return content.map(part => {
          if (!part || typeof part !== 'object') return escHtml(String(part));
          if (part.type === 'text') return escHtml(part.text || '');
          if (part.type === 'image_url') {
            const u = (part.image_url && part.image_url.url) || '';
            return '<div class="rd-img">🖼 image: ' + escHtml(u.length > 80 ? u.slice(0, 80) + '…' : u) + '</div>';
          }
          return escHtml(JSON.stringify(part));
        }).join('');
      }
      if (content == null) return '<span class="rd-note">(empty)</span>';
      return escHtml(JSON.stringify(content));
    }

    function rdMessage(role, content, reasoning) {
      const cls = 'rd-role-' + (['system','user','assistant','tool'].includes(role) ? role : 'system');
      let html = '<div class="rd-msg"><div class="rd-msg-role ' + cls + '">' + escHtml(role || 'message') + '</div>';
      html += '<div class="rd-msg-content">' + (rdContentToHtml(content) || '<span class="rd-note">(no content)</span>') + '</div>';
      if (reasoning) {
        html += '<details class="rd-reasoning"><summary>Reasoning (' + reasoning.length + ' chars)</summary>'
             + '<div class="rd-msg-content">' + escHtml(reasoning) + '</div></details>';
      }
      html += '</div>';
      return html;
    }

    function renderRequestDetail(r) {
      const epCls = r.endpoint === '8080' ? 'reqlog-ep-8080' : 'reqlog-ep-28082';
      let html = '<div class="rd-meta">';
      const metas = [
        ['Endpoint', '<span class="reqlog-ep ' + epCls + '">' + escHtml(r.endpoint || '?') + '</span>'],
        ['Time', escHtml(fmtLocalTs(r.ts_iso))],
        ['Origin IP', escHtml(r.client_ip || '--')],
        ['User-Agent', escHtml(r.user_agent || '--')],
        ['Method', escHtml(r.method || '--')],
        ['Path', escHtml(r.path || '--')],
        ['Status', escHtml(r.status == null ? '--' : String(r.status))],
        ['Latency', escHtml(fmtLatency(r.latency_ms))],
        ['Stream', r.stream ? 'yes' : 'no'],
        ['Tokens (p/c/total)', escHtml((r.prompt_tokens ?? '?') + ' / ' + (r.completion_tokens ?? '?') + ' / ' + (r.total_tokens ?? '?'))],
      ];
      html += metas.map(m => '<div class="rd-m"><span class="rd-k">' + m[0] + '</span><span class="rd-v">' + m[1] + '</span></div>').join('');
      html += '</div>';

      // Request
      let req = null;
      try { req = r.request_body ? JSON.parse(r.request_body) : null; } catch (e) {}
      html += '<div class="rd-section-title">Request</div>';
      if (req) {
        const paramKeys = ['model','temperature','top_p','top_k','max_tokens','presence_penalty','frequency_penalty','stream'];
        const params = paramKeys.filter(k => req[k] !== undefined)
          .map(k => '<span class="rd-param"><b>' + k + '</b>: ' + escHtml(JSON.stringify(req[k])) + '</span>').join('');
        if (params) html += '<div class="rd-params">' + params + '</div>';
        const msgs = Array.isArray(req.messages) ? req.messages : null;
        if (msgs) {
          html += msgs.map(m => rdMessage(m.role, m.content, m.reasoning_content)).join('');
        } else if (typeof req.prompt === 'string') {
          html += rdMessage('prompt', req.prompt);
        }
      } else if (r.request_body) {
        html += '<div class="rd-note">Request body too large to parse (' + r.request_body.length + ' chars, likely truncated). See Raw JSON below.</div>';
      } else {
        html += '<div class="rd-note">No request body stored for this request (only chat/completions/embeddings bodies are captured).</div>';
      }

      // Response
      let resp = null;
      try { resp = r.response_body ? JSON.parse(r.response_body) : null; } catch (e) {}
      html += '<div class="rd-section-title">Response' + (resp && resp._reconstructed_from_stream ? ' <span class="rd-note">(reconstructed from stream)</span>' : '') + '</div>';
      if (resp && Array.isArray(resp.choices) && resp.choices.length) {
        const ch = resp.choices[0];
        const msg = ch.message || ch.delta || {};
        html += rdMessage('assistant', msg.content, msg.reasoning_content);
        const bits = [];
        if (ch.finish_reason) bits.push('<span class="rd-param"><b>finish</b>: ' + escHtml(ch.finish_reason) + '</span>');
        if (resp.usage) bits.push('<span class="rd-param"><b>usage</b>: ' + escHtml(JSON.stringify(resp.usage)) + '</span>');
        if (bits.length) html += '<div class="rd-params">' + bits.join('') + '</div>';
      } else if (resp) {
        html += '<div class="rd-note">Non-chat response.</div>';
      } else if (r.response_body) {
        html += '<div class="rd-note">Response body too large to parse (' + r.response_body.length + ' chars, likely truncated). See Raw JSON below.</div>';
      } else {
        html += '<div class="rd-note">No response body stored.</div>';
      }

      // Raw
      if (r.request_body || r.response_body) {
        html += '<details class="rd-raw"><summary>Raw JSON</summary>';
        if (r.request_body) html += '<div class="rd-k" style="margin-top:8px">request_body</div><pre>' + escHtml(r.request_body) + '</pre>';
        if (r.response_body) html += '<div class="rd-k" style="margin-top:8px">response_body</div><pre>' + escHtml(r.response_body) + '</pre>';
        html += '</details>';
      }
      return html;
    }

    async function openRequestDetail(id) {
      const modal = document.getElementById('reqDetailModal');
      const bodyEl = document.getElementById('reqDetailBody');
      const titleEl = document.getElementById('reqDetailTitle');
      titleEl.textContent = 'Request #' + id;
      bodyEl.innerHTML = '<div class="rd-note">Loading…</div>';
      modal.classList.add('active');
      try {
        const r = await fetch('/api/request-log/detail?id=' + encodeURIComponent(id), { cache: 'no-store' });
        const d = await r.json();
        if (!d.row) {
          bodyEl.innerHTML = '<div class="rd-note">Not found' + (d.error ? ': ' + escHtml(d.error) : '') + '</div>';
          return;
        }
        titleEl.textContent = (d.row.method || '') + ' ' + (d.row.path || '') + '  ·  #' + id;
        bodyEl.innerHTML = renderRequestDetail(d.row);
      } catch (e) {
        bodyEl.innerHTML = '<div class="rd-note">Error: ' + escHtml(e.message) + '</div>';
      }
    }

    document.addEventListener('keydown', (e) => { if (e.key === 'Escape') { closeRequestDetail(); closeSidebar(); } });

    initChart();
    refresh();
    setInterval(refresh, 500);
    refreshRequestLog(true);
    setInterval(() => refreshRequestLog(false), 3000);
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, data: dict, status: int = 200) -> None:
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
        self._request_status = status

    def _send_html(self, html: str, status: int = 200) -> None:
        payload = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
        self._request_status = status

    def _caller_ip(self) -> str:
        forwarded = self.headers.get("X-Forwarded-For", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return self.client_address[0]

    def _caller_ua(self) -> str:
        return self.headers.get("User-Agent", "")

    def _begin_request(self) -> None:
        self._request_start = time.time()
        self._request_status = 0
        self._request_body: dict | None = None

    def _finish_request(self, method: str, path: str) -> None:
        elapsed_ms = (time.time() - self._request_start) * 1000
        caller = self._caller_ip()
        ua = self._caller_ua()
        status = getattr(self, "_request_status", 0) or 200
        action = _human_readable_path(path, getattr(self, "_request_body", None))
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        ua_suffix = f'  UA: "{ua}"' if ua else ""
        print(f"[{ts}] {caller}  {method} {path} → {status}  [{action}]  {elapsed_ms:.1f}ms{ua_suffix}")
        if path in ("/api/switch", "/api/stop", "/api/restart", "/api/benchmark"):
            sev = "error" if status >= 400 else "info"
            _append_log_event(
                sev,
                "request",
                f"[{caller}] {action}" + (f"  [{ua[:80]}]" if ua else ""),
            )

    def do_GET(self) -> None:
        self._begin_request()
        parsed = urlparse(self.path)
        route = parsed.path
        if route in ("/", "/index.html"):
            self._send_html(INDEX_HTML)
        elif route == "/api/status":
            status = build_status(self)
            self._send_json(status)
        elif route == "/api/request-log":
            self._send_json(read_request_log(parse_qs(parsed.query)))
        elif route == "/api/request-log/detail":
            self._send_json(read_request_detail(parse_qs(parsed.query)))
        else:
            self._send_json({"error": "Not found"}, 404)
        self._finish_request("GET", self.path)

    def do_POST(self) -> None:
        self._begin_request()

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
            body = json.loads(raw.decode("utf-8"))
            self._request_body = body
        except Exception:
            self._send_json({"error": "Invalid JSON body"}, 400)
            self._finish_request("POST", self.path)
            return

        if self.path == "/api/switch":
            model_key = body.get("model", "")
            accepted, msg = start_switch(model_key)
            if not accepted:
                self._send_json({"error": msg}, 409)
            else:
                self._send_json({"message": msg, "status": build_status(self)})
            self._finish_request("POST", self.path)
            return

        if self.path == "/api/benchmark":
            profile = body.get("profile", "balanced")
            if isinstance(profile, str) and profile.strip():
                profile = profile.strip().lower()
            else:
                profile = "balanced"

            if profile not in {"balanced", "full"}:
                self._send_json({"error": "Unsupported benchmark profile"}, 400)
                self._finish_request("POST", self.path)
                return

            models, _ = get_models()
            containers = list_llama_compose_containers()
            active_key = detect_active_model_key(containers)
            if not active_key or active_key not in models:
                self._send_json({"error": "No active model detected"}, 409)
                self._finish_request("POST", self.path)
                return
            if not is_model_server_healthy(active_key, containers):
                self._send_json({"error": "Active model is not healthy yet"}, 409)
                self._finish_request("POST", self.path)
                return

            with STATE_LOCK:
                if STATE["switch_in_progress"]:
                    self._send_json({"error": "Cannot benchmark while switch is in progress"}, 409)
                    self._finish_request("POST", self.path)
                    return

            with BENCHMARK_LOCK:
                if BENCHMARK_STATE["in_progress"]:
                    self._send_json({"error": "A benchmark is already in progress"}, 409)
                    self._finish_request("POST", self.path)
                    return
                BENCHMARK_STATE["in_progress"] = True
                BENCHMARK_STATE["profile"] = profile
                BENCHMARK_STATE["started_at"] = now_iso()
                BENCHMARK_STATE["completed_at"] = None
                BENCHMARK_STATE["last_error"] = None

            with STATE_LOCK:
                STATE["last_message"] = (
                    f"Running {profile} benchmark for {models[active_key]['label']}..."
                )

            def _benchmark_worker(model_key: str, benchmark_profile: str) -> None:
                started = time.time()
                started_iso = now_iso()
                result = None
                error = None
                model_label = models[model_key]["label"]
                _append_log_event("info", "benchmark", f"Benchmark ({benchmark_profile}) started for {model_label}")
                try:
                    local_containers = list_llama_compose_containers()
                    result, error = run_benchmark_profile(
                        model_key,
                        local_containers,
                        benchmark_profile,
                        progress_cb=lambda msg: _set_last_message(msg),
                    )
                except Exception as exc:
                    error = f"Unhandled benchmark error: {exc}"

                completed_iso = now_iso()
                elapsed = max(0.0, time.time() - started)

                with BENCHMARK_LOCK:
                    BENCHMARK_STATE["in_progress"] = False
                    BENCHMARK_STATE["completed_at"] = completed_iso
                    if error:
                        BENCHMARK_STATE["last_error"] = error
                        _append_log_event("error", "benchmark", f"Benchmark failed for {model_label}: {error}")
                        BENCHMARK_STATE["history"].append({
                            "profile": benchmark_profile,
                            "model_key": model_key,
                            "started_at": started_iso,
                            "completed_at": completed_iso,
                            "duration_sec": elapsed,
                            "success": False,
                            "error": error,
                        })
                    elif result is not None:
                        result["started_at"] = started_iso
                        result["completed_at"] = completed_iso
                        result["duration_sec"] = elapsed
                        BENCHMARK_STATE["last_result"] = result
                        BENCHMARK_STATE["last_error"] = None
                        _append_log_event("info", "benchmark", f"Benchmark ({benchmark_profile}) for {model_label}: prefill {result['prefill_tps']:.1f} T/S, gen {result['gen_tps']:.1f} T/S")
                        BENCHMARK_STATE["history"].append({
                            "profile": benchmark_profile,
                            "model_key": model_key,
                            "started_at": started_iso,
                            "completed_at": completed_iso,
                            "duration_sec": elapsed,
                            "success": True,
                            "prefill_tps": result["prefill_tps"],
                            "gen_tps": result["gen_tps"],
                        })
                    else:
                        BENCHMARK_STATE["last_error"] = "Benchmark failed without result"
                    BENCHMARK_STATE["history"] = BENCHMARK_STATE["history"][-FULL_BENCHMARK_MAX_HISTORY:]

                with STATE_LOCK:
                    if error:
                        STATE["last_message"] = f"Benchmark failed: {error}"
                    elif result is not None:
                        STATE["last_message"] = (
                            f"{benchmark_profile} benchmark complete: prefill {result['prefill_tps']:.2f} T/S, "
                            f"gen {result['gen_tps']:.2f} T/S"
                        )
                    else:
                        STATE["last_message"] = "Benchmark failed without result"

            threading.Thread(
                target=_benchmark_worker,
                args=(active_key, profile),
                daemon=True,
            ).start()
            self._send_json(
                {"message": f"{profile.capitalize()} benchmark started", "status": build_status(self)}
            )
            self._finish_request("POST", self.path)
            return

        if self.path == "/api/stop":
            with STATE_LOCK:
                if STATE["switch_in_progress"]:
                    self._send_json({"error": "A switch is already in progress"}, 409)
                    self._finish_request("POST", self.path)
                    return
                STATE["switch_in_progress"] = True
                STATE["last_message"] = "Stopping all models..."
            _append_log_event("info", "stop", "Stop-all initiated")

            def _stop_worker() -> None:
                cmd = [
                    "bash", "-lc",
                    f"cd {shlex.quote(str(LLAMA_DIR))} && {shlex.quote(str(SWITCH_SCRIPT))} --stop-all",
                ]
                exit_code, output = run_command(cmd)
                with STATE_LOCK:
                    STATE["switch_in_progress"] = False
                    STATE["last_exit_code"] = exit_code
                    STATE["last_output"] = output[-2000:]
                    STATE["last_message"] = "All models stopped." if exit_code == 0 else "Stop-all failed."
                sev = "info" if exit_code == 0 else "error"
                msg = "All models stopped" if exit_code == 0 else f"Stop-all failed (exit {exit_code})"
                _append_log_event(sev, "stop", msg)

            threading.Thread(target=_stop_worker, daemon=True).start()
            self._send_json({"message": "Stop-all initiated"})
            self._finish_request("POST", self.path)
            return

        if self.path == "/api/restart":
            models, _ = get_models()
            containers = list_llama_compose_containers()
            active_key = detect_active_model_key(containers)
            if not active_key:
                self._send_json({"error": "No active model to restart"}, 409)
                self._finish_request("POST", self.path)
                return
            accepted, msg = start_switch(active_key)
            if not accepted:
                self._send_json({"error": msg}, 409)
            else:
                self._send_json({"message": msg, "status": build_status(self)})
            self._finish_request("POST", self.path)
            return

        if self.path == "/api/reset":
            with STATE_LOCK:
                STATE["switch_in_progress"] = False
                STATE["last_message"] = "State reset by user."
            _append_log_event("info", "reset", "State reset by user")
            self._send_json({"message": "State reset"})
            self._finish_request("POST", self.path)
            return

        self._send_json({"error": "Not found"}, 404)
        self._finish_request("POST", self.path)

    def log_message(self, fmt: str, *args) -> None:
        pass


def main() -> None:
    if not SWITCH_SCRIPT.exists():
        raise SystemExit(f"Missing switch script: {SWITCH_SCRIPT}")

    threading.Thread(target=_run_log_watcher, daemon=True, name="log-watcher").start()

    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"dashboard server listening on {HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
