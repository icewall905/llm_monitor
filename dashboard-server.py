#!/usr/bin/env python3
import ast
import json
import os
import random
import re
import select
import shlex
import sqlite3
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HOST = "0.0.0.0"
PORT = int(os.environ.get("DASHBOARD_PORT", "8080"))
REQUEST_LOG_DB = os.environ.get("REQUEST_LOG_DB", "/request-logs/requests.db")
VLLM_API_KEY_FILE = os.environ.get("VLLM_API_KEY_FILE", "")

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

# ── User settings ────────────────────────────────────────────────────────────
# docker-compose.yml sets LLAMA_DIR/STACKS_DIR/VLLM_DIR/... as environment variables,
# and config.yaml documents env as always winning. A settings UI that lost to env on
# this deployment would silently do nothing, so values saved from the dashboard are
# kept in their own file and given the highest precedence. The UI shows which source
# each effective value came from, so the override is never invisible.
SETTINGS_PATH = Path(os.environ.get("DASHBOARD_SETTINGS", "/request-logs/settings.json"))
SETTINGS_LOCK = threading.Lock()
USER_SETTINGS: dict = {}


def load_user_settings() -> dict:
    try:
        if SETTINGS_PATH.is_file():
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {k: v for k, v in data.items() if isinstance(v, str) and v.strip()}
    except Exception:  # noqa: BLE001
        pass
    return {}


USER_SETTINGS = load_user_settings()


def resolve_setting(key: str, env_name: str, default: str) -> tuple[str, str]:
    """Effective value for a path setting plus where it came from."""
    if USER_SETTINGS.get(key):
        return USER_SETTINGS[key], "settings"
    if os.environ.get(env_name):
        return os.environ[env_name], "env"
    if RUNTIME_CONFIG.get(key):
        return str(RUNTIME_CONFIG[key]), "config"
    return default, "default"


_llama_dir_val, _ = resolve_setting("llama_dir", "LLAMA_DIR", "/opt/llama.cpp")
DEFAULT_LLAMA_DIR = str(RUNTIME_CONFIG.get("llama_dir") or "/opt/llama.cpp")
LLAMA_DIR = Path(_llama_dir_val)

DEFAULT_STACKS_DIR = str(RUNTIME_CONFIG.get("stacks_dir") or (LLAMA_DIR / "stacks"))
STACKS_DIR = Path(resolve_setting("stacks_dir", "STACKS_DIR", str(LLAMA_DIR / "stacks"))[0])

MODELS_DIR = Path(resolve_setting("models_dir", "MODELS_DIR", str(LLAMA_DIR / "models"))[0])

DEFAULT_SWITCH_SCRIPT = str(RUNTIME_CONFIG.get("switch_script") or (LLAMA_DIR / "switch-llm.sh"))
SWITCH_SCRIPT = Path(os.environ.get("SWITCH_SCRIPT", DEFAULT_SWITCH_SCRIPT))

LLAMA_WORKING_DIR_LABEL = os.environ.get(
    "LLAMA_WORKING_DIR_LABEL",
    str(RUNTIME_CONFIG.get("llama_working_dir_label") or "/opt/llama.cpp"),
)

# ── vLLM config ──────────────────────────────────────────────────────────────
DEFAULT_VLLM_DIR = str(RUNTIME_CONFIG.get("vllm_dir") or "/opt/vllm")
VLLM_DIR = Path(resolve_setting("vllm_dir", "VLLM_DIR", "/opt/vllm")[0])
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
BEELLAMA_DIR = Path(resolve_setting("beellama_dir", "BEELLAMA_DIR", "/opt/beellama.cpp")[0])
BEELLAMA_WORKING_DIR_LABEL = os.environ.get(
    "BEELLAMA_WORKING_DIR_LABEL",
    str(RUNTIME_CONFIG.get("beellama_working_dir_label") or str(BEELLAMA_DIR)),
)
BEELLAMA_STACKS_DIR = Path(resolve_setting(
    "beellama_stacks_dir", "BEELLAMA_STACKS_DIR", str(BEELLAMA_DIR / "stacks")
)[0])

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
# ── Metrics history ──────────────────────────────────────────────────────────
# The dashboard's in-page buffers only ever held 60 samples and died on reload, so
# anything past a couple of minutes has to be persisted server-side. SQLite on the
# same rw volume the request log already uses, so it survives container rebuilds.
METRICS_DB = os.environ.get("METRICS_DB", "/request-logs/metrics.db")
METRICS_SAMPLE_SEC = float(os.environ.get("METRICS_SAMPLE_SEC", "5"))
METRICS_RETENTION_DAYS = float(os.environ.get("METRICS_RETENTION_DAYS", "31"))
# Points returned per query. Buckets are sized to hit this, so a 30-day range costs
# the same over the wire as a 5-minute one.
METRICS_MAX_POINTS = int(os.environ.get("METRICS_MAX_POINTS", "240"))
METRICS_PRUNE_EVERY_SEC = 3600.0
METRICS_RANGES = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "24h": 86400,
    "7d": 604800,
    "30d": 2592000,
}
# A throughput reading older than this is stale — record NULL rather than a flat line
# that implies the model was idle when really nothing was sampling it.
METRICS_LIVE_MAX_AGE_SEC = 20.0
METRICS_LIVE_LOCK = threading.Lock()
METRICS_LIVE = {"ts": 0.0, "tps": None, "ingest": None}
METRICS_DB_LOCK = threading.Lock()
METRICS_DB_READY = False

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


SETTING_DEFS = [
    {
        "key": "stacks_dir",
        "env": "STACKS_DIR",
        "label": "llama.cpp stacks folder",
        "help": "Scanned for *.yml containing '# LLM_META display_name=' headers.",
        "scans": True,
    },
    {
        "key": "vllm_dir",
        "env": "VLLM_DIR",
        "label": "vLLM stacks folder",
        "help": "Scanned for vLLM compose files. Leave as-is if you do not run vLLM.",
        "scans": True,
    },
    {
        "key": "beellama_stacks_dir",
        "env": "BEELLAMA_STACKS_DIR",
        "label": "BeeLlama stacks folder",
        "help": "Scanned for BeeLlama compose files.",
        "scans": True,
    },
    {
        "key": "llama_dir",
        "env": "LLAMA_DIR",
        "label": "llama.cpp root",
        "help": "Working directory for switch-llm.sh. Stack paths are resolved relative to it.",
        "scans": False,
    },
    {
        "key": "models_dir",
        "env": "MODELS_DIR",
        "label": "Models folder",
        "help": "Where the GGUF files live. Informational only.",
        "scans": False,
    },
    {
        "key": "beellama_dir",
        "env": "BEELLAMA_DIR",
        "label": "BeeLlama root",
        "help": "BeeLlama compose project root.",
        "scans": False,
    },
]
_SETTING_KEYS = {d["key"] for d in SETTING_DEFS}


def _current_setting_path(key: str) -> Path:
    return {
        "stacks_dir": STACKS_DIR,
        "vllm_dir": VLLM_DIR,
        "beellama_stacks_dir": BEELLAMA_STACKS_DIR,
        "llama_dir": LLAMA_DIR,
        "models_dir": MODELS_DIR,
        "beellama_dir": BEELLAMA_DIR,
    }[key]


def inspect_stack_dir(path: Path) -> dict:
    """Validate a candidate folder from inside the container.

    A host path that is not bind-mounted into this container simply does not exist
    here, which is the most common way to get this wrong — so say that plainly.
    """
    info = {"exists": False, "is_dir": False, "readable": False, "yml_count": 0, "stack_count": 0}
    try:
        info["exists"] = path.exists()
        info["is_dir"] = path.is_dir()
        if not info["is_dir"]:
            return info
        ymls = sorted(path.glob("*.yml")) + sorted(path.glob("*.yaml"))
        info["readable"] = True
        info["yml_count"] = len(ymls)
        info["stack_count"] = sum(
            1 for p in ymls if parse_llm_meta(p).get("display_name")
        )
    except PermissionError:
        info["readable"] = False
    except Exception:  # noqa: BLE001
        pass
    return info


def describe_settings() -> dict:
    items = []
    for d in SETTING_DEFS:
        _, source = resolve_setting(d["key"], d["env"], "")
        value = str(_current_setting_path(d["key"]))
        item = {
            "key": d["key"],
            "label": d["label"],
            "help": d["help"],
            "value": value,
            "source": source,
            "env_name": d["env"],
            "env_value": os.environ.get(d["env"]) or None,
            "config_value": str(RUNTIME_CONFIG.get(d["key"])) if RUNTIME_CONFIG.get(d["key"]) else None,
            "overridden": bool(USER_SETTINGS.get(d["key"])),
        }
        item.update(inspect_stack_dir(Path(value)))
        item["scans"] = d["scans"]
        items.append(item)
    return {
        "settings": items,
        "settings_path": str(SETTINGS_PATH),
        "writable": os.access(SETTINGS_PATH.parent, os.W_OK) if SETTINGS_PATH.parent.exists() else False,
    }


def apply_settings(new_values: dict) -> tuple[bool, str]:
    """Persist and hot-apply directory settings. No restart needed for discovery."""
    global USER_SETTINGS, LLAMA_DIR, STACKS_DIR, MODELS_DIR
    global VLLM_DIR, BEELLAMA_DIR, BEELLAMA_STACKS_DIR
    global _models_ts

    cleaned = {}
    for key, raw in new_values.items():
        if key not in _SETTING_KEYS:
            continue
        if raw is None:
            continue
        val = str(raw).strip()
        if not val:
            continue  # empty means "fall back to env/config/default"
        if not val.startswith("/"):
            return False, f"{key}: path must be absolute (got {val!r})"
        cleaned[key] = val.rstrip("/") or "/"

    with SETTINGS_LOCK:
        merged = dict(USER_SETTINGS)
        for key in _SETTING_KEYS:
            if key in new_values:
                # Present-but-empty clears the override.
                merged.pop(key, None)
        merged.update(cleaned)
        try:
            SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = SETTINGS_PATH.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(merged, indent=2), encoding="utf-8")
            tmp.replace(SETTINGS_PATH)
        except Exception as exc:  # noqa: BLE001
            return False, f"Could not write {SETTINGS_PATH}: {exc}"
        USER_SETTINGS = merged

    LLAMA_DIR = Path(resolve_setting("llama_dir", "LLAMA_DIR", "/opt/llama.cpp")[0])
    STACKS_DIR = Path(resolve_setting("stacks_dir", "STACKS_DIR", str(LLAMA_DIR / "stacks"))[0])
    MODELS_DIR = Path(resolve_setting("models_dir", "MODELS_DIR", str(LLAMA_DIR / "models"))[0])
    VLLM_DIR = Path(resolve_setting("vllm_dir", "VLLM_DIR", "/opt/vllm")[0])
    BEELLAMA_DIR = Path(resolve_setting("beellama_dir", "BEELLAMA_DIR", "/opt/beellama.cpp")[0])
    BEELLAMA_STACKS_DIR = Path(resolve_setting(
        "beellama_stacks_dir", "BEELLAMA_STACKS_DIR", str(BEELLAMA_DIR / "stacks")
    )[0])

    _models_ts = 0.0  # force rediscovery on the next get_models()
    _append_log_event("info", "settings", f"Settings updated: {', '.join(sorted(cleaned)) or 'cleared'}")
    return True, "Settings saved"


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
                # Slot cap for the "Streams" card's denominator. vLLM does NOT publish
                # max_num_seqs in /metrics (checked 2026-08-23: cache_config_info carries
                # kv_cache_size_tokens and kv_cache_max_concurrency, but no slot count), so
                # it has to come from the stack header like ctx_size and quant do.
                "max_seqs":       meta.get("max_seqs", ""),
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
# Minimum delta window for the vLLM live-rate calculations. Must be comfortably WIDER
# than VLLM_METRICS_CACHE_TTL_SEC, or the baseline advances faster than tokens accumulate
# and the rate collapses to 0 under steady load.
VLLM_TPS_MIN_WINDOW_SEC = float(os.environ.get("VLLM_TPS_MIN_WINDOW_SEC", "3.0"))
VLLM_TPS_STATE_LOCK = threading.Lock()
VLLM_TPS_STATE = {
    "container": None,
    "sampled_at": 0.0,
    "generation_tokens": None,
    "last_tps": None,
}
VLLM_INGEST_STATE_LOCK = threading.Lock()
VLLM_INGEST_STATE = {
    "container": None,
    "sampled_at": 0.0,
    "prompt_tokens": None,
    "prefill_seconds": None,
    "last_ingest_tps": None,
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


# Categories that mean the load will never succeed. Seeing one of these is grounds for
# ending the wait immediately instead of sitting on "Loading..." until the timeout.
FATAL_LOAD_CATEGORIES = ("oom", "model_load", "config", "restart_loop")


def _fatal_load_event_since(since_ts: float) -> dict | None:
    """Newest fatal load event logged after `since_ts`, or None.

    The timestamp filter matters: the event ring buffer outlives a switch, so without it
    an error from a *previous* failed load is picked up the instant a new switch starts.
    """
    with LOG_LOCK:
        events = list(LOG_STATE["events"])
    for evt in reversed(events):
        if evt.get("_ts_f", 0.0) < since_ts:
            break
        if evt.get("category") in FATAL_LOAD_CATEGORIES:
            return evt
    return None


def _loading_progress_message_since(since_ts: float, label: str) -> str | None:
    with LOG_LOCK:
        events = list(LOG_STATE["events"])
    for evt in reversed(events):
        if evt.get("_ts_f", 0.0) < since_ts:
            break
        if evt.get("category") == "loading":
            pct = re.search(r"(\d+(?:\.\d+)?)\s*[%％]", evt["message"])
            return f"Loading {label}: {pct.group(1)}%" if pct else f"Loading {label}..."
    return None


def _clear_switch_state(message: str | None = None, *, log: str | None = None) -> bool:
    """Drop out of the switching state. Returns True if a switch was actually in progress.

    Every switch worker funnels its exit through here, including the exception path, so a
    worker thread that dies unexpectedly can never strand the UI in "Loading...".
    """
    with STATE_LOCK:
        was_active = bool(STATE["switch_in_progress"])
        STATE["switch_in_progress"] = False
        STATE["last_completed_at"] = now_iso()
        if message is not None:
            STATE["last_message"] = message
    if was_active and log:
        _append_log_event("warning", "switch", log)
    return was_active


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
    if path == "/api/metrics/history":
        return "Metrics history"
    if path == "/api/settings":
        return "Dashboard settings"
    if path == "/api/settings/inspect":
        return "Inspect stack folder"
    if path == "/api/switch/clear":
        return "Clear stuck switch state"
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

        # Check if this is a vLLM container. vLLM stacks may use their own service
        # name, so recognize every discovered vLLM stack rather than only the
        # legacy default VLLM_SERVER_SERVICE.
        if working_dir.rstrip("/") == VLLM_WORKING_DIR_LABEL.rstrip("/"):
            all_models, _ = get_models()
            vllm_services = {
                VLLM_SERVER_SERVICE.lower(),
                *(m.get("server_service", "").lower()
                  for m in all_models.values() if m.get("vllm")),
            }
            if service.strip().lower() in vllm_services:
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

    models, _ = get_models()
    vllm_services = {
        VLLM_SERVER_SERVICE.lower(),
        *(m.get("server_service", "").lower()
          for m in models.values() if m.get("vllm")),
    }
    containers = []
    for line in output.splitlines():
        parts = line.split("|", 4)
        if len(parts) != 5:
            continue
        working_dir, config_files, service, status, name = parts
        if working_dir.rstrip("/") not in valid_dirs_list:
            continue
        # Check if this container belongs to any discovered vLLM server service.
        if service.strip().lower() in vllm_services:
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


KV_CACHE_SIZE_TOKENS_PATTERN = re.compile(r'kv_cache_size_tokens="(\d+)"')


def _parse_vllm_kv_cache_size_tokens(raw: str) -> int | None:
    """Total KV-cache capacity in tokens, from vllm:cache_config_info labels."""
    for line in raw.splitlines():
        if line.startswith("vllm:cache_config_info"):
            m = KV_CACHE_SIZE_TOKENS_PATTERN.search(line)
            if m:
                try:
                    val = int(m.group(1))
                except ValueError:
                    return None
                return val if val > 0 else None
    return None


KV_MAX_CONCURRENCY_PATTERN = re.compile(r'kv_cache_max_concurrency="([0-9.]+)"')


def _parse_vllm_kv_max_concurrency(raw: str) -> float | None:
    """How many max-length requests the KV pool can hold, from vllm:cache_config_info.

    vLLM computes this itself (pool tokens / max_model_len) and publishes it as a label,
    so there is no need to recompute it here and get the hybrid-allocator arithmetic wrong.
    """
    for line in raw.splitlines():
        if line.startswith("vllm:cache_config_info"):
            m = KV_MAX_CONCURRENCY_PATTERN.search(line)
            if m:
                try:
                    val = float(m.group(1))
                except ValueError:
                    return None
                return val if val > 0 else None
    return None


def _sum_labeled_metric(raw: str, name: str) -> float | None:
    """Sum a counter across all its label sets.

    _parse_prometheus_metrics() keys on the bare metric name, so for a metric published
    once per label value (request_success_total{finished_reason=...},
    num_requests_waiting_by_reason{reason=...}) it keeps only whichever line came last.
    Anything that needs the TOTAL has to re-walk the payload.
    """
    prefix = f"vllm:{name}{{"
    total = None
    for line in raw.splitlines():
        if line.startswith(prefix):
            val = _parse_prometheus_value(line)
            if val is not None:
                total = val if total is None else total + val
    return total


def _labeled_metric_map(raw: str, name: str, label: str) -> dict:
    """{label_value: counter} for a metric published once per label value."""
    prefix = f"vllm:{name}{{"
    pat = re.compile(label + r'="([^"]*)"')
    out = {}
    for line in raw.splitlines():
        if line.startswith(prefix):
            m = pat.search(line)
            val = _parse_prometheus_value(line)
            if m and val is not None:
                out[m.group(1)] = out.get(m.group(1), 0.0) + val
    return out


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

    # Prefill / decode phase time (vLLM reports these per finished request)
    prefill_time_sum = get_metric("request_prefill_time_seconds_sum")
    prefill_time_count = get_metric("request_prefill_time_seconds_count")

    # ⭐ THE HONEST "AVERAGE RATE" FOR vLLM. request_time_per_output_token_seconds is
    # vLLM's own per-request mean inter-token time, so count/sum is the true average output
    # rate across every completed request — not, as the llama.cpp-shaped accounting did, an
    # average of instantaneous delta-TPS readings taken at whatever moments the request
    # counter happened to tick. Those readings are sampled irregularly and skewed by
    # whichever polls landed mid-prefill.
    tpot_sum = get_metric("request_time_per_output_token_seconds_sum")
    tpot_count = get_metric("request_time_per_output_token_seconds_count")
    avg_tpot_sec = tpot_sum / tpot_count if tpot_count > 0 else None
    avg_output_rate_tps = (1.0 / avg_tpot_sec) if avg_tpot_sec and avg_tpot_sec > 0 else None

    # Mean prefill throughput over all completed requests, same treatment: total prompt
    # tokens accounted for divided by the total time vLLM spent in its prefill phase.
    avg_prefill_rate_tps = (
        prompt_total / prefill_time_sum
        if prefill_time_sum and prefill_time_sum > 0 and prompt_total > 0
        else None
    )

    # Cache usage. vLLM renamed gpu_cache_usage_perc -> kv_cache_usage_perc in 0.10, and
    # get_metric() defaults missing metrics to 0 — which is why the context widget read a
    # hard 0 on modern vLLM. Keep None when neither name is present so "unknown" and
    # "genuinely empty" stay distinguishable.
    gpu_cache_usage = None
    for _usage_name in ("kv_cache_usage_perc", "gpu_cache_usage_perc"):
        for _key in (f"vllm:{_usage_name}", _usage_name):
            if _key in parsed:
                gpu_cache_usage = float(parsed[_key])
                break
        if gpu_cache_usage is not None:
            break

    # Total tokens the KV cache can hold, from the cache_config_info labels.
    kv_cache_size_tokens = _parse_vllm_kv_cache_size_tokens(raw)
    kv_max_concurrency = _parse_vllm_kv_max_concurrency(raw)

    # ── Prefix caching. THE prefill lever on a box that offloads KV to host RAM: a cold
    # prefill runs ~1,100-1,250 tok/s here while a cache hit effectively runs ~12,000, so
    # hit rate is the single number that predicts perceived latency. Two tiers:
    #   prefix_cache_*          = GPU-resident blocks
    #   external_prefix_cache_* = the host-RAM offload tier (--kv-offloading-size)
    prefix_q = get_metric("prefix_cache_queries_total")
    prefix_h = get_metric("prefix_cache_hits_total")
    prefix_hit_rate = (prefix_h / prefix_q) if prefix_q > 0 else None
    ext_q = get_metric("external_prefix_cache_queries_total")
    ext_h = get_metric("external_prefix_cache_hits_total")
    ext_hit_rate = (ext_h / ext_q) if ext_q > 0 else None

    # Share of all prompt tokens that were served from cache instead of prefilled. This is
    # the saving in the units that matter (tokens not computed), independent of hit COUNT.
    prompt_cached = get_metric("prompt_tokens_cached_total")
    prompt_cached_frac = (prompt_cached / prompt_total) if prompt_total > 0 else None

    # Preemptions: the engine evicting a running request because the pool ran out. Each one
    # is a re-prefill later, so a climbing count means the pool is too small for the traffic
    # — exactly the trade max_num_seqs and max_num_batched_tokens are balanced against.
    preemptions = get_metric("num_preemptions_total")

    # Mean tokens per engine step. With speculation off this is effectively the batch size,
    # so it says whether concurrency is actually being used or requests are arriving serially.
    # (avg_iter_tokens below is the same figure; kept named for the detail panel.)

    # End-to-end request latency, and inter-token latency as vLLM measures it.
    e2e_sum = get_metric("e2e_request_latency_seconds_sum")
    e2e_count = get_metric("e2e_request_latency_seconds_count")
    avg_e2e_sec = e2e_sum / e2e_count if e2e_count > 0 else None
    itl_sum = get_metric("inter_token_latency_seconds_sum")
    itl_count = get_metric("inter_token_latency_seconds_count")
    avg_itl_sec = itl_sum / itl_count if itl_count > 0 else None

    # ── Speculative decoding. ABSENT ENTIRELY when --speculative-config is off, which is
    # why every field here stays None rather than 0: "no drafter" and "drafter accepting
    # nothing" must not look the same. Acceptance collapsing to 0% while the drafter still
    # burns a forward pass per position is a real failure mode on this box (measured
    # 2026-08-23: 0.0% on prefix-cache-hit long-context requests, 14.5 tok/s instead of ~45).
    spec_drafts = get_metric("spec_decode_num_drafts_total")
    spec_draft_tokens = get_metric("spec_decode_num_draft_tokens_total")
    spec_accepted = get_metric("spec_decode_num_accepted_tokens_total")
    spec_present = "vllm:spec_decode_num_draft_tokens_total" in parsed
    spec_accept_rate = (spec_accepted / spec_draft_tokens) if spec_present and spec_draft_tokens > 0 else None
    spec_accept_len = (1.0 + spec_accepted / spec_drafts) if spec_present and spec_drafts > 0 else None

    # KV offload traffic, so the host-RAM tier is visible as more than a config flag.
    offload_read_bytes = get_metric("kv_offload_load_bytes_total") or _sum_labeled_metric(raw, "kv_offload_total_bytes_total") or 0.0
    offload_store_bytes = get_metric("kv_offload_store_bytes_total") or 0.0
    offload_cpu_usage = get_metric("kv_offload_cpu_cache_usage_perc")

    # Terminal states, summed across finished_reason labels, plus the queue-reason split.
    success_by_reason = _labeled_metric_map(raw, "request_success_total", "finished_reason")
    waiting_by_reason = _labeled_metric_map(raw, "num_requests_waiting_by_reason", "reason")

    # Live scheduler occupancy. These are GAUGES, not counters — they are the only two
    # numbers here that describe the engine RIGHT NOW rather than a lifetime average, so
    # they must never be served from a stale cache without saying so.
    # _parse_prometheus_metrics() already ingests every metric in the payload, so these
    # were being parsed and discarded before 2026-08-23.
    requests_running = get_metric("num_requests_running")
    requests_waiting = get_metric("num_requests_waiting")

    # Tokens actually resident in the pool. usage is a fraction OF THE POOL, not of one
    # context window — multiplying it by max_model_len (what the context card used to do)
    # reports a number that is neither.
    kv_tokens_used = None
    if gpu_cache_usage is not None and kv_cache_size_tokens:
        kv_tokens_used = int(round(gpu_cache_usage * kv_cache_size_tokens))

    # Compute live TPS (delta-based).
    #
    # ⚠️ THE BASELINE MUST NOT ADVANCE ON EVERY SCRAPE. It used to, and with a 1 s metrics
    # cache TTL that made the window ~1 s wide — short enough that a window containing no
    # completed decode step reported a hard 0.0 tok/s while the engine was visibly
    # generating (observed 2026-08-23: 0.0, 0.0, 48.6 on consecutive polls under steady
    # load). Hold the baseline until at least VLLM_TPS_MIN_WINDOW_SEC has passed, and serve
    # the last good value in between, so the card reads steady instead of strobing.
    now = time.time()
    with VLLM_TPS_STATE_LOCK:
        prev_container = VLLM_TPS_STATE.get("container")
        prev_sampled = VLLM_TPS_STATE.get("sampled_at", 0)
        prev_gen = VLLM_TPS_STATE.get("generation_tokens")
        last_tps = VLLM_TPS_STATE.get("last_tps")

        live_tps = None
        advance = True
        if (prev_container == active_key and prev_gen is not None and
                prev_sampled > 0 and prev_gen <= gen_total):
            dt = now - prev_sampled
            if dt >= VLLM_TPS_MIN_WINDOW_SEC:
                live_tps = (gen_total - prev_gen) / dt
                VLLM_TPS_STATE["last_tps"] = live_tps
            else:
                # Too soon to measure: reuse the last real number and KEEP the baseline.
                live_tps = last_tps
                advance = False

        if advance:
            VLLM_TPS_STATE["container"] = active_key
            VLLM_TPS_STATE["sampled_at"] = now
            VLLM_TPS_STATE["generation_tokens"] = gen_total

    # Compute live ingest TPS.
    # Divide the prompt-token delta by the delta of vLLM's own PREFILL phase time, not by
    # wall time between polls. prompt_tokens_total jumps by the whole prompt the moment a
    # request is accounted for, so wall-clock division reports six-figure "ingest" rates
    # whenever the poll window happens to be shorter than the prefill it contains.
    with VLLM_INGEST_STATE_LOCK:
        prev_ingest_container = VLLM_INGEST_STATE.get("container")
        prev_ingest_sampled = VLLM_INGEST_STATE.get("sampled_at", 0)
        prev_ingest = VLLM_INGEST_STATE.get("prompt_tokens")
        prev_prefill_seconds = VLLM_INGEST_STATE.get("prefill_seconds")

        live_ingest_tps = None
        if (prev_ingest_container and prev_ingest is not None and
                prev_ingest_sampled > 0 and prev_ingest <= prompt_total):
            d_tokens = prompt_total - prev_ingest
            d_prefill = None
            if prev_prefill_seconds is not None and prefill_time_sum >= prev_prefill_seconds:
                d_prefill = prefill_time_sum - prev_prefill_seconds
            if d_tokens > 0 and d_prefill and d_prefill > 0:
                live_ingest_tps = d_tokens / d_prefill

        # Same baseline-holding rule as the decode TPS above: a sub-window with no
        # prefill in it must not overwrite the reference point, or long stretches of pure
        # decode reset the ingest reading to nothing.
        if live_ingest_tps is not None:
            VLLM_INGEST_STATE["last_ingest_tps"] = live_ingest_tps
        elif (now - prev_ingest_sampled) < VLLM_TPS_MIN_WINDOW_SEC:
            live_ingest_tps = VLLM_INGEST_STATE.get("last_ingest_tps")
        if (now - prev_ingest_sampled) >= VLLM_TPS_MIN_WINDOW_SEC or prev_ingest is None:
            VLLM_INGEST_STATE["container"] = active_key
            VLLM_INGEST_STATE["sampled_at"] = now
            VLLM_INGEST_STATE["prompt_tokens"] = prompt_total
            VLLM_INGEST_STATE["prefill_seconds"] = prefill_time_sum if prefill_time_count > 0 else None

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
        "kv_cache_size_tokens": kv_cache_size_tokens,
        "kv_tokens_used": kv_tokens_used,
        "kv_max_concurrency": kv_max_concurrency,
        "requests_running": requests_running,
        "requests_waiting": requests_waiting,
        "avg_tpot_sec": avg_tpot_sec,
        "avg_output_rate_tps": avg_output_rate_tps,
        "avg_prefill_rate_tps": avg_prefill_rate_tps,
        "avg_e2e_sec": avg_e2e_sec,
        "avg_itl_sec": avg_itl_sec,
        "prefix_hit_rate": prefix_hit_rate,
        "prefix_queries": prefix_q or None,
        "external_prefix_hit_rate": ext_hit_rate,
        "prompt_tokens_cached": prompt_cached or None,
        "prompt_cached_frac": prompt_cached_frac,
        "preemptions": preemptions,
        "spec_present": spec_present,
        "spec_accept_rate": spec_accept_rate,
        "spec_accept_len": spec_accept_len,
        "spec_draft_tokens": spec_draft_tokens if spec_present else None,
        "offload_read_bytes": offload_read_bytes or None,
        "offload_store_bytes": offload_store_bytes or None,
        "offload_cpu_usage_perc": offload_cpu_usage,
        "success_by_reason": success_by_reason or None,
        "waiting_by_reason": waiting_by_reason or None,
        "completed_requests": tpot_count or gen_tokens_count,
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
            "kv_cache_size_tokens": parsed["kv_cache_size_tokens"],
            "kv_tokens_used": parsed["kv_tokens_used"],
            "kv_max_concurrency": parsed["kv_max_concurrency"],
            "requests_running": parsed["requests_running"],
            "requests_waiting": parsed["requests_waiting"],
            "avg_tpot_sec": parsed["avg_tpot_sec"],
            "avg_output_rate_tps": parsed["avg_output_rate_tps"],
            "avg_prefill_rate_tps": parsed["avg_prefill_rate_tps"],
            "completed_requests": parsed["completed_requests"],
            # Everything below is vLLM-only detail for the engine panel. Passed through
            # wholesale so adding a metric means touching parse_vllm_metrics() and the
            # frontend only — never this dict and the cache replay separately again.
            **{k: parsed[k] for k in (
                "avg_e2e_sec", "avg_itl_sec", "prefix_hit_rate", "prefix_queries",
                "external_prefix_hit_rate", "prompt_tokens_cached", "prompt_cached_frac",
                "preemptions", "spec_present", "spec_accept_rate", "spec_accept_len",
                "spec_draft_tokens", "offload_read_bytes", "offload_store_bytes",
                "offload_cpu_usage_perc", "success_by_reason", "waiting_by_reason",
            )},
        }

    with VLLM_METRICS_CACHE_LOCK:
        VLLM_METRICS_CACHE["container"] = container
        VLLM_METRICS_CACHE["checked_at"] = now
        if code == 0:
            VLLM_METRICS_CACHE["raw"] = output
            # ⭐ STORE THE WHOLE RESULT, NOT A HAND-PICKED SUBSET.
            # Until 2026-08-23 this wrote 5 keys while _build_vllm_throughput_from_cache()
            # read 12, so every cached read returned None for avg TTFT, avg queue time, avg
            # tokens per request AND reported total_requests_approx = 0. With a 1 s TTL and
            # two builder calls per /api/status poll, the cached path is the COMMON path, so
            # the dashboard showed those fields as empty essentially always. The original
            # code even said so in a comment ("I should probably just store the parsed
            # dict"). This is that fix — do not go back to enumerating fields here.
            VLLM_METRICS_CACHE["result"] = result
        else:
            VLLM_METRICS_CACHE["raw"] = None
            VLLM_METRICS_CACHE["result"] = None

    return result


def _build_vllm_throughput_from_cache(container: str) -> dict:
    """Replay the last live scrape's result.

    Every field is served verbatim from the stored result, so a cached read and a live read
    are indistinguishable except for `state`. See the STORE THE WHOLE RESULT note above for
    why this is not a field-by-field rebuild any more.

    Two things are deliberately NOT replayed:
      * `requests_running` / `requests_waiting` are GAUGES describing the engine right now.
        Replaying a value up to VLLM_METRICS_CACHE_TTL_SEC old is fine for a rate or a
        lifetime average, but a stale "3 streams running" is a lie about the present, so
        the freshness of the reading is published alongside it as `gauge_age_sec` and the
        frontend can decide.
      * `state`, which becomes "cached" so callers can tell.
    """
    cache = VLLM_METRICS_CACHE
    stored = cache.get("result")
    if not stored:
        # No successful scrape yet on this container. Say so rather than inventing zeroes:
        # a 0 here is indistinguishable from a genuinely idle engine.
        return {
            "tokens_per_second": None,
            "ingest_tps": None,
            "source": "prometheus",
            "updated_at": now_iso(),
            "container": container,
            "state": "collecting",
            "detail": "No vLLM metrics scraped yet",
            "generation_tokens_total": 0,
            "prompt_tokens_total": 0,
            "avg_queue_time_sec": None,
            "avg_ttft_sec": None,
            "avg_iter_tokens": None,
            "avg_gen_per_request": None,
            "avg_prompt_per_request": None,
            "total_requests_approx": 0,
            "gpu_cache_usage_perc": None,
            "kv_cache_size_tokens": None,
            "kv_tokens_used": None,
            "kv_max_concurrency": None,
            "requests_running": None,
            "requests_waiting": None,
        }

    result = dict(stored)
    result["state"] = "cached"
    result["detail"] = "Cached vLLM Prometheus metrics"
    result["updated_at"] = now_iso()
    result["container"] = container
    result["gauge_age_sec"] = max(0.0, time.time() - cache.get("checked_at", 0.0))
    return result


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


# Common short English words: this tokenizes at ~1.0 tokens/word on the models we run,
# so word count is a usable proxy for prompt token count without a /tokenize round trip.
BENCHMARK_FILLER_WORDS = (
    "time person year way day thing man world life hand part child eye woman place "
    "work week case point government company number group problem fact water room "
    "mother area money story month book job word business issue side kind head house "
    "service friend father power hour game line end member law car city name team "
    "minute idea kid body back parent face level office door health art war history "
    "party result change morning reason research girl guy moment air teacher force "
    "education river market winter copper wire table light rain ship coin train field "
    "stone bridge glass paper metal wind cloud road window garden summer letter voice "
    "number season picture street corner branch circle island valley"
).split()

BENCHMARK_TOKENS_PER_FILLER_WORD = 1.0

# Instruction wrapper is ~75 tokens; subtract it so the requested prompt size is roughly honoured.
BENCHMARK_PROMPT_OVERHEAD_TOKENS = 75

BENCHMARK_TASK_INSTRUCTION = (
    "TASK: Write a long, original, non-repetitive technical essay on the history of "
    "memory hierarchies in computer architecture: caches, TLBs, NUMA, HBM. "
    "Never repeat a sentence.\n\nESSAY:\n"
)


def build_benchmark_prompt(estimated_tokens: int, seed: int = 0, nonce: str = "") -> str:
    """Build a benchmark prompt that produces *representative* generation work.

    A prompt of one repeated token (the previous implementation) is a pathological
    best case for speculative decoding: the drafter predicts every token, acceptance
    hits ~100%, and reported generation throughput lands several times above what any
    real request sees. Likewise, padding the prompt with coherent prose makes the model
    copy that prose back, which the drafter also predicts perfectly.

    So: high-entropy filler the model is told to ignore, plus a real instruction that
    forces novel text. Deterministic per (estimated_tokens, seed) so runs are comparable.
    """
    pad_tokens = max(0, estimated_tokens - BENCHMARK_PROMPT_OVERHEAD_TOKENS)
    word_count = max(16, int(round(pad_tokens / BENCHMARK_TOKENS_PER_FILLER_WORD)))
    rnd = random.Random(seed * 7919 + estimated_tokens)
    filler = " ".join(rnd.choice(BENCHMARK_FILLER_WORDS) for _ in range(word_count))
    # The nonce sits at the very front so no block of the prompt can be served from a
    # prefix cache. A cached prefill measures cache lookup speed, not ingest speed.
    prefix = f"RUN-ID {nonce}\n" if nonce else ""
    return (
        prefix
        + "You are a throughput benchmark harness. The block below is opaque filler. "
        "Do not read, repeat, quote, or refer to it.\n\nFILLER:\n"
        + filler
        + "\n\n"
        + BENCHMARK_TASK_INSTRUCTION
    )


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


def _vllm_phase_snapshot(container: str, port: int, auth_args: list[str]) -> dict | None:
    """Read the vLLM Prometheus counters the benchmark needs, or None if unreadable."""
    code, out = run_command([
        "docker", "exec", container, "curl", "-fsS", "--max-time", "5", *auth_args,
        f"http://127.0.0.1:{port}/metrics",
    ])
    if code != 0:
        return None
    parsed = _parse_prometheus_metrics(out)

    def g(name: str) -> float:
        return float(parsed.get(f"vllm:{name}", parsed.get(name, 0)))

    return {
        "prefill_sum": g("request_prefill_time_seconds_sum"),
        "prefill_count": g("request_prefill_time_seconds_count"),
        "decode_sum": g("request_decode_time_seconds_sum"),
        "decode_count": g("request_decode_time_seconds_count"),
        "cache_queries": g("prefix_cache_queries_total"),
        "cache_hits": g("prefix_cache_hits_total"),
    }


def run_single_benchmark(
    active_key: str,
    containers: list[dict],
    *,
    prompt_tokens: int,
    n_predict: int,
    seed: int = 0,
) -> tuple[dict | None, str | None]:
    container = find_model_server_container_name(active_key, containers)
    if not container:
        if _key_is_vllm(active_key):
            container = find_vllm_server_container_name()
        if not container:
            return None, "Active model server container not found"

    is_vllm = _key_is_vllm(active_key)

    port = _get_server_port(active_key)

    auth_args = []
    if is_vllm:
        if VLLM_API_KEY_FILE:
            try:
                api_key = Path(VLLM_API_KEY_FILE).read_text(encoding="utf-8").strip()
                if api_key:
                    auth_args = ["-H", f"Authorization: Bearer {api_key}"]
            except OSError:
                pass
        # Fetch valid model ID from vLLM
        model_id = "qwen"
        code_model, out_model = run_command([
            "docker", "exec", container, "curl", "-s", *auth_args,
            f"http://127.0.0.1:{port}/v1/models",
        ])
        if code_model == 0:
            try:
                models_data = json.loads(out_model)
                if "data" in models_data and len(models_data["data"]) > 0:
                    model_id = models_data["data"][0]["id"]
            except Exception:
                pass

        # vLLM OpenAI completions API.
        # ignore_eos + min_tokens pin the generated token count to n_predict so every run
        # measures the same amount of decode work. Sampling params are deliberately left
        # to the server's own configuration so the benchmark reflects real serving settings.
        payload = {
            "model": model_id,
            "prompt": build_benchmark_prompt(prompt_tokens, seed, nonce=uuid.uuid4().hex[:16]),
            "max_tokens": n_predict,
            "min_tokens": n_predict,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True}
        }
        endpoint = f"http://127.0.0.1:{port}/v1/completions"
    else:
        # llama.cpp / BeeLlama completions API.
        # cache_prompt=False defeats the prompt cache so prefill is actually measured;
        # ignore_eos pins predicted_n to n_predict. Sampling is left to the server config.
        payload = {
            "prompt": build_benchmark_prompt(prompt_tokens, seed),
            "n_predict": n_predict,
            "ignore_eos": True,
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
        *auth_args,
        "-d",
        json.dumps(payload),
    ]
    pre_snapshot = _vllm_phase_snapshot(container, port, auth_args) if is_vllm else None

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
        
        # Re-parse output for usage block.
        # NOT a regex: vLLM's usage stopped being flat once the launcher passed
        # --enable-prompt-tokens-details (upstream qwen38-27b-rtx3090 bb739e4,
        # 2026-09-01), which nests prompt_tokens_details inside it. The old
        # r'"usage":\s*({[^}]+})' stopped at the FIRST closing brace, captured
        # unbalanced JSON, and every run failed with "usage missing".
        # Walk the SSE chunks instead and keep the last one that carries usage.
        for _line in output.splitlines():
            _line = _line.strip()
            if not _line.startswith("data:"):
                continue
            _payload = _line[5:].strip()
            if not _payload or _payload == "[DONE]":
                continue
            try:
                _chunk = json.loads(_payload)
            except Exception:
                continue
            if isinstance(_chunk, dict) and _chunk.get("usage"):
                usage = _chunk["usage"]
        
        if not usage:
            # Fallback to common choice-based usage if available
            return None, f"Benchmark usage missing from vLLM stream response. Output: {output[:100]}"
        
        prompt_n = usage.get("prompt_tokens", prompt_tokens)
        gen_n = usage.get("completion_tokens", n_predict)

        # Phase times come from vLLM's own PREFILL/DECODE histograms, not from curl.
        # %{time_starttransfer} is the first *byte* of the response, and vLLM flushes SSE
        # headers the moment the stream opens — long before the first token — so using it
        # as TTFT made prefill look ~100x faster than it is (six-figure ingest T/S).
        prefill_sec = None
        decode_sec = None
        cache_hit_ratio = None
        timing_source = "vllm_phase_metrics"
        post_snapshot = None
        if pre_snapshot:
            # The histograms are only updated when the request finishes, which can land
            # just after the response body does — retry briefly before giving up.
            for _ in range(6):
                post_snapshot = _vllm_phase_snapshot(container, port, auth_args)
                if post_snapshot and post_snapshot["prefill_count"] > pre_snapshot["prefill_count"]:
                    break
                time.sleep(0.25)
        if pre_snapshot and post_snapshot:
            d_prefill_count = post_snapshot["prefill_count"] - pre_snapshot["prefill_count"]
            d_decode_count = post_snapshot["decode_count"] - pre_snapshot["decode_count"]
            # Only trust the deltas if exactly our request finished in the window;
            # concurrent traffic would fold someone else's phases into ours.
            if d_prefill_count == 1:
                val = post_snapshot["prefill_sum"] - pre_snapshot["prefill_sum"]
                if val > 0:
                    prefill_sec = val
            if d_decode_count == 1:
                val = post_snapshot["decode_sum"] - pre_snapshot["decode_sum"]
                if val > 0:
                    decode_sec = val
            d_queries = post_snapshot["cache_queries"] - pre_snapshot["cache_queries"]
            d_hits = post_snapshot["cache_hits"] - pre_snapshot["cache_hits"]
            if d_queries > 0:
                cache_hit_ratio = d_hits / d_queries

        if prefill_sec is None:
            # Fallback: wall duration minus measured decode, else the old TTFT estimate.
            timing_source = "wall_clock_fallback"
            if decode_sec is not None and duration > decode_sec:
                prefill_sec = duration - decode_sec
            elif ttft > 0:
                prefill_sec = ttft
        if decode_sec is None:
            timing_source = "wall_clock_fallback"
            if prefill_sec is not None and duration > prefill_sec:
                decode_sec = duration - prefill_sec

        prefill_tps = prompt_n / prefill_sec if prefill_sec and prefill_sec > 0 else 0.0
        gen_tps = gen_n / decode_sec if decode_sec and decode_sec > 0 else 0.0

        return (
            {
                "container": container,
                "prefill_tps": prefill_tps,
                "gen_tps": gen_tps,
                "prompt_tokens": prompt_n,
                "gen_tokens": gen_n,
                # Expose the same ms fields llama.cpp gives us so the full-profile
                # aggregate can weight every run the same way.
                "prompt_ms": (prefill_sec or 0.0) * 1000.0,
                "gen_ms": (decode_sec or 0.0) * 1000.0,
                "is_vllm": True,
                "duration_sec": duration,
                "ttft_sec": ttft,
                "prefix_cache_hit_ratio": cache_hit_ratio,
                "timing_source": timing_source,
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

    # Speculative-decoding acceptance rate. A run sitting near 100% means the drafter is
    # predicting the output almost perfectly, which inflates gen_tps far above what real
    # traffic sees — surface it so an unrepresentative benchmark is visible, not silent.
    draft_n = timings.get("draft_n")
    draft_accepted = timings.get("draft_n_accepted")
    draft_acceptance = None
    if isinstance(draft_n, (int, float)) and draft_n > 0 and isinstance(draft_accepted, (int, float)):
        draft_acceptance = float(draft_accepted) / float(draft_n)

    return (
        {
            "container": container,
            "prefill_tps": prefill_tps,
            "gen_tps": gen_tps,
            "prompt_tokens": timings.get("prompt_n"),
            "gen_tokens": timings.get("predicted_n"),
            "prompt_ms": timings.get("prompt_ms"),
            "gen_ms": timings.get("predicted_ms"),
            "draft_n": draft_n,
            "draft_n_accepted": draft_accepted,
            "draft_acceptance": draft_acceptance,
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
                    seed=rep,
                )
                if error:
                    return None, f"Full benchmark failed on {spec['name']} (run {rep + 1}/{repeats}): {error}"
                run["name"] = f"{spec['name']}#{rep + 1}"
                run["requested_prompt_tokens"] = spec["prompt_tokens"]
                run["requested_n_predict"] = spec["n_predict"]
                runs.append(run)

        prompt_tokens_total = sum(int(r.get("prompt_tokens") or 0) for r in runs)
        gen_tokens_total = sum(int(r.get("gen_tokens") or 0) for r in runs)
        prompt_ms_total = sum(float(r.get("prompt_ms") or 0.0) for r in runs)
        gen_ms_total = sum(float(r.get("gen_ms") or 0.0) for r in runs)

        # Token-weighted, not a mean of per-run rates. The specs span 32 to 2048 generated
        # tokens; averaging their rates lets the shortest, noisiest runs count as much as
        # the long ones. Total tokens over total time is the throughput that actually held.
        if prompt_ms_total > 0 and prompt_tokens_total > 0:
            prefill_avg = prompt_tokens_total / (prompt_ms_total / 1000.0)
        else:
            prefill_avg = sum(float(r["prefill_tps"]) for r in runs) / len(runs)
        if gen_ms_total > 0 and gen_tokens_total > 0:
            gen_avg = gen_tokens_total / (gen_ms_total / 1000.0)
        else:
            gen_avg = sum(float(r["gen_tps"]) for r in runs) / len(runs)

        draft_n_total = sum(int(r.get("draft_n") or 0) for r in runs)
        draft_accepted_total = sum(int(r.get("draft_n_accepted") or 0) for r in runs)
        draft_acceptance = (draft_accepted_total / draft_n_total) if draft_n_total > 0 else None

        return (
            {
                "profile": "full",
                "container": runs[0]["container"],
                "prefill_tps": prefill_avg,
                "gen_tps": gen_avg,
                "prompt_tokens": prompt_tokens_total,
                "gen_tokens": gen_tokens_total,
                "prompt_ms": prompt_ms_total,
                "gen_ms": gen_ms_total,
                "draft_n": draft_n_total or None,
                "draft_n_accepted": draft_accepted_total or None,
                "draft_acceptance": draft_acceptance,
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
        switch_started_ts = time.time()
        try:
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
                    STATE["last_exit_code"] = exit_code
                    STATE["last_output"] = output[-4000:]
                _clear_switch_state(f"Switch failed for {model['label']}")
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
                    _clear_switch_state(f"Ready: {model['label']}")
                    _append_log_event("info", "switch", f"Switch complete: {model['label']} is ready")
                    return

                # A crash during load is terminal — stop waiting instead of showing
                # "Loading..." for the rest of SWITCH_READY_TIMEOUT_SEC.
                fatal = _fatal_load_event_since(switch_started_ts)
                if fatal is not None:
                    _clear_switch_state(f"Load failed for {model['label']}: {fatal['message'][:120]}")
                    _append_log_event(
                        "error", "switch",
                        f"Load aborted for {model['label']}: {fatal['category']}: {fatal['message'][:160]}",
                    )
                    return

                progress = _loading_progress_message_since(switch_started_ts, model["label"])
                if progress:
                    _set_last_message(progress)

                time.sleep(SWITCH_POLL_SEC)

            _clear_switch_state(f"Load timeout for {model['label']}")
            _append_log_event("error", "switch", f"Switch timeout: {model['label']} did not become healthy within {SWITCH_READY_TIMEOUT_SEC}s")
        except Exception as exc:
            # Without this the thread dies silently and switch_in_progress stays True forever,
            # which locks every model button in the UI.
            _clear_switch_state(f"Switch error for {model['label']}: {exc}")
            _append_log_event("error", "switch", f"Switch worker crashed for {model['label']}: {exc}")
        finally:
            _clear_switch_state()

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
        switch_started_ts = time.time()
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
                    _clear_switch_state(f"Ready: {label}")
                    _append_log_event("info", "switch", f"Ready: {label}")
                    return

                fatal = _fatal_load_event_since(switch_started_ts)
                if fatal is not None:
                    _clear_switch_state(f"Load failed for {label}: {fatal['message'][:120]}")
                    _append_log_event(
                        "error", "switch",
                        f"Load aborted for {label}: {fatal['category']}: {fatal['message'][:160]}",
                    )
                    return

            _clear_switch_state(f"{label} load timeout")
            _append_log_event("error", "switch", f"{label} load timeout")

        except Exception as exc:
            _append_log_event("error", "switch", f"vLLM switch error: {exc}")
            _clear_switch_state(f"vLLM switch error: {exc}")
        finally:
            _clear_switch_state()

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
        switch_started_ts = time.time()
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

            bee_label = model.get("label", "BeeLlama")
            deadline = time.time() + SWITCH_READY_TIMEOUT_SEC
            while time.time() < deadline:
                local_containers = list_llama_compose_containers()
                if is_model_server_healthy(model_key, local_containers):
                    _clear_switch_state(f"Ready: {bee_label}")
                    _append_log_event("info", "switch", f"Switch complete: {bee_label} is ready")
                    return

                fatal = _fatal_load_event_since(switch_started_ts)
                if fatal is not None:
                    _clear_switch_state(f"Load failed for {bee_label}: {fatal['message'][:120]}")
                    _append_log_event(
                        "error", "switch",
                        f"Load aborted for {bee_label}: {fatal['category']}: {fatal['message'][:160]}",
                    )
                    return

                time.sleep(SWITCH_POLL_SEC)

            _clear_switch_state(f"Load timeout for {bee_label}")
            _append_log_event("error", "switch", "BeeLlama load timeout")
        except Exception as e:
            _clear_switch_state(f"BeeLlama switch error: {e}")
            _append_log_event("error", "switch", f"BeeLlama switch error: {e}")
        finally:
            _clear_switch_state()

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


def _metrics_connect(readonly: bool = False):
    if readonly:
        return sqlite3.connect(f"file:{METRICS_DB}?mode=ro", uri=True, timeout=5.0)
    return sqlite3.connect(METRICS_DB, timeout=5.0)


def init_metrics_db() -> bool:
    """Create the samples table. Returns False if the volume is not writable."""
    global METRICS_DB_READY
    try:
        parent = os.path.dirname(METRICS_DB)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with METRICS_DB_LOCK:
            conn = _metrics_connect()
            try:
                # WAL so the sampler's writes never block a dashboard read.
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS samples (
                        ts       INTEGER PRIMARY KEY,
                        util     REAL,
                        vram_pct REAL,
                        vram_mb  REAL,
                        tps      REAL,
                        ingest   REAL,
                        power    REAL,
                        temp     REAL,
                        fan      REAL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS benchmarks (
                        id               INTEGER PRIMARY KEY AUTOINCREMENT,
                        ts               INTEGER NOT NULL,
                        profile          TEXT,
                        model_key        TEXT,
                        model_label      TEXT,
                        success          INTEGER,
                        prefill_tps      REAL,
                        gen_tps          REAL,
                        draft_acceptance REAL,
                        duration_sec     REAL,
                        error            TEXT,
                        started_at       TEXT,
                        completed_at     TEXT
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_benchmarks_model "
                    "ON benchmarks (model_key, success, ts DESC)"
                )
                conn.commit()
            finally:
                conn.close()
        METRICS_DB_READY = True
        return True
    except Exception as exc:  # noqa: BLE001
        _append_log_event("warning", "metrics", f"Metrics history disabled: {exc}")
        METRICS_DB_READY = False
        return False


def note_live_throughput(tps, ingest) -> None:
    """Cache the newest throughput reading for the sampler.

    Throughput costs a docker exec to measure; the status endpoint already pays for it
    on every poll, so the sampler reuses that instead of doubling the load.
    """
    with METRICS_LIVE_LOCK:
        METRICS_LIVE["ts"] = time.time()
        METRICS_LIVE["tps"] = float(tps) if isinstance(tps, (int, float)) else None
        METRICS_LIVE["ingest"] = float(ingest) if isinstance(ingest, (int, float)) else None


def record_metric_sample() -> None:
    gpus = get_gpu_stats()
    if not gpus:
        return
    gpu = gpus[0]

    with METRICS_LIVE_LOCK:
        fresh = (time.time() - METRICS_LIVE["ts"]) <= METRICS_LIVE_MAX_AGE_SEC
        tps = METRICS_LIVE["tps"] if fresh else None
        ingest = METRICS_LIVE["ingest"] if fresh else None

    mem_total = gpu.get("mem_total") or 0
    mem_used = gpu.get("mem_used") or 0
    vram_pct = (mem_used / mem_total * 100.0) if mem_total > 0 else None

    row = (
        int(time.time()),
        gpu.get("util"),
        vram_pct,
        mem_used or None,
        tps,
        ingest,
        gpu.get("power"),
        gpu.get("temp"),
        gpu.get("fan"),
    )
    with METRICS_DB_LOCK:
        conn = _metrics_connect()
        try:
            # One row per wall-clock second; REPLACE keeps a restart from colliding.
            conn.execute(
                "INSERT OR REPLACE INTO samples "
                "(ts, util, vram_pct, vram_mb, tps, ingest, power, temp, fan) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                row,
            )
            conn.commit()
        finally:
            conn.close()


def prune_metrics() -> None:
    cutoff = int(time.time() - METRICS_RETENTION_DAYS * 86400)
    with METRICS_DB_LOCK:
        conn = _metrics_connect()
        try:
            conn.execute("DELETE FROM samples WHERE ts < ?", (cutoff,))
            conn.commit()
        finally:
            conn.close()


def read_metrics_history(range_key: str) -> dict:
    """Bucket-averaged samples for a range, capped at METRICS_MAX_POINTS points."""
    span = METRICS_RANGES.get(range_key)
    if span is None:
        return {"error": "Unknown range", "points": []}
    if not METRICS_DB_READY or not os.path.exists(METRICS_DB):
        return {"error": "Metrics history unavailable", "points": [], "range": range_key}

    bucket = max(int(METRICS_SAMPLE_SEC), span // METRICS_MAX_POINTS)
    since = int(time.time()) - span
    try:
        conn = _metrics_connect(readonly=True)
        try:
            conn.execute("PRAGMA query_only=ON")
            rows = conn.execute(
                """
                SELECT (ts / ?) * ?        AS b,
                       AVG(util)           AS util,
                       AVG(vram_pct)       AS vram_pct,
                       AVG(power)          AS power,
                       AVG(temp)           AS temp,
                       AVG(fan)            AS fan,
                       AVG(tps)            AS tps,
                       MAX(tps)            AS tps_max,
                       AVG(ingest)         AS ingest,
                       MAX(ingest)         AS ingest_max
                FROM samples
                WHERE ts >= ?
                GROUP BY b
                ORDER BY b
                """,
                (bucket, bucket, since),
            ).fetchall()
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc), "points": [], "range": range_key}

    rnd = lambda v: round(v, 2) if isinstance(v, (int, float)) else None  # noqa: E731
    points = [
        {
            "t": int(r[0]),
            "util": rnd(r[1]),
            "vram_pct": rnd(r[2]),
            "power": rnd(r[3]),
            "temp": rnd(r[4]),
            "fan": rnd(r[5]),
            "tps": rnd(r[6]),
            "tps_max": rnd(r[7]),
            "ingest": rnd(r[8]),
            "ingest_max": rnd(r[9]),
        }
        for r in rows
    ]
    return {
        "range": range_key,
        "span_sec": span,
        "bucket_sec": bucket,
        "points": points,
        "error": None,
    }


def save_benchmark_run(entry: dict) -> None:
    """Persist one benchmark history entry. Benchmarks used to live only in memory,
    so every container rebuild wiped the record."""
    if not METRICS_DB_READY:
        return
    try:
        with METRICS_DB_LOCK:
            conn = _metrics_connect()
            try:
                conn.execute(
                    "INSERT INTO benchmarks (ts, profile, model_key, model_label, success, "
                    "prefill_tps, gen_tps, draft_acceptance, duration_sec, error, "
                    "started_at, completed_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        int(time.time()),
                        entry.get("profile"),
                        entry.get("model_key"),
                        entry.get("model_label"),
                        1 if entry.get("success") else 0,
                        entry.get("prefill_tps"),
                        entry.get("gen_tps"),
                        entry.get("draft_acceptance"),
                        entry.get("duration_sec"),
                        entry.get("error"),
                        entry.get("started_at"),
                        entry.get("completed_at"),
                    ),
                )
                conn.commit()
            finally:
                conn.close()
    except Exception as exc:  # noqa: BLE001
        _append_log_event("warning", "benchmark", f"Could not persist benchmark run: {exc}")


_BENCHMARK_ROW_COLUMNS = (
    "profile", "model_key", "model_label", "success", "prefill_tps", "gen_tps",
    "draft_acceptance", "duration_sec", "error", "started_at", "completed_at",
)


def load_benchmark_history(limit: int = FULL_BENCHMARK_MAX_HISTORY) -> list[dict]:
    if not METRICS_DB_READY or not os.path.exists(METRICS_DB):
        return []
    try:
        conn = _metrics_connect(readonly=True)
        try:
            conn.execute("PRAGMA query_only=ON")
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"SELECT {','.join(_BENCHMARK_ROW_COLUMNS)} FROM benchmarks "
                "ORDER BY ts DESC LIMIT ?",
                (limit,),
            ).fetchall()
        finally:
            conn.close()
    except Exception:  # noqa: BLE001
        return []
    # Stored newest-first; the in-memory history is oldest-first.
    out = []
    for r in reversed(rows):
        d = dict(r)
        d["success"] = bool(d["success"])
        out.append(d)
    return out


def model_benchmark_summary() -> dict:
    """Most recent successful benchmark per model, for the sidebar cards."""
    if not METRICS_DB_READY or not os.path.exists(METRICS_DB):
        return {}
    try:
        conn = _metrics_connect(readonly=True)
        try:
            conn.execute("PRAGMA query_only=ON")
            rows = conn.execute(
                """
                SELECT b.model_key, b.profile, b.prefill_tps, b.gen_tps,
                       b.draft_acceptance, b.ts
                FROM benchmarks b
                JOIN (
                    SELECT model_key, MAX(ts) AS ts
                    FROM benchmarks
                    WHERE success = 1 AND model_key IS NOT NULL
                    GROUP BY model_key
                ) latest
                  ON latest.model_key = b.model_key AND latest.ts = b.ts
                WHERE b.success = 1
                """
            ).fetchall()
        finally:
            conn.close()
    except Exception:  # noqa: BLE001
        return {}
    return {
        r[0]: {
            "profile": r[1],
            "prefill_tps": r[2],
            "gen_tps": r[3],
            "draft_acceptance": r[4],
            "ts": r[5],
        }
        for r in rows
    }


def restore_benchmark_history() -> None:
    rows = load_benchmark_history()
    if not rows:
        return
    with BENCHMARK_LOCK:
        BENCHMARK_STATE["history"] = rows[-FULL_BENCHMARK_MAX_HISTORY:]


def _run_metrics_sampler() -> None:
    last_prune = 0.0
    while True:
        try:
            record_metric_sample()
            now = time.time()
            if now - last_prune > METRICS_PRUNE_EVERY_SEC:
                prune_metrics()
                last_prune = now
        except Exception as exc:  # noqa: BLE001
            _append_log_event("warning", "metrics", f"Metrics sampler error: {exc}")
        time.sleep(METRICS_SAMPLE_SEC)


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
            "max_seqs": m.get("max_seqs", ""),
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
    if _key_is_vllm(active_key):
        # ⭐ vLLM KEEPS ITS OWN LIFETIME AVERAGES — USE THEM.
        # build_model_stats() accumulates "average rate" and "completed runs" the llama.cpp
        # way: it watches a completion key change and averages the instantaneous delta-TPS
        # reading it happened to hold at that moment. On vLLM that produced a number frozen
        # for hours (17.39 tok/s observed 2026-08-23 while the engine served 44-49) and a
        # run count unrelated to actual requests. vLLM publishes request_time_per_output_
        # token_seconds and the request histograms, which are exact. Prefer them, and fall
        # back to the accumulated values only where vLLM has nothing yet.
        vllm_rate = throughput.get("avg_output_rate_tps")
        if vllm_rate is not None:
            model_stats["average_rate_tps"] = vllm_rate
        vllm_done = throughput.get("completed_requests")
        if vllm_done:
            model_stats["completed_count"] = int(vllm_done)
        if throughput.get("avg_prefill_rate_tps") is not None:
            model_stats["avg_prefill_rate_tps"] = throughput["avg_prefill_rate_tps"]
        # Surface the scheduler gauges and pool figures where the frontend already looks.
        for _k in ("requests_running", "requests_waiting", "avg_ttft_sec",
                   "avg_queue_time_sec", "avg_tpot_sec", "kv_max_concurrency",
                   "avg_e2e_sec", "avg_itl_sec", "avg_iter_tokens", "avg_gen_per_request",
                   "avg_prompt_per_request", "prefix_hit_rate", "external_prefix_hit_rate",
                   "prompt_cached_frac", "preemptions", "spec_present", "spec_accept_rate",
                   "spec_accept_len", "offload_read_bytes", "offload_store_bytes",
                   "offload_cpu_usage_perc", "success_by_reason", "waiting_by_reason",
                   "kv_tokens_used", "kv_cache_size_tokens", "generation_tokens_total",
                   "prompt_tokens_total", "total_requests_approx", "state", "gauge_age_sec"):
            model_stats[_k] = throughput.get(_k)
    # Hand the freshly-measured throughput to the metrics sampler so it never has to
    # run its own docker exec to get the same numbers.
    note_live_throughput(
        model_stats.get("live_tps"),
        model_stats.get("last_live_ingest_tps") or model_stats.get("last_ingest_tps"),
    )

    context_info = {"n_ctx": None, "n_past": None}
    if _key_is_vllm(active_key):
        try:
            m = models.get(active_key) or {}
            ctx_size = int(m.get("ctx_size", 32768))
            context_info["n_ctx"] = ctx_size
            usage = throughput.get("gpu_cache_usage_perc")
            kv_tokens = throughput.get("kv_cache_size_tokens")

            # ⚠️ POOL RESIDENCY IS NOT CONTEXT-WINDOW USAGE, AND THIS USED TO CONFLATE THEM.
            # `usage` is a fraction OF THE WHOLE KV POOL, which on this box holds 383,434
            # tokens spread over up to 16 concurrent requests — more than one 262,144-token
            # window. The old code did min(ctx_size, usage * kv_tokens) and labelled the
            # result "% of window", so a healthy multi-request pool at 40% read as a single
            # request 104k tokens into its context. Observed climbing 41k -> 104k under
            # ordinary concurrent traffic 2026-08-23.
            #
            # Publish the two as SEPARATE facts:
            #   pool_*  = what the pool holds right now, out of its real capacity
            #   n_past  = a per-request figure, which vLLM does NOT expose as a gauge; the
            #             honest stand-in is its own average prompt+generation per request.
            context_info["pool_tokens"] = kv_tokens
            context_info["pool_used"] = throughput.get("kv_tokens_used")
            context_info["pool_usage_perc"] = usage
            context_info["pool_max_concurrency"] = throughput.get("kv_max_concurrency")
            context_info["requests_running"] = throughput.get("requests_running")
            context_info["requests_waiting"] = throughput.get("requests_waiting")

            avg_prompt = throughput.get("avg_prompt_per_request")
            avg_gen = throughput.get("avg_gen_per_request")
            if avg_prompt is not None:
                context_info["n_past"] = min(ctx_size, int(round(avg_prompt + (avg_gen or 0))))
                context_info["n_past_is_average"] = True
            else:
                # No completed request yet on this boot — leave it unknown rather than
                # substituting a pool number that means something else.
                context_info["n_past"] = None
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
                    stale_entry = {
                        "profile": BENCHMARK_STATE.get("profile", "balanced"),
                        "model_key": active_key,
                        "model_label": (models.get(active_key) or {}).get("label") or active_key,
                        "started_at": BENCHMARK_STATE.get("started_at"),
                        "completed_at": BENCHMARK_STATE.get("completed_at"),
                        "duration_sec": age,
                        "success": False,
                        "error": BENCHMARK_STATE["last_error"],
                    }
                    BENCHMARK_STATE["history"].append(stale_entry)
                    save_benchmark_run(stale_entry)
                    BENCHMARK_STATE["history"] = BENCHMARK_STATE["history"][-FULL_BENCHMARK_MAX_HISTORY:]

    with BENCHMARK_LOCK:
        benchmark = dict(BENCHMARK_STATE)

    bench_by_model = model_benchmark_summary()

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
                # Latest successful benchmark for this model, so the sidebar card can
                # show measured speeds without the model being loaded.
                "bench": bench_by_model.get(key),
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
      --bg: #06080f;
      --bg-glow-1: rgba(59,130,246,0.10);
      --bg-glow-2: rgba(16,185,129,0.07);
      --sidebar-bg: #0a0e18;
      --card: #0e1320;
      --card-hi: #131a2b;
      --card-active: #1b2438;
      --border: #1b2334;
      --border-strong: #2b364c;
      --text: #f4f7fb;
      --text-muted: #7c8798;
      --text-dim: #a7b2c4;
      --accent: #4f8dfb;
      --accent-soft: rgba(79,141,251,0.14);
      --success: #21c98a;
      --warning: #f5a524;
      --danger: #f2555a;
      --sidebar-w: 268px;
      --header-h: 58px;
      --radius: 14px;
      --radius-sm: 9px;
      --shadow: 0 1px 2px rgba(0,0,0,0.4), 0 8px 24px -12px rgba(0,0,0,0.7);
      --shadow-lg: 0 24px 60px -20px rgba(0,0,0,0.85);
      --ease: cubic-bezier(0.4, 0, 0.2, 1);
      --font-mono: ui-monospace, 'JetBrains Mono', 'SF Mono', 'Fira Code', 'Roboto Mono', monospace;
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
      /* 6 cards on llama.cpp, 9 on vLLM (KV Pool, Streams and Prefix Cache are vLLM-only).
         auto-fit + minmax covers both without a second set of breakpoints. The floor is
         142px, not 165px: at 165px a 1488px row fits exactly 8 columns, which left the
         ninth vLLM card orphaned on a second line. 9 x 142 + 8 x 14 gap = 1390px, so all
         nine sit on one line at this width and still wrap cleanly on narrower screens. */
      grid-template-columns: repeat(auto-fit, minmax(142px, 1fr));
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
      /* Order: stats → switch status → GPU chart & details. The request log is a
         fixed bottom drawer, not part of this flow. */
      .main-content { display: flex; flex-direction: column; }
      .main-content > * { flex: 0 0 auto; }   /* don't let flex shrink panels below content height */
      .stats-row { order: 0; }
      .switch-status-bar { order: 1; }
      .content-grid { order: 2; }
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

    /* ══════════════════════════════════════════════════════════════════════
       Visual refresh. Layered on top of the rules above so the structural
       CSS stays readable and every selector here is an intentional override.
       ══════════════════════════════════════════════════════════════════════ */

    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
      /* Two soft light sources keep the flat near-black from reading as dead space. */
      background:
        radial-gradient(1100px 620px at 12% -12%, var(--bg-glow-1), transparent 62%),
        radial-gradient(900px 520px at 92% 0%, var(--bg-glow-2), transparent 58%),
        var(--bg);
      background-attachment: fixed;
    }

    /* Numbers that change every second must not reflow their own width. */
    .stat-value, .info-value, .event-ts, .reqlog-time, td, .bh-metric, .switch-elapsed {
      font-variant-numeric: tabular-nums;
      font-feature-settings: "tnum" 1;
    }

    /* Scrollbars: Firefox + WebKit, consistent everywhere. */
    * { scrollbar-width: thin; scrollbar-color: var(--border-strong) transparent; }
    ::-webkit-scrollbar { width: 9px; height: 9px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 99px; border: 2px solid transparent; background-clip: content-box; }
    ::-webkit-scrollbar-thumb:hover { background: #3d4a63; background-clip: content-box; }

    /* One visible keyboard focus treatment for every interactive element. */
    a:focus-visible, button:focus-visible, input:focus-visible,
    select:focus-visible, summary:focus-visible, .model-row:focus-visible {
      outline: 2px solid var(--accent);
      outline-offset: 2px;
      border-radius: var(--radius-sm);
    }

    /* ── Header ── */
    .app-header {
      background: linear-gradient(180deg, rgba(16,22,36,0.92), rgba(10,14,24,0.82));
      backdrop-filter: blur(14px) saturate(160%);
      -webkit-backdrop-filter: blur(14px) saturate(160%);
      border-bottom: 1px solid rgba(255,255,255,0.06);
      padding: 0 22px;
      gap: 14px;
    }
    .brand {
      font-size: 13px; font-weight: 800; letter-spacing: 1.6px;
      background: linear-gradient(92deg, #7fb2ff, #56e0b0);
      -webkit-background-clip: text; background-clip: text;
      -webkit-text-fill-color: transparent; color: transparent;
      white-space: nowrap;
    }
    .status-badge {
      background: rgba(255,255,255,0.045);
      border: 1px solid rgba(255,255,255,0.09);
      border-radius: 999px;
      padding: 5px 13px;
      font-size: 11px; letter-spacing: 0.8px;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);
      white-space: nowrap;
    }
    .status-dot { width: 7px; height: 7px; }
    .dot-success { box-shadow: 0 0 0 3px rgba(33,201,138,0.16), 0 0 10px var(--success); }
    .dot-danger  { box-shadow: 0 0 0 3px rgba(242,85,90,0.16); }
    .dot-warning { box-shadow: 0 0 0 3px rgba(245,165,36,0.16); }
    .menu-toggle { border-radius: var(--radius-sm); border-color: rgba(255,255,255,0.1); transition: background 0.15s var(--ease); }

    /* ── Sidebar ── */
    .sidebar { background: linear-gradient(180deg, #0b101c, #080c15); border-right: 1px solid rgba(255,255,255,0.055); }
    .nav-link { border-radius: var(--radius-sm); transition: background 0.15s var(--ease), color 0.15s var(--ease), transform 0.12s var(--ease); }
    .nav-link:hover { background: rgba(255,255,255,0.055); transform: translateX(2px); }
    .family-label { transition: color 0.15s var(--ease); }
    .family-label:hover { color: var(--text-dim); }
    .model-row {
      margin: 2px 8px;
      padding: 9px 12px;
      border-radius: 10px;
      border-left: 0;
      border: 1px solid transparent;
      transition: background 0.15s var(--ease), border-color 0.15s var(--ease), transform 0.12s var(--ease);
    }
    .model-row:hover { background: rgba(255,255,255,0.05); border-color: rgba(255,255,255,0.07); transform: translateX(2px); }
    .model-row.active {
      background: linear-gradient(90deg, rgba(33,201,138,0.16), rgba(33,201,138,0.045));
      border-color: rgba(33,201,138,0.32);
    }
    /* The removed left border used to mark the active row; a dot does it without shifting layout. */
    .model-row.active::before {
      content: ''; position: absolute; left: 4px; top: 50%; transform: translateY(-50%);
      width: 3px; height: 20px; border-radius: 99px; background: var(--success);
      box-shadow: 0 0 10px rgba(33,201,138,0.8);
    }
    .tag-pill { border-radius: 5px; padding: 1.5px 5px; letter-spacing: 0.3px; }
    .sidebar-search { border-radius: 9px; padding: 8px 11px; transition: border-color 0.15s var(--ease), background 0.15s var(--ease); }

    .btn-restart, .btn-stop-all, .btn-benchmark, .btn-benchmark-full {
      border-radius: 10px;
      padding: 9px;
      transition: background 0.15s var(--ease), transform 0.12s var(--ease), box-shadow 0.15s var(--ease);
    }
    .btn-restart:hover:not(:disabled), .btn-stop-all:hover:not(:disabled),
    .btn-benchmark:hover:not(:disabled), .btn-benchmark-full:hover:not(:disabled) {
      transform: translateY(-1px);
      box-shadow: 0 6px 16px -8px rgba(0,0,0,0.9);
    }
    .btn-restart:active:not(:disabled), .btn-stop-all:active:not(:disabled),
    .btn-benchmark:active:not(:disabled), .btn-benchmark-full:active:not(:disabled) { transform: translateY(0); }

    /* ── Surfaces ── */
    .stat-card, .chart-panel, .info-panel, .table-panel, .activity-panel {
      background: linear-gradient(168deg, var(--card-hi), var(--card));
      border: 1px solid rgba(255,255,255,0.065);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
    }
    .stat-card {
      position: relative;
      overflow: hidden;
      transition: border-color 0.18s var(--ease), transform 0.18s var(--ease);
    }
    /* Hairline of light along the top edge — reads as a lit surface, not a flat rectangle. */
    .stat-card::after {
      content: ''; position: absolute; inset: 0 0 auto 0; height: 1px;
      background: linear-gradient(90deg, transparent, rgba(255,255,255,0.16), transparent);
    }
    .stat-card:hover { border-color: rgba(255,255,255,0.12); transform: translateY(-2px); }
    .stat-label { font-size: 10px; letter-spacing: 0.9px; color: var(--text-muted); }
    .stat-value { font-size: 27px; letter-spacing: -0.6px; }
    .stat-bar-bg { height: 5px; border-radius: 99px; background: rgba(255,255,255,0.07); }
    .stat-bar-fill {
      border-radius: 99px;
      background: linear-gradient(90deg, var(--accent), #7fb2ff);
      box-shadow: 0 0 12px -3px rgba(79,141,251,0.9);
    }
    .section-title { letter-spacing: 1.1px; color: var(--text-muted); }
    .info-value { letter-spacing: -0.2px; }

    /* ── Tables ── */
    th { background: rgba(255,255,255,0.025); border-bottom: 1px solid var(--border); }
    tbody tr { transition: background 0.12s var(--ease); }
    .table-panel tbody tr:hover { background: rgba(79,141,251,0.055); }

    /* ── Switch status: a real banner, with an escape hatch when it wedges ── */
    .switch-status-bar { display: flex; align-items: center; gap: 10px; }
    .switch-status-bar.switch-active {
      background: linear-gradient(90deg, rgba(245,165,36,0.13), rgba(245,165,36,0.03));
      border: 1px solid rgba(245,165,36,0.3);
      border-radius: 11px;
      padding: 10px 14px;
      color: var(--text);
    }
    .switch-status-bar.switch-stuck {
      background: linear-gradient(90deg, rgba(242,85,90,0.13), rgba(242,85,90,0.03));
      border-color: rgba(242,85,90,0.36);
    }
    .switch-spin {
      width: 13px; height: 13px; flex-shrink: 0; border-radius: 50%;
      border: 2px solid rgba(245,165,36,0.28); border-top-color: var(--warning);
      animation: spin 0.7s linear infinite;
    }
    .switch-stuck .switch-spin { border-color: rgba(242,85,90,0.28); border-top-color: var(--danger); }
    .switch-msg { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .switch-elapsed { color: var(--text-muted); flex-shrink: 0; }
    .switch-clear {
      flex-shrink: 0;
      background: rgba(255,255,255,0.06);
      border: 1px solid rgba(255,255,255,0.16);
      color: var(--text-dim);
      font: inherit; font-size: 11px; font-weight: 700;
      padding: 5px 11px; border-radius: 8px; cursor: pointer;
      transition: background 0.15s var(--ease), color 0.15s var(--ease);
    }
    .switch-clear:hover { background: rgba(255,255,255,0.13); color: var(--text); }
    .switch-stuck .switch-clear { border-color: rgba(242,85,90,0.5); color: #ffb3b6; }

    /* ── Benchmark history ── */
    .bench-history { max-height: 190px; gap: 6px; font-family: inherit; }
    .bench-history-row {
      border: 1px solid rgba(255,255,255,0.07);
      border-radius: 10px;
      padding: 8px 10px;
      background: rgba(255,255,255,0.022);
      display: flex; flex-direction: column; gap: 5px;
      transition: border-color 0.15s var(--ease), background 0.15s var(--ease);
    }
    .bench-history-row:hover { border-color: rgba(255,255,255,0.14); background: rgba(255,255,255,0.045); }
    .bench-history-row.bh-fail { border-color: rgba(242,85,90,0.3); background: rgba(242,85,90,0.05); }
    .bh-head { display: flex; align-items: center; gap: 6px; min-width: 0; }
    .bh-profile {
      font-size: 8.5px; font-weight: 800; letter-spacing: 0.6px;
      padding: 2px 5px; border-radius: 4px; flex-shrink: 0;
      background: rgba(79,141,251,0.16); color: #8fb8ff; border: 1px solid rgba(79,141,251,0.28);
    }
    .bh-profile-full { background: rgba(33,201,138,0.16); color: #6fe3b6; border-color: rgba(33,201,138,0.3); }
    .bh-model {
      font-size: 11px; font-weight: 600; color: var(--text);
      overflow: hidden; text-overflow: ellipsis; white-space: nowrap; min-width: 0;
    }
    .bh-when { font-size: 9.5px; color: var(--text-muted); font-family: var(--font-mono); }
    .bh-metrics { display: flex; flex-wrap: wrap; gap: 4px; }
    .bh-metric {
      font-size: 10px; font-family: var(--font-mono); color: var(--text-dim);
      background: rgba(255,255,255,0.045); border-radius: 5px; padding: 2px 6px;
    }
    .bh-metric b { color: var(--text-muted); font-weight: 700; }
    .bh-error { font-size: 10px; color: #ffa9ac; font-family: var(--font-mono); word-break: break-word; white-space: normal; }

    /* ── Activity panel ── */
    .activity-tab { transition: color 0.15s var(--ease), border-color 0.15s var(--ease); }
    .activity-tab.active { border-bottom-width: 2px; }
    .tab-chip { border-radius: 999px; }
    .reqlog-input, .reqlog-refresh { border-radius: 9px; }
    .reqlog-table thead th { background: #0c1220; }
    .reqlog-ep, .status-pill { border-radius: 999px; padding: 2px 8px; }

    /* ── Modal ── */
    .modal-overlay { background: rgba(3,5,10,0.78); backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px); }
    .modal {
      background: linear-gradient(168deg, var(--card-hi), var(--card));
      border: 1px solid rgba(255,255,255,0.1);
      border-radius: 18px;
      box-shadow: var(--shadow-lg);
      transform: scale(0.97);
      transition: transform 0.2s var(--ease);
    }
    .modal-overlay.active .modal { transform: scale(1); }
    .btn { border-radius: 10px; transition: filter 0.15s var(--ease), transform 0.12s var(--ease); }
    .btn:hover { filter: brightness(1.12); transform: translateY(-1px); }
    .btn:active { transform: translateY(0); }

    @media (max-width: 600px) {
      .switch-status-bar.switch-active { flex-wrap: wrap; }
      .switch-msg { flex-basis: 100%; white-space: normal; }
      .stat-value { font-size: 21px; }
    }

    /* ══════════════════════════════════════════════════════════════════════
       Density + flair pass
       ══════════════════════════════════════════════════════════════════════ */

    /* Reclaim the outer margins; the panels already separate themselves. */
    .main-content { padding: 12px 16px 14px; gap: 10px; }
    .stats-row { gap: 10px; }

    /* An idle status bar was costing ~46px of the log's height for an empty line. */
    .switch-status-bar:empty { display: none; }

    /* The right side is one card spanning both rows: Service Details on top,
       Benchmark below, split by a hairline. Fewer seams than two stacked cards,
       and the card's natural height sets how much is left for the request log. */
    .content-grid {
      grid-template-columns: minmax(0, 1fr) 348px;
      grid-template-rows: auto minmax(152px, auto);
      grid-template-areas: "chart details" "lower details";
      gap: 10px; align-items: stretch;
    }
    .content-grid > .chart-panel { grid-area: chart; }
    .content-grid > .lower-row { grid-area: lower; }
    .content-grid > .details-bench-panel { grid-area: details; min-height: 0; }
    .info-panel { padding: 13px 15px; gap: 9px; }
    .details-bench-sep { border-top: 1px solid var(--border); margin: 4px 0 1px; }
    /* The process list is almost always a single llama-server row, so it gets a
       half-width slot and the reclaimed space goes to a second chart beside it. */
    /* min-height, not height: the panel stretches when Service Details is the
       taller cell in row one (it was 258px in a flex column, which is what
       made the right-hand seam drift). */
    .chart-panel {
      height: auto; min-height: 258px; padding: 13px 15px 10px;
      display: flex; flex-direction: column;
    }
    .chart-canvas-wrap { flex: 1; min-height: 0; position: relative; }

    /* ── Chart header + range picker ── */
    .chart-head {
      display: flex; align-items: center; justify-content: space-between;
      gap: 12px; flex-wrap: wrap; margin-bottom: 12px;
    }
    .range-picker {
      display: inline-flex; gap: 2px; padding: 2px;
      background: rgba(255,255,255,0.045);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 9px;
    }
    .range-btn {
      background: none; border: none; color: var(--text-muted);
      font: inherit; font-size: 11px; font-weight: 700; letter-spacing: 0.3px;
      padding: 4px 9px; border-radius: 7px; cursor: pointer;
      transition: background 0.15s var(--ease), color 0.15s var(--ease);
    }
    .range-btn:hover { color: var(--text-dim); background: rgba(255,255,255,0.05); }
    .range-btn.active {
      background: linear-gradient(180deg, rgba(79,141,251,0.28), rgba(79,141,251,0.16));
      color: #cfe0ff;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.12);
    }
    /* A class rule with display:flex outranks the UA's [hidden]{display:none}, so
       without this the overlay never actually hides and dims the chart permanently. */
    .chart-note[hidden] { display: none; }
    .chart-note {
      position: absolute; inset: 0;
      display: flex; align-items: center; justify-content: center;
      text-align: center; padding: 16px;
      font-size: 12px; color: var(--text-muted);
      background: rgba(10,14,24,0.72);
      border-radius: 10px;
      pointer-events: none;
    }
    .lower-row {
      display: grid;
      grid-template-columns: minmax(0, 0.82fr) minmax(0, 1.18fr);
      gap: 10px;
      min-height: 152px;
    }
    .chart-panel-sm { height: auto; }
    .table-panel {
      flex: 1 1 auto; display: flex; flex-direction: column; min-height: 0;
      container-type: inline-size;
    }
    .table-scroll { overflow: auto; flex: 1; min-height: 0; }
    /* Seven columns in a narrowed panel: trim padding and never wrap a cell. */
    .table-panel th, .table-panel td { padding: 8px 12px; white-space: nowrap; }
    .table-panel th { font-size: 9.5px; letter-spacing: 0.4px; }
    .table-panel td { font-size: 11.5px; }
    /* Below the width where all seven fit, drop the two least-load-bearing columns
       rather than introduce a horizontal scrollbar. */
    @container (max-width: 540px) {
      .table-panel th:nth-child(4), .table-panel td:nth-child(4),
      .table-panel th:nth-child(5), .table-panel td:nth-child(5) { display: none; }
    }
    @container (max-width: 420px) {
      .table-panel th:nth-child(6), .table-panel td:nth-child(6) { display: none; }
    }
    .table-header { padding: 13px 16px 9px; }

    /* ── Stat cards: per-metric identity ── */
    .stat-card { padding: 10px 13px 11px; }
    .stat-card::after { background: linear-gradient(90deg, transparent, var(--sc), transparent); opacity: 0.65; }
    .stat-card { --sc: rgba(255,255,255,0.16); }
    .sc-gen    { --sc: #4f8dfb; }
    .sc-ingest { --sc: #21c98a; }
    .sc-ctx    { --sc: #a877f7; }
    .sc-util   { --sc: #f5a524; }
    .sc-vram   { --sc: #2ed3c6; }
    .sc-temp   { --sc: #f2555a; }
    .sc-pool    { --sc: #7c8cf8; }
    .sc-streams { --sc: #f77fb5; }
    .sc-cache   { --sc: #34c3e8; }
    /* vLLM engine panel: dense two-column readouts, same visual weight as info-pair. */
    .eng-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px 14px; }
    .eng-item { min-width: 0; }
    .eng-label { font-size: 10px; font-weight: 700; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.4px; }
    .eng-value { font-size: 12.5px; font-weight: 600; line-height: 1.35; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .eng-value.warn { color: #f5a524; }
    .eng-value.bad  { color: #f2555a; }
    .eng-value.good { color: #21c98a; }
    .eng-note { font-size: 10.5px; color: var(--text-muted); line-height: 1.4; margin-top: 2px; }
    /* A faint wash in the card's own colour, strongest at the lit top edge. */
    .stat-card {
      background:
        radial-gradient(120% 90% at 50% -25%, color-mix(in srgb, var(--sc) 16%, transparent), transparent 70%),
        linear-gradient(168deg, var(--card-hi), var(--card));
    }
    .stat-card:hover { border-color: color-mix(in srgb, var(--sc) 40%, transparent); }
    .stat-value {
      font-size: 23px;
      background: linear-gradient(180deg, #ffffff, color-mix(in srgb, var(--sc) 55%, #ffffff));
      -webkit-background-clip: text; background-clip: text;
      -webkit-text-fill-color: transparent; color: transparent;
    }
    .stat-sub { font-size: 10.5px; margin-top: 1px; }
    .stat-bar-bg { margin-top: 7px; }
    .stat-bar-fill {
      background: linear-gradient(90deg, color-mix(in srgb, var(--sc) 55%, transparent), var(--sc));
      box-shadow: 0 0 12px -3px var(--sc);
    }

    /* Sparkline fills the otherwise-empty lower half of the two speed cards. */
    .stat-spark { margin-top: 6px; height: 22px; }
    .stat-spark svg { display: block; width: 100%; height: 22px; overflow: visible; }
    .stat-spark .spark-area { fill: var(--sc); opacity: 0.16; }
    .stat-spark .spark-line { fill: none; stroke: var(--sc); stroke-width: 1.6; stroke-linejoin: round; stroke-linecap: round; }
    .stat-spark .spark-dot { fill: var(--sc); }
    .stat-spark .spark-empty { fill: none; stroke: rgba(255,255,255,0.1); stroke-width: 1.4; stroke-dasharray: 3 4; }

    /* ── Header flair ── */
    .app-header { position: relative; }
    /* Slow drifting light along the header seam — the one moving thing when idle. */
    .app-header::after {
      content: ''; position: absolute; left: 0; right: 0; bottom: -1px; height: 1px;
      background: linear-gradient(90deg, transparent, var(--accent), #21c98a, transparent);
      background-size: 42% 100%; background-repeat: no-repeat;
      animation: header-sweep 9s ease-in-out infinite;
      opacity: 0.75;
    }
    @keyframes header-sweep {
      0%   { background-position: -45% 0; }
      50%  { background-position: 145% 0; }
      100% { background-position: -45% 0; }
    }
    .header-center { font-size: 12.5px; gap: 8px; }
    .dot-success { animation: pulse-ring 2.4s ease-out infinite; }
    @keyframes pulse-ring {
      0%   { box-shadow: 0 0 0 0 rgba(33,201,138,0.45), 0 0 10px var(--success); }
      70%  { box-shadow: 0 0 0 7px rgba(33,201,138,0), 0 0 10px var(--success); }
      100% { box-shadow: 0 0 0 0 rgba(33,201,138,0), 0 0 10px var(--success); }
    }

    /* ── Benchmark card ── */
    .bench-btn-row { grid-template-columns: 1fr 1fr; gap: 8px; }
    .bench-headline {
      display: flex; align-items: center; gap: 12px;
      padding: 9px 12px;
      border-radius: 11px;
      background: linear-gradient(140deg, rgba(79,141,251,0.10), rgba(33,201,138,0.07));
      border: 1px solid rgba(255,255,255,0.07);
    }
    .bench-hl-item { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 3px; }
    .bench-hl-sep { width: 1px; align-self: stretch; background: rgba(255,255,255,0.1); }
    .bench-hl-value {
      font-family: var(--font-mono); font-size: 16px; font-weight: 700;
      letter-spacing: -0.5px; color: #9cc0ff; white-space: nowrap;
    }
    .bench-hl-gen { color: #6fe3b6; }
    /* The last item (history) absorbs leftover card height and scrolls, so the
       card cannot push the request log down. */
    .details-bench-panel > .info-item:last-child { flex: 1; min-height: 0; display: flex; flex-direction: column; gap: 3px; }
    .bench-history { max-height: none; flex: 1; min-height: 84px; }

    /* ── Sidebar polish ── */
    .sidebar-nav { padding: 9px 9px 7px; }
    .model-row { padding: 8px 12px; }
    .model-name { font-size: 12.5px; }

    /* Measured speeds on the model card */
    .model-bench { display: flex; gap: 4px; margin-top: 5px; }
    .mb-pill {
      display: inline-flex; align-items: baseline; gap: 3px;
      font-family: var(--font-mono); font-size: 9.5px; font-weight: 700;
      padding: 1.5px 5px; border-radius: 5px;
      background: rgba(255,255,255,0.045);
      border: 1px solid rgba(255,255,255,0.07);
      font-variant-numeric: tabular-nums;
    }
    .mb-gen { color: #6fe3b6; border-color: rgba(33,201,138,0.24); background: rgba(33,201,138,0.09); }
    .mb-pre { color: #8fb8ff; border-color: rgba(79,141,251,0.24); background: rgba(79,141,251,0.09); }
    .mb-unit { font-size: 8px; font-weight: 600; opacity: 0.6; letter-spacing: 0.2px; }

    /* ── Fill the viewport: the grid keeps its natural height, the content
       column scrolls if it exceeds the viewport, and the request log is a fixed
       bottom drawer that no longer consumes flow space. ──
       Guarded on a tall-enough desktop window; below that the page keeps its normal
       scrolling behaviour rather than squeezing every panel into a sliver. */
    @media (min-width: 1201px) and (min-height: 760px) {
      .content-grid { flex: 0 0 auto; }
      /* Scoped to the top chart: .chart-panel-sm shares the .chart-panel class and
         must stay auto-height so it fills its grid row. */
      .content-grid > .chart-panel { height: auto; min-height: 248px; }
      .chart-panel-sm { height: auto; }
      /* The merged card spans both rows, so an unbounded benchmark list would
         inflate the whole grid; cap the list (min-height 84 above clamps it just below one row). */
      .bench-history { max-height: 76px; }
      .info-panel { gap: 7px; padding: 12px 15px; }
    }

    /* ── Wide screens: the six stat cards get roomy, so let content breathe ── */
    @media (min-width: 1700px) {
      .content-grid { grid-template-columns: minmax(0, 1fr) 380px; }
      .main-content { padding: 16px 22px 20px; }
    }
    /* These must restate grid-template-columns: this block sits after the base
       responsive rules, so an unqualified two-column value here would win at every width. */
    @media (max-width: 1200px) {
      .content-grid {
        grid-template-columns: minmax(0, 1fr);
        grid-template-rows: none;
        grid-template-areas: "chart" "lower" "details";
      }
      .chart-panel { height: 300px; }
      .chart-panel-sm { height: 260px; }
    }
    @media (max-width: 980px) {
      .lower-row { grid-template-columns: minmax(0, 1fr); }
    }
    @media (max-width: 760px) {
      .chart-panel { height: 260px; }
      .chart-panel-sm { height: 240px; }
      .table-panel { max-height: none; }
    }
    /* ── Request log drawer ──
       The activity panel is not part of the page flow: it is a fixed drawer
       the bottom bar unfolds over the lower ~2/3 of the viewport. The page
       above stays visible and interactive; the content column scrolls
       clear of the bar via padding-bottom. */
    .activity-panel {
      position: fixed;
      left: var(--sidebar-w);
      right: 0;
      bottom: 44px;
      height: calc(66vh - 44px);
      transform: translateY(calc(100% + 60px));
      transition: transform 0.22s ease;
      z-index: 300;
      margin-top: 0;
      border-bottom: none;
      border-radius: 12px 12px 0 0;
    }
    .activity-panel.open { transform: translateY(0); }
    .activity-body { flex: 1 1 auto; min-height: 0; display: flex; flex-direction: column; }
    .activity-body[hidden] { display: none; }
    .reqlog-scroll { max-height: none; flex: 1; min-height: 0; }
    .events-body { max-height: none; flex: 1; min-height: 0; }
    .log-drawer-bar {
      position: fixed;
      left: var(--sidebar-w);
      right: 0;
      bottom: 0;
      height: 44px;
      z-index: 320;
      display: flex;
      align-items: center;
      padding: 0 16px;
      background: var(--card);
      border-top: 1px solid var(--border-strong);
    }
    .log-drawer-btn {
      display: flex;
      align-items: center;
      gap: 8px;
      background: none;
      border: none;
      color: var(--text-muted);
      font: inherit;
      font-size: 13px;
      font-weight: 700;
      line-height: 1;
      padding: 9px 14px;
      border-radius: 8px;
      cursor: pointer;
    }
    .log-drawer-btn:hover { color: var(--text); background: rgba(59,130,246,0.08); }
    .log-drawer-count {
      background: rgba(59,130,246,0.15);
      border: 1px solid rgba(59,130,246,0.4);
      color: var(--accent);
      border-radius: 8px;
      padding: 0 7px;
      font-size: 11px;
      line-height: 17px;
      font-family: var(--font-mono);
    }
    .log-drawer-count:empty { display: none; }
    .main-content { padding-bottom: 64px; }
    @media (max-width: 768px) {
      .log-drawer-bar, .activity-panel { left: 0; }
    }

    /* ── Settings modal ── */
    .settings-modal { max-width: 760px; width: 94vw; max-height: 88vh; padding: 0; text-align: left; display: flex; flex-direction: column; }
    .settings-foot {
      display: flex; align-items: center; gap: 10px;
      padding: 14px 22px; border-top: 1px solid var(--border); flex-shrink: 0;
    }
    .settings-status { flex: 1; font-size: 12px; color: var(--text-muted); min-width: 0; }
    .settings-status.ok { color: var(--success); }
    .settings-status.err { color: var(--danger); }
    .settings-foot .btn { font-size: 12.5px; padding: 8px 18px; }
    .set-intro {
      font-size: 12px; color: var(--text-dim); line-height: 1.6;
      background: rgba(79,141,251,0.07); border: 1px solid rgba(79,141,251,0.18);
      border-radius: 10px; padding: 11px 13px; margin-bottom: 16px;
    }
    .set-intro code { font-family: var(--font-mono); font-size: 11px; color: var(--text); }
    .set-group { margin-bottom: 8px; }
    .set-section {
      font-size: 10px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.9px;
      color: var(--text-muted); margin: 18px 0 10px;
    }
    .set-section:first-child { margin-top: 0; }
    .set-row {
      border: 1px solid var(--border); border-radius: 11px;
      padding: 12px 13px; margin-bottom: 10px;
      background: rgba(255,255,255,0.018);
    }
    .set-row-head { display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap; margin-bottom: 3px; }
    .set-label { font-size: 12.5px; font-weight: 700; color: var(--text); }
    .set-src {
      font-size: 8.5px; font-weight: 800; letter-spacing: 0.5px; text-transform: uppercase;
      padding: 2px 6px; border-radius: 4px;
      background: rgba(255,255,255,0.06); color: var(--text-muted); border: 1px solid rgba(255,255,255,0.1);
    }
    .set-src-settings { background: rgba(79,141,251,0.16); color: #8fb8ff; border-color: rgba(79,141,251,0.3); }
    .set-src-env      { background: rgba(245,165,36,0.15); color: #f5c579; border-color: rgba(245,165,36,0.3); }
    .set-help { font-size: 11px; color: var(--text-muted); line-height: 1.5; margin-bottom: 8px; }
    .set-input {
      width: 100%; padding: 8px 11px; border-radius: 9px;
      background: var(--bg); border: 1px solid var(--border-strong);
      color: var(--text); font-family: var(--font-mono); font-size: 11.5px; outline: none;
      transition: border-color 0.15s var(--ease);
    }
    .set-input:focus { border-color: var(--accent); background: rgba(79,141,251,0.06); }
    .set-check { display: flex; align-items: center; gap: 7px; margin-top: 7px; font-size: 11px; font-family: var(--font-mono); }
    .set-check-ok   { color: #6fe3b6; }
    .set-check-warn { color: #f5c579; }
    .set-check-bad  { color: #ffa9ac; }
    .set-check-idle { color: var(--text-muted); }
    .set-envnote { font-size: 10.5px; color: var(--text-muted); margin-top: 6px; font-family: var(--font-mono); word-break: break-all; }

    /* Respect the OS setting — every transition above is decorative. */
    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
      }
    }
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
        <button class="nav-link" onclick="openSettings()"><span class="nav-ico">&#9881;</span> Settings</button>
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
        <div class="stat-card sc-gen">
          <div class="stat-label">Generation Speed</div>
          <div class="stat-value" id="val-tps">--</div>
          <div class="stat-sub" id="val-tps-sub">tokens / sec</div>
          <div class="stat-spark" id="spark-tps"></div>
        </div>
        <div class="stat-card sc-ingest">
          <div class="stat-label">Ingest Speed</div>
          <div class="stat-value" id="val-ingest">--</div>
          <div class="stat-sub" id="val-ingest-sub">tokens / sec</div>
          <div class="stat-spark" id="spark-ingest"></div>
        </div>
        <div class="stat-card sc-ctx" id="stat-ctx-card">
          <div class="stat-label" id="stat-ctx-label">Context</div>
          <div class="stat-value" id="val-ctx">-- / --</div>
          <div class="stat-sub" id="val-ctx-sub">window used</div>
          <div class="stat-bar-bg"><div id="bar-ctx" class="stat-bar-fill" style="width:0%"></div></div>
        </div>
        <div class="stat-card sc-pool" id="stat-pool-card" hidden>
          <div class="stat-label">KV Pool</div>
          <div class="stat-value" id="val-pool">-- / --</div>
          <div class="stat-sub" id="val-pool-sub">tokens resident</div>
          <div class="stat-bar-bg"><div id="bar-pool" class="stat-bar-fill" style="width:0%"></div></div>
        </div>
        <div class="stat-card sc-streams" id="stat-streams-card" hidden>
          <div class="stat-label">Streams</div>
          <div class="stat-value" id="val-streams">--</div>
          <div class="stat-sub" id="val-streams-sub">concurrent requests</div>
          <div class="stat-bar-bg"><div id="bar-streams" class="stat-bar-fill" style="width:0%"></div></div>
        </div>
        <div class="stat-card sc-cache" id="stat-cache-card" hidden>
          <div class="stat-label">Prefix Cache</div>
          <div class="stat-value" id="val-cache">--%</div>
          <div class="stat-sub" id="val-cache-sub">prompt tokens reused</div>
          <div class="stat-bar-bg"><div id="bar-cache" class="stat-bar-fill" style="width:0%"></div></div>
        </div>
        <div class="stat-card sc-util">
          <div class="stat-label">GPU Utilization</div>
          <div class="stat-value" id="val-util">--%</div>
          <div class="stat-sub" id="val-util-sub">compute load</div>
          <div class="stat-bar-bg"><div id="bar-util" class="stat-bar-fill" style="width:0%"></div></div>
        </div>
        <div class="stat-card sc-vram">
          <div class="stat-label">VRAM Usage</div>
          <div class="stat-value" id="val-vram">-- GB</div>
          <div class="stat-sub" id="val-vram-sub">of -- GB</div>
          <div class="stat-bar-bg"><div id="bar-vram" class="stat-bar-fill" style="width:0%"></div></div>
        </div>
        <div class="stat-card sc-temp">
          <div class="stat-label">Temperature</div>
          <div class="stat-value" id="val-temp">--&deg;C</div>
          <div class="stat-sub" id="val-fan">Fan: --%</div>
          <div class="stat-bar-bg"><div id="bar-temp" class="stat-bar-fill" style="width:0%"></div></div>
        </div>
      </div>

      <!-- Switch status bar -->
      <div class="switch-status-bar" id="switchStatus"></div>

      <!-- Chart + Info -->
      <div class="content-grid">
          <div class="chart-panel">
            <div class="chart-head">
              <div class="section-title" style="margin:0" id="perfChartTitle">Hardware Performance History</div>
              <div class="range-picker" id="rangePicker" role="group" aria-label="Chart time range"></div>
            </div>
            <div class="chart-canvas-wrap">
              <canvas id="historyChart"></canvas>
              <div class="chart-note" id="chartNote" hidden></div>
            </div>
          </div>

          <!-- Power/thermals chart + GPU processes share the lower band -->
          <div class="lower-row">
            <div class="chart-panel chart-panel-sm">
              <div class="section-title" id="thermalChartTitle">Power &amp; Thermals</div>
              <div class="chart-canvas-wrap"><canvas id="thermalChart"></canvas></div>
            </div>

            <div class="table-panel">
              <div class="table-header"><div class="section-title" style="margin:0">Active GPU Processes</div></div>
              <div class="table-scroll">
                <table>
                  <thead><tr><th>PID</th><th>Application</th><th>VRAM</th><th>CPU%</th><th>RAM</th><th>GPU%</th><th>Status</th></tr></thead>
                  <tbody id="procTable">
                    <tr><td colspan="7" style="text-align:center; color:var(--text-muted)">Scanning processes...</td></tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div class="info-panel details-bench-panel">
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
                <div class="info-label" id="info-avg-rate-label">Avg Rate</div>
                <div class="info-value" id="info-avg-rate">-- T/S</div>
              </div>
              <div class="info-item">
                <div class="info-label" id="info-runs-label">Completed Runs</div>
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
            <div id="engine-panel" hidden>
              <div class="details-bench-sep"></div>
              <div class="section-title">vLLM Engine</div>
              <div class="eng-grid">
                <div class="eng-item"><div class="eng-label">TTFT avg</div><div class="eng-value" id="eng-ttft">--</div></div>
                <div class="eng-item"><div class="eng-label">Queue wait avg</div><div class="eng-value" id="eng-queue">--</div></div>
                <div class="eng-item"><div class="eng-label">Inter-token / req</div><div class="eng-value" id="eng-itl">--</div></div>
                <div class="eng-item"><div class="eng-label">E2E latency avg</div><div class="eng-value" id="eng-e2e">--</div></div>
                <div class="eng-item"><div class="eng-label">Batch (tok/step)</div><div class="eng-value" id="eng-batch">--</div></div>
                <div class="eng-item"><div class="eng-label">Preemptions</div><div class="eng-value" id="eng-preempt">--</div></div>
                <div class="eng-item"><div class="eng-label">Avg prompt</div><div class="eng-value" id="eng-avgprompt">--</div></div>
                <div class="eng-item"><div class="eng-label">Avg generated</div><div class="eng-value" id="eng-avggen">--</div></div>
                <div class="eng-item"><div class="eng-label">Cache: GPU / host</div><div class="eng-value" id="eng-cachetiers">--</div></div>
                <div class="eng-item"><div class="eng-label">Spec decode</div><div class="eng-value" id="eng-spec">--</div></div>
                <div class="eng-item"><div class="eng-label">KV offload r/w</div><div class="eng-value" id="eng-offload">--</div></div>
                <div class="eng-item"><div class="eng-label">Finish reasons</div><div class="eng-value" id="eng-finish">--</div></div>
              </div>
              <div class="eng-note" id="eng-note"></div>
            </div>

            <div class="details-bench-sep"></div>

            <div class="section-title">Benchmark</div>
            <div class="bench-btn-row">
              <button class="btn-benchmark" id="btnBenchmark" onclick="triggerBenchmark('balanced')">QUICK</button>
              <button class="btn-benchmark-full" id="btnBenchmarkFull" onclick="triggerBenchmark('full')">FULL</button>
            </div>
            <div class="bench-headline">
              <div class="bench-hl-item">
                <div class="info-label">Prefill</div>
                <div class="bench-hl-value" id="info-bench-prefill">-- T/S</div>
              </div>
              <div class="bench-hl-sep"></div>
              <div class="bench-hl-item">
                <div class="info-label">Generation</div>
                <div class="bench-hl-value bench-hl-gen" id="info-bench-gen">-- T/S</div>
              </div>
            </div>
            <div class="info-item">
              <div class="info-label">Last Run</div>
              <div class="info-value" id="info-bench-last" style="font-size:11px; line-height:1.45">--</div>
            </div>
            <div class="info-item">
              <div class="info-label">History (Last 10)</div>
              <div class="bench-history" id="info-bench-history">
                <div class="bench-history-empty">No benchmark runs yet.</div>
              </div>
            </div>
          </div>
      </div>


    </div><!-- /.main-content -->
  </div><!-- /.app-body -->
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
  <!-- Request log drawer bar: unfold/fold the activity panel over the lower page -->
  <div class="log-drawer-bar" id="reqlogBar">
    <button class="log-drawer-btn" id="reqlogDrawerBtn" onclick="toggleReqlogDrawer()" aria-expanded="false" aria-controls="activityPanel">
      <span id="reqlogDrawerChevron" aria-hidden="true">&#9650;</span>
      Request Log
      <span class="log-drawer-count" id="reqlogBarCount"></span>
    </button>
  </div>

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

  <!-- Settings modal -->
  <div id="settingsModal" class="modal-overlay" onclick="if(event.target===this)closeSettings()">
    <div class="modal settings-modal">
      <div class="reqdetail-head">
        <h2>Settings</h2>
        <button class="reqdetail-close" onclick="closeSettings()" title="Close">&#10005;</button>
      </div>
      <div class="reqdetail-body" id="settingsBody">Loading…</div>
      <div class="settings-foot">
        <span class="settings-status" id="settingsStatus"></span>
        <button class="btn btn-cancel" onclick="closeSettings()">CLOSE</button>
        <button class="btn btn-confirm" id="btnSaveSettings" onclick="saveSettings()">SAVE</button>
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
    let thermalChart = null;
    let isRefreshing = false;
    let pendingAction = null;
    let lastUpdated = Date.now();
    let lastData = null;
    let eventsPanelOpen = true;
    let eventsLastCount = 0;
    let searchQuery = '';
    const history = {
      util: Array(60).fill(0), vram: Array(60).fill(0),
      tps: Array(60).fill(0), ingest: Array(60).fill(0),
      power: Array(60).fill(0), temp: Array(60).fill(0), fan: Array(60).fill(0)
    };

    // ── Chart time range ──────────────────────────────────────────────────────
    // '1m' is the live in-page buffer (sub-second resolution, no server round trip).
    // Everything longer is served from the sampler's SQLite store.
    const RANGE_OPTIONS = [
      { key: '1m',  label: '1m',  live: true },
      { key: '5m',  label: '5m'  },
      { key: '15m', label: '15m' },
      { key: '1h',  label: '1h'  },
      { key: '24h', label: '24h' },
      { key: '7d',  label: '7d'  },
      { key: '30d', label: '1mo' }
    ];
    const RANGE_LABELS = { '1m': '60s', '5m': '5 min', '15m': '15 min', '1h': '1 hour',
                           '24h': '24 hours', '7d': '7 days', '30d': '30 days' };
    let chartRange = localStorage.getItem('chartRange') || '1m';
    if (!RANGE_OPTIONS.some(o => o.key === chartRange)) chartRange = '1m';
    let historyPoints = null;      // last fetched historical payload
    let historyFetching = false;
    let historyTimer = null;

    function isLiveRange() { return chartRange === '1m'; }

    function buildRangePicker() {
      const el = document.getElementById('rangePicker');
      el.innerHTML = RANGE_OPTIONS.map(o =>
        `<button class="range-btn${o.key === chartRange ? ' active' : ''}" data-range="${o.key}"
           onclick="setChartRange('${o.key}')">${o.label}</button>`).join('');
      document.getElementById('perfChartTitle').textContent =
        `Hardware Performance History (${RANGE_LABELS[chartRange]})`;
      const t = document.getElementById('thermalChartTitle');
      if (t) t.textContent = `Power & Thermals (${RANGE_LABELS[chartRange]})`;
    }

    function setChartRange(key) {
      if (!RANGE_OPTIONS.some(o => o.key === key)) return;
      chartRange = key;
      try { localStorage.setItem('chartRange', key); } catch (e) {}
      historyPoints = null;
      buildRangePicker();
      if (historyTimer) { clearInterval(historyTimer); historyTimer = null; }
      if (isLiveRange()) {
        setChartNote(null);
        applyLiveSeries();
      } else {
        setChartNote('Loading history…');
        fetchHistory();
        // Buckets are minutes-to-hours wide past 1h; polling faster just burns queries.
        const period = (key === '5m' || key === '15m') ? 15000 : 60000;
        historyTimer = setInterval(fetchHistory, period);
      }
    }

    function setChartNote(msg) {
      const el = document.getElementById('chartNote');
      if (!el) return;
      if (!msg) { el.hidden = true; el.textContent = ''; return; }
      el.hidden = false;
      el.textContent = msg;
    }

    async function fetchHistory() {
      if (historyFetching || isLiveRange()) return;
      historyFetching = true;
      const requested = chartRange;
      try {
        const r = await fetch(`/api/metrics/history?range=${encodeURIComponent(requested)}`);
        const d = await r.json();
        // The user may have clicked another range while this was in flight.
        if (requested !== chartRange) return;
        if (d.error) { setChartNote(d.error); return; }
        if (!d.points || !d.points.length) {
          setChartNote('No samples recorded for this range yet.');
          historyPoints = null;
          return;
        }
        historyPoints = d;
        setChartNote(null);
        applyHistorySeries(d);
      } catch (e) {
        if (requested === chartRange) setChartNote(`History unavailable: ${e}`);
      } finally {
        historyFetching = false;
      }
    }

    function fmtBucketLabel(tSec, span) {
      const d = new Date(tSec * 1000);
      if (span <= 3600) return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      if (span <= 86400) return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      return d.toLocaleDateString([], { month: 'short', day: 'numeric' }) + ' ' +
             d.toLocaleTimeString([], { hour: '2-digit' });
    }

    function applyHistorySeries(d) {
      const labels = d.points.map(p => fmtBucketLabel(p.t, d.span_sec));
      chart.data.labels = labels;
      chart.data.datasets[0].data = d.points.map(p => p.util);
      chart.data.datasets[1].data = d.points.map(p => p.vram_pct);
      // Averaging generation speed across a multi-minute bucket that is mostly idle
      // reads as ~0. The bucket peak is the honest answer to "how fast did it run".
      chart.data.datasets[2].data = d.points.map(p => p.tps_max);
      chart.data.datasets[2].label = 'Gen Speed (peak T/S)';
      chart.options.scales.x.display = true;
      chart.update();

      if (thermalChart) {
        thermalChart.data.labels = labels;
        thermalChart.data.datasets[0].data = d.points.map(p => p.power);
        thermalChart.data.datasets[1].data = d.points.map(p => p.temp);
        thermalChart.data.datasets[2].data = d.points.map(p => p.fan);
        thermalChart.options.scales.x.display = true;
        thermalChart.update();
      }
    }

    function applyLiveSeries() {
      chart.data.labels = Array(60).fill('');
      chart.data.datasets[0].data = history.util;
      chart.data.datasets[1].data = history.vram;
      chart.data.datasets[2].data = history.tps;
      chart.data.datasets[2].label = 'Gen Speed (T/S)';
      chart.options.scales.x.display = false;
      chart.update();

      if (thermalChart) {
        thermalChart.data.labels = Array(60).fill('');
        thermalChart.data.datasets[0].data = history.power;
        thermalChart.data.datasets[1].data = history.temp;
        thermalChart.data.datasets[2].data = history.fan;
        thermalChart.options.scales.x.display = false;
        thermalChart.update();
      }
    }

    // Inline SVG sparkline. Cheap enough to redraw on every 2s poll and it fills the
    // dead lower half of the two speed cards with something that actually says something.
    function sparkline(values) {
      const W = 100, H = 26;
      const max = Math.max(...values);
      if (!(max > 0)) {
        return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-hidden="true">
          <line class="spark-empty" x1="0" y1="${H - 1}" x2="${W}" y2="${H - 1}"/></svg>`;
      }
      const n = values.length;
      const y = (v) => H - 1 - (v / max) * (H - 3);
      const pts = values.map((v, i) => `${((i / (n - 1)) * W).toFixed(2)},${y(v).toFixed(2)}`);
      const last = values[n - 1];
      return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-hidden="true">
        <polygon class="spark-area" points="0,${H} ${pts.join(' ')} ${W},${H}"/>
        <polyline class="spark-line" points="${pts.join(' ')}" vector-effect="non-scaling-stroke"/>
        ${last > 0 ? `<circle class="spark-dot" cx="${W - 0.8}" cy="${y(last).toFixed(2)}" r="1.8"/>` : ''}
      </svg>`;
    }

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
      // Vertical fades instead of flat fills — the three series stay readable where they overlap.
      const fade = (hex) => {
        const g = ctx.createLinearGradient(0, 0, 0, 230);
        g.addColorStop(0, hex + '44');
        g.addColorStop(1, hex + '00');
        return g;
      };
      chart = new Chart(ctx, {
        type: 'line',
        data: {
          labels: Array(60).fill(''),
          datasets: [
            { label: 'GPU Util %', data: history.util, borderColor: '#4f8dfb', backgroundColor: fade('#4f8dfb'), fill: true, tension: 0.4, pointRadius: 0, borderWidth: 2, yAxisID: 'y' },
            { label: 'VRAM %',     data: history.vram, borderColor: '#21c98a', backgroundColor: fade('#21c98a'), fill: true, tension: 0.4, pointRadius: 0, borderWidth: 2, yAxisID: 'y' },
            { label: 'Gen Speed (T/S)', data: history.tps, borderColor: '#f5a524', backgroundColor: fade('#f5a524'), fill: true, tension: 0.4, pointRadius: 0, borderWidth: 2, yAxisID: 'y1' }
          ]
        },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          interaction: { mode: 'index', intersect: false },
          scales: {
            y: {
              beginAtZero: true, max: 100,
              border: { display: false },
              grid: { color: 'rgba(255,255,255,0.045)' },
              ticks: { color: '#7c8798', font: { size: 11 }, padding: 8 }
            },
            y1: {
              beginAtZero: true,
              // Without a floor an idle GPU auto-scales the axis to 0–1, which reads as
              // a broken chart. 20 T/S is a sane empty-state ceiling.
              suggestedMax: 20,
              position: 'right',
              border: { display: false },
              grid: { drawOnChartArea: false },
              ticks: { color: '#f5a524', font: { size: 11 }, padding: 8 },
              title: { display: true, text: 'Tokens/Sec', color: '#f5a524', font: { size: 10 } }
            },
            x: {
              display: false,
              border: { display: false },
              grid: { display: false },
              ticks: {
                color: '#7c8798', font: { size: 9.5 },
                maxRotation: 0, autoSkip: true, maxTicksLimit: 7
              }
            }
          },
          plugins: {
            legend: { labels: { color: '#a7b2c4', boxWidth: 8, boxHeight: 8, usePointStyle: true, pointStyle: 'circle', font: { size: 11 }, padding: 16 } },
            tooltip: {
              backgroundColor: 'rgba(14,19,32,0.96)',
              borderColor: 'rgba(255,255,255,0.12)',
              borderWidth: 1,
              titleColor: '#f4f7fb',
              bodyColor: '#a7b2c4',
              padding: 10,
              cornerRadius: 8,
              displayColors: true,
              usePointStyle: true
            }
          }
        }
      });
    }

    function initThermalChart() {
      const el = document.getElementById('thermalChart');
      if (!el) return;
      const ctx = el.getContext('2d');
      const fade = (hex) => {
        const g = ctx.createLinearGradient(0, 0, 0, 200);
        g.addColorStop(0, hex + '3a');
        g.addColorStop(1, hex + '00');
        return g;
      };
      thermalChart = new Chart(ctx, {
        type: 'line',
        data: {
          labels: Array(60).fill(''),
          datasets: [
            { label: 'Power (W)', data: history.power, borderColor: '#a877f7', backgroundColor: fade('#a877f7'), fill: true, tension: 0.4, pointRadius: 0, borderWidth: 2, yAxisID: 'yw' },
            { label: 'Temp (°C)', data: history.temp,  borderColor: '#f2555a', backgroundColor: fade('#f2555a'), fill: true, tension: 0.4, pointRadius: 0, borderWidth: 2, yAxisID: 'y' },
            { label: 'Fan (%)',   data: history.fan,   borderColor: '#2ed3c6', backgroundColor: 'transparent',   fill: false, tension: 0.4, pointRadius: 0, borderWidth: 1.6, borderDash: [4, 3], yAxisID: 'y' }
          ]
        },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          interaction: { mode: 'index', intersect: false },
          scales: {
            // °C and % share the left axis — both are 0-100 scales and read together.
            y: {
              beginAtZero: true, max: 100,
              border: { display: false },
              grid: { color: 'rgba(255,255,255,0.045)' },
              ticks: { color: '#7c8798', font: { size: 10 }, padding: 6, stepSize: 25 }
            },
            yw: {
              beginAtZero: true, suggestedMax: 100,
              position: 'right',
              border: { display: false },
              grid: { drawOnChartArea: false },
              ticks: { color: '#c9a6ff', font: { size: 10 }, padding: 6 },
              title: { display: true, text: 'Watts', color: '#c9a6ff', font: { size: 10 } }
            },
            x: {
              display: false,
              border: { display: false },
              grid: { display: false },
              ticks: {
                color: '#7c8798', font: { size: 9.5 },
                maxRotation: 0, autoSkip: true, maxTicksLimit: 7
              }
            }
          },
          plugins: {
            legend: { labels: { color: '#a7b2c4', boxWidth: 8, boxHeight: 8, usePointStyle: true, pointStyle: 'circle', font: { size: 10.5 }, padding: 12 } },
            tooltip: {
              backgroundColor: 'rgba(14,19,32,0.96)',
              borderColor: 'rgba(255,255,255,0.12)',
              borderWidth: 1,
              titleColor: '#f4f7fb',
              bodyColor: '#a7b2c4',
              padding: 10,
              cornerRadius: 8,
              usePointStyle: true
            }
          }
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
          // Measured speeds from the last successful benchmark, persisted per model.
          let benchRow = '';
          if (m.bench && (m.bench.gen_tps || m.bench.prefill_tps)) {
            const when = m.bench.ts ? new Date(m.bench.ts * 1000).toLocaleString() : 'unknown time';
            const prof = (m.bench.profile || '').toUpperCase();
            const t = `Last ${prof} benchmark — ${when}`;
            const gen = m.bench.gen_tps ? Number(m.bench.gen_tps).toFixed(1) : '--';
            const pre = m.bench.prefill_tps ? Math.round(m.bench.prefill_tps) : '--';
            benchRow = `<div class="model-bench" title="${escHtml(t)}">
              <span class="mb-pill mb-gen">▶ ${gen}<span class="mb-unit">gen</span></span>
              <span class="mb-pill mb-pre">▼ ${pre}<span class="mb-unit">ingest</span></span>
            </div>`;
          }
          html += `<div class="${cls}" onclick="confirmSwitch('${m.key}','${escHtml(m.label)}')">
            <div class="model-row-inner">
              <div class="model-row-top">${spin}<span class="model-name">${escHtml(m.label)}</span></div>
              ${tags ? `<div class="model-tags">${tags}</div>` : ''}
              ${benchRow}
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

      // ⚠️ vLLM HAS NO llama.cpp-STYLE INGEST READING. last_ingest_tps / best_ingest_tps
      // are filled by the llama.cpp log watcher and INGEST_LIVE_STATE, which never see a
      // vLLM container, so this card read a flat 0.0 on every vLLM stack. The live
      // prompt-token delta only produces a value when a poll window happens to contain a
      // prefill, so fall back to vLLM's own lifetime mean (prompt tokens / prefill seconds).
      const isVllmStack = (data.active?.family === 'vllm');
      let ingestIsAvg = false;
      if (isVllmStack && (ingestVal === '0.0' || ingestVal === '--')) {
        const liveIng = live?.ingest_tps ?? stats.ingest_tps;
        if (liveIng != null && liveIng > 0) {
          ingestVal = liveIng.toFixed(1);
        } else if (stats.avg_prefill_rate_tps != null && stats.avg_prefill_rate_tps > 0) {
          ingestVal = stats.avg_prefill_rate_tps.toFixed(0);
          ingestIsAvg = true;
        }
      }

      document.getElementById('val-tps').textContent = genDisplay;
      document.getElementById('val-ingest').textContent = ingestVal;
      const tpsSubEl = document.getElementById('val-tps-sub');
      if (tpsSubEl) {
        // average_rate_tps on vLLM is count/sum of request_time_per_output_token_seconds —
        // the true mean output rate over completed requests, not an average of samples.
        let tpsSub = 'tokens / sec';
        if (isVllmStack) {
          // ⚠️ THESE ARE DIFFERENT UNITS AND MUST SAY SO.
          // val-tps is AGGREGATE across every running stream (delta of
          // generation_tokens_total). average_rate_tps is PER REQUEST (1 / mean
          // request_time_per_output_token_seconds). With 3 concurrent streams the first
          // reads ~95 tok/s and the second ~24 - both correct, 3 x 24 ~ 72-95. Shown
          // side by side unlabelled, the pair reads as "generation is crawling", which is
          // what prompted this fix. MEASURED 2026-08-23: one request alone gets 44.9 tok/s,
          // 35.8-36.2 with one other running, 13.4 with two others. Per-stream rate falling
          // as streams are added while aggregate rises is normal, not a regression.
          const runNow = stats.requests_running || 0;
          const perStream = (liveTps > 0 && runNow > 1) ? (liveTps / runNow) : null;
          // Build ONE trailing clause, never a chain: prefixing a state word onto a string
          // that already began with "tok/s ·" produced "idle · tok/s · 25.0 per request avg".
          const avgTxt = stats.average_rate_tps
            ? (perStream
                ? `${perStream.toFixed(1)}/stream now`
                : `${stats.average_rate_tps.toFixed(1)}/req avg`)
            : null;
          // ⚠️ 0 tok/s WITH A REQUEST RUNNING IS NOT AN IDLE ENGINE — it is a request in its
          // PREFILL phase, which emits no output tokens while pinning the GPU. Observed at
          // 95% GPU utilisation with 1 stream running and an 8.3k-token average prompt.
          // Reading that card as "the model stopped" is exactly the wrong conclusion.
          const state = (liveTps >= 0.5) ? 'tok/s aggregate'
                      : (stats.requests_running > 0 ? 'prefilling' : 'idle');
          tpsSub = avgTxt ? `${state} · ${avgTxt}` : (liveTps >= 0.5 ? 'tokens / sec' : state);
        }
        tpsSubEl.textContent = tpsSub;
      }
      const ingSubEl = document.getElementById('val-ingest-sub');
      if (ingSubEl) {
        // Say EFFECTIVE, not raw: prompt_tokens_total counts prefix-cache hits, which cost
        // almost no prefill time, so this mean is inflated far above cold prefill (measured
        // ~1,100-1,250 tok/s cold on this box vs ~5,000 effective). Do not read it as raw
        // prefill bandwidth.
        ingSubEl.textContent = ingestIsAvg ? 'tok/s · effective avg (incl. cache hits)' : 'tokens / sec';
      }
      document.getElementById('val-util').textContent = gpu.util != null ? `${gpu.util}%` : '--%';
      document.getElementById('bar-util').style.width = `${gpu.util || 0}%`;
      const vramGB = ((gpu.mem_used || 0) / 1024).toFixed(1);
      document.getElementById('val-vram').textContent = `${vramGB} GB`;
      const vramPct = gpu.mem_total > 0 ? (gpu.mem_used / gpu.mem_total) * 100 : 0;
      document.getElementById('bar-vram').style.width = `${vramPct.toFixed(1)}%`;
      document.getElementById('val-temp').textContent = gpu.temp != null ? `${gpu.temp}°C` : '--°C';
      document.getElementById('val-fan').textContent = `Fan: ${gpu.fan || 0}%`;

      document.getElementById('val-util-sub').textContent =
        gpu.util != null ? (gpu.util > 5 ? 'compute active' : 'idle') : 'compute load';
      document.getElementById('val-vram-sub').textContent =
        gpu.mem_total > 0 ? `of ${(gpu.mem_total / 1024).toFixed(1)} GB · ${vramPct.toFixed(0)}%` : 'of -- GB';
      // Temperature bar spans a useful 30–95 °C, not 0–100, so normal running is visible movement.
      const tempPct = gpu.temp != null ? Math.max(0, Math.min(100, ((gpu.temp - 30) / 65) * 100)) : 0;
      document.getElementById('bar-temp').style.width = `${tempPct.toFixed(1)}%`;

      // Context card
      const nCtx = ctxInfo.n_ctx;
      const nPast = ctxInfo.n_past;
      const ctxEl = document.getElementById('val-ctx');
      const ctxBar = document.getElementById('bar-ctx');
      const ctxSub = document.getElementById('val-ctx-sub');
      // ⚠️ For vLLM, n_past is the AVERAGE prompt+generation per request, because vLLM
      // publishes no per-request context gauge. Saying "% of window" about an average, or
      // about the old pool-derived number, was the misleading part. Label what it is.
      const ctxIsAvg = ctxInfo.n_past_is_average === true;
      const ctxLabelEl = document.getElementById('stat-ctx-label');
      if (ctxLabelEl) ctxLabelEl.textContent = ctxIsAvg ? 'Context / request' : 'Context';
      if (nCtx != null && nPast > 0) {
        const share = `${((nPast / nCtx) * 100).toFixed(1)}% of window`;
        ctxSub.textContent = ctxIsAvg ? `avg per request · ${share}` : share;
      } else if (nCtx != null) {
        ctxSub.textContent = ctxIsAvg ? 'no completed requests yet' : 'window idle';
      } else {
        ctxSub.textContent = 'window used';
      }
      if (nCtx != null) {
        const ctxMax = nCtx >= 1024 ? `${Math.round(nCtx/1024)}k` : `${nCtx}`;
        if (nPast != null && nPast > 0) {
          const ctxUsed = nPast >= 1024 ? `${(nPast/1024).toFixed(1)}k` : `${nPast}`;
          ctxEl.textContent = `${ctxUsed} / ${ctxMax}`;
          const pct = Math.min(100, (nPast / nCtx) * 100);
          ctxBar.style.width = `${pct.toFixed(1)}%`;
          // Overriding --sc keeps the fill, its glow and the card accent in sync.
          const ctxCard = document.getElementById('stat-ctx-card');
          if (pct > 90) ctxCard.style.setProperty('--sc', '#f2555a');
          else if (pct > 70) ctxCard.style.setProperty('--sc', '#f5a524');
          else ctxCard.style.removeProperty('--sc');
        } else {
          ctxEl.textContent = `0 / ${ctxMax}`;
          ctxBar.style.width = '0%';
          document.getElementById('stat-ctx-card').style.removeProperty('--sc');
        }
      } else {
        ctxEl.textContent = '-- / --';
        ctxBar.style.width = '0%';
      }

      // ── vLLM-only cards and engine panel ────────────────────────────────────────
      // Everything here is driven by model_stats, which build_status() fills from the
      // engine's own Prometheus metrics when the active stack is vLLM. For llama.cpp the
      // fields are absent, so the cards stay hidden rather than rendering zeroes.
      const isVllm = (data.active?.family === 'vllm');
      const fmtTok = (n) => n == null ? '--'
        : (n >= 1000 ? `${(n/1000).toFixed(n >= 100000 ? 0 : 1)}k` : `${Math.round(n)}`);
      const fmtSec = (v) => v == null ? '--'
        : (v < 1 ? `${(v*1000).toFixed(v < 0.1 ? 1 : 0)} ms` : `${v.toFixed(v < 10 ? 2 : 1)} s`);
      const fmtGB  = (b) => b == null ? '--' : `${(b/1073741824).toFixed(1)} GB`;
      const pct1   = (f) => f == null ? '--' : `${(f*100).toFixed(1)}%`;

      const poolCard = document.getElementById('stat-pool-card');
      const streamCard = document.getElementById('stat-streams-card');
      const cacheCard = document.getElementById('stat-cache-card');
      const engPanel = document.getElementById('engine-panel');
      poolCard.hidden = !isVllm; streamCard.hidden = !isVllm;
      cacheCard.hidden = !isVllm; engPanel.hidden = !isVllm;

      if (isVllm) {
        // KV Pool — tokens actually resident out of real pool capacity. NOT context-window
        // usage: the pool spans every concurrent request, so this and the Context card
        // measure different things on purpose.
        const poolUsed = ctxInfo.pool_used, poolTot = ctxInfo.pool_tokens;
        const poolPct = (poolUsed != null && poolTot) ? (poolUsed/poolTot)*100 : 0;
        document.getElementById('val-pool').textContent =
          (poolUsed != null && poolTot) ? `${fmtTok(poolUsed)} / ${fmtTok(poolTot)}` : '-- / --';
        const conc = ctxInfo.pool_max_concurrency;
        document.getElementById('val-pool-sub').textContent =
          (poolUsed != null && poolTot)
            ? `${poolPct.toFixed(1)}% resident${conc ? ` · ${conc.toFixed(2)}x capacity` : ''}`
            : 'tokens resident';
        document.getElementById('bar-pool').style.width = `${Math.min(100, poolPct).toFixed(1)}%`;
        if (poolPct > 90) poolCard.style.setProperty('--sc', '#f2555a');
        else if (poolPct > 75) poolCard.style.setProperty('--sc', '#f5a524');
        else poolCard.style.removeProperty('--sc');

        // Streams — running now, out of the slot cap from the stack's LLM_META max_seqs.
        // Queue depth matters more than the running count on this box: capacity-waiting
        // with free slots means the prefill token budget is the bottleneck, not the slots.
        const run = stats.requests_running, wait = stats.requests_waiting;
        const maxSeqs = parseInt(data.active?.max_seqs || '0', 10) || null;
        document.getElementById('val-streams').textContent =
          run == null ? '--' : (maxSeqs ? `${Math.round(run)} / ${maxSeqs}` : `${Math.round(run)}`);
        const wr = stats.waiting_by_reason || {};
        let waitTxt = 'concurrent requests';
        if (wait != null && wait > 0) {
          const why = (wr.capacity > 0 && wr.deferred > 0) ? 'capacity + deferred'
                    : (wr.capacity > 0 ? 'waiting on capacity'
                    : (wr.deferred > 0 ? 'deferred' : 'queued'));
          waitTxt = `${Math.round(wait)} ${why}`;
        } else if (run != null) {
          waitTxt = run > 0 ? 'no queue' : 'idle';
        }
        document.getElementById('val-streams-sub').textContent = waitTxt;
        const sPct = (run != null && maxSeqs) ? Math.min(100, (run/maxSeqs)*100) : 0;
        document.getElementById('bar-streams').style.width = `${sPct.toFixed(1)}%`;
        if (wait > 0) streamCard.style.setProperty('--sc', '#f5a524');
        else streamCard.style.removeProperty('--sc');

        // Prefix Cache — headline is the share of PROMPT TOKENS served from cache, which is
        // the saving in the unit that matters. Hit-rate percentages of the two tiers go in
        // the sub-line; a hit on a tiny block and a hit on 100k tokens count the same there.
        const frac = stats.prompt_cached_frac;
        document.getElementById('val-cache').textContent = frac == null ? '--%' : pct1(frac);
        document.getElementById('val-cache-sub').textContent =
          (stats.prefix_hit_rate == null && stats.external_prefix_hit_rate == null)
            ? 'prompt tokens reused'
            : `GPU ${pct1(stats.prefix_hit_rate)} · host ${pct1(stats.external_prefix_hit_rate)}`;
        document.getElementById('bar-cache').style.width =
          `${frac == null ? 0 : Math.min(100, frac*100).toFixed(1)}%`;

        // ── engine panel ──
        const ttft = stats.avg_ttft_sec, queue = stats.avg_queue_time_sec;
        const ttftEl = document.getElementById('eng-ttft');
        ttftEl.textContent = fmtSec(ttft);
        ttftEl.className = 'eng-value' + (ttft == null ? '' : (ttft > 10 ? ' bad' : (ttft > 3 ? ' warn' : ' good')));
        const qEl = document.getElementById('eng-queue');
        qEl.textContent = fmtSec(queue);
        // Queue as a share of TTFT is the actionable number: if most of the wait is queue,
        // the fix is scheduling/token budget, not model or quantisation.
        const qShare = (ttft && queue != null && ttft > 0) ? queue/ttft : null;
        qEl.className = 'eng-value' + (qShare == null ? '' : (qShare > 0.5 ? ' bad' : (qShare > 0.2 ? ' warn' : ' good')));
        document.getElementById('eng-itl').textContent = fmtSec(stats.avg_itl_sec ?? stats.avg_tpot_sec);
        document.getElementById('eng-e2e').textContent = fmtSec(stats.avg_e2e_sec);
        document.getElementById('eng-batch').textContent =
          stats.avg_iter_tokens == null ? '--' : stats.avg_iter_tokens.toFixed(1);
        const pre = stats.preemptions;
        const preEl = document.getElementById('eng-preempt');
        preEl.textContent = pre == null ? '--' : Math.round(pre).toLocaleString();
        preEl.className = 'eng-value' + (pre == null ? '' : (pre > 0 ? ' warn' : ' good'));
        document.getElementById('eng-avgprompt').textContent = fmtTok(stats.avg_prompt_per_request) + ' tok';
        document.getElementById('eng-avggen').textContent = fmtTok(stats.avg_gen_per_request) + ' tok';
        document.getElementById('eng-cachetiers').textContent =
          `${pct1(stats.prefix_hit_rate)} / ${pct1(stats.external_prefix_hit_rate)}`;

        // Spec decode: "off" and "on but accepting nothing" are completely different
        // situations and must never render identically. spec_present is false when the
        // engine was started without --speculative-config at all.
        const specEl = document.getElementById('eng-spec');
        if (!stats.spec_present) {
          specEl.textContent = 'off';
          specEl.className = 'eng-value';
        } else {
          const ar = stats.spec_accept_rate, al = stats.spec_accept_len;
          specEl.textContent = ar == null ? 'on' :
            `${pct1(ar)} accept${al ? ` · ${al.toFixed(2)}x` : ''}`;
          // 0% acceptance means the drafter burns a forward pass per position for nothing.
          specEl.className = 'eng-value' + (ar == null ? '' : (ar < 0.05 ? ' bad' : (ar < 0.2 ? ' warn' : ' good')));
        }
        document.getElementById('eng-offload').textContent =
          (stats.offload_read_bytes == null && stats.offload_store_bytes == null) ? '--'
          : `${fmtGB(stats.offload_read_bytes)} / ${fmtGB(stats.offload_store_bytes)}`;
        const sbr = stats.success_by_reason || {};
        const finish = Object.entries(sbr).filter(([, v]) => v > 0)
          .sort((a, b) => b[1] - a[1]).map(([k, v]) => `${k} ${Math.round(v)}`).join(' · ');
        document.getElementById('eng-finish').textContent = finish || '--';

        // One plain-language line naming the current bottleneck, if there is one.
        const notes = [];
        if (wait > 0 && wr.capacity > 0 && maxSeqs && run != null && run < maxSeqs) {
          notes.push(`${Math.round(wait)} request(s) waiting on capacity with ${maxSeqs - Math.round(run)} slot(s) free — the prefill token budget is the limit, not max_num_seqs.`);
        }
        if (qShare != null && qShare > 0.5) {
          notes.push(`${pct1(qShare)} of TTFT is queue wait.`);
        }
        if (pre > 0) notes.push(`${Math.round(pre)} preemption(s): the KV pool is being exhausted.`);
        if (stats.spec_present && stats.spec_accept_rate != null && stats.spec_accept_rate < 0.05) {
          notes.push('Speculative decoding is accepting ~nothing — it is pure overhead right now.');
        }
        document.getElementById('eng-note').textContent = notes.join(' ');
      }

      // History chart
      const tpsVal = stats.live_tps || 0;
      const ingestLive = (live?.state !== 'idle' && liveIngest > 0) ? liveIngest : 0;
      history.util.push(gpu.util || 0); history.util.shift();
      history.vram.push(vramPct); history.vram.shift();
      history.tps.push(tpsVal); history.tps.shift();
      history.ingest.push(ingestLive); history.ingest.shift();
      history.power.push(gpu.power || 0); history.power.shift();
      history.temp.push(gpu.temp || 0); history.temp.shift();
      history.fan.push(gpu.fan || 0); history.fan.shift();
      // Track the card's real ceiling so the watts axis isn't dwarfed by a 280W limit.
      if (thermalChart && gpu.power_limit > 0) {
        thermalChart.options.scales.yw.suggestedMax = gpu.power_limit;
      }
      // In a historical range the charts are driven by fetchHistory, not by this poll.
      if (isLiveRange()) {
        chart.update();
        if (thermalChart) thermalChart.update();
      }

      document.getElementById('spark-tps').innerHTML = sparkline(history.tps);
      document.getElementById('spark-ingest').innerHTML = sparkline(history.ingest);

      // Service details
      const infoModel = document.getElementById('info-model');
      const infoBadges = document.getElementById('info-badges');
      const infoAvgRate = document.getElementById('info-avg-rate');
      const infoRuns = document.getElementById('info-runs');

      if (data.active) {
        infoModel.textContent = data.active.label;
        // On vLLM these two are exact engine figures, not accumulated samples — rename them
        // so nobody reads "Avg Rate" as the llama.cpp average-of-completed-runs it used to be.
        const rateLabel = document.getElementById('info-avg-rate-label');
        const runsLabel = document.getElementById('info-runs-label');
        // Name the unit: this is a PER-REQUEST mean, not the aggregate on the speed card.
        if (rateLabel) rateLabel.textContent = (data.active.family === 'vllm') ? 'Avg Rate / Request' : 'Avg Rate';
        if (runsLabel) runsLabel.textContent = (data.active.family === 'vllm') ? 'Requests Served' : 'Completed Runs';
        let badges = '';
        if (data.active.params) badges += `<span class="badge badge-ctx">${escHtml(data.active.params)}</span>`;
        if (data.active.family === 'vllm' && data.active.max_seqs) {
          badges += `<span class="badge badge-ctx">${escHtml(data.active.max_seqs)} slots</span>`;
        }
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
        btnBenchmark.textContent = (bench.in_progress && bench.profile === 'balanced') ? 'RUNNING…' : 'QUICK';
      }
      if (btnBenchmarkFull) {
        btnBenchmarkFull.disabled = !canRun;
        btnBenchmarkFull.textContent = (bench.in_progress && bench.profile === 'full') ? 'RUNNING…' : 'FULL';
      }

      const benchRes = bench.last_result || {};
      infoBenchPrefill.textContent = benchRes.prefill_tps ? `${benchRes.prefill_tps.toFixed(2)} T/S` : '-- T/S';
      infoBenchGen.textContent = benchRes.gen_tps ? `${benchRes.gen_tps.toFixed(2)} T/S` : '-- T/S';
      if (bench.in_progress && bench.started_at) {
        infoBenchLast.textContent = `${(bench.profile || 'balanced').toUpperCase()} running since ${fmtLocalTs(bench.started_at)}`;
      } else if (bench.last_error) {
        infoBenchLast.textContent = `Failed: ${bench.last_error}`;
      } else if (benchRes.completed_at) {
        const who = benchRes.model_label ? ` · ${benchRes.model_label}` : '';
        infoBenchLast.textContent = `${(benchRes.profile || 'balanced').toUpperCase()}${who} at ${fmtLocalTs(benchRes.completed_at)}`;
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
          // Older entries predate model_label; fall back to the key so they still identify a model.
          const model = r.model_label || r.model_key || 'unknown model';
          const head = `<div class="bh-head">
              <span class="bh-profile bh-profile-${profile.toLowerCase()}">${profile}</span>
              <span class="bh-model" title="${escHtml(model)}">${escHtml(model)}</span>
            </div>
            <div class="bh-when">${escHtml(when)}</div>`;
          if (!r.success) {
            return `<div class="bench-history-row bh-fail">${head}
              <div class="bh-metrics bh-error">FAIL: ${escHtml(r.error || 'Unknown error')}</div>
            </div>`;
          }
          const p = Number(r.prefill_tps || 0).toFixed(1);
          const g = Number(r.gen_tps || 0).toFixed(1);
          const acc = (typeof r.draft_acceptance === 'number')
            ? `<span class="bh-metric"><b>Draft</b> ${(r.draft_acceptance * 100).toFixed(0)}%</span>` : '';
          return `<div class="bench-history-row">${head}
            <div class="bh-metrics">
              <span class="bh-metric"><b>Prefill</b> ${p}</span>
              <span class="bh-metric"><b>Gen</b> ${g}</span>${acc}
            </div>
          </div>`;
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
      renderSwitchStatus(data);

      // Sidebar
      buildSidebar(data);

      // Events & Logs
      updateEventsPanel(data);
    }

    // Seconds the dashboard has been showing "switching" — drives the stuck-state hint.
    const SWITCH_STUCK_HINT_SEC = 180;

    function renderSwitchStatus(data) {
      const el = document.getElementById('switchStatus');
      const msg = data.last_message || '';
      if (!data.switch_in_progress) {
        el.className = 'switch-status-bar';
        el.textContent = msg;
        return;
      }
      const startedMs = data.last_started_at ? Date.parse(data.last_started_at) : NaN;
      const elapsed = Number.isNaN(startedMs) ? 0 : Math.max(0, Math.round((Date.now() - startedMs) / 1000));
      const stuck = elapsed >= SWITCH_STUCK_HINT_SEC;
      el.className = 'switch-status-bar switch-active' + (stuck ? ' switch-stuck' : '');
      el.innerHTML = `
        <span class="switch-spin"></span>
        <span class="switch-msg">${escHtml(msg || 'Switching...')}</span>
        <span class="switch-elapsed">${fmtElapsed(elapsed)}</span>
        <button class="switch-clear" onclick="clearSwitchState()"
          title="Force the dashboard out of the switching state. Does not stop the container.">
          Force clear
        </button>`;
    }

    function fmtElapsed(sec) {
      if (sec < 60) return `${sec}s`;
      const m = Math.floor(sec / 60);
      return `${m}m ${String(sec % 60).padStart(2, '0')}s`;
    }

    async function clearSwitchState() {
      try {
        const r = await fetch('/api/switch/clear', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: '{}'
        });
        const d = await r.json().catch(() => ({}));
        if (d.status) { lastData = d.status; updateDashboard(d.status); }
      } catch (e) {
        const el = document.getElementById('switchStatus');
        el.className = 'switch-status-bar';
        el.textContent = `Could not clear switch state: ${e}`;
      }
      refresh();
    }

    // ── Settings ──────────────────────────────────────────────────────────────
    let settingsData = null;
    const settingsInspectTimers = {};

    async function openSettings() {
      closeSidebar();
      document.getElementById('settingsModal').classList.add('active');
      setSettingsStatus('');
      const body = document.getElementById('settingsBody');
      body.innerHTML = '<div class="rd-note">Loading…</div>';
      try {
        const r = await fetch('/api/settings');
        settingsData = await r.json();
        renderSettings(settingsData);
      } catch (e) {
        body.innerHTML = `<div class="rd-note">Could not load settings: ${escHtml(String(e))}</div>`;
      }
    }

    function closeSettings() {
      document.getElementById('settingsModal').classList.remove('active');
    }

    function setSettingsStatus(msg, kind) {
      const el = document.getElementById('settingsStatus');
      el.className = 'settings-status' + (kind ? ' ' + kind : '');
      el.textContent = msg || '';
    }

    function checkLine(item) {
      if (!item.exists) {
        return `<span class="set-check set-check-bad">✕ Not found inside the container — is this path bind-mounted?</span>`;
      }
      if (!item.is_dir) return `<span class="set-check set-check-bad">✕ Exists but is not a folder</span>`;
      if (!item.readable) return `<span class="set-check set-check-bad">✕ Not readable</span>`;
      if (!item.scans) return `<span class="set-check set-check-ok">✓ Folder found</span>`;
      if (!item.stack_count) {
        return `<span class="set-check set-check-warn">⚠ Folder found, but no *.yml with an <code>LLM_META display_name</code> header (${item.yml_count} yml file${item.yml_count === 1 ? '' : 's'})</span>`;
      }
      return `<span class="set-check set-check-ok">✓ ${item.stack_count} stack${item.stack_count === 1 ? '' : 's'} found (${item.yml_count} yml file${item.yml_count === 1 ? '' : 's'})</span>`;
    }

    function renderSettings(d) {
      const scanned = d.settings.filter(s => s.scans);
      const other = d.settings.filter(s => !s.scans);
      const row = (s) => `
        <div class="set-row" data-key="${s.key}">
          <div class="set-row-head">
            <span class="set-label">${escHtml(s.label)}</span>
            <span class="set-src set-src-${s.source}">${s.source}</span>
          </div>
          <div class="set-help">${escHtml(s.help)}</div>
          <input class="set-input" id="set-${s.key}" value="${escHtml(s.value)}"
                 spellcheck="false" autocomplete="off"
                 oninput="onSettingInput('${s.key}')" placeholder="/absolute/path" />
          <div id="check-${s.key}">${checkLine(s)}</div>
          ${s.env_value && s.source === 'settings'
            ? `<div class="set-envnote">Overrides ${escHtml(s.env_name)}=${escHtml(s.env_value)}</div>` : ''}
        </div>`;
      document.getElementById('settingsBody').innerHTML = `
        <div class="set-intro">
          Paths are resolved <b>inside the gpu-monitor container</b>. A folder that is not
          bind-mounted in <code>docker-compose.yml</code> will not be visible here, however
          correct it looks on the host.<br/>
          Saved values are written to <code>${escHtml(d.settings_path)}</code> and take
          precedence over the <code>${'$'}STACKS_DIR</code>-style environment variables that
          compose sets. Clear a field to fall back to env / config.
          ${d.writable ? '' : '<br/><b style="color:var(--danger)">This file is not writable — saving will fail.</b>'}
        </div>
        <div class="set-section">Stack folders</div>
        ${scanned.map(row).join('')}
        <div class="set-section">Other paths</div>
        ${other.map(row).join('')}`;
    }

    function onSettingInput(key) {
      setSettingsStatus('');
      clearTimeout(settingsInspectTimers[key]);
      const box = document.getElementById(`check-${key}`);
      box.innerHTML = '<span class="set-check set-check-idle">Checking…</span>';
      // Debounced so a typed path is not stat()ed on every keystroke.
      settingsInspectTimers[key] = setTimeout(async () => {
        const val = document.getElementById(`set-${key}`).value.trim();
        if (!val) {
          box.innerHTML = '<span class="set-check set-check-idle">Empty — will fall back to env / config / default</span>';
          return;
        }
        const meta = settingsData.settings.find(s => s.key === key) || { scans: false };
        try {
          const r = await fetch(`/api/settings/inspect?path=${encodeURIComponent(val)}`);
          const d = await r.json();
          if (d.error) { box.innerHTML = `<span class="set-check set-check-bad">✕ ${escHtml(d.error)}</span>`; return; }
          box.innerHTML = checkLine({ ...d, scans: meta.scans });
        } catch (e) {
          box.innerHTML = `<span class="set-check set-check-bad">✕ ${escHtml(String(e))}</span>`;
        }
      }, 350);
    }

    async function saveSettings() {
      if (!settingsData) return;
      const btn = document.getElementById('btnSaveSettings');
      const payload = {};
      for (const s of settingsData.settings) {
        payload[s.key] = document.getElementById(`set-${s.key}`).value.trim();
      }
      btn.disabled = true;
      setSettingsStatus('Saving…');
      try {
        const r = await fetch('/api/settings', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ settings: payload })
        });
        const d = await r.json();
        if (!r.ok) { setSettingsStatus(d.error || `Save failed (${r.status})`, 'err'); return; }
        settingsData = d;
        renderSettings(d);
        setSettingsStatus('Saved — model list reloaded.', 'ok');
        refresh();
      } catch (e) {
        setSettingsStatus(String(e), 'err');
      } finally {
        btn.disabled = false;
      }
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
    // The request log lives in a fixed bottom drawer (see .log-drawer-bar).
    function openReqlogDrawer() {
      document.getElementById('activityPanel').classList.add('open');
      document.getElementById('reqlogDrawerBtn').setAttribute('aria-expanded', 'true');
      document.getElementById('reqlogDrawerChevron').innerHTML = '&#9660;';
      refreshRequestLog(true);
    }
    function closeReqlogDrawer() {
      document.getElementById('activityPanel').classList.remove('open');
      document.getElementById('reqlogDrawerBtn').setAttribute('aria-expanded', 'false');
      document.getElementById('reqlogDrawerChevron').innerHTML = '&#9650;';
    }
    function toggleReqlogDrawer() {
      if (document.getElementById('activityPanel').classList.contains('open')) closeReqlogDrawer();
      else openReqlogDrawer();
    }
    window.addEventListener('resize', () => { if (window.innerWidth > 768) closeSidebar(); });

    // Quick navigation from the drawer/sidebar to a section.
    function goTo(section) {
      closeSidebar();
      setTimeout(() => {
        if (section === 'overview') {
          document.querySelector('.main-content').scrollTo({ top: 0, behavior: 'smooth' });
        } else if (section === 'requests') {
          switchActivityTab('reqlog');
          openReqlogDrawer();
        } else if (section === 'events') {
          switchActivityTab('events');
          openReqlogDrawer();
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
        const barCount = document.getElementById('reqlogBarCount');
        if (!d.available) {
          body.innerHTML = '<tr><td colspan="10" class="reqlog-empty">No request log yet — start a model stack, then send a request to port 8080 or 28082.</td></tr>';
          countEl.textContent = '';
          if (barCount) barCount.textContent = '';
          return;
        }
        if (d.error) {
          body.innerHTML = '<tr><td colspan="10" class="reqlog-empty">Error reading log: ' + escHtml(d.error) + '</td></tr>';
          countEl.textContent = '';
          if (barCount) barCount.textContent = '';
          return;
        }
        const rows = d.rows || [];
        countEl.textContent = rows.length ? String(rows.length) : '';
        if (barCount) barCount.textContent = rows.length ? String(rows.length) : '';
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

    document.addEventListener('keydown', (e) => { if (e.key === 'Escape') { closeRequestDetail(); closeSettings(); closeSidebar(); } });

    initChart();
    initThermalChart();
    buildRangePicker();
    setChartRange(chartRange);
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
        elif route == "/api/settings":
            self._send_json(describe_settings())
        elif route == "/api/settings/inspect":
            q = parse_qs(parsed.query)
            raw = (q.get("path") or [""])[0].strip()
            if not raw.startswith("/"):
                self._send_json({"error": "Path must be absolute", "path": raw}, 400)
            else:
                self._send_json({"path": raw, **inspect_stack_dir(Path(raw))})
        elif route == "/api/metrics/history":
            q = parse_qs(parsed.query)
            self._send_json(read_metrics_history((q.get("range") or ["1h"])[0]))
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

        if self.path == "/api/settings":
            values = body.get("settings")
            if not isinstance(values, dict):
                self._send_json({"error": "Expected a 'settings' object"}, 400)
                self._finish_request("POST", self.path)
                return
            ok, msg = apply_settings(values)
            if not ok:
                self._send_json({"error": msg}, 400)
            else:
                self._send_json({"message": msg, **describe_settings()})
            self._finish_request("POST", self.path)
            return

        if self.path == "/api/switch/clear":
            # Manual escape hatch: drop a wedged "Loading..." state so the UI unlocks
            # without waiting for SWITCH_STALE_SEC or restarting the dashboard.
            was_active = _clear_switch_state(
                "Switch state cleared manually",
                log="Switch state cleared manually from the dashboard",
            )
            self._send_json({
                "message": "Switch state cleared" if was_active else "No switch was in progress",
                "cleared": was_active,
                "status": build_status(self),
            })
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
                        entry = {
                            "profile": benchmark_profile,
                            "model_key": model_key,
                            "model_label": model_label,
                            "started_at": started_iso,
                            "completed_at": completed_iso,
                            "duration_sec": elapsed,
                            "success": False,
                            "error": error,
                        }
                        BENCHMARK_STATE["history"].append(entry)
                        save_benchmark_run(entry)
                    elif result is not None:
                        result["started_at"] = started_iso
                        result["completed_at"] = completed_iso
                        result["duration_sec"] = elapsed
                        result["model_key"] = model_key
                        result["model_label"] = model_label
                        BENCHMARK_STATE["last_result"] = result
                        BENCHMARK_STATE["last_error"] = None
                        accept = result.get("draft_acceptance")
                        accept_txt = f", draft accept {accept * 100:.0f}%" if isinstance(accept, float) else ""
                        _append_log_event("info", "benchmark", f"Benchmark ({benchmark_profile}) for {model_label}: prefill {result['prefill_tps']:.1f} T/S, gen {result['gen_tps']:.1f} T/S{accept_txt}")
                        entry = {
                            "profile": benchmark_profile,
                            "model_key": model_key,
                            "model_label": model_label,
                            "started_at": started_iso,
                            "completed_at": completed_iso,
                            "duration_sec": elapsed,
                            "success": True,
                            "prefill_tps": result["prefill_tps"],
                            "gen_tps": result["gen_tps"],
                            "draft_acceptance": result.get("draft_acceptance"),
                        }
                        BENCHMARK_STATE["history"].append(entry)
                        save_benchmark_run(entry)
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

    if init_metrics_db():
        restore_benchmark_history()
        threading.Thread(target=_run_metrics_sampler, daemon=True, name="metrics-sampler").start()

    threading.Thread(target=_run_log_watcher, daemon=True, name="log-watcher").start()

    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"dashboard server listening on {HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
