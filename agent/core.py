#!/usr/bin/env python3
import ast
import json
import os
import random
import zlib
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
    if _one("only", "") == "inference":
        # Hide monitoring traffic (/status, /props, /v1/models, /metrics...) and keep
        # the requests that actually ran the model.
        where.append(
            "(IFNULL(path,'') LIKE '%completions%' OR IFNULL(path,'') LIKE '%chat%' "
            "OR IFNULL(path,'') LIKE '%embeddings%' OR IFNULL(path,'') LIKE '%generate%' "
            "OR IFNULL(path,'') LIKE '%messages%' OR IFNULL(path,'') LIKE '%responses%')"
        )
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
# "kind" picks the context register (see BENCHMARK_CONTEXT_KINDS). The full profile
# deliberately spans general prose, code and chat transcripts so the aggregate is not
# one register's acceptance rate. Aggregates from before 2026-09-20 were taken with
# opaque filler throughout and are NOT comparable with these.
FULL_BENCHMARK_SPECS = [
    {"name": "prefill_xl", "prompt_tokens": 4096, "n_predict": 32, "kind": "general"},
    {"name": "prefill_heavy", "prompt_tokens": 2048, "n_predict": 64, "kind": "code"},
    {"name": "prefill_medium", "prompt_tokens": 1024, "n_predict": 128, "kind": "chat"},
    {"name": "mixed", "prompt_tokens": 1024, "n_predict": 256, "kind": "general"},
    {"name": "mixed_even", "prompt_tokens": 512, "n_predict": 512, "kind": "code"},
    {"name": "gen_heavy", "prompt_tokens": 256, "n_predict": 1024, "kind": "chat"},
    {"name": "gen_xl", "prompt_tokens": 128, "n_predict": 2048, "kind": "general"},
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
METRICS_LIVE = {"ts": 0.0, "tps": None, "ingest": None, "running": None, "waiting": None,
                "ctx_pct": None}
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


def _guess_vllm_backend(path: Path) -> str:
    """Stacks in VLLM_DIR may run vLLM, Strata or TabbyAPI. LLM_META backend="..." wins;
    otherwise sniff the compose file for the engine's image name."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace").lower()
    except OSError:
        return "vllm"
    if "strata" in text:
        return "strata"
    if "tabbyapi" in text or "exl3" in text:
        return "tabbyapi"
    return "vllm"


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
                "backend":        meta.get("backend") or "llamacpp",
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
                "internal_port":  int(meta.get("internal_port", "8080")),
                "backend":        meta.get("backend") or _guess_vllm_backend(path),
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
                "backend":        "beellama",
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
VLLM_METRICS_SCRAPE = {"running": False}  # one background scrape at a time
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
        STATE["force_restart"] = False
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
        f0 = lambda v: "?" if v is None else f"{v:.0f}"  # noqa: E731
        parts.append(f"GPU: {f0(g0['util'])}%  VRAM: {f0(g0['mem_used'])}/{f0(g0['mem_total'])}MB  {f0(g0['temp'])}°C")

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


# (2026-11-02) Missing-image self-healing. `docker image prune -a` (or a disk cleanup)
# removes locally built engine images, and `up` then fails with "pull access denied" /
# "No such image" because they have `pull_policy: never` or exist on no registry.
# Before every stack start, ensure_stack_images() checks each service image and rebuilds
# what is missing, in this order:
#   1. a recipe in the compose file's top-level `x-gpu-monitor-images` extension:
#        x-gpu-monitor-images:
#          - image: local/base:tag            # listed in dependency order (bases first)
#            context: /abs/path/to/source     # must be readable inside this container
#            dockerfile: Dockerfile           # optional, relative to context
#            args: {KEY: value}               # optional --build-arg
#      A recipe is also built when a later missing image needs it (FROM base).
#   2. the service's own `build:` section -> `docker compose build <service>`
#   3. otherwise `docker pull <image>` (unless pull_policy: never)
IMAGE_BUILD_TIMEOUT_SEC = int(os.environ.get("IMAGE_BUILD_TIMEOUT_SEC", "7200"))


def _docker_image_exists(image: str) -> bool:
    try:
        return subprocess.run(["docker", "image", "inspect", image],
                              capture_output=True, timeout=30).returncode == 0
    except Exception:
        return False


def _run_logged(cmd: list[str], timeout: int) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, ((proc.stdout or "") + (proc.stderr or "")).strip()
    except subprocess.TimeoutExpired:
        return 124, f"timeout after {timeout}s: {' '.join(cmd)}"


def ensure_stack_images(compose_file: str | Path, project: str | None = None,
                        cwd: str | Path | None = None,
                        status_cb=None) -> tuple[bool, str]:
    """Rebuild/pull any image a compose stack needs that is missing locally.
    Returns (ok, message). Never raises."""
    compose_file = Path(compose_file)
    if not compose_file.is_absolute() and cwd:
        compose_file = Path(cwd) / compose_file
    base = ["docker", "compose"]
    if project:
        base += ["-p", project]
    if cwd:
        base += ["--project-directory", str(cwd)]
    base += ["-f", str(compose_file)]
    code, out = _run_logged(base + ["config", "--format", "json"], 120)
    if code != 0:
        return False, f"compose config failed: {out[-500:]}"
    try:
        cfg = json.loads(out)
    except Exception as exc:
        return False, f"compose config not JSON: {exc}"

    recipes = {r["image"]: r for r in (cfg.get("x-gpu-monitor-images") or [])
               if isinstance(r, dict) and r.get("image")}
    order = list(recipes)

    def note(msg: str, level: str = "info") -> None:
        _append_log_event(level, "switch", msg)
        if status_cb:
            try:
                status_cb(msg)
            except Exception:
                pass

    def build_recipe(image: str) -> tuple[bool, str]:
        r = recipes[image]
        # Bases listed before this recipe are built first if they are missing too.
        for dep in order[:order.index(image)]:
            if not _docker_image_exists(dep):
                ok, msg = build_recipe(dep)
                if not ok:
                    return ok, msg
        ctx = str(r.get("context") or "")
        if not ctx or not Path(ctx).is_dir():
            return False, f"build context for {image} not readable in gpu-monitor: {ctx!r}"
        cmd = ["docker", "build", "-t", image]
        if r.get("dockerfile"):
            cmd += ["-f", str(Path(ctx) / str(r["dockerfile"]))]
        for k, v in (r.get("args") or {}).items():
            cmd += ["--build-arg", f"{k}={v}"]
        cmd.append(ctx)
        note(f"Image {image} missing - building (can take a long time)...")
        t0 = time.time()
        code, out = _run_logged(cmd, IMAGE_BUILD_TIMEOUT_SEC)
        if code != 0:
            return False, f"build of {image} failed (exit {code}): {out[-1500:]}"
        note(f"Built {image} in {int(time.time() - t0)}s")
        return True, ""

    fixed = []
    for svc_name, svc in (cfg.get("services") or {}).items():
        image = svc.get("image")
        if image and _docker_image_exists(image):
            continue
        if image and image in recipes:
            ok, msg = build_recipe(image)
        elif svc.get("build"):
            note(f"Image for service {svc_name} missing - docker compose build...")
            code, out = _run_logged(base + ["build", svc_name], IMAGE_BUILD_TIMEOUT_SEC)
            ok, msg = code == 0, f"compose build {svc_name} failed (exit {code}): {out[-1500:]}"
        elif image and svc.get("pull_policy") != "never":
            note(f"Image {image} missing - pulling...")
            code, out = _run_logged(["docker", "pull", image], IMAGE_BUILD_TIMEOUT_SEC)
            ok, msg = code == 0, f"pull of {image} failed (exit {code}): {out[-800:]}"
        else:
            ok, msg = False, (f"image {image} for service {svc_name} is missing and has no "
                              f"x-gpu-monitor-images recipe or build section")
        if not ok:
            note(msg, "error")
            return False, msg
        fixed.append(image or svc_name)
    return True, ("rebuilt: " + ", ".join(fixed)) if fixed else "all images present"


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


def _current_stack_model_key(compose_to_model: dict) -> str | None:
    """Resolve the active vLLM stack from the switcher's .current-stack state file.

    A running container's compose label is stamped at `up` time and goes stale the
    moment a stack file is renamed, so the label basename can name a file that no
    longer exists. switch-vllm.sh rewrites .current-stack on every start, so it is
    the more current of the two. Returns None if the file is missing or names a
    stack that is no longer discovered.
    """
    try:
        raw = (VLLM_DIR / ".current-stack").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not raw:
        return None
    return compose_to_model.get(Path(raw).name)


def detect_active_model_key(containers: list[dict] | None = None) -> str | None:
    models, compose_to_model = get_models()
    records = containers if containers is not None else list_llama_compose_containers()
    for c in records:
        model_key = compose_to_model.get(c["compose"])
        if model_key and c["status"].startswith("Up"):
            return model_key
    # Fallback: check vLLM directly; resolve to specific stack key if possible
    vllm_records = list_vllm_compose_containers()
    for c in vllm_records:
        if c.get("vllm") and c["status"].startswith("Up"):
            model_key = compose_to_model.get(c["compose"])
            if model_key:
                return model_key
            # Stale/unknown compose label (e.g. the stack file was renamed after
            # `up`). Fall back to .current-stack, then to the generic "vllm" entry
            # -- but only if it actually exists, since discover_models() creates it
            # only when NO stack files were found. Returning a key that is absent
            # from `models` used to raise KeyError in build_status().
            return _current_stack_model_key(compose_to_model) or (
                "vllm" if "vllm" in models else None
            )
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
        compose = model.get("compose")
        return is_vllm_healthy(containers, compose if compose and compose != "vllm" else None)

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


def is_vllm_healthy(containers: list[dict] | None = None, compose: str | None = None) -> bool:
    """Check if the vLLM container is healthy.

    With `compose` (the stack's file name) only that stack's server counts, so a
    leftover container from another stack can't report the requested one healthy.
    """
    records = list_vllm_compose_containers()
    for c in records:
        if not c.get("vllm"):
            continue
        if compose and c.get("compose") and c.get("compose") != compose:
            continue
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


def _vllm_auth_args() -> list[str]:
    """curl header args carrying the shared vLLM-stack API key, or [] if there is none."""
    if VLLM_API_KEY_FILE:
        try:
            api_key = Path(VLLM_API_KEY_FILE).read_text(encoding="utf-8").strip()
            if api_key:
                return ["-H", f"Authorization: Bearer {api_key}"]
        except OSError:
            pass
    return []


def _strata_metrics_to_prometheus(raw: str) -> str:
    """(2026-09-30) Strata's /metrics is JSON (its own web UI reads it); map it to the vLLM
    Prometheus names parse_vllm_metrics() reads. Anything that is not JSON passes through.

    Counters come from Strata's lifetime `totals` (updated when a request finishes) plus the
    in-flight request's `live.generated`, so live T/s moves while a reply is written. Strata
    runs one generation at a time, so the "pool" is one context window.
    """
    if not raw.lstrip().startswith("{"):
        return raw
    try:
        data = json.loads(raw)
    except ValueError:
        return raw
    engine = data.get("engine") or {}
    live = data.get("live") or {}
    totals = data.get("totals") or {}

    def num(v) -> float:
        try:
            return float(v or 0)
        except (TypeError, ValueError):
            return 0.0

    ctx = num(engine.get("context") or engine.get("max_context"))
    busy = live.get("state") not in (None, "idle")
    live_gen = num(live.get("generated")) if busy else 0.0
    # prompt_read counts up during prefill and is None once it answers (whole prompt read)
    read = live.get("prompt_read")
    live_read = num(live.get("prompt_tokens") if read is None else read) if busy else 0.0
    live_total = num(live.get("prompt_total") or live.get("prompt_tokens")) if busy else 0.0
    # (2026-10-31) v0.2.9 batch slots: live.running counts generating requests (solo or in a
    # slot); live.waiting/admitting are queued. Older engines: one request, live.queued.
    if "running" in live:
        running = num(live.get("running"))
        waiting = num(live.get("waiting")) + num(live.get("admitting"))
    else:
        running = 1.0 if busy else 0.0
        waiting = num(live.get("queued"))
    n = num(totals.get("requests"))
    prompt = num(totals.get("prompt_tokens"))
    reused = num(totals.get("reused"))
    out = num(totals.get("output_tokens"))
    prompt_s = num(totals.get("prompt_ms")) / 1000.0
    decode_s = num(totals.get("decode_ms")) / 1000.0

    lines = [
        f"vllm:generation_tokens_total {out + live_gen}",
        f"vllm:prompt_tokens_total {max(prompt - reused, 0.0)}",
        f"vllm:prompt_tokens_cached_total {reused}",
        f"vllm:prefix_cache_queries_total {prompt}",
        f"vllm:prefix_cache_hits_total {reused}",
        f"vllm:num_requests_running {running}",
        f"vllm:num_requests_waiting {waiting}",
        f"vllm:request_prompt_tokens_sum {prompt}",
        f"vllm:request_prompt_tokens_count {n}",
        f"vllm:request_generation_tokens_sum {out}",
        f"vllm:request_generation_tokens_count {n}",
        f"vllm:request_prefill_time_seconds_sum {prompt_s}",
        f"vllm:request_prefill_time_seconds_count {n}",
        f"vllm:request_decode_time_seconds_sum {decode_s}",
        f"vllm:request_decode_time_seconds_count {n}",
        f"vllm:time_to_first_token_seconds_sum {prompt_s}",
        f"vllm:time_to_first_token_seconds_count {n}",
        f"vllm:e2e_request_latency_seconds_sum {prompt_s + decode_s}",
        f"vllm:e2e_request_latency_seconds_count {n}",
        f"vllm:request_time_per_output_token_seconds_sum {decode_s / out * n if out else 0.0}",
        f"vllm:request_time_per_output_token_seconds_count {n}",
        f'vllm:request_success_total{{finished_reason="stop"}} {n}',
    ]
    if ctx:
        used = (live_read + live_gen) if busy else 0.0
        lines += [
            f'vllm:cache_config_info{{kv_cache_size_tokens="{int(ctx)}"}} 1',
            f"vllm:kv_cache_usage_perc {min(used / ctx, 1.0)}",
        ]
    if num(engine.get("spec")) > 0 or num(engine.get("mtp_max")) > 0:
        # MTP drafting is on. v0.2.9 totals carry drafts_offered/drafts_accepted (lifetime,
        # finished requests); older engines have none, so a present-but-zero counter still
        # renders "on", not "off". No proposal-round count, so acceptance length stays blank.
        lines.append(f"vllm:spec_decode_num_draft_tokens_total {num(totals.get('drafts_offered'))}")
        if "drafts_accepted" in totals:
            lines.append(f"vllm:spec_decode_num_accepted_tokens_total {num(totals.get('drafts_accepted'))}")
    if busy:
        lines += [
            f"tabby:ctx_live_tokens {live_read + live_gen}",
            f"tabby:ctx_live_target_tokens {live_total + live_gen}",
        ]
    return "\n".join(lines) + "\n"


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
    # Promotions actually performed (CPU tier -> GPU). THE number that says whether the
    # host tier earns its pinned RAM: bytes can look busy while this stays at 0, which is
    # exactly the write-only failure the MTP eagle-group bug produced (28 GB stored, 1
    # load, all of it from a synthetic test). Tracked as a COUNT, not bytes, because a
    # single large promotion and a steady trickle are different situations.
    offload_load_count = get_metric("kv_offload_load_size_count") or 0.0
    # Tier lookup hit rate. vLLM 0.28 dropped external_prefix_cache_{queries,hits}_total
    # (which is why the "host" figure read blank on this build) in favour of the
    # per-tier chunk counters.
    tier_q = _sum_labeled_metric(raw, "kv_offload_tiering_chunk_queries_total") or 0.0
    tier_h = _sum_labeled_metric(raw, "kv_offload_tiering_chunk_hits_total") or 0.0
    tier_hit_rate = (tier_h / tier_q) if tier_q > 0 else None

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
    # (2026-09-30) Live per-request context from engines that publish it (the TabbyAPI
    # shim, tabby_metrics.py): tokens already in the largest in-flight request's KV, which
    # climbs chunk by chunk during prefill, and that request's full prompt+output size.
    live_ctx_tokens = parsed.get("tabby:ctx_live_tokens")
    live_ctx_target = parsed.get("tabby:ctx_live_target_tokens")

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

    # (2026-09-30) Decode rate of the request(s) that finished since the last scrape, from
    # the per-request generation-token and decode-time sums. The window rate above is
    # diluted by idle time around a request's end, which made "last completed" read ~5 T/s.
    last_request_tps = None
    req_gen_sum = get_metric("request_generation_tokens_sum")
    req_decode_sum = get_metric("request_decode_time_seconds_sum")
    with VLLM_TPS_STATE_LOCK:
        if VLLM_TPS_STATE.get("req_container") == active_key:
            d_gen = req_gen_sum - (VLLM_TPS_STATE.get("req_gen_sum") or 0.0)
            d_dec = req_decode_sum - (VLLM_TPS_STATE.get("req_decode_sum") or 0.0)
            if d_gen > 0 and d_dec > 0:
                VLLM_TPS_STATE["last_request_tps"] = d_gen / d_dec
            last_request_tps = VLLM_TPS_STATE.get("last_request_tps")
        else:
            VLLM_TPS_STATE["last_request_tps"] = None
        VLLM_TPS_STATE["req_container"] = active_key
        VLLM_TPS_STATE["req_gen_sum"] = req_gen_sum
        VLLM_TPS_STATE["req_decode_sum"] = req_decode_sum

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
        "live_ctx_tokens": live_ctx_tokens,
        "last_request_tps": last_request_tps,
        "live_ctx_target": live_ctx_target,
        "avg_tpot_sec": avg_tpot_sec,
        "avg_output_rate_tps": avg_output_rate_tps,
        "avg_prefill_rate_tps": avg_prefill_rate_tps,
        "avg_e2e_sec": avg_e2e_sec,
        "avg_itl_sec": avg_itl_sec,
        "prefix_hit_rate": prefix_hit_rate,
        "prefix_queries": prefix_q or None,
        "external_prefix_hit_rate": ext_hit_rate if ext_hit_rate is not None else tier_hit_rate,
        "offload_load_count": offload_load_count or None,
        "offload_tier_queries": tier_q or None,
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

    # ⭐ NEVER SCRAPE ON THE REQUEST PATH (2026-09-27). TabbyAPI stops answering HTTP
    # while it prefills a long prompt: /metrics was measured at 0.35-7 s routinely and
    # 42.8 s once, during a 32K-token agent turn. Scraped inline, every /api/status poll
    # waited on it (up to the 5 s curl cap plus ~1 s of docker exec on a loaded host), and
    # the whole dashboard froze. Now the poll always answers from the last good scrape
    # and, if that is stale, starts ONE background scrape. `gauge_age_sec` says how old
    # the numbers are.
    now = time.time()
    with VLLM_METRICS_CACHE_LOCK:
        stale = (VLLM_METRICS_CACHE["container"] != container
                 or (now - VLLM_METRICS_CACHE["checked_at"]) >= VLLM_METRICS_CACHE_TTL_SEC)
        start = stale and not VLLM_METRICS_SCRAPE["running"]
        if start:
            VLLM_METRICS_SCRAPE["running"] = True
        same_container = VLLM_METRICS_CACHE["container"] == container
    if start:
        threading.Thread(target=_scrape_vllm_metrics_bg, args=(container,),
                         daemon=True, name="vllm-metrics-scrape").start()
    if not same_container:
        # A different engine than the cached one: its old numbers would be wrong, so report
        # "collecting" until the first scrape of this container lands.
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
        }
    with VLLM_METRICS_CACHE_LOCK:
        return _build_vllm_throughput_from_cache(container)


# Engine port inside the active vLLM/Strata container. build_status() sets it from the
# stack's LLM_META internal_port (default 8080) before any scrape is started.
VLLM_ACTIVE_PORT = {"port": 8080}


def _scrape_vllm_metrics_bg(container: str) -> None:
    try:
        _scrape_vllm_metrics(container)
    except Exception as exc:  # noqa: BLE001
        _append_log_event("warning", "metrics", f"vLLM metrics scrape failed: {exc}")
    finally:
        with VLLM_METRICS_CACHE_LOCK:
            VLLM_METRICS_SCRAPE["running"] = False


def _scrape_vllm_metrics(container: str) -> dict:
    """Scrape and parse the engine's /metrics once and store it in VLLM_METRICS_CACHE.

    Runs on the background scrape thread only; see build_vllm_throughput_status().
    """
    code, output = run_command(
        ["docker", "exec", container, "curl", "-fsS", "--max-time", "20", *_vllm_auth_args(),
         f"http://127.0.0.1:{VLLM_ACTIVE_PORT['port']}/metrics"]
    )
    if code == 0:
        output = _strata_metrics_to_prometheus(output)
    now = time.time()
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
            "live_ctx_tokens": parsed.get("live_ctx_tokens"),
            "live_ctx_target": parsed.get("live_ctx_target"),
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
                "offload_load_count", "offload_tier_queries",
                "offload_cpu_usage_perc", "success_by_reason", "waiting_by_reason",
                "last_request_tps",
            )},
        }

    with VLLM_METRICS_CACHE_LOCK:
        prev_container = VLLM_METRICS_CACHE["container"]
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
            # Keep the last good result: a scrape that timed out during a prefill stall
            # says nothing new about the engine, and blanking the cards is what made the
            # dashboard look dead. gauge_age_sec keeps growing, so staleness stays visible.
            VLLM_METRICS_CACHE["raw"] = None
            if prev_container != container:
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
            "live_ctx_tokens": None,
            "live_ctx_target": None,
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

        completion_tps = completed.get("last_request_tps") or completed.get("tokens_per_second")
        completion_ingest = completed.get("ingest_tps")
        completion_key = completed.get("completion_key")
        # (2026-09-30) "cached" too: status polls are served from the background scrape's
        # cache, so a vLLM-family reading is almost never "ok" and this never fired.
        # completion_key (the finished-request count) already dedups repeats.
        should_add_completion = (
            completed.get("state") in ("ok", "cached")
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

# ── Realistic benchmark context (added 2026-09-20) ───────────────────────────
# The original prompt padded with random high-entropy words. That was a considered
# choice, not an oversight: a prompt of one repeated token, or of prose the model
# then copies back, is a best case for speculative decoding and inflates reported
# generation throughput. This block keeps that ANTI-COPY GUARD -- every task asks
# for something the context does not contain -- while making the context real text
# in three registers, so the profile covers general prose, code and chat rather
# than one synthetic distribution. Keep the guard when editing these.
#
# ⚠️ WHAT THIS DOES AND DOES NOT BUY, all MEASURED on Swift-Qwen3.8-27B with MTP,
# fixed 1024-token prompt and 512 generated tokens, /v1/completions + ignore_eos
# (the harness's own request shape), idle box:
#
#     opaque filler  + essay task (the old prompt)      49.0% acceptance
#     real context   + essay task                       50.5%
#     real context   + ordinary answer task (SHIPS)     52.2%
#     short question, no padding at all                 58.5%
#
# So realistic context is worth about +1.5 points of draft acceptance and an
# ordinary "answer this" task another +1.7. Both are real, both are small.
# ⛔ What dominates is PROMPT LENGTH, not padding content: the same model on short
# unpadded questions accepts 58.5% against ~50% on any 1024-token padded prompt.
# A padded benchmark prompt therefore reports decode throughput BELOW what short
# real requests get, and no choice of filler text fixes that -- it is inherent to
# measuring long-prompt behaviour. Do not read the full profile's acceptance as
# "the model's acceptance rate"; read it as acceptance at that prompt length.
# (An earlier version of this comment blamed opaque filler for the whole gap. It
# was wrong: filler accounts for ~1.5 points of roughly nine.)
#
# ⚠️ COMPARABILITY: numbers recorded before 2026-09-20 used kind="filler" with the
# essay task and are NOT comparable with the realistic kinds. kind="filler" is
# retained so an old measurement can be reproduced deliberately.
#
# Units are sampled WITHOUT replacement in a seeded order and only reshuffled once
# exhausted, so a large prompt repeats material but never adjacently. The pools are
# deliberately small enough to keep this file self-contained: dashboard-server.py is
# baked into the image, so it cannot read a corpus off disk -- and a code change here
# needs `docker compose up -d --build`, since a restart alone keeps the old copy.

BENCHMARK_CONTEXT_GENERAL = [
    "The storage tier was provisioned for throughput, not latency, and the difference only became visible under mixed read and write load.",
    "Each regional office kept its own copy of the pricing table, which is how three different answers to the same question reached the customer.",
    "We assumed the queue drained faster than it filled, and for eleven months that assumption held.",
    "The migration plan called for a read-only window of four hours; the actual cutover took nineteen minutes and nobody believed the dashboard.",
    "Procurement rejected the first proposal because the warranty terms did not cover drives that failed inside the first ninety days.",
    "Support tickets clustered around Monday mornings, which turned out to reflect a weekly batch job rather than user behaviour.",
    "The report distinguished between capacity that exists and capacity that can be reached without a restart, and only the second number mattered.",
    "Two teams measured the same endpoint and disagreed by a factor of three, because one of them counted queue time and the other did not.",
    "The vendor benchmark used a working set that fit entirely in cache, which is a legitimate configuration and a misleading headline.",
    "After the reorganisation the on-call rotation covered nine services with six people, and the escalation policy was never updated to match.",
    "Compliance required an audit trail for every schema change, so the team wrote a migration tool that refused to run without a ticket reference.",
    "The forecast was revised downward twice, both times because a single large contract slipped a quarter.",
    "Latency at the ninety-ninth percentile was four times the median, and the median was the only figure in the quarterly review.",
    "The cache hit rate improved after deployment, but so did the error rate, and the two turned out to share a cause.",
    "Field engineers carried a spare power supply because the failure they actually saw was never the one the manual described.",
    "A postmortem is useful in proportion to how specific it is about what was believed at the time, not about what is obvious afterwards.",
    "The contract specified availability per calendar month, which meant a six-hour outage on the thirty-first was cheaper than one on the first.",
    "Inventory turnover looked healthy in aggregate and terrible once seasonal lines were separated out.",
    "The new index made the reporting query fast and the ingest path noticeably slower, which surprised nobody who had read the write path.",
    "Training materials were rewritten after it became clear most errors came from a single ambiguous field label.",
    "The board asked for one number; the honest answer required three, and the compromise was a footnote nobody read.",
    "Retention improved in the cohort that received slower but more accurate responses, which contradicted the original hypothesis.",
    "Because the retry policy was exponential but unbounded, a brief upstream failure produced traffic for the next forty minutes.",
    "The building's cooling capacity, not the power budget, set the practical limit on how many racks could be populated.",
    "Every estimate in the plan was a median with no stated spread, so the schedule had no slack anywhere in it.",
    "The data was correct, the aggregation was correct, and the label on the axis was wrong, which is the hardest class of error to catch.",
    "Rolling back the configuration restored service in under a minute and destroyed the evidence needed to explain the incident.",
    "Suppliers quoted lead times in weeks and delivered in months, so the team learned to order against the pessimistic figure.",
    "The audit found no unauthorised access and considerable unnecessary access, which is a finding worth acting on.",
    "Demand was elastic below a threshold price and almost perfectly inelastic above it, which made the pricing experiment cheap to run.",
    "Documentation described the intended behaviour; the tests described the actual behaviour; the two had diverged over about two years.",
    "Because the clocks drifted, events arrived out of order and the deduplication logic silently discarded the newer record.",
    "The pilot succeeded partly because the team running it had written the tool and knew which paths to avoid.",
    "Capacity planning based on peak-to-mean ratios worked until a marketing campaign changed the shape of the traffic entirely.",
    "The simplest explanation for the discrepancy was a unit error, and after two days of analysis that is exactly what it was.",
    "Maintenance windows were negotiated annually, which meant urgent work waited and routine work was done under pressure.",
]

BENCHMARK_CONTEXT_CODE = [
    'def load_config(path, defaults=None):\n    """Read a JSON config, layering it over defaults."""\n    cfg = dict(defaults or {})\n    try:\n        with open(path) as fh:\n            cfg.update(json.load(fh))\n    except FileNotFoundError:\n        pass\n    except json.JSONDecodeError as exc:\n        raise ValueError(f"{path}: {exc}") from exc\n    return cfg',
    'class RingBuffer:\n    def __init__(self, capacity):\n        if capacity <= 0:\n            raise ValueError("capacity must be positive")\n        self._buf = [None] * capacity\n        self._head = 0\n        self._len = 0\n\n    def append(self, item):\n        self._buf[(self._head + self._len) % len(self._buf)] = item\n        if self._len < len(self._buf):\n            self._len += 1\n        else:\n            self._head = (self._head + 1) % len(self._buf)\n\n    def __len__(self):\n        return self._len',
    'def retry(fn, attempts=5, base=0.1, cap=5.0):\n    last = None\n    for i in range(attempts):\n        try:\n            return fn()\n        except TransientError as exc:\n            last = exc\n            time.sleep(min(cap, base * (2 ** i)) * (0.5 + random.random()))\n    raise last',
    'def percentile(values, q):\n    if not values:\n        return None\n    ordered = sorted(values)\n    idx = (len(ordered) - 1) * (q / 100.0)\n    lo = math.floor(idx)\n    hi = math.ceil(idx)\n    if lo == hi:\n        return ordered[int(idx)]\n    return ordered[lo] * (hi - idx) + ordered[hi] * (idx - lo)',
    'async def fan_out(urls, limit=8):\n    sem = asyncio.Semaphore(limit)\n\n    async def fetch(session, url):\n        async with sem:\n            async with session.get(url, timeout=30) as resp:\n                resp.raise_for_status()\n                return url, await resp.json()\n\n    async with aiohttp.ClientSession() as session:\n        return await asyncio.gather(*(fetch(session, u) for u in urls), return_exceptions=True)',
    'def merge_intervals(intervals):\n    out = []\n    for start, end in sorted(intervals):\n        if out and start <= out[-1][1]:\n            out[-1] = (out[-1][0], max(out[-1][1], end))\n        else:\n            out.append((start, end))\n    return out',
    'CREATE TABLE benchmarks (\n  id        INTEGER PRIMARY KEY,\n  ts        REAL    NOT NULL,\n  profile   TEXT    NOT NULL,\n  model_key TEXT    NOT NULL,\n  success   INTEGER NOT NULL DEFAULT 0,\n  gen_tps   REAL,\n  UNIQUE (model_key, ts)\n);\nCREATE INDEX idx_benchmarks_model ON benchmarks (model_key, success, ts DESC);',
    'def parse_prometheus(text):\n    out = {}\n    for line in text.splitlines():\n        if not line or line.startswith("#"):\n            continue\n        name, _, value = line.rpartition(" ")\n        base = name.split("{", 1)[0]\n        try:\n            out.setdefault(base, []).append(float(value))\n        except ValueError:\n            continue\n    return out',
    'function debounce(fn, waitMs) {\n  let timer = null;\n  return (...args) => {\n    if (timer !== null) clearTimeout(timer);\n    timer = setTimeout(() => {\n      timer = null;\n      fn(...args);\n    }, waitMs);\n  };\n}',
    'def chunked(iterable, size):\n    if size < 1:\n        raise ValueError("size must be >= 1")\n    it = iter(iterable)\n    while True:\n        batch = list(itertools.islice(it, size))\n        if not batch:\n            return\n        yield batch',
    '@dataclass(frozen=True)\nclass RunResult:\n    name: str\n    prompt_tokens: int\n    gen_tokens: int\n    prompt_ms: float\n    gen_ms: float\n\n    @property\n    def gen_tps(self) -> float:\n        return self.gen_tokens / (self.gen_ms / 1000.0) if self.gen_ms else 0.0',
    # Short units: a single function is a coarse packing quantum, and without
    # something smaller a 512-token code prompt lands ~14% under target.
    'def clamp(x, lo, hi):\n    return lo if x < lo else hi if x > hi else x',
    'def human_bytes(n):\n    for unit in ("B", "KiB", "MiB", "GiB"):\n        if abs(n) < 1024:\n            return f"{n:.1f}{unit}"\n        n /= 1024\n    return f"{n:.1f}TiB"',
    'export const sleep = (ms) => new Promise((res) => setTimeout(res, ms));',
    'SELECT model_key, COUNT(*) AS runs, AVG(gen_tps) AS mean_tps\n  FROM benchmarks\n WHERE success = 1\n GROUP BY model_key\n ORDER BY mean_tps DESC;',
    'def is_power_of_two(n):\n    return n > 0 and (n & (n - 1)) == 0',
    'def bisect_left(a, x, lo=0, hi=None):\n    if hi is None:\n        hi = len(a)\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if a[mid] < x:\n            lo = mid + 1\n        else:\n            hi = mid\n    return lo',
]

BENCHMARK_CONTEXT_CHAT = [
    "User: The nightly export finished but the row count is about 12% lower than yesterday. Where should I look first?\nAssistant: Compare the source query's date filter against the job's start time. A job that now starts before midnight in local time will clip the last partition.",
    "User: We set max_connections to 400 and the pool still exhausts under load.\nAssistant: Check whether the limit is per process. With eight workers each holding its own pool, the effective ceiling is the per-worker size, not the server setting.",
    "User: Can I run the migration without taking the service down?\nAssistant: Adding a nullable column is safe online. Backfilling it in one statement is not, because it takes a lock for the duration; batch the backfill instead.",
    "User: Our p99 doubled after the deploy but the median is unchanged.\nAssistant: That pattern usually means a new slow path on a minority of requests rather than a general regression. Group latency by endpoint and by cache hit or miss.",
    "User: Should we store timestamps as text or integers?\nAssistant: Store them as an explicit timestamp type with a timezone if the database has one. Integers are compact but every consumer then reimplements the conversion, usually inconsistently.",
    "User: The retry storm took down the upstream again.\nAssistant: Bound the total retry window and add jitter. Unbounded exponential backoff still produces a long tail of traffic after a brief failure.",
    "User: How big should the batch be for the nightly reindex?\nAssistant: Size it by wall time rather than row count, targeting a few seconds per batch, so the job adapts when rows get wider.",
    "User: Two services disagree about the customer's balance.\nAssistant: Establish which one owns the value. If both write it, the disagreement is a design outcome, not a bug to be patched.",
    "User: Is it worth adding an index for this report?\nAssistant: Measure the write cost first. A reporting query run four times a day rarely justifies slowing every insert.",
    "User: The container gets OOM killed but the process shows low RSS.\nAssistant: Page cache and tmpfs count against the cgroup limit. Check memory.current and memory.stat rather than the process's own RSS.",
    "User: Can we cache this endpoint?\nAssistant: Only if the response is a function of the request alone. Anything that depends on the caller's permissions needs the identity in the cache key or it will leak.",
    "User: Our deploy rolled back automatically and we lost the logs.\nAssistant: Ship logs off the instance before the rollback completes, or keep the failed instance quarantined instead of terminating it.",
    "User: Why is the first request after a deploy so slow?\nAssistant: Cold caches, lazy connection pools and JIT or graph compilation all land on that request. Warm the path in the readiness check.",
    "User: We want exactly-once processing.\nAssistant: Aim for at-least-once delivery with idempotent handlers. Exactly-once across a network boundary requires coordination most pipelines cannot afford.",
    "User: The backup completes but the restore test fails.\nAssistant: Then the backup is not verified and should not be counted as one. Restore into a scratch environment as part of the schedule.",
    "User: How do I pick a timeout for this call?\nAssistant: Base it on the downstream's measured p99 plus headroom, and make sure it is shorter than the caller's own timeout, otherwise retries pile up behind a request nobody is waiting for.",
]

# Wrapper + task instruction cost, MEASURED with the Qwen3.8 tokenizer 2026-09-20,
# per kind because the instructions differ in length. The legacy filler path keeps
# using BENCHMARK_PROMPT_OVERHEAD_TOKENS so its historical sizing is untouched.
# The per-unit estimators below are calibrated against the same tokenizer: with
# these values a requested 4096-token prompt lands within a few percent, against
# the 11% undershoot the first uncalibrated guess produced. Re-measure if the
# instructions or the pools change materially.
BENCHMARK_CONTEXT_OVERHEAD_TOKENS = {"general": 90, "code": 91, "chat": 87}

# Per kind: (units, tokens-per-unit estimator, context label, task instruction).
# The task always asks for material the context does NOT supply, so the model must
# generate rather than copy. See the note above before changing these.
BENCHMARK_CONTEXT_KINDS = {
    "general": (
        BENCHMARK_CONTEXT_GENERAL,
        lambda u: len(u.split()) * 1.20,
        "NOTES",
        "TASK: Using the notes above as background, answer this for a new team "
        "member: when is it better to buy more capacity than to optimise what we "
        "already run? Explain your reasoning. Do not quote or restate the notes.\n\n"
        "ANSWER:\n",
    ),
    "code": (
        BENCHMARK_CONTEXT_CODE,
        lambda u: len(u) / 3.37,
        "SOURCE",
        "TASK: Using the source above as background, answer this for a colleague: "
        "how should we structure retries for background jobs that touch a shared "
        "database? Cover interfaces and failure handling, and explain your reasoning. "
        "Do not quote or restate the source.\n\nANSWER:\n",
    ),
    "chat": (
        BENCHMARK_CONTEXT_CHAT,
        lambda u: len(u.split()) * 1.33,
        "TRANSCRIPT",
        "TASK: Using the transcript above as background, answer this question from a "
        "new on-call engineer: what should I check first when latency rises after a "
        "deploy? Explain your reasoning. Do not quote or restate the transcript.\n\n"
        "ANSWER:\n",
    ),
}

BENCHMARK_TOKENS_PER_FILLER_WORD = 1.0

# Instruction wrapper is ~75 tokens; subtract it so the requested prompt size is roughly honoured.
BENCHMARK_PROMPT_OVERHEAD_TOKENS = 75

BENCHMARK_TASK_INSTRUCTION = (
    "TASK: Write a long, original, non-repetitive technical essay on the history of "
    "memory hierarchies in computer architecture: caches, TLBs, NUMA, HBM. "
    "Never repeat a sentence.\n\nESSAY:\n"
)


def build_benchmark_prompt(
    estimated_tokens: int, seed: int = 0, nonce: str = "", kind: str = "general"
) -> str:
    """Build a benchmark prompt that produces *representative* generation work.

    A prompt of one repeated token is a pathological best case for speculative
    decoding: the drafter predicts every token and reported throughput lands
    several times above what any real request sees. Padding with prose the model
    then copies back has the same effect. Opaque random filler avoids both, but
    overshoots into the worst case instead and understates decode throughput --
    see the note above BENCHMARK_CONTEXT_GENERAL for the measured gap.

    So: real text as background context, plus a task that demands novel output
    about something the context does not cover. kind selects the register the
    context comes from ("general", "code", "chat"); kind="filler" reproduces the
    pre-2026-09-20 opaque-filler prompt for comparison with old numbers.

    Deterministic per (estimated_tokens, seed, kind) so runs are comparable.
    """
    overhead = BENCHMARK_CONTEXT_OVERHEAD_TOKENS.get(kind, BENCHMARK_PROMPT_OVERHEAD_TOKENS)
    pad_tokens = max(0, estimated_tokens - overhead)
    # kind is folded into the seed so the same (tokens, seed) pair does not hand
    # every register an identically ordered context.
    rnd = random.Random(seed * 7919 + estimated_tokens + (zlib.crc32(kind.encode()) % 100003))

    if kind == "filler":
        legacy_pad = max(0, estimated_tokens - BENCHMARK_PROMPT_OVERHEAD_TOKENS)
        word_count = max(16, int(round(legacy_pad / BENCHMARK_TOKENS_PER_FILLER_WORD)))
        filler = " ".join(rnd.choice(BENCHMARK_FILLER_WORDS) for _ in range(word_count))
        prefix = f"RUN-ID {nonce}\n" if nonce else ""
        return (
            prefix
            + "You are a throughput benchmark harness. The block below is opaque filler. "
            "Do not read, repeat, quote, or refer to it.\n\nFILLER:\n"
            + filler
            + "\n\n"
            + BENCHMARK_TASK_INSTRUCTION
        )

    units, est, label, instruction = BENCHMARK_CONTEXT_KINDS.get(
        kind, BENCHMARK_CONTEXT_KINDS["general"]
    )
    # Sample without replacement, reshuffling only once the pool is exhausted, so a
    # large prompt repeats material but never places the same unit twice in a row.
    chosen: list[str] = []
    budget = 0.0
    bag: list[str] = []
    while budget < pad_tokens:
        if not bag:
            bag = list(units)
            rnd.shuffle(bag)
        unit = bag.pop()
        cost = est(unit)
        # Small targets: one unit can exceed the whole budget. Take it only if we
        # have nothing yet, or if it lands closer to the target than stopping does.
        if chosen and budget + cost - pad_tokens > pad_tokens - budget:
            break
        chosen.append(unit)
        budget += cost
    # The nonce sits at the very front so no block of the prompt can be served from a
    # prefix cache. A cached prefill measures cache lookup speed, not ingest speed.
    prefix = f"RUN-ID {nonce}\n" if nonce else ""
    joiner = "\n\n" if kind == "code" else "\n"
    return (
        prefix
        + "You are a throughput benchmark harness. The block below is background "
        "material. Use it only as context; do not quote or repeat it.\n\n"
        + label
        + ":\n"
        + joiner.join(chosen)
        + "\n\n"
        + instruction
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
    out = _strata_metrics_to_prometheus(out)
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
    kind: str = "general",
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
            "prompt": build_benchmark_prompt(
                prompt_tokens, seed, nonce=uuid.uuid4().hex[:16], kind=kind
            ),
            "max_tokens": n_predict,
            "min_tokens": n_predict,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True}
        }
        endpoint = f"http://127.0.0.1:{port}/v1/completions"

        # (2026-09-30) Strata has no /v1/completions: chat API instead, thinking off. It has no
        # ignore_eos either, so the generated count can end short of n_predict (usage reports
        # the real count). Recognised by its JSON /metrics.
        code_m, out_m = run_command([
            "docker", "exec", container, "curl", "-fsS", "--max-time", "5", *auth_args,
            f"http://127.0.0.1:{port}/metrics",
        ])
        if code_m == 0 and out_m.lstrip().startswith("{") and '"engine"' in out_m:
            payload = {
                "model": model_id,
                "messages": [{"role": "user", "content": payload["prompt"]}],
                "max_tokens": n_predict,
                "reasoning_effort": "none",
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            endpoint = f"http://127.0.0.1:{port}/v1/chat/completions"
    else:
        # llama.cpp / BeeLlama completions API.
        # cache_prompt=False defeats the prompt cache so prefill is actually measured;
        # ignore_eos pins predicted_n to n_predict. Sampling is left to the server config.
        payload = {
            "prompt": build_benchmark_prompt(prompt_tokens, seed, kind=kind),
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
        stream_timings = None   # llama.cpp-style per-request timings (Strata's last chunk)
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
            if isinstance(_chunk, dict) and isinstance(_chunk.get("timings"), dict):
                stream_timings = _chunk["timings"]
        
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

        # (2026-09-30) Strata reports this request's own phase times and MTP draft counts in
        # its final chunk: exact, and immune to other traffic, so they win over the deltas.
        draft_n = draft_accepted = None
        if stream_timings:
            p_ms, d_ms = stream_timings.get("prompt_ms"), stream_timings.get("predicted_ms")
            if isinstance(p_ms, (int, float)) and p_ms > 0 and isinstance(d_ms, (int, float)) and d_ms > 0:
                prefill_sec, decode_sec = p_ms / 1000.0, d_ms / 1000.0
                prompt_n = stream_timings.get("prompt_n") or prompt_n
                gen_n = stream_timings.get("predicted_n") or gen_n
                timing_source = "strata_timings"
                cached_n = stream_timings.get("cache_n") or 0
                if prompt_n + cached_n > 0:
                    cache_hit_ratio = cached_n / (prompt_n + cached_n)
            draft_n = stream_timings.get("draft_n")
            draft_accepted = stream_timings.get("draft_n_accepted")

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
                "draft_n": draft_n,
                "draft_n_accepted": draft_accepted,
                "draft_acceptance": (draft_accepted / draft_n) if draft_n and draft_accepted is not None else None,
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


# ── Benchmark request gate (added 2026-11-26) ────────────────────────────────
# Clean numbers need an idle engine. Every client reaches the model through an
# access-logger container (the single choke point; see access-logger.py "Benchmark
# gate"), which exposes a lease-based hold/release API on 127.0.0.1:8099 inside the
# container. Around a benchmark we: close the gate (new requests wait there), wait
# for the requests already past it to finish, run the benchmark directly against
# the engine, then reopen the gate so the held requests resume in order.
# The hold is a lease renewed every BENCHMARK_GATE_RENEW_SEC; if the agent dies the
# logger reopens on its own after BENCHMARK_GATE_TTL_SEC.
BENCHMARK_GATE_PORT = int(os.environ.get("BENCHMARK_GATE_PORT", "8099"))
BENCHMARK_GATE_TTL_SEC = int(os.environ.get("BENCHMARK_GATE_TTL_SEC", "120"))
BENCHMARK_GATE_RENEW_SEC = int(os.environ.get("BENCHMARK_GATE_RENEW_SEC", "30"))
BENCHMARK_DRAIN_TIMEOUT_SEC = int(os.environ.get("BENCHMARK_DRAIN_TIMEOUT_SEC", "900"))
BENCHMARK_GATE_ENABLED = os.environ.get("BENCHMARK_GATE", "true").lower() in ("1", "true", "yes", "on")

_GATE_CLIENT = (
    "import sys,urllib.request;"
    "r=urllib.request.Request(sys.argv[1],method=sys.argv[2],data=b'' if sys.argv[2]=='POST' else None);"
    "print(urllib.request.urlopen(r,timeout=5).read().decode())"
)


def find_access_logger_containers() -> list[str]:
    """Running containers that bind-mount access-logger.py (vLLM fleet or a llama.cpp stack)."""
    code, out = run_command(["docker", "ps", "-q"])
    ids = out.split() if code == 0 else []
    if not ids:
        return []
    code, out = run_command([
        "docker", "inspect", "--format",
        "{{.Name}}|{{range .Mounts}}{{.Source}};{{end}}", *ids,
    ])
    if code != 0:
        return []
    names = []
    for line in out.splitlines():
        name, _, mounts = line.partition("|")
        if any(m.endswith("/access-logger.py") for m in mounts.split(";")):
            names.append(name.lstrip("/"))
    return sorted(names)


def _gate_call(container: str, method: str, path: str) -> dict | None:
    url = f"http://127.0.0.1:{BENCHMARK_GATE_PORT}{path}"
    code, out = run_command(["docker", "exec", container, "python", "-c", _GATE_CLIENT, url, method])
    if code != 0:
        return None
    try:
        return json.loads(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None


class BenchmarkRequestGate:
    """Context manager: hold client traffic, drain in-flight requests, release on exit.

    `error` is set (and the gate already reopened) when draining timed out; the
    caller must then skip the benchmark."""

    def __init__(self, progress_cb=None):
        self.progress_cb = progress_cb
        self.containers: list[str] = []
        self.error: str | None = None
        self._stop = threading.Event()
        self._renewer: threading.Thread | None = None

    def _progress(self, msg: str) -> None:
        if callable(self.progress_cb):
            self.progress_cb(msg)

    def _hold_all(self) -> list[str]:
        ok = []
        for c in self.containers:
            if _gate_call(c, "POST", f"/gate/hold?ttl={BENCHMARK_GATE_TTL_SEC}&owner=gpu-monitor-benchmark"):
                ok.append(c)
        return ok

    def _renew_loop(self) -> None:
        while not self._stop.wait(BENCHMARK_GATE_RENEW_SEC):
            self._hold_all()

    def _release_all(self) -> int:
        released = 0
        for c in self.containers:
            st = _gate_call(c, "POST", "/gate/release")
            if st is None:
                _append_log_event("warning", "benchmark",
                                  f"Could not reopen request gate on {c}; it reopens itself "
                                  f"within {BENCHMARK_GATE_TTL_SEC}s")
            else:
                released += int(st.get("waiting") or 0)
        return released

    def __enter__(self) -> "BenchmarkRequestGate":
        if not BENCHMARK_GATE_ENABLED:
            return self
        found = find_access_logger_containers()
        if not found:
            _append_log_event("warning", "benchmark",
                              "No access-logger container found; benchmarking without holding client requests")
            return self
        self.containers = found
        held = self._hold_all()
        missing = sorted(set(found) - set(held))
        if missing:
            _append_log_event("warning", "benchmark",
                              f"Request gate unavailable on {', '.join(missing)} (restart it to load the "
                              "gate-capable access-logger.py); its traffic is NOT held")
        self.containers = held
        if not held:
            return self
        _append_log_event("info", "benchmark", f"Holding new client requests at {', '.join(held)}")
        self._renewer = threading.Thread(target=self._renew_loop, daemon=True, name="bench-gate-renew")
        self._renewer.start()
        try:
            self._drain()
        except BaseException:
            self._close()
            raise
        return self

    def _drain(self) -> None:
        held = self.containers
        deadline = time.time() + BENCHMARK_DRAIN_TIMEOUT_SEC
        while True:
            states = [_gate_call(c, "GET", "/gate") for c in held]
            active = sum(int((st or {}).get("active") or 0) for st in states)
            waiting = sum(int((st or {}).get("waiting") or 0) for st in states)
            if active == 0 and all(st is not None for st in states):
                break
            if time.time() >= deadline:
                self.error = (f"{active} in-flight request(s) still running after "
                              f"{BENCHMARK_DRAIN_TIMEOUT_SEC}s; benchmark skipped")
                self._close()
                return
            self._progress(f"Benchmark: waiting for {active} in-flight request(s) to finish"
                           f" ({waiting} new request(s) held)...")
            time.sleep(1.0)
        _append_log_event("info", "benchmark", "In-flight requests drained; engine idle")

    def _close(self) -> None:
        if not self.containers:
            return
        self._stop.set()
        released = self._release_all()
        _append_log_event("info", "benchmark",
                          f"Request gate reopened; {released} held request(s) resumed")
        self.containers = []

    def __exit__(self, *_exc) -> bool:
        self._close()
        return False


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
                    progress_cb(
                        f"full benchmark: pass {idx}/{len(specs)} "
                        f"({spec['name']}/{spec.get('kind', 'general')}, run {rep + 1}/{repeats})..."
                    )
                run, error = run_single_benchmark(
                    active_key,
                    containers,
                    prompt_tokens=spec["prompt_tokens"],
                    n_predict=spec["n_predict"],
                    seed=rep,
                    kind=spec.get("kind", "general"),
                )
                if error:
                    return None, f"Full benchmark failed on {spec['name']} (run {rep + 1}/{repeats}): {error}"
                run["name"] = f"{spec['name']}#{rep + 1}"
                run["requested_prompt_tokens"] = spec["prompt_tokens"]
                run["requested_n_predict"] = spec["n_predict"]
                run["kind"] = spec.get("kind", "general")
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


def start_switch(model_key: str, force: bool = False) -> tuple[bool, str]:
    """Switch to model_key. force=True restarts it even when it is already healthy."""
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
            force=force,
        )

    # Handle BeeLlama switching
    if model.get("beellama"):
        return start_beellama_switch(model_key, force=force)

    current_containers = list_llama_compose_containers()
    current_active = detect_active_model_key(current_containers)
    if not force and current_active == model_key and is_model_server_healthy(model_key, current_containers):
        return False, f"{model['label']} is already active and healthy"

    with STATE_LOCK:
        if STATE["switch_in_progress"]:
            return False, "A model switch is already in progress"
        STATE["switch_in_progress"] = True
        STATE["force_restart"] = force
        STATE["last_requested_model"] = model_key
        STATE["last_started_at"] = now_iso()
        STATE["last_message"] = f"Switching to {model['label']}..."
    _append_log_event("info", "switch", f"Switch requested: {model['label']}")

    def _worker() -> None:
        switch_started_ts = time.time()
        try:
            # (2026-11-02) Build missing images FIRST, while the current stack still serves;
            # only then stop BeeLlama / let switch-llm.sh down the old stack.
            compose_path = model["compose"]

            def _img_status(msg: str) -> None:
                with STATE_LOCK:
                    STATE["last_message"] = msg

            img_ok, img_msg = ensure_stack_images(compose_path, cwd=LLAMA_DIR, status_cb=_img_status)
            if not img_ok:
                with STATE_LOCK:
                    STATE["last_exit_code"] = 1
                    STATE["last_output"] = img_msg[-4000:]
                _clear_switch_state(f"Image build failed for {model['label']}: {img_msg[:120]}")
                return
            # Stop BeeLlama if running before starting a llama.cpp model
            if BEELLAMA_DIR.is_dir():
                _stop_beellama()
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


def _start_vllm_proxy_fleet(compose_path: Path) -> None:
    """Bring the shared proxy fleet up for stacks that do not bake it in.

    Proxies, routers and the access logger were extracted to llm-proxies.yml on
    2026-09-08. Stacks written after that date carry ONLY the model server, so
    the plain `up -d --remove-orphans` in start_vllm_switch() leaves nothing
    listening on 8080/11434/28080 — and reaps the fleet as an orphan when it was
    already running. Mirrors start_stack() in switch-vllm.sh.
    """
    proxies_file = VLLM_DIR / "llm-proxies.yml"
    if not proxies_file.is_file():
        _append_log_event(
            "warning", "switch",
            f"{proxies_file} is missing; only the model server is up and "
            "nothing is published on 8080.")
        return

    # Older stacks still declare the fleet themselves — starting it twice would
    # collide on container names and ports.
    exit_code, output = run_command([
        "bash", "-lc",
        f"docker compose --project-directory {shlex.quote(str(VLLM_DIR))} "
        f"-f {shlex.quote(str(compose_path))} config --services",
    ])
    if exit_code == 0 and "access-logger" in output.split():
        return

    _append_log_event("info", "switch",
                      "Stack ships no proxies — starting llm-proxies.yml...")
    # COMPOSE_IGNORE_ORPHANS: this invocation sees the model server as an orphan
    # (different file, same project) and would nag on every switch. Do NOT "fix"
    # that with --remove-orphans — it would delete the server.
    # /props is synthesized by access-logger from PROPS_* env, which llm-proxies.yml
    # reads from MAX_SEQS/TEMP/TOP_P/TOP_K/MIN_P. Feed them from this stack's LLM_META
    # (same as switch-vllm.sh) so /props describes the stack actually running instead of
    # the fleet defaults (8 slots, temp 0.7) — e.g. Strata serves 1 stream.
    meta = parse_llm_meta(compose_path)
    props_env = []
    for field, env_name in (("max_seqs", "MAX_SEQS"), ("props_temp", "TEMP"),
                            ("props_top_p", "TOP_P"), ("props_top_k", "TOP_K"),
                            ("props_min_p", "MIN_P")):
        if meta.get(field):
            props_env.append(f"{env_name}={shlex.quote(meta[field])}")
    if props_env:
        _append_log_event("info", "switch", "Proxy /props from stack: " + " ".join(props_env))
    exit_code, output = run_command([
        "bash", "-lc",
        f"{' '.join(props_env)} COMPOSE_IGNORE_ORPHANS=True docker compose -p vllm "
        f"--project-directory {shlex.quote(str(VLLM_DIR))} "
        f"-f {shlex.quote(str(proxies_file))} up -d",
    ])
    if exit_code != 0:
        _append_log_event(
            "error", "switch",
            f"Proxy fleet start failed (exit {exit_code}): {output[-500:]}")


def start_vllm_switch(
    compose_file: str | None = None,
    model_key: str = "vllm",
    label: str = "vLLM",
    force: bool = False,
) -> tuple[bool, str]:
    """Start/restart a vLLM stack (stop llama.cpp or a different vLLM stack first).

    force=True also takes down and recreates the stack when it is the active one."""
    if compose_file is None:
        compose_file = str(VLLM_DIR / "docker-compose.yml")

    current_containers = list_llama_compose_containers()
    current_active = detect_active_model_key(current_containers)
    vllm_containers = list_vllm_compose_containers()

    if not force and current_active == model_key and is_vllm_healthy(vllm_containers):
        return False, f"{label} is already active and healthy"

    with STATE_LOCK:
        if STATE["switch_in_progress"]:
            return False, "A model switch is already in progress"
        STATE["switch_in_progress"] = True
        STATE["force_restart"] = force
        STATE["last_requested_model"] = model_key
        STATE["last_started_at"] = now_iso()
        STATE["last_message"] = f"Switching to {label}..."
    _append_log_event("info", "switch", f"Switch requested: {label}")

    def _vllm_worker() -> None:
        switch_started_ts = time.time()
        try:
            # (2026-11-02) Rebuild missing images while the current stack still serves.
            def _img_status(msg: str) -> None:
                with STATE_LOCK:
                    STATE["last_message"] = msg

            img_ok, img_msg = ensure_stack_images(Path(compose_file), project="vllm", cwd=VLLM_DIR, status_cb=_img_status)
            if not img_ok:
                with STATE_LOCK:
                    STATE["last_exit_code"] = 1
                    STATE["last_output"] = img_msg[-4000:]
                _clear_switch_state(f"Image build failed for {label}: {img_msg[:120]}")
                return
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
            elif _key_is_vllm(current_active) and (current_active != model_key or force):
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

            # The model server is up; the proxy fleet lives in its own file and
            # was just reaped by --remove-orphans (or never started).
            _start_vllm_proxy_fleet(compose_path)

            # Poll until healthy
            deadline = time.time() + SWITCH_READY_TIMEOUT_SEC
            while time.time() < deadline:
                time.sleep(SWITCH_POLL_SEC)
                local_vllm = list_vllm_compose_containers()
                if is_vllm_healthy(local_vllm, compose_path.name):
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


def start_beellama_switch(model_key: str, force: bool = False) -> tuple[bool, str]:
    """Start BeeLlama (stop llama.cpp first, then docker compose up -d)."""
    models, _ = get_models()
    model = models.get(model_key, {})
    current_containers = list_llama_compose_containers()
    current_active = detect_active_model_key(current_containers)
    if not force and current_active == model_key and is_model_server_healthy(model_key, current_containers):
        return False, f"{model.get('label', 'BeeLlama')} is already active and healthy"

    with STATE_LOCK:
        if STATE["switch_in_progress"]:
            return False, "A model switch is already in progress"
        STATE["switch_in_progress"] = True
        STATE["force_restart"] = force
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

    def _num(v: str):
        # nvidia-smi prints "[N/A]" / "[Not Supported]" for fields a GPU lacks (laptop
        # GPUs: fan speed and power limit). One missing field must not drop the GPU.
        try:
            return float(v)
        except ValueError:
            return None

    gpus = []
    for line in output.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 13:
            try:
                index = int(parts[0])
            except ValueError:
                continue
            gpus.append({
                "index": index,
                "name": parts[1],
                "temp": _num(parts[2]),
                "util": _num(parts[3]),
                "mem_util": _num(parts[4]),
                "mem_total": _num(parts[5]),
                "mem_used": _num(parts[6]),
                "mem_free": _num(parts[7]),
                "fan": _num(parts[8]),
                "power": _num(parts[9]),
                "power_limit": _num(parts[10]),
                "clock_gfx": _num(parts[11]),
                "clock_mem": _num(parts[12]),
            })
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
                # Scheduler gauges, added 2026-09-27. ALTER rather than CREATE so an existing
                # samples table keeps its history; old rows read back as NULL.
                have = {r[1] for r in conn.execute("PRAGMA table_info(samples)")}
                for col in ("running", "waiting", "ctx_pct"):
                    if col not in have:
                        conn.execute(f"ALTER TABLE samples ADD COLUMN {col} REAL")
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
                # Last KV pool the engine actually allocated, per model. The pool size is
                # only knowable while the stack is up, but it is the figure you want when
                # deciding WHICH stack to boot, so it has to outlive the container.
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS kv_pools (
                        model_key       TEXT PRIMARY KEY,
                        ts              INTEGER NOT NULL,
                        pool_tokens     INTEGER,
                        max_concurrency REAL,
                        ctx_size        INTEGER
                    )
                    """
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


def note_live_context(context_info: dict) -> None:
    """Work out the charted context % and cache it for the sampler (2026-09-29/30).

    Order: a live per-request figure (TabbyAPI shim; climbs during prefill) -> 0 when that
    engine reports nothing in flight -> KV-pool share (vLLM, no per-request gauge) ->
    llama.cpp slot n_past / n_ctx. Stored back as context_info["ctx_pct"] so the
    browser's live chart shows exactly what the sampler stores.
    """
    pct = None
    n_past, n_ctx = context_info.get("n_past"), context_info.get("n_ctx")
    used, pool = context_info.get("pool_used"), context_info.get("pool_tokens")
    have_window = isinstance(n_past, (int, float)) and isinstance(n_ctx, (int, float)) and n_ctx > 0
    if context_info.get("n_past_live") and have_window:
        pct = 100.0 * float(n_past) / float(n_ctx)
    elif context_info.get("live_ctx_supported"):
        pct = 0.0
    elif isinstance(used, (int, float)) and isinstance(pool, (int, float)) and pool > 0:
        pct = 100.0 * float(used) / float(pool)
    elif not context_info.get("n_past_is_average") and have_window:
        pct = 100.0 * float(n_past) / float(n_ctx)
    if pct is not None:
        pct = max(0.0, min(100.0, pct))
    context_info["ctx_pct"] = pct
    with METRICS_LIVE_LOCK:
        METRICS_LIVE["ctx_pct"] = pct


def note_live_throughput(tps, ingest, running=None, waiting=None) -> None:
    """Cache the newest throughput reading for the sampler.

    Throughput costs a docker exec to measure; the status endpoint already pays for it
    on every poll, so the sampler reuses that instead of doubling the load.
    """
    with METRICS_LIVE_LOCK:
        METRICS_LIVE["ts"] = time.time()
        METRICS_LIVE["tps"] = float(tps) if isinstance(tps, (int, float)) else None
        METRICS_LIVE["ingest"] = float(ingest) if isinstance(ingest, (int, float)) else None
        METRICS_LIVE["running"] = float(running) if isinstance(running, (int, float)) else None
        METRICS_LIVE["waiting"] = float(waiting) if isinstance(waiting, (int, float)) else None


def record_metric_sample() -> None:
    gpus = get_gpu_stats()
    if not gpus:
        return
    gpu = gpus[0]

    with METRICS_LIVE_LOCK:
        fresh = (time.time() - METRICS_LIVE["ts"]) <= METRICS_LIVE_MAX_AGE_SEC
        tps = METRICS_LIVE["tps"] if fresh else None
        ingest = METRICS_LIVE["ingest"] if fresh else None
        running = METRICS_LIVE["running"] if fresh else None
        waiting = METRICS_LIVE["waiting"] if fresh else None
        ctx_pct = METRICS_LIVE["ctx_pct"] if fresh else None

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
        running,
        waiting,
        ctx_pct,
    )
    with METRICS_DB_LOCK:
        conn = _metrics_connect()
        try:
            # One row per wall-clock second; REPLACE keeps a restart from colliding.
            conn.execute(
                "INSERT OR REPLACE INTO samples "
                "(ts, util, vram_pct, vram_mb, tps, ingest, power, temp, fan, running, waiting, ctx_pct) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
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
                       MAX(ingest)         AS ingest_max,
                       AVG(waiting)        AS waiting,
                       MAX(waiting)        AS waiting_max,
                       MAX(running)        AS running_max,
                       AVG(ctx_pct)        AS ctx_pct,
                       MAX(ctx_pct)        AS ctx_pct_max
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
            "waiting": rnd(r[10]),
            "waiting_max": rnd(r[11]),
            "running_max": rnd(r[12]),
            "ctx_pct": rnd(r[13]),
            "ctx_pct_max": rnd(r[14]),
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


_KV_POOL_LAST_WRITE: dict[str, tuple[int, float]] = {}
_KV_POOL_WRITE_EVERY_SEC = 900


def record_kv_pool(model_key: str, pool_tokens, max_concurrency, ctx_size) -> None:
    """Persist the KV pool the live engine reports for this model.

    Written on change, and otherwise at most every 15 minutes, so the status poll
    (a few seconds) does not turn into a write loop.
    """
    if not METRICS_DB_READY or not model_key or not pool_tokens:
        return
    try:
        tokens = int(pool_tokens)
    except (TypeError, ValueError):
        return
    if tokens <= 0:
        return
    now = time.time()
    prev = _KV_POOL_LAST_WRITE.get(model_key)
    if prev and prev[0] == tokens and (now - prev[1]) < _KV_POOL_WRITE_EVERY_SEC:
        return
    try:
        with METRICS_DB_LOCK:
            conn = _metrics_connect()
            try:
                conn.execute(
                    "INSERT INTO kv_pools (model_key, ts, pool_tokens, max_concurrency, ctx_size) "
                    "VALUES (?,?,?,?,?) "
                    "ON CONFLICT(model_key) DO UPDATE SET ts=excluded.ts, "
                    "pool_tokens=excluded.pool_tokens, max_concurrency=excluded.max_concurrency, "
                    "ctx_size=excluded.ctx_size",
                    (model_key, int(now), tokens,
                     float(max_concurrency) if isinstance(max_concurrency, (int, float)) else None,
                     int(ctx_size) if isinstance(ctx_size, int) else None),
                )
                conn.commit()
            finally:
                conn.close()
        _KV_POOL_LAST_WRITE[model_key] = (tokens, now)
    except Exception as exc:  # noqa: BLE001
        _append_log_event("warning", "metrics", f"Could not persist KV pool size: {exc}")


def model_kv_pool_summary() -> dict:
    """Last observed KV pool per model, for the sidebar cards."""
    if not METRICS_DB_READY or not os.path.exists(METRICS_DB):
        return {}
    try:
        conn = _metrics_connect(readonly=True)
        try:
            conn.execute("PRAGMA query_only=ON")
            rows = conn.execute(
                "SELECT model_key, ts, pool_tokens, max_concurrency, ctx_size FROM kv_pools"
            ).fetchall()
        finally:
            conn.close()
    except Exception:  # noqa: BLE001
        return {}
    return {
        r[0]: {
            "ts": r[1],
            "pool_tokens": r[2],
            "max_concurrency": r[3],
            "ctx_size": r[4],
        }
        for r in rows
    }


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

            # A forced restart starts with the old (healthy) container still up; only the
            # worker knows when the new one is ready.
            if switched and STATE.get("force_restart"):
                switched = False
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
    m = models.get(active_key) if active_key else None
    if m is not None:
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
            "backend": m.get("backend") or ("vllm" if m.get("vllm") else "llamacpp"),
        }
    if m is not None and m.get("vllm"):
        try:
            VLLM_ACTIVE_PORT["port"] = int(m.get("internal_port") or 8080)
        except (TypeError, ValueError):
            VLLM_ACTIVE_PORT["port"] = 8080
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
                   "offload_load_count", "offload_tier_queries",
                   "offload_cpu_usage_perc", "success_by_reason", "waiting_by_reason",
                   "last_request_tps", "kv_tokens_used", "kv_cache_size_tokens", "generation_tokens_total",
                   "prompt_tokens_total", "total_requests_approx", "state", "gauge_age_sec"):
            model_stats[_k] = throughput.get(_k)
    # Hand the freshly-measured throughput to the metrics sampler so it never has to
    # run its own docker exec to get the same numbers.
    note_live_throughput(
        model_stats.get("live_tps"),
        model_stats.get("last_live_ingest_tps") or model_stats.get("last_ingest_tps"),
        model_stats.get("requests_running"),
        model_stats.get("requests_waiting"),
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
            # Remember it: once the stack is down the pool figure is unrecoverable.
            record_kv_pool(active_key, kv_tokens,
                           throughput.get("kv_max_concurrency"), ctx_size)

            avg_prompt = throughput.get("avg_prompt_per_request")
            avg_gen = throughput.get("avg_gen_per_request")
            if avg_prompt is not None:
                context_info["n_past"] = min(ctx_size, int(round(avg_prompt + (avg_gen or 0))))
                context_info["n_past_is_average"] = True
            else:
                # No completed request yet on this boot — leave it unknown rather than
                # substituting a pool number that means something else.
                context_info["n_past"] = None
            live_ctx = throughput.get("live_ctx_tokens")
            if live_ctx is not None:
                context_info["live_ctx_supported"] = True
                live_target = throughput.get("live_ctx_target") or 0
                if live_target > 0:
                    context_info["n_past"] = min(ctx_size, int(live_ctx))
                    context_info["n_past_target"] = min(ctx_size, int(live_target))
                    context_info["n_past_is_average"] = False
                    context_info["n_past_live"] = True
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
    if (not context_info.get("n_past_live")
            and (context_info["n_past"] is None or (isinstance(context_info["n_past"], int) and context_info["n_past"] < 10))):
        last_good = ctx_state.get("last_good_n_past")
        if last_good is not None and last_good >= 10:
            context_info["n_past"] = last_good
    note_live_context(context_info)

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
    kv_pool_by_model = model_kv_pool_summary()

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
                "backend": m.get("backend") or ("vllm" if m.get("vllm") else "llamacpp"),
                "max_seqs": m.get("max_seqs", ""),
                # Latest successful benchmark for this model, so the sidebar card can
                # show measured speeds without the model being loaded.
                "bench": bench_by_model.get(key),
                # Last KV pool this stack allocated when it was up, so the card shows the
                # likely pool size before you boot it.
                "kv_pool": kv_pool_by_model.get(key),
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




HOST_SHM_DIR = Path(os.environ.get("HOST_SHM_DIR", "/host-shm"))
SHM_SWEEP_INTERVAL_S = int(os.environ.get("SHM_SWEEP_INTERVAL_S", "60"))
SHM_SWEEP_MIN_AGE_S = 120


_VLLM_ENGINE_IMAGE_RE = re.compile(r"vllm|hyperqwen|qwen38-27b|gemma4-31b", re.I)


def _vllm_engine_running() -> bool | None:
    """True if any running container looks like a vLLM engine (image name or a
    VLLM_* env var). None if docker could not be queried (fail safe)."""
    try:
        ids = subprocess.run(["docker", "ps", "-q"], capture_output=True, text=True,
                             timeout=20, check=True).stdout.split()
        if not ids:
            return False
        out = subprocess.run(["docker", "inspect", *ids], capture_output=True, text=True,
                             timeout=30, check=True).stdout
        me = os.environ.get("HOSTNAME", "")
        for c in json.loads(out):
            if me and c.get("Id", "").startswith(me):
                continue  # gpu-monitor itself carries VLLM_DIR etc.
            cfg = c.get("Config") or {}
            img = cfg.get("Image", "")
            env = cfg.get("Env") or []
            if any(e.startswith("VLLM_") for e in env):
                return True
            # proxies live in the vllm compose project too: match engine images only
            if _VLLM_ENGINE_IMAGE_RE.search(img) and "proxy" not in img:
                return True
        return False
    except Exception as e:
        print(f"shm-sweep: docker query failed: {e}")
        return None


def sweep_orphan_vllm_shm() -> None:
    """Delete leaked vLLM KV-offload regions. With ipc: host a stopped vLLM
    engine leaves a ~24 GiB /dev/shm/vllm_offload_*.mmap in host RAM/swap."""
    if not HOST_SHM_DIR.is_dir():
        return
    now = time.time()
    cands = {p.name: p for p in HOST_SHM_DIR.glob("vllm_offload_*.mmap")
             if now - p.stat().st_mtime > SHM_SWEEP_MIN_AGE_S}
    if not cands:
        return
    # Only one vLLM engine runs at a time on this host. If none is running, no
    # process can own an offload region, so every one left is a leak. While an
    # engine runs, leave everything alone (its own boot sweep handles older ones).
    if _vllm_engine_running() is not False:
        return
    for name, p in cands.items():
        try:
            size_gb = p.stat().st_size / 2**30
            p.unlink()
            msg = f"Removed orphaned vLLM shm region {name} ({size_gb:.1f} GiB)"
            print(f"shm-sweep: {msg}")
            try:
                _append_log_event("info", "shm-sweep", msg)
            except Exception:
                pass
        except FileNotFoundError:
            pass
        except OSError as e:
            print(f"shm-sweep: failed to remove {name}: {e}")


def _run_shm_sweeper() -> None:
    while True:
        try:
            sweep_orphan_vllm_shm()
        except Exception as e:
            print(f"shm-sweep: error: {e}")
        time.sleep(SHM_SWEEP_INTERVAL_S)


