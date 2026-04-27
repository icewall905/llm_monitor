#!/usr/bin/env python3
import ast
import json
import os
import re
import select
import shlex
import subprocess
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HOST = "0.0.0.0"
PORT = int(os.environ.get("DASHBOARD_PORT", "8080"))


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
SWITCH_READY_TIMEOUT_SEC = int(os.environ.get("SWITCH_READY_TIMEOUT_SEC", "900"))
SWITCH_POLL_SEC = float(os.environ.get("SWITCH_POLL_SEC", "5"))
SWITCH_STALE_SEC = int(
    os.environ.get("SWITCH_STALE_SEC", str(max(SWITCH_READY_TIMEOUT_SEC + 120, 1020)))
)
THROUGHPUT_LOG_TAIL_LINES = int(os.environ.get("THROUGHPUT_LOG_TAIL_LINES", "300"))
THROUGHPUT_CACHE_TTL_SEC = float(os.environ.get("THROUGHPUT_CACHE_TTL_SEC", "2"))
BENCHMARK_TIMEOUT_SEC = int(os.environ.get("BENCHMARK_TIMEOUT_SEC", "180"))
BENCHMARK_PROMPT_TOKENS = int(os.environ.get("BENCHMARK_PROMPT_TOKENS", "768"))
BENCHMARK_N_PREDICT = int(os.environ.get("BENCHMARK_N_PREDICT", "256"))
FULL_BENCHMARK_MAX_HISTORY = int(os.environ.get("BENCHMARK_HISTORY_LIMIT", "10"))
BENCHMARK_STALE_SEC = int(
    os.environ.get("BENCHMARK_STALE_SEC", str(max(BENCHMARK_TIMEOUT_SEC * 4, 600)))
)
LOG_MAX_EVENTS = int(os.environ.get("LOG_MAX_EVENTS", "100"))
LOG_WATCHER_INTERVAL_SEC = float(os.environ.get("LOG_WATCHER_INTERVAL_SEC", "2.0"))
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
    if not STACKS_DIR.is_dir():
        return models
    entries = []
    for path in STACKS_DIR.glob("*.yml"):
        meta = parse_llm_meta(path)
        if not meta.get("display_name"):
            continue
        rel = f"stacks/{path.name}"
        try:
            order = int(meta.get("sort_order", 999))
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
            "family":         meta.get("family", "other"),
            "ctx_size":       meta.get("ctx_size", ""),
            "quant":          meta.get("quant", ""),
            "params":         meta.get("params", ""),
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
LIVE_TPS_CACHE_TTL_SEC = float(os.environ.get("LIVE_TPS_CACHE_TTL_SEC", "0.5"))
LIVE_TPS_CACHE_LOCK = threading.Lock()
LIVE_TPS_CACHE = {
    "active_key": None,
    "container": None,
    "checked_at": 0.0,
    "result": None,
}
LIVE_INGEST_STATE_LOCK = threading.Lock()
LIVE_INGEST_STATE = {
    "active_key": None,
    "container": None,
    "sampled_at": 0.0,
    "n_past": None,
}

EVAL_TPS_PATTERN = re.compile(
    r"eval time\s*=.*?\(\s*[0-9.]+\s+ms per token,\s*([0-9.]+)\s+tokens per second\)"
)
INGEST_TPS_PATTERN = re.compile(
    r"prompt eval time\s*=.*?\(\s*[0-9.]+\s+ms per token,\s*([0-9.]+)\s+tokens per second\)"
)
TIMING_TASK_PATTERN = re.compile(r"slot print_timing: id\s+\d+\s+\|\s+task\s+(\d+)\s+\|")
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


def _process_log_line(raw_line: str) -> None:
    match = _LOG_TS_RE.search(raw_line)
    ts_iso = match.group(0).strip() if match else None
    clean = _LOG_TS_RE.sub("", raw_line.strip(), count=1)
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


def _ingest_log_tail(container: str, tail: int = 50) -> None:
    code, output = run_command(["docker", "logs", "--tail", str(tail), container])
    if code == 0:
        for line in output.splitlines():
            _process_log_line(line)


def run_command(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    combined = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, combined.strip()


def _run_log_watcher() -> None:
    """Daemon: streams docker logs and emits events into LOG_STATE."""
    proc = None
    current_container: str | None = None
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
                    current_container = None
                time.sleep(LOG_WATCHER_INTERVAL_SEC)
                continue

            target = find_model_server_container_name(active_key, containers)
            if not target:
                time.sleep(LOG_WATCHER_INTERVAL_SEC)
                continue

            _check_restart_count(target)

            for c in containers:
                if c["name"] == target and "(unhealthy)" in c.get("status", ""):
                    if not _has_recent_event("health", 30):
                        _append_log_event(
                            "warning", "health",
                            f"Container {target} is unhealthy — health check failing",
                        )

            if target != current_container:
                if proc is not None:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    proc = None
                _ingest_log_tail(target, tail=50)
                proc = subprocess.Popen(
                    ["docker", "logs", "--follow", "--tail", "0", "--timestamps", target],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
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
                    _process_log_line(line)
                    count += 1
                if proc.poll() is not None:
                    proc = None
                    current_container = None
            else:
                time.sleep(LOG_WATCHER_INTERVAL_SEC)

        except Exception:
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
    return None


def is_model_server_healthy(model_key: str, containers: list[dict] | None = None) -> bool:
    models, _ = get_models()
    model = models.get(model_key)
    if not model:
        return False
    records = containers if containers is not None else list_llama_compose_containers()
    target_compose = model["compose"]
    target_service = model["server_service"]

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


def parse_latest_completion(log_text: str) -> tuple[float | None, float | None, int | None, str | None, str | None, float | None]:
    latest_gen_tps = None
    latest_ingest_tps = None
    latest_task_id = None
    latest_eval_line = None
    latest_ingest_line = None
    latest_ts_f = None
    current_task_id = None
    for line in log_text.splitlines():
        # Handle timestamp prefix
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
            val = float(ingest_match.group(1))
            if val < 20000: # Sanity check
                latest_ingest_tps = val
                latest_ingest_line = line.strip()
                if line_ts_f: latest_ts_f = line_ts_f

        match = EVAL_TPS_PATTERN.search(line)
        if match:
            val = float(match.group(1))
            if val < 20000: # Sanity check
                latest_gen_tps = val
                latest_task_id = current_task_id
                latest_eval_line = line.strip()
                if line_ts_f: latest_ts_f = line_ts_f

    return latest_gen_tps, latest_ingest_tps, latest_task_id, latest_eval_line, latest_ingest_line, latest_ts_f


def parse_live_tps_from_slots(slots_text: str, active_key: str, container: str) -> dict:
    try:
        slots = json.loads(slots_text)
    except Exception:
        return {
            "tokens_per_second": None,
            "n_ctx": None,
            "n_past": None,
            "live_ingest_tps": None,
            "state": "error",
            "detail": "Failed to parse /slots response",
        }

    if not isinstance(slots, list):
        return {
            "tokens_per_second": None,
            "n_ctx": None,
            "n_past": None,
            "live_ingest_tps": None,
            "state": "error",
            "detail": "Unexpected /slots format",
        }

    processing = False
    decoded_tokens = 0
    n_ctx = None
    n_past = None
    for slot in slots:
        if not isinstance(slot, dict):
            continue
        slot_n_ctx = slot.get("n_ctx")
        slot_n_past = slot.get("n_past")
        if isinstance(slot_n_ctx, int) and (n_ctx is None or slot_n_ctx > n_ctx):
            n_ctx = slot_n_ctx
        if not slot.get("is_processing"):
            continue
        processing = True
        if isinstance(slot_n_past, int):
            n_past = max(n_past or 0, slot_n_past)
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
            with LIVE_INGEST_STATE_LOCK:
                LIVE_INGEST_STATE["active_key"] = active_key
                LIVE_INGEST_STATE["container"] = container
                LIVE_INGEST_STATE["sampled_at"] = now
                LIVE_INGEST_STATE["n_past"] = None
            return {
                "tokens_per_second": 0.0,
                "n_ctx": n_ctx,
                "n_past": None,
                "live_ingest_tps": None,
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

    live_ingest_tps = None
    with LIVE_INGEST_STATE_LOCK:
        prev_valid_ingest = (
            LIVE_INGEST_STATE["active_key"] == active_key
            and LIVE_INGEST_STATE["container"] == container
            and LIVE_INGEST_STATE["n_past"] is not None
            and LIVE_INGEST_STATE["sampled_at"] > 0
        )
        if prev_valid_ingest and n_past is not None:
            dt_ingest = now - float(LIVE_INGEST_STATE["sampled_at"])
            prev_n_past = int(LIVE_INGEST_STATE["n_past"])
            if dt_ingest > 0 and n_past > prev_n_past:
                live_ingest_tps = (n_past - prev_n_past) / dt_ingest
        LIVE_INGEST_STATE["active_key"] = active_key
        LIVE_INGEST_STATE["container"] = container
        LIVE_INGEST_STATE["sampled_at"] = now
        LIVE_INGEST_STATE["n_past"] = n_past

    if live_tps is None:
        return {
            "tokens_per_second": None,
            "n_ctx": n_ctx,
            "n_past": n_past,
            "live_ingest_tps": live_ingest_tps,
            "state": "warming",
            "detail": "Collecting live TPS sample",
        }
    return {
        "tokens_per_second": live_tps,
        "n_ctx": n_ctx,
        "n_past": n_past,
        "live_ingest_tps": live_ingest_tps,
        "state": "ok",
        "detail": "Live decode throughput from /slots",
    }


def fetch_live_tps(active_key: str | None, container: str) -> dict:
    code, output = run_command(
        ["docker", "exec", container, "curl", "-fsS", "http://127.0.0.1:8080/slots"]
    )
    if code != 0:
        return {
            "tokens_per_second": None,
            "n_ctx": None,
            "n_past": None,
            "live_ingest_tps": None,
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
        "n_past": parsed.get("n_past"),
        "live_ingest_tps": parsed.get("live_ingest_tps"),
        "source": "slots",
        "updated_at": now_iso(),
        "container": container,
        "state": parsed["state"],
        "detail": parsed["detail"],
    }


def build_live_throughput_status(active_key: str | None, containers: list[dict]) -> dict:
    if not active_key:
        return {
            "tokens_per_second": None,
            "n_ctx": None,
            "n_past": None,
            "live_ingest_tps": None,
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
            "n_past": None,
            "live_ingest_tps": None,
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
        gen_tps, ingest_tps, completion_id, completion_line, ingest_line, ts_f = parse_latest_completion(output)
        completion_key = (
            f"task:{completion_id}" if completion_id is not None else f"line:{completion_line}"
        )
        if gen_tps is None:
            result = {
                "tokens_per_second": None,
                "ingest_tps": None,
                "completion_id": None,
                "completion_key": None,
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

        live_ingest = live.get("live_ingest_tps")
        if isinstance(live_ingest, (int, float)) and live_ingest > 0:
            stats["last_live_ingest_tps"] = float(live_ingest)
            if stats["best_ingest_tps"] is None or float(live_ingest) > stats["best_ingest_tps"]:
                stats["best_ingest_tps"] = float(live_ingest)

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
            if isinstance(completion_ingest, (int, float)):
                stats["last_ingest_tps"] = float(completion_ingest)
                if stats["best_ingest_tps"] is None or float(completion_ingest) > stats["best_ingest_tps"]:
                    stats["best_ingest_tps"] = float(completion_ingest)
            stats["last_completed_at"] = now
            stats["last_completed_at_f"] = completed.get("ts_f")
            stats["last_completed_completion_key"] = completion_key
            stats["completed_count"] += 1
            stats["completed_sum_tps"] += float(completion_tps)
            stats["average_rate_tps"] = stats["completed_sum_tps"] / stats["completed_count"]

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


def run_single_benchmark(
    active_key: str,
    containers: list[dict],
    *,
    prompt_tokens: int,
    n_predict: int,
) -> tuple[dict | None, str | None]:
    container = find_model_server_container_name(active_key, containers)
    if not container:
        return None, "Active model server container not found"

    payload = {
        "prompt": build_benchmark_prompt(prompt_tokens),
        "n_predict": n_predict,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": False,
        "cache_prompt": False,
    }
    cmd = [
        "docker",
        "exec",
        container,
        "curl",
        "-fsS",
        "--max-time",
        str(BENCHMARK_TIMEOUT_SEC),
        "-X",
        "POST",
        "http://127.0.0.1:8080/completion",
        "-H",
        "Content-Type: application/json",
        "-d",
        json.dumps(payload),
    ]
    code, output = run_command(cmd)
    if code != 0:
        return None, "Benchmark request failed"

    try:
        body = json.loads(output)
    except Exception:
        return None, "Benchmark returned invalid JSON"

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
        specs = [
            {"name": "prefill_heavy", "prompt_tokens": 2048, "n_predict": 64},
            {"name": "mixed", "prompt_tokens": 1024, "n_predict": 256},
            {"name": "gen_heavy", "prompt_tokens": 256, "n_predict": 512},
        ]
        runs = []
        for idx, spec in enumerate(specs, start=1):
            if callable(progress_cb):
                progress_cb(f"full benchmark: pass {idx}/{len(specs)} ({spec['name']})...")
            run, error = run_single_benchmark(
                active_key,
                containers,
                prompt_tokens=spec["prompt_tokens"],
                n_predict=spec["n_predict"],
            )
            if error:
                return None, f"Full benchmark failed on {spec['name']}: {error}"
            run["name"] = spec["name"]
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

    def _worker() -> None:
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

    threading.Thread(target=_worker, daemon=True).start()
    return True, f"Switch request accepted: {model['label']}"


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
            "family": m["family"],
            "ctx_size": m["ctx_size"],
            "quant": m["quant"],
            "params": m["params"],
        }
    throughput = build_throughput_status(active_key, containers)
    live_throughput = build_live_throughput_status(active_key, containers)
    model_stats = build_model_stats(active_key, live_throughput, throughput)

    context_info = {"n_ctx": None, "n_past": None}
    if live_throughput.get("n_ctx") is not None:
        context_info["n_ctx"] = live_throughput["n_ctx"]
    if live_throughput.get("n_past") is not None:
        context_info["n_past"] = live_throughput["n_past"]
    if context_info["n_ctx"] is None and active_key and active_key in models:
        ctx_str = models[active_key].get("ctx_size", "")
        try:
            context_info["n_ctx"] = int(ctx_str) if ctx_str else None
        except ValueError:
            pass

    # Safety valve: clear stale benchmark runs that never flipped state (e.g. worker crash).
    with BENCHMARK_LOCK:
        if BENCHMARK_STATE["in_progress"]:
            started_dt = parse_iso_utc(BENCHMARK_STATE.get("started_at"))
            if started_dt is not None:
                age = (datetime.now(timezone.utc) - started_dt).total_seconds()
                if age > BENCHMARK_STALE_SEC:
                    BENCHMARK_STATE["in_progress"] = False
                    BENCHMARK_STATE["completed_at"] = now_iso()
                    BENCHMARK_STATE["last_error"] = (
                        f"Benchmark marked stale after {int(age)}s and auto-reset"
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
                "family": m["family"],
                "ctx_size": m["ctx_size"],
                "quant": m["quant"],
                "params": m["params"],
            }
            for key, m in models.items()
        ],
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
    @media (max-width: 600px) {
      .stats-row { grid-template-columns: repeat(2, 1fr); }
      .sidebar { display: none; }
    }

    /* ── Events & Logs panel ── */
    .events-panel { background: var(--card); border: 1px solid var(--border-strong); border-radius: 12px; overflow: hidden; margin-top: 16px; }
    .events-panel-header { display:flex; align-items:center; justify-content:space-between; padding:12px 18px; border-bottom:1px solid var(--border); cursor:pointer; user-select:none; }
    .events-panel-header:hover { background:rgba(255,255,255,0.02); }
    .events-title { font-size:11px; font-weight:700; color:var(--text-muted); text-transform:uppercase; letter-spacing:1px; }
    .events-err-badge { font-size:10px; font-weight:800; padding:2px 7px; border-radius:999px; background:rgba(239,68,68,0.15); color:var(--danger); border:1px solid rgba(239,68,68,0.3); display:none; }
    .events-body { max-height:220px; overflow-y:auto; padding:6px 0; font-family:var(--font-mono); font-size:12px; }
    .events-body::-webkit-scrollbar { width:4px; }
    .events-body::-webkit-scrollbar-thumb { background:var(--border-strong); border-radius:2px; }
    .events-collapsed .events-body { display:none; }
    .event-row { display:flex; gap:10px; align-items:baseline; padding:3px 18px; border-bottom:1px solid rgba(255,255,255,0.03); }
    .event-row:last-child { border-bottom:none; }
    .event-ts { color:var(--text-muted); font-size:10px; white-space:nowrap; flex-shrink:0; }
    .event-sev { font-size:10px; font-weight:800; width:52px; flex-shrink:0; }
    .ev-error { color:var(--danger); } .ev-warning { color:var(--warning); } .ev-info { color:var(--success); }
    .event-msg { flex:1; color:var(--text-dim); word-break:break-word; }
    .events-empty-msg { text-align:center; color:var(--text-muted); padding:16px; font-size:12px; }
    .hdr-err-badge { font-size:10px; font-weight:800; padding:2px 8px; border-radius:999px; background:rgba(239,68,68,0.2); color:var(--danger); border:1px solid rgba(239,68,68,0.4); display:none; margin-right:8px; }
  </style>
</head>
<body>

  <!-- Header -->
  <div class="app-header">
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

    <!-- Sidebar -->
    <div class="sidebar">
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
        <div class="stat-card">
          <div class="stat-label">Context</div>
          <div class="stat-value" id="val-ctx">-- / --</div>
          <div class="stat-bar-bg"><div id="bar-ctx" class="stat-bar-fill" style="width:0%"></div></div>
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

      <!-- Events & Logs -->
      <div class="events-panel" id="eventsPanel">
        <div class="events-panel-header" onclick="toggleEventsPanel()">
          <span class="events-title">Events &amp; Logs</span>
          <div style="display:flex;align-items:center;gap:8px">
            <span id="eventsBadge" class="events-err-badge"></span>
            <span id="evToggleIcon" style="color:var(--text-muted);font-size:12px">&#9660;</span>
          </div>
        </div>
        <div class="events-body" id="eventsBody">
          <div class="events-empty-msg">No events yet.</div>
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
    const FAMILY_ORDER = ['gemma','gptoss','glm','mistral','nemotron','qwen','other'];
    const FAMILY_LABELS = {
      qwen: 'QWEN', nemotron: 'NEMOTRON', gptoss: 'GPT-OSS',
      glm: 'GLM', mistral: 'MISTRAL', gemma: 'GEMMA', other: 'OTHER'
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
      for (const fam of FAMILY_ORDER) {
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
      const stats = data.model_stats || {};
      const bench = data.benchmark || {};

      // Quick stats - show live TPS (0 when idle)
      const liveTps = stats.live_tps || 0;
      document.getElementById('val-tps').textContent = liveTps > 0 ? liveTps.toFixed(1) : '0.0';

      // Ingest Speed: prefer live ingest from /slots, fall back to best known
      let ingestVal = '0.0';
      const liveIngest = stats.last_live_ingest_tps || data.live_throughput?.live_ingest_tps;
      const bestIngest = stats.best_ingest_tps;
      if (liveIngest != null && liveIngest > 0) {
        ingestVal = liveIngest.toFixed(1);
      } else if (bestIngest != null && bestIngest > 0) {
        ingestVal = bestIngest.toFixed(1);
      }
      document.getElementById('val-ingest').textContent = ingestVal;
      document.getElementById('val-util').textContent = gpu.util != null ? `${gpu.util}%` : '--%';
      document.getElementById('bar-util').style.width = `${gpu.util || 0}%`;
      const vramGB = ((gpu.mem_used || 0) / 1024).toFixed(1);
      document.getElementById('val-vram').textContent = `${vramGB} GB`;
      const vramPct = gpu.mem_total > 0 ? (gpu.mem_used / gpu.mem_total) * 100 : 0;
      document.getElementById('bar-vram').style.width = `${vramPct.toFixed(1)}%`;
      document.getElementById('val-temp').textContent = gpu.temp != null ? `${gpu.temp}°C` : '--°C';
      document.getElementById('val-fan').textContent = `Fan: ${gpu.fan || 0}%`;

      // Context usage
      const ctxInfo = data.context_info || {};
      const nCtx = ctxInfo.n_ctx;
      const nPast = ctxInfo.n_past;
      const ctxEl = document.getElementById('val-ctx');
      const ctxBar = document.getElementById('bar-ctx');
      if (nCtx != null) {
        const ctxMax = nCtx >= 1024 ? `${Math.round(nCtx/1024)}k` : `${nCtx}`;
        if (nPast != null && nPast > 0) {
          const ctxUsed = nPast >= 1024 ? `${(nPast/1024).toFixed(1)}k` : `${nPast}`;
          ctxEl.textContent = `${ctxUsed} / ${ctxMax}`;
          const pct = (nPast / nCtx) * 100;
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

      // History chart - show live TPS (0 when idle) and last ingest
      const tpsVal = (stats && stats.live_tps) || 0;
      history.util.push(gpu.util || 0); history.util.shift();
      history.vram.push(vramPct); history.vram.shift();
      history.tps.push(tpsVal); history.tps.shift();
      chart.update();

      // Service details
      const infoModel = document.getElementById('info-model');
      const infoBadges = document.getElementById('info-badges');
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
        infoBadges.innerHTML = badges;
      } else {
        infoModel.textContent = 'NONE';
        infoBadges.innerHTML = '';
      }

      document.getElementById('info-avg-rate').textContent =
        stats.average_rate_tps ? `${stats.average_rate_tps.toFixed(2)} T/S` : '-- T/S';
      document.getElementById('info-runs').textContent = stats.completed_count || 0;
      document.getElementById('info-clocks').textContent =
        `${gpu.clock_gfx || '--'} / ${gpu.clock_mem || '--'} MHz`;
      document.getElementById('info-power').textContent =
        `${gpu.power ? gpu.power.toFixed(0) : '--'}W / ${gpu.power_limit ? gpu.power_limit.toFixed(0) : '--'}W`;
      const btnBenchmark = document.getElementById('btnBenchmark');
      const btnBenchmarkFull = document.getElementById('btnBenchmarkFull');
      if (btnBenchmark) {
        const canRun = !!(data.active?.healthy) && !data.switch_in_progress && !bench.in_progress;
        btnBenchmark.disabled = !canRun;
        btnBenchmark.textContent = (bench.in_progress && bench.profile === 'balanced') ? 'BENCHMARKING...' : 'RUN BENCHMARK';
      }
      if (btnBenchmarkFull) {
        const canRun = !!(data.active?.healthy) && !data.switch_in_progress && !bench.in_progress;
        btnBenchmarkFull.disabled = !canRun;
        btnBenchmarkFull.textContent = (bench.in_progress && bench.profile === 'full') ? 'BENCHMARKING...' : 'RUN FULL BENCHMARK';
      }

      const benchRes = bench.last_result || {};
      document.getElementById('info-bench-prefill').textContent =
        benchRes.prefill_tps ? `${benchRes.prefill_tps.toFixed(2)} T/S` : '-- T/S';
      document.getElementById('info-bench-gen').textContent =
        benchRes.gen_tps ? `${benchRes.gen_tps.toFixed(2)} T/S` : '-- T/S';
      if (bench.in_progress && bench.started_at) {
        document.getElementById('info-bench-last').textContent = `${(bench.profile || 'balanced').toUpperCase()} running since ${fmtLocalTs(bench.started_at)}`;
      } else if (bench.last_error) {
        document.getElementById('info-bench-last').textContent = `Failed: ${bench.last_error}`;
      } else if (benchRes.completed_at) {
        document.getElementById('info-bench-last').textContent =
          `${(benchRes.profile || 'balanced').toUpperCase()} at ${fmtLocalTs(benchRes.completed_at)}`;
      } else {
        document.getElementById('info-bench-last').textContent = '--';
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
      if (lastData?.active?.key === key && lastData?.active?.healthy) {
        const el = document.getElementById('switchStatus');
        el.textContent = `${label} is already active and healthy.`;
        setTimeout(() => { el.textContent = lastData?.last_message || ''; }, 3000);
        return;
      }
      if (lastData?.switch_in_progress) return;
      pendingAction = { type: 'switch', key };
      document.getElementById('modalTitle').textContent = 'Switch Model?';
      document.getElementById('modalText').textContent =
        `Switch to ${label}? The current LLM will be stopped and the new model loaded into VRAM.`;
      document.getElementById('btnConfirmSwitch').textContent = 'PROCEED';
      document.getElementById('btnConfirmSwitch').style.background = 'var(--accent)';
      document.getElementById('confirmModal').classList.add('active');
    }

    function confirmStopAll() {
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

    function toggleEventsPanel() {
      eventsPanelOpen = !eventsPanelOpen;
      document.getElementById('eventsPanel').classList.toggle('events-collapsed', !eventsPanelOpen);
      document.getElementById('evToggleIcon').textContent = eventsPanelOpen ? '▼' : '▶';
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

    initChart();
    refresh();
    setInterval(refresh, 1000);
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

    def _send_html(self, html: str, status: int = 200) -> None:
        payload = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send_html(INDEX_HTML)
            return

        if self.path == "/api/status":
            status = build_status(self)
            self._send_json(status)
            return

        self._send_json({"error": "Not found"}, 404)

    def do_POST(self) -> None:
        if self.path == "/api/switch":
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
                body = json.loads(raw.decode("utf-8"))
            except Exception:
                self._send_json({"error": "Invalid JSON body"}, 400)
                return

            model_key = body.get("model", "")
            accepted, msg = start_switch(model_key)
            if not accepted:
                self._send_json({"error": msg}, 409)
                return
            self._send_json({"message": msg, "status": build_status(self)})
            return

        if self.path == "/api/benchmark":
            profile = "balanced"
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
                body = json.loads(raw.decode("utf-8"))
                if isinstance(body, dict):
                    requested = body.get("profile")
                    if isinstance(requested, str) and requested.strip():
                        profile = requested.strip().lower()
            except Exception:
                self._send_json({"error": "Invalid JSON body"}, 400)
                return

            if profile not in {"balanced", "full"}:
                self._send_json({"error": "Unsupported benchmark profile"}, 400)
                return

            models, _ = get_models()
            containers = list_llama_compose_containers()
            active_key = detect_active_model_key(containers)
            if not active_key or active_key not in models:
                self._send_json({"error": "No active model detected"}, 409)
                return
            if not is_model_server_healthy(active_key, containers):
                self._send_json({"error": "Active model is not healthy yet"}, 409)
                return

            with STATE_LOCK:
                if STATE["switch_in_progress"]:
                    self._send_json({"error": "Cannot benchmark while switch is in progress"}, 409)
                    return

            with BENCHMARK_LOCK:
                if BENCHMARK_STATE["in_progress"]:
                    self._send_json({"error": "A benchmark is already in progress"}, 409)
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
                        BENCHMARK_STATE["history"].append(
                            {
                                "profile": benchmark_profile,
                                "model_key": model_key,
                                "started_at": started_iso,
                                "completed_at": completed_iso,
                                "duration_sec": elapsed,
                                "success": False,
                                "error": error,
                            }
                        )
                    elif result is not None:
                        result["started_at"] = started_iso
                        result["completed_at"] = completed_iso
                        result["duration_sec"] = elapsed
                        BENCHMARK_STATE["last_result"] = result
                        BENCHMARK_STATE["last_error"] = None
                        BENCHMARK_STATE["history"].append(
                            {
                                "profile": benchmark_profile,
                                "model_key": model_key,
                                "started_at": started_iso,
                                "completed_at": completed_iso,
                                "duration_sec": elapsed,
                                "success": True,
                                "prefill_tps": result["prefill_tps"],
                                "gen_tps": result["gen_tps"],
                            }
                        )
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
            return

        if self.path == "/api/stop":
            with STATE_LOCK:
                if STATE["switch_in_progress"]:
                    self._send_json({"error": "A switch is already in progress"}, 409)
                    return
                STATE["switch_in_progress"] = True
                STATE["last_message"] = "Stopping all models..."

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

            threading.Thread(target=_stop_worker, daemon=True).start()
            self._send_json({"message": "Stop-all initiated"})
            return

        if self.path == "/api/restart":
            models, _ = get_models()
            containers = list_llama_compose_containers()
            active_key = detect_active_model_key(containers)
            if not active_key:
                self._send_json({"error": "No active model to restart"}, 409)
                return
            accepted, msg = start_switch(active_key)
            if not accepted:
                self._send_json({"error": msg}, 409)
                return
            self._send_json({"message": msg, "status": build_status(self)})
            return

        # Emergency reset — clears stuck switch_in_progress without touching containers
        if self.path == "/api/reset":
            with STATE_LOCK:
                STATE["switch_in_progress"] = False
                STATE["last_message"] = "State reset by user."
            self._send_json({"message": "State reset"})
            return

        self._send_json({"error": "Not found"}, 404)

    def log_message(self, fmt: str, *args) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] {self.address_string()} {fmt % args}")


def main() -> None:
    if not SWITCH_SCRIPT.exists():
        raise SystemExit(f"Missing switch script: {SWITCH_SCRIPT}")

    threading.Thread(target=_run_log_watcher, daemon=True, name="log-watcher").start()

    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"dashboard server listening on {HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
