#!/usr/bin/env python3
"""LLM host agent.

Runs next to the GPU on every LLM host. It owns everything that needs host access
(docker socket, container logs, nvidia-smi, stack folders, the request-log DB) and
exposes it over a small token-protected JSON API that the hub polls.

All collection and control logic lives in core.py, moved unchanged from the old
single-host dashboard-server.py. This file adds:
  * a background collector that builds the status snapshot once a second, so the
    API answers from memory and history is sampled even with no browser open;
  * the /v1 HTTP API with bearer-token auth.
"""
import hmac
import json
import os
import shlex
import socket
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import core

AGENT_VERSION = "2.0.0"
AGENT_HOST = os.environ.get("AGENT_HOST", "0.0.0.0")
AGENT_PORT = int(os.environ.get("AGENT_PORT", "9360"))
AGENT_TOKEN = os.environ.get("AGENT_TOKEN", "").strip()
AGENT_TOKEN_FILE = os.environ.get("AGENT_TOKEN_FILE", "").strip()
AGENT_NAME = os.environ.get("AGENT_NAME", "").strip() or socket.gethostname()
COLLECT_INTERVAL_SEC = float(os.environ.get("AGENT_COLLECT_INTERVAL_SEC", "1.0"))

if not AGENT_TOKEN and AGENT_TOKEN_FILE:
    try:
        with open(AGENT_TOKEN_FILE, encoding="utf-8") as fh:
            AGENT_TOKEN = fh.read().strip()
    except OSError as exc:
        raise SystemExit(f"Cannot read AGENT_TOKEN_FILE {AGENT_TOKEN_FILE}: {exc}")

STARTED_AT = time.time()

# ---------------------------------------------------------------------------
# Snapshot collector
# ---------------------------------------------------------------------------
SNAPSHOT_LOCK = threading.Lock()
SNAPSHOT: dict = {"ready": False}
COLLECT_NOW = threading.Event()


def _collect_once() -> None:
    started = time.time()
    status = core.build_status(None)
    status["host"] = AGENT_NAME
    status["agent_version"] = AGENT_VERSION
    status["collected_at"] = time.time()
    status["collect_ms"] = round((time.time() - started) * 1000, 1)
    status["ready"] = True
    with SNAPSHOT_LOCK:
        SNAPSHOT.clear()
        SNAPSHOT.update(status)


def _run_collector() -> None:
    while True:
        try:
            _collect_once()
        except Exception as exc:  # noqa: BLE001
            core._append_log_event("warning", "agent", f"Snapshot collection failed: {exc}")
        COLLECT_NOW.wait(COLLECT_INTERVAL_SEC)
        COLLECT_NOW.clear()


def get_snapshot() -> dict:
    with SNAPSHOT_LOCK:
        if not SNAPSHOT.get("ready"):
            pass
        else:
            return dict(SNAPSHOT)
    # First request before the collector finished: build one synchronously.
    _collect_once()
    with SNAPSHOT_LOCK:
        return dict(SNAPSHOT)


def agent_info() -> dict:
    models, _ = core.get_models()
    backends = sorted({(m.get("backend") or ("vllm" if m.get("vllm") else "llamacpp"))
                       for m in models.values()})
    return {
        "name": AGENT_NAME,
        "version": AGENT_VERSION,
        "uptime_sec": round(time.time() - STARTED_AT),
        "gpus": core.get_gpu_stats(),
        "backends": backends,
        "model_count": len(models),
        "capabilities": ["snapshot", "history", "requests", "switch", "benchmark", "settings"],
        "request_log": os.path.exists(core.REQUEST_LOG_DB),
        "metrics_history": bool(core.METRICS_DB_READY),
    }


# ---------------------------------------------------------------------------
# Actions (moved from the old Handler.do_POST)
# ---------------------------------------------------------------------------
def action_switch(body: dict) -> tuple[int, dict]:
    accepted, msg = core.start_switch(str(body.get("model", "")))
    return (200, {"message": msg}) if accepted else (409, {"error": msg})


def action_restart(_body: dict) -> tuple[int, dict]:
    containers = core.list_llama_compose_containers()
    active_key = core.detect_active_model_key(containers)
    if not active_key:
        return 409, {"error": "No active model to restart"}
    accepted, msg = core.start_switch(active_key, force=True)
    return (200, {"message": msg}) if accepted else (409, {"error": msg})


def action_clear(_body: dict) -> tuple[int, dict]:
    was_active = core._clear_switch_state(
        "Switch state cleared manually",
        log="Switch state cleared manually from the dashboard",
    )
    return 200, {
        "message": "Switch state cleared" if was_active else "No switch was in progress",
        "cleared": was_active,
    }


def action_stop(_body: dict) -> tuple[int, dict]:
    with core.STATE_LOCK:
        if core.STATE["switch_in_progress"]:
            return 409, {"error": "A switch is already in progress"}
        core.STATE["switch_in_progress"] = True
        core.STATE["last_message"] = "Stopping all models..."
    core._append_log_event("info", "stop", "Stop-all initiated")

    def _stop_worker() -> None:
        cmd = [
            "bash", "-lc",
            f"cd {shlex.quote(str(core.LLAMA_DIR))} && {shlex.quote(str(core.SWITCH_SCRIPT))} --stop-all",
        ]
        exit_code, output = core.run_command(cmd)
        with core.STATE_LOCK:
            core.STATE["switch_in_progress"] = False
            core.STATE["last_exit_code"] = exit_code
            core.STATE["last_output"] = output[-2000:]
            core.STATE["last_message"] = "All models stopped." if exit_code == 0 else "Stop-all failed."
        sev = "info" if exit_code == 0 else "error"
        msg = "All models stopped" if exit_code == 0 else f"Stop-all failed (exit {exit_code})"
        core._append_log_event(sev, "stop", msg)
        COLLECT_NOW.set()

    threading.Thread(target=_stop_worker, daemon=True).start()
    return 200, {"message": "Stop-all initiated"}


def action_benchmark(body: dict) -> tuple[int, dict]:
    profile = body.get("profile", "balanced")
    profile = profile.strip().lower() if isinstance(profile, str) and profile.strip() else "balanced"
    if profile not in {"balanced", "full"}:
        return 400, {"error": "Unsupported benchmark profile"}

    models, _ = core.get_models()
    containers = core.list_llama_compose_containers()
    active_key = core.detect_active_model_key(containers)
    if not active_key or active_key not in models:
        return 409, {"error": "No active model detected"}
    if not core.is_model_server_healthy(active_key, containers):
        return 409, {"error": "Active model is not healthy yet"}
    with core.STATE_LOCK:
        if core.STATE["switch_in_progress"]:
            return 409, {"error": "Cannot benchmark while switch is in progress"}
    with core.BENCHMARK_LOCK:
        if core.BENCHMARK_STATE["in_progress"]:
            return 409, {"error": "A benchmark is already in progress"}
        core.BENCHMARK_STATE["in_progress"] = True
        core.BENCHMARK_STATE["profile"] = profile
        core.BENCHMARK_STATE["started_at"] = core.now_iso()
        core.BENCHMARK_STATE["completed_at"] = None
        core.BENCHMARK_STATE["last_error"] = None
    with core.STATE_LOCK:
        core.STATE["last_message"] = f"Running {profile} benchmark for {models[active_key]['label']}..."

    def _worker(model_key: str, benchmark_profile: str) -> None:
        started = time.time()
        started_iso = core.now_iso()
        result = None
        error = None
        model_label = models[model_key]["label"]
        core._append_log_event("info", "benchmark", f"Benchmark ({benchmark_profile}) started for {model_label}")
        try:
            result, error = core.run_benchmark_profile(
                model_key,
                core.list_llama_compose_containers(),
                benchmark_profile,
                progress_cb=lambda msg: core._set_last_message(msg),
            )
        except Exception as exc:  # noqa: BLE001
            error = f"Unhandled benchmark error: {exc}"

        completed_iso = core.now_iso()
        elapsed = max(0.0, time.time() - started)
        entry = {
            "profile": benchmark_profile,
            "model_key": model_key,
            "model_label": model_label,
            "started_at": started_iso,
            "completed_at": completed_iso,
            "duration_sec": elapsed,
        }
        with core.BENCHMARK_LOCK:
            core.BENCHMARK_STATE["in_progress"] = False
            core.BENCHMARK_STATE["completed_at"] = completed_iso
            if error:
                core.BENCHMARK_STATE["last_error"] = error
                core._append_log_event("error", "benchmark", f"Benchmark failed for {model_label}: {error}")
                entry.update(success=False, error=error)
                core.BENCHMARK_STATE["history"].append(entry)
                core.save_benchmark_run(entry)
            elif result is not None:
                result.update(started_at=started_iso, completed_at=completed_iso, duration_sec=elapsed,
                              model_key=model_key, model_label=model_label)
                core.BENCHMARK_STATE["last_result"] = result
                core.BENCHMARK_STATE["last_error"] = None
                accept = result.get("draft_acceptance")
                accept_txt = f", draft accept {accept * 100:.0f}%" if isinstance(accept, float) else ""
                core._append_log_event(
                    "info", "benchmark",
                    f"Benchmark ({benchmark_profile}) for {model_label}: prefill {result['prefill_tps']:.1f} T/S, "
                    f"gen {result['gen_tps']:.1f} T/S{accept_txt}",
                )
                entry.update(success=True, prefill_tps=result["prefill_tps"], gen_tps=result["gen_tps"],
                             draft_acceptance=result.get("draft_acceptance"))
                core.BENCHMARK_STATE["history"].append(entry)
                core.save_benchmark_run(entry)
            else:
                core.BENCHMARK_STATE["last_error"] = "Benchmark failed without result"
            core.BENCHMARK_STATE["history"] = core.BENCHMARK_STATE["history"][-core.FULL_BENCHMARK_MAX_HISTORY:]

        with core.STATE_LOCK:
            if error:
                core.STATE["last_message"] = f"Benchmark failed: {error}"
            elif result is not None:
                core.STATE["last_message"] = (
                    f"{benchmark_profile} benchmark complete: prefill {result['prefill_tps']:.2f} T/S, "
                    f"gen {result['gen_tps']:.2f} T/S"
                )
            else:
                core.STATE["last_message"] = "Benchmark failed without result"
        COLLECT_NOW.set()

    threading.Thread(target=_worker, args=(active_key, profile), daemon=True).start()
    return 200, {"message": f"{profile.capitalize()} benchmark started"}


def action_settings(body: dict) -> tuple[int, dict]:
    values = body.get("settings")
    if not isinstance(values, dict):
        return 400, {"error": "Expected a 'settings' object"}
    ok, msg = core.apply_settings(values)
    if not ok:
        return 400, {"error": msg}
    return 200, {"message": msg, **core.describe_settings()}


POST_ROUTES = {
    "/v1/switch": action_switch,
    "/v1/restart": action_restart,
    "/v1/switch/clear": action_clear,
    "/v1/stop": action_stop,
    "/v1/benchmark": action_benchmark,
    "/v1/settings": action_settings,
}
LOGGED_ACTIONS = {"/v1/switch", "/v1/restart", "/v1/stop", "/v1/benchmark"}


def request_log_endpoints() -> dict:
    """Distinct endpoint labels in the request log, so the UI never hard-codes them."""
    if not os.path.exists(core.REQUEST_LOG_DB):
        return {"endpoints": [], "available": False}
    try:
        conn = sqlite3.connect(f"file:{core.REQUEST_LOG_DB}?mode=ro", uri=True, timeout=5)
        try:
            rows = conn.execute(
                "SELECT endpoint, COUNT(*) FROM requests GROUP BY endpoint ORDER BY endpoint"
            ).fetchall()
        finally:
            conn.close()
        return {"available": True, "endpoints": [{"endpoint": r[0], "count": r[1]} for r in rows if r[0]]}
    except sqlite3.Error as exc:
        return {"available": False, "endpoints": [], "error": str(exc)}


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = f"llm-agent/{AGENT_VERSION}"

    def _send_json(self, data, status: int = 200) -> None:
        payload = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _authorized(self) -> bool:
        if not AGENT_TOKEN:
            return True
        header = self.headers.get("Authorization", "")
        token = header[7:].strip() if header.lower().startswith("bearer ") else ""
        return bool(token) and hmac.compare_digest(token.encode(), AGENT_TOKEN.encode())

    def _caller(self) -> str:
        via = self.headers.get("X-Hub-User") or ""
        ip = self.headers.get("X-Forwarded-For", "").split(",")[0].strip() or self.client_address[0]
        return f"{ip}{' via ' + via if via else ''}"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path.rstrip("/") or "/"
        q = parse_qs(parsed.query)
        if route == "/healthz":
            self._send_json({"ok": True})
            return
        if not self._authorized():
            self._send_json({"error": "Unauthorized"}, 401)
            return
        try:
            if route == "/v1/info":
                self._send_json(agent_info())
            elif route == "/v1/snapshot":
                self._send_json(get_snapshot())
            elif route == "/v1/history":
                self._send_json(core.read_metrics_history((q.get("range") or ["1h"])[0]))
            elif route == "/v1/requests":
                self._send_json(core.read_request_log(q))
            elif route == "/v1/requests/detail":
                self._send_json(core.read_request_detail(q))
            elif route == "/v1/requests/endpoints":
                self._send_json(request_log_endpoints())
            elif route == "/v1/settings":
                self._send_json(core.describe_settings())
            elif route == "/v1/settings/inspect":
                raw = (q.get("path") or [""])[0].strip()
                if not raw.startswith("/"):
                    self._send_json({"error": "Path must be absolute", "path": raw}, 400)
                else:
                    self._send_json({"path": raw, **core.inspect_stack_dir(core.Path(raw))})
            else:
                self._send_json({"error": "Not found"}, 404)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": f"Agent error: {exc}"}, 500)

    def do_POST(self) -> None:
        route = urlparse(self.path).path.rstrip("/")
        if not self._authorized():
            self._send_json({"error": "Unauthorized"}, 401)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            body = json.loads(raw.decode("utf-8") or "{}")
            if not isinstance(body, dict):
                raise ValueError("body must be an object")
        except Exception:  # noqa: BLE001
            self._send_json({"error": "Invalid JSON body"}, 400)
            return
        fn = POST_ROUTES.get(route)
        if fn is None:
            self._send_json({"error": "Not found"}, 404)
            return
        try:
            status, data = fn(body)
        except Exception as exc:  # noqa: BLE001
            status, data = 500, {"error": f"Agent error: {exc}"}
        if route in LOGGED_ACTIONS:
            action = core._human_readable_path(route.replace("/v1/", "/api/"), body)
            core._append_log_event("error" if status >= 400 else "info", "request",
                                   f"[{self._caller()}] {action}")
        COLLECT_NOW.set()
        self._send_json(data, status)

    def log_message(self, fmt: str, *args) -> None:
        pass


def main() -> None:
    if not AGENT_TOKEN and os.environ.get("AGENT_ALLOW_NO_TOKEN") != "1":
        raise SystemExit(
            "AGENT_TOKEN is not set. The agent can start and stop containers, so it refuses to "
            "run unauthenticated. Set AGENT_TOKEN (or AGENT_TOKEN_FILE), or AGENT_ALLOW_NO_TOKEN=1."
        )
    if not core.SWITCH_SCRIPT.exists():
        raise SystemExit(f"Missing switch script: {core.SWITCH_SCRIPT}")

    threading.Thread(target=core._run_shm_sweeper, daemon=True, name="shm-sweeper").start()
    if core.init_metrics_db():
        core.restore_benchmark_history()
        threading.Thread(target=core._run_metrics_sampler, daemon=True, name="metrics-sampler").start()
    threading.Thread(target=core._run_log_watcher, daemon=True, name="log-watcher").start()
    threading.Thread(target=_run_collector, daemon=True, name="collector").start()

    server = ThreadingHTTPServer((AGENT_HOST, AGENT_PORT), Handler)
    print(f"llm-agent {AGENT_VERSION} '{AGENT_NAME}' listening on {AGENT_HOST}:{AGENT_PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
