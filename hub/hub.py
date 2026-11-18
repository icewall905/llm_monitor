#!/usr/bin/env python3
"""LLM fleet hub.

Serves the web dashboard and talks to one llm-agent per GPU host. Each agent is
registered with a URL and a bearer token (Settings page or nodes.json). The hub
polls every agent's snapshot once a second, streams the fleet to browsers over
SSE, and proxies history, request-log reads and control actions. Agent tokens
never leave the hub.
"""
import base64
import hashlib
import hmac
import json
import mimetypes
import os
import re
import secrets
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlencode, urlparse

HUB_VERSION = "2.0.0"
HOST = os.environ.get("HUB_HOST", "0.0.0.0")
PORT = int(os.environ.get("HUB_PORT", "80"))
DATA_DIR = Path(os.environ.get("HUB_DATA_DIR", "/data"))
STATIC_DIR = Path(os.environ.get("HUB_STATIC_DIR", Path(__file__).parent / "static"))
NODES_FILE = DATA_DIR / "nodes.json"
POLL_INTERVAL_SEC = float(os.environ.get("HUB_POLL_INTERVAL_SEC", "1.0"))
POLL_TIMEOUT_SEC = float(os.environ.get("HUB_POLL_TIMEOUT_SEC", "5"))
STREAM_INTERVAL_SEC = float(os.environ.get("HUB_STREAM_INTERVAL_SEC", "1.0"))
HUB_PASSWORD = os.environ.get("HUB_PASSWORD", "")
HUB_USER = os.environ.get("HUB_USER", "admin")

NODE_COLORS = ["#5b8def", "#22c55e", "#a78bfa", "#f59e0b", "#14b8a6", "#ef4444", "#ec4899", "#06b6d4"]

# Snapshot keys the browser never needs on the 1 s stream (models go out only on change).
STREAM_DROP_KEYS = {"models", "throughput"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Agent client
# ---------------------------------------------------------------------------
class AgentError(Exception):
    def __init__(self, message: str, status: int = 502, body=None):
        super().__init__(message)
        self.status = status
        self.body = body


def agent_call(url: str, token: str, path: str, method: str = "GET", body=None,
               timeout: float = POLL_TIMEOUT_SEC, query: dict | None = None):
    full = url.rstrip("/") + path
    if query:
        full += "?" + urlencode(query, doseq=True)
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(full, data=data, method=method)
    req.add_header("Accept", "application/json")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read().decode("utf-8") or "{}")
        except Exception:  # noqa: BLE001
            payload = {"error": f"HTTP {exc.code}"}
        if exc.code == 401:
            raise AgentError("Agent rejected the token (401)", 401, payload) from None
        return exc.code, payload
    except urllib.error.URLError as exc:
        raise AgentError(f"Unreachable: {exc.reason}") from None
    except TimeoutError:
        raise AgentError("Timed out") from None
    except (ValueError, OSError) as exc:
        raise AgentError(f"Bad response: {exc}") from None


# ---------------------------------------------------------------------------
# Node registry + pollers
# ---------------------------------------------------------------------------
class Node:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.snapshot: dict | None = None
        self.models_hash = ""
        self.online = False
        self.error: str | None = None
        self.last_seen: float | None = None
        self.latency_ms: float | None = None
        self.stop = threading.Event()
        self.kick = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name=f"poll-{cfg['id']}")

    @property
    def id(self) -> str:
        return self.cfg["id"]

    def _run(self) -> None:
        while not self.stop.is_set():
            if self.cfg.get("enabled", True):
                self.poll()
            self.kick.wait(POLL_INTERVAL_SEC)
            self.kick.clear()

    def poll(self) -> None:
        started = time.time()
        try:
            status, snap = agent_call(self.cfg["url"], self.cfg.get("token", ""), "/v1/snapshot")
            if status != 200:
                raise AgentError(snap.get("error") or f"HTTP {status}")
            models = snap.get("models") or []
            mh = hashlib.sha1(json.dumps(models, sort_keys=True, default=str).encode()).hexdigest()[:12]
            with self.lock:
                self.snapshot = snap
                self.models_hash = mh
                if not self.online:
                    log(f"node {self.cfg['name']} online")
                self.online = True
                self.error = None
                self.last_seen = time.time()
                self.latency_ms = round((time.time() - started) * 1000, 1)
        except AgentError as exc:
            with self.lock:
                if self.online or self.error != str(exc):
                    log(f"node {self.cfg['name']} offline: {exc}")
                self.online = False
                self.error = str(exc)

    def public(self, include_models: bool = True, stream: bool = False) -> dict:
        with self.lock:
            snap = self.snapshot
            out = {
                "id": self.id,
                "name": self.cfg.get("name") or self.id,
                "url": self.cfg.get("url"),
                "color": self.cfg.get("color"),
                "enabled": self.cfg.get("enabled", True),
                "has_token": bool(self.cfg.get("token")),
                "online": self.online,
                "error": self.error,
                "last_seen": self.last_seen,
                "latency_ms": self.latency_ms,
                "models_hash": self.models_hash,
            }
        if snap is not None:
            if stream:
                s = {k: v for k, v in snap.items() if k not in STREAM_DROP_KEYS}
                if include_models:
                    s["models"] = snap.get("models") or []
                out["snapshot"] = s
            else:
                out["snapshot"] = snap
        return out


class Registry:
    def __init__(self):
        self.lock = threading.Lock()
        self.nodes: dict[str, Node] = {}

    def load(self) -> None:
        cfgs = []
        if NODES_FILE.exists():
            try:
                cfgs = json.loads(NODES_FILE.read_text(encoding="utf-8")).get("nodes", [])
            except (OSError, ValueError) as exc:
                log(f"cannot read {NODES_FILE}: {exc}")
        else:
            cfgs = bootstrap_nodes()
            if cfgs:
                self._write(cfgs)
        for cfg in cfgs:
            self._start(self._normalize(cfg))

    def _normalize(self, cfg: dict) -> dict:
        cfg = dict(cfg)
        cfg["id"] = cfg.get("id") or slugify(cfg.get("name") or "node")
        cfg["name"] = (cfg.get("name") or cfg["id"]).strip()
        cfg["url"] = (cfg.get("url") or "").strip().rstrip("/")
        cfg["token"] = cfg.get("token") or ""
        cfg["enabled"] = bool(cfg.get("enabled", True))
        if not cfg.get("color"):
            cfg["color"] = NODE_COLORS[len(self.nodes) % len(NODE_COLORS)]
        return cfg

    def _start(self, cfg: dict) -> Node:
        node = Node(cfg)
        with self.lock:
            self.nodes[node.id] = node
        node.thread.start()
        return node

    def _write(self, cfgs: list[dict] | None = None) -> None:
        if cfgs is None:
            with self.lock:
                cfgs = [n.cfg for n in self.nodes.values()]
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = NODES_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps({"nodes": cfgs}, indent=2), encoding="utf-8")
        os.chmod(tmp, 0o600)
        tmp.replace(NODES_FILE)

    def get(self, node_id: str) -> Node | None:
        with self.lock:
            return self.nodes.get(node_id)

    def all(self) -> list[Node]:
        with self.lock:
            return list(self.nodes.values())

    def add(self, cfg: dict) -> Node:
        cfg = self._normalize(cfg)
        base, n = cfg["id"], 2
        with self.lock:
            while cfg["id"] in self.nodes:
                cfg["id"] = f"{base}-{n}"
                n += 1
        node = self._start(cfg)
        self._write()
        return node

    def update(self, node_id: str, changes: dict) -> Node | None:
        node = self.get(node_id)
        if node is None:
            return None
        cfg = dict(node.cfg)
        for key in ("name", "url", "color", "enabled"):
            if key in changes:
                cfg[key] = changes[key]
        if changes.get("token"):  # empty = keep existing token
            cfg["token"] = changes["token"]
        cfg = self._normalize(cfg)
        node.cfg = cfg
        with node.lock:
            node.snapshot = None if changes.get("url") else node.snapshot
        node.kick.set()
        self._write()
        return node

    def remove(self, node_id: str) -> bool:
        with self.lock:
            node = self.nodes.pop(node_id, None)
        if node is None:
            return False
        node.stop.set()
        node.kick.set()
        self._write()
        return True


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:32] or "node"


def bootstrap_nodes() -> list[dict]:
    """First run: register the agent(s) named in the environment."""
    raw = os.environ.get("HUB_BOOTSTRAP_NODES", "").strip()
    if raw:
        try:
            nodes = json.loads(raw)
            if isinstance(nodes, list):
                return nodes
        except ValueError:
            log("HUB_BOOTSTRAP_NODES is not valid JSON; ignoring")
    url = os.environ.get("HUB_LOCAL_AGENT_URL", "").strip()
    if url:
        return [{
            "name": os.environ.get("HUB_LOCAL_AGENT_NAME", "local"),
            "url": url,
            "token": os.environ.get("HUB_LOCAL_AGENT_TOKEN", ""),
        }]
    return []


REGISTRY = Registry()

# ---------------------------------------------------------------------------
# Proxy routes
# ---------------------------------------------------------------------------
NODE_GET = {
    "info": "/v1/info",
    "snapshot": "/v1/snapshot",
    "history": "/v1/history",
    "requests": "/v1/requests",
    "requests/detail": "/v1/requests/detail",
    "requests/endpoints": "/v1/requests/endpoints",
    "settings": "/v1/settings",
    "settings/inspect": "/v1/settings/inspect",
}
NODE_POST = {
    "switch": "/v1/switch",
    "restart": "/v1/restart",
    "stop": "/v1/stop",
    "switch/clear": "/v1/switch/clear",
    "benchmark": "/v1/benchmark",
    "settings": "/v1/settings",
}
POOL = ThreadPoolExecutor(max_workers=16)


def fleet_requests(query: dict) -> dict:
    """Merge request-log rows from every online node, newest first."""
    wanted = (query.get("node") or ["all"])[0]
    nodes = [n for n in REGISTRY.all() if n.online and (wanted in ("all", "") or n.id == wanted)]
    q = {k: v for k, v in query.items() if k != "node"}
    limit = int((query.get("limit") or ["300"])[0] or 300)

    def fetch(node: Node):
        try:
            _, data = agent_call(node.cfg["url"], node.cfg.get("token", ""), "/v1/requests",
                                 query=q, timeout=10)
            return node, data, None
        except AgentError as exc:
            return node, None, str(exc)

    rows, errors = [], {}
    for node, data, err in POOL.map(fetch, nodes):
        if err:
            errors[node.id] = err
            continue
        for row in (data or {}).get("rows", []):
            row["node"] = node.id
            rows.append(row)
        if data and data.get("error"):
            errors[node.id] = data["error"]
    rows.sort(key=lambda r: r.get("ts") or 0, reverse=True)
    return {"rows": rows[:limit], "errors": errors}


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = f"llm-hub/{HUB_VERSION}"
    protocol_version = "HTTP/1.1"

    # -- helpers -----------------------------------------------------------
    def _send(self, status: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, data, status: int = 200) -> None:
        self._send(status, json.dumps(data, default=str).encode(), "application/json; charset=utf-8",
                   {"Cache-Control": "no-store"})

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        data = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        if not isinstance(data, dict):
            raise ValueError("JSON body must be an object")
        return data

    def _authorized(self) -> bool:
        if not HUB_PASSWORD:
            return True
        header = self.headers.get("Authorization", "")
        if not header.startswith("Basic "):
            return False
        try:
            user, _, pw = base64.b64decode(header[6:]).decode().partition(":")
        except Exception:  # noqa: BLE001
            return False
        return hmac.compare_digest(user, HUB_USER) and hmac.compare_digest(pw, HUB_PASSWORD)

    def _challenge(self) -> None:
        self._send(401, b"Authentication required", "text/plain",
                   {"WWW-Authenticate": 'Basic realm="LLM Fleet"'})

    def _node_or_404(self, node_id: str) -> Node | None:
        node = REGISTRY.get(node_id)
        if node is None:
            self._json({"error": f"Unknown node '{node_id}'"}, 404)
        return node

    def _proxy(self, node: Node, path: str, method: str = "GET", body=None, query=None) -> None:
        try:
            status, data = agent_call(node.cfg["url"], node.cfg.get("token", ""), path, method,
                                      body, timeout=30 if method == "POST" else 15, query=query)
        except AgentError as exc:
            self._json({"error": str(exc)}, exc.status)
            return
        if method == "POST":
            node.kick.set()
        self._json(data, status)

    # -- GET ---------------------------------------------------------------
    def do_HEAD(self) -> None:
        self.do_GET()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path
        query = parse_qs(parsed.query)
        if route == "/healthz":
            self._json({"ok": True})
            return
        if not self._authorized():
            self._challenge()
            return
        if route == "/api/stream":
            self._stream()
            return
        if route == "/api/fleet":
            self._json({"version": HUB_VERSION, "server_time": time.time(),
                        "nodes": [n.public(stream=True) for n in REGISTRY.all()]})
            return
        if route == "/api/nodes":
            self._json({"nodes": [{k: v for k, v in n.public().items() if k != "snapshot"}
                                  for n in REGISTRY.all()]})
            return
        if route == "/api/requests":
            self._json(fleet_requests(query))
            return
        m = re.fullmatch(r"/api/nodes/([^/]+)/(.+)", route)
        if m:
            node = self._node_or_404(m.group(1))
            if node is None:
                return
            sub = m.group(2)
            if sub == "models":
                with node.lock:
                    snap = node.snapshot or {}
                self._json({"models": snap.get("models") or []})
                return
            target = NODE_GET.get(sub)
            if target is None:
                self._json({"error": "Not found"}, 404)
                return
            self._proxy(node, target, query=query)
            return
        if route.startswith("/api/"):
            self._json({"error": "Not found"}, 404)
            return
        self._static(route)

    def _static(self, route: str) -> None:
        rel = route.lstrip("/") or "index.html"
        path = (STATIC_DIR / rel).resolve()
        if not str(path).startswith(str(STATIC_DIR.resolve())) or not path.is_file():
            path = STATIC_DIR / "index.html"  # SPA fallback
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
            ctype += "; charset=utf-8"
        cache = "no-cache" if path.name == "index.html" else "public, max-age=60"
        self._send(200, path.read_bytes(), ctype, {"Cache-Control": cache})

    def _stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        sent_models: dict[str, str] = {}
        try:
            self.wfile.write(b"retry: 2000\n\n")
            while True:
                nodes = []
                for n in REGISTRY.all():
                    include = sent_models.get(n.id) != n.models_hash
                    nodes.append(n.public(include_models=include, stream=True))
                    if include and n.snapshot is not None:
                        sent_models[n.id] = n.models_hash
                payload = json.dumps({"server_time": time.time(), "nodes": nodes}, default=str)
                self.wfile.write(f"event: fleet\ndata: {payload}\n\n".encode())
                self.wfile.flush()
                time.sleep(STREAM_INTERVAL_SEC)
        except (BrokenPipeError, ConnectionResetError, OSError):
            return

    # -- POST / PUT / DELETE ----------------------------------------------
    def do_POST(self) -> None:
        if not self._authorized():
            self._challenge()
            return
        route = urlparse(self.path).path
        try:
            body = self._body()
        except ValueError as exc:
            self._json({"error": f"Invalid JSON: {exc}"}, 400)
            return

        if route == "/api/nodes":
            if not body.get("url"):
                self._json({"error": "url is required"}, 400)
                return
            node = REGISTRY.add(body)
            node.kick.set()
            self._json({"node": {k: v for k, v in node.public().items() if k != "snapshot"}})
            return
        if route == "/api/nodes/test":
            url = (body.get("url") or "").strip()
            token = body.get("token") or ""
            if not token and body.get("id"):
                existing = REGISTRY.get(body["id"])
                token = existing.cfg.get("token", "") if existing else ""
            if not url:
                self._json({"ok": False, "error": "url is required"}, 400)
                return
            started = time.time()
            try:
                status, info = agent_call(url, token, "/v1/info", timeout=6)
                ok = status == 200
                self._json({"ok": ok, "info": info if ok else None,
                            "error": None if ok else info.get("error"),
                            "latency_ms": round((time.time() - started) * 1000, 1)})
            except AgentError as exc:
                self._json({"ok": False, "error": str(exc)})
            return
        if route == "/api/token":
            self._json({"token": secrets.token_urlsafe(32)})
            return
        m = re.fullmatch(r"/api/nodes/([^/]+)/(.+)", route)
        if m:
            node = self._node_or_404(m.group(1))
            if node is None:
                return
            target = NODE_POST.get(m.group(2))
            if target is None:
                self._json({"error": "Not found"}, 404)
                return
            self._proxy(node, target, "POST", body)
            return
        self._json({"error": "Not found"}, 404)

    def do_PUT(self) -> None:
        if not self._authorized():
            self._challenge()
            return
        m = re.fullmatch(r"/api/nodes/([^/]+)", urlparse(self.path).path)
        if not m:
            self._json({"error": "Not found"}, 404)
            return
        try:
            body = self._body()
        except ValueError as exc:
            self._json({"error": f"Invalid JSON: {exc}"}, 400)
            return
        node = REGISTRY.update(m.group(1), body)
        if node is None:
            self._json({"error": "Unknown node"}, 404)
            return
        self._json({"node": {k: v for k, v in node.public().items() if k != "snapshot"}})

    def do_DELETE(self) -> None:
        if not self._authorized():
            self._challenge()
            return
        m = re.fullmatch(r"/api/nodes/([^/]+)", urlparse(self.path).path)
        if not m:
            self._json({"error": "Not found"}, 404)
            return
        self._json({"removed": REGISTRY.remove(m.group(1))})

    def log_message(self, fmt: str, *args) -> None:
        pass


def main() -> None:
    mimetypes.add_type("application/javascript", ".js")
    mimetypes.add_type("text/css", ".css")
    mimetypes.add_type("image/svg+xml", ".svg")
    REGISTRY.load()
    log(f"llm-hub {HUB_VERSION} on {HOST}:{PORT} with {len(REGISTRY.all())} node(s); data in {DATA_DIR}")
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.daemon_threads = True
    server.serve_forever()


if __name__ == "__main__":
    main()
