# Adding a GPU host to LLM Fleet

Each GPU host runs one **agent** (`llm-agent`). The agent watches the host's GPU, its LLM stacks (llama.cpp, vLLM, Strata, TabbyAPI, BeeLlama) and its request log, and lets the hub switch models there. The **hub** (the dashboard at llm.tdgs.org / port 9358) polls every agent you register.

```
hub (rtx3090 :9358)  ──Bearer token──>  llm-agent (new host :9360)
```

Takes about 10 minutes. Nothing on the host's running stacks is restarted.

---

## 1. Before you start

On the new host you need:

- Docker with Compose v2 (`docker compose version`).
- An NVIDIA driver (`nvidia-smi` works on the host).
- Your LLM stacks as compose files with `# LLM_META` headers (see the [README](../README.md#stack-discovery)).
- Port **9360** reachable from the hub host (10.0.10.23).

Collect these facts — you will put them in the compose file:

| What | How to find it | Example (hbawl) |
|---|---|---|
| Stack folders | where your `*.yml` with `LLM_META` live | `/root/compose/strata` |
| Compose project of a running stack | `docker inspect <server> --format '{{index .Config.Labels "com.docker.compose.project.working_dir"}}'` | `/root/compose/strata` |
| Request log folder | `docker inspect <access-logger> --format '{{range .Mounts}}{{.Source}} -> {{.Destination}}{{println}}{{end}}'` — the folder mounted at `/logs` | `/root/compose/gpu-monitor/request-logs` |
| Engine API key file (vLLM/Strata) | the file the stack mounts as its key | `/root/.config/strata/api-key` |
| NVML library version | `ls /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.*` | `libnvidia-ml.so.580.178.04` |

## 2. Get the agent onto the host

Either clone the repo:

```bash
git clone https://github.com/icewall905/llm_monitor.git /root/compose/llm-agent
cd /root/compose/llm-agent
```

…or copy just what the image needs from a machine that has it:

```bash
ssh root@NEW_HOST 'mkdir -p /root/compose/llm-agent/agent'
scp agent/agent.py agent/core.py agent/Dockerfile root@NEW_HOST:/root/compose/llm-agent/agent/
scp switch-llm.sh root@NEW_HOST:/root/compose/llm-agent/
```

The agent image is built from `agent/Dockerfile` with the repo root (the folder holding `switch-llm.sh`) as build context.

## 3. Create a token

Each host gets its own token. On the new host:

```bash
cd /root/compose/llm-agent
umask 077
printf 'AGENT_TOKEN=%s\nAGENT_NAME=%s\n' "$(openssl rand -base64 32 | tr -d '/+=' | head -c 40)" "$(hostname)" > .env
```

Keep `.env` private — anyone with the token can start and stop containers on this host.

## 4. Write the compose file

Create `/root/compose/llm-agent/docker-compose.yml` (a cloned repo already has a `docker-compose.yml` for the hub host — write this one as `agent-compose.yml` and pass `-f agent-compose.yml` below). Start from [`agent/compose.example.yml`](../agent/compose.example.yml). This is the one used on hbawl, a Strata-only host:

```yaml
services:
  llm-agent:
    build:
      context: .
      dockerfile: agent/Dockerfile
    container_name: llm-agent
    pid: "host"                        # CPU/RAM per GPU process
    environment:
      - AGENT_TOKEN=${AGENT_TOKEN:?set AGENT_TOKEN in .env}
      - AGENT_NAME=${AGENT_NAME:-hbawl}
      # Stack folders. Missing ones are skipped.
      - LLAMA_DIR=/root/compose/llama.cpp
      - STACKS_DIR=/root/compose/llama.cpp/stacks
      - LLAMA_WORKING_DIR_LABEL=/root/compose/llama.cpp
      - VLLM_DIR=/root/compose/strata          # vLLM, Strata and TabbyAPI stacks
      - VLLM_SERVER_SERVICE=strata             # default service name of the server
      - VLLM_API_KEY_FILE=/run/secrets/engine-api-key
      # Request log written by the access-logger; history is kept beside it.
      - REQUEST_LOG_DB=/request-logs/requests.db
      - METRICS_DB=/request-logs/metrics.db
      - DASHBOARD_SETTINGS=/request-logs/settings.json
    ports:
      - "9360:9360"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      # Mount stack folders at the SAME path as on the host.
      - /root/compose/llama.cpp:/root/compose/llama.cpp:ro
      - /root/compose/strata:/root/compose/strata:ro
      - /root/.config/strata/api-key:/run/secrets/engine-api-key:ro
      - /root/compose/gpu-monitor/request-logs:/request-logs:rw
      # NVIDIA userspace from the host — match your driver version.
      - /usr/bin/nvidia-smi:/usr/bin/nvidia-smi:ro
      - /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1:ro
      - /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.178.04:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.178.04:ro
    devices:
      - /dev/nvidia0:/dev/nvidia0
      - /dev/nvidiactl:/dev/nvidiactl
      - /dev/nvidia-uvm:/dev/nvidia-uvm
    restart: always
```

Rules that matter:

- **Same paths inside and outside.** Docker labels running containers with their host compose path; the agent matches stacks by that path. `/srv/x:/srv/x`, never `/srv/x:/stacks`.
- **Stacks read-only, request-log folder read-write** (SQLite needs to create `-wal`/`-shm` files).
- **Optional extras:**
  - `/dev/shm:/host-shm:rw` — lets the agent delete leaked vLLM KV-offload regions. Only for hosts running vLLM with `ipc: host`.
  - Build-context folders (read-only, same path) for stacks whose images the agent must rebuild when pruned (`x-gpu-monitor-images` or `build:` in the stack).
  - More GPUs: add `/dev/nvidia1`, … under `devices`.
- **What switching runs:** llama.cpp stacks go through `switch-llm.sh` (project `llamacpp`); stacks in `VLLM_DIR` run as project `vllm`, and `VLLM_DIR/llm-proxies.yml` is started after any stack that has no `access-logger` of its own.

## 5. Start and check it

```bash
docker compose up -d --build
docker ps --filter name=llm-agent            # should become (healthy)
```

From the **hub host** (replace the token):

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://NEW_HOST:9360/v1/info         # 401 = auth works
curl -s -H "Authorization: Bearer TOKEN" http://NEW_HOST:9360/v1/info
```

Look for: your GPU under `gpus`, `model_count` > 0, `request_log: true`.

## 6. Register it in the hub

1. Open the dashboard → **Settings** (gear, bottom left) → **Add host**.
2. Name, URL `http://NEW_HOST:9360`, the token, a colour.
3. **Test** — should say `Connected to "<name>" · 1 GPU · N stacks`.
4. **Add host.** It appears on the Fleet page within a second.

Hosts are stored in `data/hub/nodes.json` on the hub host. The token never leaves the hub.

## 7. If you had the old dashboard on that host

The old single-host `gpu-monitor` writes the same `metrics.db`. Once the agent works, stop it but keep it around:

```bash
docker update --restart=no gpu-monitor && docker stop gpu-monitor
```

Pointing `METRICS_DB` at the old file (as above) keeps its charts and benchmark history.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Test says **Unreachable** | Port 9360 blocked, wrong IP, or container down. `curl http://NEW_HOST:9360/healthz` from the hub host. |
| Test says **rejected the token (401)** | Token in hub ≠ `AGENT_TOKEN` in the host's `.env`. Edit the host in Settings and paste it again. |
| Agent exits: `AGENT_TOKEN is not set` | `.env` missing or not next to the compose file. |
| `gpus: []` / no GPU on the card | `docker exec llm-agent nvidia-smi` — if that fails, the NVML library mount doesn't match the driver version or a `/dev/nvidia*` device is missing. |
| `model_count: 0` | Stack folder not mounted at the same path, or no `# LLM_META display_name=` in the files. Settings → Stack folders shows what the agent sees. |
| Model shows but never **Healthy** | The server container has no healthcheck, or its `server_service` / compose path doesn't match. Check `LLM_META server_service`. |
| Requests page empty for this host | `REQUEST_LOG_DB` doesn't point at the access-logger's `requests.db`, or the folder is mounted read-only. |
| Engine card shows no vLLM/Strata metrics | `VLLM_API_KEY_FILE` missing or wrong, or the stack's `internal_port` isn't the engine port inside its container. |

## Updating the agent

```bash
cd /root/compose/llm-agent
git pull                 # or scp the new agent/*.py
docker compose up -d --build
```

## Removing a host

Settings → host → remove (the agent keeps running), then on the host: `docker compose down`.
