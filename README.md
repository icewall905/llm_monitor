# LLM Fleet

Dashboard and model switcher for self-hosted LLM stacks (llama.cpp, vLLM, Strata,
TabbyAPI, BeeLlama) across one or more GPU hosts.

```
browser ──SSE──> hub (:9358)  ──bearer token──>  agent (:9360) on each GPU host
                                                  docker.sock · nvidia-smi · stack dirs · request log
```

- **agent/** runs on every GPU host. It discovers stacks from `# LLM_META` headers,
  reads the GPU, container logs and engine metrics, keeps metrics/benchmark history,
  serves the request log, and performs switches (one model per host). All of it is
  exposed over a token-protected JSON API (`/v1/...`).
- **hub/** is the web dashboard. It polls each registered agent once a second,
  streams the fleet to the browser over SSE, and proxies actions. Agent tokens never
  reach the browser.
- **legacy/** is the previous single-host dashboard, kept runnable for comparison.

## Views

| View | What it shows |
|---|---|
| Fleet | One card per host: active model + backend, live gen/ingest t/s, context, per-GPU util/VRAM/temp/power, switch progress |
| Host | Hero with actions, switch stepper, KPI tiles (gen, ingest, context, KV pool, streams, prefix cache, GPU, VRAM, temp), performance + thermal charts (live → 30 d), engine metrics with a bottleneck note, benchmarks, GPU processes |
| Models | Every stack on every host, grouped by family, filter by host/backend/reasoning/vision, last benchmark and KV pool, one-click switch |
| Requests | Request log across hosts (inference-only filter), detail sheet with the chat transcript |
| Events | Switches, loads, OOMs, heartbeats from every agent |
| Settings | Add/edit/test hosts, per-host stack folders, theme |

Keyboard: `⌘K`/`Ctrl+K` or `/` command palette (switch any model, jump anywhere),
`f m r e s` views, `1-9` hosts.

## Run (this host: hub + agent)

```bash
echo "AGENT_TOKEN=$(openssl rand -base64 32)" > .env   # once
docker compose up -d --build
```

Dashboard on `:9358`. The local agent is registered automatically on first start;
hosts are then stored in `./data/hub/nodes.json` and managed in Settings.
Set `HUB_PASSWORD` in `.env` to put basic auth on the dashboard.

## Add another GPU host

1. Copy `agent/compose.example.yml` to the host, adjust the stack paths (mount them
   at the same path inside the container) and set `AGENT_TOKEN` in a `.env` beside it.
2. `docker compose -f compose.example.yml up -d --build`
3. In the hub: Settings → Add host → `http://<host>:9360` + the token → Test → Add.

## Stack discovery

Each stack file starts with `LLM_META` lines:

```yaml
# LLM_META display_name="Example 8B (Q4_K_M)"
# LLM_META family="example"          # grouping; family="infra" is hidden from the model list
# LLM_META params="8B"
# LLM_META quant="Q4_K_M"
# LLM_META ctx_size="32768"
# LLM_META thinking="false"
# LLM_META vision="false"
# LLM_META server_service="llm-server"
# LLM_META internal_port="8080"      # engine port inside its container
# LLM_META max_seqs="8"              # vLLM slot cap; also fed to the proxy /props
# LLM_META backend="strata"          # optional; otherwise guessed from the file
# LLM_META sort_order="100"
```

Folders scanned by the agent (env, `config.yaml`, or Settings in the UI):
`STACKS_DIR` (llama.cpp), `VLLM_DIR` (vLLM/Strata/TabbyAPI), `BEELLAMA_STACKS_DIR`.
llama.cpp switches call `switch-llm.sh --stack stacks/<file>.yml` / `--stop-all`.
vLLM-dir stacks that ship only the model server get `VLLM_DIR/llm-proxies.yml`
started after them, with `/props` values from the stack's `max_seqs`/`props_*` meta.

## Agent API

All routes need `Authorization: Bearer $AGENT_TOKEN` (except `/healthz`).

| | |
|---|---|
| `GET /v1/info` | name, version, GPUs, backends |
| `GET /v1/snapshot` | full live status (refreshed every second in the background) |
| `GET /v1/history?range=5m\|15m\|1h\|24h\|7d\|30d` | bucketed metrics |
| `GET /v1/requests?limit&endpoint&q&only=inference`, `/v1/requests/detail?id=`, `/v1/requests/endpoints` | request log |
| `POST /v1/switch {model}`, `/v1/restart`, `/v1/stop`, `/v1/switch/clear`, `/v1/benchmark {profile}` | control |
| `GET/POST /v1/settings`, `GET /v1/settings/inspect?path=` | stack folders |

## Tests

```bash
cd playwright-tests && BASE_URL=http://localhost:9358 npx playwright test
```

## License

MIT. See [`LICENSE`](./LICENSE).
