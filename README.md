# LLM Monitor

GPU and model switch dashboard for `llama.cpp` Docker stacks.

This project runs a web UI plus terminal monitor (`nvtop` or `btop`) and discovers available models from stack files in a `stacks/` directory using `# LLM_META` headers.

## What It Does

- Discovers available models from compose stack files (`*.yml`) in a stacks directory.
- Shows active model status and health.
- Triggers model switches by calling an external `switch-llm.sh` script.
- Exposes benchmark and restart/stop actions from the UI.

## Requirements

- Docker with GPU runtime configured.
- NVIDIA device access on host (`/dev/nvidia*`, `nvidia-smi`, `libnvidia-ml.so.*`).
- External llama.cpp workspace that includes:
  - `switch-llm.sh`
  - `stacks/*.yml`
  - model files directory

## Configuration

Primary config file: [`config.yaml`](./config.yaml)

Optional local override: copy [`config.local.example.yaml`](./config.local.example.yaml) to `config.local.yaml`.
`config.local.yaml` is ignored by git.

Supported keys:

- `llama_dir`: root with `switch-llm.sh` and stacks.
- `stacks_dir`: path containing stack files with `# LLM_META` headers.
- `models_dir`: path containing model files (documented and surfaced for setup consistency).
- `llama_working_dir_label`: expected Docker Compose `working_dir` label.
- `switch_script` (optional): explicit path to `switch-llm.sh`.

Environment variable overrides are supported and take precedence over file values:

- `LLAMA_DIR`
- `STACKS_DIR`
- `MODELS_DIR`
- `LLAMA_WORKING_DIR_LABEL`
- `SWITCH_SCRIPT`
- `DASHBOARD_CONFIG` (defaults to `config.yaml`)
- `DASHBOARD_CONFIG_LOCAL` (defaults to `config.local.yaml`)

## Stack File Format

Each stack file must begin with `LLM_META` lines for discovery:

```yaml
# LLM_META display_name="Example 8B (Q4_K_M)"
# LLM_META family="example"
# LLM_META params="8B"
# LLM_META quant="Q4_K_M"
# LLM_META ctx_size="32768"
# LLM_META thinking="false"
# LLM_META server_service="llm-server"
# LLM_META sort_order="100"
services:
  llm-server:
    image: ghcr.io/ggml-org/llama.cpp:server-cuda
```

See sanitized examples in [`examples/stacks/`](./examples/stacks/).

## External Switch Script Contract

The dashboard depends on an external `switch-llm.sh` and calls it like:

- `switch-llm.sh --stack stacks/<file>.yml`
- `switch-llm.sh --stop-all`

Your script should handle those arguments and return non-zero on failure.

## Run With Docker Compose

```bash
docker compose up -d --build
```

Default ports:

- Dashboard: `9358`
- Terminal monitor (ttyd): `9359`

## Security Notes

- Do not commit real tokens, private model paths, or host-specific secrets.
- Keep local overrides in `config.local.yaml`.
- Use sanitized stack examples for sharing.

## License

MIT. See [`LICENSE`](./LICENSE).
