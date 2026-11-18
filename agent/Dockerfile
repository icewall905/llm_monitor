# llm-agent: runs on each GPU host. Build context is the repo root:
#   docker build -f agent/Dockerfile -t llm-agent .
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl python3 docker.io procps && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /usr/local/lib/docker/cli-plugins && \
    curl -fsSL -o /usr/local/lib/docker/cli-plugins/docker-compose \
      https://github.com/docker/compose/releases/download/v2.40.1/docker-compose-linux-x86_64 && \
    chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

COPY switch-llm.sh /gpu-monitor/switch-llm.sh
COPY agent/core.py agent/agent.py /opt/llm-agent/
RUN chmod +x /gpu-monitor/switch-llm.sh

ENV AGENT_PORT=9360 SWITCH_SCRIPT=/gpu-monitor/switch-llm.sh
EXPOSE 9360
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=5 \
  CMD curl -fsS http://localhost:${AGENT_PORT}/healthz || exit 1
WORKDIR /opt/llm-agent
CMD ["python3", "-u", "agent.py"]
