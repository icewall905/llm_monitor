FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install tools and ttyd.
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl nvtop btop python3 docker.io nginx && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /usr/local/lib/docker/cli-plugins && \
    curl -fsSL -o /usr/local/lib/docker/cli-plugins/docker-compose \
      https://github.com/docker/compose/releases/download/v2.40.1/docker-compose-linux-x86_64 && \
    chmod +x /usr/local/lib/docker/cli-plugins/docker-compose && \
    curl -fsSL -o /usr/local/bin/ttyd \
      https://github.com/tsl0922/ttyd/releases/download/1.7.7/ttyd.x86_64 && \
    chmod +x /usr/local/bin/ttyd

COPY launch-monitor.sh /usr/local/bin/launch-monitor.sh
COPY switch-llm.sh /gpu-monitor/switch-llm.sh
COPY start-services.sh /usr/local/bin/start-services.sh
COPY dashboard-server.py /usr/local/bin/dashboard-server.py
COPY nginx.conf /etc/nginx/nginx.conf
COPY btop.conf /tmp/btop.conf
RUN mkdir -p /gpu-monitor /root/.config/btop && \
    mv /tmp/btop.conf /root/.config/btop/btop.conf && \
    chmod +x /usr/local/bin/launch-monitor.sh /usr/local/bin/start-services.sh /gpu-monitor/switch-llm.sh

EXPOSE 80 7681

CMD ["/usr/local/bin/start-services.sh"]
