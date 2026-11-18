# llm-hub: the web dashboard. Build context is hub/.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 HUB_PORT=80 HUB_DATA_DIR=/data
COPY hub.py /opt/llm-hub/hub.py
COPY static /opt/llm-hub/static
VOLUME /data
EXPOSE 80
HEALTHCHECK --interval=15s --timeout=5s --start-period=5s --retries=5 \
  CMD python3 -c "import urllib.request,os; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"HUB_PORT\",\"80\")}/healthz', timeout=4)"
WORKDIR /opt/llm-hub
CMD ["python3", "-u", "hub.py"]
