# ThermalView operator console. Build once the pipeline has produced artifacts/ and results/
# (or mount them at run time, as docker-compose.yml does).
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY thermalview/ thermalview/
COPY scripts/ scripts/
COPY docs/ docs/
COPY config.toml .
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/spec', timeout=4)"
CMD ["python", "-m", "thermalview", "serve", "--lan"]
