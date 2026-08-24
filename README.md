# ThermalView

ThermalView is an infrastructure monitoring and thermal-aware workload scheduling prototype for enterprise server environments.

## Highlights

- HPE iLO / Redfish-oriented server discovery and telemetry
- Server health, thermal, power, and monitoring views
- Predictive thermal analysis and thermal-risk classification
- Thermal-aware workload/server ranking and scheduling recommendations
- FastAPI web application with SQLite persistence
- Prometheus metrics and Grafana dashboards
- Docker-based deployment support
- Operator decision auditing and authentication support

## Tech Stack

Python, FastAPI, SQLModel, SQLite, Scikit-learn, HPE iLO/Redfish, Prometheus, Grafana, Docker, HTML/Jinja2.

## Local Development

```bash
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/`.

## Docker

```bash
docker compose up --build -d
```

## Security

Secrets, local databases, cookies, virtual environments, and runtime artifacts are intentionally excluded from version control. Copy `.env.example` to `.env` and provide your own credentials and secrets.

## Project Structure

```text
app/           FastAPI application, monitoring, prediction, scheduling
app/templates/ Web UI templates
grafana/       Grafana provisioning and dashboards
monitoring/    Prometheus configuration and alert rules
tests/         Automated tests
scripts/       Deployment/release utilities
tools/         Supporting infrastructure utilities
```
