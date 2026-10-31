"""The ThermalView HTTP API, described once.

The same description is served at GET /api/v1/spec (and rendered on the
console's API page) and written to docs/API.md by
`python -m thermalview apidoc`, so the documentation cannot drift from the
list of routes: tests/test_web.py checks that every documented route exists.
"""
from __future__ import annotations

INTRO = """ThermalView exposes two kinds of access.

* **Console API** (`/api/...`): used by the web console. Sign in with `POST /api/login`; the
  session cookie it returns authorises later calls. Role `viewer` can read and export; role
  `operator` can also place workloads, drain and retire servers, inject faults and acknowledge
  alerts. Every state-changing call is written to the audit log.
* **Integration API** (`/api/v1/...`): for monitoring agents and other systems. Authenticate
  with `Authorization: Bearer <token>`; tokens are listed in `config.toml` → `[api].ingest_tokens`.

All bodies are JSON. Errors return `{"error": "..."}` with status 401 (not signed in / bad
token), 403 (role not allowed), 404 (unknown server or alert), 409 (not allowed in the current
state, e.g. retiring a server that is not drained) or 422 (invalid input)."""

INGEST_NOTES = """### Where a real data centre plugs in

ThermalView never talks to hardware itself. A small agent per site (or the operator's existing
monitoring stack) reads each server's sensors and posts them here once a minute. A server must
first be registered as **external** (Inventory page, or `POST /api/servers` with
`"external": true`); from then on its readings come only from the API, and it is treated exactly
like a simulated server: forecast, ranked, alerted on, and excluded from placement if its
telemetry goes stale for more than `[recommender].stale_after_min` minutes.

| ThermalView field | Unit | Required | Typical Redfish source (DMTF Redfish, via the BMC) |
|---|---|---|---|
| `cpu_temp_c` | °C | yes | `Chassis/{id}/Thermal` → `Temperatures[]` where `PhysicalContext` is `CPU`: take the max `ReadingCelsius` (newer BMCs: `ThermalSubsystem/ThermalMetrics`) |
| `load_pct` | % | yes | Not in standard Redfish. From the OS (node exporter, SNMP `hrProcessorLoad`, Ganglia `100 - cpu_idle`) or a vendor `ProcessorMetrics` extension |
| `power_w` | W | no | `Chassis/{id}/Power` → `PowerControl[0].PowerConsumedWatts` (newer: `EnvironmentMetrics.PowerWatts`) |
| `inlet_c` | °C | no | `Thermal` → `Temperatures[]` with `PhysicalContext` = `Intake` |
| `fan_rpm` | rpm | no | `Thermal` → mean of `Fans[].Reading` where `ReadingUnits` = `RPM` |

Missing optional fields are estimated from the fleet-median heat model, so a minimal agent can
send only temperature and load. Readings outside the physical ranges in `thermalview/schema.py`
are rejected per reading, not per batch. `scripts/telemetry_agent.py` is a working example agent:
it can generate synthetic readings or map a Redfish `Thermal`/`Power` response."""

ENDPOINTS = [
    # method, path, auth, summary, body example, response example
    ("POST", "/api/login", "none", "Sign in. Sets an HttpOnly session cookie.",
     {"username": "operator", "password": "…"}, {"username": "operator", "role": "operator", "display_name": "Shift operator"}),
    ("POST", "/api/logout", "session", "Sign out.", None, {"ok": True}),
    ("GET", "/api/me", "session", "The signed-in user.", None, {"username": "viewer", "role": "viewer"}),
    ("POST", "/api/me/password", "session", "Change your own password (needs the current one). Your other sessions are signed out.",
     {"current": "old-password", "password": "new-password"}, {"username": "viewer"}),
    ("GET", "/api/users", "operator", "All accounts with role and where they are defined (config.toml or added in the console). Never passwords.",
     None, {"users": [{"username": "operator", "display_name": "Shift operator", "role": "operator", "source": "config.toml"}]}),
    ("POST", "/api/users", "operator", "Add an account. Username 3-32 lowercase characters; password at least 8 characters. "
     "Stored as a salted PBKDF2 hash in data/users.json, so it survives restarts.",
     {"username": "priya", "display_name": "Priya (NOC)", "role": "viewer", "password": "at-least-8"}, {"username": "priya", "role": "viewer"}),
    ("DELETE", "/api/users/{username}", "operator", "Remove an account added in the console and sign it out. You cannot remove yourself, "
     "the last operator, or an account defined in config.toml.", None, {"removed": "priya"}),
    ("POST", "/api/users/{username}/password", "operator", "Reset someone's password; their sessions are signed out.",
     {"password": "new-password"}, {"username": "priya"}),
    ("GET", "/api/fleet", "viewer", "Fleet snapshot: every server with location, readings, forecasts and state; per-site summary; active alerts.", None,
     {"sim_time": "…", "threshold": 72.0, "servers": ["…"], "sites": ["…"], "alerts": {"critical": 1, "warning": 2}}),
    ("GET", "/api/servers/{id}/history", "viewer", "Last `minutes` (default 180) of one server's telemetry, with the 5-min forecast issued for each minute.", None,
     {"server_id": 3, "rows": ["…"], "forecast": {"5": 61.2, "15": 62.0, "30": 62.4}}),
    ("GET", "/api/inventory", "viewer", "All servers ever registered, including retired ones.", None, {"servers": ["…"]}),
    ("POST", "/api/servers", "operator", "Add a server to a rack. `external: true` registers a server whose telemetry will arrive via /api/v1/telemetry.",
     {"site": "PNQ1", "hall": "A", "rack": "A01", "external": False, "name": None}, {"server_id": 51, "name": "PNQ1-A01-07", "kind": "simulated"}),
    ("POST", "/api/servers/{id}/maintenance", "operator", "Drain (`on: true`) or return to service. A drained server gets no new placements; running jobs finish.",
     {"on": True, "reason": "firmware update"}, {"server_id": 3, "maintenance": True}),
    ("POST", "/api/servers/{id}/retire", "operator", "Remove from the fleet. The server must be drained and idle unless `force: true`.",
     {"force": False}, {"server_id": 3, "status": "retired"}),
    ("POST", "/api/servers/{id}/fault", "operator", "Simulate or clear a failure on a simulated server: `fan` (true/false), `sensor` (true = stop reporting telemetry) or `runaway_pct` (a stuck process using this much CPU; 0 clears it).",
     {"fan": True}, {"server_id": 3, "fan_fault": True}),
    ("POST", "/api/whatif", "viewer", "Predict a failure's effect without applying it: `kind` = cooling (`site`, `hall`, `magnitude` °C), fan (`server_id`) or runaway (`server_id`, `magnitude` % CPU). Returns each affected server's predicted temperature at +5/15/30 min.",
     {"kind": "cooling", "site": "HYD2", "hall": "A", "magnitude": 8},
     {"summary": "Cooling in HYD2 hall A degrades: inlet air +8 °C. 7 of 16 servers would be over the limit within 30 minutes.", "servers": ["…"]}),
    ("GET", "/api/resilience", "viewer", "For every hall: how many servers would cross the limit within 30 minutes if its cooling degraded by `rise` °C now (default 6).", None,
     {"rise_c": 6, "halls": [{"site": "PNQ1", "hall": "A", "servers": 12, "would_cross": 2}]}),
    ("POST", "/api/halls/cooling", "operator", "Simulate degraded cooling in one hall: raise inlet air by `inlet_rise_c` (0 clears it).",
     {"site": "HYD2", "hall": "A", "inlet_rise_c": 6}, {"site": "HYD2", "hall": "A", "inlet_rise_c": 6.0}),
    ("POST", "/api/recommend", "operator", "Rank servers for a workload, optionally only in one site. Logged with the full ranking.",
     {"name": "Analytics batch", "cpu_pct": 25, "mem_gb": 32, "duration_min": 45, "site": None},
     {"recommendation_id": 12, "candidates": [{"rank": 1, "name": "HYD2-A02-03", "status": "recommended", "predicted_temp": 63.1, "explanation": "…"}]}),
    ("POST", "/api/place", "operator", "Place a workload on a server (eligibility re-checked now). Any server other than ThermalView's current top pick (within the same `site` filter) is an override and needs a `reason`, which is audited.",
     {"name": "Analytics batch", "cpu_pct": 25, "mem_gb": 32, "duration_min": 45, "server_id": 17, "recommendation_id": 12, "reason": ""},
     {"job_id": 88, "server_id": 17, "server_name": "HYD2-A02-03", "override": False}),
    ("GET", "/api/queue", "viewer", "Jobs waiting for an operator decision (manual mode, or auto mode when every eligible server would exceed the limit), each with ThermalView's current suggestion.", None,
     {"mode": "manual", "queue": [{"queue_id": 4, "name": "batch-31", "suggested": {"name": "PNQ1-A01-04", "status": "recommended"}}]}),
    ("POST", "/api/queue/{id}/approve", "operator", "Place a held job: on the suggestion, or on `server_id` (an override needs a `reason`).", {"server_id": None, "reason": ""},
     {"job_id": 90, "server_name": "PNQ1-A01-04", "queue_id": 4}),
    ("POST", "/api/queue/{id}/reject", "operator", "Reject a held job; `reason` is required and audited.", {"reason": "duplicate submission"}, {"queue_id": 4, "rejected": True}),
    ("GET", "/api/jobs", "viewer", "Recent placements (operator and scheduler).", None, {"jobs": ["…"]}),
    ("GET", "/api/alerts", "viewer", "Alerts. `status` = active (default), open, acknowledged, resolved or all. Each has a suggested `action`, and `escalated` when a critical alert went unacknowledged too long.", None, {"alerts": ["…"]}),
    ("POST", "/api/alerts/{id}/ack", "operator", "Acknowledge an alert, with an optional note.", {"note": "Facilities informed"},
     {"alert_id": 4, "status": "acknowledged", "acked_by": "operator"}),
    ("GET", "/api/audit", "viewer", "Audit log, newest first. Filters: `user`, `action`, `scheduler=0` to hide automatic placements.", None, {"entries": ["…"]}),
    ("GET", "/api/export/{kind}.csv", "viewer", "CSV download. `kind` = fleet, alerts, audit, jobs or telemetry (`minutes`, optional `server_id`).", None, "text/csv"),
    ("POST", "/api/control", "operator", "Clock and modes: `pause`, `resume`, `speed` (1-10), `auto_jobs` (job arrivals on/off), `mode` (auto or manual placement), `step`.", {"action": "mode", "value": "manual"}, {"ok": True}),
    ("GET", "/api/results", "viewer", "Offline experiment results from `results/` (what the paper reports).", None, {"forecast_metrics": ["…"]}),
    ("GET", "/api/v1/spec", "none", "This API description as JSON.", None, {"endpoints": ["…"]}),
    ("GET", "/metrics", "token", "Prometheus metrics: per-server temperature, 5/15/30-min forecasts, load, power, state, drained flag, jobs; active alerts; queue length; live forecast error. Scrape with the bearer token.", None, "text/plain (Prometheus exposition format)"),
    ("GET", "/api/v1/servers", "token", "Active servers with id, name, location and kind, so an agent can map its hosts.", None,
     {"servers": [{"server_id": 51, "name": "PNQ1-A01-07", "kind": "external"}]}),
    ("POST", "/api/v1/telemetry", "token", "Send readings for external servers (one per server per minute; up to 1000 per call).",
     {"readings": [{"server": "PNQ1-A01-07", "cpu_temp_c": 58.5, "load_pct": 41, "power_w": 610, "inlet_c": 23.1, "fan_rpm": 7200}]},
     {"accepted": 1, "rejected": [], "sim_time": "…"}),
]

AUTH_LABEL = {"none": "public", "session": "signed in", "viewer": "viewer or operator", "operator": "operator only",
              "token": "Bearer token"}


def spec() -> dict:
    return {"title": "ThermalView API", "version": "1", "intro": INTRO, "ingest_notes": INGEST_NOTES,
            "endpoints": [{"method": m, "path": p, "auth": a, "auth_label": AUTH_LABEL[a], "summary": s,
                           "body": b, "response": r} for m, p, a, s, b, r in ENDPOINTS]}


def to_markdown() -> str:
    import json
    lines = ["# ThermalView API", "", "_Generated by `python -m thermalview apidoc` from `thermalview/web/apispec.py`._",
             "", INTRO, "", "## Endpoints", "", "| Method | Path | Access | What it does |", "|---|---|---|---|"]
    for m, p, a, s, _b, _r in ENDPOINTS:
        lines.append(f"| {m} | `{p}` | {AUTH_LABEL[a]} | {s} |")
    lines += ["", INGEST_NOTES, "", "## Examples", ""]
    for m, p, a, s, b, r in ENDPOINTS:
        if b is None and not isinstance(r, dict):
            continue
        lines += [f"### {m} {p}", "", s, ""]
        if b is not None:
            lines += ["Request:", "```json", json.dumps(b, indent=2, ensure_ascii=False), "```"]
        if isinstance(r, dict):
            lines += ["Response:", "```json", json.dumps(r, indent=2, ensure_ascii=False), "```"]
        lines.append("")
    lines += ["### curl: send one reading", "", "```bash",
              "curl -X POST http://127.0.0.1:8000/api/v1/telemetry \\",
              "  -H 'Authorization: Bearer tv-demo-agent-token' -H 'Content-Type: application/json' \\",
              "  -d '{\"readings\":[{\"server\":\"PNQ1-A01-07\",\"cpu_temp_c\":58.5,\"load_pct\":41}]}'", "```", ""]
    return "\n".join(lines)
