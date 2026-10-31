"""Operator console: fleet, roles, audit, alerts, inventory, telemetry API.

A tiny environment is built in a temp folder (fixture telemetry, fitted
heat models, a stand-in forecaster that predicts "no change"), so these
tests take seconds and need no pipeline run.
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from thermalview import thermal_model as TM
from thermalview.config import load_config
from thermalview.data.fixture import make_fixture
from thermalview.data.prepare import regularise, save_table
from thermalview.recommender import Workload
from thermalview.web import apispec
from thermalview.web.live import ActionError, LiveApp

COLS = ["temp", "temp_d1", "temp_d5", "temp_d15", "temp_mean15", "temp_std15", "load", "load_d1", "load_d5",
        "load_mean15", "power", "power_d5", "inlet", "inlet_d15", "fan", "fan_d5"]
OP = {"username": "operator", "role": "operator"}


def _env(tmp_path: Path):
    cfg = load_config()
    cfg.root = tmp_path
    cfg["paths"] = {"raw_m100": "raw", "processed": "processed", "artifacts": "artifacts", "results": "results",
                    "database": "tv.db"}
    table = regularise(make_fixture(n_nodes=6, days=2, seed=3), 1, 2)
    save_table(table, cfg.path("processed"), "telemetry", {"source": "dev-fixture"})
    split_end = table["ts"].min() + pd.Timedelta(hours=30)
    params = TM.fit_all(table, split_end, 0.005, 0.8)
    TM.save_params(params, cfg.path("artifacts") / "heat_params.json")
    model = DummyRegressor(strategy="constant", constant=0.0).fit([[0] * len(COLS)], [0.0])
    joblib.dump({"models": {5: model, 15: model, 30: model}, "chosen": {5: "persistence", 15: "persistence", 30: "persistence"},
                 "feature_columns": COLS, "optional_present": ["power_w", "inlet_c", "fan_rpm"], "threshold_c": 70.0,
                 "primary_horizon": 5, "data_source": "dev-fixture"}, cfg.path("artifacts") / "forecaster.joblib")
    res = cfg.path("results")
    res.mkdir(parents=True)
    (res / "forecast_summary.json").write_text(json.dumps({"splits": {"5": {"holdout_nodes": [], "t_val_end": str(split_end)}},
                                                           "config": {"primary_horizon": 5}}))
    return cfg


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    return LiveApp(_env(tmp_path_factory.mktemp("tv")))


def test_fleet_has_configured_topology(live):
    f = live.fleet()
    assert len(f["servers"]) == 50
    assert [s["code"] for s in f["sites"]] == ["PNQ1", "HYD2", "MAA1"]
    assert sum(len(r["servers"]) for s in f["sites"] for h in s["halls"] for r in h["racks"]) == 50
    names = {s["name"] for s in f["servers"]}
    assert "PNQ1-A01-01" in names and "MAA1-A01-10" in names
    # servers copying the same node start at different points of its trace
    same = [s for s in live.sim.servers.values() if s.params.node_id == live.node_pool[0]]
    assert len(same) > 1 and len({s.trace_offset for s in same}) == len(same)


def test_users_and_passwords(live):
    assert live.store.check_user("operator", "operator-demo")["role"] == "operator"
    assert live.store.check_user("viewer", "viewer-demo")["role"] == "viewer"
    assert live.store.check_user("operator", "wrong") is None
    assert live.store.check_user("nobody", "x") is None
    row = live.store.db.execute("SELECT pw_hash FROM users WHERE username='operator'").fetchone()
    assert b"operator-demo" not in row[0]


def test_drain_blocks_placement_and_retire_needs_drain(live):
    sid = live.server_by_ref("PNQ1-A01-02").server_id
    with pytest.raises(ActionError):
        live.retire_server(OP, sid)
    live.set_maintenance(OP, sid, True, "test")
    ranked = {c.server_id: c for c in live.rank(Workload(5, 1, 10))}
    assert ranked[sid].status == "ineligible" and "maintenance" in ranked[sid].reasons[0]
    with pytest.raises(ActionError):
        live.place_for(OP, Workload(5, 1, 10), sid, None)
    live.retire_server(OP, sid)
    assert sid not in live.sim.servers
    inv = {s["server_id"]: s for s in live.store.inventory()}
    assert inv[sid]["status"] == "retired"


def test_recommend_by_site_and_audit_trail(live):
    r = live.recommend_for(OP, Workload(10, 8, 20, "t"), "MAA1")
    assert {c["site"] for c in r["candidates"]} == {"MAA1"}
    top = next(c for c in r["candidates"] if c["status"] != "ineligible")
    live.place_for(OP, Workload(10, 8, 20, "t"), top["server_id"], r["recommendation_id"], site="MAA1")
    actions = [e["action"] for e in live.store.audit_log(include_scheduler=False)]
    assert actions[:2] == ["place", "recommend"]
    assert {"drain", "server_retire", "start"} <= set(actions)


def test_external_server_ingest_stale_and_validation(live):
    out = live.add_server(OP, "HYD2", "A", "A01", True, "HYD2-EDGE-01")
    sid = out["server_id"]
    assert out["kind"] == "external"
    live.advance()
    assert live.fleet()["servers"][-1]["state"] == "stale"
    res = live.ingest([{"server": "HYD2-EDGE-01", "cpu_temp_c": 55, "load_pct": 30},
                       {"server": "PNQ1-A01-01", "cpu_temp_c": 55, "load_pct": 30},
                       {"server": "HYD2-EDGE-01", "cpu_temp_c": 900, "load_pct": 30},
                       {"server": "HYD2-EDGE-01", "load_pct": 30}])
    assert res["accepted"] == 1
    errs = [x["error"] for x in res["rejected"]]
    assert "simulated server" in errs[0] and "outside" in errs[1] and "required" in errs[2]
    for _ in range(20):                       # an agent sending every minute keeps it fresh
        live.ingest([{"server_id": sid, "cpu_temp_c": 55, "load_pct": 30}])
        live.advance()
    assert next(s for s in live.server_rows() if s["server_id"] == sid)["state"] != "stale"
    assert sid in live.forecasts[5]           # forecasts work from ingested history
    for _ in range(5):                        # agent goes silent
        live.advance()
    srv = next(s for s in live.server_rows() if s["server_id"] == sid)
    assert srv["state"] == "stale"
    assert any(a["kind"] == "stale" and a["server_id"] == sid for a in live.store.alerts("active"))
    with pytest.raises(ActionError):
        live.set_fan_fault(OP, sid, True)


def test_alert_opens_acks_and_resolves(live):
    sid = live.server_by_ref("MAA1-A01-03").server_id
    live.set_fan_fault(OP, sid, True)
    live.advance()
    a = next(a for a in live.store.alerts("active") if a["server_id"] == sid and a["kind"] == "fan_fault")
    assert a["severity"] == "critical"
    acked = live.ack_alert(OP, a["alert_id"], "tech dispatched")
    assert acked["status"] == "acknowledged" and acked["acked_by"] == "operator"
    live.set_fan_fault(OP, sid, False)
    for _ in range(6):
        live.advance()
    assert next(x for x in live.store.alerts("all") if x["alert_id"] == a["alert_id"])["status"] == "resolved"


def test_hall_cooling_raises_inlet_only_in_that_hall(live):
    hyd = [s for s in live.sim.servers.values() if live.loc[s.server_id]["site"] == "HYD2" and not s.external]
    pnq = [s for s in live.sim.servers.values() if live.loc[s.server_id]["site"] == "PNQ1"]
    before = (hyd[0].inlet_bias, pnq[0].inlet_bias)
    live.set_cooling(OP, "HYD2", "A", 6)
    assert hyd[0].inlet_bias == pytest.approx(before[0] + 6) and pnq[0].inlet_bias == before[1]
    live.set_cooling(OP, "HYD2", "A", 0)


def test_documented_routes_exist(tmp_path):
    from thermalview.web.server import create_app
    cfg = _env(tmp_path)
    app = create_app(cfg)
    paths = {getattr(r, "path", None) for r in app.routes}
    for e in apispec.spec()["endpoints"]:
        p = e["path"].replace("{id}", "{sid:int}").replace("/api/alerts/{sid:int}", "/api/alerts/{aid:int}")
        if p.startswith("/api/queue/{sid:int}/"):
            p = "/api/queue/{qid:int}/{decision:str}"
        p = p.replace("{kind}.csv", "{kind}.csv")
        assert p in paths, e["path"]
    md = apispec.to_markdown()
    assert "/api/v1/telemetry" in md and "Redfish" in md


def test_http_roles_and_ingest_token(tmp_path):
    pytest.importorskip("httpx")
    from starlette.testclient import TestClient
    from thermalview.web.server import create_app
    app = create_app(_env(tmp_path))
    with TestClient(app) as c:
        assert c.get("/api/fleet").status_code == 401
        assert c.post("/api/login", json={"username": "viewer", "password": "nope"}).status_code == 401
        assert c.post("/api/login", json={"username": "viewer", "password": "viewer-demo"}).status_code == 200
        assert c.get("/api/fleet").status_code == 200
        assert c.post("/api/servers/1/maintenance", json={"on": True}).status_code == 403
        csv = c.get("/api/export/alerts.csv")
        assert csv.status_code == 200 and csv.headers["content-type"].startswith("text/csv")
        c.post("/api/logout")
        c.post("/api/login", json={"username": "operator", "password": "operator-demo"})
        assert c.post("/api/servers/1/maintenance", json={"on": True}).json()["maintenance"] is True
        assert c.post("/api/servers/1/retire", json={}).status_code in (200, 409)
        assert c.post("/api/v1/telemetry", json={"readings": [{}]}).status_code == 401
        r = c.post("/api/v1/telemetry", headers={"Authorization": "Bearer tv-demo-agent-token"},
                   json={"readings": [{"server": "PNQ1-A01-03", "cpu_temp_c": 50, "load_pct": 1}]})
        assert r.status_code == 200 and r.json()["accepted"] == 0
        actions = [e["action"] for e in c.get("/api/audit").json()["entries"]]
        assert "login_failed" in actions and "export" in actions and "drain" in actions


def test_redfish_mapping():
    import telemetry_agent as agent
    base = Path(__file__).resolve().parent.parent / "scripts" / "sample_redfish"
    r = agent.redfish_to_reading(json.loads((base / "thermal.json").read_text()),
                                 json.loads((base / "power.json").read_text()), 40)
    assert r == {"cpu_temp_c": 63.5, "load_pct": 40.0, "inlet_c": 23.0, "fan_rpm": 7200.0, "power_w": 640.0}
    with pytest.raises(ValueError):
        agent.redfish_to_reading({"Temperatures": []}, None, 10)


def test_only_operators_manage_users_and_accounts_persist(tmp_path):
    from starlette.testclient import TestClient
    from thermalview.web.server import create_app
    cfg = _env(tmp_path)
    op, vw = TestClient(create_app(cfg)), None
    assert op.post("/api/login", json={"username": "operator", "password": "operator-demo"}).status_code == 200
    app = op.app
    vw = TestClient(app)
    assert vw.post("/api/login", json={"username": "viewer", "password": "viewer-demo"}).status_code == 200
    # viewers cannot list, add or remove accounts
    assert vw.get("/api/users").status_code == 403
    assert vw.post("/api/users", json={"username": "x-user", "password": "longenough", "role": "operator"}).status_code == 403
    # operators can
    assert op.post("/api/users", json={"username": "priya", "password": "short", "role": "viewer"}).status_code == 422
    r = op.post("/api/users", json={"username": "priya", "password": "priya-pass-1", "role": "viewer", "display_name": "Priya"})
    assert r.status_code == 200
    assert op.post("/api/users", json={"username": "priya", "password": "priya-pass-1", "role": "viewer"}).status_code == 409
    names = {u["username"]: u for u in op.get("/api/users").json()["users"]}
    assert names["priya"]["source"] == "added in console" and names["operator"]["source"] == "config.toml"
    assert "pw_hash" not in names["priya"] and "salt" not in names["priya"]
    p = TestClient(app)
    assert p.post("/api/login", json={"username": "priya", "password": "priya-pass-1"}).status_code == 200
    # self-service password change needs the current password
    assert p.post("/api/me/password", json={"current": "wrong", "password": "new-pass-123"}).status_code == 403
    assert p.post("/api/me/password", json={"current": "priya-pass-1", "password": "new-pass-123"}).status_code == 200
    # rules: not yourself, not config accounts
    assert op.delete("/api/users/operator").status_code == 409
    assert op.delete("/api/users/viewer").status_code == 409
    # survives a restart (fresh app, same folder); the file holds hashes, never the password
    text = (cfg.root / "data" / "users.json").read_text()
    assert "new-pass-123" not in text and "priya" in text
    again = TestClient(create_app(_env_same(cfg)))
    assert again.post("/api/login", json={"username": "priya", "password": "new-pass-123"}).status_code == 200
    # removal signs the user out at once
    assert op.delete("/api/users/priya").status_code == 200
    assert p.get("/api/me").status_code == 401
    audit = [e["action"] for e in op.get("/api/audit").json()["entries"]]
    assert {"user_add", "user_password", "user_remove"} <= set(audit)


def _env_same(cfg):
    from thermalview.config import load_config
    c = load_config()
    c.root = cfg.root
    c["paths"] = dict(cfg["paths"], database="tv2.db")
    return c
