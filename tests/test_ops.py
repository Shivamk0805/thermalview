"""Operations behaviour of the console: alert engine, failures, what-if, manual mode."""
import http.server
import json
import threading
import time

import numpy as np
import pytest

from thermalview.recommender import Workload
from thermalview.web.live import ActionError, LiveApp

from test_web import OP, _env


def _live(tmp_path, **alerts):
    cfg = _env(tmp_path)
    cfg.setdefault("alerts", {}).update(alerts)
    return LiveApp(cfg)


def _force_temp(live, sid, temp):
    """Pin a server's replayed temperature (and its forecast) to a value."""
    s = live.sim.servers[sid]
    s.temp_trace = np.full(len(s.temp_trace), float(temp))   # traces are shared read-only arrays: replace, never write
    s.delta = 0.0


def test_debounce_escalation_and_resolution(tmp_path):
    live = _live(tmp_path, open_after_min=2, escalate_after_min=3, resolve_after_min=2, incident_min_servers=99)
    sid = live.server_by_ref("PNQ1-A01-01").server_id
    _force_temp(live, sid, live.threshold + 5)
    live.advance()
    assert not [a for a in live.store.alerts("active") if a["server_id"] == sid]       # 1 minute: not yet
    live.advance()
    a = next(a for a in live.store.alerts("active") if a["server_id"] == sid and a["kind"] == "breach")
    assert a["action"] and not a["escalated"]
    for _ in range(4):
        live.advance()
    assert live.store.get_alert(a["alert_id"])["escalated"] == 1
    _force_temp(live, sid, live.threshold - 20)
    for _ in range(4):
        live.advance()
    assert live.store.get_alert(a["alert_id"])["status"] == "resolved"


def test_flicker_does_not_open_alerts(tmp_path):
    live = _live(tmp_path, open_after_min=2, incident_min_servers=99)
    sid = live.server_by_ref("PNQ1-A01-02").server_id
    for k in range(10):                        # over, under, over, under ...
        _force_temp(live, sid, live.threshold + (2 if k % 2 == 0 else -15))
        live.advance()
    assert not [a for a in live.store.alerts("all") if a["server_id"] == sid and a["kind"] == "breach"]


def test_hall_incident_groups_hot_servers(tmp_path):
    live = _live(tmp_path, open_after_min=1, incident_min_servers=3)
    hyd = [i for i, l in live.loc.items() if l["site"] == "HYD2"]
    for i in hyd[:3]:                          # 3 of 16 = 19%: below the 25% share, no incident
        _force_temp(live, i, live.threshold + 4)
    live.advance()
    assert not [a for a in live.store.alerts("active") if a["kind"].startswith("hall_incident")]
    for i in hyd[3:5]:                         # 5 of 16 = 31%: incident
        _force_temp(live, i, live.threshold + 4)
    live.advance()
    inc = [a for a in live.store.alerts("active") if a["kind"].startswith("hall_incident")]
    assert len(inc) == 1 and inc[0]["site"] == "HYD2" and inc[0]["severity"] == "critical"
    assert "cooling" in inc[0]["action"]


def test_webhook_receives_open_and_resolve(tmp_path):
    got = []

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            got.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.end_headers()

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    live = _live(tmp_path, open_after_min=1, resolve_after_min=1, incident_min_servers=99,
                 webhook_url=f"http://127.0.0.1:{srv.server_port}/hook")
    sid = live.server_by_ref("MAA1-A01-01").server_id
    live.set_fan_fault(OP, sid, True)
    live.advance()
    live.set_fan_fault(OP, sid, False)
    for _ in range(3):
        live.advance()
    for _ in range(50):
        if any(g["event"] == "resolve" for g in got):
            break
        time.sleep(0.05)
    srv.shutdown()
    events = [(g["event"], g["alert"]["kind"]) for g in got if g["alert"]["server_id"] == sid]
    assert ("open", "fan_fault") in events and ("resolve", "fan_fault") in events
    assert all("text" in g for g in got)                 # Slack/Teams-compatible field


def test_sensor_loss_goes_stale_and_is_excluded(tmp_path):
    live = _live(tmp_path)
    sid = live.server_by_ref("PNQ1-B01-01").server_id
    live.set_sensor_loss(OP, sid, True)
    for _ in range(5):
        live.advance()
    row = next(r for r in live.server_rows() if r["server_id"] == sid)
    assert row["state"] == "stale"
    c = next(c for c in live.rank(Workload(5, 1, 10)) if c.server_id == sid)
    assert c.status == "ineligible" and "telemetry" in c.reasons[0]
    assert any(a["kind"] == "stale" and a["server_id"] == sid for a in live.store.alerts("active"))
    live.set_sensor_loss(OP, sid, False)
    live.advance()
    assert next(r for r in live.server_rows() if r["server_id"] == sid)["state"] != "stale"


def test_runaway_process_heats_and_whatif_predicts_it(tmp_path):
    live = _live(tmp_path)
    sid = live.server_by_ref("PNQ1-B02-01").server_id
    w = live.whatif("runaway", sid=sid, magnitude=60)
    p = w["servers"][0]
    assert p["rise_c"] > 0 and p["pred"]["30"] > p["pred"]["5"] - 1e-9
    t0 = live.sim.servers[sid].temp
    live.set_runaway(OP, sid, 60)
    for _ in range(30):
        live.advance()
    assert live.sim.servers[sid].temp > t0 + 0.5 * p["rise_c"] - 3     # heats roughly as predicted
    assert next(r for r in live.server_rows() if r["server_id"] == sid)["runaway_pct"] == 60


def test_whatif_cooling_and_resilience(tmp_path):
    live = _live(tmp_path)
    w = live.whatif("cooling", "HYD2", "A", magnitude=8)
    assert w["n_affected"] == 16 and all(abs(r["rise_c"] - 8) < 1e-9 for r in w["servers"])
    assert w["servers"][0]["pred"]["30"] >= w["servers"][-1]["pred"]["30"]
    res = live.resilience(6)
    assert {(r["site"], r["hall"]) for r in res} == {("PNQ1", "A"), ("PNQ1", "B"), ("HYD2", "A"), ("MAA1", "A")}
    with pytest.raises(ValueError):
        live.whatif("meteor")


def test_manual_mode_queue_approve_reject_expire(tmp_path):
    live = _live(tmp_path)
    live.mode = "manual"
    live.queue_expire_min = 3
    live.fc["background_jobs_per_server_per_min"] = 0.05
    live.advance()
    rows = live.queue_rows()
    assert rows and all(r["suggested"] for r in rows)
    first, second = rows[0], rows[1] if len(rows) > 1 else None
    out = live.decide_queued(OP, first["queue_id"], True, None, "")
    assert out["server_id"] == first["suggested"]["server_id"] and not out["override"]
    if second:
        with pytest.raises(ValueError):
            live.decide_queued(OP, second["queue_id"], False, None, "")      # reject needs a reason
        live.decide_queued(OP, second["queue_id"], False, None, "duplicate job")
    live.fc["background_jobs_per_server_per_min"] = 0.0
    for _ in range(4):
        live.advance()
    assert not live.queue                                                   # the rest expired
    acts = {e["action"] for e in live.store.audit_log()}
    assert {"job_held", "place", "job_expired"} <= acts


def test_override_needs_reason_and_is_audited(tmp_path):
    live = _live(tmp_path)
    w = Workload(10, 8, 20, "o")
    ranked = [c for c in live.rank(w) if c.status != "ineligible"]
    other = ranked[3]
    with pytest.raises(ValueError):
        live.place_for(OP, w, other.server_id, None, "")
    out = live.place_for(OP, w, other.server_id, None, "customer asked for this rack")
    assert out["override"]
    e = next(e for e in live.store.audit_log() if e["action"] == "place_override")
    d = json.loads(e["detail"])
    assert d["reason"] == "customer asked for this rack" and d["thermalview_pick"] == ranked[0].name


def test_scheduler_holds_jobs_when_every_server_would_overheat(tmp_path):
    live = _live(tmp_path)
    for i in live.sim.servers:
        _force_temp(live, i, live.threshold + 3)
    live.advance()                       # forecasts now see the hot readings
    start = live.sim.minute
    live.fc["background_jobs_per_server_per_min"] = 0.05
    live.advance()
    live.advance()
    assert live.queue and all("exceed" in j["why"] for j in live.queue.values())
    assert not [j for j in live.store.jobs(100) if j["origin"] == "background" and j["start_minute"] >= start]


def test_external_servers_are_never_placed_on(tmp_path):
    live = _live(tmp_path)
    out = live.add_server(OP, "MAA1", "A", "A01", True, "EXT-1")
    for _ in range(20):
        live.ingest([{"server": "EXT-1", "cpu_temp_c": 30, "load_pct": 1}])
        live.advance()
    c = next(c for c in live.rank(Workload(5, 1, 10)) if c.server_id == out["server_id"])
    assert c.status == "ineligible" and "external" in c.reasons[0]
    with pytest.raises(ActionError):
        live.place_for(OP, Workload(5, 1, 10), out["server_id"], None, "test")
