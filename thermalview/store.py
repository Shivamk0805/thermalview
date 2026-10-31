"""SQLite persistence for the live app.

Tables
  servers          inventory: location (site/hall/rack/slot), which real node it copies, status
  telemetry        one row per server per simulated minute (pruned to a window)
  predictions      5-minute forecasts as issued, so they can be compared later
  jobs             workloads placed by an operator or the background scheduler
  recommendations  every ranking shown, with the workload and the full ranking
  alerts           threshold / forecast / fault / stale-telemetry alerts and their acknowledgement
  audit            who did what, when (wall clock and simulated clock)
  users            operator accounts (password hashes only)
Experiment results are files in results/, not database rows.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS servers (
  server_id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, source_node TEXT NOT NULL,
  site TEXT, hall TEXT, rack TEXT, slot INTEGER, kind TEXT DEFAULT 'simulated',
  status TEXT DEFAULT 'active', alpha REAL, gain REAL, memory_gb REAL,
  added_at TEXT, retired_at TEXT);
CREATE TABLE IF NOT EXISTS telemetry (
  server_id INTEGER NOT NULL, minute INTEGER NOT NULL, ts TEXT NOT NULL,
  temp REAL, load REAL, power REAL, inlet REAL, fan REAL, fan_fault INTEGER, n_jobs INTEGER,
  PRIMARY KEY (server_id, minute));
CREATE TABLE IF NOT EXISTS predictions (
  server_id INTEGER NOT NULL, minute INTEGER NOT NULL, horizon INTEGER NOT NULL, predicted REAL,
  PRIMARY KEY (server_id, minute, horizon));
CREATE TABLE IF NOT EXISTS jobs (
  job_id INTEGER PRIMARY KEY, name TEXT, server_id INTEGER, cpu_pct REAL, mem_gb REAL,
  duration_min INTEGER, start_minute INTEGER, origin TEXT, placed_by TEXT);
CREATE TABLE IF NOT EXISTS recommendations (
  rec_id INTEGER PRIMARY KEY AUTOINCREMENT, minute INTEGER, workload TEXT, ranking TEXT,
  chosen_server INTEGER, requested_by TEXT);
CREATE TABLE IF NOT EXISTS alerts (
  alert_id INTEGER PRIMARY KEY AUTOINCREMENT, server_id INTEGER, server_name TEXT, site TEXT,
  kind TEXT, severity TEXT, text TEXT, status TEXT DEFAULT 'open',
  opened_minute INTEGER, opened_ts TEXT, peak_value REAL,
  acked_by TEXT, acked_ts TEXT, ack_note TEXT, resolved_minute INTEGER, resolved_ts TEXT,
  hall TEXT, action TEXT, escalated INTEGER DEFAULT 0, escalated_ts TEXT);
CREATE TABLE IF NOT EXISTS audit (
  audit_id INTEGER PRIMARY KEY AUTOINCREMENT, wall_ts TEXT, sim_ts TEXT, username TEXT, role TEXT,
  action TEXT, target TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS users (
  username TEXT PRIMARY KEY, display_name TEXT, role TEXT NOT NULL, salt BLOB, pw_hash BLOB);
CREATE INDEX IF NOT EXISTS alerts_open ON alerts(status, server_id, kind);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def hash_password(password: str, salt: bytes | None = None) -> tuple[bytes, bytes]:
    salt = salt or os.urandom(16)
    return salt, hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000)


class Store:
    def __init__(self, path: Path, fresh: bool = True):
        path.parent.mkdir(parents=True, exist_ok=True)
        if fresh and path.exists():
            path.unlink()          # each app start is a new simulation run
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def commit(self) -> None:
        self.db.commit()

    def _rows(self, sql: str, args=()) -> list[dict]:
        return [dict(r) for r in self.db.execute(sql, args).fetchall()]

    # ---- users ----------------------------------------------------------
    def add_user(self, username: str, password: str, role: str, display_name: str = "") -> None:
        if role not in ("viewer", "operator"):
            raise ValueError(f"unknown role {role!r}")
        salt, h = hash_password(password)
        self.db.execute("INSERT OR REPLACE INTO users VALUES (?,?,?,?,?)",
                        (username, display_name or username, role, salt, h))
        self.db.commit()

    def put_user_hash(self, username: str, role: str, display_name: str, salt: bytes, pw_hash: bytes) -> None:
        self.db.execute("INSERT OR REPLACE INTO users VALUES (?,?,?,?,?)", (username, display_name, role, salt, pw_hash))
        self.db.commit()

    def user_row(self, username: str) -> dict | None:
        r = self.db.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        return dict(r) if r else None

    def list_users(self) -> list[dict]:
        return self._rows("SELECT username, display_name, role FROM users ORDER BY role DESC, username")

    def delete_user(self, username: str) -> None:
        self.db.execute("DELETE FROM users WHERE username=?", (username,))
        self.db.commit()

    def check_user(self, username: str, password: str) -> dict | None:
        r = self.db.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        if r is None:
            hash_password(password)              # same cost whether or not the user exists
            return None
        _, h = hash_password(password, r["salt"])
        if not hmac.compare_digest(h, r["pw_hash"]):
            return None
        return {"username": r["username"], "display_name": r["display_name"], "role": r["role"]}

    # ---- inventory ------------------------------------------------------
    def add_server(self, row: dict) -> None:
        row = {"kind": "simulated", "status": "active", "added_at": _now(), "retired_at": None,
               "alpha": None, "gain": None, "memory_gb": None, **row}
        self.db.execute(
            "INSERT OR REPLACE INTO servers (server_id,name,source_node,site,hall,rack,slot,kind,status,alpha,gain,"
            "memory_gb,added_at,retired_at) VALUES (:server_id,:name,:source_node,:site,:hall,:rack,:slot,:kind,"
            ":status,:alpha,:gain,:memory_gb,:added_at,:retired_at)", row)

    def add_servers(self, rows: list[dict]) -> None:
        for r in rows:
            self.add_server(r)
        self.db.commit()

    def set_server_status(self, server_id: int, status: str) -> None:
        self.db.execute("UPDATE servers SET status=?, retired_at=CASE WHEN ?='retired' THEN ? ELSE retired_at END "
                        "WHERE server_id=?", (status, status, _now(), server_id))

    def inventory(self) -> list[dict]:
        return self._rows("SELECT server_id,name,site,hall,rack,slot,kind,status,source_node,memory_gb,added_at,"
                          "retired_at FROM servers ORDER BY site, hall, rack, slot")

    # ---- telemetry ------------------------------------------------------
    def add_telemetry(self, rows: list[dict]) -> None:
        self.db.executemany(
            "INSERT OR REPLACE INTO telemetry VALUES (:server_id,:minute,:ts,:temp,:load,:power,:inlet,:fan,:fan_fault,:n_jobs)",
            [{**r, "ts": str(r["ts"]), "fan_fault": int(r["fan_fault"])} for r in rows])

    def add_predictions(self, minute: int, horizon: int, preds: dict[int, float]) -> None:
        self.db.executemany("INSERT OR REPLACE INTO predictions VALUES (?,?,?,?)",
                            [(sid, minute, horizon, v) for sid, v in preds.items()])

    def prune(self, keep_from_minute: int) -> None:
        self.db.execute("DELETE FROM telemetry WHERE minute < ?", (keep_from_minute,))
        self.db.execute("DELETE FROM predictions WHERE minute < ?", (keep_from_minute,))

    def history(self, server_id: int, since_minute: int) -> list[dict]:
        return self._rows(
            "SELECT t.minute, t.ts, t.temp, t.load, t.power, t.inlet, t.fan, t.fan_fault, p.predicted AS pred5_issued "
            "FROM telemetry t LEFT JOIN predictions p ON p.server_id=t.server_id AND p.minute=t.minute-5 AND p.horizon=5 "
            "WHERE t.server_id=? AND t.minute>=? ORDER BY t.minute", (server_id, since_minute))

    def telemetry_export(self, since_minute: int, server_id: int | None = None) -> list[dict]:
        sql = ("SELECT s.name AS server, s.site, s.hall, s.rack, t.ts, t.minute, t.temp AS cpu_temp_c, "
               "t.load AS load_pct, t.power AS power_w, t.inlet AS inlet_c, t.fan AS fan_rpm, t.fan_fault, t.n_jobs, "
               "p.predicted AS forecast_5min_issued_5min_earlier FROM telemetry t JOIN servers s USING(server_id) "
               "LEFT JOIN predictions p ON p.server_id=t.server_id AND p.minute=t.minute-5 AND p.horizon=5 "
               "WHERE t.minute>=?")
        args: list = [since_minute]
        if server_id is not None:
            sql += " AND t.server_id=?"
            args.append(server_id)
        return self._rows(sql + " ORDER BY t.minute, s.name", args)

    def forecast_error(self, since_minute: int) -> dict:
        """Live check: MAE of 5-minute forecasts issued in the app vs what then happened."""
        r = self.db.execute(
            "SELECT COUNT(*) n, AVG(ABS(p.predicted - t.temp)) mae, AVG(ABS(t0.temp - t.temp)) persist "
            "FROM predictions p JOIN telemetry t ON t.server_id=p.server_id AND t.minute=p.minute+5 "
            "JOIN telemetry t0 ON t0.server_id=p.server_id AND t0.minute=p.minute "
            "WHERE p.horizon=5 AND p.minute>=?", (since_minute,)).fetchone()
        return dict(r)

    # ---- jobs and recommendations --------------------------------------
    def add_job(self, job, name: str, origin: str, placed_by: str = "") -> None:
        self.db.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?)",
                        (job.job_id, name, job.server_id, job.cpu_pct, job.mem_gb, job.duration_min,
                         job.start_min, origin, placed_by))

    def jobs(self, limit: int = 40, origin: str | None = None) -> list[dict]:
        where, args = ("WHERE j.origin=?", [origin]) if origin else ("", [])
        return self._rows("SELECT j.*, s.name AS server_name, s.site FROM jobs j LEFT JOIN servers s USING(server_id) "
                          f"{where} ORDER BY job_id DESC LIMIT ?", (*args, limit))

    def add_recommendation(self, minute: int, workload: dict, ranking: list[dict], chosen: int | None,
                           requested_by: str = "") -> int:
        cur = self.db.execute("INSERT INTO recommendations (minute, workload, ranking, chosen_server, requested_by) "
                              "VALUES (?,?,?,?,?)", (minute, json.dumps(workload), json.dumps(ranking), chosen,
                                                     requested_by))
        return int(cur.lastrowid)

    def mark_chosen(self, rec_id: int, server_id: int) -> None:
        self.db.execute("UPDATE recommendations SET chosen_server=? WHERE rec_id=?", (server_id, rec_id))

    # ---- alerts ---------------------------------------------------------
    def open_alerts(self) -> dict[tuple[int, str], dict]:
        rows = self._rows("SELECT * FROM alerts WHERE status IN ('open','acknowledged')")
        return {(r["server_id"], r["kind"]): r for r in rows}

    def open_alert(self, a: dict) -> int:
        cur = self.db.execute(
            "INSERT INTO alerts (server_id,server_name,site,hall,kind,severity,text,action,status,opened_minute,opened_ts,"
            "peak_value) VALUES (:server_id,:server_name,:site,:hall,:kind,:severity,:text,:action,'open',:minute,:ts,:value)",
            {"hall": None, "action": "", **a})
        return int(cur.lastrowid)

    def update_alert_peak(self, alert_id: int, value: float, text: str) -> None:
        self.db.execute("UPDATE alerts SET peak_value=MAX(COALESCE(peak_value,?),?), text=? WHERE alert_id=?",
                        (value, value, text, alert_id))

    def escalate_alert(self, alert_id: int, ts: str) -> None:
        self.db.execute("UPDATE alerts SET escalated=1, escalated_ts=? WHERE alert_id=?", (ts, alert_id))

    def get_alert(self, alert_id: int) -> dict | None:
        r = self.db.execute("SELECT * FROM alerts WHERE alert_id=?", (alert_id,)).fetchone()
        return dict(r) if r else None

    def resolve_alert(self, alert_id: int, minute: int, ts: str) -> None:
        self.db.execute("UPDATE alerts SET status='resolved', resolved_minute=?, resolved_ts=? WHERE alert_id=?",
                        (minute, ts, alert_id))

    def ack_alert(self, alert_id: int, user: str, note: str) -> dict | None:
        r = self.db.execute("SELECT * FROM alerts WHERE alert_id=?", (alert_id,)).fetchone()
        if r is None:
            return None
        if r["status"] == "open":
            self.db.execute("UPDATE alerts SET status='acknowledged', acked_by=?, acked_ts=?, ack_note=? "
                            "WHERE alert_id=?", (user, _now(), note[:200], alert_id))
        return dict(self.db.execute("SELECT * FROM alerts WHERE alert_id=?", (alert_id,)).fetchone())

    def alerts(self, status: str | None = None, limit: int = 200) -> list[dict]:
        if status == "active":
            return self._rows("SELECT * FROM alerts WHERE status IN ('open','acknowledged') "
                              "ORDER BY CASE severity WHEN 'critical' THEN 0 ELSE 1 END, alert_id DESC LIMIT ?", (limit,))
        if status and status != "all":
            return self._rows("SELECT * FROM alerts WHERE status=? ORDER BY alert_id DESC LIMIT ?", (status, limit))
        return self._rows("SELECT * FROM alerts ORDER BY alert_id DESC LIMIT ?", (limit,))

    # ---- audit ----------------------------------------------------------
    def audit(self, user: str, role: str, action: str, target: str = "", detail: dict | None = None,
              sim_ts: str = "") -> None:
        self.db.execute("INSERT INTO audit (wall_ts,sim_ts,username,role,action,target,detail) VALUES (?,?,?,?,?,?,?)",
                        (_now(), sim_ts, user, role, action, target, json.dumps(detail or {}, default=str)))

    def audit_log(self, limit: int = 300, user: str | None = None, action: str | None = None,
                  include_scheduler: bool = True) -> list[dict]:
        sql, args = "SELECT * FROM audit WHERE 1=1", []
        if user:
            sql, args = sql + " AND username=?", args + [user]
        if action:
            sql, args = sql + " AND action=?", args + [action]
        if not include_scheduler:
            sql += " AND username<>'scheduler'"
        return self._rows(sql + " ORDER BY audit_id DESC LIMIT ?", (*args, limit))
