"""The live fleet behind the operator console.

A multi-site fleet (sites -> halls -> racks -> servers) runs in the
closed-loop simulator. Every simulated server copies one real M100 node:
its fitted heat model, and that node's recorded load and inlet traces
replayed from the unseen test period. Servers registered as "external"
take no simulated physics at all: their readings arrive through the
telemetry API (scripts/telemetry_agent.py, or a real monitoring agent).

The forecaster, recommender and eligibility rules are the same objects the
offline evaluation uses. This module adds inventory, maintenance, alerts
and the audit trail.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import queue
import threading
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from thermalview import simulator as S
from thermalview import thermal_model as TM
from thermalview.config import Config
from thermalview.data.prepare import load_table
from thermalview.recommender import MLForecaster, Workload, recommend
from thermalview.schema import VALID_RANGES
from thermalview.store import Store

log = logging.getLogger("thermalview")
HORIZONS = (5, 15, 30)
SYSTEM = ("scheduler", "system")


ACTIONS = {
    "breach": "Avoid new work here; ThermalView already ranks it last. If it stays hot for more than "
              "10 minutes, drain it and check its fans and airflow.",
    "forecast": "Expected to cross the limit soon. Avoid new work here; no action is needed if the alert "
                "clears by itself.",
    "fan_fault": "Drain the server so no new work lands on it, and schedule a fan replacement. Running jobs "
                 "keep running.",
    "stale": "Check the server's monitoring agent and network. It is excluded from placement until readings "
             "resume.",
    "hall_incident": "Several servers in this hall are hot at the same time, which usually means a cooling "
                     "problem rather than a server problem. Contact facilities; consider draining the hottest "
                     "servers in the hall.",
}


class Notifier:
    """Posts alert events to a webhook (Slack/Teams-compatible "text" field plus
    structured fields) on a background thread, so a slow endpoint never stalls
    the console. Disabled when no URL is configured."""

    def __init__(self, url: str, events: list[str], org: str):
        self.url, self.events, self.org = url, set(events), org
        self.q: queue.Queue = queue.Queue(maxsize=500)
        self.sent = self.failed = 0
        if url:
            threading.Thread(target=self._run, daemon=True).start()

    def notify(self, event: str, alert: dict) -> None:
        if not self.url or event not in self.events:
            return
        icon = {"open": "🔴" if alert.get("severity") == "critical" else "🟡", "escalate": "⏫", "resolve": "✅"}[event]
        text = f"{icon} [{self.org}] {event.upper()}: {alert.get('text', '')}"
        if event != "resolve" and alert.get("action"):
            text += f" — {alert['action']}"
        try:
            self.q.put_nowait({"text": text, "event": event, "alert": alert})
        except queue.Full:
            self.failed += 1

    def _run(self) -> None:
        while True:
            body = self.q.get()
            try:
                req = urllib.request.Request(self.url, data=json.dumps(body, default=str).encode(),
                                             headers={"Content-Type": "application/json"}, method="POST")
                urllib.request.urlopen(req, timeout=10).close()
                self.sent += 1
            except Exception as exc:  # never let notifications break the console
                self.failed += 1
                log.warning("webhook failed: %s", exc)


class SetupError(RuntimeError):
    pass


class ActionError(ValueError):
    """A request that is understood but not allowed in the current state (HTTP 409)."""


class LiveApp:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        art, res = cfg.path("artifacts"), cfg.path("results")
        needed = [art / "forecaster.joblib", art / "heat_params.json", res / "forecast_summary.json"]
        missing = [str(p) for p in needed if not p.exists()]
        if missing:
            raise SetupError("Missing " + ", ".join(missing) +
                             ". Run `python -m thermalview all` (or `all --dev --quick` for development data) first.")
        self.forecaster = MLForecaster(art / "forecaster.joblib")
        self.params = TM.load_params(art / "heat_params.json")
        summ = json.loads((res / "forecast_summary.json").read_text())
        split = summ["splits"][str(summ["config"]["primary_horizon"])]
        table, meta = load_table(cfg.path("processed"), "telemetry")
        self.data_source = meta.get("source", "unknown")
        # The research threshold (a percentile of training temperatures) is the default limit.
        # An operator would normally set their own: [alerts].limit_c (e.g. the vendor's
        # throttling point minus a margin). It drives alerts, states and recommendations alike.
        lim = cfg.get("alerts", {}).get("limit_c", "model")
        self.threshold = self.forecaster.threshold if lim in ("model", "", None) else float(lim)
        self.limit_source = "research threshold" if lim in ("model", "", None) else "operator setting"
        self.horizon = self.forecaster.horizon
        self.fc = cfg["fleet"]
        self.stale_after = int(cfg["recommender"]["stale_after_min"])

        # Node pool: fitted nodes, ones the forecaster never trained on first.
        self.node_pool = S.pick_nodes(self.params, len(self.params), split["holdout_nodes"])
        if not self.node_pool:
            raise SetupError("no fitted heat models; run `python -m thermalview thermal`")
        trace_start = pd.Timestamp(split["t_val_end"])
        sim_cfg = {"memory_gb": self.fc["memory_gb"], "background_memory_frac": self.fc["background_memory_frac"]}
        self.templates = {s.params.node_id: s
                          for s in S.build_servers(table, self.params, self.node_pool, trace_start, sim_cfg)}
        del table
        fitted = [p for p in self.params.values() if p.fitted]
        med = {f.name: float(np.median([getattr(p, f.name) for p in fitted]))
               for f in dataclasses.fields(TM.NodeParams) if f.type in ("float", float)}
        self.median_params = dataclasses.replace(fitted[0], **med, node_id="fleet-median", fitted=False, n_rows=0)

        self.sites = {s["code"]: s for s in self.fc["sites"]}
        self.cooling: dict[tuple[str, str], float] = {}     # (site, hall) -> extra inlet °C
        self.loc: dict[int, dict] = {}
        self.sim = S.FleetSim([], cpu_cap=float(self.fc["cpu_cap_pct"]),
                              bg_scale=float(self.fc["background_load_scale"]), noise_scale=float(self.fc["noise_scale"]),
                              seed=1, history=int(cfg["app"]["history_minutes"]),
                              start_ts=pd.Timestamp.now(tz="UTC").floor("min"))
        self.store = Store(cfg.path("database"))
        self.config_users = set()
        for u in cfg.get("auth", {}).get("users", []):
            self.store.add_user(u["username"], u["password"], u["role"], u.get("name", ""))
            self.config_users.add(u["username"])
        uf = Path(cfg.get("auth", {}).get("users_file", "data/users.json"))
        self.users_file = uf if uf.is_absolute() else cfg.root / uf
        self._load_users_file()
        self._next_id = 1
        self._node_i = 0
        for site in self.fc["sites"]:
            for hall in site["halls"]:
                for r in range(1, int(hall["racks"]) + 1):
                    for slot in range(1, int(hall["servers_per_rack"]) + 1):
                        self._create(site["code"], hall["name"], f"{hall['name']}{r:02d}", slot, external=False)
        self.store.commit()

        self.running, self.speed, self.auto_jobs = True, 1, True
        self.mode = str(self.fc.get("placement_mode", "auto"))   # auto | manual
        if self.mode not in ("auto", "manual"):
            raise SetupError("[fleet].placement_mode must be auto or manual")
        self.queue: dict[int, dict] = {}            # jobs waiting for an operator decision
        self._next_q = 1
        self.queue_expire_min = int(self.fc.get("queue_expire_min", 15))
        self.rng = np.random.default_rng(7)
        self.next_job_id = 1
        self.forecasts: dict[int, dict[int, float]] = {}
        self._clear_since: dict[int, int] = {}
        self._pending: dict[tuple, int] = {}       # condition -> consecutive minutes seen (debounce)
        ac = cfg.get("alerts", {})
        self.alert_cfg = {"open_after_min": int(ac.get("open_after_min", 2)),
                          "resolve_after_min": int(ac.get("resolve_after_min", cfg["app"].get("alert_resolve_after_min", 5))),
                          "escalate_after_min": int(ac.get("escalate_after_min", 10)),
                          "incident_min_servers": int(ac.get("incident_min_servers", 3)),
                          "incident_min_fraction": float(ac.get("incident_min_fraction", 0.25))}
        self.notifier = Notifier(str(ac.get("webhook_url", "") or ""), list(ac.get("webhook_events", ["open", "escalate", "resolve"])),
                                 self.fc.get("org_name", "ThermalView"))
        self.audit("system", "system", "start", "", {"servers": len(self.sim.servers), "data_source": self.data_source,
                                                      "model": self.forecaster.name, "threshold_c": self.threshold})
        for _ in range(S.WINDOW_READY + 5):     # history so forecasts exist immediately
            self.advance()

    # ---- inventory --------------------------------------------------------
    def _bias(self, sid: int) -> float:
        loc = self.loc[sid]
        return (float(self.sites[loc["site"]].get("inlet_bias_c", 0.0)) +
                self.cooling.get((loc["site"], loc["hall"]), 0.0) + 0.1 * (loc["slot"] - 1))

    def _create(self, site: str, hall: str, rack: str, slot: int, external: bool, name: str | None = None) -> S.SimServer:
        sid = self._next_id
        self._next_id += 1
        name = name or f"{site}-{rack}-{slot:02d}"
        traces = {}
        if external:
            params, node = self.median_params, "external"
            bg, inlet = np.zeros(60), np.full(60, self.median_params.inlet_mean)
        else:
            node = self.node_pool[self._node_i % len(self.node_pool)]
            self._node_i += 1
            params = self.params[node]
            t = self.templates[node]
            bg, inlet = t.bg_load, t.inlet_trace
            traces = {"temp_trace": t.temp_trace, "power_trace": t.power_trace, "fan_trace": t.fan_trace}
        self.loc[sid] = {"site": site, "hall": hall, "rack": rack, "slot": slot}
        s = S.SimServer(server_id=sid, name=name, params=params, bg_load=bg, inlet_trace=inlet,
                        mem_gb=float(self.fc["memory_gb"]), bg_mem_frac=float(self.fc["background_memory_frac"]),
                        trace_offset=(sid * 137) % len(bg), external=external, **traces)
        s.inlet_bias = self._bias(sid)
        self.sim.add_server(s)
        self.store.add_server({"server_id": sid, "name": name, "source_node": node, "site": site, "hall": hall,
                               "rack": rack, "slot": slot, "kind": "external" if external else "simulated",
                               "alpha": params.alpha, "gain": params.gain, "memory_gb": s.mem_gb})
        return s

    def racks(self) -> dict[tuple[str, str, str], list[int]]:
        out: dict[tuple[str, str, str], list[int]] = {}
        for sid, l in self.loc.items():
            if sid in self.sim.servers:
                out.setdefault((l["site"], l["hall"], l["rack"]), []).append(sid)
        return out

    # ---- user accounts (operators manage them; kept across restarts in users_file) -----------------
    def _load_users_file(self) -> None:
        """Accounts created or changed in the console: password hashes only, never passwords.
        They are added on top of config.toml, and a password changed here overrides config.toml."""
        import base64
        self.console_users: set[str] = set()
        if not self.users_file.exists():
            return
        for u in json.loads(self.users_file.read_text(encoding="utf-8")).get("users", []):
            self.store.put_user_hash(u["username"], u["role"], u.get("display_name", u["username"]),
                                     base64.b64decode(u["salt"]), base64.b64decode(u["pw_hash"]))
            self.console_users.add(u["username"])

    def _save_users_file(self) -> None:
        import base64
        rows = []
        for u in self.store.list_users():
            if u["username"] in self.config_users and u["username"] not in self.console_users:
                continue
            r = self.store.user_row(u["username"])
            rows.append({"username": r["username"], "display_name": r["display_name"], "role": r["role"],
                         "salt": base64.b64encode(r["salt"]).decode(), "pw_hash": base64.b64encode(r["pw_hash"]).decode()})
        self.users_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.users_file.with_suffix(".tmp")
        tmp.write_text(json.dumps({"_note": "Managed by the ThermalView console (Users page). Password hashes only "
                                            "(PBKDF2-SHA256); delete this file to go back to config.toml accounts.",
                                   "users": rows}, indent=2), encoding="utf-8")
        tmp.replace(self.users_file)

    def list_users(self) -> list[dict]:
        return [{**u, "source": ("config.toml, password changed here" if u["username"] in self.console_users else
                                 "config.toml") if u["username"] in self.config_users else "added in console"}
                for u in self.store.list_users()]

    def add_user(self, actor: dict, username: str, password: str, role: str, display_name: str = "") -> dict:
        import re as _re
        username = (username or "").strip().lower()
        if not _re.fullmatch(r"[a-z0-9._-]{3,32}", username):
            raise ValueError("username: 3-32 characters, lowercase letters, digits, dot, dash or underscore")
        if role not in ("viewer", "operator"):
            raise ValueError("role must be viewer or operator")
        if len(password or "") < 8:
            raise ValueError("password must have at least 8 characters")
        if self.store.user_row(username):
            raise ActionError(f"user {username} already exists")
        self.store.add_user(username, password, role, (display_name or username).strip()[:64])
        self.console_users.add(username)
        self._save_users_file()
        self.audit(actor["username"], actor["role"], "user_add", username, {"role": role})
        self.store.commit()
        return {"username": username, "role": role}

    def set_password(self, actor: dict, username: str, password: str) -> dict:
        r = self.store.user_row(username)
        if r is None:
            raise KeyError(f"no user {username}")
        if len(password or "") < 8:
            raise ValueError("password must have at least 8 characters")
        self.store.add_user(username, password, r["role"], r["display_name"])
        self.console_users.add(username)
        self._save_users_file()
        self.audit(actor["username"], actor["role"], "user_password", username, {})
        self.store.commit()
        return {"username": username}

    def delete_user(self, actor: dict, username: str) -> dict:
        r = self.store.user_row(username)
        if r is None:
            raise KeyError(f"no user {username}")
        if username == actor["username"]:
            raise ActionError("you cannot remove your own account")
        if username in self.config_users:
            raise ActionError(f"{username} is defined in config.toml; remove it there and restart")
        if r["role"] == "operator" and sum(u["role"] == "operator" for u in self.store.list_users()) <= 1:
            raise ActionError("at least one operator must remain")
        self.store.delete_user(username)
        self.console_users.discard(username)
        self._save_users_file()
        self.audit(actor["username"], actor["role"], "user_remove", username, {"role": r["role"]})
        self.store.commit()
        return {"removed": username}

    def add_server(self, user: dict, site: str, hall: str, rack: str, external: bool, name: str | None) -> dict:
        if site not in self.sites:
            raise ValueError(f"unknown site {site!r}")
        if not any(h["name"] == hall for h in self.sites[site]["halls"]):
            raise ValueError(f"site {site} has no hall {hall!r}")
        if not rack or not rack.startswith(hall) or len(rack) > 8:
            raise ValueError(f"rack must start with the hall letter, e.g. {hall}01")
        used = [self.loc[i]["slot"] for i in self.racks().get((site, hall, rack), [])]
        if len(used) >= 42:
            raise ActionError(f"rack {rack} is full")
        if name:
            name = name.strip()[:32]
            if any(s.name == name for s in self.sim.servers.values()):
                raise ActionError(f"a server called {name} already exists")
        slot = next(i for i in range(1, 43) if i not in used)
        s = self._create(site, hall, rack, slot, external, name)
        self.audit(user["username"], user["role"], "server_add", s.name,
                   {"site": site, "hall": hall, "rack": rack, "slot": slot, "kind": "external" if external else "simulated",
                    "copies_node": s.params.node_id})
        self.store.commit()
        return {"server_id": s.server_id, "name": s.name, "kind": "external" if external else "simulated"}

    def retire_server(self, user: dict, sid: int, force: bool = False) -> dict:
        s = self._server(sid)
        if not s.maintenance and not force:
            raise ActionError(f"drain {s.name} before retiring it")
        if s.jobs and not force:
            raise ActionError(f"{s.name} still runs {len(s.jobs)} job(s); wait for them to finish")
        self.sim.remove_server(sid)
        self.store.set_server_status(sid, "retired")
        for (alert_sid, _kind), a in self.store.open_alerts().items():
            if alert_sid == sid:
                self.store.resolve_alert(a["alert_id"], self.sim.minute, str(self.sim.now()))
        self.audit(user["username"], user["role"], "server_retire", s.name, {"forced": force, "jobs_dropped": len(s.jobs)})
        self.store.commit()
        return {"server_id": sid, "name": s.name, "status": "retired"}

    def set_maintenance(self, user: dict, sid: int, on: bool, reason: str = "") -> dict:
        s = self._server(sid)
        s.maintenance = bool(on)
        self.store.set_server_status(sid, "maintenance" if on else "active")
        self.audit(user["username"], user["role"], "drain" if on else "undrain", s.name,
                   {"reason": reason[:200], "jobs_still_running": len(s.jobs)})
        self.store.commit()
        return {"server_id": sid, "name": s.name, "maintenance": s.maintenance}

    def set_fan_fault(self, user: dict, sid: int, on: bool) -> dict:
        s = self._server(sid)
        if s.external:
            raise ActionError(f"{s.name} reports real telemetry; faults can only be simulated on simulated servers")
        s.fan_fault = bool(on)
        self.audit(user["username"], user["role"], "fault_fan_inject" if on else "fault_fan_clear", s.name, {})
        self.store.commit()
        return {"server_id": sid, "name": s.name, "fan_fault": s.fan_fault}

    def set_cooling(self, user: dict, site: str, hall: str, offset: float) -> dict:
        if site not in self.sites or not any(h["name"] == hall for h in self.sites[site]["halls"]):
            raise ValueError(f"unknown hall {site}/{hall}")
        offset = float(np.clip(offset, 0.0, 12.0))
        self.cooling[(site, hall)] = offset
        for sid, l in self.loc.items():
            if sid in self.sim.servers and l["site"] == site and l["hall"] == hall:
                self.sim.servers[sid].inlet_bias = self._bias(sid)
        self.audit(user["username"], user["role"], "fault_cooling" if offset > 0 else "fault_cooling_clear",
                   f"{site} hall {hall}", {"inlet_rise_c": offset})
        self.store.commit()
        return {"site": site, "hall": hall, "inlet_rise_c": offset}

    def set_sensor_loss(self, user: dict, sid: int, on: bool) -> dict:
        s = self._server(sid)
        if s.external:
            raise ActionError(f"{s.name} reports through an agent; stop the agent to simulate lost telemetry")
        s.sensor_lost = bool(on)
        self.audit(user["username"], user["role"], "fault_sensor_inject" if on else "fault_sensor_clear", s.name, {})
        self.store.commit()
        return {"server_id": sid, "name": s.name, "sensor_lost": s.sensor_lost}

    def set_runaway(self, user: dict, sid: int, cpu_pct: float) -> dict:
        s = self._server(sid)
        if s.external:
            raise ActionError(f"{s.name} reports real telemetry; faults can only be simulated on simulated servers")
        s.runaway_pct = float(np.clip(cpu_pct, 0.0, 100.0))
        self.audit(user["username"], user["role"], "fault_runaway_inject" if s.runaway_pct else "fault_runaway_clear",
                   s.name, {"cpu_pct": s.runaway_pct})
        self.store.commit()
        return {"server_id": sid, "name": s.name, "runaway_pct": s.runaway_pct}

    # ---- what-if: predict a failure's effect without applying it ------------
    def whatif(self, kind: str, site: str | None = None, hall: str | None = None, sid: int | None = None,
               magnitude: float | None = None) -> dict:
        """Predicted temperature of every affected server 5/15/30 minutes after a failure,
        if it happened now: the current forecast plus the heat-model response to the change.
        Nothing is applied."""
        if kind == "cooling":
            if site not in self.sites or not any(h["name"] == hall for h in self.sites[site]["halls"]):
                raise ValueError(f"unknown hall {site}/{hall}")
            rise = float(np.clip(magnitude if magnitude is not None else 6.0, 0.0, 12.0))
            ids = [i for i, l in self.loc.items() if i in self.sim.servers and l["site"] == site and l["hall"] == hall
                   and not self.sim.servers[i].external]
            label = f"Cooling in {site} hall {hall} degrades: inlet air +{rise:g} °C"
        elif kind in ("fan", "runaway"):
            s = self._server(int(sid))
            if s.external:
                raise ActionError(f"{s.name} reports real telemetry; faults can only be simulated on simulated servers")
            ids = [s.server_id]
            rise = float(np.clip(magnitude if magnitude is not None else 60.0, 0.0, 100.0))
            label = f"Fan failure on {s.name}" if kind == "fan" else f"Stuck process on {s.name} using {rise:g}% CPU"
        else:
            raise ValueError("kind must be cooling, fan or runaway")
        out = []
        for i in ids:
            s, p = self.sim.servers[i], self.sim.servers[i].params
            now = float(s.last.get("temp", s.temp))
            if kind == "cooling":
                target, alpha = p.inlet_k * rise, p.alpha
            elif kind == "fan":
                target, alpha = S.FAN_FAULT_RISE_C, p.alpha * S.FAN_FAULT_ALPHA
            else:
                extra = max(0.0, min(100.0, self.sim.load(s) + rise) - self.sim.load(s))
                target, alpha = p.gain * extra, p.alpha
            pred = {}
            for h in (5, 15, 30):
                base = self.forecasts.get(h, {}).get(i, now)
                pred[h] = round(base + target * (1 - (1 - alpha) ** h), 2)
            first = next((h for h in (5, 15, 30) if pred[h] >= self.threshold), None)
            out.append({"server_id": i, "name": s.name, "now": round(now, 2), "pred": {str(h): v for h, v in pred.items()},
                        "rise_c": round(target, 2), "crosses_within_min": first,
                        "already_over": now >= self.threshold})
        out.sort(key=lambda r: -r["pred"]["30"])
        n_cross = sum(r["crosses_within_min"] is not None for r in out)
        return {"kind": kind, "label": label, "threshold": self.threshold, "servers": out,
                "n_affected": len(out), "n_cross": n_cross,
                "summary": (f"{label}. {n_cross} of {len(out)} server{'s' if len(out) != 1 else ''} would be over the "
                            f"{self.threshold:.1f} °C limit within 30 minutes." if out else f"{label}. No servers affected.")}

    def resilience(self, rise: float = 6.0) -> list[dict]:
        """For every hall: how many servers would cross the limit within 30 minutes if its cooling
        degraded by `rise` °C now. A planning view for 'what if we lose a cooling unit?'."""
        rows = []
        for code, site in self.sites.items():
            for h in site["halls"]:
                w = self.whatif("cooling", code, h["name"], magnitude=rise)
                rows.append({"site": code, "hall": h["name"], "servers": w["n_affected"], "would_cross": w["n_cross"],
                             "worst": w["servers"][0]["name"] if w["servers"] else None,
                             "worst_temp_30": w["servers"][0]["pred"]["30"] if w["servers"] else None})
        return rows

    def _server(self, sid: int) -> S.SimServer:
        if sid not in self.sim.servers:
            raise KeyError(f"no active server {sid}")
        return self.sim.servers[sid]

    def server_by_ref(self, ref) -> S.SimServer:
        if isinstance(ref, int) or (isinstance(ref, str) and ref.isdigit()):
            return self._server(int(ref))
        for s in self.sim.servers.values():
            if s.name == ref:
                return s
        raise KeyError(f"no active server {ref!r}")

    # ---- telemetry ingest ---------------------------------------------------
    def ingest(self, readings: list) -> dict:
        accepted, rejected = 0, []
        for i, r in enumerate(readings):
            try:
                if not isinstance(r, dict):
                    raise ValueError("each reading must be an object")
                s = self.server_by_ref(r.get("server_id", r.get("server")))
                if not s.external:
                    raise ValueError(f"{s.name} is a simulated server; register an external server in Inventory "
                                     "to send real telemetry")
                vals = {}
                for col, key in (("cpu_temp_c", "temp"), ("load_pct", "load"), ("power_w", "power"),
                                 ("inlet_c", "inlet"), ("fan_rpm", "fan")):
                    v = r.get(col)
                    if v is None:
                        if col in ("cpu_temp_c", "load_pct"):
                            raise ValueError(f"{col} is required")
                        continue
                    v = float(v)
                    lo, hi = VALID_RANGES[col]
                    if not (lo <= v <= hi):
                        raise ValueError(f"{col}={v} is outside {lo}-{hi}")
                    vals[key] = v
                p = s.params
                vals.setdefault("power", p.power_0 + p.power_1 * vals["load"])
                vals.setdefault("inlet", p.inlet_mean)
                vals.setdefault("fan", p.fan_0 + p.fan_1 * max(0.0, vals["temp"] - 50.0))
                row = self.sim.ingest(s.server_id, vals)
                self.store.add_telemetry([row])
                accepted += 1
            except (KeyError, ValueError, TypeError) as exc:
                rejected.append({"index": i, "error": str(exc.args[0]) if exc.args else str(exc)})
        self.store.commit()
        return {"accepted": accepted, "rejected": rejected, "sim_time": str(self.sim.now())}

    # ---- simulation loop ----------------------------------------------------
    def advance(self) -> None:
        rows = self.sim.step()
        self.store.add_telemetry(rows)
        if self.auto_jobs and self.sim.minute > S.WINDOW_READY:
            self._background_jobs()
        bufs = self.sim.buffers()
        minute = self.sim.minute - 1
        self.forecasts = {h: self.forecaster.predict(bufs, h) for h in HORIZONS}
        self.store.add_predictions(minute, 5, self.forecasts[5])
        self._alerts()
        if minute % 60 == 0:
            self.store.prune(minute - int(self.cfg["app"]["history_minutes"]))
        self.store.commit()

    def _background_jobs(self) -> None:
        """Jobs arriving from users' batch systems. In auto mode the scheduler places each on
        ThermalView's top pick; if every eligible server would exceed the limit the job is held
        for an operator instead, and if no server can take it at all it is rejected. In manual
        mode every job waits for an operator decision. Every outcome is audited."""
        wl = self.cfg["workload"]
        rate = float(self.fc["background_jobs_per_server_per_min"]) * len(self.sim.servers)
        for _ in range(self.rng.poisson(rate)):
            self._arrivals = getattr(self, "_arrivals", 0) + 1
            w = Workload(cpu_pct=float(self.rng.uniform(wl["cpu_min"], wl["cpu_max"] * 0.7)),
                         mem_gb=float(self.rng.uniform(wl["mem_min_gb"], wl["mem_max_gb"] / 2)),
                         duration_min=max(5, int(self.rng.lognormal(np.log(wl["duration_median_min"]), wl["duration_sigma"]))),
                         name=f"batch-{self._arrivals}")
            top = self._top(w)
            if self.mode == "manual":
                self._enqueue(w, "manual mode: waiting for an operator")
            elif top is None:
                self.audit("scheduler", "system", "job_rejected", w.name,
                           {"cpu_pct": round(w.cpu_pct, 1), "mem_gb": round(w.mem_gb, 1), "reason": "no server can take it"})
            elif top.status == "not_recommended":
                self._enqueue(w, "every eligible server would exceed the limit")
            else:
                job = self._place(w, top.server_id, "background", "scheduler")
                self.audit("scheduler", "system", "place", top.name,
                           {"job_id": job.job_id, "cpu_pct": round(w.cpu_pct, 1), "mem_gb": round(w.mem_gb, 1),
                            "predicted_temp": round(top.predicted_temp, 2), "status": top.status, "rank": 1})
        for qid in [q for q, j in self.queue.items() if self.sim.minute - j["arrived_minute"] >= self.queue_expire_min]:
            j = self.queue.pop(qid)
            self.audit("system", "system", "job_expired", j["workload"].name,
                       {"queue_id": qid, "waited_min": self.sim.minute - j["arrived_minute"], "why_held": j["why"]})

    def _top(self, w: Workload, site: str | None = None):
        ranked = self.rank(w, site)
        return next((c for c in ranked if c.status != "ineligible" and not self.sim.servers[c.server_id].external), None)

    def _enqueue(self, w: Workload, why: str) -> None:
        qid = self._next_q
        self._next_q += 1
        self.queue[qid] = {"queue_id": qid, "workload": w, "arrived_minute": self.sim.minute,
                           "arrived_ts": str(self.sim.now()), "why": why}
        self.audit("scheduler", "system", "job_held", w.name, {"queue_id": qid, "why": why,
                                                                "cpu_pct": round(w.cpu_pct, 1), "mem_gb": round(w.mem_gb, 1)})

    def queue_rows(self) -> list[dict]:
        out = []
        for qid, j in sorted(self.queue.items()):
            w = j["workload"]
            top = self._top(w)
            out.append({"queue_id": qid, "name": w.name, "cpu_pct": round(w.cpu_pct, 1), "mem_gb": round(w.mem_gb, 1),
                        "duration_min": w.duration_min, "arrived_ts": j["arrived_ts"], "why": j["why"],
                        "waiting_min": self.sim.minute - j["arrived_minute"],
                        "expires_in_min": max(0, self.queue_expire_min - (self.sim.minute - j["arrived_minute"])),
                        "suggested": None if top is None else {"server_id": top.server_id, "name": top.name,
                                                               "status": top.status,
                                                               "predicted_temp": round(top.predicted_temp, 2),
                                                               "explanation": top.explanation}})
        return out

    def decide_queued(self, user: dict, qid: int, approve: bool, sid: int | None, reason: str) -> dict:
        if qid not in self.queue:
            raise KeyError(f"no queued job {qid} (it may have been decided or expired)")
        w = self.queue[qid]["workload"]
        if not approve:
            if not reason.strip():
                raise ValueError("give a reason for rejecting the job")
            self.queue.pop(qid)
            self.audit(user["username"], user["role"], "job_rejected", w.name, {"queue_id": qid, "reason": reason[:200]})
            self.store.commit()
            return {"queue_id": qid, "rejected": True}
        top = self._top(w)
        if sid is None:
            if top is None:
                raise ActionError("no server can take this job right now; reject it or wait")
            sid = top.server_id
        out = self.place_for(user, w, int(sid), None, reason, origin="approved")
        self.queue.pop(qid)
        return {**out, "queue_id": qid}

    def _alerts(self) -> None:
        """Turn conditions into alerts.
        * breach and forecast must hold for open_after_min consecutive minutes before an
          alert opens (temperatures here sit near the limit and flicker across it);
          fan faults and lost telemetry open at once.
        * an alert resolves after its condition has been clear for resolve_after_min.
        * a critical alert nobody has acknowledged for escalate_after_min is escalated.
        * incident_min_servers or more servers in one hall over (or forecast over) the limit
          at once open one hall incident, which points at cooling rather than at servers.
        * a drained server gets no forecast warnings (no new work goes there anyway)."""
        minute, ts = self.sim.minute - 1, str(self.sim.now() - pd.Timedelta(minutes=1))
        ac = self.alert_cfg
        views = {v.server_id: v for v in self.sim.views()}
        pred = self.forecasts.get(self.horizon, {})
        cond: dict[tuple, dict] = {}
        hot_by_hall: dict[tuple[str, str], list[str]] = {}
        for sid, v in views.items():
            s, l = self.sim.servers[sid], self.loc[sid]
            base = {"server_id": sid, "server_name": s.name, "site": l["site"], "hall": l["hall"], "minute": minute, "ts": ts}
            if v.fan_fault:
                cond[(sid, "fan_fault")] = {**base, "kind": "fan_fault", "severity": "critical", "value": v.temp,
                                            "text": f"{s.name}: fan failure, {v.temp:.1f} °C"}
            if v.age_min > self.stale_after:
                age = "never reported" if v.age_min >= 10 ** 6 else f"no telemetry for {v.age_min} min"
                cond[(sid, "stale")] = {**base, "kind": "stale", "severity": "warning", "value": None,
                                        "text": f"{s.name}: {age}; excluded from placement"}
                continue
            if v.temp >= self.threshold:
                cond[(sid, "breach")] = {**base, "kind": "breach", "severity": "critical", "value": v.temp,
                                         "text": f"{s.name} at {v.temp:.1f} °C, above the {self.threshold:.1f} °C limit"}
                hot_by_hall.setdefault((l["site"], l["hall"]), []).append(s.name)
            elif sid in pred and pred[sid] >= self.threshold:
                if not s.maintenance:
                    cond[(sid, "forecast")] = {**base, "kind": "forecast", "severity": "warning", "value": pred[sid],
                                               "text": f"{s.name} forecast to reach {pred[sid]:.1f} °C within {self.horizon} min"}
                hot_by_hall.setdefault((l["site"], l["hall"]), []).append(s.name)
        hall_size: dict[tuple[str, str], int] = {}
        for sid in views:
            hall_size[(self.loc[sid]["site"], self.loc[sid]["hall"])] = hall_size.get((self.loc[sid]["site"], self.loc[sid]["hall"]), 0) + 1
        for (site, hall), names in hot_by_hall.items():
            if (len(names) >= ac["incident_min_servers"]
                    and len(names) >= ac["incident_min_fraction"] * hall_size.get((site, hall), 1)):
                cond[(0, f"hall_incident:{site}/{hall}")] = {
                    "server_id": 0, "server_name": f"{site} hall {hall}", "site": site, "hall": hall, "minute": minute,
                    "ts": ts, "kind": f"hall_incident:{site}/{hall}", "severity": "critical", "value": float(len(names)),
                    "text": f"{site} hall {hall}: {len(names)} servers over or about to cross the limit "
                            f"({', '.join(sorted(names)[:5])}{'…' if len(names) > 5 else ''})"}
        # debounce: only conditions seen for long enough become alerts
        for key in list(self._pending):
            if key not in cond:
                del self._pending[key]
        ready = {}
        for key, a in cond.items():
            self._pending[key] = self._pending.get(key, 0) + 1
            needs = ac["open_after_min"] if a["kind"] in ("breach", "forecast") else 1
            if self._pending[key] >= needs:
                ready[key] = a
        open_ = self.store.open_alerts()
        for key, a in ready.items():
            if key in open_:
                self._clear_since.pop(open_[key]["alert_id"], None)
                if a["value"] is not None:
                    self.store.update_alert_peak(open_[key]["alert_id"], a["value"], a["text"])
            else:
                kind = a["kind"].split(":")[0]
                aid = self.store.open_alert({**a, "action": ACTIONS.get(kind, "")})
                self.notifier.notify("open", self.store.get_alert(aid))
        for key, a in open_.items():
            if key in cond:
                if (a["severity"] == "critical" and a["status"] == "open" and not a["escalated"]
                        and minute - a["opened_minute"] >= ac["escalate_after_min"]):
                    self.store.escalate_alert(a["alert_id"], ts)
                    self.notifier.notify("escalate", self.store.get_alert(a["alert_id"]))
                continue
            since = self._clear_since.setdefault(a["alert_id"], minute)
            gone = key[0] != 0 and key[0] not in views
            if minute - since >= ac["resolve_after_min"] or gone:
                self.store.resolve_alert(a["alert_id"], minute, ts)
                self._clear_since.pop(a["alert_id"], None)
                self.notifier.notify("resolve", self.store.get_alert(a["alert_id"]))

    def ack_alert(self, user: dict, alert_id: int, note: str) -> dict:
        a = self.store.ack_alert(alert_id, user["username"], note)
        if a is None:
            raise KeyError(f"no alert {alert_id}")
        self.audit(user["username"], user["role"], "alert_ack", a["server_name"],
                   {"alert_id": alert_id, "kind": a["kind"], "note": note[:200]})
        self.store.commit()
        return a

    # ---- recommender --------------------------------------------------------
    def rank(self, w: Workload, site: str | None = None):
        views = [v for v in self.sim.views() if site is None or self.loc[v.server_id]["site"] == site]
        ranked = recommend(views, w, self.forecasts.get(self.horizon, {}), self.threshold, self.horizon,
                           self.cfg["recommender"]["safety_margin_c"], self.stale_after,
                           getattr(self.forecaster, "interval_q", {}).get(self.horizon))
        # external servers are monitored, not managed: the console cannot start work on them
        ext = [c for c in ranked if self.sim.servers[c.server_id].external and c.status != "ineligible"]
        for c in ext:
            c.status, c.reasons = "ineligible", ["external server: monitored only, the console cannot start work there"]
            c.explanation = "Excluded: " + c.reasons[0] + "."
        if ext:
            from thermalview.recommender import STATUS_ORDER
            ranked.sort(key=lambda c: (STATUS_ORDER[c.status], c.predicted_temp if c.predicted_temp is not None
                                       and c.status != "ineligible" else float("inf"), -c.free_cpu, c.server_id))
            for i, c in enumerate(ranked, 1):
                c.rank = i
        return ranked

    def _place(self, w: Workload, sid: int, origin: str, who: str) -> S.Job:
        job = S.Job(job_id=self.next_job_id, cpu_pct=w.cpu_pct, mem_gb=w.mem_gb, duration_min=int(w.duration_min),
                    arrival_min=self.sim.minute)
        self.next_job_id += 1
        self.sim.place(job, sid)
        self.store.add_job(job, w.name, origin, who)
        return job

    def recommend_for(self, user: dict, w: Workload, site: str | None) -> dict:
        if site and site not in self.sites:
            raise ValueError(f"unknown site {site!r}")
        ranked = self.rank(w, site)
        ranking = [dict(c.as_dict(), **self.loc[c.server_id]) for c in ranked]
        rec_id = self.store.add_recommendation(self.sim.minute, {**vars(w), "site": site}, ranking, None, user["username"])
        top = next((c for c in ranked if c.status != "ineligible"), None)
        self.audit(user["username"], user["role"], "recommend", w.name,
                   {"recommendation_id": rec_id, "cpu_pct": w.cpu_pct, "mem_gb": w.mem_gb, "site": site or "any",
                    "eligible": sum(c.status != "ineligible" for c in ranked),
                    "top": top.name if top else None, "top_predicted_temp": round(top.predicted_temp, 2) if top else None})
        self.store.commit()
        return {"recommendation_id": rec_id, "workload": vars(w), "site": site, "threshold": self.threshold,
                "horizon": self.horizon, "model": self.forecaster.name, "candidates": ranking}

    def place_for(self, user: dict, w: Workload, sid: int, rec_id: int | None, reason: str = "",
                  origin: str = "operator", site: str | None = None) -> dict:
        """Place on a server the operator chose. Eligibility is re-checked now. Choosing anything
        other than ThermalView's current top pick is an override and needs a written reason,
        which is kept in the audit log with what ThermalView would have done."""
        ranked = {c.server_id: c for c in self.rank(w)}
        if sid not in ranked:
            raise KeyError(f"no active server {sid}")
        c = ranked[sid]
        if c.status == "ineligible":
            raise ActionError(f"{c.name} cannot take this workload: " + "; ".join(c.reasons))
        if self.sim.servers[sid].external:
            raise ActionError(f"{c.name} is an external server: the console only receives its telemetry and cannot "
                              "start work on it")
        top = self._top(w, site)          # compared within the same site filter the operator ranked with
        override = top is not None and top.server_id != sid
        if (override or c.status == "not_recommended") and not reason.strip():
            raise ValueError(f"{c.name} is not ThermalView's recommendation"
                             + (f" (that is {top.name})" if override else "") + "; give a reason to override")
        job = self._place(w, sid, origin, user["username"])
        if rec_id:
            self.store.mark_chosen(int(rec_id), sid)
        self.audit(user["username"], user["role"], "place_override" if override else "place", c.name,
                   {"job_id": job.job_id, "workload": w.name, "cpu_pct": round(w.cpu_pct, 1), "mem_gb": round(w.mem_gb, 1),
                    "recommendation_id": rec_id, "status": c.status, "predicted_temp": round(c.predicted_temp, 2),
                    "thermalview_pick": top.name if top else None, "reason": reason[:200] or None, "via": origin})
        self.store.commit()
        return {"job_id": job.job_id, "server_id": sid, "server_name": c.name, "status": c.status, "override": override}

    # ---- audit helper -------------------------------------------------------
    def audit(self, user: str, role: str, action: str, target: str = "", detail: dict | None = None) -> None:
        self.store.audit(user, role, action, target, detail, sim_ts=str(self.sim.now()))

    # ---- payloads -------------------------------------------------------------
    def server_rows(self) -> list[dict]:
        views = {v.server_id: v for v in self.sim.views()}
        margin = float(self.cfg["recommender"]["safety_margin_c"])
        out = []
        for sid, s in self.sim.servers.items():
            v, l = views[sid], self.loc[sid]
            f = {h: self.forecasts.get(h, {}).get(sid) for h in HORIZONS}
            pred = f[self.horizon]
            stale = v.age_min > self.stale_after
            state = ("stale" if stale else "fault" if v.fan_fault else "hot" if v.temp >= self.threshold else
                     "warning" if pred is not None and pred >= self.threshold else
                     "watch" if pred is not None and pred >= self.threshold - margin else "ok")
            has = bool(s.last)
            out.append({"server_id": sid, "name": s.name, **l, "kind": "external" if s.external else "simulated",
                        "source_node": "agent" if s.external else s.params.node_id,
                        "temp": round(v.temp, 2) if has else None, "load": round(v.load, 1) if has else None,
                        "power": round(v.power) if has and np.isfinite(v.power) else None,
                        "inlet": round(v.inlet, 1) if has and np.isfinite(v.inlet) else None,
                        "fan_fault": v.fan_fault, "maintenance": s.maintenance, "n_jobs": v.n_jobs,
                        "sensor_lost": s.sensor_lost, "runaway_pct": s.runaway_pct,
                        "age_min": None if v.age_min >= 10 ** 6 else v.age_min,
                        "free_cpu": round(v.free_cpu, 1), "free_mem": round(v.free_mem, 1),
                        "forecast": {str(h): (round(x, 2) if x is not None else None) for h, x in f.items()},
                        "state": state})
        return out

    def fleet(self) -> dict:
        servers = self.server_rows()
        sites = []
        rack_map = self.racks()
        for code, site in self.sites.items():
            halls = []
            for h in site["halls"]:
                racks = sorted({r for (sc, hn, r) in rack_map if sc == code and hn == h["name"]})
                halls.append({"name": h["name"], "inlet_rise_c": self.cooling.get((code, h["name"]), 0.0),
                              "racks": [{"rack": r, "servers": sorted(rack_map[(code, h["name"], r)],
                                                                      key=lambda i: self.loc[i]["slot"])} for r in racks]})
            members = [s for s in servers if s["site"] == code]
            temps = [s["temp"] for s in members if s["temp"] is not None]
            counts: dict[str, int] = {}
            for s in members:
                counts[s["state"]] = counts.get(s["state"], 0) + 1
            sites.append({"code": code, "name": site["name"], "halls": halls, "n": len(members), "states": counts,
                          "max_temp": max(temps) if temps else None,
                          "mean_temp": round(float(np.mean(temps)), 2) if temps else None,
                          "mean_load": round(float(np.mean([s["load"] for s in members if s["load"] is not None] or [0])), 1),
                          "maintenance": sum(s["maintenance"] for s in members)})
        active = self.store.alerts("active", limit=50)
        err = self.store.forecast_error(max(0, self.sim.minute - 120))
        return {"org": self.fc.get("org_name", ""), "sim_time": str(self.sim.now()), "minute": self.sim.minute,
                "running": self.running, "speed": self.speed, "auto_jobs": self.auto_jobs,
                "threshold": self.threshold, "limit_source": self.limit_source, "horizon": self.horizon, "safety_margin": self.cfg["recommender"]["safety_margin_c"],
                "model": self.forecaster.name, "data_source": self.data_source,
                "servers": servers, "sites": sites,
                "mode": self.mode, "queue": self.queue_rows(),
                "notifier": {"enabled": bool(self.notifier.url), "sent": self.notifier.sent, "failed": self.notifier.failed},
                "alerts": {"active": active[:20], "critical": sum(a["severity"] == "critical" for a in active),
                           "escalated": sum(bool(a["escalated"]) for a in active),
                           "warning": sum(a["severity"] == "warning" for a in active),
                           "unacked": sum(a["status"] == "open" for a in active)},
                "live_forecast_check": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in err.items()}}

    def metrics_text(self) -> str:
        """Prometheus text exposition of the fleet's current state (scraped by Prometheus, shown in Grafana)."""
        def esc(v) -> str:
            return str(v).replace("\\", "\\\\").replace('"', '\\"')
        lines = []
        def metric(name, help_, typ, samples):
            lines.append(f"# HELP {name} {help_}")
            lines.append(f"# TYPE {name} {typ}")
            for labels, value in samples:
                if value is None or (isinstance(value, float) and not np.isfinite(value)):
                    continue
                lab = ",".join(f'{k}="{esc(v)}"' for k, v in labels.items())
                lines.append(f"{name}{{{lab}}} {float(value):.4g}" if lab else f"{name} {float(value):.4g}")
        rows = self.server_rows()
        key = lambda r: {"server": r["name"], "site": r["site"], "hall": r["hall"], "rack": r["rack"]}
        metric("thermalview_cpu_temperature_celsius", "Latest CPU temperature (hottest core).", "gauge",
               [(key(r), r["temp"]) for r in rows])
        for h in HORIZONS:
            metric(f"thermalview_forecast_{h}m_celsius", f"Forecast CPU temperature {h} minutes ahead.", "gauge",
                   [(key(r), r["forecast"][str(h)]) for r in rows])
        metric("thermalview_cpu_load_percent", "CPU load.", "gauge", [(key(r), r["load"]) for r in rows])
        metric("thermalview_power_watts", "Node power.", "gauge", [(key(r), r["power"]) for r in rows])
        metric("thermalview_inlet_celsius", "Inlet air temperature.", "gauge", [(key(r), r["inlet"]) for r in rows])
        metric("thermalview_server_state", "1 for the server's current state.", "gauge",
               [({**key(r), "state": r["state"]}, 1) for r in rows])
        metric("thermalview_server_drained", "1 if drained for maintenance.", "gauge",
               [(key(r), int(r["maintenance"])) for r in rows])
        metric("thermalview_jobs_running", "Jobs running on the server.", "gauge", [(key(r), r["n_jobs"]) for r in rows])
        metric("thermalview_thermal_limit_celsius", "Thermal limit in use.", "gauge", [({}, self.threshold)])
        active = self.store.alerts("active", limit=10_000)
        sev = {}
        for a in active:
            sev[(a["severity"], a["status"])] = sev.get((a["severity"], a["status"]), 0) + 1
        metric("thermalview_alerts_active", "Active alerts by severity and status.", "gauge",
               [({"severity": s, "status": st}, n) for (s, st), n in sorted(sev.items())] or [({"severity": "none", "status": "none"}, 0)])
        metric("thermalview_queue_length", "Jobs waiting for an operator decision.", "gauge", [({"mode": self.mode}, len(self.queue))])
        err = self.store.forecast_error(max(0, self.sim.minute - 120))
        metric("thermalview_live_forecast_mae_celsius", "5-min forecast MAE over the last 2 simulated hours.", "gauge",
               [({"model": self.forecaster.name}, err.get("mae")), ({"model": "persistence"}, err.get("persist"))])
        metric("thermalview_simulated_minute", "Simulation clock (minutes since start).", "counter", [({}, self.sim.minute)])
        return "\n".join(lines) + "\n"

    def results(self) -> dict:
        res = self.cfg.path("results")
        out = {"data_source": self.data_source, "eval_month": str(self.cfg["m100"].get("year_month") or "")}
        for name in ("forecast_summary.json", "placement_meta.json"):
            p = res / name
            out[name.split(".")[0]] = json.loads(p.read_text()) if p.exists() else None
        for name in ("transfer_summary.json", "job_event_summary_transfer.json"):
            p = res / name
            out[name.split(".")[0]] = json.loads(p.read_text()) if p.exists() else None
        for name in ("forecast_metrics.csv", "placement_summary.csv", "simulator_fidelity.csv", "feature_importance.csv",
                     "transfer_metrics.csv", "sensitivity_summary.csv", "job_event_heat_model_transfer.csv",
                     "job_event_forecast_transfer.csv"):
            p = res / name
            out[name.split(".")[0]] = (pd.read_csv(p).replace({np.nan: None}).to_dict("records") if p.exists() else None)
        figs = res / "figures"
        out["figures"] = sorted(p.name for p in figs.glob("*.png")) if figs.exists() else []
        return out

