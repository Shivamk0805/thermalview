"""Closed-loop fleet simulator.

Each simulated server copies one real M100 node. It replays that node's
recordings from the unseen test period (load, temperature, inlet air, power,
fan speed) and adds only what the simulation changes: background load is
scaled down to make room for placed jobs, jobs add load on top, and inlet or
fan-fault scenarios add heat. The temperature is the recorded temperature
plus the node's fitted response to that difference in load (see
thermal_model.py), so a placement decision changes the server's future
temperature while everything the model cannot explain stays as recorded.
Everything random is driven by seeded generators.

This is a simulation calibrated from real data. Results produced with it
are simulation evidence, not measurements of real servers.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from thermalview.features import LONG_WINDOW
from thermalview.thermal_model import NodeParams

WINDOW_READY = LONG_WINDOW + 1  # minutes of history before every feature set has a value

FAN_FAULT_ALPHA = 0.7      # a failed fan removes heat more slowly ...
FAN_FAULT_RISE_C = 12.0    # ... and settles hotter


@dataclass
class Job:
    job_id: int
    cpu_pct: float
    mem_gb: float
    duration_min: int
    arrival_min: int
    server_id: int | None = None
    start_min: int | None = None

    @property
    def end_min(self) -> int | None:
        return None if self.start_min is None else self.start_min + self.duration_min


@dataclass
class SimServer:
    server_id: int
    name: str
    params: NodeParams
    bg_load: np.ndarray
    inlet_trace: np.ndarray
    mem_gb: float
    bg_mem_frac: float
    temp: float = 0.0
    fan_fault: bool = False
    jobs: list[Job] = field(default_factory=list)
    buf: dict = field(default_factory=dict)
    last: dict = field(default_factory=dict)
    trace_offset: int = 0          # where in the replayed trace this server starts
    inlet_bias: float = 0.0        # extra inlet air (e.g. a hall with degraded cooling)
    maintenance: bool = False      # drained: no new placements
    external: bool = False         # telemetry arrives through the ingest API, not the simulator
    temp_trace: np.ndarray | None = None    # recorded temperature (trace-anchored mode)
    power_trace: np.ndarray | None = None
    fan_trace: np.ndarray | None = None
    delta: float = 0.0             # modelled deviation from the recorded temperature
    sensor_lost: bool = False      # fault: the server keeps running but stops reporting telemetry
    runaway_pct: float = 0.0       # fault: a stuck process burning this much CPU outside any placed job


@dataclass
class ServerView:
    """What the recommender and policies are allowed to see."""
    server_id: int
    name: str
    temp: float
    load: float
    power: float
    inlet: float
    free_cpu: float
    free_mem: float
    fan_fault: bool
    age_min: int
    alpha: float
    gain: float
    n_jobs: int
    maintenance: bool = False


class FleetSim:
    def __init__(self, servers: list[SimServer], cpu_cap: float, bg_scale: float, noise_scale: float,
                 seed: int, history: int = 720, start_ts: pd.Timestamp | None = None, trace_offset: int = 0):
        self.servers = {s.server_id: s for s in servers}
        self.cpu_cap = cpu_cap
        self.bg_scale = bg_scale
        self.noise_scale = noise_scale
        self.rng = np.random.default_rng(seed)
        self.minute = 0
        self.trace_offset = trace_offset
        self.inlet_offset = 0.0
        self.start_ts = start_ts or pd.Timestamp("2026-01-01", tz="UTC")
        self.history = history
        for s in servers:
            self._init_server(s)

    def _init_server(self, s: SimServer) -> None:
        s.buf = {k: deque(maxlen=self.history) for k in ("ts", "temp", "load", "power", "inlet", "fan")}
        if s.external:
            return
        if s.temp_trace is not None:
            s.delta = self._delta_target(s)
            s.temp = float(s.temp_trace[self._idx(s)]) + s.delta
        else:
            s.temp = s.params.steady_state(self._bg(s), self._inlet(s))

    def add_server(self, s: SimServer) -> None:
        if s.server_id in self.servers:
            raise ValueError(f"server id {s.server_id} already exists")
        self._init_server(s)
        self.servers[s.server_id] = s

    def remove_server(self, server_id: int) -> SimServer:
        return self.servers.pop(server_id)

    def ingest(self, server_id: int, reading: dict) -> dict:
        """Record one reading for an external server (from the telemetry API).
        It is stamped with the current simulated minute."""
        s = self.servers[server_id]
        row = {"server_id": server_id, "ts": self.now(), "minute": self.minute,
               "temp": float(reading["temp"]), "load": float(reading["load"]),
               "power": float(reading.get("power") if reading.get("power") is not None else float("nan")),
               "inlet": float(reading.get("inlet") if reading.get("inlet") is not None else float("nan")),
               "fan": float(reading.get("fan") if reading.get("fan") is not None else float("nan")),
               "fan_fault": s.fan_fault, "n_jobs": len(s.jobs)}
        if s.last.get("minute") == self.minute:          # one reading per minute: replace
            for k in ("ts", "temp", "load", "power", "inlet", "fan"):
                s.buf[k].pop()
        for k in ("temp", "load", "power", "inlet", "fan"):
            s.buf[k].append(row[k])
        s.buf["ts"].append(row["ts"])
        s.temp, s.last = row["temp"], row
        return row

    # ---- state helpers -------------------------------------------------
    def _idx(self, s: SimServer) -> int:
        return (self.minute + self.trace_offset + s.trace_offset) % len(s.bg_load)

    def _bg(self, s: SimServer) -> float:
        return float(s.bg_load[self._idx(s)]) * self.bg_scale

    def _inlet(self, s: SimServer) -> float:
        return float(s.inlet_trace[self._idx(s)]) + self.inlet_offset + s.inlet_bias

    def _rec_load(self, s: SimServer) -> float:
        return float(s.bg_load[self._idx(s)])

    def _delta_target(self, s: SimServer) -> float:
        """Steady-state deviation from the recorded temperature right now."""
        p = s.params
        extra_inlet = self.inlet_offset + s.inlet_bias
        return (p.gain * (self.load(s) - self._rec_load(s)) + p.inlet_k * extra_inlet
                + (FAN_FAULT_RISE_C if s.fan_fault else 0.0))

    def load(self, s: SimServer) -> float:
        if s.external:          # the agent's reading already includes whatever runs there
            return float(s.last.get("load", 0.0))
        return min(100.0, self._bg(s) + s.runaway_pct + sum(j.cpu_pct for j in s.jobs))

    def free_cpu(self, s: SimServer) -> float:
        return self.cpu_cap - self.load(s)

    def free_mem(self, s: SimServer) -> float:
        return s.mem_gb * (1 - s.bg_mem_frac) - sum(j.mem_gb for j in s.jobs)

    def now(self) -> pd.Timestamp:
        return self.start_ts + pd.Timedelta(minutes=self.minute)

    # ---- simulation ----------------------------------------------------
    def step(self) -> list[dict]:
        """Record telemetry for the current minute, then advance one minute."""
        rows = []
        ts = self.now()
        for s in self.servers.values():
            s.jobs = [j for j in s.jobs if j.end_min is None or j.end_min > self.minute]
            if s.external:
                continue
            p = s.params
            load = self.load(s)
            inlet = self._inlet(s)
            if s.temp_trace is not None:
                # trace-anchored: update the deviation with this minute's load, then report
                i = self._idx(s)
                alpha = p.alpha * (FAN_FAULT_ALPHA if s.fan_fault else 1.0)
                s.delta += alpha * (self._delta_target(s) - s.delta)
                s.temp = float(s.temp_trace[i]) + s.delta
                dl = load - self._rec_load(s)
                power = float(s.power_trace[i]) + max(p.power_1, 0.0) * dl
                fan = float(s.fan_trace[i]) + max(p.fan_1, 0.0) * max(0.0, s.delta)
                row = {"server_id": s.server_id, "ts": ts, "minute": self.minute, "temp": s.temp, "load": load,
                       "power": power, "inlet": inlet, "fan": fan, "fan_fault": s.fan_fault, "n_jobs": len(s.jobs),
                       "rec_temp": float(s.temp_trace[i])}
                if s.sensor_lost:          # physics continues, nothing is reported
                    continue
                for k in ("temp", "load", "power", "inlet", "fan"):
                    s.buf[k].append(row[k])
                s.buf["ts"].append(ts)
                s.last = row
                rows.append(row)
                continue
            power = p.power_0 + p.power_1 * load
            fan = p.fan_0 + p.fan_1 * max(0.0, s.temp - 50.0)
            # a failed fan is not modelled as a lower rpm reading: the forecaster
            # learned on real data where low fan speed means a cool node, so a
            # synthetic low reading would mislead it. The fault shows up as heat.
            row = {"server_id": s.server_id, "ts": ts, "minute": self.minute, "temp": s.temp, "load": load,
                   "power": power, "inlet": inlet, "fan": fan, "fan_fault": s.fan_fault, "n_jobs": len(s.jobs)}
            if not s.sensor_lost:
                for k in ("temp", "load", "power", "inlet", "fan"):
                    s.buf[k].append(row[k])
                s.buf["ts"].append(ts)
                s.last = row
                rows.append(row)
            alpha = p.alpha * (FAN_FAULT_ALPHA if s.fan_fault else 1.0)
            target = p.steady_state(load, inlet) + (FAN_FAULT_RISE_C if s.fan_fault else 0.0)
            s.temp = s.temp + alpha * (target - s.temp) + self.rng.normal(0.0, p.sigma * self.noise_scale)
        self.minute += 1
        return rows

    def place(self, job: Job, server_id: int) -> None:
        job.server_id = server_id
        job.start_min = self.minute
        self.servers[server_id].jobs.append(job)

    def views(self) -> list[ServerView]:
        out = []
        for s in self.servers.values():
            last = s.last or {}
            out.append(ServerView(
                server_id=s.server_id, name=s.name, temp=float(last.get("temp", s.temp)),
                load=self.load(s), power=float(last.get("power", 0.0)), inlet=float(last.get("inlet", 0.0)),
                free_cpu=self.free_cpu(s), free_mem=self.free_mem(s), fan_fault=s.fan_fault,
                age_min=(max(0, self.minute - 1 - int(last["minute"])) if "minute" in last else 10 ** 6),
                alpha=s.params.alpha * (FAN_FAULT_ALPHA if s.fan_fault else 1.0), gain=s.params.gain,
                n_jobs=len(s.jobs), maintenance=s.maintenance))
        return out

    def buffers(self) -> dict[int, dict]:
        return {sid: {k: list(v) for k, v in s.buf.items() if k != "ts"} for sid, s in self.servers.items()}


def generate_jobs(minutes: int, rate: float, wl: dict, seed: int) -> list[Job]:
    rng = np.random.default_rng(seed)
    jobs, jid = [], 0
    for m in range(minutes):
        for _ in range(rng.poisson(rate)):
            jobs.append(Job(job_id=jid, arrival_min=m,
                            cpu_pct=float(rng.uniform(wl["cpu_min"], wl["cpu_max"])),
                            mem_gb=float(rng.uniform(wl["mem_min_gb"], wl["mem_max_gb"])),
                            duration_min=max(1, int(rng.lognormal(np.log(wl["duration_median_min"]), wl["duration_sigma"])))))
            jid += 1
    return jobs


def pick_nodes(params: dict[str, NodeParams], n: int, prefer: list[str]) -> list[str]:
    """Fitted nodes only; nodes the forecaster never saw come first."""
    fitted = sorted(k for k, p in params.items() if p.fitted)
    first = [k for k in prefer if k in fitted]
    rest = [k for k in fitted if k not in first]
    return (first + rest)[:n]


def build_servers(table: pd.DataFrame, params: dict[str, NodeParams], node_ids: list[str],
                  trace_start: pd.Timestamp, sim_cfg: dict) -> list[SimServer]:
    """Servers that replay each node's recordings from trace_start onwards."""
    servers = []
    for i, node in enumerate(node_ids):
        p = params[node]
        g = table[(table["node_id"] == node) & (table["ts"] >= trace_start)].sort_values("ts")
        if g["load_pct"].notna().sum() < 60 or g["cpu_temp_c"].notna().sum() < 60:
            raise ValueError(f"node {node} has too little trace data after {trace_start}")

        def trace(col: str, default: np.ndarray | float) -> np.ndarray:
            x = g[col].interpolate(limit_direction="both")
            return (x.fillna(pd.Series(default, index=g.index)) if not np.isscalar(default) else x.fillna(default)).to_numpy(float)

        load = trace("load_pct", 0.0)
        servers.append(SimServer(
            server_id=i + 1, name=f"TV-{i + 1:02d}", params=p, bg_load=load,
            inlet_trace=trace("inlet_c", p.inlet_mean), temp_trace=trace("cpu_temp_c", float(g["cpu_temp_c"].median())),
            power_trace=trace("power_w", p.power_0 + p.power_1 * load),
            fan_trace=trace("fan_rpm", p.fan_0),
            mem_gb=float(sim_cfg["memory_gb"]), bg_mem_frac=float(sim_cfg["background_memory_frac"])))
    return servers
