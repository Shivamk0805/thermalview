"""Workload placement recommender: filter, predict, rank, explain.

For each eligible server the predicted temperature after placing the job is

    forecast(t + h)                      where the server is already heading
  + gain * cpu * (1 - (1 - alpha)^h)     extra heat this job adds (heat model)

The forecast comes from the trained ML model (or, in the ablation, from
persistence). The job's own effect comes from the per-node heat model,
because a tree model asked "what if load jumps?" would be extrapolating.
The same function is used by the live app and by the policy evaluation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from thermalview.features import fast_columns, features_from_buffers, increment_from_load
from thermalview.simulator import ServerView

STATUS_ORDER = {"recommended": 0, "caution": 1, "not_recommended": 2, "ineligible": 3}


@dataclass
class Workload:
    cpu_pct: float
    mem_gb: float
    duration_min: int = 30
    name: str = "workload"


@dataclass
class Candidate:
    server_id: int
    name: str
    status: str
    rank: int = 0
    reasons: list[str] = field(default_factory=list)
    current_temp: float = 0.0
    forecast_temp: float | None = None
    job_increment: float | None = None
    predicted_temp: float | None = None
    margin_c: float | None = None
    upper_temp: float | None = None    # predicted + calibrated interval half-width (None if no interval)
    load: float = 0.0
    free_cpu: float = 0.0
    free_mem: float = 0.0
    explanation: str = ""

    def as_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 2)
        return d


class MLForecaster:
    def __init__(self, bundle_path: Path):
        b = joblib.load(bundle_path)
        self.bundle = b
        self.horizon = int(b["primary_horizon"])
        for m in b["models"].values():
            if hasattr(m, "n_jobs"):
                m.n_jobs = 1               # predicting ~10 rows: threads only add overhead
        self.cols = b["feature_columns"]
        self.cols_by_h = {int(k): v for k, v in b.get("feature_columns_by_h", {}).items()}
        self.interval_q = {int(k): float(v) for k, v in b.get("interval_q", {}).items()}
        self.optional = b["optional_present"]
        self.threshold = float(b["threshold_c"])
        self.name = b["chosen"][self.horizon]
        self.data_source = b.get("data_source", "unknown")

    def predict(self, buffers: dict[int, dict], horizon: int | None = None) -> dict[int, float]:
        h = horizon or self.horizon
        model = self.bundle["models"][h]
        ids, rows = [], []
        for sid, buf in buffers.items():
            row = features_from_buffers(buf, self.optional)
            if row is not None:
                ids.append(sid)
                rows.append(row)
        if not rows:
            return {}
        X = pd.DataFrame(rows, columns=fast_columns(self.optional))[self.cols_by_h.get(h, self.cols)]
        ok = X.notna().all(axis=1).to_numpy()   # extended features need 60 minutes of history
        if not ok.all():
            ids = [i for i, k in zip(ids, ok) if k]
            X = X[ok]
            if X.empty:
                return {}
        delta = model.predict(X)
        return {sid: float(buffers[sid]["temp"][-1] + d) for sid, d in zip(ids, delta)}


class PersistenceForecaster:
    name = "persistence"

    def __init__(self, horizon: int):
        self.horizon = horizon

    def predict(self, buffers: dict[int, dict], horizon: int | None = None) -> dict[int, float]:
        return {sid: float(buf["temp"][-1]) for sid, buf in buffers.items() if buf["temp"]}


def eligibility(v: ServerView, w: Workload, stale_after: int) -> list[str]:
    """Reasons a server cannot take the workload (empty list = eligible)."""
    reasons = []
    if v.maintenance:
        reasons.append("in maintenance (drained)")
    if v.fan_fault:
        reasons.append("active fan fault")
    if v.age_min >= 10 ** 6:
        reasons.append("no telemetry received yet")
    elif v.age_min > stale_after:
        reasons.append(f"telemetry is {v.age_min} min old")
    if v.free_cpu < w.cpu_pct:
        reasons.append(f"needs {w.cpu_pct:.0f}% CPU, only {max(0.0, v.free_cpu):.0f}% free")
    if v.free_mem < w.mem_gb:
        reasons.append(f"needs {w.mem_gb:.0f} GB, only {max(0.0, v.free_mem):.0f} GB free")
    return reasons


def recommend(views: list[ServerView], w: Workload, forecasts: dict[int, float], threshold: float,
              horizon: int, safety_margin: float, stale_after: int,
              interval_q: float | None = None) -> list[Candidate]:
    """Rank servers for workload w. interval_q, when given, is the calibrated
    half-width of the forecast interval; it is reported (upper bound), not used
    to rank, so the ranking is the one evaluated in the paper."""
    cands = []
    for v in views:
        c = Candidate(server_id=v.server_id, name=v.name, status="ineligible", current_temp=v.temp,
                      load=v.load, free_cpu=v.free_cpu, free_mem=v.free_mem)
        c.reasons = eligibility(v, w, stale_after)
        if not c.reasons and v.server_id not in forecasts:
            c.reasons = ["not enough history for a forecast yet"]
        if c.reasons:
            c.explanation = "Excluded: " + "; ".join(c.reasons) + "."
            cands.append(c)
            continue
        c.forecast_temp = forecasts[v.server_id]
        c.job_increment = increment_from_load(w.cpu_pct, v.gain, v.alpha, horizon)
        c.predicted_temp = c.forecast_temp + c.job_increment
        c.margin_c = threshold - c.predicted_temp
        if interval_q is not None:
            c.upper_temp = c.predicted_temp + interval_q
        c.status = ("recommended" if c.margin_c >= safety_margin else
                    "caution" if c.margin_c >= 0 else "not_recommended")
        trend = c.forecast_temp - v.temp
        heading = "rising" if trend > 0.3 else "falling" if trend < -0.3 else "steady"
        c.explanation = (f"Now {v.temp:.1f} °C and {heading}; forecast {c.forecast_temp:.1f} °C in {horizon} min, "
                         f"+{c.job_increment:.1f} °C from this job = {c.predicted_temp:.1f} °C, "
                         f"{abs(c.margin_c):.1f} °C {'below' if c.margin_c >= 0 else 'above'} the "
                         f"{threshold:.1f} °C limit. {v.free_cpu - w.cpu_pct:.0f}% CPU would remain free.")
        if c.upper_temp is not None:
            c.explanation += (f" Forecast uncertainty: 9 times in 10 it stays below {c.upper_temp:.1f} °C"
                              + (" (above the limit)." if c.upper_temp >= threshold else "."))
        cands.append(c)
    cands.sort(key=lambda c: (STATUS_ORDER[c.status],
                              c.predicted_temp if c.predicted_temp is not None else np.inf,
                              -c.free_cpu, c.server_id))
    for i, c in enumerate(cands, start=1):
        c.rank = i
    return cands
