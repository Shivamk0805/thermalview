"""Per-node heat response, fitted from real telemetry.

A node's CPU temperature is modelled as its *recorded* temperature plus a
deviation caused by load that differs from the recording:

    D(t) = D(t-1) + alpha * (gain * (L_sim(t) - L_rec(t)) + inlet_k * dI(t) - D(t-1))
    T_sim(t) = T_rec(t) + D(t)

alpha is the fraction of the remaining response reached per minute; gain is
the steady-state rise in degrees C per percentage point of load. Anchoring to
the recorded trace keeps everything the data cannot explain (GPU heat,
neighbours, sensor quantisation) exactly as it happened, and lets the model
govern only what placement changes. Fitting an absolute steady state instead
was tried first on M100 and drifted badly (see the project history).

alpha and gain are fitted per node on data before the test period: for each
alpha on a grid, gain is the least-squares slope of the observed temperature
change over 1..15 minutes on the filtered load change; the alpha with the
lowest mean absolute error wins. A node whose load barely varied in the fitting
period (sd < 10 points or fewer than 5 one-minute changes of 10+ points) is not
fitted and takes the fleet median: its slope would be noise. inlet_k is fixed at 1 (CPU temperature moves
one-for-one with inlet air at steady state); it is an assumption used only by
the cooling scenarios, not fitted. Power and fan speed get linear maps.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class NodeParams:
    node_id: str
    alpha: float
    base: float
    gain: float          # steady-state C per % load
    inlet_k: float
    sigma: float         # residual std of one-minute change
    power_0: float
    power_1: float       # W per % load
    fan_0: float
    fan_1: float         # rpm per C above 50
    inlet_mean: float
    fitted: bool         # False = fell back to fleet median parameters
    n_rows: int

    def steady_state(self, load: float, inlet: float) -> float:
        return self.base + self.gain * load + self.inlet_k * inlet


def _lstsq(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(X, y, rcond=None)[0]


ALPHA_GRID = np.round(np.arange(0.05, 1.0001, 0.05), 2)
FIT_HORIZON = 15        # minutes of response used in the fit
FIT_EVERY = 7           # start a fitting segment every 7 minutes


def _segments(g: pd.DataFrame, length: int, every: int) -> tuple[np.ndarray, np.ndarray]:
    """Complete (no missing minute) temperature and load windows of length+1 minutes."""
    T = g["cpu_temp_c"].to_numpy(float)
    L = g["load_pct"].to_numpy(float)
    ok = ~(np.isnan(T) | np.isnan(L))
    Ts, Ls = [], []
    for s0 in range(0, len(g) - length - 1, every):
        if ok[s0:s0 + length + 1].all():
            Ts.append(T[s0:s0 + length + 1])
            Ls.append(L[s0:s0 + length + 1])
    return np.array(Ts), np.array(Ls)


def filtered_change(loads: np.ndarray, alpha: float) -> np.ndarray:
    """Response state x[k] to load changes relative to loads[..., 0], for each step k
    (x[0] = 0). Works on one window (1-D) or many (2-D, windows in rows)."""
    d = loads - loads[..., :1]
    x = np.zeros_like(d, dtype=float)
    for k in range(1, d.shape[-1]):
        x[..., k] = x[..., k - 1] + alpha * (d[..., k] - x[..., k - 1])
    return x


MIN_LOAD_SD = 10.0      # load must actually vary for its effect to be identifiable ...
MIN_LOAD_STEPS = 5      # ... with at least this many one-minute changes of >= 10 points


def fit_node(g: pd.DataFrame, min_alpha: float, max_alpha: float) -> NodeParams | None:
    g = g.sort_values("ts")
    load = g["load_pct"]
    if load.std() < MIN_LOAD_SD or int((load.diff().abs() >= 10).sum()) < MIN_LOAD_STEPS:
        return None             # a node that never changed load cannot tell us its gain
    Ts, Ls = _segments(g, FIT_HORIZON, FIT_EVERY)
    if len(Ts) < 50:
        return None
    Y = (Ts - Ts[:, :1])[:, 1:].ravel()
    best = None
    for a in ALPHA_GRID:
        if not (min_alpha <= a <= max_alpha):
            continue
        X = filtered_change(Ls, a)[:, 1:].ravel()
        xx = float(X @ X)
        if xx <= 0:
            continue
        gain = float(X @ Y) / xx
        err = float(np.abs(Y - gain * X).mean())
        if best is None or err < best[0]:
            best = (err, float(a), gain)
    if best is None or best[2] <= 0:
        return None
    _, alpha, gain = best
    X1 = filtered_change(Ls[:, :2], alpha)[:, 1]
    sigma = float(np.std((Ts[:, 1] - Ts[:, 0]) - gain * X1))
    p0, p1 = 300.0, 10.0
    pw = g[["load_pct", "power_w"]].dropna()
    if len(pw) > 200 and pw["load_pct"].std() > 0:
        p1, p0 = np.polyfit(pw["load_pct"], pw["power_w"], 1)
    f0, f1 = 3000.0, 80.0
    fn = g[["cpu_temp_c", "fan_rpm"]].dropna()
    if len(fn) > 200 and fn["cpu_temp_c"].std() > 0:
        f1, f0 = np.polyfit(np.clip(fn["cpu_temp_c"] - 50, 0, None), fn["fan_rpm"], 1)
    has_inlet = g["inlet_c"].notna().mean() > 0.5
    return NodeParams(node_id=str(g["node_id"].iloc[0]), alpha=alpha,
                      base=float(np.nanmedian(g["cpu_temp_c"]) - gain * np.nanmedian(g["load_pct"])),
                      gain=gain, inlet_k=1.0, sigma=sigma, power_0=float(p0), power_1=float(p1),
                      fan_0=float(f0), fan_1=float(f1),
                      inlet_mean=float(g["inlet_c"].mean()) if has_inlet else 22.0, fitted=True, n_rows=int(len(Ts)))


def fit_all(table: pd.DataFrame, fit_until: pd.Timestamp, min_alpha: float, max_alpha: float) -> dict[str, NodeParams]:
    """Fit every node on data before fit_until. Nodes without enough clean data,
    or with a non-positive gain, get the fleet-median parameters (fitted=False)."""
    train = table[table["ts"] < fit_until]
    fitted = {}
    for node, g in train.groupby("node_id"):
        p = fit_node(g, min_alpha, max_alpha)
        if p is not None:
            fitted[node] = p
    if not fitted:
        raise ValueError("no node produced a valid heat-model fit")
    med = {k: float(np.median([getattr(p, k) for p in fitted.values()]))
           for k in ("alpha", "base", "gain", "inlet_k", "sigma", "power_0", "power_1", "fan_0", "fan_1", "inlet_mean")}
    out = dict(fitted)
    for node in train["node_id"].unique():
        if node not in out:
            out[node] = NodeParams(node_id=node, fitted=False, n_rows=0, **med)
    return out


def fidelity(table: pd.DataFrame, params: dict[str, NodeParams], start: pd.Timestamp,
             horizons=(5, 15, 30), every: int = 30) -> pd.DataFrame:
    """Simulator check on unseen time. From each start point, the heat model is
    given the node's real future load and predicts the temperature change it
    causes (T0 + gain * filtered load change); this is compared with what the
    node actually did, and with persistence (no change). It tests the response
    the simulator uses for placed jobs; it is not a forecast (future load is known)."""
    hmax = max(horizons)
    rows = []
    for node, g in table[table["ts"] >= start].groupby("node_id"):
        p = params.get(node)
        if p is None:
            continue
        Ts, Ls = _segments(g.sort_values("ts"), hmax, every)
        if not len(Ts):
            continue
        pred = Ts[:, :1] + p.gain * filtered_change(Ls, p.alpha)
        for h in horizons:
            for e_sim, e_per in zip(np.abs(pred[:, h] - Ts[:, h]), np.abs(Ts[:, 0] - Ts[:, h])):
                rows.append({"node_id": node, "horizon_min": h, "sim_abs_err": e_sim,
                             "persistence_abs_err": e_per, "fitted": p.fitted})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    out = (df.groupby("horizon_min")
             .agg(n=("sim_abs_err", "size"), sim_mae=("sim_abs_err", "mean"),
                  persistence_mae=("persistence_abs_err", "mean"))
             .reset_index())
    out["skill"] = 1 - out["sim_mae"] / out["persistence_mae"]
    return out


def save_params(params: dict[str, NodeParams], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: asdict(v) for k, v in params.items()}, indent=2))


def load_params(path: Path) -> dict[str, NodeParams]:
    return {k: NodeParams(**v) for k, v in json.loads(path.read_text()).items()}
