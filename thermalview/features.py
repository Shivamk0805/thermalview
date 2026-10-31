"""Feature engineering, shared by offline training and the live app.

All features at time t use only data at or before t. Targets are the
temperature *change* over the horizon, temp(t+h) - temp(t); models add it
back to the current temperature. Predicting the change makes "no change"
(persistence) the natural zero point, so any skill a model shows is skill
beyond persistence.
"""
from __future__ import annotations

import pandas as pd

WINDOW = 15       # minutes of history for the base rolling features
LONG_WINDOW = 60  # minutes of history for the "extended" features

BASE_FEATURES = [
    "temp", "temp_d1", "temp_d5", "temp_d15", "temp_mean15", "temp_std15",
    "load", "load_d1", "load_d5", "load_mean15",
]
OPTIONAL_FEATURES = {
    "power_w": ["power", "power_d5"],
    "inlet_c": ["inlet", "inlet_d15"],
    "fan_rpm": ["fan", "fan_d5"],
}


# "extended" mode adds longer history and the shape of recent load, all
# relative (no absolute levels). Chosen on the validation period of the
# evaluation month; see docs/paper (feature-group ablation).
EXTENDED_FEATURES = [
    "temp_d30", "temp_d60", "temp_std30", "temp_std60", "temp_dev60",
    "load_d30", "load_d60", "load_mean30", "load_mean60", "load_max15", "load_min15", "min_since_load_step",
]
MODES = ("all", "relative", "extended")


def lookback(mode: str) -> int:
    """Minutes of history a feature row needs in this mode."""
    return LONG_WINDOW if mode == "extended" else WINDOW


# Absolute levels that drift between periods and nodes (inlet air, fan, power,
# the temperature itself). In "relative" mode (the default, chosen on the
# validation period of the development month) models see only changes, spreads
# and the load, so they cannot memorise one period's levels.
LEVEL_FEATURES = {"temp", "temp_mean15", "power", "inlet", "fan"}


def feature_columns(optional_present: list[str], mode: str = "all") -> list[str]:
    cols = list(BASE_FEATURES)
    for src, names in OPTIONAL_FEATURES.items():
        if src in optional_present:
            cols += names
    if mode in ("relative", "extended"):
        cols = [c for c in cols if c not in LEVEL_FEATURES]
    if mode == "extended":
        cols += EXTENDED_FEATURES
    elif mode not in MODES:
        raise ValueError(f"unknown feature mode {mode!r}")
    return cols


def _minutes_since_step(load, step: float = 10.0, cap: int = LONG_WINDOW):
    """Minutes since the load last changed by >= `step` points in one minute (capped)."""
    import numpy as np
    out = np.full(len(load), float(cap))
    last = None
    for i in range(1, len(load)):
        d = load[i] - load[i - 1]
        if d == d and abs(d) >= step:   # d == d skips NaN
            last = i
        if last is not None:
            out[i] = min(cap, i - last)
    return out


def fast_columns(optional_present: list[str]) -> list[str]:
    """Column order of features_from_buffers(): every base, optional and extended feature."""
    return feature_columns(optional_present, "all") + EXTENDED_FEATURES


def _node_features(g: pd.DataFrame, optional_present: list[str], horizons: list[int]) -> pd.DataFrame:
    temp, load = g["cpu_temp_c"], g["load_pct"]
    f = pd.DataFrame({"node_id": g["node_id"], "ts": g["ts"]}, index=g.index)
    f["temp"] = temp
    f["temp_d1"] = temp - temp.shift(1)
    f["temp_d5"] = temp - temp.shift(5)
    f["temp_d15"] = temp - temp.shift(15)
    f["temp_mean15"] = temp.rolling(WINDOW, min_periods=WINDOW).mean()
    f["temp_std15"] = temp.rolling(WINDOW, min_periods=WINDOW).std()
    f["load"] = load
    f["load_d1"] = load - load.shift(1)
    f["load_d5"] = load - load.shift(5)
    f["load_mean15"] = load.rolling(WINDOW, min_periods=WINDOW).mean()
    if "power_w" in optional_present:
        f["power"] = g["power_w"]
        f["power_d5"] = g["power_w"] - g["power_w"].shift(5)
    if "inlet_c" in optional_present:
        f["inlet"] = g["inlet_c"]
        f["inlet_d15"] = g["inlet_c"] - g["inlet_c"].shift(15)
    if "fan_rpm" in optional_present:
        f["fan"] = g["fan_rpm"]
        f["fan_d5"] = g["fan_rpm"] - g["fan_rpm"].shift(5)
    # extended (longer history and load shape)
    for k in (30, 60):
        f[f"temp_d{k}"] = temp - temp.shift(k)
        f[f"temp_std{k}"] = temp.rolling(k, min_periods=k).std()
        f[f"load_d{k}"] = load - load.shift(k)
        f[f"load_mean{k}"] = load.rolling(k, min_periods=k).mean()
    f["temp_dev60"] = temp - temp.rolling(LONG_WINDOW, min_periods=LONG_WINDOW).mean()
    f["load_max15"] = load.rolling(WINDOW, min_periods=WINDOW).max()
    f["load_min15"] = load.rolling(WINDOW, min_periods=WINDOW).min()
    f["min_since_load_step"] = _minutes_since_step(load.to_numpy())
    for h in horizons:
        f[f"future_{h}"] = temp.shift(-h)
        f[f"y_{h}"] = f[f"future_{h}"] - temp
    return f


def build_features(df: pd.DataFrame, optional_present: list[str], horizons: list[int] | None = None,
                   dropna: bool = True, mode: str = "all") -> pd.DataFrame:
    """df must be on a regular 1-minute grid per node (see data.prepare)."""
    horizons = horizons or []
    df = df.sort_values(["node_id", "ts"])
    parts = [_node_features(g, optional_present, horizons) for _, g in df.groupby("node_id", sort=False)]
    out = pd.concat(parts, ignore_index=True)
    if dropna:
        out = out.dropna(subset=feature_columns(optional_present, mode))
    return out


def latest_features(history: pd.DataFrame, optional_present: list[str]) -> pd.DataFrame:
    """Feature row for the newest minute of each node (used live)."""
    f = build_features(history, optional_present, horizons=[], dropna=False)
    last = f.groupby("node_id").tail(1)
    cols = feature_columns(optional_present)
    return last.dropna(subset=cols)


def increment_from_load(delta_load: float, gain_c_per_pct: float, alpha: float, horizon: int) -> float:
    """Extra temperature after `horizon` minutes caused by a step increase of
    `delta_load` percent, for a first-order node with response rate alpha."""
    return float(delta_load * gain_c_per_pct * (1.0 - (1.0 - alpha) ** horizon))



def features_from_buffers(buf: dict, optional_present: list[str]) -> list[float] | None:
    """Fast path used inside the simulator: the same features as
    build_features(), computed from the last 16 minutes held in arrays
    (oldest first). Returns None until enough history exists.
    buf keys: temp, load, and optionally power, inlet, fan."""
    temp, load = buf["temp"], buf["load"]
    if len(temp) < WINDOW + 1:
        return None
    extended = len(temp) >= LONG_WINDOW + 1
    t, l = temp[-WINDOW - 1:], load[-WINDOW - 1:]
    last15 = t[-WINDOW:]
    mean15 = sum(last15) / WINDOW
    var15 = sum((x - mean15) ** 2 for x in last15) / (WINDOW - 1)
    row = [t[-1], t[-1] - t[-2], t[-1] - t[-6], t[-1] - t[0], mean15, var15 ** 0.5,
           l[-1], l[-1] - l[-2], l[-1] - l[-6], sum(l[-WINDOW:]) / WINDOW]
    if "power_w" in optional_present:
        p = buf["power"][-WINDOW - 1:]
        row += [p[-1], p[-1] - p[-6]]
    if "inlet_c" in optional_present:
        i = buf["inlet"][-WINDOW - 1:]
        row += [i[-1], i[-1] - i[0]]
    if "fan_rpm" in optional_present:
        fa = buf["fan"][-WINDOW - 1:]
        row += [fa[-1], fa[-1] - fa[-6]]
    if extended:
        row += _extended_row(list(temp)[-LONG_WINDOW - 1:], list(load)[-LONG_WINDOW - 1:])
    else:
        row += [float("nan")] * len(EXTENDED_FEATURES)
    return row


def _std(x):
    m = sum(x) / len(x)
    return (sum((v - m) ** 2 for v in x) / (len(x) - 1)) ** 0.5


def _extended_row(t, l):
    """EXTENDED_FEATURES from the last 61 minutes (oldest first), same order."""
    since = LONG_WINDOW
    for k in range(1, len(l)):
        if abs(l[-k] - l[-k - 1]) >= 10.0:
            since = k - 1
            break
    l15 = l[-WINDOW:]
    return [t[-1] - t[-31], t[-1] - t[0], _std(t[-30:]), _std(t[-60:]), t[-1] - sum(t[-60:]) / 60,
            l[-1] - l[-31], l[-1] - l[0], sum(l[-30:]) / 30, sum(l[-60:]) / 60, max(l15), min(l15), float(since)]
