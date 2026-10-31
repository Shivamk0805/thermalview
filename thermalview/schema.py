"""The canonical telemetry schema shared by every data source.

Whatever the source (M100 replay, the simulator, or a future Redfish
adapter), telemetry is normalised to one row per node per minute with
these columns and units before anything else touches it.
"""
from __future__ import annotations

import pandas as pd

KEY = ["node_id", "ts"]
REQUIRED = ["cpu_temp_c", "load_pct"]          # forecasting needs these
OPTIONAL = ["power_w", "inlet_c", "fan_rpm"]   # used when present
COLUMNS = KEY + REQUIRED + OPTIONAL

VALID_RANGES = {
    "cpu_temp_c": (5.0, 120.0),
    "load_pct": (0.0, 100.0),
    "power_w": (0.0, 5000.0),
    "inlet_c": (0.0, 60.0),
    "fan_rpm": (0.0, 40000.0),
}


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce types, enforce units/ranges, drop duplicates. Out-of-range
    readings become NaN (treated as missing), they are never clipped."""
    missing = [c for c in KEY + REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"telemetry is missing required columns: {missing}")
    out = df.copy()
    for col in OPTIONAL:
        if col not in out.columns:
            out[col] = float("nan")
    out["node_id"] = out["node_id"].astype(str)
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    for col, (lo, hi) in VALID_RANGES.items():
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
        out.loc[(out[col] < lo) | (out[col] > hi), col] = float("nan")
    out = out.drop_duplicates(subset=KEY, keep="last").sort_values(KEY)
    return out[COLUMNS].reset_index(drop=True)


def available_optional(df: pd.DataFrame, min_coverage: float = 0.5) -> list[str]:
    """Optional columns with enough non-missing data to be used as features."""
    return [c for c in OPTIONAL if c in df.columns and df[c].notna().mean() >= min_coverage]
