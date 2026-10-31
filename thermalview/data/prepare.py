"""Cleaning: regular 1-minute grid per node, short-gap interpolation,
coverage filter and seeded node selection. Also table save/load."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from thermalview import schema

NUMERIC = ["cpu_temp_c", "load_pct", "power_w", "inlet_c", "fan_rpm"]


def _fill_short_gaps(s: pd.Series, max_len: int) -> pd.Series:
    """Interpolate interior gaps of at most max_len points; longer gaps are
    left entirely missing (pandas' own `limit` would half-fill them)."""
    missing = s.isna()
    if not missing.any() or max_len <= 0:
        return s
    run_id = (missing != missing.shift()).cumsum()
    run_len = missing.groupby(run_id).transform("sum")
    filled = s.interpolate(limit_area="inside")
    return filled.where(~missing | (run_len <= max_len))


def regularise(df: pd.DataFrame, resample_minutes: int = 1, max_gap_fill: int = 2) -> pd.DataFrame:
    """Put every node on the same full minute grid. Gaps up to max_gap_fill
    minutes are linearly interpolated; longer gaps stay missing."""
    df = schema.validate(df)
    freq = f"{resample_minutes}min"
    df["ts"] = df["ts"].dt.floor(freq)
    df = df.groupby(["node_id", "ts"], as_index=False)[NUMERIC].mean()
    grid = pd.date_range(df["ts"].min(), df["ts"].max(), freq=freq)
    parts = []
    for node, g in df.groupby("node_id"):
        g = g.set_index("ts").reindex(grid)
        g.index.name = "ts"
        for col in NUMERIC:
            g[col] = _fill_short_gaps(g[col], max_gap_fill)
        g["node_id"] = node
        parts.append(g.reset_index())
    return pd.concat(parts, ignore_index=True)[schema.COLUMNS]


def coverage(df: pd.DataFrame) -> pd.Series:
    ok = df["cpu_temp_c"].notna() & df["load_pct"].notna()
    return ok.groupby(df["node_id"]).mean()


def select_nodes(df: pd.DataFrame, min_coverage: float, n_nodes: int, seed: int) -> tuple[pd.DataFrame, dict]:
    cov = coverage(df)
    eligible = sorted(cov[cov >= min_coverage].index)
    if not eligible:
        raise ValueError(f"no node has coverage >= {min_coverage}")
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(eligible, size=min(n_nodes, len(eligible)), replace=False).tolist())
    info = {"nodes_total": int(len(cov)), "nodes_eligible": len(eligible), "nodes_selected": len(chosen),
            "coverage_selected_mean": float(cov[chosen].mean())}
    return df[df["node_id"].isin(chosen)].reset_index(drop=True), info


def summarise(df: pd.DataFrame) -> dict:
    ok = df.dropna(subset=["cpu_temp_c", "load_pct"])
    return {
        "nodes": int(df["node_id"].nunique()),
        "rows": int(len(df)),
        "valid_rows": int(len(ok)),
        "start": str(df["ts"].min()),
        "end": str(df["ts"].max()),
        "days": round((df["ts"].max() - df["ts"].min()) / pd.Timedelta(days=1), 2),
        "cpu_temp_c": {k: round(float(v), 2) for k, v in ok["cpu_temp_c"].describe().items() if k != "count"},
        "load_pct": {k: round(float(v), 2) for k, v in ok["load_pct"].describe().items() if k != "count"},
        "optional_columns_used": schema.available_optional(df),
    }


def _has_pyarrow() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def save_table(df: pd.DataFrame, directory: Path, name: str, meta: dict | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    if _has_pyarrow():
        path = directory / f"{name}.parquet"
        df.to_parquet(path, index=False)
    else:
        path = directory / f"{name}.csv.gz"
        df.to_csv(path, index=False)
    if meta is not None:
        (directory / f"{name}.meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return path


def load_table(directory: Path, name: str) -> tuple[pd.DataFrame, dict]:
    pq, csv = directory / f"{name}.parquet", directory / f"{name}.csv.gz"
    if pq.exists():
        df = pd.read_parquet(pq)
    elif csv.exists():
        df = pd.read_csv(csv)
    else:
        raise FileNotFoundError(f"no {name} table in {directory}; run `python -m thermalview prepare` first")
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["node_id"] = df["node_id"].astype(str)
    meta_path = directory / f"{name}.meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return df, meta
