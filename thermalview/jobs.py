"""Real job starts from the Slurm job table: an event study.

The simulator assumes a placed job heats its node by gain x load change with
response rate alpha. The M100 job table records when real jobs started on which
nodes, so the assumption can be checked on events the model never chose:

  * the average load and temperature around a real job start (minute -15 to +30);
  * the heat model's error after a job start, given the node's real load, against
    persistence (the same check as Table 2, restricted to job starts);
  * the deployed forecaster's error in the 15 minutes after a job start versus all
    other minutes, which shows how much of the remaining error is the scheduler's
    knowledge that the node's own history cannot contain.

  python -m thermalview jobs          (uses [m100].year_month and the evaluation artifacts)

Node ids: the job table lists nodes per job. The step reports how many listed
nodes match telemetry node ids and the average load step at a start; a load step
near zero means the ids do not correspond and the numbers must not be used.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from thermalview import features as F
from thermalview import thermal_model as TM
from thermalview.config import Config

PRE, POST = 15, 30


def find_job_table(raw_root: Path, year_month: str | None) -> list[Path]:
    out = []
    for p in sorted(Path(raw_root).rglob("*.parquet")):
        parts = [x.split("=", 1)[-1] for x in p.relative_to(raw_root).parts]
        if "job_table" in parts and (not year_month or year_month in parts):
            out.append(p)
    return out


def _to_utc(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        unit = "ms" if s.dropna().median() > 1e11 else "s"
        return pd.to_datetime(s, unit=unit, utc=True)
    t = pd.to_datetime(s, utc=True, errors="coerce")
    return t


def node_list(v) -> list[str]:
    """The job table's node field as a list of node-id strings, whatever its encoding."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return []
    if isinstance(v, (list, tuple, np.ndarray)):
        return [str(x).strip() for x in v if str(x).strip()]
    s = str(v).strip()
    if s.startswith("["):
        try:
            return [str(x).strip() for x in ast.literal_eval(s)]
        except (ValueError, SyntaxError):
            s = s.strip("[]")
    return [x.strip().strip("'\"") for x in s.replace(";", ",").split(",") if x.strip()]


def load_jobs(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        try:
            import pyarrow.parquet as pq
            have = set(pq.read_schema(p).names)
        except ImportError:                       # pandas fallback reads everything
            have = None
        cols = [c for c in ("job_id", "start_time", "end_time", "nodes", "num_nodes", "num_cpus", "job_state",
                            "partition") if have is None or c in have]
        frames.append(pd.read_parquet(p, columns=cols if have is not None else None))
    return prepare_jobs(pd.concat(frames, ignore_index=True))


def prepare_jobs(j: pd.DataFrame) -> pd.DataFrame:
    j = j.copy()
    j["start_time"] = _to_utc(j["start_time"])
    if "end_time" in j:
        j["end_time"] = _to_utc(j["end_time"])
    j["node_list"] = j["nodes"].map(node_list)
    return j.dropna(subset=["start_time"]).drop_duplicates(subset=["job_id"] if "job_id" in j else None)


def starts_per_node(jobs: pd.DataFrame, node_ids: set[str], t0, t1) -> pd.DataFrame:
    rows = []
    for r in jobs.itertuples():
        if not (t0 <= r.start_time <= t1):
            continue
        for n in r.node_list:
            if n in node_ids:
                rows.append({"node_id": n, "start": r.start_time.floor("min"), "job_id": getattr(r, "job_id", None),
                             "num_nodes": getattr(r, "num_nodes", np.nan)})
    return pd.DataFrame(rows, columns=["node_id", "start", "job_id", "num_nodes"]).drop_duplicates(["node_id", "start"])


def _windows(table: pd.DataFrame, starts: pd.DataFrame, before: int, after: int):
    """For every start, the node's temperature and load from `before` minutes before to `after`
    minutes after, as arrays (n_starts x width), NaN where data is missing. Vectorised per node."""
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    width = before + after + 1
    Ts, Ls, keep = [], [], []
    for node, st in starts.groupby("node_id", sort=False):
        g = table[table["node_id"] == node]
        if g.empty:
            continue
        m = ((pd.to_datetime(g["ts"], utc=True) - epoch) // pd.Timedelta(minutes=1)).to_numpy()
        m0 = m.min()
        T = np.full(m.max() - m0 + 1 + width * 2, np.nan)
        L = T.copy()
        T[m - m0 + width], L[m - m0 + width] = g["cpu_temp_c"].to_numpy(), g["load_pct"].to_numpy()
        s0 = ((pd.to_datetime(st["start"], utc=True) - epoch) // pd.Timedelta(minutes=1)).to_numpy() - m0 + width
        ok = (s0 - before >= 0) & (s0 + after < len(T))
        cols = (s0[ok] - before)[:, None] + np.arange(width)[None, :]
        Ts.append(T[cols])
        Ls.append(L[cols])
        keep.append(st.index[ok])
    if not Ts:
        return np.empty((0, width)), np.empty((0, width))
    return np.vstack(Ts), np.vstack(Ls)


def event_curves(table: pd.DataFrame, starts: pd.DataFrame) -> pd.DataFrame:
    """Mean change in temperature and load from the minute before a start, minute -PRE..+POST."""
    T, L = _windows(table, starts, PRE, POST)
    ok = (np.isnan(T).mean(axis=1) <= 0.2) & np.isfinite(T[:, PRE - 1]) & np.isfinite(L[:, PRE - 1])
    dT = T[ok] - T[ok, PRE - 1][:, None]
    dL = L[ok] - L[ok, PRE - 1][:, None]
    rows = []
    for j, k in enumerate(range(-PRE, POST + 1)):
        n = int(np.isfinite(dT[:, j]).sum())
        if n:
            rows.append({"minute": k, "n": n, "d_temp": float(np.nanmean(dT[:, j])), "d_load": float(np.nanmean(dL[:, j])),
                         "d_temp_se": float(np.nanstd(dT[:, j], ddof=1) / np.sqrt(max(1, n)))})
    return pd.DataFrame(rows)


def heat_model_at_starts(table: pd.DataFrame, starts: pd.DataFrame, params: dict, horizons=(5, 15, 30)) -> pd.DataFrame:
    """Heat model given the real load from the minute before a start, versus persistence."""
    hmax = max(horizons)
    rows = []
    for node, st in starts.groupby("node_id", sort=False):
        p = params.get(node)
        if p is None:
            continue
        T, L = _windows(table, st, 1, hmax - 1)          # minute -1 .. hmax-1  (hmax+1 columns)
        ok = np.isfinite(T).all(axis=1) & np.isfinite(L).all(axis=1)
        if not ok.any():
            continue
        T, L = T[ok], L[ok]
        pred = T[:, :1] + p.gain * TM.filtered_change(L, p.alpha)
        for h in horizons:
            for i in range(len(T)):
                rows.append({"horizon_min": h, "sim_abs_err": abs(pred[i, h] - T[i, h]),
                             "persistence_abs_err": abs(T[i, 0] - T[i, h]), "load_step": L[i, min(h, 5)] - L[i, 0],
                             "actual_change": T[i, h] - T[i, 0], "model_change": pred[i, h] - T[i, 0]})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    out = df.groupby("horizon_min").agg(n=("sim_abs_err", "size"), sim_mae=("sim_abs_err", "mean"),
                                        persistence_mae=("persistence_abs_err", "mean"),
                                        mean_load_step=("load_step", "mean"),
                                        mean_actual_change=("actual_change", "mean"),
                                        mean_model_change=("model_change", "mean")).reset_index()
    out["skill"] = 1 - out["sim_mae"] / out["persistence_mae"]
    return out


def forecast_near_starts(table: pd.DataFrame, starts: pd.DataFrame, bundle: dict, t_from) -> pd.DataFrame:
    """Deployed forecaster: error on minutes within 15 min after a job start vs all other minutes (test period)."""
    optional = bundle["optional_present"]
    horizons = sorted(int(h) for h in bundle["models"])
    feat = F.build_features(table, optional, horizons, dropna=False)
    feat = feat[feat["ts"] >= t_from]
    cols_by_h = {int(k): v for k, v in bundle.get("feature_columns_by_h", {}).items()}
    near = np.zeros(len(feat), bool)
    key = feat["node_id"].astype(str).to_numpy()
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    as_min = lambda s_: ((pd.to_datetime(s_, utc=True) - epoch) // pd.Timedelta(minutes=1)).to_numpy()  # noqa: E731
    ts = as_min(feat["ts"])
    by_node = {n: np.sort(as_min(g["start"])) for n, g in starts.groupby("node_id")}
    for n, st in by_node.items():
        m = key == n
        if not m.any():
            continue
        t = ts[m]
        pos = np.searchsorted(st, t, side="right") - 1
        since = np.where(pos >= 0, t - st[np.clip(pos, 0, None)], np.inf)
        near[np.where(m)[0]] = (since >= 0) & (since < 15)
    rows = []
    for h in horizons:
        cols = cols_by_h.get(h, bundle["feature_columns"])
        d = feat.dropna(subset=cols + [f"y_{h}"])
        nr = near[feat.index.get_indexer(d.index)]
        pred = bundle["models"][h].predict(d[cols])
        err = np.abs(pred - d[f"y_{h}"].to_numpy())
        per = np.abs(d[f"y_{h}"].to_numpy())
        for name, m in (("after_job_start", nr), ("other_minutes", ~nr)):
            if m.sum():
                rows.append({"horizon_min": h, "minutes": name, "n": int(m.sum()), "model_mae": float(err[m].mean()),
                             "persistence_mae": float(per[m].mean()), "skill": float(1 - err[m].mean() / per[m].mean())})
    return pd.DataFrame(rows)


def run(cfg: Config, test_cfg: Config | None = None, log=print) -> dict:
    """Event study on the evaluation month (cfg), or, with test_cfg, on another month
    using the evaluation month's frozen models (every start is then out of sample)."""
    from thermalview.data.prepare import load_table
    res, art = cfg.path("results"), cfg.path("artifacts")
    data_cfg = test_cfg or cfg
    tag = "" if test_cfg is None else "_transfer"
    paths = find_job_table(data_cfg.path("raw_m100"), data_cfg["m100"].get("year_month"))
    if not paths:
        raise FileNotFoundError("no job table found; run: python scripts\\fetch_m100.py <month> --plugin job_table")
    jobs = load_jobs(paths)
    log(f"job table: {len(jobs)} jobs from {', '.join(str(p) for p in paths)}")
    log(f"  example node field: {jobs['nodes'].iloc[0]!r} -> {jobs['node_list'].iloc[0][:5]}")
    log(f"  job start times {jobs['start_time'].min()} to {jobs['start_time'].max()}")
    table, _ = load_table(data_cfg.path("processed"), "telemetry")
    table["node_id"] = table["node_id"].astype(str)
    node_ids = set(table["node_id"].unique())
    all_listed = {n for lst in jobs["node_list"] for n in lst}
    starts = starts_per_node(jobs, node_ids, table["ts"].min() + pd.Timedelta(minutes=PRE),
                             table["ts"].max() - pd.Timedelta(minutes=POST))
    log(f"  telemetry {table['ts'].min()} to {table['ts'].max()}; {len(all_listed)} distinct nodes in the job "
        f"table, {len(all_listed & node_ids)} of our {len(node_ids)} telemetry nodes appear; {len(starts)} job "
        f"starts on our nodes in the window")
    curves = event_curves(table, starts)
    curves.to_csv(res / f"job_event_curve{tag}.csv", index=False)
    val = lambda col, m: float(curves.loc[curves.minute == m, col].iloc[0]) if (curves.minute == m).any() else float("nan")  # noqa: E731
    step, rise = val("d_load", 2), val("d_temp", 15)
    log(f"  mean load change 2 min after a start {step:+.1f} points; temperature after 15 min {rise:+.2f} C")
    params = TM.load_params(art / "heat_params.json")
    fitted = [p for p in params.values() if p.fitted]
    med = {k: float(np.median([getattr(p, k) for p in fitted]))
           for k in ("alpha", "base", "gain", "inlet_k", "sigma", "power_0", "power_1", "fan_0", "fan_1", "inlet_mean")}
    params = {n: params.get(n) or TM.NodeParams(node_id=n, fitted=False, n_rows=0, **med) for n in node_ids}
    if test_cfg is None:          # evaluation month: only starts after the fitting/training periods are out of sample
        summ = json.loads((res / "forecast_summary.json").read_text())
        t_from = pd.Timestamp(summ["splits"][str(summ["config"]["primary_horizon"])]["t_val_end"])
    else:
        t_from = table["ts"].min()
    oos = starts[starts["start"] >= t_from]
    hm_all = heat_model_at_starts(table, starts, params)
    hm_oos = heat_model_at_starts(table, oos, params)
    hm_all.to_csv(res / f"job_event_heat_model_all{tag}.csv", index=False)
    hm_oos.to_csv(res / f"job_event_heat_model{tag}.csv", index=False)
    fc = forecast_near_starts(table, oos, joblib.load(art / "forecaster.joblib"), t_from)
    fc.to_csv(res / f"job_event_forecast{tag}.csv", index=False)
    out = {"n_jobs": int(len(jobs)), "n_listed_nodes": len(all_listed), "n_matched_nodes": len(all_listed & node_ids),
           "n_starts": int(len(starts)), "n_out_of_sample_starts": int(len(oos)),
           "out_of_sample_from": str(t_from), "job_start_min": str(jobs["start_time"].min()),
           "job_start_max": str(jobs["start_time"].max()), "telemetry_start": str(table["ts"].min()),
           "telemetry_end": str(table["ts"].max()),
           "mean_load_step_2min": step, "mean_temp_rise_15min": rise, "temp_minus15_vs_minus1": val("d_temp", -PRE),
           "ids_look_valid": bool(len(all_listed & node_ids) > 0 and step > 2.0)}
    (res / f"job_event_summary{tag}.json").write_text(json.dumps(out, indent=2))
    if not out["ids_look_valid"]:
        log("  WARNING: job-table node ids do not seem to match telemetry ids (no load step at job starts); "
            "do not use these numbers")
    log("heat model at job starts (all):\n" + hm_all.round(3).to_string(index=False))
    log("heat model at out-of-sample job starts:\n" + hm_oos.round(3).to_string(index=False))
    log("forecaster after job starts vs other minutes:\n" + fc.round(3).to_string(index=False))
    return out
