"""Placement-policy evaluation in the closed-loop simulator.

Every policy sees the same fleet, the same job stream and the same sensor
noise for a given seed; only the placement choice differs. All policies
apply the same hard constraints (CPU cap, memory, no faulted server), so
the comparison isolates *which* feasible server is chosen.

Policies
  random                pick any feasible server
  least_loaded          lowest current CPU load
  coolest_now           lowest current temperature (reactive)
  predictive_persist    ThermalView ranking with persistence instead of ML (ablation)
  thermalview           ThermalView ranking with the ML forecaster (full system)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from thermalview import simulator as S
from thermalview.recommender import PersistenceForecaster, Workload, eligibility, recommend
from thermalview.thermal_model import NodeParams

POLICIES = ["random", "least_loaded", "coolest_now", "predictive_persist", "thermalview"]
LABELS = {"random": "Random", "least_loaded": "Least loaded", "coolest_now": "Coolest now (reactive)",
          "predictive_persist": "Predictive, persistence (ablation)", "thermalview": "ThermalView (ML + heat model)"}
WARMUP_MIN = 60


def auto_rate(servers: list[S.SimServer], bg_scale: float, wl: dict, load_factor: float) -> float:
    """Arrival rate that brings the mean simulated load to load_factor times the
    fleet's recorded mean load (background is scaled by bg_scale, jobs fill the rest)."""
    rec = float(np.mean([s.bg_load.mean() for s in servers]))
    mean_cpu = (wl["cpu_min"] + wl["cpu_max"]) / 2
    mean_dur = wl["duration_median_min"] * np.exp(wl["duration_sigma"] ** 2 / 2)
    need = max(0.0, load_factor * rec - bg_scale * rec) * len(servers)
    return need / (mean_cpu * mean_dur)


def _choose(policy, sim, job, ctx, rng):
    views = sim.views()
    w = Workload(cpu_pct=job.cpu_pct, mem_gb=job.mem_gb, duration_min=job.duration_min)
    feasible = [v for v in views if not eligibility(v, w, ctx["stale_after"])]
    if not feasible:
        return None
    if policy == "random":
        return feasible[int(rng.integers(len(feasible)))].server_id
    if policy == "least_loaded":
        return min(feasible, key=lambda v: (v.load, v.server_id)).server_id
    if policy == "coolest_now":
        return min(feasible, key=lambda v: (v.temp, v.server_id)).server_id
    forecaster = ctx["ml"] if policy == "thermalview" else ctx["persist"]
    forecasts = forecaster.predict(sim.buffers())
    ranked = recommend(feasible, w, forecasts, ctx["threshold"], ctx["horizon"], ctx["safety"], ctx["stale_after"])
    best = ranked[0]
    return best.server_id if best.status != "ineligible" else None


def run_one(policy: str, seed: int, table: pd.DataFrame, params: dict[str, NodeParams], node_ids: list[str],
            trace_start: pd.Timestamp, cfg: dict, ctx: dict) -> dict:
    sim_cfg, wl = cfg["simulator"], cfg["workload"]
    servers = S.build_servers(table, params, node_ids, trace_start, sim_cfg)
    trace_len = min(len(s.bg_load) for s in servers)
    offset = int(np.random.default_rng(seed).integers(trace_len))
    sim = S.FleetSim(servers, cpu_cap=sim_cfg["cpu_cap_pct"], bg_scale=sim_cfg["background_load_scale"],
                     noise_scale=sim_cfg["noise_scale"], seed=10_000 + seed, trace_offset=offset)
    minutes = int(cfg["evaluation"]["sim_days"] * 1440)
    jobs = S.generate_jobs(minutes, ctx["rate"], wl, seed=20_000 + seed)
    pick_rng = np.random.default_rng(30_000 + seed)
    thr = ctx["threshold"]

    by_minute: dict[int, list[S.Job]] = {}
    for j in jobs:
        by_minute.setdefault(j.arrival_min, []).append(j)
    temps, loads, recs = [], [], []
    rejected = placed = 0
    for m in range(minutes):
        rows = sim.step()
        if m >= WARMUP_MIN:
            temps.append([r["temp"] for r in rows])
            loads.append([r["load"] for r in rows])
            recs.append([r.get("rec_temp", float("nan")) for r in rows])
        if m < S.WINDOW_READY:
            continue
        for job in by_minute.get(m, []):
            sid = _choose(policy, sim, job, ctx, pick_rng)
            if sid is None:
                rejected += 1
            else:
                sim.place(job, sid)
                placed += 1
    T, L, R = np.array(temps), np.array(loads), np.array(recs)
    over = T >= thr
    # placement-caused: over the limit now, but the replayed recording was not
    caused = over & (R < thr)
    caused_excess = np.clip(T - np.maximum(thr, np.nan_to_num(R, nan=-np.inf)), 0, None)
    starts = over & ~np.vstack([np.zeros((1, over.shape[1]), bool), over[:-1]])
    return {
        "policy": policy, "seed": seed, "jobs": placed + rejected, "placed": placed, "rejected": rejected,
        "caused_violation_pct": float(caused.mean() * 100),
        "caused_excess_degree_min": float(caused_excess.sum()),
        "violation_pct": float(over.mean() * 100),
        "recorded_violation_pct": float((R >= thr).mean() * 100),
        "violation_server_min": int(over.sum()),
        "violation_episodes": int(starts.sum()),
        "excess_degree_min": float(np.clip(T - thr, 0, None).sum()),
        "peak_temp": float(T.max()), "p99_temp": float(np.percentile(T, 99)), "mean_temp": float(T.mean()),
        "mean_load": float(L.mean()), "load_imbalance_std": float(L.std(axis=1).mean()),
    }


METRICS = ["caused_violation_pct", "caused_excess_degree_min", "violation_pct", "recorded_violation_pct", "violation_episodes", "excess_degree_min", "peak_temp", "p99_temp",
           "mean_temp", "rejected", "load_imbalance_std"]


def summarise(runs: pd.DataFrame, reference: str = "coolest_now") -> pd.DataFrame:
    """Per-policy summary with paired differences against the reference.
    If runs carry a load_factor column, each level is summarised
    separately (the column is kept in the output)."""
    if "load_factor" in runs.columns:
        parts = []
        for u, g in runs.groupby("load_factor", sort=True):
            part = summarise(g.drop(columns="load_factor"), reference)
            part.insert(0, "load_factor", u)
            parts.append(part)
        return pd.concat(parts, ignore_index=True)
    metrics = METRICS
    out = []
    ref = runs[runs["policy"] == reference].set_index("seed")
    for policy, g in runs.groupby("policy", sort=False):
        g = g.set_index("seed")
        for m in metrics:
            row = {"policy": policy, "label": LABELS[policy], "metric": m,
                   "mean": g[m].mean(), "std": g[m].std(ddof=1), "n_seeds": len(g)}
            if policy != reference:
                d = (g[m] - ref.loc[g.index, m]).to_numpy()
                row["diff_vs_" + reference] = d.mean()
                if len(d) > 1 and np.std(d) > 0:
                    half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
                    row["diff_ci_low"], row["diff_ci_high"] = d.mean() - half, d.mean() + half
                    row["wilcoxon_p"] = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
            out.append(row)
    return pd.DataFrame(out)


def run(table, params, ml_forecaster, cfg, prefer_nodes, trace_start, out_dir: Path, log=print) -> dict:
    sim_cfg, wl = cfg["simulator"], cfg["workload"]
    node_ids = S.pick_nodes(params, int(sim_cfg["n_servers"]), prefer_nodes)
    servers = S.build_servers(table, params, node_ids, trace_start, sim_cfg)
    utils = wl.get("load_factors") or [1.0]
    base = {"ml": ml_forecaster, "persist": PersistenceForecaster(ml_forecaster.horizon),
            "threshold": ml_forecaster.threshold, "horizon": ml_forecaster.horizon,
            "safety": cfg["recommender"]["safety_margin_c"], "stale_after": cfg["recommender"]["stale_after_min"]}
    trace_min = min(len(s.bg_load) for s in servers)
    log(f"fleet: {node_ids}; threshold {base['threshold']} C; background trace {trace_min} min "
        f"(wraps if shorter than {cfg['evaluation']['sim_days']} sim days)")
    runs, rates = [], {}
    for u in utils:
        rate = wl["arrival_rate_per_min"]
        if rate == "auto":
            rate = auto_rate(servers, sim_cfg["background_load_scale"], wl, float(u))
        rates[str(u)] = float(rate)
        ctx = dict(base, rate=float(rate))
        log(f"load {u}x recorded: arrival rate {rate:.3f}/min")
        for seed in cfg["evaluation"]["seeds"]:
            for policy in POLICIES:
                r = run_one(policy, seed, table, params, node_ids, trace_start, cfg, ctx)
                r = {"load_factor": float(u), **r}
                runs.append(r)
                log(f"  u={u} seed {seed:>2} {policy:18s} violations {r['violation_pct']:6.3f}%  "
                    f"peak {r['peak_temp']:.1f}  rejected {r['rejected']}")
    runs_df = pd.DataFrame(runs)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_df.to_csv(out_dir / "placement_runs.csv", index=False)
    summary = summarise(runs_df)
    summary.to_csv(out_dir / "placement_summary.csv", index=False)
    meta = {"fleet_nodes": node_ids, "arrival_rate_per_min": rates, "threshold_c": base["threshold"],
            "horizon_min": base["horizon"], "forecaster": ml_forecaster.name, "background_trace_min": trace_min,
            "data_source": ml_forecaster.data_source, "config": {k: cfg[k] for k in
            ("simulator", "workload", "evaluation", "recommender")}}
    (out_dir / "placement_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return meta


def run_sensitivity(table, params, ml_forecaster, cfg, prefer_nodes, trace_start, t_train_end,
                    out_dir: Path, log=print) -> pd.DataFrame:
    """Robustness of the main comparison to two choices made before evaluation:
    the thermal limit (percentile of training temperatures) and the safety
    margin. One load level, every seed, the reactive reference, the ablation
    and the full system. Only one setting changes at a time."""
    sim_cfg, wl, sc = cfg["simulator"], cfg["workload"], cfg.get("sensitivity", {})
    node_ids = S.pick_nodes(params, int(sim_cfg["n_servers"]), prefer_nodes)
    servers = S.build_servers(table, params, node_ids, trace_start, sim_cfg)
    u = float(sc.get("load_factor", 1.5))
    rate = wl["arrival_rate_per_min"]
    if rate == "auto":
        rate = auto_rate(servers, sim_cfg["background_load_scale"], wl, u)
    train_t = table.loc[table["ts"] < t_train_end, "cpu_temp_c"].dropna()
    base_margin = float(cfg["recommender"]["safety_margin_c"])
    settings = [("limit_percentile", float(p), float(np.round(train_t.quantile(p / 100) * 2) / 2), base_margin)
                for p in sc.get("limit_percentiles", [85, 90, 95, 99])]
    settings += [("safety_margin_c", float(m), float(ml_forecaster.threshold), float(m))
                 for m in sc.get("safety_margins_c", [])]
    policies = sc.get("policies", ["coolest_now", "predictive_persist", "thermalview"])
    rows = []
    for kind, value, thr, margin in settings:
        ctx = {"ml": ml_forecaster, "persist": PersistenceForecaster(ml_forecaster.horizon), "threshold": thr,
               "horizon": ml_forecaster.horizon, "safety": margin, "stale_after": cfg["recommender"]["stale_after_min"],
               "rate": float(rate)}
        for seed in cfg["evaluation"]["seeds"]:
            for policy in policies:
                r = run_one(policy, seed, table, params, node_ids, trace_start, cfg, ctx)
                rows.append({"setting": kind, "value": value, "threshold_c": thr, "safety_margin_c": margin,
                             "load_factor": u, **r})
        log(f"  {kind}={value}: limit {thr} C, margin {margin} C done")
    runs = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(out_dir / "sensitivity_runs.csv", index=False)
    parts = []
    for (kind, value), g in runs.groupby(["setting", "value"], sort=True):
        s = summarise(g.drop(columns=["setting", "value", "threshold_c", "safety_margin_c", "load_factor"]))
        s = s[s.metric.isin(["caused_violation_pct", "caused_excess_degree_min", "rejected"])]
        s.insert(0, "value", value)
        s.insert(0, "setting", kind)
        s.insert(2, "threshold_c", g.threshold_c.iloc[0])
        parts.append(s)
    summary = pd.concat(parts, ignore_index=True)
    summary.to_csv(out_dir / "sensitivity_summary.csv", index=False)
    return summary
