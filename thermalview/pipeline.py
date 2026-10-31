"""Pipeline steps behind the command line (python -m thermalview <step>)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from thermalview import evaluation, forecasting
from thermalview import thermal_model as TM
from thermalview.config import Config
from thermalview.data import m100
from thermalview.data.fixture import make_fixture
from thermalview.data.prepare import load_table, regularise, save_table, select_nodes, summarise
from thermalview.recommender import MLForecaster

TABLE = "telemetry"


def step_probe(cfg: Config, log=print) -> Path:
    report = m100.probe(cfg.path("raw_m100"), cfg["m100"])
    out = cfg.path("results") / "m100_probe.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    log(report)
    log(f"\nreport saved to {out}")
    return out


def step_prepare(cfg: Config, dev: bool = False, log=print) -> dict:
    pc = cfg["prepare"]
    if dev:
        log("DEV MODE: generating synthetic fixture data (never report these numbers)")
        raw = make_fixture(n_nodes=16, days=10, seed=pc["seed"])
        source = "dev-fixture"
    else:
        prefer = None
        if pc.get("include_nodes_from"):    # keep another month's nodes (cross-month test), then fill at random
            d = Path(pc["include_nodes_from"])
            prev, _ = load_table(d if d.is_absolute() else cfg.root / d, TABLE)
            prefer = sorted(prev["node_id"].astype(str).unique())
        raw = m100.load_m100(cfg.path("raw_m100"), cfg["m100"], pc["resample_minutes"], pc["n_nodes"],
                             pc["min_coverage"], pc["seed"], log=log, prefer_nodes=prefer)
        source = "M100"
    table = regularise(raw, pc["resample_minutes"], pc["max_gap_fill_minutes"])
    table, node_info = select_nodes(table, pc["min_coverage"], pc["n_nodes"], pc["seed"])
    node_info["node_ids"] = sorted(table["node_id"].astype(str).unique())
    meta = {"source": source, "selection": node_info, "summary": summarise(table), "config": pc}
    ok = table.dropna(subset=["cpu_temp_c", "load_pct"])
    corr = float(ok.groupby("node_id")[["cpu_temp_c", "load_pct"]].corr().xs("load_pct", level=1)["cpu_temp_c"].median())
    meta["load_temp_corr_median"] = round(corr, 3)
    log(f"median per-node corr(load, temp) = {corr:.2f}")
    if corr < 0:
        log("WARNING: load and temperature are negatively correlated - load is probably inverted. "
            "Check [m100].load_transform before trusting any result.")
    path = save_table(table, cfg.path("processed"), TABLE, meta)
    log(json.dumps(meta["summary"], indent=2))
    log(f"saved {path}")
    return meta


def _check_source(meta: dict, log) -> None:
    src = meta.get("source", "unknown")
    if src != "M100":
        log("=" * 70 + f"\nWARNING: processed data source is '{src}', not M100. "
            "These results must not be reported.\n" + "=" * 70)


def step_forecast(cfg: Config, log=print) -> dict:
    table, meta = load_table(cfg.path("processed"), TABLE)
    _check_source(meta, log)
    return forecasting.run(table, cfg, meta, cfg.path("results"), cfg.path("artifacts"), log=log)


def _forecast_summary(cfg: Config) -> dict:
    p = cfg.path("results") / "forecast_summary.json"
    if not p.exists():
        raise FileNotFoundError("run `python -m thermalview forecast` first")
    return json.loads(p.read_text())


def step_thermal(cfg: Config, log=print) -> pd.DataFrame:
    table, _meta = load_table(cfg.path("processed"), TABLE)
    summ = _forecast_summary(cfg)
    split = summ["splits"][str(summ["config"]["primary_horizon"])]
    t_val_end = pd.Timestamp(split["t_val_end"])
    tm = cfg["thermal_model"]
    params = TM.fit_all(table, t_val_end, tm["min_alpha"], tm["max_alpha"])
    TM.save_params(params, cfg.path("artifacts") / "heat_params.json")
    n_fit = sum(p.fitted for p in params.values())
    log(f"heat model: {n_fit}/{len(params)} nodes fitted, others use fleet median")
    fid = TM.fidelity(table, params, t_val_end,
                      horizons=tuple(cfg["evaluation"].get("fidelity_horizons", [5, 15, 30])))
    fid.to_csv(cfg.path("results") / "simulator_fidelity.csv", index=False)
    pd.DataFrame([vars(p) for p in params.values()]).to_csv(cfg.path("results") / "heat_params.csv", index=False)
    log(fid.round(3).to_string(index=False))
    return fid


def step_placement(cfg: Config, log=print) -> dict:
    table, _meta = load_table(cfg.path("processed"), TABLE)
    summ = _forecast_summary(cfg)
    split = summ["splits"][str(summ["config"]["primary_horizon"])]
    params = TM.load_params(cfg.path("artifacts") / "heat_params.json")
    ml = MLForecaster(cfg.path("artifacts") / "forecaster.joblib")
    return evaluation.run(table, params, ml, cfg, split["holdout_nodes"], pd.Timestamp(split["t_val_end"]),
                          cfg.path("results"), log=log)


def step_sensitivity(cfg: Config, log=print) -> pd.DataFrame:
    table, _meta = load_table(cfg.path("processed"), TABLE)
    summ = _forecast_summary(cfg)
    split = summ["splits"][str(summ["config"]["primary_horizon"])]
    params = TM.load_params(cfg.path("artifacts") / "heat_params.json")
    ml = MLForecaster(cfg.path("artifacts") / "forecaster.joblib")
    return evaluation.run_sensitivity(table, params, ml, cfg, split["holdout_nodes"], pd.Timestamp(split["t_val_end"]),
                                      pd.Timestamp(split["t_train_end"]), cfg.path("results"), log=log)


def step_figures(cfg: Config, log=print) -> list[Path]:
    from thermalview import figures
    return figures.make_all(cfg.path("results"), log=log)


def step_all(cfg: Config, dev: bool = False, log=print) -> None:
    step_prepare(cfg, dev=dev, log=log)
    step_forecast(cfg, log=log)
    step_thermal(cfg, log=log)
    step_placement(cfg, log=log)
    step_sensitivity(cfg, log=log)
    step_figures(cfg, log=log)
