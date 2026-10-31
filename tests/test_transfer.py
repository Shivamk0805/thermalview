"""Forecast pipeline end to end on synthetic data: feature-set choice on validation,
calibrated intervals, and the frozen cross-month transfer step."""
import json

import pandas as pd

from thermalview import forecasting, transfer
from thermalview import thermal_model as TM
from thermalview.config import load_config
from thermalview.data.fixture import make_fixture
from thermalview.data.prepare import regularise, save_table


def _cfg(root, sub):
    cfg = load_config(overrides={"forecast": {"max_train_rows": 20_000, "bootstrap_iters": 50,
                                         "models": ["persistence", "ridge"]}})
    cfg.root = root
    cfg["paths"] = {"raw_m100": "raw", "processed": f"{sub}/processed", "artifacts": f"{sub}/artifacts",
                    "results": f"{sub}/results", "database": "tv.db"}
    return cfg


def test_forecast_then_transfer(tmp_path):
    a = _cfg(tmp_path, "a")
    table = regularise(make_fixture(n_nodes=6, days=3, seed=3), 1, 2)
    save_table(table, a.path("processed"), "telemetry", {"source": "dev-fixture"})
    summ = forecasting.run(table, a, {"source": "dev-fixture", "summary": {}}, a.path("results"), a.path("artifacts"),
                           log=lambda *_: None)
    assert set(summ["feature_sets_compared"]) == {"relative", "extended"}
    assert set(map(int, summ["interval_q"])) == {5, 15, 30}
    fm = pd.read_csv(a.path("results") / "forecast_metrics.csv")
    assert {"features", "dm_p", "interval_coverage", "upper_crossing_recall"} <= set(fm.columns)
    cov = fm[(fm.split == "test_seen") & (fm.model != "persistence")].interval_coverage
    assert cov.between(0.75, 0.99).all()               # calibrated on validation, roughly holds on test
    ab = pd.read_csv(a.path("results") / "forecast_feature_ablation.csv")
    assert set(ab.features) == {"relative", "extended"}
    t_val_end = pd.Timestamp(summ["splits"][5]["t_val_end"])
    TM.save_params(TM.fit_all(table, t_val_end, 0.005, 0.8), a.path("artifacts") / "heat_params.json")

    b = _cfg(tmp_path, "b")                             # "another month": different seed, two new nodes
    other = regularise(make_fixture(n_nodes=8, days=2, seed=11), 1, 2)
    save_table(other, b.path("processed"), "telemetry", {"source": "dev-fixture"})
    out = transfer.run(a, b, log=lambda *_: None)
    tm = pd.read_csv(a.path("results") / "transfer_metrics.csv")
    assert set(tm.group) >= {"all", "trained_nodes"}
    assert tm.skill_vs_persistence.notna().all()
    assert out["n_test_nodes"] == 8
    assert (a.path("results") / "transfer_fidelity.csv").exists()
    json.loads((a.path("results") / "transfer_summary.json").read_text())

    import joblib
    from thermalview import jobs
    node = sorted(table.node_id.unique())[0]
    st = pd.DataFrame({"node_id": [node] * 3, "start": [t_val_end + pd.Timedelta(hours=h) for h in (1, 3, 5)]})
    fc = jobs.forecast_near_starts(table, st, joblib.load(a.path("artifacts") / "forecaster.joblib"), t_val_end)
    assert set(fc.minutes) == {"after_job_start", "other_minutes"}
    assert (fc[fc.minutes == "after_job_start"].n > 0).all()


def test_job_event_study_on_synthetic_starts(tmp_path):
    import numpy as np
    from thermalview import jobs
    assert jobs.node_list("[1, 2, 3]") == ["1", "2", "3"]
    assert jobs.node_list("r1n01,r1n02") == ["r1n01", "r1n02"]
    assert jobs.node_list(np.array([5, 6])) == ["5", "6"]
    table = regularise(make_fixture(n_nodes=3, days=2, seed=4), 1, 2)
    nodes = sorted(table.node_id.unique())
    t0 = table.ts.min()
    starts = [t0 + pd.Timedelta(hours=h) for h in (5, 9, 30, 40)]
    # make a real load step at each start so the ids "match"
    for st in starts:
        m = (table.node_id == nodes[0]) & (table.ts >= st) & (table.ts < st + pd.Timedelta(minutes=40))
        table.loc[m, "load_pct"] = np.minimum(100, table.loc[m, "load_pct"] + 40)
    jt = jobs.prepare_jobs(pd.DataFrame({"job_id": range(len(starts)), "start_time": [s.tz_convert(None) for s in starts],
                                         "nodes": [f"[{nodes[0]!r}]"] * len(starts)}))
    st = jobs.starts_per_node(jt, set(nodes), t0, table.ts.max())
    assert len(st) == 4 and set(st.node_id) == {nodes[0]}
    curve = jobs.event_curves(table, st)
    assert curve.loc[curve.minute == 2, "d_load"].iloc[0] > 20
    params = TM.fit_all(table, t0 + pd.Timedelta(hours=30), 0.005, 0.8)
    hm = jobs.heat_model_at_starts(table, st, params)
    assert set(hm.horizon_min) == {5, 15, 30}
