import numpy as np
import pandas as pd
import pytest

from thermalview import simulator as S
from thermalview import thermal_model as TM
from thermalview.recommender import PersistenceForecaster, Workload, recommend


def _first_order(alpha=0.1, base=30.0, gain=0.3, inlet_k=0.5, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    load = np.repeat(rng.uniform(5, 95, n // 60 + 1), 60)[:n]
    inlet = 22 + rng.normal(0, 0.5, n)
    temp = np.empty(n)
    temp[0] = base + gain * load[0] + inlet_k * inlet[0]
    for k in range(1, n):
        # minute-averaged temperature responds to the same minute's load (as on M100)
        temp[k] = temp[k - 1] + alpha * (base + gain * load[k] + inlet_k * inlet[k] - temp[k - 1]) + rng.normal(0, 0.1)
    ts = pd.date_range("2021-03-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({"node_id": "x", "ts": ts, "cpu_temp_c": temp, "load_pct": load, "power_w": np.nan,
                         "inlet_c": inlet, "fan_rpm": np.nan})


def test_heat_model_recovers_known_parameters():
    p = TM.fit_node(_first_order(), 0.005, 0.8)
    assert p is not None
    assert p.alpha == pytest.approx(0.1, rel=0.1)
    assert p.gain == pytest.approx(0.3, rel=0.1)


def test_heat_model_rejects_nonsense_and_falls_back():
    df = _first_order()
    flat = df.copy()
    flat["cpu_temp_c"] = 50.0 + np.random.default_rng(1).normal(0, 0.1, len(df))   # temp unrelated to load
    flat["node_id"] = "flat"
    params = TM.fit_all(pd.concat([df, flat]), df["ts"].max(), 0.005, 0.8)
    assert params["x"].fitted
    assert not params["flat"].fitted


def _sim(seed=1):
    df = _first_order()
    params = {"x": TM.fit_node(df, 0.005, 0.8)}
    cfg = {"memory_gb": 256, "background_memory_frac": 0.3}
    servers = S.build_servers(df.assign(load_pct=20.0, cpu_temp_c=50.0), params, ["x"], df["ts"].min(), cfg)
    return S.FleetSim(servers, cpu_cap=90, bg_scale=1.0, noise_scale=0.0, seed=seed)


def test_placing_load_raises_future_temperature():
    a, b = _sim(), _sim()
    for _ in range(30):
        a.step(); b.step()
    b.place(S.Job(job_id=1, cpu_pct=50, mem_gb=8, duration_min=60, arrival_min=30), 1)
    for _ in range(30):
        ra, rb = a.step(), b.step()
    assert rb[0]["temp"] > ra[0]["temp"] + 5
    assert rb[0]["load"] == pytest.approx(ra[0]["load"] + 50)


def test_jobs_finish_and_fan_fault_heats():
    s = _sim()
    s.place(S.Job(job_id=1, cpu_pct=30, mem_gb=8, duration_min=5, arrival_min=0), 1)
    for _ in range(7):
        s.step()
    assert s.views()[0].n_jobs == 0
    t0 = s.views()[0].temp
    s.servers[1].fan_fault = True
    for _ in range(60):
        s.step()
    assert s.views()[0].temp > t0 + 4


def test_recommender_filters_ranks_and_explains():
    def view(i, temp, free_cpu, fault=False, age=0):
        return S.ServerView(server_id=i, name=f"S{i}", temp=temp, load=90 - free_cpu, power=0, inlet=22,
                            free_cpu=free_cpu, free_mem=100, fan_fault=fault, age_min=age, alpha=0.1, gain=0.3, n_jobs=0)
    views = [view(1, 55, 50), view(2, 45, 50), view(3, 40, 10), view(4, 30, 60, fault=True), view(5, 30, 60, age=9),
             view(6, 58, 50)]
    w = Workload(cpu_pct=20, mem_gb=10)
    fc = {v.server_id: v.temp for v in views}
    ranked = recommend(views, w, fc, threshold=60.0, horizon=5, safety_margin=3.0, stale_after=3)
    assert [c.server_id for c in ranked[:3]] == [2, 1, 6]
    assert ranked[0].status == "recommended"
    assert ranked[2].status == "not_recommended"                     # 58 + increment > 60
    statuses = {c.server_id: c.status for c in ranked}
    assert statuses[3] == statuses[4] == statuses[5] == "ineligible"
    assert "CPU" in next(c for c in ranked if c.server_id == 3).explanation
    assert ranked[0].predicted_temp == pytest.approx(45 + 20 * 0.3 * (1 - 0.9 ** 5))


def test_trace_anchored_replay_and_fidelity():
    """With background unscaled and no jobs, a server replays its recorded
    temperature exactly; the fidelity check beats persistence on a node that
    really follows the model."""
    df = _first_order(alpha=0.5, gain=0.3, n=3000)
    p = TM.fit_node(df, 0.005, 1.0)
    assert p.alpha == pytest.approx(0.5, abs=0.1) and p.gain == pytest.approx(0.3, rel=0.15)
    cfg = {"memory_gb": 256, "background_memory_frac": 0.3}
    sim = S.FleetSim(S.build_servers(df, {"x": p}, ["x"], df["ts"].min(), cfg), cpu_cap=90, bg_scale=1.0,
                     noise_scale=0.0, seed=1)
    temps = [sim.step()[0]["temp"] for _ in range(200)]
    assert np.allclose(temps, df["cpu_temp_c"].to_numpy()[:200])
    fid = TM.fidelity(df, {"x": p}, df["ts"].min(), horizons=(1, 5), every=1)
    assert fid["skill"].min() > 0.15 and fid["skill"].max() > 0.5


def test_persistence_forecaster():
    f = PersistenceForecaster(5)
    assert f.predict({1: {"temp": [40.0, 41.5]}}) == {1: 41.5}


def test_sensitivity_runs_each_setting(tmp_path):
    from thermalview import evaluation
    from thermalview.config import load_config
    from thermalview.data.fixture import make_fixture
    from thermalview.data.prepare import regularise
    table = regularise(make_fixture(n_nodes=4, days=2, seed=5), 1, 2)
    t_train_end = table["ts"].min() + pd.Timedelta(hours=20)
    params = TM.fit_all(table, t_train_end, 0.005, 0.8)
    ml = PersistenceForecaster(5)
    ml.threshold, ml.data_source = 70.0, "dev-fixture"
    cfg = load_config(overrides={"evaluation": {"sim_days": 0.1, "seeds": [1, 2]},
                                 "simulator": {"n_servers": 4},
                                 "sensitivity": {"limit_percentiles": [90, 97.5], "safety_margins_c": [1.0]}})
    out = evaluation.run_sensitivity(table, params, ml, cfg, [], t_train_end, t_train_end, tmp_path, log=lambda *_: None)
    assert set(out.setting) == {"limit_percentile", "safety_margin_c"}
    lim = out[out.setting == "limit_percentile"].groupby("value").threshold_c.first()
    assert lim[90.0] <= lim[97.5]
    assert (tmp_path / "sensitivity_runs.csv").exists()
