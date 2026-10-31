import numpy as np
import pandas as pd

from thermalview import features as F
from thermalview import forecasting
from thermalview.data.fixture import make_fixture
from thermalview.data.prepare import regularise


def _table():
    return regularise(make_fixture(n_nodes=6, days=3, seed=3))


def test_buffer_features_match_dataframe_features():
    t = _table()
    node = t[t["node_id"] == "dev001"].iloc[:400]
    opt = ["power_w", "inlet_c", "fan_rpm"]
    full = F.build_features(node, opt, horizons=[], dropna=True).iloc[-1]
    g = node.iloc[: full.name + 1] if full.name < len(node) else node
    g = node[node["ts"] <= full["ts"]]
    buf = {"temp": g["cpu_temp_c"].tolist(), "load": g["load_pct"].tolist(), "power": g["power_w"].tolist(),
           "inlet": g["inlet_c"].tolist(), "fan": g["fan_rpm"].tolist()}
    fast = F.features_from_buffers(buf, opt)
    np.testing.assert_allclose(fast, full[F.fast_columns(opt)].to_numpy(dtype=float), rtol=1e-9, atol=1e-9)


def test_extended_features_match_and_use_only_the_past():
    t = _table()
    node = t[t["node_id"] == "dev001"].iloc[:600].copy()
    node.loc[node.index[500], "load_pct"] += 40          # a clear load step 99 rows before the end
    opt = ["power_w"]
    full = F.build_features(node, opt, horizons=[], dropna=True, mode="extended")
    for i in (-1, -60, -95, -100):
        row = full.iloc[i]
        g = node[node["ts"] <= row["ts"]]
        buf = {"temp": g["cpu_temp_c"].tolist(), "load": g["load_pct"].tolist(), "power": g["power_w"].tolist()}
        np.testing.assert_allclose(F.features_from_buffers(buf, opt), row[F.fast_columns(opt)].to_numpy(dtype=float),
                                   rtol=1e-9, atol=1e-9)
    assert set(F.EXTENDED_FEATURES) <= set(F.feature_columns(opt, "extended"))
    assert not (F.LEVEL_FEATURES & set(F.feature_columns(opt, "extended")))


def test_features_use_only_the_past():
    t = _table()
    a = F.build_features(t, ["power_w"], horizons=[5])
    t2 = t.copy()
    cutoff = t2["ts"].min() + pd.Timedelta(hours=10)
    t2.loc[t2["ts"] > cutoff, ["cpu_temp_c", "load_pct", "power_w"]] += 30.0   # change the future only
    b = F.build_features(t2, ["power_w"], horizons=[5])
    cols = F.feature_columns(["power_w"])
    ka = a[a["ts"] <= cutoff].set_index(["node_id", "ts"])[cols]
    kb = b[b["ts"] <= cutoff].set_index(["node_id", "ts"])[cols]
    pd.testing.assert_frame_equal(ka, kb.loc[ka.index])


def test_split_has_no_overlap_or_leakage():
    feat = F.build_features(_table(), [], horizons=[15])
    sp = forecasting.make_split(feat, 15, 0.6, 0.2, 0.2, seed=7)
    h = pd.Timedelta(minutes=15)
    assert (sp.train["ts"] + h <= sp.t_train_end).all()
    assert sp.val["ts"].min() >= sp.t_train_end + pd.Timedelta(minutes=F.WINDOW)
    assert (sp.val["ts"] + h <= sp.t_val_end).all()
    assert sp.test_seen["ts"].min() >= sp.t_val_end + pd.Timedelta(minutes=F.WINDOW)
    assert not set(sp.holdout_nodes) & set(sp.train["node_id"])
    assert set(sp.test_unseen["node_id"]) <= set(sp.holdout_nodes)


def test_persistence_has_zero_skill_and_metrics_are_consistent():
    feat = F.build_features(_table(), [], horizons=[5]).dropna(subset=["y_5"])
    m = forecasting.metrics(feat, np.zeros(len(feat)), 5, threshold=60.0, event_rise=2.0)
    assert abs(m["skill_vs_persistence"]) < 1e-12
    assert m["mae"] == m["persistence_mae"]
    assert m["rmse"] >= m["mae"]
