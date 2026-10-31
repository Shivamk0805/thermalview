import numpy as np
import pandas as pd
import pytest

from thermalview import schema
from thermalview.data import m100
from thermalview.data.prepare import regularise, select_nodes


def test_validate_marks_out_of_range_as_missing():
    df = pd.DataFrame({"node_id": ["a", "a"], "ts": ["2021-03-01 00:00", "2021-03-01 00:01"],
                       "cpu_temp_c": [50.0, 400.0], "load_pct": [10.0, -5.0]})
    out = schema.validate(df)
    assert out["cpu_temp_c"].isna().tolist() == [False, True]
    assert out["load_pct"].isna().tolist() == [False, True]
    assert list(out.columns) == schema.COLUMNS


def _frames(metric, node, values, start="2021-03-01 00:00:00"):
    ts = pd.date_range(start, periods=len(values), freq="20s", tz="UTC")
    return pd.DataFrame({"node": node, "ts": ts, "value": values, "metric": metric})


def test_reshape_takes_hottest_core_and_converts_idle_to_load():
    frames = {
        "cpu_temp": [_frames("p0_core0_temp", "n1", [50, 50, 50]), _frames("p0_core1_temp", "n1", [60, 62, 64])],
        "load": [_frames("cpu_idle", "n1", [30, 30, 30])],
    }
    out = m100.reshape_fields(frames, {"cpu_temp": "max", "load": "mean"}, 1, "100_minus")
    assert len(out) == 1
    assert out["cpu_temp_c"].iloc[0] == 62.0       # mean of core1 in the minute, max over cores
    assert out["load_pct"].iloc[0] == 70.0


def test_discover_reads_plugin_and_metric_from_folders(tmp_path):
    p = tmp_path / "year_month=21-03" / "plugin=ipmi_pub" / "metric=ambient"
    p.mkdir(parents=True)
    (p / "part-0.parquet").write_bytes(b"")
    q = tmp_path / "21-03" / "ganglia_pub" / "cpu_idle"
    q.mkdir(parents=True)
    (q / "a.parquet").write_bytes(b"")
    found = {(f.plugin, f.metric) for f in m100.discover(tmp_path)}
    assert found == {("ipmi_pub", "ambient"), ("ganglia_pub", "cpu_idle")}


def test_regularise_fills_short_gaps_only():
    ts = pd.date_range("2021-03-01", periods=20, freq="1min", tz="UTC")
    temp = np.arange(20, dtype=float) + 40
    df = pd.DataFrame({"node_id": "a", "ts": ts, "cpu_temp_c": temp, "load_pct": 50.0})
    df = df.drop(index=[3, 4] + list(range(10, 15)))            # 2-minute gap and 5-minute gap
    out = regularise(df, 1, max_gap_fill=2)
    assert len(out) == 20
    assert out.loc[3, "cpu_temp_c"] == 43.0                     # interpolated
    assert out.loc[10:14, "cpu_temp_c"].isna().all()            # left missing


def test_select_nodes_is_seeded_and_respects_coverage():
    ts = pd.date_range("2021-03-01", periods=100, freq="1min", tz="UTC")
    parts = []
    for i in range(6):
        temp = np.full(100, 50.0)
        if i == 0:
            temp[:60] = np.nan                                   # 40% coverage
        parts.append(pd.DataFrame({"node_id": f"n{i}", "ts": ts, "cpu_temp_c": temp, "load_pct": 10.0}))
    df = regularise(pd.concat(parts))
    a, info = select_nodes(df, 0.85, 3, seed=1)
    b, _ = select_nodes(df, 0.85, 3, seed=1)
    assert info["nodes_eligible"] == 5 and "n0" not in set(a["node_id"])
    assert sorted(a["node_id"].unique()) == sorted(b["node_id"].unique())


def test_load_m100_end_to_end_with_window_and_streaming(tmp_path, monkeypatch=None):
    """Writes a tiny M100-shaped dataset (pickles stand in for Parquet so the
    test needs no pyarrow) and runs the real two-pass loader on it."""
    rng = np.random.default_rng(0)
    ts = pd.date_range("2021-03-01", "2021-03-08", freq="20s", tz="UTC", inclusive="left")
    nodes = [str(n) for n in range(10, 16)]

    def write(plugin, metric, fn):
        rows = []
        for n in nodes:
            keep = ts if n != "15" else ts[ts < "2021-03-02"]          # node 15 mostly offline
            rows.append(pd.DataFrame({"timestamp": keep, "value": fn(len(keep)), "node": n}))
        df = pd.concat(rows)
        df = df[(df["timestamp"] < "2021-03-06") | (df["node"] == "10")]  # outage for most nodes after 03-06
        d = tmp_path / plugin / metric
        d.mkdir(parents=True)
        df.to_pickle(d / "part.parquet")

    for c in range(3):
        write("ipmi_pub", f"p0_core{c}_temp", lambda k, c=c: 50 + c + rng.normal(0, 0.5, k))
    write("ipmi_pub", "ambient", lambda k: 21 + rng.normal(0, 0.1, k))
    write("ganglia_pub", "cpu_idle", lambda k: np.full(k, 40.0))
    cfg = {"plugin": "ipmi_pub", "load_plugin": "ganglia_pub", "load_transform": "100_minus",
           "cpu_temp": {"pattern": "^p[0-9]+_core[0-9]+_temp$", "agg": "max"},
           "inlet": {"pattern": "^ambient$", "agg": "mean"},
           "load": {"pattern": "^cpu_idle$", "agg": "mean"}}
    original = pd.read_parquet
    pd.read_parquet = pd.read_pickle
    try:
        out = m100.load_m100(tmp_path, cfg, 1, n_nodes=10, min_coverage=0.85, seed=1, log=lambda *_: None)
        few = m100.load_m100(tmp_path, cfg, 1, n_nodes=3, min_coverage=0.85, seed=1, log=lambda *_: None,
                             prefer_nodes=["12", "14", "15"])     # 15 is not eligible
    finally:
        pd.read_parquet = original
    assert out["ts"].min() >= pd.Timestamp("2021-03-01", tz="UTC")
    assert out["ts"].max() < pd.Timestamp("2021-03-06", tz="UTC")       # densest window only
    assert "15" not in set(out["node_id"])                               # low coverage dropped
    assert set(out["node_id"]) == {"10", "11", "12", "13", "14"}
    assert {"12", "14"} <= set(few["node_id"]) and len(set(few["node_id"])) == 3 and "15" not in set(few["node_id"])
    assert out["load_pct"].round(6).eq(60.0).all()
    assert out["cpu_temp_c"].mean() > 51.5                               # max over cores, not mean


def test_densest_window_breaks_on_calendar_gap_and_trims():
    """Real M100 March: data on 03-01..03-03 (last day partial) and again on
    03-30. The gap must end the run, and the window must stop at the last
    reading rather than at midnight four days later."""
    ts = list(pd.date_range("2021-03-01", "2021-03-03 10:00", freq="1min", tz="UTC"))
    ts += list(pd.date_range("2021-03-30", "2021-03-31", freq="1min", tz="UTC", inclusive="left"))
    first = pd.DataFrame({"ts": ts * 2, "node": ["a"] * len(ts) + ["b"] * len(ts), "value": 50.0})
    t0, t1, per_day = m100.densest_window(first)
    assert t0 == pd.Timestamp("2021-03-01", tz="UTC")
    assert t1 == pd.Timestamp("2021-03-03 10:01", tz="UTC")
    assert len(per_day) == 4


def test_load_m100_real_parquet_streaming(tmp_path):
    """Same as above but with genuine Parquet files, small batches and two
    months side by side: exercises the pyarrow streaming path (runs where
    pyarrow is installed, e.g. the project laptop)."""
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    rng = np.random.default_rng(1)
    nodes = [10, 11, 12, 13]

    def write(month, plugin, metric, start, fn):
        ts = pd.date_range(start, periods=3 * 1440 * 3, freq="20s", tz="UTC")
        df = pd.concat([pd.DataFrame({"timestamp": ts, "value": fn(len(ts)), "node": n}) for n in nodes])
        d = tmp_path / f"year_month={month}" / f"plugin={plugin}" / f"metric={metric}"
        d.mkdir(parents=True)
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), d / "a_0.parquet", row_group_size=5000)

    for month, start, base in (("21-02", "2021-02-01", 50.0), ("21-03", "2021-03-01", 90.0)):
        for c in range(2):
            write(month, "ipmi_pub", f"p0_core{c}_temp", start, lambda k, c=c, b=base: b + c + rng.normal(0, 0.3, k))
        write(month, "ipmi_pub", "total_power", start, lambda k: 400 + rng.normal(0, 5, k))
        write(month, "ganglia_pub", "cpu_idle", start, lambda k: np.full(k, 70.0))
    cfg = {"plugin": "ipmi_pub", "load_plugin": "ganglia_pub", "load_transform": "100_minus", "year_month": "21-02",
           "cpu_temp": {"pattern": "^p[0-9]+_core[0-9]+_temp$", "agg": "max"},
           "power": {"pattern": "^total_power$", "agg": "mean"},
           "load": {"pattern": "^cpu_idle$", "agg": "mean"}}
    old = m100.BATCH_ROWS
    m100.BATCH_ROWS = 7000                      # force many batches
    try:
        out = m100.load_m100(tmp_path, cfg, 1, n_nodes=3, min_coverage=0.8, seed=1, log=lambda *_: None)
    finally:
        m100.BATCH_ROWS = old
    assert out["ts"].min() >= pd.Timestamp("2021-02-01", tz="UTC") and out["ts"].max() < pd.Timestamp("2021-03-01", tz="UTC")
    assert out["node_id"].nunique() == 3 and set(out["node_id"]) <= {"10", "11", "12", "13"}
    assert 50.5 < out["cpu_temp_c"].mean() < 52                  # Feb only, max over cores
    assert out["load_pct"].round(6).eq(30.0).all()
    assert out.groupby("node_id")["ts"].nunique().min() > 0.95 * 3 * 1440
