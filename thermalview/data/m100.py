"""Reader for the M100 ExaData dataset (CINECA Marconi100, CC BY 4.0).

Dataset: Borghesi et al., "M100 ExaData: a data collection campaign on the
CINECA's Marconi100 Tier-0 supercomputer", Scientific Data 10, 2023.
Download: https://zenodo.org/records/7588815 and https://zenodo.org/records/7589131

Each monthly tarball is a partitioned Parquet dataset laid out as
  <month>/<plugin>/<metric>/...parquet   (possibly as key=value folders).
Exact column names inside the files are detected at run time, because they
are not documented in the Zenodo records.

The reader is split in two:
  * file discovery and Parquet reading (needs pyarrow; runs on your laptop)
  * reshape_fields(), a pure-pandas step that is unit tested.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TS_CANDIDATES = ["timestamp", "time", "ts", "date", "datetime"]
VALUE_CANDIDATES = ["value", "val", "values", "reading"]
NODE_CANDIDATES = ["node", "hostname", "host", "node_id", "nodename", "name"]

FIELD_TO_COLUMN = {
    "cpu_temp": "cpu_temp_c",
    "load": "load_pct",
    "power": "power_w",
    "inlet": "inlet_c",
    "fan": "fan_rpm",
}


@dataclass
class MetricFile:
    plugin: str
    metric: str
    path: Path


def _clean(part: str) -> str:
    return part.split("=", 1)[1] if "=" in part else part


def discover(root: Path, year_month: str | None = None) -> list[MetricFile]:
    """Find every Parquet file and infer (plugin, metric) from its folders.
    With year_month (e.g. "21-02"), only files under that month's folder are used,
    so several extracted months can sit side by side in data/raw/m100."""
    files = []
    for p in sorted(Path(root).rglob("*.parquet")):
        raw_parts = p.relative_to(root).parts[:-1]
        if year_month and not any(_clean(x) == year_month for x in raw_parts):
            continue
        parts = [_clean(x) for x in raw_parts]
        if len(parts) >= 2:
            files.append(MetricFile(plugin=parts[-2], metric=parts[-1], path=p))
        elif len(parts) == 1:
            files.append(MetricFile(plugin=parts[0], metric=p.stem, path=p))
    return files


def _pick(columns, candidates, what):
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    raise KeyError(f"cannot find a {what} column among {list(columns)}")


def standardise(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return columns node, ts, value, metric from an M100 metric frame."""
    ts = _pick(df.columns, TS_CANDIDATES, "timestamp")
    val = _pick(df.columns, VALUE_CANDIDATES, "value")
    node = _pick(df.columns, NODE_CANDIDATES, "node")
    out = pd.DataFrame({
        "node": df[node].astype(str),
        "ts": pd.to_datetime(df[ts], utc=True, errors="coerce"),
        "value": pd.to_numeric(df[val], errors="coerce"),
    })
    out["metric"] = metric
    return out.dropna(subset=["ts", "value"])


def minute_means(df: pd.DataFrame, freq: str) -> pd.Series:
    """One metric -> mean value per (node, minute)."""
    return df.assign(ts=df["ts"].dt.floor(freq)).groupby(["node", "ts"])["value"].mean()


class FieldAccumulator:
    """Combines one metric at a time into a field (max over cores, mean over
    fans, ...) so the 48 core-temperature files never sit in memory together."""

    def __init__(self, agg: str):
        if agg not in ("max", "mean", "sum", "min"):
            raise ValueError(f"unsupported agg {agg!r}")
        self.agg, self.acc, self.count = agg, None, None

    def add(self, s: pd.Series) -> None:
        if self.acc is None:
            self.acc, self.count = s.astype(float), pd.Series(1.0, index=s.index)
            return
        a, b = self.acc.align(s, join="outer")
        if self.agg == "max":
            self.acc = np.fmax(a, b)
        elif self.agg == "min":
            self.acc = np.fmin(a, b)
        else:
            self.acc = a.fillna(0) + b.fillna(0)
            c, d = self.count.align(s.notna().astype(float), join="outer")
            self.count = c.fillna(0) + d.fillna(0)

    def result(self) -> pd.Series | None:
        if self.acc is None:
            return None
        if self.agg == "mean":
            return self.acc / self.count.reindex(self.acc.index)
        return self.acc


def finish_table(per_field: dict[str, pd.Series], load_transform: str) -> pd.DataFrame:
    cols = [s.rename(FIELD_TO_COLUMN[f]) for f, s in per_field.items() if s is not None]
    if not cols:
        raise ValueError("no metric data matched the configured patterns")
    table = pd.concat(cols, axis=1)
    table.index.names = ["node", "ts"]
    table = table.reset_index().rename(columns={"node": "node_id"})
    if "load_pct" in table:
        if load_transform == "100_minus":
            table["load_pct"] = 100.0 - table["load_pct"]
        elif load_transform == "power_proxy":
            def scale(s: pd.Series) -> pd.Series:
                lo, hi = s.quantile(0.05), s.quantile(0.95)
                if not np.isfinite(hi - lo) or hi - lo <= 0:
                    return s * 0.0
                return ((s - lo) / (hi - lo) * 100.0).clip(0, 100)
            table["load_pct"] = table.groupby("node_id")["load_pct"].transform(scale)
    return table


def reshape_fields(frames: dict[str, list[pd.DataFrame]], aggs: dict[str, str],
                   resample_minutes: int = 1, load_transform: str = "none") -> pd.DataFrame:
    """Combine standardised metric frames (columns node, ts, value, metric)
    into the canonical per-minute table. Each metric is averaged within the
    minute first; the metrics of one field are then combined with aggs[field]."""
    freq = f"{resample_minutes}min"
    per_field = {}
    for field, parts in frames.items():
        acc = FieldAccumulator(aggs.get(field, "mean"))
        for part in parts:
            for _metric, g in part.groupby("metric"):
                acc.add(minute_means(g, freq))
        per_field[field] = acc.result()
    return finish_table(per_field, load_transform)


def densest_window(first: pd.DataFrame, min_share: float = 0.5) -> tuple[pd.Timestamp, pd.Timestamp, pd.Series]:
    """Longest run of consecutive *calendar* days on which at least min_share
    of the busiest day's node count reported data. Days with no data at all
    count as bad days, so a gap (e.g. 03-03 then 03-30) breaks the run.
    The returned window is trimmed to the first and last reading inside the
    run, so a partial first or last day does not dilute node coverage."""
    day = first["ts"].dt.floor("D")
    per_day = first.assign(day=day).groupby("day")["node"].nunique()
    calendar = pd.date_range(per_day.index.min(), per_day.index.max(), freq="D")
    per_day = per_day.reindex(calendar, fill_value=0)
    ok = per_day >= per_day.max() * min_share
    best, run, best_start, start = 0, 0, None, None
    for d, good in ok.items():
        if good:
            start = d if run == 0 else start
            run += 1
            if run > best:
                best, best_start = run, start
        else:
            run = 0
    d0, d1 = best_start, best_start + pd.Timedelta(days=best)
    inside = first["ts"][(first["ts"] >= d0) & (first["ts"] < d1)]
    t0 = inside.min().floor("min")
    t1 = inside.max().floor("min") + pd.Timedelta(minutes=1)
    return t0, t1, per_day[per_day > 0]


def _match(files: list[MetricFile], plugin: str, pattern: str) -> list[MetricFile]:
    rx = re.compile(pattern)
    return [f for f in files if f.plugin == plugin and rx.search(f.metric)]


def _field_specs(cfg_m100: dict) -> dict[str, tuple[str, str, str]]:
    specs = {}
    for field in FIELD_TO_COLUMN:
        if field in cfg_m100:
            plugin = cfg_m100.get("load_plugin", cfg_m100["plugin"]) if field == "load" else cfg_m100["plugin"]
            specs[field] = (plugin, cfg_m100[field]["pattern"], cfg_m100[field].get("agg", "mean"))
    return specs


def _is_parquet(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            return fh.read(4) == b"PAR1"
    except OSError:
        return False


def _arrow():
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
        return pa, pc, pq
    except ImportError:
        return None


BATCH_ROWS = 2_000_000


def _batches(path: Path, metric: str, nodes: set[str] | None = None, value: bool = True):
    """Yield standardised frames (node, ts, value, metric) from one metric file,
    a row batch at a time, keeping only `nodes` when given. A full M100 month has
    tens of millions of rows per metric, so files are never loaded whole.
    Falls back to pandas for non-Parquet test files or when pyarrow is missing."""
    arrow = _arrow()
    if arrow is None or not _is_parquet(path):
        df = standardise(pd.read_parquet(path), metric)
        yield df[df["node"].isin(nodes)] if nodes is not None else df
        return
    pa, pc, pq = arrow
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    ts, node = _pick(names, TS_CANDIDATES, "timestamp"), _pick(names, NODE_CANDIDATES, "node")
    val = _pick(names, VALUE_CANDIDATES, "value")
    cols = [ts, node, val] if value else [ts, node]
    wanted = pa.array(sorted(nodes), type=pa.string()) if nodes is not None else None
    for rb in pf.iter_batches(batch_size=BATCH_ROWS, columns=cols):
        if wanted is not None:
            node_str = pc.cast(rb.column(cols.index(node)), pa.string())
            rb = rb.filter(pc.is_in(node_str, value_set=wanted))
            if rb.num_rows == 0:
                continue
        df = rb.to_pandas()
        if value:
            out = standardise(df, metric)
        else:
            out = pd.DataFrame({"node": df[node].astype(str),
                                "ts": pd.to_datetime(df[ts], utc=True, errors="coerce")}).dropna(subset=["ts"])
        yield out


def _read(path: Path, nodes: set[str] | None, t0=None, t1=None, metric: str = "") -> pd.DataFrame:
    parts = []
    for df in _batches(path, metric, nodes):
        if t0 is not None:
            df = df[(df["ts"] >= t0) & (df["ts"] < t1)]
        parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["node", "ts", "value", "metric"])


def _minute_means_streamed(path: Path, nodes: set[str], t0, t1, metric: str, freq: str) -> pd.Series:
    """Per-(node, minute) mean of one metric, accumulated batch by batch (sum and count)."""
    sums, counts = [], []
    for df in _batches(path, metric, nodes):
        df = df[(df["ts"] >= t0) & (df["ts"] < t1)]
        if df.empty:
            continue
        g = df.assign(ts=df["ts"].dt.floor(freq)).groupby(["node", "ts"])["value"]
        sums.append(g.sum())
        counts.append(g.count())
    if not sums:
        return pd.Series(dtype=float, index=pd.MultiIndex.from_arrays([[], []], names=["node", "ts"]))
    s = pd.concat(sums).groupby(level=[0, 1]).sum()
    c = pd.concat(counts).groupby(level=[0, 1]).sum()
    return s / c


def _node_minutes(path: Path, metric: str, freq: str) -> pd.DataFrame:
    """Pass 1: the unique (node, minute) pairs that have a reading, as a compact frame."""
    parts = []
    for df in _batches(path, metric, None, value=False):
        parts.append(df.assign(ts=df["ts"].dt.floor(freq))[["node", "ts"]].drop_duplicates())
    out = pd.concat(parts, ignore_index=True).drop_duplicates()
    out["node"] = out["node"].astype("category")
    return out


def probe(root: Path, cfg_m100: dict, max_list: int = 60) -> str:
    """Human-readable report of what the extracted dataset contains."""
    ym = cfg_m100.get("year_month") or None
    files = discover(root, ym)
    lines = [f"Parquet files found: {len(files)} under {root}" + (f" for month {ym}" if ym else "")]
    by_plugin: dict[str, set[str]] = defaultdict(set)
    for f in files:
        by_plugin[f.plugin].add(f.metric)
    for plugin, metrics in sorted(by_plugin.items()):
        names = sorted(metrics)
        lines.append(f"\n[{plugin}] {len(names)} metrics")
        lines.append("  " + ", ".join(names[:max_list]) + (" ..." if len(names) > max_list else ""))
    lines.append("\nConfigured field patterns:")
    for field, (plugin, pattern, agg) in _field_specs(cfg_m100).items():
        hits = sorted({f.metric for f in _match(files, plugin, pattern)})
        lines.append(f"  {field:9s} plugin={plugin} pattern={pattern!r} agg={agg} -> {len(hits)} metrics "
                     f"{hits[:8]}{' ...' if len(hits) > 8 else ''}")
    for plugin in sorted(by_plugin):
        sample = next(f for f in files if f.plugin == plugin)
        try:
            arrow = _arrow()
            if arrow is not None and _is_parquet(sample.path):
                pf = arrow[2].ParquetFile(sample.path)
                n_rows = pf.metadata.num_rows
                df = next(pf.iter_batches(batch_size=5)).to_pandas()
            else:
                df = pd.read_parquet(sample.path)
                n_rows = len(df)
            lines.append(f"\nSample file for [{plugin}] {sample.metric}: {n_rows} rows, columns={list(df.columns)}")
            lines.append(df.head(3).to_string())
        except Exception as exc:  # report, keep going
            lines.append(f"\nCould not read {sample.path}: {exc}")
    return "\n".join(lines)


def load_m100(root: Path, cfg_m100: dict, resample_minutes: int, n_nodes: int,
              min_coverage: float, seed: int, log=print, prefer_nodes: list | None = None) -> pd.DataFrame:
    """Pass 1 reads one core-temperature metric for every node to find the
    densest stretch of days and the nodes with good coverage in it. Pass 2
    streams every configured metric for only those nodes and days."""
    ym = cfg_m100.get("year_month") or None
    files = discover(root, ym)
    if not files:
        raise ValueError(f"no Parquet files under {root}" + (f" for year_month={ym}" if ym else ""))
    log(f"reading M100 month {ym or '(all months found)'}: {len(files)} metric files")
    specs = _field_specs(cfg_m100)
    if "cpu_temp" not in specs or "load" not in specs:
        raise ValueError("config [m100] must define cpu_temp and load patterns")
    freq = f"{resample_minutes}min"
    temp_files = _match(files, *specs["cpu_temp"][:2])
    if not temp_files:
        raise ValueError("no files match the cpu_temp pattern; run `python -m thermalview probe`")
    probe_metric = temp_files[0].metric
    log(f"pass 1: coverage from {probe_metric}")
    first = pd.concat([_node_minutes(f.path, probe_metric, freq) for f in temp_files if f.metric == probe_metric])
    first["node"] = first["node"].astype(str)
    t0, t1, per_day = densest_window(first)
    max_days = float(cfg_m100.get("max_days", 0) or 0)
    if max_days and (t1 - t0) > pd.Timedelta(days=max_days):
        t1 = t0 + pd.Timedelta(days=max_days)
        log(f"  window capped at [m100].max_days = {max_days:g}")
    log("  nodes reporting per day: " + ", ".join(f"{d:%m-%d}:{n}" for d, n in per_day.items()))
    log(f"  using {t0:%Y-%m-%d %H:%M} to {t1:%Y-%m-%d %H:%M} ({(t1 - t0) / pd.Timedelta(days=1):.2f} days)")
    win = first[(first["ts"] >= t0) & (first["ts"] < t1)]
    minutes = (t1 - t0) / pd.Timedelta(minutes=resample_minutes)
    coverage = win.assign(ts=win["ts"].dt.floor(freq)).groupby("node")["ts"].nunique() / minutes
    eligible = sorted(coverage[coverage >= min_coverage].index)
    log(f"  {len(coverage)} nodes, {len(eligible)} with coverage >= {min_coverage:.0%} "
        f"(coverage quartiles {coverage.quantile([.25, .5, .75]).round(2).tolist()})")
    if not eligible:
        raise ValueError(f"no node meets min_coverage in that window (best {coverage.max():.0%}); "
                         "lower [prepare].min_coverage")
    rng = np.random.default_rng(seed)
    prefer = [n for n in eligible if str(n) in {str(x) for x in (prefer_nodes or [])}]
    if prefer_nodes:
        log(f"  {len(prefer)} of {len(prefer_nodes)} preferred nodes (from another month) are eligible here")
    rest = [n for n in eligible if n not in set(prefer)]
    k = max(0, min(n_nodes - len(prefer), len(rest)))
    nodes = set(prefer[:n_nodes]) | set(rng.choice(rest, size=k, replace=False).tolist() if k else [])
    del first, win

    per_field = {}
    for field, (plugin, pattern, agg) in specs.items():
        matched = _match(files, plugin, pattern)
        log(f"pass 2: {field}: {len(matched)} metric files")
        acc = FieldAccumulator(agg)
        for i, f in enumerate(matched, 1):
            acc.add(_minute_means_streamed(f.path, nodes, t0, t1, f.metric, freq))
            if i % 12 == 0:
                log(f"    {i}/{len(matched)}")
        per_field[field] = acc.result()
    return finish_table(per_field, cfg_m100.get("load_transform", "none"))
