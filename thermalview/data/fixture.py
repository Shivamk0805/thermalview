"""DEVELOPMENT-ONLY synthetic telemetry shaped like the processed M100 table.

This exists so the pipeline, tests and app can be built and checked before
the real dataset is downloaded. Anything produced from it is marked
source="dev-fixture" and must never be reported as a result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_fixture(n_nodes: int = 24, days: int = 14, seed: int = 0,
                 start: str = "2021-03-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    minutes = days * 24 * 60
    ts = pd.date_range(start, periods=minutes, freq="1min", tz="UTC")
    t = np.arange(minutes)
    rows = []
    for i in range(n_nodes):
        alpha = rng.uniform(0.05, 0.22)
        gain = rng.uniform(0.25, 0.45)          # C of steady-state rise per % load
        base = rng.uniform(24.0, 32.0)
        inlet_k = rng.uniform(0.6, 1.0)

        # HPC-like load: jobs run near full load, separated by idle gaps
        load = np.empty(minutes)
        pos = 0
        while pos < minutes:
            busy = int(rng.lognormal(np.log(90), 0.7))
            idle = int(rng.lognormal(np.log(15), 0.8))
            level = rng.uniform(65, 100)
            load[pos:pos + busy] = level
            load[pos + busy:pos + busy + idle] = rng.uniform(2, 15)
            pos += busy + idle
        load = np.clip(load + rng.normal(0, 2.0, minutes), 0, 100)

        inlet = 22 + 1.5 * np.sin(2 * np.pi * t / 1440 + rng.uniform(0, 6)) \
            + np.cumsum(rng.normal(0, 0.01, minutes))
        temp = np.empty(minutes)
        temp[0] = base + gain * load[0]
        for k in range(1, minutes):
            target = base + gain * load[k - 1] + inlet_k * (inlet[k - 1] - 22)
            a = alpha * (1.6 if temp[k - 1] > 68 else 1.0)   # fan curve: stronger cooling when hot
            temp[k] = temp[k - 1] + a * (target - temp[k - 1]) + rng.normal(0, 0.25)
        power = 320 + 11.5 * load + rng.normal(0, 8, minutes)
        fan = 3000 + 90 * np.clip(temp - 50, 0, None) + rng.normal(0, 40, minutes)

        df = pd.DataFrame({"node_id": f"dev{i:03d}", "ts": ts, "cpu_temp_c": temp, "load_pct": load,
                           "power_w": power, "inlet_c": inlet, "fan_rpm": fan})
        # sensor dropouts: short gaps and a few long outages
        for _ in range(rng.integers(5, 15)):
            s = rng.integers(0, minutes - 5)
            df.loc[s:s + rng.integers(1, 3), ["cpu_temp_c", "load_pct"]] = np.nan
        for _ in range(rng.integers(0, 3)):
            s = rng.integers(0, minutes - 200)
            df = df.drop(index=range(s, s + int(rng.integers(30, 180))), errors="ignore")
        rows.append(df)
    return pd.concat(rows, ignore_index=True)
