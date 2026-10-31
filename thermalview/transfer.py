"""Cross-month test: the models trained on the evaluation month, frozen, scored on
another month that played no part in any choice (January 2021 for the paper).

Nothing is refitted. The forecaster bundle (models, feature columns, interval
widths, threshold) and the heat-model parameters come from the evaluation
month's artifacts; the other month's cleaned table only supplies test data.
Nodes are reported in two groups: nodes the forecaster was trained on in the
evaluation month, and nodes it never saw.

  python -m thermalview transfer --test-config config_jan.toml
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from thermalview import features as F
from thermalview import thermal_model as TM
from thermalview.config import Config
from thermalview.data.prepare import load_table
from thermalview.forecasting import bootstrap_skill, dm_test, interval_metrics, metrics


def run(cfg: Config, test_cfg: Config, log=print) -> dict:
    art, res = cfg.path("artifacts"), cfg.path("results")
    bundle = joblib.load(art / "forecaster.joblib")
    table, meta = load_table(test_cfg.path("processed"), "telemetry")
    if meta.get("source") != "M100":
        log(f"WARNING: test table source is {meta.get('source')!r}, not M100")
    optional = bundle["optional_present"]
    missing = [c for c in optional if c not in table.columns]
    if missing:
        raise ValueError(f"test month lacks signals the model needs: {missing}")
    horizons = sorted(int(h) for h in bundle["models"])
    feat = F.build_features(table, optional, horizons, dropna=False)
    cols_by_h = {int(k): v for k, v in bundle.get("feature_columns_by_h", {}).items()}
    need = sorted(set().union(*[cols_by_h.get(h, bundle["feature_columns"]) for h in horizons]))
    feat = feat.dropna(subset=need).reset_index(drop=True)
    train_nodes = set(map(str, bundle.get("train_nodes", [])))
    feat["group"] = np.where(feat["node_id"].isin(train_nodes), "trained_nodes", "new_nodes")
    fc = cfg["forecast"]
    threshold = float(bundle["threshold_c"])
    q = {int(k): float(v) for k, v in bundle.get("interval_q", {}).items()}
    rows = []
    for h in horizons:
        model, name = bundle["models"][h], bundle["chosen"][h] if h in bundle["chosen"] else bundle["chosen"][str(h)]
        cols = cols_by_h.get(h, bundle["feature_columns"])
        df_h = feat.dropna(subset=[f"y_{h}"])
        for group in ("all", "trained_nodes", "new_nodes"):
            d = df_h if group == "all" else df_h[df_h["group"] == group]
            if d.empty:
                continue
            pred = model.predict(d[cols])
            r = {"horizon_min": h, "model": name, "group": group, "nodes": int(d["node_id"].nunique())}
            r.update(metrics(d, pred, h, threshold, fc["event_rise_c"]))
            r["skill_ci_low"], r["skill_ci_high"] = bootstrap_skill(d, pred, h, int(fc["bootstrap_iters"]), int(fc["seed"]))
            r["dm_p"] = dm_test(d, pred, h)
            if h in q:
                r.update(interval_metrics(d, pred, h, q[h], threshold))
            rows.append(r)
            log(f"  h={h:>2} {group:13s} {r['nodes']:3d} nodes  skill {r['skill_vs_persistence']:+.3f} "
                f"({r['skill_ci_low']:+.3f} to {r['skill_ci_high']:+.3f})"
                + (f"  coverage {r['interval_coverage']:.3f}" if h in q else ""))
    out = pd.DataFrame(rows)
    out.to_csv(res / "transfer_metrics.csv", index=False)

    # heat model: evaluation-month parameters on the other month (fleet median for unseen nodes)
    params = TM.load_params(art / "heat_params.json")
    med = {k: float(np.median([getattr(p, k) for p in params.values() if p.fitted]))
           for k in ("alpha", "base", "gain", "inlet_k", "sigma", "power_0", "power_1", "fan_0", "fan_1", "inlet_mean")}
    all_params = {n: params.get(n) or TM.NodeParams(node_id=n, fitted=False, n_rows=0, **med)
                  for n in table["node_id"].unique()}
    fid = TM.fidelity(table, all_params, table["ts"].min(),
                      horizons=tuple(cfg["evaluation"].get("fidelity_horizons", [5, 15, 30])))
    fid.to_csv(res / "transfer_fidelity.csv", index=False)
    log(fid.round(3).to_string(index=False))
    summary = {"test_dataset": meta.get("summary", {}), "test_selection": meta.get("selection", {}),
               "n_trained_nodes_in_test": int(table["node_id"].isin(train_nodes).groupby(table["node_id"]).first().sum()),
               "n_test_nodes": int(table["node_id"].nunique()), "threshold_c": threshold,
               "heat_nodes_with_own_params": int(sum(n in params for n in table["node_id"].unique()))}
    (res / "transfer_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
