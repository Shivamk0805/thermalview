"""Forecasting experiment: four models, leakage-safe splits, paper metrics.

Splits (decided before looking at any results):
  * nodes: a seeded 20% of nodes is held out entirely ("unseen nodes")
  * time:  first 60% train, next 20% validation, last 20% test
  * embargo: a training row is only used if its target time is still inside
    the training period, and validation/test rows start WINDOW minutes after
    the boundary (the longest feature window), so no feature window or target
    overlaps another split.
Hyper-parameters, the feature set and the deployed model are chosen on
validation only; the test periods are scored once. A split-conformal
interval (validation residuals of the train-only model) gives each forecast
a calibrated upper bound, whose test coverage is reported.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from thermalview import features as F
from thermalview import models as M
from thermalview import schema


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test_seen: pd.DataFrame
    test_unseen: pd.DataFrame
    t_train_end: pd.Timestamp
    t_val_end: pd.Timestamp
    holdout_nodes: list[str]


def make_split(feat: pd.DataFrame, horizon: int, train_frac: float, val_frac: float,
               holdout_node_frac: float, seed: int, gap_min: int = F.WINDOW) -> Split:
    nodes = sorted(feat["node_id"].unique())
    rng = np.random.default_rng(seed)
    n_hold = max(1, int(round(len(nodes) * holdout_node_frac))) if len(nodes) > 2 else 0
    holdout = sorted(rng.choice(nodes, size=n_hold, replace=False).tolist()) if n_hold else []
    t0, t1 = feat["ts"].min(), feat["ts"].max()
    span = t1 - t0
    t_train_end = t0 + span * train_frac
    t_val_end = t0 + span * (train_frac + val_frac)
    h = pd.Timedelta(minutes=horizon)
    gap = pd.Timedelta(minutes=gap_min)
    target = f"y_{horizon}"
    f = feat.dropna(subset=[target])
    seen = ~f["node_id"].isin(holdout)
    return Split(
        train=f[seen & (f["ts"] + h <= t_train_end)],
        val=f[seen & (f["ts"] >= t_train_end + gap) & (f["ts"] + h <= t_val_end)],
        test_seen=f[seen & (f["ts"] >= t_val_end + gap)],
        test_unseen=f[~seen & (f["ts"] >= t_val_end + gap)],
        t_train_end=t_train_end, t_val_end=t_val_end, holdout_nodes=holdout,
    )


def _sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    return df if len(df) <= n else df.sample(n=n, random_state=seed)


def metrics(df: pd.DataFrame, pred_delta: np.ndarray, horizon: int, threshold: float,
            event_rise: float) -> dict:
    temp = df["temp"].to_numpy()
    actual = df[f"future_{horizon}"].to_numpy()
    pred = temp + pred_delta
    err = pred - actual
    abs_err = np.abs(err)
    persist_abs = np.abs(temp - actual)
    rising = (actual - temp) >= event_rise
    crossing = (temp < threshold) & (actual >= threshold)
    pred_cross = (temp < threshold) & (pred >= threshold)
    tp = int((crossing & pred_cross).sum())
    out = {
        "n": int(len(df)),
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "persistence_mae": float(persist_abs.mean()),
        "skill_vs_persistence": float(1 - abs_err.mean() / persist_abs.mean()) if persist_abs.mean() > 0 else 0.0,
        "n_heating_events": int(rising.sum()),
        "mae_heating_events": float(abs_err[rising].mean()) if rising.any() else float("nan"),
        "n_threshold_crossings": int(crossing.sum()),
        "crossing_recall": tp / int(crossing.sum()) if crossing.any() else float("nan"),
        "crossing_precision": tp / int(pred_cross.sum()) if pred_cross.any() else float("nan"),
    }
    return out


def bootstrap_skill(df: pd.DataFrame, pred_delta: np.ndarray, horizon: int, iters: int, seed: int) -> tuple[float, float]:
    """95% CI of skill vs persistence, resampling whole (node, day) blocks so
    that autocorrelated minutes are not treated as independent."""
    temp = df["temp"].to_numpy()
    actual = df[f"future_{horizon}"].to_numpy()
    blocks = pd.factorize(df["node_id"].astype(str) + "|" + df["ts"].dt.strftime("%Y-%m-%d"))[0]
    m = np.bincount(blocks, weights=np.abs(temp + pred_delta - actual))
    p = np.bincount(blocks, weights=np.abs(temp - actual))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(m), size=(iters, len(m)))
    skills = 1 - m[idx].sum(axis=1) / np.maximum(p[idx].sum(axis=1), 1e-12)
    lo, hi = np.percentile(skills, [2.5, 97.5])
    return float(lo), float(hi)


def dm_test(df: pd.DataFrame, pred_delta: np.ndarray, horizon: int) -> float:
    """Two-sided p-value of a Diebold-Mariano-style test of equal absolute error
    against persistence. Loss differences are averaged within (node, day)
    blocks first, so autocorrelation inside a day does not inflate the test."""
    from math import erf, sqrt
    temp = df["temp"].to_numpy()
    actual = df[f"future_{horizon}"].to_numpy()
    d = np.abs(temp + pred_delta - actual) - np.abs(temp - actual)
    blocks = pd.factorize(df["node_id"].astype(str) + "|" + df["ts"].dt.strftime("%Y-%m-%d"))[0]
    db = np.bincount(blocks, weights=d) / np.bincount(blocks)
    if len(db) < 3 or db.std(ddof=1) == 0:
        return float("nan")
    z = db.mean() / (db.std(ddof=1) / sqrt(len(db)))
    return float(2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2)))))


def interval_metrics(df: pd.DataFrame, pred_delta: np.ndarray, horizon: int, q: float, threshold: float) -> dict:
    """Coverage of [pred - q, pred + q] and threshold-crossing detection using the upper bound."""
    temp = df["temp"].to_numpy()
    actual = df[f"future_{horizon}"].to_numpy()
    pred = temp + pred_delta
    crossing = (temp < threshold) & (actual >= threshold)
    alarm = (temp < threshold) & (pred + q >= threshold)
    tp = int((crossing & alarm).sum())
    return {
        "interval_q": float(q),
        "interval_coverage": float((np.abs(pred - actual) <= q).mean()),
        "upper_crossing_recall": tp / int(crossing.sum()) if crossing.any() else float("nan"),
        "upper_crossing_precision": tp / int(alarm.sum()) if alarm.any() else float("nan"),
        "upper_alarm_rate": float(alarm.mean()),
    }


def resolve_threshold(cfg_value, train_temps: pd.Series) -> float:
    if isinstance(cfg_value, (int, float)):
        return float(cfg_value)
    return float(np.round(train_temps.quantile(0.95) * 2) / 2)


def run(table: pd.DataFrame, cfg: dict, data_meta: dict, out_dir: Path, artifact_dir: Path, log=print) -> dict:
    fc = cfg["forecast"]
    horizons = list(fc["horizons"])
    seed = int(fc["seed"])
    optional = schema.available_optional(table)
    mode_cfg = fc.get("features", "auto")
    modes = list(fc.get("feature_candidates", ["relative", "extended"])) if mode_cfg == "auto" else [mode_cfg]
    colsets = {m: F.feature_columns(optional, m) for m in modes}
    for m, c in colsets.items():
        log(f"feature set {m} ({len(c)}): {c}")
    coverage = float(fc.get("interval_coverage", 0.9))
    feat = F.build_features(table, optional, horizons, dropna=False)
    # every feature set is scored on exactly the same rows
    feat = feat.dropna(subset=sorted(set().union(*colsets.values()))).reset_index(drop=True)
    gap = max(F.lookback(m) for m in modes)
    ablation_rows, intervals, cols_by_h = [], {}, {}
    model_names = list(fc.get("models", M.MODELS))     # compared models (default: all)

    rows, ci_rows, chosen, trained = [], [], {}, {}
    sample_preds = []
    split_info = {}
    threshold = None
    for h in horizons:
        sp = make_split(feat, h, fc["train_frac"], fc["val_frac"], fc["holdout_node_frac"], seed, gap)
        if threshold is None:
            threshold = resolve_threshold(fc["threshold_c"], sp.train["temp"])
        split_info[h] = {k: int(len(getattr(sp, k))) for k in ("train", "val", "test_seen", "test_unseen")}
        split_info[h].update(t_train_end=str(sp.t_train_end), t_val_end=str(sp.t_val_end),
                             holdout_nodes=sp.holdout_nodes)
        target = f"y_{h}"
        train = _sample(sp.train, int(fc["max_train_rows"]), seed)
        val = _sample(sp.val, 100_000, seed)
        best, fitted = {}, {}
        val_persist = float(np.abs(val[target]).mean())
        for name, (_label, _f, param_grid) in M.MODELS.items():
            if name not in model_names:
                continue
            best_mae, best_params, best_mode = np.inf, {}, modes[0]
            for mode in (modes if name != "persistence" else modes[:1]):
                cols = colsets[mode]
                mode_mae = np.inf
                for params in M.grid(param_grid):
                    model = M.build(name, params, seed).fit(train[cols], train[target])
                    mae = float(np.abs(model.predict(val[cols]) - val[target]).mean())
                    mode_mae = min(mode_mae, mae)
                    if mae < best_mae:
                        best_mae, best_params, best_mode = mae, params, mode
                        fitted[name] = model
                ablation_rows.append({"horizon_min": h, "model": name, "features": mode, "val_mae": mode_mae,
                                      "val_skill": 1 - mode_mae / val_persist if val_persist else 0.0})
            best[name] = (best_mae, best_params, best_mode)
            log(f"  h={h:>2} {name:14s} val MAE={best_mae:.3f} features={best_mode} params={best_params}")
        # refit each model on train+val with its chosen parameters, score test once
        trainval = _sample(pd.concat([sp.train, sp.val[sp.val["ts"] + pd.Timedelta(minutes=h) <= sp.t_val_end]]),
                           int(fc["max_train_rows"]), seed)
        # split-conformal interval for each model: |residual| quantile on validation, train-only fit
        qs = {}
        for name, (_mae, _params, mode) in best.items():
            res_v = np.abs(fitted[name].predict(val[colsets[mode]]) - val[target].to_numpy()) if name in fitted \
                else np.abs(val[target].to_numpy())
            n_v = len(res_v)
            qs[name] = float(np.quantile(res_v, min(1.0, np.ceil((n_v + 1) * coverage) / n_v)))
        for name, (val_mae, params, mode) in best.items():
            cols = colsets[mode]
            model = M.build(name, params, seed).fit(trainval[cols], trainval[target])
            for split_name in ("test_seen", "test_unseen"):
                test = getattr(sp, split_name)
                if test.empty:
                    continue
                t = time.perf_counter()
                pred = model.predict(test[cols])
                us_per_row = (time.perf_counter() - t) / len(test) * 1e6
                row = {"horizon_min": h, "model": name, "label": M.MODELS[name][0], "split": split_name,
                       "features": mode, "val_mae": val_mae, "params": json.dumps(params),
                       "inference_us_per_row": us_per_row}
                row.update(metrics(test, pred, h, threshold, fc["event_rise_c"]))
                row.update(interval_metrics(test, pred, h, qs[name], threshold))
                if name != "persistence":
                    row["skill_ci_low"], row["skill_ci_high"] = bootstrap_skill(
                        test, pred, h, int(fc["bootstrap_iters"]), seed)
                    row["dm_p"] = dm_test(test, pred, h)
                rows.append(row)
                if split_name == "test_seen":
                    keep_nodes = sorted(test["node_id"].unique())[:3]
                    sub = test[test["node_id"].isin(keep_nodes)]
                    sample_preds.append(pd.DataFrame({
                        "horizon_min": h, "model": name, "node_id": sub["node_id"].values, "ts": sub["ts"].values,
                        "actual": sub[f"future_{h}"].values,
                        "predicted": sub["temp"].values + model.predict(sub[cols])}))
            trained[(h, name)] = model
        # deployed model: lowest validation MAE (test results never used to choose)
        chosen[h] = min(best, key=lambda n: best[n][0])
        cols_by_h[h] = colsets[best[chosen[h]][2]]
        intervals[h] = qs[chosen[h]]
        log(f"  h={h:>2} chosen on validation: {chosen[h]} with {best[chosen[h]][2]} features; "
            f"{coverage:.0%} interval +/-{intervals[h]:.2f} C")
        # feature-set ablation for the chosen model family, reported on test (not used to choose)
        name, (_v, params, chosen_mode) = chosen[h], best[chosen[h]]
        for mode in modes:
            m_ = trained[(h, name)] if mode == chosen_mode else \
                M.build(name, params, seed).fit(trainval[colsets[mode]], trainval[target])
            for split_name in ("test_seen", "test_unseen"):
                test = getattr(sp, split_name)
                if test.empty:
                    continue
                pred = m_.predict(test[colsets[mode]])
                lo, hi = bootstrap_skill(test, pred, h, int(fc["bootstrap_iters"]), seed)
                ablation_rows.append({"horizon_min": h, "model": name, "features": mode, "split": split_name,
                                      "skill_vs_persistence": metrics(test, pred, h, threshold,
                                                                      fc["event_rise_c"])["skill_vs_persistence"],
                                      "skill_ci_low": lo, "skill_ci_high": hi, "chosen": mode == chosen_mode})

    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(out_dir / "forecast_metrics.csv", index=False)
    pd.concat(sample_preds).to_csv(out_dir / "forecast_predictions_sample.csv", index=False)
    pd.DataFrame(ablation_rows).to_csv(out_dir / "forecast_feature_ablation.csv", index=False)

    primary = int(fc["primary_horizon"])
    cols = cols_by_h[primary]
    importance = _permutation_importance(trained[(primary, chosen[primary])], feat, cols, primary, seed,
                                         split_info[primary])
    importance.to_csv(out_dir / "feature_importance.csv", index=False)

    bundle = {
        "models": {h: trained[(h, chosen[h])] for h in horizons},
        "chosen": chosen, "feature_columns": cols, "feature_columns_by_h": cols_by_h,
        "interval_q": intervals, "interval_coverage": coverage, "optional_present": optional,
        "train_nodes": sorted(set(feat["node_id"]) - set(split_info[horizons[0]]["holdout_nodes"])),
        "threshold_c": threshold, "primary_horizon": primary, "data_source": data_meta.get("source", "unknown"),
    }
    joblib.dump(bundle, artifact_dir / "forecaster.joblib")
    summary = {
        "data_source": data_meta.get("source", "unknown"),
        "dataset": data_meta.get("summary", {}),
        "threshold_c": threshold, "feature_columns": cols, "feature_columns_by_h": cols_by_h,
        "feature_sets_compared": modes, "interval_q": intervals, "interval_coverage": coverage,
        "splits": split_info,
        "chosen_model_by_horizon": chosen, "config": cfg["forecast"],
    }
    (out_dir / "forecast_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def _permutation_importance(model, feat, cols, h, seed, info) -> pd.DataFrame:
    test = feat[feat["ts"] >= pd.Timestamp(info["t_val_end"])].dropna(subset=[f"y_{h}"])
    test = _sample(test, 20_000, seed)
    X, y = test[cols].copy(), test[f"y_{h}"].to_numpy()
    base = np.abs(model.predict(X) - y).mean()
    rng = np.random.default_rng(seed)
    out = []
    for c in cols:
        Xp = X.copy()
        Xp[c] = rng.permutation(Xp[c].to_numpy())
        out.append({"feature": c, "mae_increase": float(np.abs(model.predict(Xp) - y).mean() - base)})
    return pd.DataFrame(out).sort_values("mae_increase", ascending=False)
