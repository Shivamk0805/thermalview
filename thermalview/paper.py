"""Fill the paper draft with numbers from results/.

docs/paper/draft.md is written with {{placeholders}}. `python -m thermalview
paper` computes every placeholder from the files in results/ and writes
docs/paper/draft_filled.md, plus docs/paper/numbers.json (every value, for
checking). Numbers are never typed by hand, so the paper cannot drift from
the experiments.

If the results did not come from M100, the output is written as
draft_DEV_NOT_RESULTS.md with a banner on top, and must not be used.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from thermalview.config import Config

MODEL = {"persistence": "Persistence", "ridge": "Ridge regression", "random_forest": "Random forest",
         "hist_gb": "Gradient boosting (HistGB)", "mlp": "Neural network (MLP)"}
FEATURE_SET = {"relative": "relative", "extended": "extended", "all": "all (with absolute levels)"}
POLICY = {"random": "Random", "least_loaded": "Least loaded", "coolest_now": "Coolest now (reactive)",
          "predictive_persist": "Predictive, persistence (ablation)", "thermalview": "ThermalView (ML + heat model)"}
FEATURE = {"temp": "current temperature", "temp_d1": "1-min temperature change", "temp_d5": "5-min temperature change",
           "temp_d15": "15-min temperature change", "temp_mean15": "15-min mean temperature",
           "temp_std15": "15-min temperature spread", "load": "current load", "load_d1": "1-min load change",
           "load_d5": "5-min load change", "load_mean15": "15-min mean load", "power": "power",
           "power_d5": "5-min power change", "inlet": "inlet air temperature", "inlet_d15": "15-min inlet change",
           "fan": "fan speed", "fan_d5": "5-min fan-speed change",
           "temp_d30": "30-min temperature change", "temp_d60": "60-min temperature change",
           "temp_std30": "30-min temperature spread", "temp_std60": "60-min temperature spread",
           "temp_dev60": "temperature minus its 60-min mean", "load_d30": "30-min load change",
           "load_d60": "60-min load change", "load_mean30": "30-min mean load", "load_mean60": "60-min mean load",
           "load_max15": "15-min max load", "load_min15": "15-min min load",
           "min_since_load_step": "minutes since last load step"}


def _p(x) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return "< 0.001" if x < 0.001 else f"{x:.3f}"


def _f(x, d=2) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x:.{d}f}"


def _pct(x, d=1) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{100 * x:.{d}f}%"


def _md_table(header: list[str], rows: list[list[str]], align: str | None = None) -> str:
    align = align or "l" + "r" * (len(header) - 1)
    sep = ["---:" if a == "r" else "---" for a in align]
    return "\n".join(["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"] +
                     ["| " + " | ".join(r) + " |" for r in rows])


def _sig(lo, hi, better="better than", worse="worse than", same="not distinguishable from") -> str:
    if lo is None or pd.isna(lo):
        return same
    return better if lo > 0 else worse if hi < 0 else same


def numbers(res: Path) -> dict:
    n: dict = {}
    summ = json.loads((res / "forecast_summary.json").read_text())
    n["data_source"] = summ.get("data_source", "unknown")
    ds = summ["dataset"]
    n.update(n_nodes=ds["nodes"], n_days=(f"{ds['days']:.0f}" if abs(ds["days"] - round(ds["days"])) < 0.05 else _f(ds["days"], 1)), data_start=ds["start"][:16], data_end=ds["end"][:16],
             n_rows=f"{ds['rows']:,}", n_valid_rows=f"{ds['valid_rows']:,}", temp_mean=_f(ds["cpu_temp_c"]["mean"], 1),
             temp_std=_f(ds["cpu_temp_c"]["std"], 1), temp_p50=_f(ds["cpu_temp_c"]["50%"], 1),
             temp_max=_f(ds["cpu_temp_c"]["max"], 1), load_mean=_f(ds["load_pct"]["mean"], 1),
             load_p50=_f(ds["load_pct"]["50%"], 1), optional_cols=", ".join(ds.get("optional_columns_used", [])) or "none",
             threshold_c=_f(summ["threshold_c"], 1))
    cfg = summ["config"]
    sp = summ["splits"][str(cfg["primary_horizon"])]
    n.update(n_holdout_nodes=len(sp["holdout_nodes"]), n_train=f"{sp['train']:,}", n_val=f"{sp['val']:,}",
             n_test_seen=f"{sp['test_seen']:,}", n_test_unseen=f"{sp['test_unseen']:,}",
             train_pct=int(cfg["train_frac"] * 100), val_pct=int(cfg["val_frac"] * 100),
             test_pct=int(round((1 - cfg["train_frac"] - cfg["val_frac"]) * 100)),
             holdout_pct=int(cfg["holdout_node_frac"] * 100), bootstrap_iters=cfg["bootstrap_iters"],
             event_rise_c=_f(cfg["event_rise_c"], 1), max_train_rows=f"{cfg['max_train_rows']:,}")

    fm = pd.read_csv(res / "forecast_metrics.csv")
    chosen = summ["chosen_model_by_horizon"]
    models = [m for m in MODEL if m in set(fm.model)]
    n["n_models"] = len(models)
    n["interval_cov_target"] = _pct(summ.get("interval_coverage", 0.9), 0)
    rows_int = []
    rows5, rowsH = [], []
    for h in (5, 15, 30):
        best = chosen[str(h)]
        n[f"best_h{h}"] = MODEL[best]
        for split, tag in (("test_seen", ""), ("test_unseen", "_unseen")):
            r = fm[(fm.horizon_min == h) & (fm.model == best) & (fm.split == split)].iloc[0]
            p = fm[(fm.horizon_min == h) & (fm.model == "persistence") & (fm.split == split)].iloc[0]
            n[f"mae_best_h{h}{tag}"] = _f(r.mae)
            n[f"mae_persist_h{h}{tag}"] = _f(p.mae)
            n[f"skill_h{h}{tag}"] = _pct(r.skill_vs_persistence)
            n[f"skill_h{h}{tag}_ci"] = f"{_pct(r.skill_ci_low)} to {_pct(r.skill_ci_high)}"
            n[f"sig_h{h}{tag}"] = _sig(r.skill_ci_low, r.skill_ci_high)
            n[f"heat_mae_best_h{h}{tag}"] = _f(r.mae_heating_events)
            n[f"heat_mae_persist_h{h}{tag}"] = _f(p.mae_heating_events)
            n[f"n_heat_events_h{h}{tag}"] = int(r.n_heating_events)
            n[f"crossings_h{h}{tag}"] = int(r.n_threshold_crossings)
            n[f"recall_best_h{h}{tag}"] = _pct(r.crossing_recall, 0)
            n[f"precision_best_h{h}{tag}"] = _pct(r.crossing_precision, 0)
            n[f"dm_p_h{h}{tag}"] = _p(r.get("dm_p"))
            if "interval_q" in fm.columns:
                n[f"int_q_h{h}"] = _f(r.interval_q)
                n[f"int_cov_h{h}{tag}"] = _pct(r.interval_coverage, 0)
                n[f"up_recall_h{h}{tag}"] = _pct(r.upper_crossing_recall, 0)
                n[f"up_precision_h{h}{tag}"] = _pct(r.upper_crossing_precision, 0)
                n[f"up_alarm_h{h}{tag}"] = _pct(r.upper_alarm_rate, 1)
                n[f"up_recall_persist_h{h}{tag}"] = _pct(p.upper_crossing_recall, 0)
                n[f"up_precision_persist_h{h}{tag}"] = _pct(p.upper_crossing_precision, 0)
                n[f"up_alarm_persist_h{h}{tag}"] = _pct(p.upper_alarm_rate, 1)
                n[f"int_q_persist_h{h}"] = _f(p.interval_q)
                if split == "test_seen":
                    rows_int.append([f"{h} min", "Persistence", f"±{_f(p.interval_q)}", _pct(p.interval_coverage, 1),
                                     f"{int(p.n_threshold_crossings)}", "0%", "n/a", _pct(p.upper_crossing_recall, 0),
                                     _pct(p.upper_crossing_precision, 0), _pct(p.upper_alarm_rate, 1)])
                    rows_int.append([f"{h} min", MODEL[best] + " *", f"±{_f(r.interval_q)}", _pct(r.interval_coverage, 1),
                                     f"{int(r.n_threshold_crossings)}", _pct(r.crossing_recall, 0),
                                     _pct(r.crossing_precision, 0), _pct(r.upper_crossing_recall, 0),
                                     _pct(r.upper_crossing_precision, 0), _pct(r.upper_alarm_rate, 1)])
        n[f"feat_h{h}"] = FEATURE_SET.get(str(r.get("features", "relative")), str(r.get("features", "relative")))
        for m in models:
            for split, tag in (("test_seen", ""), ("test_unseen", "_unseen")):
                r_ = fm[(fm.horizon_min == h) & (fm.model == m) & (fm.split == split)].iloc[0]
                n[f"skill_{m}_h{h}{tag}"] = _pct(r_.skill_vs_persistence)
                n[f"skill_{m}_h{h}{tag}_ci"] = f"{_pct(r_.skill_ci_low)} to {_pct(r_.skill_ci_high)}"
        for m in models:
            s = fm[(fm.horizon_min == h) & (fm.model == m) & (fm.split == "test_seen")].iloc[0]
            u = fm[(fm.horizon_min == h) & (fm.model == m) & (fm.split == "test_unseen")].iloc[0]
            if h == 5:
                ci = "" if pd.isna(s.skill_ci_low) else f" ({_pct(s.skill_ci_low)} to {_pct(s.skill_ci_high)})"
                rows5.append([MODEL[m] + (" *" if m == best else ""), _f(s.mae, 3), _f(u.mae, 3), _f(s.rmse, 3),
                              _pct(s.skill_vs_persistence) + ci, _f(s.mae_heating_events), _pct(s.crossing_recall, 0),
                              _pct(s.crossing_precision, 0), f"{s.inference_us_per_row:.1f}"])
    for m in models:
        rowsH.append([MODEL[m]] + [_f(fm[(fm.horizon_min == h) & (fm.model == m) & (fm.split == sp_)].iloc[0].mae, 3)
                                   for h in (5, 15, 30) for sp_ in ("test_seen", "test_unseen")])
    n["TABLE_FORECAST"] = _md_table(
        ["Model", "MAE seen", "MAE unseen", "RMSE seen", "Skill vs persistence (95% CI)", "MAE heating",
         "Crossing recall", "Crossing precision", "µs/row"], rows5)
    n["TABLE_HORIZON"] = _md_table(["Model", "5 min seen", "5 min unseen", "15 min seen", "15 min unseen",
                                    "30 min seen", "30 min unseen"], rowsH)
    n["TABLE_DATA"] = _md_table(["Property", "Value"], [
        ["Source", "M100 ExaData (CINECA Marconi100), CC BY 4.0" if n["data_source"] == "M100" else n["data_source"]],
        ["Window", f"{n['data_start']} to {n['data_end']} UTC ({n['n_days']} days)"],
        ["Nodes", str(n["n_nodes"])], ["Rows (1-min) / valid", f"{n['n_rows']} / {n['n_valid_rows']}"],
        ["CPU temperature mean ± sd (median, max)", f"{n['temp_mean']} ± {n['temp_std']} °C ({n['temp_p50']}, {n['temp_max']})"],
        ["CPU load mean (median)", f"{n['load_mean']}% ({n['load_p50']}%)"],
        ["Optional signals used", n["optional_cols"]], ["Thermal limit (95th pct. of training temps)", f"{n['threshold_c']} °C"],
        ["Train / val / test rows (5 min)", f"{n['n_train']} / {n['n_val']} / {n['n_test_seen']} (+{n['n_test_unseen']} unseen-node)"],
    ], "lr")

    if rows_int:
        n["TABLE_INTERVALS"] = _md_table(
            ["Horizon", "Model", f"{n['interval_cov_target']} interval (°C)", "Test coverage", "Crossings",
             "Recall (point)", "Precision (point)", "Recall (upper bound)", "Precision (upper bound)",
             "Alarm rate"], rows_int)
    ab_path = res / "forecast_feature_ablation.csv"
    if ab_path.exists():
        ab = pd.read_csv(ab_path)
        val = ab[ab["split"].isna()] if "split" in ab.columns else ab
        test = ab[ab["split"] == "test_seen"] if "split" in ab.columns else ab.iloc[0:0]
        rows_ab = []
        for h in (5, 15, 30):
            t = test[test.horizon_min == h]
            if t.empty:
                continue
            m = t.model.iloc[0]
            cells = [f"{h} min", MODEL.get(m, m)]
            for fs in summ.get("feature_sets_compared", ["relative", "extended"]):
                v = val[(val.horizon_min == h) & (val.model == m) & (val.features == fs)]
                tt = t[t.features == fs]
                cells.append(_pct(v.val_skill.iloc[0]) if len(v) else "n/a")
                cells.append(f"{_pct(tt.skill_vs_persistence.iloc[0])} ({_pct(tt.skill_ci_low.iloc[0])} to "
                             f"{_pct(tt.skill_ci_high.iloc[0])})" if len(tt) else "n/a")
                n[f"abl_{fs}_h{h}"] = _pct(tt.skill_vs_persistence.iloc[0]) if len(tt) else "n/a"
                n[f"abl_{fs}_h{h}_val"] = _pct(v.val_skill.iloc[0]) if len(v) else "n/a"
            rows_ab.append(cells)
        hdr = ["Horizon", "Deployed model"]
        for fs in summ.get("feature_sets_compared", ["relative", "extended"]):
            hdr += [f"{fs}: validation skill", f"{fs}: test skill (95% CI)"]
        n["TABLE_ABLATION_FEATURES"] = _md_table(hdr, rows_ab)
        # best validation skill per model and feature set, all horizons (shows MLP and others too)
        rows_v = []
        for m in models:
            if m == "persistence":
                continue
            row = [MODEL[m]]
            for h in (5, 15, 30):
                for fs in summ.get("feature_sets_compared", ["relative", "extended"]):
                    v = val[(val.horizon_min == h) & (val.model == m) & (val.features == fs)]
                    row.append(_pct(v.val_skill.iloc[0]) if len(v) else "n/a")
            rows_v.append(row)
        hdr = ["Model"] + [f"{h} min {fs[:3]}." for h in (5, 15, 30)
                           for fs in summ.get("feature_sets_compared", ["relative", "extended"])]
        n["TABLE_VALIDATION"] = _md_table(hdr, rows_v)

    imp = pd.read_csv(res / "feature_importance.csv").sort_values("mae_increase", ascending=False)
    for i, r in enumerate(imp.head(3).itertuples(), 1):
        n[f"imp{i}"] = FEATURE.get(r.feature, r.feature)
        n[f"imp{i}_val"] = _f(r.mae_increase, 3)
    n["TABLE_IMPORTANCE"] = _md_table(["Feature", "MAE increase when shuffled (°C)"],
                                      [[FEATURE.get(r.feature, r.feature), _f(r.mae_increase, 3)]
                                       for r in imp.head(10).itertuples()]
                                      + ([[f"({len(imp) - 10} more, each below {_f(imp.mae_increase.iloc[10], 3)})", ""]]
                                         if len(imp) > 10 else []))

    fid = pd.read_csv(res / "simulator_fidelity.csv")
    for r in fid.itertuples():
        n[f"fid_h{r.horizon_min}_sim"] = _f(r.sim_mae)
        n[f"fid_h{r.horizon_min}_pers"] = _f(r.persistence_mae)
    n["fid_n"] = int(fid.n.iloc[0])
    n["TABLE_FIDELITY"] = _md_table(["Open-loop horizon", "Heat model MAE (°C)", "Persistence MAE (°C)", "Checks"],
                                    [[f"{r.horizon_min} min", _f(r.sim_mae), _f(r.persistence_mae), str(r.n)] for r in fid.itertuples()])
    hp = pd.read_csv(res / "heat_params.csv")
    fitted = hp[hp.fitted]
    n.update(n_fitted=len(fitted), n_param_nodes=len(hp), alpha_med=_f(fitted.alpha.median(), 3),
             alpha_lo=_f(fitted.alpha.quantile(.1), 3), alpha_hi=_f(fitted.alpha.quantile(.9), 3),
             gain_med=_f(fitted.gain.median(), 3), inlet_k_med=_f(fitted.inlet_k.median(), 2),
             tau_med=_f(np.log(0.1) / np.log(max(1e-9, 1 - min(fitted.alpha.median(), 0.999))), 1))

    pm = json.loads((res / "placement_meta.json").read_text())
    ev = pm["config"]["evaluation"]
    n.update(n_seeds=len(ev["seeds"]), sim_days=ev["sim_days"], n_servers=len(pm["fleet_nodes"]),
             safety_margin=_f(pm["config"]["recommender"]["safety_margin_c"], 1),
             cpu_cap=_f(pm["config"]["simulator"]["cpu_cap_pct"], 0), bg_scale=_f(pm["config"]["simulator"]["background_load_scale"], 2))
    ps = pd.read_csv(res / "placement_summary.csv")
    if "load_factor" not in ps:
        ps["load_factor"] = 1.0
    rows, utils = [], sorted(ps.load_factor.unique())
    n["utils"] = ", ".join(f"{u:.2f}×" for u in utils)
    n["n_utils"] = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}.get(len(utils), str(len(utils)))
    for u in utils:
        k = int(round(u * 100))
        sub = ps[ps.load_factor == u]
        get = lambda pol, met: sub[(sub.policy == pol) & (sub.metric == met)].iloc[0]  # noqa: E731
        for pol in POLICY:
            if not ((sub.policy == pol).any()):
                continue
            v, ex, rj, tot = (get(pol, "caused_violation_pct"), get(pol, "caused_excess_degree_min"), get(pol, "rejected"),
                              get(pol, "violation_pct"))
            diff = "reference" if pol == "coolest_now" else (
                f"{v.diff_vs_coolest_now:+.3f} ({_f(v.diff_ci_low, 3)} to {_f(v.diff_ci_high, 3)}; p = {_f(v.wilcoxon_p, 3)})")
            rows.append([f"{u:.2f}×", POLICY[pol], f"{_f(v['mean'], 3)} ± {_f(v['std'], 3)}", _f(ex["mean"], 0),
                         _f(tot["mean"], 2), _f(rj["mean"], 1), diff])
        n[f"rec_viol_{k}"] = _f(get("coolest_now", "recorded_violation_pct")["mean"], 2)
        n[f"tv_total_{k}"], n[f"cn_total_{k}"] = _f(get("thermalview", "violation_pct")["mean"], 2), _f(get("coolest_now", "violation_pct")["mean"], 2)
        tv, cn, pp = get("thermalview", "caused_violation_pct"), get("coolest_now", "caused_violation_pct"), get("predictive_persist", "caused_violation_pct")
        n[f"tv_viol_{k}"], n[f"cn_viol_{k}"], n[f"pp_viol_{k}"] = _f(tv["mean"], 3), _f(cn["mean"], 3), _f(pp["mean"], 3)
        n[f"tv_diff_{k}"] = f"{tv.diff_vs_coolest_now:+.3f}"
        n[f"tv_diff_{k}_ci"] = f"{_f(tv.diff_ci_low, 3)} to {_f(tv.diff_ci_high, 3)}"
        n[f"tv_diff_{k}_p"] = _f(tv.wilcoxon_p, 3)
        n[f"tv_sig_{k}"] = _sig(-tv.diff_ci_high if not pd.isna(tv.diff_ci_high) else None,
                                -tv.diff_ci_low if not pd.isna(tv.diff_ci_low) else None,
                                "lower than", "higher than", "not distinguishable from")
        n[f"tv_rej_{k}"], n[f"cn_rej_{k}"] = _f(get("thermalview", "rejected")["mean"], 1), _f(get("coolest_now", "rejected")["mean"], 1)
        n[f"tv_exc_{k}"], n[f"cn_exc_{k}"] = (_f(get("thermalview", "caused_excess_degree_min")["mean"], 0),
                                            _f(get("coolest_now", "caused_excess_degree_min")["mean"], 0))
        rel = (tv["mean"] - cn["mean"]) / cn["mean"] if cn["mean"] else float("nan")
        n[f"tv_rel_{k}"] = _pct(rel, 0)
        n[f"tv_less_{k}"] = _pct(-rel, 0)
        n[f"tv_diff_{k}_p"] = _f(tv.wilcoxon_p, 3)
        n[f"pp_diff_{k}"] = f"{pp.diff_vs_coolest_now:+.3f}"
        n[f"pp_diff_{k}_ci"] = f"{_f(pp.diff_ci_low, 3)} to {_f(pp.diff_ci_high, 3)}"
        n[f"pp_less_{k}"] = _pct(-(pp["mean"] - cn["mean"]) / cn["mean"] if cn["mean"] else float("nan"), 0)
        n[f"rnd_viol_{k}"] = _f(get("random", "caused_violation_pct")["mean"], 2)
        n[f"ll_viol_{k}"] = _f(get("least_loaded", "caused_violation_pct")["mean"], 2)
        pe = get("predictive_persist", "caused_excess_degree_min")
        te = get("thermalview", "caused_excess_degree_min")
        n[f"pp_exc_{k}"] = _f(pe["mean"], 0)
        n[f"tv_exc_less_{k}"] = _pct(-te.diff_vs_coolest_now / get("coolest_now", "caused_excess_degree_min")["mean"], 0)
        n[f"tv_exc_{k}_ci"] = f"{_f(te.diff_ci_low, 0)} to {_f(te.diff_ci_high, 0)}"
        n[f"tv_exc_{k}_p"] = _f(te.wilcoxon_p, 3)
        n[f"pp_exc_{k}_ci"] = f"{_f(pe.diff_ci_low, 0)} to {_f(pe.diff_ci_high, 0)}"
        n[f"pp_diff_{k}_p"] = _f(pp.wilcoxon_p, 3)
        n[f"pp_sig_{k}"] = _sig(-pp.diff_ci_high if not pd.isna(pp.diff_ci_high) else None,
                                -pp.diff_ci_low if not pd.isna(pp.diff_ci_low) else None,
                                "lower than", "higher than", "not distinguishable from")
        runs_path = res / "placement_runs.csv"
        if runs_path.exists():                 # the forecast's own value: ThermalView minus the ablation, paired
            pr = pd.read_csv(runs_path)
            pr = pr[pr.load_factor == u] if "load_factor" in pr else pr
            for met, tag in (("caused_violation_pct", ""), ("caused_excess_degree_min", "_exc")):
                a_ = pr[pr.policy == "thermalview"].set_index("seed")[met]
                b_ = pr[pr.policy == "predictive_persist"].set_index("seed")[met]
                d = (a_ - b_.loc[a_.index]).to_numpy()
                if len(d) > 1 and d.std() > 0:
                    from scipy import stats as _st
                    half = _st.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
                    n[f"tvpp{tag}_diff_{k}"] = f"{d.mean():+.3f}" if not tag else f"{d.mean():+.0f}"
                    n[f"tvpp{tag}_diff_{k}_ci"] = (f"{d.mean() - half:+.3f} to {d.mean() + half:+.3f}" if not tag
                                                  else f"{d.mean() - half:+.0f} to {d.mean() + half:+.0f}")
                    n[f"tvpp{tag}_p_{k}"] = _f(_st.wilcoxon(d).pvalue, 3)
    n["TABLE_PLACEMENT"] = _md_table(["Load", "Policy", "Caused % server-min over limit (mean ± sd)", "Caused °C·min",
                                      "Total % over limit", "Rejected jobs", "Δ caused vs coolest-now (95% CI; Wilcoxon p)"],
                                     rows, "llrrrrr")

    sens_path = res / "sensitivity_summary.csv"
    if sens_path.exists():
        se = pd.read_csv(sens_path)
        rows_s = []
        for (kind, value), g in se.groupby(["setting", "value"], sort=True):
            c = g[g.metric == "caused_violation_pct"].set_index("policy")
            if "coolest_now" not in c.index or "thermalview" not in c.index:
                continue
            tv, cn = c.loc["thermalview"], c.loc["coolest_now"]
            pp = c.loc["predictive_persist"] if "predictive_persist" in c.index else None
            label = f"{value:g}th pct." if kind == "limit_percentile" else f"margin {value:g} °C"
            key = f"{'lim' if kind == 'limit_percentile' else 'mar'}_{str(value).replace('.', '_')}"
            rel = (tv["mean"] - cn["mean"]) / cn["mean"] if cn["mean"] else float("nan")
            n[f"sens_{key}_tv"], n[f"sens_{key}_cn"] = _f(tv["mean"]), _f(cn["mean"])
            n[f"sens_{key}_rel"] = _pct(-rel, 0) if rel == rel else "n/a"
            n[f"sens_{key}_sig"] = _sig(-tv.diff_ci_high, -tv.diff_ci_low, "lower than", "higher than",
                                         "not distinguishable from")
            rows_s.append([("Limit " if kind == "limit_percentile" else "Safety ") + label,
                           f"{g.threshold_c.iloc[0]:.1f}", _f(cn["mean"]), _f(tv["mean"]),
                           f"{tv.diff_vs_coolest_now:+.2f} ({tv.diff_ci_low:+.2f} to {tv.diff_ci_high:+.2f})",
                           f"{pp.diff_vs_coolest_now:+.2f} ({pp.diff_ci_low:+.2f} to {pp.diff_ci_high:+.2f})"
                           if pp is not None else "n/a"])
        n["TABLE_SENSITIVITY"] = _md_table(["Setting", "Limit (°C)", "Coolest-now caused %", "ThermalView caused %",
                                            "Δ ThermalView (95% CI)", "Δ ablation (95% CI)"], rows_s)
        lims = se[se.setting == "limit_percentile"].threshold_c
        if len(lims):
            n["sens_lo"], n["sens_hi"], n["sens_span"] = _f(lims.min(), 1), _f(lims.max(), 1), _f(lims.max() - lims.min(), 1)
        n["sens_load"] = f"{float(se.load_factor.iloc[0]) if 'load_factor' in se else 1.5:g}×"

    tr_path = res / "transfer_metrics.csv"
    if tr_path.exists():
        tr = pd.read_csv(tr_path)
        ts = json.loads((res / "transfer_summary.json").read_text())
        td = ts.get("test_dataset", {})
        n.update(tr_start=str(td.get("start", ""))[:10], tr_end=str(td.get("end", ""))[:10],
                 tr_days=_f(td.get("days"), 1) if td.get("days") is not None else "n/a",
                 tr_nodes=ts.get("n_test_nodes"), tr_trained_nodes=ts.get("n_trained_nodes_in_test"))
        rows_t = []
        for r in tr.itertuples():
            tag = {"all": "", "trained_nodes": "_trained", "new_nodes": "_new"}[r.group]
            n[f"tr_skill_h{r.horizon_min}{tag}"] = _pct(r.skill_vs_persistence)
            n[f"tr_skill_h{r.horizon_min}{tag}_ci"] = f"{_pct(r.skill_ci_low)} to {_pct(r.skill_ci_high)}"
            n[f"tr_sig_h{r.horizon_min}{tag}"] = _sig(r.skill_ci_low, r.skill_ci_high)
            n[f"tr_mae_h{r.horizon_min}{tag}"] = _f(r.mae)
            n[f"tr_pmae_h{r.horizon_min}{tag}"] = _f(r.persistence_mae)
            if hasattr(r, "interval_coverage"):
                n[f"tr_cov_h{r.horizon_min}{tag}"] = _pct(r.interval_coverage, 0)
            rows_t.append([f"{r.horizon_min} min", {"all": "all", "trained_nodes": "trained on", "new_nodes": "new"}[r.group],
                           str(r.nodes), MODEL.get(r.model, r.model), _f(r.persistence_mae, 3), _f(r.mae, 3),
                           f"{_pct(r.skill_vs_persistence)} ({_pct(r.skill_ci_low)} to {_pct(r.skill_ci_high)})",
                           _pct(getattr(r, "interval_coverage", float("nan")), 1)])
        n["TABLE_TRANSFER"] = _md_table(["Horizon", "Nodes", "n", "Model (frozen)", "Persistence MAE", "Model MAE",
                                         "Skill (95% CI)", "Interval coverage"], rows_t)
        tf = pd.read_csv(res / "transfer_fidelity.csv")
        for r in tf.itertuples():
            n[f"tr_fid_h{r.horizon_min}_sim"], n[f"tr_fid_h{r.horizon_min}_pers"] = _f(r.sim_mae), _f(r.persistence_mae)
            n[f"tr_fid_h{r.horizon_min}_skill"] = _pct(r.skill)


    for tag, px in (("", "je"), ("_transfer", "jt")):
        je_path = res / f"job_event_summary{tag}.json"
        if not je_path.exists():
            continue
        je = json.loads(je_path.read_text())
        if not je.get("ids_look_valid"):
            continue
        n.update({f"{px}_jobs": f"{je['n_jobs']:,}", f"{px}_starts": f"{je['n_starts']:,}",
                  f"{px}_oos_starts": f"{je.get('n_out_of_sample_starts', 0):,}", f"{px}_nodes": je["n_matched_nodes"],
                  f"{px}_load_step": _f(je["mean_load_step_2min"], 1), f"{px}_rise15": _f(je["mean_temp_rise_15min"], 1),
                  f"{px}_pre_dip": _f(je.get("temp_minus15_vs_minus1"), 1)})
        hm_file = res / f"job_event_heat_model{tag}.csv"
        hm = pd.read_csv(hm_file) if hm_file.exists() and hm_file.stat().st_size > 5 else pd.DataFrame()
        rows_h = []
        for r in hm.itertuples():
            h = r.horizon_min
            n[f"{px}_sim_h{h}"], n[f"{px}_pers_h{h}"] = _f(r.sim_mae), _f(r.persistence_mae)
            n[f"{px}_skill_h{h}"] = _pct(r.skill)
            n[f"{px}_act_h{h}"], n[f"{px}_mod_h{h}"] = _f(r.mean_actual_change, 1), _f(r.mean_model_change, 1)
            n[f"{px}_hm_n"] = f"{int(r.n):,}"
            rows_h.append([f"{h} min", f"{int(r.n):,}", f"{_f(r.mean_load_step, 1)}", _f(r.mean_actual_change, 2),
                           _f(r.mean_model_change, 2), _f(r.persistence_mae), _f(r.sim_mae), _pct(r.skill)])
        if rows_h:
            n[f"TABLE_{px.upper()}_HEAT"] = _md_table(["Horizon", "Starts", "Load step (points)", "Actual ΔT (°C)",
                                                    "Heat model ΔT (°C)", "Persistence MAE", "Heat model MAE",
                                                    "Skill"], rows_h)
        fc_file = res / f"job_event_forecast{tag}.csv"
        fc = pd.read_csv(fc_file) if fc_file.exists() and fc_file.stat().st_size > 5 else pd.DataFrame()
        rows_j = []
        for h, g in (fc.groupby("horizon_min") if len(fc) else []):
            a_ = g[g.minutes == "after_job_start"]
            o_ = g[g.minutes == "other_minutes"].iloc[0]
            if a_.empty:
                continue
            a_ = a_.iloc[0]
            n[f"{px}_fc_after_h{h}"], n[f"{px}_fc_other_h{h}"] = _pct(a_.skill), _pct(o_.skill)
            n[f"{px}_fc_after_mae_h{h}"], n[f"{px}_fc_other_mae_h{h}"] = _f(a_.model_mae), _f(o_.model_mae)
            n[f"{px}_pers_after_mae_h{h}"], n[f"{px}_pers_other_mae_h{h}"] = _f(a_.persistence_mae), _f(o_.persistence_mae)
            rows_j.append([f"{h} min", f"{int(a_.n):,}", _f(a_.persistence_mae), _f(a_.model_mae), _pct(a_.skill),
                           f"{int(o_.n):,}", _f(o_.persistence_mae), _f(o_.model_mae), _pct(o_.skill)])
        if rows_j:
            n[f"TABLE_{px.upper()}_FORECAST"] = _md_table(
                ["Horizon", "Minutes ≤15 after a start", "Persistence MAE", "Model MAE", "Skill",
                 "Other minutes", "Persistence MAE", "Model MAE", "Skill"], rows_j)
    return n


def fill(template: str, n: dict) -> tuple[str, list[str]]:
    missing: list[str] = []

    def rep(m):
        k = m.group(1)
        if k not in n:
            missing.append(k)
            return f"[[missing: {k}]]"
        return str(n[k])
    return re.sub(r"\{\{\s*([A-Za-z0-9_]+)\s*\}\}", rep, template), missing


CITE = re.compile(r"\[(@[A-Za-z0-9_]+(?:\s*;\s*@[A-Za-z0-9_]+)*)\]")


def cite(text: str, refs: dict) -> tuple[str, list[str]]:
    """Replace [@a; @b] with IEEE numbers in order of first appearance and fill {{REFERENCES}}."""
    order: list[str] = []
    unknown: list[str] = []

    def rep(m):
        keys = [k.strip().lstrip("@") for k in m.group(1).split(";")]
        nums = []
        for k in keys:
            if k not in refs:
                unknown.append(k)
                continue
            if k not in order:
                order.append(k)
            nums.append(order.index(k) + 1)
        return "[" + ", ".join(str(x) for x in sorted(nums)) + "]" if nums else f"[[unknown ref: {m.group(1)}]]"
    text = CITE.sub(rep, text)
    listing = "\n\n".join(f"[{i}] {refs[k]['text']}" for i, k in enumerate(order, 1))
    unknown += [f"{k} (cited but not yet verified)" for k in order if not refs[k].get("verified")]
    return text.replace("{{REFERENCES}}", listing), unknown


def build(cfg: Config, log=print) -> Path:
    res = cfg.path("results")
    paper_dir = cfg.root / "docs" / "paper"
    n = numbers(res)
    template = re.sub(r"\A\s*<!--.*?-->\s*", "", (paper_dir / "draft.md").read_text(encoding="utf-8"), flags=re.S)
    refs_file = paper_dir / "refs.json"
    refs = {k: v for k, v in json.loads(refs_file.read_text(encoding="utf-8")).items()
            if not k.startswith("_")} if refs_file.exists() else {}
    # <!-- if: KEY --> ... <!-- endif --> keeps a block only when KEY has a value (e.g. a step that was run)
    template = re.sub(r"<!-- if: (\w+) -->\n?(.*?)<!-- endif -->\n?",
                      lambda m: m.group(2) if m.group(1) in n else "", template, flags=re.S)
    template, unknown = cite(template, refs)
    text, missing = fill(template, n)
    month = pd.Timestamp(json.loads((res / "forecast_summary.json").read_text())["dataset"]["start"]).strftime("%y-%m")
    eval_month = str(cfg["m100"].get("year_month") or "")
    real = n["data_source"] == "M100" and (not eval_month or month == eval_month)
    if not real:
        why = (f"data source is {n['data_source']}" if n["data_source"] != "M100" else
               f"these results are from M100 month {month}, but the evaluation month is {eval_month}")
        text = (f"> **NOT THE PAPER'S RESULTS ({why}). Do not copy any number from this file.**\n"
                "> Run `python -m thermalview all` on the evaluation month, then `python -m thermalview paper` again.\n\n" + text)
    out = paper_dir / ("draft_filled.md" if real else "draft_DEV_NOT_RESULTS.md")
    out.write_text(text, encoding="utf-8")
    (paper_dir / ("numbers.json" if real else "numbers_DEV.json")).write_text(
        json.dumps({k: v for k, v in n.items() if not k.startswith("TABLE_")}, indent=2, default=str), encoding="utf-8")
    log(f"wrote {out}")
    if unknown:
        log("references to check: " + ", ".join(sorted(set(unknown))))
    if missing:
        log("placeholders with no value: " + ", ".join(sorted(set(missing))))
    if not real:
        log(f"WARNING: not the evaluation month's M100 results ({month}); the draft is marked DEV and must not be used.")
    return out
