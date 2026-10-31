"""Paper figures (PNG, 300 dpi) built only from files in results/."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# reference categorical palette (fixed order), neutral gray for baselines
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
NEUTRAL = "#8c8b85"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
MODEL_COLORS = {"persistence": NEUTRAL, "ridge": SERIES[0], "random_forest": SERIES[1], "hist_gb": SERIES[2],
                "mlp": SERIES[4]}
MODEL_LABELS = {"persistence": "Persistence", "ridge": "Ridge", "random_forest": "Random forest",
                "hist_gb": "HistGB", "mlp": "MLP"}
POLICY_COLORS = {"random": NEUTRAL, "least_loaded": "#b4b2a9", "coolest_now": SERIES[1],
                 "predictive_persist": SERIES[3], "thermalview": SERIES[0]}


def _style(ax, title, ylabel, xlabel=None):
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_ylabel(ylabel, fontsize=9, color=INK2)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b4b2a9")
    ax.tick_params(colors=INK2, labelsize=8)


def _source_note(fig, results: Path):
    p = results / "forecast_summary.json"
    src = json.loads(p.read_text()).get("data_source", "?") if p.exists() else "?"
    note = f"data: {src}" + ("  -  DEVELOPMENT DATA, NOT A RESULT" if src == "dev-fixture" else "")
    fig.text(0.99, 0.01, note, ha="right", va="bottom", fontsize=7,
             color="#e34948" if src == "dev-fixture" else INK2)


def fig_forecast_mae(results: Path, out: Path) -> Path:
    m = pd.read_csv(results / "forecast_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, split, title in zip(axes, ["test_seen", "test_unseen"], ["Seen nodes, future time", "Unseen nodes, future time"]):
        d = m[m["split"] == split]
        horizons = sorted(d["horizon_min"].unique())
        models = [k for k in MODEL_COLORS if k in set(d["model"])]
        width = 0.8 / len(models)
        for i, name in enumerate(models):
            dd = d[d["model"] == name].set_index("horizon_min").reindex(horizons)
            xs = [h_i + (i - (len(models) - 1) / 2) * width for h_i in range(len(horizons))]
            ax.bar(xs, dd["mae"], width=width * 0.92, color=MODEL_COLORS[name], label=MODEL_LABELS[name])
        ax.set_xticks(range(len(horizons)), [f"{h} min" for h in horizons])
        _style(ax, title, "MAE (°C)" if split == "test_seen" else "", "Forecast horizon")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    _source_note(fig, results)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path = out / "fig_forecast_mae.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def fig_skill(results: Path, out: Path) -> Path:
    m = pd.read_csv(results / "forecast_metrics.csv")
    d = m[(m["model"] != "persistence") & (m["split"] == "test_seen")]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for name, g in d.groupby("model"):
        g = g.sort_values("horizon_min")
        lo = g["skill_vs_persistence"] - g["skill_ci_low"]
        hi = g["skill_ci_high"] - g["skill_vs_persistence"]
        ax.errorbar(g["horizon_min"], g["skill_vs_persistence"] * 100, yerr=[lo * 100, hi * 100],
                    color=MODEL_COLORS[name], marker="o", markersize=5, linewidth=2, capsize=3,
                    label=MODEL_LABELS[name])
    ax.axhline(0, color=INK2, linewidth=1, linestyle="--")
    ax.text(ax.get_xlim()[1], 0, " persistence", va="center", fontsize=8, color=INK2)
    _style(ax, "Skill over persistence (95% block-bootstrap CI)", "MAE reduction (%)", "Forecast horizon (min)")
    ax.legend(frameon=False, fontsize=8)
    _source_note(fig, results)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path = out / "fig_skill.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def fig_example(results: Path, out: Path) -> Path:
    s = pd.read_csv(results / "forecast_predictions_sample.csv", parse_dates=["ts"])
    summ = json.loads((results / "forecast_summary.json").read_text())
    h = int(summ["config"]["primary_horizon"])
    chosen = summ["chosen_model_by_horizon"][str(h)]
    s = s[(s["horizon_min"] == h)]
    node = sorted(s["node_id"].unique())[0]
    s = s[s["node_id"] == node]
    t0 = s["ts"].min()
    s = s[s["ts"] < t0 + pd.Timedelta(hours=12)]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    act = s[s["model"] == chosen]
    ax.plot(act["ts"] + pd.Timedelta(minutes=h), act["actual"], color=INK, linewidth=1.6, label="Observed")
    for name in ("persistence", chosen):
        d = s[s["model"] == name]
        ax.plot(d["ts"] + pd.Timedelta(minutes=h), d["predicted"], color=MODEL_COLORS[name], linewidth=1.4,
                label=f"{MODEL_LABELS[name]} forecast", alpha=0.95)
    _style(ax, f"{h}-minute forecasts on test data, node {node}", "CPU temperature (°C)")
    ax.legend(frameon=False, fontsize=8, ncol=3)
    _source_note(fig, results)
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path = out / "fig_example_forecast.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def fig_importance(results: Path, out: Path) -> Path:
    d = pd.read_csv(results / "feature_importance.csv").sort_values("mae_increase")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(d["feature"], d["mae_increase"], color=SERIES[0], height=0.7)
    _style(ax, "Permutation importance (chosen model)", "", "MAE increase when feature is shuffled (°C)")
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    _source_note(fig, results)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path = out / "fig_importance.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def fig_fidelity(results: Path, out: Path) -> Path:
    d = pd.read_csv(results / "simulator_fidelity.csv")
    fig, ax = plt.subplots(figsize=(6, 3.2))
    x = range(len(d))
    ax.bar([i - 0.2 for i in x], d["persistence_mae"], width=0.38, color=NEUTRAL, label="Persistence")
    ax.bar([i + 0.2 for i in x], d["sim_mae"], width=0.38, color=SERIES[0], label="Heat model, real load")
    ax.set_xticks(list(x), [f"{h} min" for h in d["horizon_min"]])
    _style(ax, "Simulator check on unseen test period", "MAE vs observed (°C)", "Open-loop horizon")
    ax.legend(frameon=False, fontsize=8)
    _source_note(fig, results)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path = out / "fig_simulator_fidelity.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def fig_placement(results: Path, out: Path) -> Path:
    runs = pd.read_csv(results / "placement_runs.csv")
    if "load_factor" not in runs:
        runs["load_factor"] = float("nan")
    order = [p for p in POLICY_COLORS if p in set(runs["policy"])]
    labels = {"random": "Random", "least_loaded": "Least loaded", "coolest_now": "Coolest now",
              "predictive_persist": "Predictive (persistence)", "thermalview": "ThermalView"}
    utils = sorted(runs["load_factor"].dropna().unique()) or [float("nan")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    width = 0.8 / len(order)
    m0, m1 = ("caused_violation_pct", "caused_excess_degree_min") if "caused_violation_pct" in runs else ("violation_pct", "excess_degree_min")
    for ax, metric, title, unit in ((axes[0], m0, "Time over the limit caused by placement", "% of server-minutes"),
                                    (axes[1], m1, "Severity caused by placement", "°C·minutes per run"),
                                    (axes[2], "rejected", "Jobs rejected", "jobs per run")):
        for k, pol in enumerate(order):
            g = runs[runs["policy"] == pol].groupby("load_factor")[metric]
            m, sd = g.mean().reindex(utils), g.std().reindex(utils)
            ax.bar([i + (k - (len(order) - 1) / 2) * width for i in range(len(utils))], m, width, yerr=sd,
                   capsize=2, color=POLICY_COLORS[pol], label=labels[pol],
                   error_kw={"ecolor": INK2, "elinewidth": 0.8})
        ax.set_xticks(range(len(utils)), [f"{u:.2f}× recorded load" if u == u else "" for u in utils])
        _style(ax, title, unit)
    axes[0].legend(fontsize=7, frameon=False)
    _source_note(fig, results)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path = out / "fig_placement.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def fig_overview(results: Path, out: Path) -> Path:
    """Figure 1: system overview (drawn, no data)."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")

    def box(x, y, w, h, title, body, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    facecolor="white", edgecolor=color, linewidth=1.6))
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=8.5, color=INK, weight="bold")
        ax.text(x + w / 2, y + h - 6.2, body, ha="center", va="top", fontsize=6.8, color=INK2, linespacing=1.35)

    def arrow(x0, y0, x1, y1, text=""):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10, color=INK2,
                                     linewidth=1.1))
        if text:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 1.2, text, ha="center", fontsize=6.3, color=INK2)

    box(1, 13, 16, 18, "Telemetry", "IPMI: core temps,\npower, inlet, fans\nGanglia: CPU load\n1-min grid", NEUTRAL)
    box(22, 22, 18, 15, "Forecaster", "relative / extended\nfeatures; model and\nset chosen on\nvalidation", SERIES[0])
    box(22, 2, 18, 15, "Heat model", "per-node gain and\nresponse rate α,\nanchored to the\nrecorded trace", SERIES[2])
    box(46, 12, 22, 18, "Recommender", "filters: CPU, memory,\nfan fault, stale, drain\nT̂ = forecast(t+h)\n+ gain·c·(1−(1−α)^h)", SERIES[1])
    box(74, 22, 24, 15, "Operator console", "ranked servers with\nexplanations, alerts,\napproval queue, audit", SERIES[3])
    box(74, 2, 24, 15, "Fleet simulator", "replays real traces,\nadds placed jobs' heat;\npolicy comparison", SERIES[4])
    arrow(17, 25, 22, 29)
    arrow(17, 19, 22, 10)
    arrow(40, 29, 46, 24)
    arrow(40, 10, 46, 17)
    ax.text(43.5, 29.5, "T(t+h)", ha="center", fontsize=6.3, color=INK2)
    ax.text(43.5, 8.0, "job heat", ha="center", fontsize=6.3, color=INK2)
    arrow(68, 24, 74, 29)
    arrow(68, 18, 74, 10)
    arrow(80, 22, 80, 17)
    ax.text(81, 19.2, "same code", fontsize=6.3, color=INK2)
    fig.tight_layout()
    path = out / "fig_overview.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def fig_feature_ablation(results: Path, out: Path) -> Path:
    ab = pd.read_csv(results / "forecast_feature_ablation.csv")
    t = ab[ab.get("split", pd.Series(dtype=str)) == "test_seen"]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    sets = list(dict.fromkeys(t["features"]))
    hs = sorted(t["horizon_min"].unique())
    w = 0.8 / max(1, len(sets))
    for i, fs in enumerate(sets):
        d = t[t.features == fs].set_index("horizon_min").reindex(hs)
        xs = [k + (i - (len(sets) - 1) / 2) * w for k in range(len(hs))]
        ax.bar(xs, d.skill_vs_persistence * 100, width=w * 0.9, color=SERIES[i], label=fs,
               yerr=[(d.skill_vs_persistence - d.skill_ci_low) * 100, (d.skill_ci_high - d.skill_vs_persistence) * 100],
               capsize=3, error_kw={"elinewidth": 1, "ecolor": INK2})
    ax.axhline(0, color=INK2, linewidth=1, linestyle="--")
    ax.set_xticks(range(len(hs)), [f"{h} min" for h in hs])
    _style(ax, "Feature sets, deployed model family (test, seen nodes)", "Skill over persistence (%)", "Horizon")
    ax.legend(frameon=False, fontsize=8)
    _source_note(fig, results)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path = out / "fig_feature_ablation.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def fig_intervals(results: Path, out: Path) -> Path:
    m = pd.read_csv(results / "forecast_metrics.csv")
    summ = json.loads((results / "forecast_summary.json").read_text())
    chosen = {int(k): v for k, v in summ["chosen_model_by_horizon"].items()}
    d = pd.concat([m[(m.horizon_min == h) & (m.model == name) & (m.split == "test_seen")] for h, name in chosen.items()])
    d = d.sort_values("horizon_min")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7))
    ax = axes[0]
    ax.bar(range(len(d)), d.interval_coverage * 100, color=SERIES[0], width=0.55)
    ax.axhline(summ.get("interval_coverage", 0.9) * 100, color=INK2, linestyle="--", linewidth=1)
    ax.set_ylim(50, 100)
    ax.set_xticks(range(len(d)), [f"{h} min" for h in d.horizon_min])
    _style(ax, "Interval coverage on test (target dashed)", "% of test minutes inside", "Horizon")
    ax = axes[1]
    xs = range(len(d))
    pers = pd.concat([m[(m.horizon_min == h) & (m.model == "persistence") & (m.split == "test_seen")]
                      for h in d.horizon_min])
    ax.bar([x - 0.27 for x in xs], d.crossing_recall * 100, width=0.25, color="#b4b2a9", label="model, point forecast")
    ax.bar(list(xs), pers.upper_crossing_recall * 100, width=0.25, color=NEUTRAL, label="persistence, upper bound")
    ax.bar([x + 0.27 for x in xs], d.upper_crossing_recall * 100, width=0.25, color=SERIES[1], label="model, upper bound")
    for x, (pr, mr, pa, ma) in zip(xs, zip(pers.upper_crossing_precision, d.upper_crossing_precision,
                                           pers.upper_alarm_rate, d.upper_alarm_rate)):
        ax.text(x + 0.13, 101, f"alarm on {pa:.0%} / {ma:.0%}", ha="center", fontsize=6.3, color=INK2)
    ax.set_xticks(list(xs), [f"{h} min" for h in d.horizon_min])
    ax.set_ylim(0, 110)
    _style(ax, "Threshold crossings anticipated", "Recall (%)", "Horizon")
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    _source_note(fig, results)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path = out / "fig_intervals.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def fig_sensitivity(results: Path, out: Path) -> Path:
    se = pd.read_csv(results / "sensitivity_summary.csv")
    se = se[(se.metric == "caused_violation_pct") & (se.policy != "coolest_now")]
    kinds = [k for k in [("limit_percentile", "Thermal limit", "percentile of training temperatures"),
                         ("safety_margin_c", "Safety margin", "°C")] if k[0] in set(se.setting)]
    fig, axes = plt.subplots(1, len(kinds), figsize=(4.8 * len(kinds), 3.3), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, (kind, title, unit) in zip(axes, kinds):
        g = se[se.setting == kind]
        for i, (pol, gg) in enumerate(g.groupby("policy", sort=False)):
            gg = gg.sort_values("value")
            ax.errorbar(gg.value + (i - 0.5) * 0.3, gg.diff_vs_coolest_now,
                        yerr=[gg.diff_vs_coolest_now - gg.diff_ci_low, gg.diff_ci_high - gg.diff_vs_coolest_now],
                        color=POLICY_COLORS.get(pol, INK2), marker="o", capsize=3, linewidth=1.8,
                        label={"thermalview": "ThermalView", "predictive_persist": "Ablation (persistence)"}.get(pol, pol))
        ax.axhline(0, color=INK2, linewidth=1, linestyle="--")
        _style(ax, title, "Δ caused % vs coolest-now" if kind == "limit_percentile" else "", unit)
    axes[0].legend(frameon=False, fontsize=8)
    _source_note(fig, results)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path = out / "fig_sensitivity.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def fig_job_events(results: Path, out: Path) -> Path:
    src = results / "job_event_curve_transfer.csv"
    c = pd.read_csv(src if src.exists() else results / "job_event_curve.csv")
    fig, ax = plt.subplots(figsize=(6, 3.3))
    ax2 = ax.twinx()
    ax2.plot(c.minute, c.d_load, color=NEUTRAL, linewidth=1.6, label="CPU load (points)")
    ax.plot(c.minute, c.d_temp, color=SERIES[1], linewidth=2, label="CPU temperature (°C)")
    ax.fill_between(c.minute, c.d_temp - 1.96 * c.d_temp_se, c.d_temp + 1.96 * c.d_temp_se, color=SERIES[1], alpha=0.2)
    ax.axvline(0, color=INK2, linewidth=1, linestyle="--")
    _style(ax, f"Around {int(c.n.max()):,} real job starts", "Change in temperature (°C)", "Minutes from job start")
    ax2.set_ylabel("Change in load (points)", fontsize=9, color=INK2)
    ax2.tick_params(colors=INK2, labelsize=8)
    for side in ("top",):
        ax2.spines[side].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper left")
    _source_note(fig, results)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path = out / "fig_job_events.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def make_all(results: Path, log=print) -> list[Path]:
    out = results / "figures"
    out.mkdir(parents=True, exist_ok=True)
    made = [fig_overview(results, out)]
    log(f"  wrote {made[-1]}")
    jobs = [(fig_forecast_mae, "forecast_metrics.csv"), (fig_skill, "forecast_metrics.csv"),
            (fig_feature_ablation, "forecast_feature_ablation.csv"), (fig_intervals, "forecast_metrics.csv"),
            (fig_example, "forecast_predictions_sample.csv"), (fig_importance, "feature_importance.csv"),
            (fig_fidelity, "simulator_fidelity.csv"), (fig_placement, "placement_runs.csv"),
            (fig_sensitivity, "sensitivity_summary.csv"), (fig_job_events, "job_event_curve.csv")]
    for fn, needs in jobs:
        if not (results / needs).exists():
            log(f"  skipped {fn.__name__}: {needs} missing")
            continue
        try:
            made.append(fn(results, out))
            log(f"  wrote {made[-1]}")
        except (KeyError, AttributeError, ValueError) as exc:     # results from an older pipeline version
            log(f"  skipped {fn.__name__}: {exc!r}")
    return made
