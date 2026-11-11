"""Explain forecast accuracy without confusing it with driving performance.

All results here are development diagnostics. No model is trained or selected.
"""

from pathlib import Path
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .forecast_metrics import summarize_predictions
from .density_coverage_report import table
from .experiment import save_json

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/prediction_audit"
LABELS = {"dqn_mean": "DQN mean", "qr_mean": "QR mean", "qr_tail": "QR lower quarter"}
DENSITIES = [1.0, 1.5, 2.0, 2.5]


def find_summary(summaries, budget, policy, density):
    """Look through saved summaries until we find the requested group.

    Check all three labels: budget, policy, and density. A missing group is
    still an error, not a made-up zero result.
    """
    for summary in summaries:
        if summary["budget"] == budget:
            if summary["policy"] == policy:
                if summary["density"] == density:
                    return summary
    raise StopIteration


def load_and_summarize_results():
    """Read the saved rollouts and keep each policy/budget/density group separate."""
    summaries = []
    # Each budget gets its own results. Keep 40K and 100K separate until the paired comparison.
    for budget in [40000, 100000]:
        # A policy is a decision rule. Several rules can share the very same QR-DQN network.
        for policy in LABELS:
            all_rows = []
            # A training seed identifies one independently trained model, not a new traffic scenario.
            for seed in range(3):
                path = OUTPUT / f"audit_{budget}_{policy}_seed{seed}.json"
                data = json.loads(path.read_text())
                all_rows.extend(data["episodes"])
            # Traffic density changes vehicle spacing. It does not add more network inputs.
            for density in DENSITIES:
                rows = []
                for row in all_rows:
                    if row["density"] == density:
                        rows.append(row)
                summaries.append(
                    {
                        "budget": budget,
                        "policy": policy,
                        "density": density,
                        **summarize_predictions(rows),
                    }
                )
    save_json(
        OUTPUT / "summary.json",
        {
            "groups": summaries,
            "scope": "Initial chosen-action predictions on development scenarios. Error ranges account for cutoff continuation, not sampling uncertainty. Policy changes also change the outcomes being predicted.",
        },
    )
    return summaries


def draw_error_comparison(summaries):
    """Draw signed and absolute errors. Whiskers bound missing future rewards, not sample noise."""
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    # A figure is the whole image; axes are its individual plotting panels.
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), sharey="row")
    for column, policy in enumerate(LABELS):
        for budget, color, offset in [(40000, "C0", -0.03), (100000, "C1", 0.03)]:
            rows = []
            for summary in summaries:
                if summary["budget"] == budget and summary["policy"] == policy:
                    rows.append(summary)
            for row_number, metric in [
                (0, "bias_bounds"),
                (1, "absolute_error_bounds"),
            ]:
                lower_error_limits = []
                for row in rows:
                    lower_error_limits.append(row[metric][0])
                lows = np.array(lower_error_limits)
                upper_error_limits = []
                for row in rows:
                    upper_error_limits.append(row[metric][1])
                highs = np.array(upper_error_limits)
                centers = (lows + highs) / 2
                axis = axes[row_number, column]
                # Whiskers here describe a range. Read the plot label: they are not automatically statistical confidence intervals.
                axis.errorbar(
                    np.array(DENSITIES) + offset,
                    centers,
                    yerr=[centers - lows, highs - centers],
                    fmt="o-",
                    color=color,
                    capsize=4,
                    label=f"{budget // 1000}K",
                )
                axis.axhline(0, color="black", linewidth=0.7)
                axis.set_xticks(DENSITIES)
                axis.set_xlabel("Traffic density")
                axis.grid(alpha=0.2)
        axes[0, column].set_title(LABELS[policy])
        axes[0, column].legend()
    axes[0, 0].set_ylabel("Prediction minus return")
    axes[1, 0].set_ylabel("Mean absolute prediction error")
    figure.suptitle(
        "Missing-continuation bounds only; whiskers are NOT sampling confidence intervals"
    )
    figure.tight_layout()
    # Save the graph for the local report. Drawing it does not change any measured result.
    figure.savefig(OUTPUT / "forecast_errors.png", dpi=150)
    plt.close(figure)


def draw_quantile_checks(summaries):
    """Draw QR distribution checks. A DQN scalar is never drawn as a quantile distribution."""
    # A quantile curve is meaningful for QR-DQN, not for DQN's scalar mean.
    # A figure is the whole image; axes are its individual plotting panels.
    figure, axes = plt.subplots(2, 4, figsize=(15, 8), sharex=True, sharey=True)
    for row_number, policy in enumerate(["qr_mean", "qr_tail"]):
        for column, density in enumerate(DENSITIES):
            axis = axes[row_number, column]
            for budget, color in [(40000, "C0"), (100000, "C1")]:
                item = find_summary(summaries, budget, policy, density)
                check = item["quantile_checks"]
                axis.fill_between(
                    check["taus"],
                    check["coverage_lower"],
                    check["coverage_upper"],
                    color=color,
                    alpha=0.25,
                    label=f"{budget // 1000}K",
                )
                axis.plot(
                    check["taus"], check["coverage_lower"], color=color, linewidth=0.8
                )
                axis.plot(
                    check["taus"], check["coverage_upper"], color=color, linewidth=0.8
                )
            axis.plot([0, 1], [0, 1], "--", color="black", linewidth=0.8)
            axis.set_title(f"{LABELS[policy]}, density {density:g}")
            axis.set_xlabel("Assigned quantile fraction")
            axis.grid(alpha=0.2)
    axes[0, 0].set_ylabel("Empirical compatibility bracket")
    axes[1, 0].set_ylabel("Empirical compatibility bracket")
    axes[0, 0].legend()
    figure.suptitle(
        "Bands account for ties and cutoff censoring; not statistical confidence bands"
    )
    figure.tight_layout()
    # Save the graph for the local report. Drawing it does not change any measured result.
    figure.savefig(OUTPUT / "quantile_checks.png", dpi=150)
    plt.close(figure)


def write_report_page(summaries):
    """Turn the measured numbers into a page. No model is changed or selected here."""
    # The long string below is page layout, not reinforcement learning. Later lines insert the measured numbers.
    page = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Prediction audit</title><style>body{font:17px/1.6 system-ui;background:#f2f5fa;color:#172a40;max-width:1160px;margin:auto;padding:32px}section{background:white;padding:24px;border-radius:14px;margin:24px 0}img{width:100%}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;text-align:left;border-bottom:1px solid #dce4ef}a{color:#165bb5}</style></head><body><p>DEEP RL / DEVELOPMENT PREDICTION AUDIT</p><h1>Did longer training improve the predictions?</h1><p><a href="../report/index.html">Interactive decision viewer</a> · <a href="../long_training-report/index.html">Frozen driving-performance study</a> · <a href="summary.json">Raw summaries</a> · <a href="design.json">Recorded diagnostic design</a></p><section><h2>What is being checked?</h2><p>40K and 100K models predict the return of their initial chosen action, then drive under the stated rule. Each policy/budget/density uses three models and 60 shared development scenarios per model. Seeds 19000–19059 are development data. No further confirmation seeds are consumed.</p><p>Different policies can choose different actions and create different outcomes. This is an audit of forecasts under their deployed behavior, not a fixed-target test isolating predictor quality. QR-tail deployment also differs from the mean-based continuation rule used in training.</p></section><section><h2>Mean prediction error</h2><p>Positive signed error means optimism. Absolute errors avoid cancellation between overestimates and underestimates. A crash gives an exact return; the 30-second recording cutoff leaves a bounded unknown future. Whiskers bound that missing continuation, not statistical uncertainty.</p><img src="forecast_errors.png"></section>"""
    # Summarize two representative diagnostics without changing any policy.
    old_dense = find_summary(summaries, 40000, "qr_tail", 2.0)
    new_dense = find_summary(summaries, 100000, "qr_tail", 2.0)
    old_normal = find_summary(summaries, 40000, "qr_mean", 1.0)
    new_normal = find_summary(summaries, 100000, "qr_mean", 1.0)
    old_dense_limits = []
    for x in old_dense["absolute_error_bounds"]:
        old_dense_limits.append(round(x, 3))
    new_dense_limits = []
    for x in new_dense["absolute_error_bounds"]:
        new_dense_limits.append(round(x, 3))
    dense_text = (
        "At density 2, cautious QR absolute-error bounds change from "
        + str(old_dense_limits)
        + " at 40K to "
        + str(new_dense_limits)
        + " at 100K."
    )
    normal_text = f"In normal traffic, QR mean's empirical compatibility gap decreases from {old_normal['quantile_checks']['minimum_mean_absolute_compatibility_gap']:.3f} to {new_normal['quantile_checks']['minimum_mean_absolute_compatibility_gap']:.3f}."
    page += (
        "<section><h2>Main observation</h2><p>Longer training does not uniformly improve prediction quality. "
        + dense_text
        + " "
        + normal_text
        + "</p><p>These are descriptive results on this diagnostic block. Better driving performance and more accurate predictions are separate properties. Missing-continuation bounds do not provide a population confidence interval, and changing a policy also changes what it experiences.</p></section>"
    )
    rows = []
    for result in summaries:
        printed_bias_limits = []
        for x in result["bias_bounds"]:
            printed_bias_limits.append(round(x, 3))
        printed_absolute_limits = []
        for x in result["absolute_error_bounds"]:
            printed_absolute_limits.append(round(x, 3))
        rows.append(
            [
                LABELS[result["policy"]],
                result["budget"],
                result["density"],
                str(printed_bias_limits),
                str(printed_absolute_limits),
                f"{100 * result['censored_fraction']:.1f}%",
                f"{100 * result['crash_rate']:.1f}%",
            ]
        )
    page += "<section><h2>All empirical error bounds</h2>"
    page += (
        table(
            [
                "Policy",
                "Steps",
                "Density",
                "Signed bias bounds",
                "Absolute error bounds",
                "Censored",
                "Collisions",
            ],
            rows,
        )
        + "</section>"
    )
    page += '<section><h2>QR-DQN distribution checks</h2><p>The dashed line is the assigned quantile fraction. Bands show which empirical coverage values are compatible with ties and unobserved continuation. They are not sampling confidence bands. Raw quantile heads are not sorted here, so crossings remain visible. Good marginal agreement does not establish accurate conditional forecasts in every state.</p><img src="quantile_checks.png">'
    rows = []
    for result in summaries:
        if "quantile_checks" in result:
            check = result["quantile_checks"]
            rows.append(
                [
                    LABELS[result["policy"]],
                    result["budget"],
                    result["density"],
                    f"{check['minimum_mean_absolute_compatibility_gap']:.3f}",
                    f"{100 * result['quantile_crossing_fraction']:.1f}%",
                ]
            )
    page += (
        table(
            [
                "Policy",
                "Steps",
                "Density",
                "Minimum compatibility gap",
                "Adjacent crossings",
            ],
            rows,
        )
        + "</section>"
    )
    page += "<section><h2>Limits and reproduction</h2><p>DQN supplies one mean estimate, so no quantile-distribution test is invented for it. Small mean bias can hide large individual errors. These comparisons do not establish why a policy crashed or guarantee safe driving.</p><p><code>python run_prediction_audit.py --jobs 4</code> validates and reuses saved data. <code>--report-only</code> rebuilds this page. No model training, risk-level selection, or speed-cap tuning occurs.</p></section></body></html>"
    (OUTPUT / "index.html").write_text(page)
    print("Prediction audit report saved.", flush=True)


def report():
    """Four short steps: read, draw two figures, then write the page."""
    summaries = load_and_summarize_results()
    draw_error_comparison(summaries)
    draw_quantile_checks(summaries)
    write_report_page(summaries)


if __name__ == "__main__":
    report()
