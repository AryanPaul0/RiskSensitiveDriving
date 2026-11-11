"""Explain the 100K study using simple tables, plots, and paired comparisons.

Nothing in this file trains a model or chooses a better-looking checkpoint.
The two budgets face the same scenarios, so we subtract matching outcomes.
"""

import json
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .analysis import summarize_group, paired_matrix, scenario_bootstrap_fixed_models
from .density_coverage_report import table
from .experiment import save_json

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/long_training_evaluation"
LABELS = {
    "qr_tail": "QR lower quarter",
    "dqn_governed": "DQN + governor",
    "qr_governed": "QR mean + governor",
    "dqn_mean": "DQN mean",
    "qr_mean": "QR mean",
}


def read(path):
    """Turn a saved JSON file into Python lists and dictionaries."""
    return json.loads(path.read_text())


def compare(left, right, density):
    """Subtract matching cases, then estimate uncertainty over fresh scenarios.

    The trained models stay fixed. Resample scenario IDs, not individual cells,
    because the same starting traffic is shared across the models.
    """
    result = {}
    # Compare one quantity at a time, using matching model and scenario IDs.
    for metric in ["crashed", "mean_speed_m_s", "distance_m", "return"]:
        seeds, differences = paired_matrix(left, right, density, metric)
        result[metric] = scenario_bootstrap_fixed_models(differences)
    return result


def load_and_compare_results():
    """Read matching cases, calculate differences, and save the full numerical summary."""
    design = read(OUTPUT / "design.json")
    data = {}
    groups = {}
    budget_effects = []
    policy_effects = []
    # Each budget gets its own results. Keep 40K and 100K separate until the paired comparison.
    for budget in [40000, 100000]:
        data[budget] = {}
        groups[budget] = {}
        # A policy is a decision rule. Several rules can share the very same QR-DQN network.
        for policy in LABELS:
            runs = []
            # A training seed identifies one independently trained model, not a new traffic scenario.
            for seed in range(3):
                path = OUTPUT / f"eval_{budget}_{policy}_seed{seed}.json"
                runs.append(read(path))
            data[budget][policy] = runs
            groups[budget][policy] = summarize_group(runs)
    # Compare budgets using matching scenario/model IDs, rather than comparing
    # the old 20000 block with the new 20100 block.
    for policy in LABELS:
        for density in design["densities"]:
            effect = compare(data[100000][policy], data[40000][policy], density)
            budget_effects.append({"policy": policy, "density": density, **effect})
    for reference in ["dqn_governed", "qr_governed", "dqn_mean", "qr_mean"]:
        for density in design["densities"]:
            effect = compare(data[100000]["qr_tail"], data[100000][reference], density)
            policy_effects.append(
                {"reference": reference, "density": density, **effect}
            )
    # Find the one comparison named in the recorded design.
    primary = None
    for comparison in policy_effects:
        if comparison["reference"] == "dqn_governed" and comparison["density"] == 1.0:
            primary = comparison
            break
    if primary is None:
        raise StopIteration
    upper = primary["crashed"]["conditional_95_percent_interval"][1]
    # Use the rule recorded before evaluating the fresh scenarios; do not move the goalposts afterward.
    criterion_met = upper < 0
    summary = {
        "design": design,
        "groups": groups,
        "budget_effects": budget_effects,
        "policy_effects_100k": policy_effects,
        "primary": primary,
        "primary_criterion_met": criterion_met,
        "prefix_verification": read(OUTPUT / "prefix_verification.json"),
    }
    # JSON object keys become text when saved; this is normal and intentional.
    save_json(OUTPUT / "summary.json", summary)
    return summary


def draw_budget_comparison(groups):
    """Plot the measured outcomes at both budgets. This is display code, not training."""
    viewer = ROOT / "results/report"
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    # A figure is the whole image; axes are its individual plotting panels.
    figure, axes = plt.subplots(2, 3, figsize=(14, 8))
    for column, policy in enumerate(["qr_tail", "dqn_governed", "qr_governed"]):
        for budget, color in [(40000, "C0"), (100000, "C1")]:
            rows = groups[budget][policy]
            densities = []
            for row in rows:
                densities.append(row["density"])
            for row_number, metric, scale in [
                (0, "crash_rate", 100),
                (1, "mean_speed_m_s", 1),
            ]:
                axis = axes[row_number, column]
                values = []
                for row in rows:
                    values.append(row[metric] * scale)
                axis.plot(
                    densities,
                    values,
                    "-o",
                    color=color,
                    label=f"{budget // 1000}K steps",
                )
                axis.set_xticks(densities)
                axis.set_xlabel("Traffic density")
                axis.grid(alpha=0.2)
        axes[0, column].set_title(LABELS[policy])
        axes[0, column].set_ylim(0, 105)
        axes[0, column].legend()
    axes[0, 0].set_ylabel("Episodes with collision (%)")
    axes[1, 0].set_ylabel("Mean episode speed (m/s)")
    figure.suptitle(
        "Same fresh scenarios, different training budgets; three models per learner"
    )
    figure.tight_layout()
    # Save the graph for the local report. Drawing it does not change any measured result.
    figure.savefig(viewer / "long_training_comparison.png", dpi=150)
    plt.close(figure)


def build_budget_tables(groups):
    """List the measured outcomes at each budget. This table does not choose a winner."""
    page = ""
    # Each budget gets its own results. Keep 40K and 100K separate until the paired comparison.
    for budget in [40000, 100000]:
        rows = []
        # A policy is a decision rule. Several rules can share the very same QR-DQN network.
        for policy in LABELS:
            for result in groups[budget][policy]:
                rows.append(
                    [
                        LABELS[policy],
                        result["density"],
                        f"{100 * result['crash_rate']:.1f}%",
                        f"{result['mean_speed_m_s']:.2f}",
                        f"{result['mean_distance_m']:.0f}",
                        f"{result['mean_return']:.2f}",
                    ]
                )
        page += f"<section><h2>{budget // 1000}K results on the new block</h2>"
        page += (
            table(
                [
                    "Policy",
                    "Density",
                    "Collisions",
                    "Speed m/s",
                    "Distance m",
                    "Mean return",
                ],
                rows,
            )
            + "</section>"
        )
    return page


def build_budget_effect_table(budget_effects):
    """Show 100K minus 40K. Negative collision differences favor the longer budget."""
    page = ""
    # Negative collision changes favor the longer training budget. Show speed
    # too: a reduced collision count might accompany slower driving.
    rows = []
    for result in budget_effects:
        crash = result["crashed"]
        # Convert this comparison interval into percentage points for the results table.
        budget_interval_percent = []
        for x in crash["conditional_95_percent_interval"]:
            budget_interval_percent.append(round(x * 100, 1))
        # Show each model pair too: a good average can hide differences between training seeds.
        seed_changes_percent = []
        for x in crash["per_training_seed_differences"]:
            seed_changes_percent.append(round(x * 100, 1))
        rows.append(
            [
                LABELS[result["policy"]],
                result["density"],
                f"{100 * crash['mean_difference']:+.1f}",
                str(budget_interval_percent),
                f"{result['mean_speed_m_s']['mean_difference']:+.2f}",
                str(seed_changes_percent),
            ]
        )
    page += "<section><h2>100K minus 40K on matching cases</h2><p>These budget comparisons are secondary. Intervals are conditional on the fixed model set and are not adjusted for multiple comparisons.</p>"
    page += (
        table(
            [
                "Policy",
                "Density",
                "Collision change pp",
                "Scenario interval pp",
                "Speed change m/s",
                "Per-model collision changes pp",
            ],
            rows,
        )
        + "</section>"
    )
    return page


def build_policy_effect_table(policy_effects):
    """Show cautious QR minus each reference, keeping speed differences visible."""
    page = ""
    rows = []
    for result in policy_effects:
        crash = result["crashed"]
        policy_interval_percent = []
        for x in crash["conditional_95_percent_interval"]:
            policy_interval_percent.append(round(x * 100, 1))
        rows.append(
            [
                LABELS[result["reference"]],
                result["density"],
                f"{100 * crash['mean_difference']:+.1f}",
                str(policy_interval_percent),
                f"{result['mean_speed_m_s']['mean_difference']:+.2f}",
            ]
        )
    page += "<section><h2>100K cautious QR minus each reference</h2>"
    page += (
        table(
            [
                "Reference",
                "Density",
                "Collision change pp",
                "Scenario interval pp",
                "Speed change m/s",
            ],
            rows,
        )
        + "</section>"
    )
    return page


def write_report_page(summary):
    """Build tables from the saved comparisons. The original primary criterion stays unchanged."""
    viewer = ROOT / "results/report"
    groups = summary["groups"]
    budget_effects = summary["budget_effects"]
    policy_effects = summary["policy_effects_100k"]
    primary = summary["primary"]
    # Use the rule recorded before evaluating the fresh scenarios; do not move the goalposts afterward.
    criterion_met = summary["primary_criterion_met"]
    interval = primary["crashed"]["conditional_95_percent_interval"]
    difference = primary["crashed"]["mean_difference"] * 100
    speed = primary["mean_speed_m_s"]["mean_difference"]
    status = "met" if criterion_met else "not met"
    # The long string below is page layout, not reinforcement learning. Later lines insert the measured numbers.
    page = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Longer training · Risk-sensitive driving</title><style>body{font:17px/1.6 system-ui;color:#18283d;background:#f4f7fb;margin:auto;max-width:1160px;padding:36px}h1{font-size:38px;line-height:1.2}section{background:white;padding:24px;margin:24px 0;border-radius:14px}img{width:100%;height:auto}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;text-align:left;border-bottom:1px solid #dde3eb}th{background:#eef3fa}a{color:#175bbc}</style></head><body><p>DEEP RL / LONGER TRAINING</p><h1>Did 100,000 training steps help?</h1><p>Six completed 100K runs are evaluated alongside their 40K versions on exactly the same fresh scenarios. No new training, risk-level selection, governor tuning or checkpoint selection happens in this study.</p><p><a href="../long_training_evaluation/design.json">Locked design</a> · <a href="../long_training_evaluation/summary.json">Full numerical results</a> · <a href="../confirmation_40k-report/index.html">Earlier confirmation study</a></p>"""
    page += "<section><h2>Primary 100K comparison</h2>"
    # Convert fractions such as 0.05 into percentage points such as 5.0 for the reader.
    primary_interval_percent = []
    for x in interval:
        primary_interval_percent.append(round(x * 100, 1))
    page += f"<p>The predeclared normal-traffic collision criterion is <strong>{status}</strong>. Cautious QR minus governed DQN: {difference:+.1f} percentage points; conditional 95% interval {primary_interval_percent} pp. Speed difference: {speed:+.2f} m/s.</p>"
    page += "<p>The interval estimates scenario uncertainty for these fixed models. It does not establish equivalence, speed-independent benefit, or reliability across new training runs.</p></section>"
    page += (
        "<section><h2>Interpreting the speed control</h2><p>The 25 m/s governor was frozen before this study. Its resulting average speed need not match a newly trained network. Here cautious QR differs from controlled DQN by "
        + f"{speed:+.2f}"
        + " m/s in normal traffic. This comparison is therefore not evidence that a risk rule alone caused a collision difference.</p><p>A primary criterion that is not met means the study has not established the specified advantage. It does not establish statistical equivalence.</p></section>"
    )
    page += '<section><h2>Paired budget comparison</h2><p>Both budgets use seeds 20100–20199: 100 scenario IDs and three trained models per policy. Those 300 rollouts are not treated as 300 independent starting scenarios. All six 40K network prefixes exactly match the previous trained models.</p><img src="long_training_comparison.png"></section>'
    page += build_budget_tables(groups)
    page += build_budget_effect_table(budget_effects)
    page += build_policy_effect_table(policy_effects)
    page += "<section><h2>Reproduce and interpret</h2><p><code>python run_long_training_evaluation.py --jobs 4</code> validates and reuses saved evaluations. <code>--report-only</code> rebuilds this page.</p><p>The primary comparison is governed DQN versus cautious QR at density 1 with 100K models. All other contrasts are secondary/exploratory. Policies share the configured training-density mixture, but their experiences differ. Only three training seeds are available.</p><p>Seeds 20100–20199 are now consumed evaluation data, alongside the earlier 20000–20099 block. Any later tuning requires a new untouched confirmation block beginning at 20200. Partial observations, prediction error and simplified traffic limit conclusions; no real-road safety guarantee is established.</p></section></body></html>"
    (viewer / "index.html").write_text(page)
    print(
        f"100K primary criterion {status}: {difference:+.1f} pp; interval {primary_interval_percent}; speed {speed:+.2f} m/s",
        flush=True,
    )


def report():
    """Keep the earlier page, then read, compare, draw, and write."""
    viewer = ROOT / "results/report"
    backup = ROOT / "results/confirmation_40k-report"
    if not backup.exists():
        shutil.copytree(viewer, backup)
    summary = load_and_compare_results()
    draw_budget_comparison(summary["groups"])
    write_report_page(summary)


if __name__ == "__main__":
    report()
