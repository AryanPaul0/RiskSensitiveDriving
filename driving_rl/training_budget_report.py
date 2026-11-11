"""Create the training budget results page.

Read saved numbers, calculate clearly labeled summaries, draw figures, and write
an HTML page. This file presents evidence; it does not train or select models.
Long HTML strings are page templates. The reinforcement-learning calculations
live in agents.py and experiment.py, not inside the page layout.

Original implementation notes:
Scientific report for the predeclared training-budget comparison.
"""

import csv
import html
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .analysis import paired_matrix, crossed_bootstrap
from .calibration import summarize_rows
from .experiment import save_json

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "results/training_budget"
LABELS = {
    "dqn_mean": "DQN",
    "qrdqn_mean": "QR mean",
    "qrdqn_tail25": "QR lower 25%",
    "qrdqn_governed": "QR + fixed governor",
}


def report():
    """Build graphs and an HTML page from saved result files.
    The arrays below hold measured results; plotting does not change the models.
    HTML strings describe page layout, while Python inserts the saved numbers.
    Keep primary comparisons separate from exploratory comparisons in the report.
    """
    viewer = ROOT / "results/report"
    backup = ROOT / "results/risk_analysis-report"
    if not backup.exists():
        shutil.copytree(viewer, backup)
    groups = {}
    summaries = []
    for name in LABELS:
        algo, policy = name.split("_", 1)
        for budget in [20000, 40000]:
            data = [
                json.loads(
                    (STAGE / f"eval_{algo}_{s}_{budget}_{policy}.json").read_text()
                )
                for s in range(3)
            ]
            groups[name, budget] = data
            for density in [1.0, 1.5, 2.0]:
                per_seed = []
                for d in data:
                    rows = [r for r in d["episodes"] if r["density"] == density]
                    returns = sorted(r["return"] for r in rows)
                    per_seed.append(
                        dict(
                            seed=d["provenance"]["training_seed"],
                            crash_rate=np.mean([r["crashed"] for r in rows]),
                            speed=np.mean([r["mean_speed_m_s"] for r in rows]),
                            distance=np.mean([r["distance_m"] for r in rows]),
                            mean_return=np.mean(returns),
                            worst10=np.mean(returns[:3]),
                        )
                    )
                summaries.append(
                    dict(
                        policy=name,
                        budget=budget,
                        density=density,
                        per_seed=per_seed,
                        **{
                            k: float(np.mean([r[k] for r in per_seed]))
                            for k in [
                                "crash_rate",
                                "speed",
                                "distance",
                                "mean_return",
                                "worst10",
                            ]
                        },
                    )
                )
    comparisons = []
    for name in LABELS:
        for density in [1.0, 1.5, 2.0]:
            row = dict(policy=name, density=density)
            # Compare one measured quantity at a time, keeping the scenarios paired.
            for metric in ["crashed", "mean_speed_m_s", "distance_m", "return"]:
                _, matrix = paired_matrix(
                    groups[name, 40000], groups[name, 20000], density, metric
                )
                row[metric] = crossed_bootstrap(matrix)
            comparisons.append(row)
    controls = []
    for density in [1.0, 1.5, 2.0]:
        item = dict(density=density)
        # Compare one measured quantity at a time, keeping the scenarios paired.
        for metric in ["crashed", "mean_speed_m_s"]:
            _, matrix = paired_matrix(
                groups["qrdqn_tail25", 40000],
                groups["qrdqn_governed", 40000],
                density,
                metric,
            )
            item[metric] = crossed_bootstrap(matrix)
        controls.append(item)
    calibration = []
    for policy in ["mean", "tail25"]:
        for budget in [20000, 40000]:
            folder = ROOT / "results/risk_analysis" if budget == 20000 else STAGE
            data = [
                json.loads((folder / f"calibration_{policy}_seed{s}.json").read_text())
                for s in range(3)
            ]
            for density in [1.0, 1.5, 2.0]:
                rows = [
                    r for d in data for r in d["episodes"] if r["density"] == density
                ]
                calibration.append(
                    dict(
                        policy=policy,
                        budget=budget,
                        density=density,
                        **summarize_rows(rows),
                    )
                )
    prefix = json.loads((STAGE / "prefix_verification.json").read_text())
    reproduction = []
    for algo in ["dqn", "qrdqn"]:
        for seed in range(3):
            for policy in (["mean"] if algo == "dqn" else ["mean", "tail25"]):
                original = json.loads(
                    (ROOT / f"results/eval_{algo}_seed{seed}_{policy}.json").read_text()
                )
                fresh = groups[f"{algo}_{policy}", 20000][seed]
                equal = original["episodes"] == fresh["episodes"]
                reproduction.append(
                    dict(
                        algorithm=algo,
                        seed=seed,
                        policy=policy,
                        exact_episode_match=equal,
                    )
                )
                if not equal:
                    raise ValueError("20k episode reproduction failed")
    save_json(STAGE / "evaluation_reproduction.json", reproduction)
    result = dict(
        evaluation_reproduction=reproduction,
        groups=summaries,
        budget_differences=comparisons,
        same_network_control=controls,
        calibration=calibration,
        prefix_verification=prefix,
        confirmation_seeds_used=False,
    )
    save_json(STAGE / "summary.json", result)
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    for col, density in enumerate([1.0, 1.5, 2.0]):
        for i, name in enumerate(LABELS):
            points = [
                r for r in summaries if r["policy"] == name and r["density"] == density
            ]
            color = f"C{i}"
            for row, metric, scale in [(0, "crash_rate", 100), (1, "speed", 1)]:
                ax = axes[row, col]
                ax.plot(
                    [20, 40],
                    [r[metric] * scale for r in points],
                    "-o",
                    color=color,
                    label=LABELS[name],
                )
                for seed in range(3):
                    ax.plot(
                        [20, 40],
                        [r["per_seed"][seed][metric] * scale for r in points],
                        color=color,
                        alpha=0.25,
                        lw=0.8,
                    )
                ax.grid(alpha=0.2)
                ax.set_xticks([20, 40])
                ax.set_xlabel("Training steps (thousands)")
        axes[0, col].set_title(f"Traffic density {density:g}")
        axes[0, col].set_ylim(0, 105)
    axes[0, 0].set_ylabel("Episodes with collision (%)")
    axes[1, 0].set_ylabel("Mean episode speed (m/s)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "More training, fixed exploration schedule; faint lines = individual training seeds"
    )
    fig.tight_layout()
    # Save a figure for the local results page; this does not affect agent behavior.
    fig.savefig(viewer / "training_budget_budget.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, policy in zip(axes, ["mean", "tail25"]):
        for budget, color, offset in [(20000, "C0", -0.04), (40000, "C1", 0.04)]:
            rows = [
                r
                for r in calibration
                if r["policy"] == policy and r["budget"] == budget
            ]
            lo = np.array([r["mean_prediction_bias_bounds"][0] for r in rows])
            hi = np.array([r["mean_prediction_bias_bounds"][1] for r in rows])
            mid = (lo + hi) / 2
            ax.errorbar(
                np.array([1.0, 1.5, 2.0]) + offset,
                mid,
                yerr=[mid - lo, hi - mid],
                fmt="o-",
                capsize=4,
                label=f"{budget//1000}k",
                color=color,
            )
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(f"QR {policy} deployment")
        ax.set_xlabel("Traffic density")
        ax.set_xticks([1, 1.5, 2])
        ax.legend()
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Prediction minus discounted return")
    fig.suptitle("Censoring bounds only; whiskers are not confidence intervals")
    fig.tight_layout()
    # Save a figure for the local results page; this does not affect agent behavior.
    fig.savefig(viewer / "training_budget_bias.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, algo in zip(axes, ["dqn", "qrdqn"]):
        for seed in range(3):
            with (STAGE / f"{algo}_seed{seed}/training.csv").open() as f:
                logs = list(csv.DictReader(f))
            steps = np.array([int(r["step"]) for r in logs])
            returns = np.array([float(r["return"]) for r in logs])
            if len(returns) >= 100:
                ax.plot(
                    steps[99:] / 1000,
                    np.convolve(returns, np.ones(100) / 100, mode="valid"),
                    label=f"Seed {seed}",
                )
        ax.axvline(20, color="black", ls="--", lw=1)
        ax.set_title(algo.upper())
        ax.set_xlabel("Training steps (thousands)")
        ax.grid(alpha=0.2)
        ax.legend()
    axes[0].set_ylabel("Training return: trailing 100 episodes")
    fig.suptitle("Exploratory training returns include epsilon-greedy actions")
    fig.tight_layout()
    # Save a figure for the local results page; this does not affect agent behavior.
    fig.savefig(viewer / "training_budget_learning.png", dpi=150)
    plt.close(fig)

    def table(headers, rows):
        """Build the HTML for a table of saved numbers. Headers label columns; each
        row supplies values in the same order. This changes presentation, not the data.
        """
        return (
            '<div class="scroll"><table><thead><tr>'
            + "".join("<th>" + html.escape(x) + "</th>" for x in headers)
            + "</tr></thead><tbody>"
            + "".join(
                "<tr>" + "".join("<td>" + str(x) + "</td>" for x in row) + "</tr>"
                for row in rows
            )
            + "</tbody></table></div>"
        )

    body = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>training budget experiment · Risk-sensitive driving</title><style>body{font:17px/1.6 system-ui;color:#18283d;background:#f4f7fb;margin:auto;max-width:1160px;padding:36px}h1{font-size:38px;line-height:1.2}section{background:white;padding:24px;margin:24px 0;border-radius:14px}img{width:100%;height:auto}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;text-align:left;border-bottom:1px solid #dde3eb}th{background:#eef3fa}a{color:#175bbc}.note{color:#47576b}</style></head><body><p>training budget experiment / CONTROLLED TRAINING BUDGET</p><h1>Does more experience improve risk-sensitive driving?</h1><p>Six training trajectories, two checkpoint budgets, and a fixed exploration schedule. This is a development study; confirmation scenarios remain reserved.</p><p><a href="../risk_analysis-report/index.html">risk analysis: interactive risk frontier, calibration and replays</a> · <a href="../training_budget/summary.json">Numerical results</a></p>"""
    normal = {(r["policy"], r["budget"]): r for r in summaries if r["density"] == 1.0}
    highlights = []
    for name in LABELS:
        a, b = normal[name, 20000], normal[name, 40000]
        highlights.append(
            f"{LABELS[name]}: collisions {100*a['crash_rate']:.1f}% to {100*b['crash_rate']:.1f}%; speed {a['speed']:.2f} to {b['speed']:.2f} m/s."
        )
    body += (
        "<section><h2>Normal-traffic findings</h2><ul>"
        + "".join("<li>" + x + "</li>" for x in highlights)
        + "</ul><p>Budget effects are paired within each trajectory. These development comparisons do not establish generalization to reserved scenarios or real driving.</p></section>"
    )
    body += '<section><h2>Training budget and physical performance</h2><p>Networks are trained from scratch for 40,000 steps. Epsilon reaches .05 at step 12,001 in both budgets. Reward, traffic, architecture and optimizer are unchanged. Solid lines average three training seeds; faint lines show their paired checkpoints.</p><img src="training_budget_budget.png"></section>'
    body += '<section><h2>Learning curves</h2><p>Training rewards include exploration and variable episode lengths; use frozen-policy evaluations below to judge performance.</p><img src="training_budget_learning.png"></section>'
    body += (
        "<section><h2>All results</h2><p>30 shared development scenarios per density per model. Speed is the mean of episode-average speeds. Worst 10% means the three lowest undiscounted episode returns per seed, then averaged across seeds.</p>"
        + table(
            [
                "Policy",
                "Steps",
                "Density",
                "Collision",
                "Speed m/s",
                "Distance m",
                "Return",
                "Worst 10%",
            ],
            [
                [
                    LABELS[r["policy"]],
                    r["budget"],
                    r["density"],
                    f"{100*r['crash_rate']:.1f}%",
                    f"{r['speed']:.2f}",
                    f"{r['distance']:.0f}",
                    f"{r['mean_return']:.2f}",
                    f"{r['worst10']:.2f}",
                ]
                for r in summaries
            ],
        )
        + "</section>"
    )
    body += (
        "<section><h2>Paired budget effects: 40k minus 20k</h2><p>Negative collision differences favor 40k. Exploratory 95% intervals resample training seeds and shared scenarios as crossed factors, preserving checkpoint pairing (5,000 replicates). Three training seeds give limited uncertainty estimates. Repeated development evaluation is not a confirmatory test.</p>"
        + table(
            [
                "Policy",
                "Density",
                "Collision change pp",
                "Exploratory interval pp",
                "Per-seed change pp",
                "Speed change m/s",
            ],
            [
                [
                    LABELS[r["policy"]],
                    r["density"],
                    f"{r['crashed']['mean_difference']*100:+.1f}",
                    str(
                        [
                            round(x * 100, 1)
                            for x in r["crashed"]["exploratory_95_percent_interval"]
                        ]
                    ),
                    str(
                        [
                            round(x * 100, 1)
                            for x in r["crashed"]["per_training_seed_differences"]
                        ]
                    ),
                    f"{r['mean_speed_m_s']['mean_difference']:+.2f}",
                ]
                for r in comparisons
            ],
        )
        + "</section>"
    )
    body += (
        "<section><h2>Frozen speed control at 40k</h2><p>Lower quarter minus mean deployment with the old 25 m/s governor. This threshold was not reselected at 40k. Check residual speed differences before attributing collision changes to the risk rule.</p>"
        + table(
            [
                "Density",
                "Collision change pp",
                "Exploratory interval pp",
                "Per-seed change pp",
                "Speed change m/s",
            ],
            [
                [
                    r["density"],
                    f"{r['crashed']['mean_difference']*100:+.1f}",
                    str(
                        [
                            round(x * 100, 1)
                            for x in r["crashed"]["exploratory_95_percent_interval"]
                        ]
                    ),
                    str(
                        [
                            round(x * 100, 1)
                            for x in r["crashed"]["per_training_seed_differences"]
                        ]
                    ),
                    f"{r['mean_speed_m_s']['mean_difference']:+.2f}",
                ]
                for r in controls
            ],
        )
        + "</section>"
    )
    body += (
        '<section><h2>Return prediction diagnostics</h2><p>60 separate development scenarios per density per model. Successful rollouts right-censor the continuing return; whiskers bound missing continuation. These are not statistical confidence intervals. Lower-quarter deployment changes the continuation policy relative to mean-greedy training.</p><img src="training_budget_bias.png">'
        + table(
            [
                "Policy",
                "Steps",
                "Density",
                "Bias bounds",
                "Compatibility gap",
                "Adjacent crossings",
            ],
            [
                [
                    r["policy"],
                    r["budget"],
                    r["density"],
                    str([round(x, 3) for x in r["mean_prediction_bias_bounds"]]),
                    f"{r['minimum_mean_absolute_compatibility_gap']:.3f}",
                    f"{100*r['quantile_crossing_fraction']:.1f}%",
                ]
                for r in calibration
            ],
        )
        + "</section>"
    )
    body += (
        "<section><h2>Reproducibility and limits</h2><p>Exact original 20k online-weight matches: "
        + str(sum(r["prefix_weights_equal"] for r in prefix))
        + "/6. Checkpoints share training history and are not independent runs. Density randomization was deferred to isolate budget. Predicted quantiles are not safety probabilities; cautious selection has no global CVaR-optimality guarantee. HighwayEnv is a simplified simulator.</p><p>Run <code>python run_training_budget.py --jobs 3</code> and <code>python -m pytest -q</code>. Existing artifacts are validated before reuse. The original risk analysis viewer is preserved.</p></section></body></html>"
    )
    (viewer / "index.html").write_text(body)
    print("training budget experiment report saved", flush=True)


# Run the entry point only when this file is executed, not when it is imported.
if __name__ == "__main__":
    report()
