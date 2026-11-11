"""Create the baseline controls results page.

Read saved numbers, calculate clearly labeled summaries, draw figures, and write
an HTML page. This file presents evidence; it does not train or select models.
Long HTML strings are page templates. The reinforcement-learning calculations
live in agents.py and experiment.py, not inside the page layout.

Original implementation notes:
Distributional and scalar baselines with equal traffic coverage and speed controls.
"""

from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .analysis import summarize_group, paired_matrix, crossed_bootstrap
from .experiment import save_json
from .density_coverage_report import table

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "results/baseline_controls"
LABELS = {
    "fixed_dqn": "Fixed DQN",
    "mixed_dqn": "Mixed DQN",
    "mixed_qr_mean": "Mixed QR mean",
    "mixed_qr_tail": "Mixed QR lower quarter",
    "matched_dqn": "Speed-targeted DQN",
    "matched_qr": "Speed-targeted QR mean",
}


def read(p):
    """Read saved JSON data into ordinary Python dictionaries and lists."""
    return json.loads(p.read_text())


def load(group, block, selections):
    """Read the result files belonging to one comparison group. The group controls
    which saved model and decision rule are represented; it does not train a model.
    """
    paths = []
    for seed in range(3):
        if group == "fixed_dqn":
            p = (
                ROOT / f"results/training_budget/eval_dqn_{seed}_40000_mean.json"
                if block == "main"
                else STAGE / f"eval_dqn_fixed_mean_seed{seed}_transfer.json"
            )
        elif group == "mixed_dqn":
            p = STAGE / f"eval_dqn_mixed_mean_seed{seed}_{block}.json"
        elif group in ["mixed_qr_mean", "mixed_qr_tail"]:
            policy = "mean" if group == "mixed_qr_mean" else "tail25"
            p = (
                ROOT
                / f"results/density_coverage/eval_mixed_{policy}_seed{seed}_{block}.json"
            )
        else:
            algo = "dqn" if group == "matched_dqn" else "qrdqn"
            cap = selections[algo]["selected_cap_m_s"]
            token = f"cap{cap:g}".replace(".", "p")
            p = STAGE / f"eval_{algo}_mixed_{token}_seed{seed}_{block}.json"
        paths.append(p)
    return [read(p) for p in paths]


def report():
    """Build graphs and an HTML page from saved result files.
    The arrays below hold measured results; plotting does not change the models.
    HTML strings describe page layout, while Python inserts the saved numbers.
    Keep primary comparisons separate from exploratory comparisons in the report.
    """
    viewer = ROOT / "results/report"
    backup = ROOT / "results/density_coverage-report"
    if not backup.exists():
        shutil.copytree(viewer, backup)
    selections = {
        algo: read(STAGE / f"{algo}_governor_selection.json")
        for algo in ["dqn", "qrdqn"]
    }
    data = {}
    groups = {}
    comparisons = []
    exposure = {}
    mean_comparisons = []
    for block, densities in [("main", [1.0, 1.5, 2.0]), ("transfer", [1.25, 2.5])]:
        data[block] = {g: load(g, block, selections) for g in LABELS}
        groups[block] = {g: summarize_group(data[block][g]) for g in LABELS}
        for other in ["mixed_dqn", "mixed_qr_mean", "matched_dqn", "matched_qr"]:
            for density in densities:
                item = {"block": block, "reference": other, "density": density}
                # Compare one measured quantity at a time, keeping the scenarios paired.
                for metric in ["crashed", "mean_speed_m_s", "distance_m", "return"]:
                    _, matrix = paired_matrix(
                        data[block]["mixed_qr_tail"],
                        data[block][other],
                        density,
                        metric,
                    )
                    item[metric] = crossed_bootstrap(matrix)
                comparisons.append(item)
        for label, left, right in [
            ("Mean selectors", "mixed_qr_mean", "mixed_dqn"),
            ("Speed-targeted mean selectors", "matched_qr", "matched_dqn"),
        ]:
            for density in densities:
                item = {"block": block, "comparison": label, "density": density}
                # Compare one measured quantity at a time, keeping the scenarios paired.
                for metric in ["crashed", "mean_speed_m_s"]:
                    _, matrix = paired_matrix(
                        data[block][left], data[block][right], density, metric
                    )
                    item[metric] = crossed_bootstrap(matrix)
                mean_comparisons.append(item)
    for algo in ["dqn", "qrdqn"]:
        folder = STAGE if algo == "dqn" else ROOT / "results/density_coverage"
        exposure[algo] = [
            {"seed": s, **read(folder / f"{algo}_seed{s}/summary.json")}
            for s in range(3)
        ]
        if any(sum(r["steps_by_density"].values()) != 40000 for r in exposure[algo]):
            raise ValueError("Unequal interaction budgets")
    save_json(
        STAGE / "summary.json",
        {
            "design": read(STAGE / "design.json"),
            "groups": groups,
            "lower_quarter_minus_reference": comparisons,
            "selections": selections,
            "training_exposure": exposure,
            "exploratory_mean_selector_comparisons": mean_comparisons,
        },
    )
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    colors = {g: f"C{i}" for i, g in enumerate(LABELS)}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for group in LABELS:
        rows = groups["main"][group]
        x = [r["density"] for r in rows]
        for ax, metric, scale, title in zip(
            axes,
            ["crash_rate", "mean_speed_m_s", "mean_distance_m"],
            [100, 1, 1],
            [
                "Episodes with collision (%)",
                "Mean episode speed (m/s)",
                "Mean distance (m)",
            ],
        ):
            ax.plot(
                x,
                [r[metric] * scale for r in rows],
                "-o",
                color=colors[group],
                label=LABELS[group],
            )
            ax.set_xlabel("Evaluation density")
            ax.set_xticks(x)
            ax.set_ylabel(title)
            ax.grid(alpha=0.2)
    axes[0].set_ylim(0, 105)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "40k interactions per model; identical configured traffic mixture for mixed learners"
    )
    fig.tight_layout()
    # Save a figure for the local results page; this does not affect agent behavior.
    fig.savefig(viewer / "baseline_controls_performance.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, density in zip(axes, [1.0, 1.5, 2.0]):
        for group in LABELS:
            row = next(r for r in groups["main"][group] if r["density"] == density)
            for seed in row["per_seed"]:
                ax.scatter(
                    seed["mean_speed_m_s"],
                    100 * seed["crash_rate"],
                    color=colors[group],
                    s=18,
                    alpha=0.35,
                )
            ax.scatter(
                row["mean_speed_m_s"],
                100 * row["crash_rate"],
                color=colors[group],
                s=75,
                marker="D",
                label=LABELS[group],
            )
        ax.set_title(f"Density {density:g}")
        ax.set_xlabel("Mean episode speed (m/s)")
        ax.set_ylim(-3, 105)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Episodes with collision (%)")
    axes[0].legend(fontsize=7)
    fig.suptitle(
        "Speed/collision tradeoffs; diamonds = means, small dots = training seeds"
    )
    fig.tight_layout()
    # Save a figure for the local results page; this does not affect agent behavior.
    fig.savefig(viewer / "baseline_controls_tradeoff.png", dpi=150)
    plt.close(fig)
    document = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Scalar baselines and speed controls · Risk-sensitive driving</title><style>body{font:17px/1.6 system-ui;color:#18283d;background:#f4f7fb;margin:auto;max-width:1160px;padding:36px}h1{font-size:38px;line-height:1.2}section{background:white;padding:24px;margin:24px 0;border-radius:14px}img{width:100%;height:auto}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;text-align:left;border-bottom:1px solid #dde3eb}th{background:#eef3fa}a{color:#175bbc}</style></head><body><p>DEEP RL / STRONGER BASELINES</p><h1>Does cautious selection outperform a comparable scalar learner?</h1><p>DQN and QR-DQN now both train for 40,000 interactions on the same configured episode-density mixture: 1, 1.5 and 2. Rewards, hidden-layer sizes, optimizer settings and exploration schedules are held fixed. Experience remains policy-dependent; there are three training seeds per learner.</p><p><a href="../density_coverage-report/index.html">Coverage comparison, return diagnostics and replays</a> · <a href="../baseline_controls/summary.json">Numerical results</a> · <a href="../baseline_controls/design.json">Recorded design</a></p>"""
    headlines = []
    for density in [1.0, 2.0]:
        contrast = next(
            r
            for r in comparisons
            if r["block"] == "main"
            and r["reference"] == "matched_dqn"
            and r["density"] == density
        )
        diff = contrast["crashed"]
        interval = [round(100 * x, 1) for x in diff["exploratory_95_percent_interval"]]
        headlines.append(
            f"At density {density:g}, cautious QR minus speed-targeted DQN is {100*diff['mean_difference']:+.1f} collision percentage points (exploratory interval {interval}), with a speed difference of {contrast['mean_speed_m_s']['mean_difference']:+.2f} m/s."
        )
    document += (
        "<section><h2>Main control comparison</h2>"
        + "".join("<p>" + line + "</p>" for line in headlines)
        + "<p>The full seed results and residual speed differences below are needed to interpret these comparisons.</p></section>"
    )
    document += '<section><h2>Equal-coverage comparison</h2><img src="baseline_controls_performance.png"><p>Mixed learners share the configured traffic distribution and step budget. The fixed-density DQN is retained to show its coverage effect. The cautious QR selector uses the lowest quarter of 32 outputs, without additional risk-level tuning.</p></section>'
    document += (
        "<section><h2>Seed-0 speed-targeted controls</h2><p>For each learner, select a cap using only seed-0 normal-traffic speed closeness to the cautious QR policy; then freeze it before evaluating seeds 1 and 2. The development threshold grid is 21–25 m/s. This does not match the full speed distribution or the action interventions.</p>"
        + table(
            ["Learner", "Selected cap m/s", "Target speed m/s", "Seed-0 residual m/s"],
            [
                [
                    a,
                    r["selected_cap_m_s"],
                    f"{r['target_speed_m_s']:.3f}",
                    f"{r['seed0_speed_difference_m_s']:+.3f}",
                ]
                for a, r in selections.items()
            ],
        )
        + '<img src="baseline_controls_tradeoff.png"></section>'
    )
    for block, title in [
        ("main", "Primary traffic scenarios"),
        ("transfer", "Intermediate and beyond-training densities"),
    ]:
        document += (
            "<section><h2>"
            + title
            + "</h2><p>"
            + (
                "Scenarios 10000–10029; densities 1, 1.5 and 2."
                if block == "main"
                else "Separate development scenarios 16000–16029; density 1.25 is intermediate and density 2.5 is beyond both training ranges."
            )
            + "</p>"
            + table(
                [
                    "Method",
                    "Density",
                    "Collisions",
                    "Speed m/s",
                    "Distance m",
                    "Mean return",
                    "Worst 10% return",
                ],
                [
                    [
                        LABELS[g],
                        r["density"],
                        f"{100*r['crash_rate']:.1f}%",
                        f"{r['mean_speed_m_s']:.2f}",
                        f"{r['mean_distance_m']:.0f}",
                        f"{r['mean_return']:.2f}",
                        f"{r['worst_10_percent_mean_return']:.2f}",
                    ]
                    for g in LABELS
                    for r in groups[block][g]
                ],
            )
            + "</section>"
        )
    document += (
        "<section><h2>Cautious QR minus each reference</h2><p>Negative collision differences favor cautious QR. Exploratory 95% intervals use 5,000 crossed bootstrap replicates over model seed IDs and shared scenarios, preserving policy pairing. Speed-control intervals condition on the selected cap and do not adjust for its selection. Three seeds and repeated development use limit inference.</p>"
        + table(
            [
                "Block",
                "Reference",
                "Density",
                "Collision change pp",
                "Exploratory interval pp",
                "Speed change m/s",
                "Per-seed collision change pp",
            ],
            [
                [
                    r["block"],
                    LABELS[r["reference"]],
                    r["density"],
                    f"{100*r['crashed']['mean_difference']:+.1f}",
                    str(
                        [
                            round(100 * x, 1)
                            for x in r["crashed"]["exploratory_95_percent_interval"]
                        ]
                    ),
                    f"{r['mean_speed_m_s']['mean_difference']:+.2f}",
                    str(
                        [
                            round(100 * x, 1)
                            for x in r["crashed"]["per_training_seed_differences"]
                        ]
                    ),
                ]
                for r in comparisons
            ],
        )
        + "</section>"
    )
    residual = []
    for reference in ["matched_dqn", "matched_qr"]:
        target = next(r for r in groups["main"]["mixed_qr_tail"] if r["density"] == 1.0)
        control = next(r for r in groups["main"][reference] if r["density"] == 1.0)
        for a, b in zip(target["per_seed"], control["per_seed"]):
            residual.append(
                [
                    LABELS[reference],
                    a["training_seed"],
                    f"{a['mean_speed_m_s']-b['mean_speed_m_s']:+.3f}",
                    f"{100*(a['crash_rate']-b['crash_rate']):+.1f}",
                ]
            )
    document += (
        "<section><h2>Exploratory comparison of the mean selectors</h2><p>QR mean minus scalar DQN, using the same training mixture. These secondary summaries were added after inspecting the primary results; no training or selection settings were changed. Distributional learning also changes the output head, loss and parameter count, so this does not isolate a single architectural cause.</p>"
        + table(
            [
                "Block",
                "Comparison",
                "Density",
                "Collision change pp",
                "Exploratory interval pp",
                "Speed change m/s",
            ],
            [
                [
                    r["block"],
                    r["comparison"],
                    r["density"],
                    f"{100*r['crashed']['mean_difference']:+.1f}",
                    str(
                        [
                            round(100 * x, 1)
                            for x in r["crashed"]["exploratory_95_percent_interval"]
                        ]
                    ),
                    f"{r['mean_speed_m_s']['mean_difference']:+.2f}",
                ]
                for r in mean_comparisons
            ],
        )
        + "</section>"
    )
    document += (
        "<section><h2>Normal-traffic residuals by seed</h2><p>Values are cautious minus control. Aggregate matching can hide substantial differences by model.</p>"
        + table(
            [
                "Reference",
                "Training seed",
                "Speed difference m/s",
                "Collision difference pp",
            ],
            residual,
        )
        + "</section>"
    )
    document += (
        "<section><h2>Actual mixed-training exposure</h2><p>Density is sampled uniformly by episode, not by transition. Different policies generate different episode lengths and exposure shares despite equal total interaction budgets.</p>"
        + table(
            [
                "Learner",
                "Seed",
                "Density-1 steps",
                "Density-1.5 steps",
                "Density-2 steps",
                "Completed episodes",
            ],
            [
                [
                    algo,
                    r["seed"],
                    r["steps_by_density"]["1.0"],
                    r["steps_by_density"]["1.5"],
                    r["steps_by_density"]["2.0"],
                    r["episodes"],
                ]
                for algo, rows in exposure.items()
                for r in rows
            ],
        )
        + "</section>"
    )
    document += "<section><h2>Reproduce and interpret</h2><p><code>python run_baseline_controls.py --jobs 3</code> validates and reuses completed runs. <code>--stage evaluate</code> skips training; <code>--stage report</code> rebuilds this viewer. <code>python -m pytest -q</code> checks learning, sampling and speed-selection rules.</p><p>All scenarios remain development data; confirmation seeds 20000 onward are untouched. Reward and alpha are not retuned. A speed governor is a decision rule rather than a physical speed guarantee. Partial observations, simplified traffic and forecast error limit conclusions. Neither lower-tail selection nor favorable simulator results guarantee real-road safety.</p></section></body></html>"
    (viewer / "index.html").write_text(document)
    print("Scalar baseline and speed-control report saved", flush=True)


# Run the entry point only when this file is executed, not when it is imported.
if __name__ == "__main__":
    report()
