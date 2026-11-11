"""Create the confirmation results page.

Read saved numbers, calculate clearly labeled summaries, draw figures, and write
an HTML page. This file presents evidence; it does not train or select models.
Long HTML strings are page templates. The reinforcement-learning calculations
live in agents.py and experiment.py, not inside the page layout.

Original implementation notes:
Locked-policy confirmation with conditional scenario-cluster uncertainty.
"""

from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .analysis import (
    summarize_group,
    paired_matrix,
    crossed_bootstrap,
    scenario_bootstrap_fixed_models,
)
from .density_coverage_report import table
from .experiment import save_json

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "results/confirmation"
LABELS = {
    "qr_tail": "QR lower quarter",
    "dqn_governed": "DQN + frozen governor",
    "qr_governed": "QR mean + frozen governor",
    "dqn_mean": "DQN mean",
    "qr_mean": "QR mean",
}


def read(p):
    """Read saved JSON data into ordinary Python dictionaries and lists."""
    return json.loads(p.read_text())


def development(policy, density):
    """Read the earlier development result for a policy and density.
    It is shown for comparison, not used to retune the confirmation policy.
    """
    rows = []
    for s in range(3):
        block = "main" if density != 2.5 else "transfer"
        if policy == "qr_tail":
            p = (
                ROOT
                / f"results/density_coverage/eval_mixed_tail25_seed{s}_{block}.json"
            )
        elif policy == "qr_mean":
            p = ROOT / f"results/density_coverage/eval_mixed_mean_seed{s}_{block}.json"
        else:
            algo = "qrdqn" if policy == "qr_governed" else "dqn"
            token = "mean" if policy == "dqn_mean" else "cap25"
            p = (
                ROOT
                / f"results/baseline_controls/eval_{algo}_mixed_{token}_seed{s}_{block}.json"
            )
        rows.append(next(r for r in read(p)["summary"] if r["density"] == density))
    return {
        "collision_fraction": float(np.mean([r["crash_rate"] for r in rows])),
        "speed_m_s": float(np.mean([r["mean_speed_m_s"] for r in rows])),
    }


def report():
    """Build graphs and an HTML page from saved result files.
    The arrays below hold measured results; plotting does not change the models.
    HTML strings describe page layout, while Python inserts the saved numbers.
    Keep primary comparisons separate from exploratory comparisons in the report.
    """
    viewer = ROOT / "results/report"
    backup = ROOT / "results/baseline_controls-report"
    if not backup.exists():
        shutil.copytree(viewer, backup)
    design = read(STAGE / "design.json")
    for configs in design["checkpoint_hashes"].values():
        for item in configs.values():
            if (
                hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest()
                != item["sha256"]
            ):
                raise ValueError("Frozen model changed")
    data = {
        p: [read(STAGE / f"eval_{p}_seed{s}.json") for s in range(3)] for p in LABELS
    }
    groups = {p: summarize_group(d) for p, d in data.items()}
    comparisons = []
    dev = []
    for reference in ["dqn_governed", "qr_governed", "dqn_mean", "qr_mean"]:
        for density in design["densities"]:
            item = {
                "reference": reference,
                "density": density,
                "primary": reference == "dqn_governed" and density == 1.0,
            }
            # Compare one measured quantity at a time, keeping the scenarios paired.
            for metric in ["crashed", "mean_speed_m_s", "distance_m", "return"]:
                _, matrix = paired_matrix(
                    data["qr_tail"], data[reference], density, metric
                )
                item[metric] = {
                    "fixed_models": scenario_bootstrap_fixed_models(matrix),
                    "exploratory_new_training": crossed_bootstrap(matrix),
                }
            comparisons.append(item)
    for policy in LABELS:
        for density in design["densities"]:
            current = next(r for r in groups[policy] if r["density"] == density)
            dev.append(
                {
                    "policy": policy,
                    "density": density,
                    "development": development(policy, density),
                    "confirmation": {
                        "collision_fraction": current["crash_rate"],
                        "speed_m_s": current["mean_speed_m_s"],
                    },
                }
            )
    primary = next(r for r in comparisons if r["primary"])
    criterion = (
        primary["crashed"]["fixed_models"]["conditional_95_percent_interval"][1] < 0
    )
    result = {
        "design": design,
        "groups": groups,
        "contrasts": comparisons,
        "primary_criterion_met": criterion,
        "primary": primary,
        "development_comparison": dev,
        "interpretation": "Primary inference is conditional on six selected trained networks and frozen deployment settings. Residual speeds/action interventions remain; no claim of speed-independent benefit, new-training robustness or real-road safety. Other contrasts are secondary/exploratory.",
        "consumed_confirmation_seeds": [20000, 20099],
        "future_untouched_seeds_begin": 20100,
    }
    save_json(STAGE / "summary.json", result)
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    colors = {
        "qr_tail": "C3",
        "dqn_governed": "C4",
        "qr_governed": "C5",
        "dqn_mean": "C1",
        "qr_mean": "C2",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for p in LABELS:
        rows = groups[p]
        x = [r["density"] for r in rows]
        for ax, key, scale, ylabel in zip(
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
                [r[key] * scale for r in rows],
                "-o",
                color=colors[p],
                label=LABELS[p],
            )
            ax.set_xticks(x)
            ax.set_xlabel("Traffic density")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.2)
    axes[0].set_ylim(0, 105)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Untouched confirmation scenarios: 100 scenario IDs, three fixed models per learner"
    )
    fig.tight_layout()
    # Save a figure for the local results page; this does not affect agent behavior.
    fig.savefig(viewer / "confirmation_performance.png", dpi=150)
    plt.close(fig)
    rows = [r for r in comparisons if r["density"] == 1.0]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for index, r in enumerate(rows):
        for ax, metric, scale in zip(axes, ["crashed", "mean_speed_m_s"], [100, 1]):
            for scope, offset, color, label in [
                ("fixed_models", -0.10, "C0", "Fixed-model scenario interval"),
                (
                    "exploratory_new_training",
                    0.10,
                    "#888888",
                    "Exploratory training/scenario interval",
                ),
            ]:
                v = r[metric][scope]
                key = (
                    "conditional_95_percent_interval"
                    if scope == "fixed_models"
                    else "exploratory_95_percent_interval"
                )
                lo, hi = np.array(v[key]) * scale
                mean = v["mean_difference"] * scale
                ax.errorbar(
                    mean,
                    index + offset,
                    xerr=[[max(0.0, mean - lo)], [max(0.0, hi - mean)]],
                    fmt="o",
                    color=color,
                    capsize=4,
                    label=label if index == 0 else None,
                )
    for ax, label in zip(
        axes, ["Collision difference (percentage points)", "Speed difference (m/s)"]
    ):
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel(label)
        ax.set_yticks(
            range(4),
            [LABELS[r["reference"]] + (" *" if r["primary"] else "") for r in rows],
        )
        ax.grid(alpha=0.2)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, fontsize=8, loc="lower center", ncol=2)
    fig.suptitle(
        "Normal traffic: cautious QR minus each reference; * = the single primary comparison"
    )
    fig.tight_layout(rect=[0, 0.09, 1, 0.95])
    # Save a figure for the local results page; this does not affect agent behavior.
    fig.savefig(viewer / "confirmation_intervals.png", dpi=150)
    plt.close(fig)
    primary_stats = primary["crashed"]["fixed_models"]
    interval = primary_stats["conditional_95_percent_interval"]
    speed = primary["mean_speed_m_s"]["fixed_models"]["mean_difference"]
    status = "met" if criterion else "not met"
    document = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Confirmation study · Risk-sensitive driving</title><style>body{font:17px/1.6 system-ui;color:#18283d;background:#f4f7fb;margin:auto;max-width:1160px;padding:36px}h1{font-size:38px;line-height:1.2}section{background:white;padding:24px;margin:24px 0;border-radius:14px}img{width:100%;height:auto}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;text-align:left;border-bottom:1px solid #dde3eb}th{background:#eef3fa}a{color:#175bbc}summary{cursor:pointer}</style></head><body><p>DEEP RL / LOCKED CONFIRMATION STUDY</p><h1>Do the development findings survive fresh scenarios?</h1><p>Six frozen mixed-density networks, five deployment rules, and no changes to training, checkpoints, risk level or speed thresholds after inspecting the reserved results. Scenario IDs 20000–20099 were verified unused before the design was locked.</p><p><a href="../confirmation/design.json">Locked design and model hashes</a> · <a href="../confirmation/preflight.json">Preflight audit</a> · <a href="../confirmation/summary.json">Full results</a> · <a href="../baseline_controls-report/index.html">Archived development comparisons</a> · <a href="../density_coverage-report/index.html">Training coverage and prediction diagnostics</a></p>"""
    document += (
        "<section><h2>Single primary comparison: normal-traffic collisions</h2><p>The predeclared criterion is <strong>"
        + status
        + "</strong>: cautious QR minus DQN with its frozen 25 m/s governor is "
        + f"{100*primary_stats['mean_difference']:+.1f}"
        + " percentage points, with an approximate conditional 95% interval of "
        + str([round(100 * v, 1) for v in interval])
        + " pp.</p><p>The average-speed difference is "
        + f"{speed:+.2f}"
        + " m/s. This is a comparison of the locked policies; a collision advantage does not establish a speed-independent effect.</p><p>The criterion requires the upper endpoint to be strictly below zero. It concerns new scenarios for this selected set of networks, not new training runs.</p></section>"
    )
    normal_tail = next(r for r in groups["qr_tail"] if r["density"] == 1.0)
    normal_dqn = next(r for r in groups["dqn_governed"] if r["density"] == 1.0)
    raw_text = f"Cautious QR: {100*normal_tail['crash_rate']:.1f}% collisions at {normal_tail['mean_speed_m_s']:.2f} m/s. Controlled DQN: {100*normal_dqn['crash_rate']:.1f}% at {normal_dqn['mean_speed_m_s']:.2f} m/s."
    seed_text = (
        "Per-model collision differences (QR minus controlled DQN) are "
        + str(
            [round(100 * v, 1) for v in primary_stats["per_training_seed_differences"]]
        )
        + " percentage points. The interval permits effects in either direction; equivalence was not tested."
    )
    document += (
        "<section><h2>Reading the primary result</h2><p>"
        + raw_text
        + "</p><p>"
        + seed_text
        + "</p></section>"
    )
    document += (
        '<section><h2>Frozen-policy performance</h2><p>Each policy/density row contains 300 rollouts: three training models evaluated on the same 100 fresh scenario IDs. Those 300 observations are not treated as 300 independent scenario samples.</p><img src="confirmation_performance.png">'
        + table(
            [
                "Policy",
                "Density",
                "Collisions",
                "Speed m/s",
                "Distance m",
                "Mean return",
                "Worst 10% return",
            ],
            [
                [
                    LABELS[p],
                    r["density"],
                    f"{100*r['crash_rate']:.1f}%",
                    f"{r['mean_speed_m_s']:.2f}",
                    f"{r['mean_distance_m']:.0f}",
                    f"{r['mean_return']:.2f}",
                    f"{r['worst_10_percent_mean_return']:.2f}",
                ]
                for p in LABELS
                for r in groups[p]
            ],
        )
        + "</section>"
    )
    document += '<section><h2>Two scopes of uncertainty</h2><p>Blue intervals resample scenario IDs while keeping all trained networks and policy pairings together. They estimate uncertainty over fresh scenarios conditional on the selected models. Gray intervals also resample training seed IDs; with only three model pairs, these remain exploratory. Only the starred collision contrast is primary; other effects are descriptive secondary comparisons without family-wise adjustment.</p><img src="confirmation_intervals.png"></section>'
    document += (
        "<section><h2>Cautious QR minus each reference</h2>"
        + table(
            [
                "Role",
                "Reference",
                "Density",
                "Collision change pp",
                "Conditional scenario interval pp",
                "Exploratory crossed interval pp",
                "Speed change m/s",
            ],
            [
                [
                    "Primary" if r["primary"] else "Secondary",
                    LABELS[r["reference"]],
                    r["density"],
                    f"{100*r['crashed']['fixed_models']['mean_difference']:+.1f}",
                    str(
                        [
                            round(100 * v, 1)
                            for v in r["crashed"]["fixed_models"][
                                "conditional_95_percent_interval"
                            ]
                        ]
                    ),
                    str(
                        [
                            round(100 * v, 1)
                            for v in r["crashed"]["exploratory_new_training"][
                                "exploratory_95_percent_interval"
                            ]
                        ]
                    ),
                    f"{r['mean_speed_m_s']['fixed_models']['mean_difference']:+.2f}",
                ]
                for r in comparisons
            ],
        )
        + "</section>"
    )
    document += (
        "<section><h2>Development versus confirmation</h2><p>Development used 30 scenarios per density/model and supported tuning and selection. Confirmation uses 100 fresh scenarios per density/model. The two blocks differ in initial traffic; these figures are shown without changing any policy.</p>"
        + table(
            [
                "Policy",
                "Density",
                "Development collisions",
                "Confirmation collisions",
                "Development speed",
                "Confirmation speed",
            ],
            [
                [
                    LABELS[r["policy"]],
                    r["density"],
                    f"{100*r['development']['collision_fraction']:.1f}%",
                    f"{100*r['confirmation']['collision_fraction']:.1f}%",
                    f"{r['development']['speed_m_s']:.2f}",
                    f"{r['confirmation']['speed_m_s']:.2f}",
                ]
                for r in dev
            ],
        )
        + "</section>"
    )
    document += (
        "<section><h2>Model variation</h2><details><summary>View every training seed</summary>"
        + table(
            [
                "Policy",
                "Density",
                "Training seed",
                "Collisions",
                "Speed m/s",
                "Distance m",
            ],
            [
                [
                    LABELS[p],
                    r["density"],
                    seed["training_seed"],
                    f"{100*seed['crash_rate']:.1f}%",
                    f"{seed['mean_speed_m_s']:.2f}",
                    f"{seed['mean_distance_m']:.0f}",
                ]
                for p in LABELS
                for r in groups[p]
                for seed in r["per_seed"]
            ],
        )
        + "</details></section>"
    )
    document += (
        '<section><h2>First reserved scenario, selected before outcomes</h2><p>Training seed 0, scenario 20000, density 1. This is one illustrative case, not evidence of typical performance.</p><div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px">'
        + "".join(
            "<article><h3>"
            + LABELS[p]
            + '</h3><img alt="Locked policy driving replay" src="../confirmation/replay_'
            + p
            + '.gif"></article>'
            for p in ["qr_tail", "dqn_governed", "qr_governed"]
        )
        + "</div></section>"
    )
    document += "<section><h2>Reproduce and retain the holdout boundary</h2><p><code>python run_confirmation.py --jobs 3</code> validates and reuses saved outputs. <code>--stage report</code> regenerates the report. Models and selection rules remain locked by hashes and recorded settings.</p><p>Seeds 20000–20099 are now consumed confirmation data for this locked study. Any further model tuning must use development blocks, and a later confirmation needs a new untouched block (20100 onward). Existing development viewers are historical snapshots from before this confirmation.</p><p>Observed speeds, action interventions, partial observations, simulator simplifications, forecast error and only three training seeds limit interpretation. Return quantiles are not collision probabilities. This experiment does not establish a globally CVaR-optimal policy or real-road safety.</p></section></body></html>"
    (viewer / "index.html").write_text(document)
    print(
        f'Primary criterion {status}: {100*primary_stats["mean_difference"]:+.1f} pp; conditional interval {[round(100*v,1) for v in interval]}; speed {speed:+.2f} m/s',
        flush=True,
    )
    print("Confirmation report saved", flush=True)


# Run the entry point only when this file is executed, not when it is imported.
if __name__ == "__main__":
    report()
