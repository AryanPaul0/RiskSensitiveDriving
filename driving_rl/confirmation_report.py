"""Create the confirmation results page.

Read saved numbers, calculate clearly labeled summaries, draw figures, and write
an HTML page. This file presents evidence; it does not train or select models.
Long HTML strings are page templates. The reinforcement-learning calculations
live in agents.py and experiment.py, not inside the page layout.

Original implementation notes:
Locked-policy confirmation with conditional scenario-cluster uncertainty.
"""

from pathlib import Path  # Make Path available for this file's calculations and workflow.
import hashlib  # Make hashlib available for this file's calculations and workflow.
import json  # Make json available for this file's calculations and workflow.
import shutil  # Make shutil available for this file's calculations and workflow.
import numpy as np  # Make numpy available for this file's calculations and workflow.
import matplotlib  # Make matplotlib available for this file's calculations and workflow.

matplotlib.use("Agg")  # Matplotlib.use applied to 'agg'.
import matplotlib.pyplot as plt  # Make matplotlib.pyplot available for this file's calculations and workflow.
from .analysis import (  # Make summarize_group, paired_matrix, crossed_bootstrap, scenario_bootstrap_fixed_models available for this file's calculations and workflow.
    summarize_group,  # Continue the calculation or input list started above.
    paired_matrix,  # Continue the calculation or input list started above.
    crossed_bootstrap,  # Continue the calculation or input list started above.
    scenario_bootstrap_fixed_models,  # Continue the calculation or input list started above.
)  # Finish assembling the grouped values above.
from .density_coverage_report import table  # Make table available for this file's calculations and workflow.
from .experiment import save_json  # Make save_json available for this file's calculations and workflow.

ROOT = Path(__file__).resolve().parents[1]  # Set ROOT to the entries of Path( file ).resolve().parents selected by 1.
STAGE = ROOT / "results/confirmation"  # Set STAGE to the path to 'results/confirmation' inside ROOT.
LABELS = {  # Set LABELS to the named settings or measurements assembled below.
    "qr_tail": "QR lower quarter",  # Record 'qr_tail' as 'QR lower quarter'.
    "dqn_governed": "DQN + frozen governor",  # Record 'dqn_governed' as 'DQN + frozen governor'.
    "qr_governed": "QR mean + frozen governor",  # Record 'qr_governed' as 'QR mean + frozen governor'.
    "dqn_mean": "DQN mean",  # Record 'dqn_mean' as 'DQN mean'.
    "qr_mean": "QR mean",  # Record 'qr_mean' as 'QR mean'.
}  # Finish assembling the grouped values above.


def read(p):  # Read saved JSON data into ordinary Python dictionaries and lists.
    """Read saved JSON data into ordinary Python dictionaries and lists."""  # Describe this operation for readers and help().
    return json.loads(p.read_text())  # Return json.loads applied to the result of p.read text.


def development(policy, density):  # Read the earlier development result for a policy and density.
    """Read the earlier development result for a policy and density.
    It is shown for comparison, not used to retune the confirmation policy.
    """
    rows = []  # Set rows to .
    for s in range(3):  # Process each s from range applied to 3.
        block = "main" if density != 2.5 else "transfer"  # Set block to 'main' when the traffic-density setting differs from 2.5, otherwise 'transfer'.
        if policy == "qr_tail":  # Use this branch when policy equals 'qr_tail'.
            p = (  # Set p to (ROOT divided by f'results/density coverage/eval mixed tail25 seed{s} {block}.json').
                ROOT  # Include (ROOT divided by f'results/density coverage/eval mixed tail25 seed{s} {block}.json') in the surrounding operation.
                / f"results/density_coverage/eval_mixed_tail25_seed{s}_{block}.json"  # Include f'results/density coverage/eval mixed tail25 seed{s} {block}.json' in the surrounding operation.
            )  # Finish assembling the grouped values above.
        elif policy == "qr_mean":  # Use this branch when policy equals 'qr_mean'.
            p = ROOT / f"results/density_coverage/eval_mixed_mean_seed{s}_{block}.json"  # Set p to (ROOT divided by f'results/density coverage/eval mixed mean seed{s} {block}.json').
        else:  # Handle the alternative when the earlier condition (policy equals 'qr_mean') is false.
            algo = "qrdqn" if policy == "qr_governed" else "dqn"  # Set algo to 'qrdqn' when policy equals 'qr_governed', otherwise 'dqn'.
            token = "mean" if policy == "dqn_mean" else "cap25"  # Set token to 'mean' when policy equals 'dqn_mean', otherwise 'cap25'.
            p = (  # Set p to (ROOT divided by f'results/baseline controls/eval {algo} mixed {token} seed{s} {block}.json').
                ROOT  # Include (ROOT divided by f'results/baseline controls/eval {algo} mixed {token} seed{s} {block}.json') in the surrounding operation.
                / f"results/baseline_controls/eval_{algo}_mixed_{token}_seed{s}_{block}.json"  # Include f'results/baseline controls/eval {algo} mixed {token} seed{s} {block}.json' in the surrounding operation.
            )  # Finish assembling the grouped values above.
        rows.append(next(r for r in read(p)["summary"] if r["density"] == density))  # Add next applied to r for each r in the entries of read(p) selected by 'summary', keeping only cases where r['density'] equals the traffic-density setting to rows.
    return {  # Return the named settings or measurements assembled below.
        "collision_fraction": float(np.mean([r["crash_rate"] for r in rows])),  # Record 'collision_fraction' as the average of the entries of r selected by 'crash_rate' for each r in rows as a floating-point value.
        "speed_m_s": float(np.mean([r["mean_speed_m_s"] for r in rows])),  # Record 'speed_m_s' as the average of the entries of r selected by 'mean_speed_m_s' for each r in rows as a floating-point value.
    }  # Finish assembling the grouped values above.


def report():  # Build graphs and an HTML page from saved result files.
    """Build graphs and an HTML page from saved result files.
    The arrays below hold measured results; plotting does not change the models.
    HTML strings describe page layout, while Python inserts the saved numbers.
    Keep primary comparisons separate from exploratory comparisons in the report.
    """
    viewer = ROOT / "results/report"  # Set viewer to the path to 'results/report' inside ROOT.
    backup = ROOT / "results/baseline_controls-report"  # Set backup to the path to 'results/baseline_controls-report' inside ROOT.
    if not backup.exists():  # Use this branch when not the result of backup.exists.
        shutil.copytree(viewer, backup)  # Shutil.copytree applied to viewer, backup.
    design = read(STAGE / "design.json")  # Set design to read applied to the path to 'design.json' inside STAGE.
    for configs in design["checkpoint_hashes"].values():  # Process each configs from the result of design['checkpoint hashes'].values.
        for item in configs.values():  # Process each item from the result of configs.values.
            if (  # Use this branch when the result of hashlib.sha256((ROOT / item['path']).read bytes()).hexdigest differs from the entries of item selected by 'sha256'.
                hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest()  # Include the result of hashlib.sha256((ROOT / item['path']).read bytes()).hexdigest differs from the entries of item selected by 'sha256' in the surrounding operation.
                != item["sha256"]  # Include the entries of item selected by 'sha256' in the surrounding operation.
            ):  # Finish assembling the grouped values above.
                raise ValueError("Frozen model changed")  # Reject this invalid case instead of continuing with misleading experiment results.
    data = {  # Set data to read applied to (STAGE divided by f'eval {p} seed{s}.json') for each s in range applied to 3 for each p in LABELS.
        p: [read(STAGE / f"eval_{p}_seed{s}.json") for s in range(3)] for p in LABELS  # Include p, read applied to (STAGE divided by f'eval {p} seed{s}.json') for each s in range applied to 3, p, LABELS in the surrounding operation.
    }  # Finish assembling the grouped values above.
    groups = {p: summarize_group(d) for p, d in data.items()}  # Set groups to summarize group applied to d for each p, d in the result of data.items.
    comparisons = []  # Set comparisons to .
    dev = []  # Set dev to .
    for reference in ["dqn_governed", "qr_governed", "dqn_mean", "qr_mean"]:  # Process each reference from 'dqn_governed', 'qr_governed', 'dqn_mean', 'qr_mean'.
        for density in design["densities"]:  # Process each the traffic-density setting from the entries of design selected by 'densities'.
            item = {  # Set item to the named settings or measurements assembled below.
                "reference": reference,  # Record 'reference' as reference.
                "density": density,  # Record 'density' as the traffic-density setting.
                "primary": reference == "dqn_governed" and density == 1.0,  # Record 'primary' as reference equals 'dqn_governed' and the traffic-density setting equals 1.0.
            }  # Finish assembling the grouped values above.
            for metric in ["crashed", "mean_speed_m_s", "distance_m", "return"]:  # Compare one measured quantity at a time, keeping the scenarios paired.
                _, matrix = paired_matrix(  # Set the unused result, matrix to paired matrix applied to the entries of data selected by 'qr_tail', the entries of data selected by reference, the traffic-density setting, metric.
                    data["qr_tail"], data[reference], density, metric  # Include the entries of data selected by 'qr_tail', the entries of data selected by reference, the traffic-density setting, metric in the surrounding operation.
                )  # Finish assembling paired matrix inputs.
                item[metric] = {  # Set the entries of item selected by metric to the named settings or measurements assembled below.
                    "fixed_models": scenario_bootstrap_fixed_models(matrix),  # Record 'fixed_models' as scenario bootstrap fixed models applied to matrix.
                    "exploratory_new_training": crossed_bootstrap(matrix),  # Record 'exploratory_new_training' as crossed bootstrap applied to matrix.
                }  # Finish assembling the grouped values above.
            comparisons.append(item)  # Add item to comparisons.
    for policy in LABELS:  # Process each policy from LABELS.
        for density in design["densities"]:  # Process each the traffic-density setting from the entries of design selected by 'densities'.
            current = next(r for r in groups[policy] if r["density"] == density)  # Set current to next applied to r for each r in the entries of groups selected by policy, keeping only cases where the entries of r selected by 'density' equals the traffic-density setting.
            dev.append(  # Add the named settings or measurements assembled below to dev.
                {  # Include the named settings or measurements assembled below in the surrounding operation.
                    "policy": policy,  # Record 'policy' as policy.
                    "density": density,  # Record 'density' as the traffic-density setting.
                    "development": development(policy, density),  # Record 'development' as development applied to policy, the traffic-density setting.
                    "confirmation": {  # Record 'confirmation' as the named settings or measurements assembled below.
                        "collision_fraction": current["crash_rate"],  # Record 'collision_fraction' as the entries of current selected by 'crash_rate'.
                        "speed_m_s": current["mean_speed_m_s"],  # Record 'speed_m_s' as the entries of current selected by 'mean_speed_m_s'.
                    },  # Finish assembling the grouped values above.
                }  # Finish assembling the grouped values above.
            )  # Finish assembling dev.append inputs.
    primary = next(r for r in comparisons if r["primary"])  # Set primary to next applied to r for each r in comparisons, keeping only cases where the entries of r selected by 'primary'.
    criterion = (  # Set criterion to the entries of the entries of the entries of primary['crashed'] selected by 'fixed_models' selected by 'conditional_95_percent_interval' selected by 1 is below 0.
        primary["crashed"]["fixed_models"]["conditional_95_percent_interval"][1] < 0  # Include the entries of the entries of the entries of primary['crashed'] selected by 'fixed_models' selected by 'conditional_95_percent_interval' selected by 1 is below 0 in the surrounding operation.
    )  # Finish assembling the grouped values above.
    result = {  # Set result to the named settings or measurements assembled below.
        "design": design,  # Record 'design' as design.
        "groups": groups,  # Record 'groups' as groups.
        "contrasts": comparisons,  # Record 'contrasts' as comparisons.
        "primary_criterion_met": criterion,  # Record 'primary_criterion_met' as criterion.
        "primary": primary,  # Record 'primary' as primary.
        "development_comparison": dev,  # Record 'development_comparison' as dev.
        "interpretation": "Primary inference is conditional on six selected trained networks and frozen deployment settings. Residual speeds/action interventions remain; no claim of speed-independent benefit, new-training robustness or real-road safety. Other contrasts are secondary/exploratory.",  # Record 'interpretation' as the report text.
        "consumed_confirmation_seeds": [20000, 20099],  # Record 'consumed_confirmation_seeds' as 20000, 20099.
        "future_untouched_seeds_begin": 20100,  # Record 'future_untouched_seeds_begin' as 20100.
    }  # Finish assembling the grouped values above.
    save_json(STAGE / "summary.json", result)  # Save the following results/settings as json: the path to 'summary.json' inside stage, result.
    plt.rcParams.update(  # Plt.rcparams.update applied to the named settings or measurements assembled below.
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}  # Record 'font.size' as 10.
    )  # Finish assembling plt.rcParams.update inputs.
    colors = {  # Set colors to the named settings or measurements assembled below.
        "qr_tail": "C3",  # Record 'qr_tail' as 'C3'.
        "dqn_governed": "C4",  # Record 'dqn_governed' as 'C4'.
        "qr_governed": "C5",  # Record 'qr_governed' as 'C5'.
        "dqn_mean": "C1",  # Record 'dqn_mean' as 'C1'.
        "qr_mean": "C2",  # Record 'qr_mean' as 'C2'.
    }  # Finish assembling the grouped values above.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))  # Set fig, axes to plt.subplots applied to 1, 3.
    for p in LABELS:  # Process each p from LABELS.
        rows = groups[p]  # Set rows to the entries of groups selected by p.
        x = [r["density"] for r in rows]  # Set x to the entries of r selected by 'density' for each r in rows.
        for ax, key, scale, ylabel in zip(  # Process each ax, key, scale, ylabel from matching entries grouped across axes, 'crash_rate', 'mean_speed_m_s', 'mean_distance_m', 100, 1, 1, 'Episodes with collision (%)', 'Mean episode speed (m/s)', 'Mean distance (m)'.
            axes,  # Include axes in the surrounding operation.
            ["crash_rate", "mean_speed_m_s", "mean_distance_m"],  # Include 'crash_rate', 'mean_speed_m_s', 'mean_distance_m' in the surrounding operation.
            [100, 1, 1],  # Include 100, 1, 1 in the surrounding operation.
            [  # Include 'Episodes with collision (%)', 'Mean episode speed (m/s)', 'Mean distance (m)' in the surrounding operation.
                "Episodes with collision (%)",  # Include 'Episodes with collision (%)' in the surrounding operation.
                "Mean episode speed (m/s)",  # Include 'Mean episode speed (m/s)' in the surrounding operation.
                "Mean distance (m)",  # Include 'Mean distance (m)' in the surrounding operation.
            ],  # Finish assembling the grouped values above.
        ):  # Finish assembling zip inputs.
            ax.plot(  # Ax.plot applied to x, (the entries of r selected by key times scale) for each r in rows, '-o'.
                x,  # Include x in the surrounding operation.
                [r[key] * scale for r in rows],  # Include (the entries of r selected by key times scale) for each r in rows in the surrounding operation.
                "-o",  # Include '-o' in the surrounding operation.
                color=colors[p],  # Include the entries of colors selected by p in the surrounding operation.
                label=LABELS[p],  # Include the entries of LABELS selected by p in the surrounding operation.
            )  # Finish assembling ax.plot inputs.
            ax.set_xticks(x)  # Ax.set xticks applied to x.
            ax.set_xlabel("Traffic density")  # Ax.set xlabel applied to 'traffic density'.
            ax.set_ylabel(ylabel)  # Ax.set ylabel applied to ylabel.
            ax.grid(alpha=0.2)  # The result of ax.grid.
    axes[0].set_ylim(0, 105)  # Axes[0].set ylim applied to 0, 105.
    axes[0].legend(fontsize=8)  # The result of axes[0].legend.
    fig.suptitle(  # Fig.suptitle applied to the report text.
        "Untouched confirmation scenarios: 100 scenario IDs, three fixed models per learner"  # Include the report text in the surrounding operation.
    )  # Finish assembling fig.suptitle inputs.
    fig.tight_layout()  # The result of fig.tight layout.
    fig.savefig(viewer / "confirmation_performance.png", dpi=150)  # Save a figure for the local results page; this does not affect agent behavior.
    plt.close(fig)  # Plt.close applied to fig.
    rows = [r for r in comparisons if r["density"] == 1.0]  # Set rows to r for each r in comparisons, keeping only cases where the entries of r selected by 'density' equals 1.0.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))  # Set fig, axes to plt.subplots applied to 1, 2.
    for index, r in enumerate(rows):  # Process each index, r from enumerate applied to rows.
        for ax, metric, scale in zip(axes, ["crashed", "mean_speed_m_s"], [100, 1]):  # Process each ax, metric, scale from matching entries grouped across axes, 'crashed', 'mean_speed_m_s', 100, 1.
            for scope, offset, color, label in [  # Process each scope, offset, color, label from 'fixed_models', negative 0.1, 'C0', 'Fixed-model scenario interval', 'exploratory_new_training', 0.1, '#888888', 'Exploratory training/scenario interval'.
                ("fixed_models", -0.10, "C0", "Fixed-model scenario interval"),  # Include 'fixed_models', negative 0.1, 'C0', 'Fixed-model scenario interval' in the surrounding operation.
                (  # Include 'exploratory_new_training', 0.1, '#888888', 'Exploratory training/scenario interval' in the surrounding operation.
                    "exploratory_new_training",  # Include 'exploratory_new_training' in the surrounding operation.
                    0.10,  # Include 0.1 in the surrounding operation.
                    "#888888",  # Include '#888888' in the surrounding operation.
                    "Exploratory training/scenario interval",  # Include 'Exploratory training/scenario interval' in the surrounding operation.
                ),  # Finish assembling the grouped values above.
            ]:  # Finish assembling the grouped values above.
                v = r[metric][scope]  # Set v to the entries of the entries of r selected by metric selected by scope.
                key = (  # Set key to 'conditional_95_percent_interval' when scope equals 'fixed_models', otherwise 'exploratory_95_percent_interval'.
                    "conditional_95_percent_interval"  # Include 'conditional_95_percent_interval' when scope equals 'fixed_models', otherwise 'exploratory_95_percent_interval' in the surrounding operation.
                    if scope == "fixed_models"  # Include scope equals 'fixed_models' in the surrounding operation.
                    else "exploratory_95_percent_interval"  # Include 'exploratory_95_percent_interval' in the surrounding operation.
                )  # Finish assembling the grouped values above.
                lo, hi = np.array(v[key]) * scale  # Set lo, hi to (the entries of v selected by key as a numerical array/tensor for batch calculations times scale).
                mean = v["mean_difference"] * scale  # Set mean to (the entries of v selected by 'mean_difference' times scale).
                ax.errorbar(  # Ax.errorbar applied to mean, (index plus offset).
                    mean,  # Include mean in the surrounding operation.
                    index + offset,  # Include (index plus offset) in the surrounding operation.
                    xerr=[[max(0.0, mean - lo)], [max(0.0, hi - mean)]],  # Include max applied to 0.0, (mean minus lo), max applied to 0.0, (hi minus mean) in the surrounding operation.
                    fmt="o",  # Include 'o' in the surrounding operation.
                    color=color,  # Include color in the surrounding operation.
                    capsize=4,  # Include 4 in the surrounding operation.
                    label=label if index == 0 else None,  # Include label when index equals 0, otherwise None in the surrounding operation.
                )  # Finish assembling ax.errorbar inputs.
    for ax, label in zip(  # Process each ax, label from matching entries grouped across axes, 'Collision difference (percentage points)', 'Speed difference (m/s)'.
        axes, ["Collision difference (percentage points)", "Speed difference (m/s)"]  # Include axes, 'Collision difference (percentage points)', 'Speed difference (m/s)' in the surrounding operation.
    ):  # Finish assembling zip inputs.
        ax.axvline(0, color="black", lw=0.8)  # Ax.axvline applied to 0.
        ax.set_xlabel(label)  # Ax.set xlabel applied to label.
        ax.set_yticks(  # Ax.set yticks applied to range applied to 4, (the entries of labels selected by r['reference'] plus ' *' when r['primary'], otherwise '') for each r in rows.
            range(4),  # Include range applied to 4 in the surrounding operation.
            [LABELS[r["reference"]] + (" *" if r["primary"] else "") for r in rows],  # Include (the entries of LABELS selected by r['reference'] plus ' *' when the entries of r selected by 'primary', otherwise '') for each r in rows in the surrounding operation.
        )  # Finish assembling ax.set yticks inputs.
        ax.grid(alpha=0.2)  # The result of ax.grid.
    handles, legend_labels = axes[0].get_legend_handles_labels()  # Set handles, legend labels to the result of axes[0].get legend handles labels.
    fig.legend(handles, legend_labels, fontsize=8, loc="lower center", ncol=2)  # Fig.legend applied to handles, legend labels.
    fig.suptitle(  # Fig.suptitle applied to the report text.
        "Normal traffic: cautious QR minus each reference; * = the single primary comparison"  # Include the report text in the surrounding operation.
    )  # Finish assembling fig.suptitle inputs.
    fig.tight_layout(rect=[0, 0.09, 1, 0.95])  # The result of fig.tight layout.
    fig.savefig(viewer / "confirmation_intervals.png", dpi=150)  # Save a figure for the local results page; this does not affect agent behavior.
    plt.close(fig)  # Plt.close applied to fig.
    primary_stats = primary["crashed"]["fixed_models"]  # Set primary stats to the entries of the entries of primary selected by 'crashed' selected by 'fixed_models'.
    interval = primary_stats["conditional_95_percent_interval"]  # Set interval to the entries of primary stats selected by 'conditional_95_percent_interval'.
    speed = primary["mean_speed_m_s"]["fixed_models"]["mean_difference"]  # Set speed to the entries of the entries of the entries of primary selected by 'mean_speed_m_s' selected by 'fixed_models' selected by 'mean_difference'.
    status = "met" if criterion else "not met"  # Set status to 'met' when criterion, otherwise 'not met'.
    document = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Confirmation study · Risk-sensitive driving</title><style>body{font:17px/1.6 system-ui;color:#18283d;background:#f4f7fb;margin:auto;max-width:1160px;padding:36px}h1{font-size:38px;line-height:1.2}section{background:white;padding:24px;margin:24px 0;border-radius:14px}img{width:100%;height:auto}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;text-align:left;border-bottom:1px solid #dde3eb}th{background:#eef3fa}a{color:#175bbc}summary{cursor:pointer}</style></head><body><p>DEEP RL / LOCKED CONFIRMATION STUDY</p><h1>Do the development findings survive fresh scenarios?</h1><p>Six frozen mixed-density networks, five deployment rules, and no changes to training, checkpoints, risk level or speed thresholds after inspecting the reserved results. Scenario IDs 20000–20099 were verified unused before the design was locked.</p><p><a href="../confirmation/design.json">Locked design and model hashes</a> · <a href="../confirmation/preflight.json">Preflight audit</a> · <a href="../confirmation/summary.json">Full results</a> · <a href="../baseline_controls-report/index.html">Archived development comparisons</a> · <a href="../density_coverage-report/index.html">Training coverage and prediction diagnostics</a></p>"""  # Set document to the report text.
    document += (  # Increase document using (((('<section><h2>Single primary comparison: normal-traffic collisions</h2><p>The predeclared criterion is <strong>' + status + '</str plus str([round(100 * v, 1) for v in interval])) plus ' pp.</p><p>The average-speed difference is ') plus f'{speed:+.2f}') plus the report text).
        "<section><h2>Single primary comparison: normal-traffic collisions</h2><p>The predeclared criterion is <strong>"  # Include (((('<section><h2>Single primary comparison: normal-traffic collisions</h2><p>The predeclared criterion is <strong>' + status + '</str plus str([round(100 * v, 1) for v in interval])) plus ' pp.</p><p>The average-speed difference is ') plus f'{speed:+.2f}') plus the report text) in the surrounding operation.
        + status  # Include status in the surrounding operation.
        + "</strong>: cautious QR minus DQN with its frozen 25 m/s governor is "  # Include the report text in the surrounding operation.
        + f"{100*primary_stats['mean_difference']:+.1f}"  # Include f"{100 * primary stats['mean difference']:+.1f}" in the surrounding operation.
        + " percentage points, with an approximate conditional 95% interval of "  # Include the report text in the surrounding operation.
        + str([round(100 * v, 1) for v in interval])  # Include round applied to (100 times v), 1 for each v in interval as a text value in the surrounding operation.
        + " pp.</p><p>The average-speed difference is "  # Include ' pp.</p><p>The average-speed difference is ' in the surrounding operation.
        + f"{speed:+.2f}"  # Include f'{speed:+.2f}' in the surrounding operation.
        + " m/s. This is a comparison of the locked policies; a collision advantage does not establish a speed-independent effect.</p><p>The criterion requires the upper endpoint to be strictly below zero. It concerns new scenarios for this selected set of networks, not new training runs.</p></section>"  # Include the report text in the surrounding operation.
    )  # Finish assembling the grouped values above.
    normal_tail = next(r for r in groups["qr_tail"] if r["density"] == 1.0)  # Set normal tail to next applied to r for each r in the entries of groups selected by 'qr_tail', keeping only cases where the entries of r selected by 'density' equals 1.0.
    normal_dqn = next(r for r in groups["dqn_governed"] if r["density"] == 1.0)  # Set normal dqn to next applied to r for each r in the entries of groups selected by 'dqn_governed', keeping only cases where the entries of r selected by 'density' equals 1.0.
    raw_text = f"Cautious QR: {100*normal_tail['crash_rate']:.1f}% collisions at {normal_tail['mean_speed_m_s']:.2f} m/s. Controlled DQN: {100*normal_dqn['crash_rate']:.1f}% at {normal_dqn['mean_speed_m_s']:.2f} m/s."  # Set raw text to f"Cautious QR: {100 * normal tail['crash rate']:.1f}% collisions at {normal tail['mean speed m s']:.2f} m/s. Controlled DQN: {100 * normal dqn['crash rate']:.1f}% at {normal dqn['m.
    seed_text = (  # Set seed text to (('Per-model collision differences (QR minus controlled DQN) are ' plus round(100 * v, 1) for each v in primary stats['per training seed differences'] as a text value) plus the report text).
        "Per-model collision differences (QR minus controlled DQN) are "  # Include (('Per-model collision differences (QR minus controlled DQN) are ' plus round(100 * v, 1) for each v in primary stats['per training seed differences'] as a text value) plus the report text) in the surrounding operation.
        + str(  # Include round applied to (100 times v), 1 for each v in the entries of primary stats selected by 'per_training_seed_differences' as a text value in the surrounding operation.
            [round(100 * v, 1) for v in primary_stats["per_training_seed_differences"]]  # Include round applied to (100 times v), 1 for each v in the entries of primary stats selected by 'per_training_seed_differences' in the surrounding operation.
        )  # Finish assembling str inputs.
        + " percentage points. The interval permits effects in either direction; equivalence was not tested."  # Include the report text in the surrounding operation.
    )  # Finish assembling the grouped values above.
    document += (  # Increase document using (((('<section><h2>Reading the primary result</h2><p>' plus raw text) plus '</p><p>') plus seed text) plus '</p></section>').
        "<section><h2>Reading the primary result</h2><p>"  # Include (((('<section><h2>Reading the primary result</h2><p>' plus raw text) plus '</p><p>') plus seed text) plus '</p></section>') in the surrounding operation.
        + raw_text  # Include raw text in the surrounding operation.
        + "</p><p>"  # Include '</p><p>' in the surrounding operation.
        + seed_text  # Include seed text in the surrounding operation.
        + "</p></section>"  # Include '</p></section>' in the surrounding operation.
    )  # Finish assembling the grouped values above.
    document += (  # Increase document using ((the report text plus table applied to the listed settings or measurements in their recorded order, [LABELS[p], r['density'], f"{100 * r['crash rate']:.1f}%", f"{r['mean speed m s']:.2f}", f"{r['mean distance m']:.0f}", f"{r['mean for each p in LABELS) plus '</section>').
        '<section><h2>Frozen-policy performance</h2><p>Each policy/density row contains 300 rollouts: three training models evaluated on the same 100 fresh scenario IDs. Those 300 observations are not treated as 300 independent scenario samples.</p><img src="confirmation_performance.png">'  # Include ((the report text plus table applied to the listed settings or measurements in their recorded order, [LABELS[p], r['density'], f"{100 * r['crash rate']:.1f}%", f"{r['mean speed m s']:.2f}", f"{r['mean distance m']:.0f}", f"{r['mean for each p in LABELS) plus '</section>') in the surrounding operation.
        + table(  # Include table applied to the listed settings or measurements in their recorded order, the listed settings or measurements in their recorded order for each p in LABELS in the surrounding operation.
            [  # Include the listed settings or measurements in their recorded order in the surrounding operation.
                "Policy",  # Include 'Policy' in the surrounding operation.
                "Density",  # Include 'Density' in the surrounding operation.
                "Collisions",  # Include 'Collisions' in the surrounding operation.
                "Speed m/s",  # Include 'Speed m/s' in the surrounding operation.
                "Distance m",  # Include 'Distance m' in the surrounding operation.
                "Mean return",  # Include 'Mean return' in the surrounding operation.
                "Worst 10% return",  # Include 'Worst 10% return' in the surrounding operation.
            ],  # Finish assembling the grouped values above.
            [  # Include the listed settings or measurements in their recorded order for each p in LABELS in the surrounding operation.
                [  # Include the listed settings or measurements in their recorded order in the surrounding operation.
                    LABELS[p],  # Include the entries of LABELS selected by p in the surrounding operation.
                    r["density"],  # Include the entries of r selected by 'density' in the surrounding operation.
                    f"{100*r['crash_rate']:.1f}%",  # Include f"{100 * r['crash rate']:.1f}%" in the surrounding operation.
                    f"{r['mean_speed_m_s']:.2f}",  # Include f"{r['mean speed m s']:.2f}" in the surrounding operation.
                    f"{r['mean_distance_m']:.0f}",  # Include f"{r['mean distance m']:.0f}" in the surrounding operation.
                    f"{r['mean_return']:.2f}",  # Include f"{r['mean return']:.2f}" in the surrounding operation.
                    f"{r['worst_10_percent_mean_return']:.2f}",  # Include f"{r['worst 10 percent mean return']:.2f}" in the surrounding operation.
                ]  # Finish assembling the grouped values above.
                for p in LABELS  # Include p, LABELS in the surrounding operation.
                for r in groups[p]  # Include r, the entries of groups selected by p in the surrounding operation.
            ],  # Finish assembling table inputs.
        )  # Finish assembling table inputs.
        + "</section>"  # Include '</section>' in the surrounding operation.
    )  # Finish assembling the grouped values above.
    document += '<section><h2>Two scopes of uncertainty</h2><p>Blue intervals resample scenario IDs while keeping all trained networks and policy pairings together. They estimate uncertainty over fresh scenarios conditional on the selected models. Gray intervals also resample training seed IDs; with only three model pairs, these remain exploratory. Only the starred collision contrast is primary; other effects are descriptive secondary comparisons without family-wise adjustment.</p><img src="confirmation_intervals.png"></section>'  # Increase document using the report text.
    document += (  # Increase document using (('<section><h2>Cautious QR minus each reference</h2>' plus table applied to the listed settings or measurements in their recorded order, ['Primary' if r['primary'] else 'Secondary', LABELS[r['reference']], r['density'], f"{100 * r['crashed']['fixed models']['mean dif for each r in comparisons) plus '</section>').
        "<section><h2>Cautious QR minus each reference</h2>"  # Include (('<section><h2>Cautious QR minus each reference</h2>' plus table applied to the listed settings or measurements in their recorded order, ['Primary' if r['primary'] else 'Secondary', LABELS[r['reference']], r['density'], f"{100 * r['crashed']['fixed models']['mean dif for each r in comparisons) plus '</section>') in the surrounding operation.
        + table(  # Include table applied to the listed settings or measurements in their recorded order, the listed settings or measurements in their recorded order for each r in comparisons in the surrounding operation.
            [  # Include the listed settings or measurements in their recorded order in the surrounding operation.
                "Role",  # Include 'Role' in the surrounding operation.
                "Reference",  # Include 'Reference' in the surrounding operation.
                "Density",  # Include 'Density' in the surrounding operation.
                "Collision change pp",  # Include 'Collision change pp' in the surrounding operation.
                "Conditional scenario interval pp",  # Include 'Conditional scenario interval pp' in the surrounding operation.
                "Exploratory crossed interval pp",  # Include 'Exploratory crossed interval pp' in the surrounding operation.
                "Speed change m/s",  # Include 'Speed change m/s' in the surrounding operation.
            ],  # Finish assembling the grouped values above.
            [  # Include the listed settings or measurements in their recorded order for each r in comparisons in the surrounding operation.
                [  # Include the listed settings or measurements in their recorded order in the surrounding operation.
                    "Primary" if r["primary"] else "Secondary",  # Include 'Primary' when the entries of r selected by 'primary', otherwise 'Secondary' in the surrounding operation.
                    LABELS[r["reference"]],  # Include the entries of LABELS selected by r['reference'] in the surrounding operation.
                    r["density"],  # Include the entries of r selected by 'density' in the surrounding operation.
                    f"{100*r['crashed']['fixed_models']['mean_difference']:+.1f}",  # Include f"{100 * r['crashed']['fixed models']['mean difference']:+.1f}" in the surrounding operation.
                    str(  # Include round applied to (100 times v), 1 for each v in the entries of the entries of r['crashed'] selected by 'fixed_models' selected by 'conditional_95_percent_interval' as a text value in the surrounding operation.
                        [  # Include round applied to (100 times v), 1 for each v in the entries of the entries of the entries of r selected by 'crashed' selected by 'fixed_models' selected by 'conditional_95_percent_interval' in the surrounding operation.
                            round(100 * v, 1)  # Include round applied to (100 times v), 1 in the surrounding operation.
                            for v in r["crashed"]["fixed_models"][  # Include v, the entries of the entries of the entries of r selected by 'crashed' selected by 'fixed_models' selected by 'conditional_95_percent_interval' in the surrounding operation.
                                "conditional_95_percent_interval"  # Include 'conditional_95_percent_interval' in the surrounding operation.
                            ]  # Finish assembling str inputs.
                        ]  # Finish assembling str inputs.
                    ),  # Finish assembling str inputs.
                    str(  # Include round applied to (100 times v), 1 for each v in the entries of the entries of r['crashed'] selected by 'exploratory_new_training' selected by 'exploratory_95_percent_interval' as a text value in the surrounding operation.
                        [  # Include round applied to (100 times v), 1 for each v in the entries of the entries of the entries of r selected by 'crashed' selected by 'exploratory_new_training' selected by 'exploratory_95_percent_interval' in the surrounding operation.
                            round(100 * v, 1)  # Include round applied to (100 times v), 1 in the surrounding operation.
                            for v in r["crashed"]["exploratory_new_training"][  # Include v, the entries of the entries of the entries of r selected by 'crashed' selected by 'exploratory_new_training' selected by 'exploratory_95_percent_interval' in the surrounding operation.
                                "exploratory_95_percent_interval"  # Include 'exploratory_95_percent_interval' in the surrounding operation.
                            ]  # Finish assembling str inputs.
                        ]  # Finish assembling str inputs.
                    ),  # Finish assembling str inputs.
                    f"{r['mean_speed_m_s']['fixed_models']['mean_difference']:+.2f}",  # Include f"{r['mean speed m s']['fixed models']['mean difference']:+.2f}" in the surrounding operation.
                ]  # Finish assembling the grouped values above.
                for r in comparisons  # Include r, comparisons in the surrounding operation.
            ],  # Finish assembling table inputs.
        )  # Finish assembling table inputs.
        + "</section>"  # Include '</section>' in the surrounding operation.
    )  # Finish assembling the grouped values above.
    document += (  # Increase document using ((the report text plus table applied to the listed settings or measurements in their recorded order, [LABELS[r['policy']], r['density'], f"{100 * r['development']['collision fraction']:.1f}%", f"{100 * r['confirmation']['collision for each r in dev) plus '</section>').
        "<section><h2>Development versus confirmation</h2><p>Development used 30 scenarios per density/model and supported tuning and selection. Confirmation uses 100 fresh scenarios per density/model. The two blocks differ in initial traffic; these figures are shown without changing any policy.</p>"  # Include ((the report text plus table applied to the listed settings or measurements in their recorded order, [LABELS[r['policy']], r['density'], f"{100 * r['development']['collision fraction']:.1f}%", f"{100 * r['confirmation']['collision for each r in dev) plus '</section>') in the surrounding operation.
        + table(  # Include table applied to the listed settings or measurements in their recorded order, the listed settings or measurements in their recorded order for each r in dev in the surrounding operation.
            [  # Include the listed settings or measurements in their recorded order in the surrounding operation.
                "Policy",  # Include 'Policy' in the surrounding operation.
                "Density",  # Include 'Density' in the surrounding operation.
                "Development collisions",  # Include 'Development collisions' in the surrounding operation.
                "Confirmation collisions",  # Include 'Confirmation collisions' in the surrounding operation.
                "Development speed",  # Include 'Development speed' in the surrounding operation.
                "Confirmation speed",  # Include 'Confirmation speed' in the surrounding operation.
            ],  # Finish assembling the grouped values above.
            [  # Include the listed settings or measurements in their recorded order for each r in dev in the surrounding operation.
                [  # Include the listed settings or measurements in their recorded order in the surrounding operation.
                    LABELS[r["policy"]],  # Include the entries of LABELS selected by r['policy'] in the surrounding operation.
                    r["density"],  # Include the entries of r selected by 'density' in the surrounding operation.
                    f"{100*r['development']['collision_fraction']:.1f}%",  # Include f"{100 * r['development']['collision fraction']:.1f}%" in the surrounding operation.
                    f"{100*r['confirmation']['collision_fraction']:.1f}%",  # Include f"{100 * r['confirmation']['collision fraction']:.1f}%" in the surrounding operation.
                    f"{r['development']['speed_m_s']:.2f}",  # Include f"{r['development']['speed m s']:.2f}" in the surrounding operation.
                    f"{r['confirmation']['speed_m_s']:.2f}",  # Include f"{r['confirmation']['speed m s']:.2f}" in the surrounding operation.
                ]  # Finish assembling the grouped values above.
                for r in dev  # Include r, dev in the surrounding operation.
            ],  # Finish assembling table inputs.
        )  # Finish assembling table inputs.
        + "</section>"  # Include '</section>' in the surrounding operation.
    )  # Finish assembling the grouped values above.
    document += (  # Increase document using ((the report text plus table applied to the listed settings or measurements in their recorded order, [LABELS[p], r['density'], seed['training seed'], f"{100 * seed['crash rate']:.1f}%", f"{seed['mean speed m s']:.2f}", f"{seed['mea for each p in LABELS) plus '</details></section>').
        "<section><h2>Model variation</h2><details><summary>View every training seed</summary>"  # Include ((the report text plus table applied to the listed settings or measurements in their recorded order, [LABELS[p], r['density'], seed['training seed'], f"{100 * seed['crash rate']:.1f}%", f"{seed['mean speed m s']:.2f}", f"{seed['mea for each p in LABELS) plus '</details></section>') in the surrounding operation.
        + table(  # Include table applied to the listed settings or measurements in their recorded order, the listed settings or measurements in their recorded order for each p in LABELS in the surrounding operation.
            [  # Include the listed settings or measurements in their recorded order in the surrounding operation.
                "Policy",  # Include 'Policy' in the surrounding operation.
                "Density",  # Include 'Density' in the surrounding operation.
                "Training seed",  # Include 'Training seed' in the surrounding operation.
                "Collisions",  # Include 'Collisions' in the surrounding operation.
                "Speed m/s",  # Include 'Speed m/s' in the surrounding operation.
                "Distance m",  # Include 'Distance m' in the surrounding operation.
            ],  # Finish assembling the grouped values above.
            [  # Include the listed settings or measurements in their recorded order for each p in LABELS in the surrounding operation.
                [  # Include the listed settings or measurements in their recorded order in the surrounding operation.
                    LABELS[p],  # Include the entries of LABELS selected by p in the surrounding operation.
                    r["density"],  # Include the entries of r selected by 'density' in the surrounding operation.
                    seed["training_seed"],  # Include the entries of seed selected by 'training_seed' in the surrounding operation.
                    f"{100*seed['crash_rate']:.1f}%",  # Include f"{100 * seed['crash rate']:.1f}%" in the surrounding operation.
                    f"{seed['mean_speed_m_s']:.2f}",  # Include f"{seed['mean speed m s']:.2f}" in the surrounding operation.
                    f"{seed['mean_distance_m']:.0f}",  # Include f"{seed['mean distance m']:.0f}" in the surrounding operation.
                ]  # Finish assembling the grouped values above.
                for p in LABELS  # Include p, LABELS in the surrounding operation.
                for r in groups[p]  # Include r, the entries of groups selected by p in the surrounding operation.
                for seed in r["per_seed"]  # Include seed, the entries of r selected by 'per_seed' in the surrounding operation.
            ],  # Finish assembling table inputs.
        )  # Finish assembling table inputs.
        + "</details></section>"  # Include '</details></section>' in the surrounding operation.
    )  # Finish assembling the grouped values above.
    document += (  # Increase document using ((the report text plus ''.join applied to '<article><h3>' + LABELS[p] + '</h3><img alt="Locked policy driving replay" src="../confirmation/replay ' + p + '.gif"></article>' for each p in ['qr tail', 'dqn governed', 'qr governed']) plus '</div></section>').
        '<section><h2>First reserved scenario, selected before outcomes</h2><p>Training seed 0, scenario 20000, density 1. This is one illustrative case, not evidence of typical performance.</p><div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px">'  # Include ((the report text plus ''.join applied to '<article><h3>' + LABELS[p] + '</h3><img alt="Locked policy driving replay" src="../confirmation/replay ' + p + '.gif"></article>' for each p in ['qr tail', 'dqn governed', 'qr governed']) plus '</div></section>') in the surrounding operation.
        + "".join(  # Include ''.join applied to (('<article><h3>' + LABELS[p] + '</h3><img alt="Locked policy driving replay" src="../confirmation/replay ' plus p) plus '.gif"></article>') for each p in 'qr_tail', 'dqn_governed', 'qr_governed' in the surrounding operation.
            "<article><h3>"  # Include (((('<article><h3>' plus LABELS[p]) plus the report text) plus p) plus '.gif"></article>') in the surrounding operation.
            + LABELS[p]  # Include the entries of LABELS selected by p in the surrounding operation.
            + '</h3><img alt="Locked policy driving replay" src="../confirmation/replay_'  # Include the report text in the surrounding operation.
            + p  # Include p in the surrounding operation.
            + '.gif"></article>'  # Include '.gif"></article>' in the surrounding operation.
            for p in ["qr_tail", "dqn_governed", "qr_governed"]  # Include p, 'qr_tail', 'dqn_governed', 'qr_governed' in the surrounding operation.
        )  # Finish assembling ''.join inputs.
        + "</div></section>"  # Include '</div></section>' in the surrounding operation.
    )  # Finish assembling the grouped values above.
    document += "<section><h2>Reproduce and retain the holdout boundary</h2><p><code>python run_confirmation.py --jobs 3</code> validates and reuses saved outputs. <code>--stage report</code> regenerates the report. Models and selection rules remain locked by hashes and recorded settings.</p><p>Seeds 20000–20099 are now consumed confirmation data for this locked study. Any further model tuning must use development blocks, and a later confirmation needs a new untouched block (20100 onward). Existing development viewers are historical snapshots from before this confirmation.</p><p>Observed speeds, action interventions, partial observations, simulator simplifications, forecast error and only three training seeds limit interpretation. Return quantiles are not collision probabilities. This experiment does not establish a globally CVaR-optimal policy or real-road safety.</p></section></body></html>"  # Increase document using the report text.
    (viewer / "index.html").write_text(document)  # (viewer / 'index.html').write text applied to document.
    print(  # Print applied to f"primary criterion {status}: {100 * primary stats['mean difference']:+.1f} pp; conditional interval {[round(100 * v, 1) for v in interval]}; speed {speed:+.2f} m/s".
        f'Primary criterion {status}: {100*primary_stats["mean_difference"]:+.1f} pp; conditional interval {[round(100*v,1) for v in interval]}; speed {speed:+.2f} m/s',  # Include f"Primary criterion {status}: {100 * primary stats['mean difference']:+.1f} pp; conditional interval {[round(100 * v, 1) for v in interval]}; speed {speed:+.2f} m/s" in the surrounding operation.
        flush=True,  # Include True in the surrounding operation.
    )  # Finish assembling print inputs.
    print("Confirmation report saved", flush=True)  # Print applied to 'confirmation report saved'.


if __name__ == "__main__":  # Run this workflow when launched from Terminal; importing its helpers must not launch experiments.
    report()  # The result of report.
