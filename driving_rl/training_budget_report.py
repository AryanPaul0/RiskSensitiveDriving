"""Create the training budget results page.

Read saved numbers, calculate clearly labeled summaries, draw figures, and write
an HTML page. This file presents evidence; it does not train or select models.
Long HTML strings are page templates. The reinforcement-learning calculations
live in agents.py and experiment.py, not inside the page layout.

Original implementation notes:
Scientific report for the predeclared training-budget comparison.
"""

import csv  # Make csv available for this file's calculations and workflow.
import html  # Make html available for this file's calculations and workflow.
import json  # Make json available for this file's calculations and workflow.
from pathlib import Path  # Make Path available for this file's calculations and workflow.
import shutil  # Make shutil available for this file's calculations and workflow.
import numpy as np  # Make numpy available for this file's calculations and workflow.
import matplotlib  # Make matplotlib available for this file's calculations and workflow.

matplotlib.use("Agg")  # Matplotlib.use applied to 'agg'.
import matplotlib.pyplot as plt  # Make matplotlib.pyplot available for this file's calculations and workflow.
from .analysis import paired_matrix, crossed_bootstrap  # Make paired_matrix, crossed_bootstrap available for this file's calculations and workflow.
from .calibration import summarize_rows  # Make summarize_rows available for this file's calculations and workflow.
from .experiment import save_json  # Make save_json available for this file's calculations and workflow.

ROOT = Path(__file__).resolve().parents[1]  # Set ROOT to the entries of Path( file ).resolve().parents selected by 1.
STAGE = ROOT / "results/training_budget"  # Set STAGE to the path to 'results/training_budget' inside ROOT.
LABELS = {  # Set LABELS to the named settings or measurements assembled below.
    "dqn_mean": "DQN",  # Record 'dqn_mean' as 'DQN'.
    "qrdqn_mean": "QR mean",  # Record 'qrdqn_mean' as 'QR mean'.
    "qrdqn_tail25": "QR lower 25%",  # Record 'qrdqn_tail25' as 'QR lower 25%'.
    "qrdqn_governed": "QR + fixed governor",  # Record 'qrdqn_governed' as 'QR + fixed governor'.
}  # Finish assembling the grouped values above.


def report():  # Build graphs and an HTML page from saved result files.
    """Build graphs and an HTML page from saved result files.
    The arrays below hold measured results; plotting does not change the models.
    HTML strings describe page layout, while Python inserts the saved numbers.
    Keep primary comparisons separate from exploratory comparisons in the report.
    """
    viewer = ROOT / "results/report"  # Set viewer to the path to 'results/report' inside ROOT.
    backup = ROOT / "results/risk_analysis-report"  # Set backup to the path to 'results/risk_analysis-report' inside ROOT.
    if not backup.exists():  # Use this branch when not the result of backup.exists.
        shutil.copytree(viewer, backup)  # Shutil.copytree applied to viewer, backup.
    groups = {}  # Set groups to the named settings or measurements assembled below.
    summaries = []  # Set summaries to .
    for name in LABELS:  # Process each name from LABELS.
        algo, policy = name.split("_", 1)  # Set algo, policy to name.split applied to '_', 1.
        for budget in [20000, 40000]:  # Process each the environment-interaction budget from 20000, 40000.
            data = [  # Set data to json.loads applied to the result of (STAGE / f'eval {algo} {s} {budget} {policy}.json').read text for each s in range applied to 3.
                json.loads(  # Include json.loads applied to the result of (STAGE / f'eval {algo} {s} {budget} {policy}.json').read text in the surrounding operation.
                    (STAGE / f"eval_{algo}_{s}_{budget}_{policy}.json").read_text()  # Include the result of (STAGE / f'eval {algo} {s} {budget} {policy}.json').read text in the surrounding operation.
                )  # Finish assembling json.loads inputs.
                for s in range(3)  # Include s, range applied to 3 in the surrounding operation.
            ]  # Finish assembling the grouped values above.
            groups[name, budget] = data  # Set the entries of groups selected by (name, budget) to data.
            for density in [1.0, 1.5, 2.0]:  # Process each the traffic-density setting from 1.0, 1.5, 2.0.
                per_seed = []  # Set per seed to .
                for d in data:  # Process each d from data.
                    rows = [r for r in d["episodes"] if r["density"] == density]  # Set rows to r for each r in the entries of d selected by 'episodes', keeping only cases where the entries of r selected by 'density' equals the traffic-density setting.
                    returns = sorted(r["return"] for r in rows)  # Set returns to sorted applied to the entries of r selected by 'return' for each r in rows.
                    per_seed.append(  # Add the result of dict to per seed.
                        dict(  # Include the result of dict in the surrounding operation.
                            seed=d["provenance"]["training_seed"],  # Include the entries of the entries of d selected by 'provenance' selected by 'training_seed' in the surrounding operation.
                            crash_rate=np.mean([r["crashed"] for r in rows]),  # Include the average of the entries of r selected by 'crashed' for each r in rows in the surrounding operation.
                            speed=np.mean([r["mean_speed_m_s"] for r in rows]),  # Include the average of the entries of r selected by 'mean_speed_m_s' for each r in rows in the surrounding operation.
                            distance=np.mean([r["distance_m"] for r in rows]),  # Include the average of the entries of r selected by 'distance_m' for each r in rows in the surrounding operation.
                            mean_return=np.mean(returns),  # Include the average of returns in the surrounding operation.
                            worst10=np.mean(returns[:3]),  # Include the average of the entries of returns selected by :3 in the surrounding operation.
                        )  # Finish assembling dict inputs.
                    )  # Finish assembling per seed.append inputs.
                summaries.append(  # Add the result of dict to summaries.
                    dict(  # Include the result of dict in the surrounding operation.
                        policy=name,  # Include name in the surrounding operation.
                        budget=budget,  # Include the environment-interaction budget in the surrounding operation.
                        density=density,  # Include the traffic-density setting in the surrounding operation.
                        per_seed=per_seed,  # Include per seed in the surrounding operation.
                        **{  # Include the average of r[k] for each r in per seed as a floating-point value for each k in 'crash_rate', 'speed', 'distance', 'mean_return', 'worst10' in the surrounding operation.
                            k: float(np.mean([r[k] for r in per_seed]))  # Include k, the average of the entries of r selected by k for each r in per seed as a floating-point value in the surrounding operation.
                            for k in [  # Include k, 'crash_rate', 'speed', 'distance', 'mean_return', 'worst10' in the surrounding operation.
                                "crash_rate",  # Include 'crash_rate' in the surrounding operation.
                                "speed",  # Include 'speed' in the surrounding operation.
                                "distance",  # Include 'distance' in the surrounding operation.
                                "mean_return",  # Include 'mean_return' in the surrounding operation.
                                "worst10",  # Include 'worst10' in the surrounding operation.
                            ]  # Finish assembling the grouped values above.
                        },  # Finish assembling dict inputs.
                    )  # Finish assembling dict inputs.
                )  # Finish assembling summaries.append inputs.
    comparisons = []  # Set comparisons to .
    for name in LABELS:  # Process each name from LABELS.
        for density in [1.0, 1.5, 2.0]:  # Process each the traffic-density setting from 1.0, 1.5, 2.0.
            row = dict(policy=name, density=density)  # Set row to the result of dict.
            for metric in ["crashed", "mean_speed_m_s", "distance_m", "return"]:  # Compare one measured quantity at a time, keeping the scenarios paired.
                _, matrix = paired_matrix(  # Set the unused result, matrix to paired matrix applied to the entries of groups selected by (name, 40000), the entries of groups selected by (name, 20000), the traffic-density setting, metric.
                    groups[name, 40000], groups[name, 20000], density, metric  # Include the entries of groups selected by (name, 40000), the entries of groups selected by (name, 20000), the traffic-density setting, metric in the surrounding operation.
                )  # Finish assembling paired matrix inputs.
                row[metric] = crossed_bootstrap(matrix)  # Set the entries of row selected by metric to crossed bootstrap applied to matrix.
            comparisons.append(row)  # Add row to comparisons.
    controls = []  # Set controls to .
    for density in [1.0, 1.5, 2.0]:  # Process each the traffic-density setting from 1.0, 1.5, 2.0.
        item = dict(density=density)  # Set item to the result of dict.
        for metric in ["crashed", "mean_speed_m_s"]:  # Compare one measured quantity at a time, keeping the scenarios paired.
            _, matrix = paired_matrix(  # Set the unused result, matrix to paired matrix applied to the entries of groups selected by ('qrdqn_tail25', 40000), the entries of groups selected by ('qrdqn_governed', 40000), the traffic-density setting, metric.
                groups["qrdqn_tail25", 40000],  # Include the entries of groups selected by ('qrdqn_tail25', 40000) in the surrounding operation.
                groups["qrdqn_governed", 40000],  # Include the entries of groups selected by ('qrdqn_governed', 40000) in the surrounding operation.
                density,  # Include the traffic-density setting in the surrounding operation.
                metric,  # Include metric in the surrounding operation.
            )  # Finish assembling paired matrix inputs.
            item[metric] = crossed_bootstrap(matrix)  # Set the entries of item selected by metric to crossed bootstrap applied to matrix.
        controls.append(item)  # Add item to controls.
    calibration = []  # Set calibration to .
    for policy in ["mean", "tail25"]:  # Process each policy from 'mean', 'tail25'.
        for budget in [20000, 40000]:  # Process each the environment-interaction budget from 20000, 40000.
            folder = ROOT / "results/risk_analysis" if budget == 20000 else STAGE  # Set folder to the path to 'results/risk_analysis' inside ROOT when the environment-interaction budget equals 20000, otherwise STAGE.
            data = [  # Set data to json.loads applied to the result of (folder / f'calibration {policy} seed{s}.json').read text for each s in range applied to 3.
                json.loads((folder / f"calibration_{policy}_seed{s}.json").read_text())  # Include json.loads applied to the result of (folder / f'calibration {policy} seed{s}.json').read text in the surrounding operation.
                for s in range(3)  # Include s, range applied to 3 in the surrounding operation.
            ]  # Finish assembling the grouped values above.
            for density in [1.0, 1.5, 2.0]:  # Process each the traffic-density setting from 1.0, 1.5, 2.0.
                rows = [  # Set rows to r for each d in data.
                    r for d in data for r in d["episodes"] if r["density"] == density  # Include r, d, data, r, the entries of d selected by 'episodes', the entries of r selected by 'density' equals the traffic-density setting in the surrounding operation.
                ]  # Finish assembling the grouped values above.
                calibration.append(  # Add the result of dict to calibration.
                    dict(  # Include the result of dict in the surrounding operation.
                        policy=policy,  # Include policy in the surrounding operation.
                        budget=budget,  # Include the environment-interaction budget in the surrounding operation.
                        density=density,  # Include the traffic-density setting in the surrounding operation.
                        **summarize_rows(rows),  # Include summarize rows applied to rows in the surrounding operation.
                    )  # Finish assembling dict inputs.
                )  # Finish assembling calibration.append inputs.
    prefix = json.loads((STAGE / "prefix_verification.json").read_text())  # Set prefix to json.loads applied to the result of (STAGE / 'prefix verification.json').read text.
    reproduction = []  # Set reproduction to .
    for algo in ["dqn", "qrdqn"]:  # Process each algo from 'dqn', 'qrdqn'.
        for seed in range(3):  # Process each seed from range applied to 3.
            for policy in (["mean"] if algo == "dqn" else ["mean", "tail25"]):  # Process each policy from 'mean' when algo equals 'dqn', otherwise 'mean', 'tail25'.
                original = json.loads(  # Set original to json.loads applied to the result of (ROOT / f'results/eval {algo} seed{seed} {policy}.json').read text.
                    (ROOT / f"results/eval_{algo}_seed{seed}_{policy}.json").read_text()  # Include the result of (ROOT / f'results/eval {algo} seed{seed} {policy}.json').read text in the surrounding operation.
                )  # Finish assembling json.loads inputs.
                fresh = groups[f"{algo}_{policy}", 20000][seed]  # Set fresh to the entries of the entries of groups selected by (f'{algo}_{policy}', 20000) selected by seed.
                equal = original["episodes"] == fresh["episodes"]  # Set equal to the entries of original selected by 'episodes' equals the entries of fresh selected by 'episodes'.
                reproduction.append(  # Add the result of dict to reproduction.
                    dict(  # Include the result of dict in the surrounding operation.
                        algorithm=algo,  # Include algo in the surrounding operation.
                        seed=seed,  # Include seed in the surrounding operation.
                        policy=policy,  # Include policy in the surrounding operation.
                        exact_episode_match=equal,  # Include equal in the surrounding operation.
                    )  # Finish assembling dict inputs.
                )  # Finish assembling reproduction.append inputs.
                if not equal:  # Use this branch when not equal.
                    raise ValueError("20k episode reproduction failed")  # Reject this invalid case instead of continuing with misleading experiment results.
    save_json(STAGE / "evaluation_reproduction.json", reproduction)  # Save the following results/settings as json: the path to 'evaluation_reproduction.json' inside stage, reproduction.
    result = dict(  # Set result to the result of dict.
        evaluation_reproduction=reproduction,  # Include reproduction in the surrounding operation.
        groups=summaries,  # Include summaries in the surrounding operation.
        budget_differences=comparisons,  # Include comparisons in the surrounding operation.
        same_network_control=controls,  # Include controls in the surrounding operation.
        calibration=calibration,  # Include calibration in the surrounding operation.
        prefix_verification=prefix,  # Include prefix in the surrounding operation.
        confirmation_seeds_used=False,  # Include False in the surrounding operation.
    )  # Finish assembling dict inputs.
    save_json(STAGE / "summary.json", result)  # Save the following results/settings as json: the path to 'summary.json' inside stage, result.
    plt.rcParams.update(  # Plt.rcparams.update applied to the named settings or measurements assembled below.
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}  # Record 'font.size' as 10.
    )  # Finish assembling plt.rcParams.update inputs.
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)  # Set fig, axes to plt.subplots applied to 2, 3.
    for col, density in enumerate([1.0, 1.5, 2.0]):  # Process each col, the traffic-density setting from enumerate applied to 1.0, 1.5, 2.0.
        for i, name in enumerate(LABELS):  # Process each i, name from enumerate applied to LABELS.
            points = [  # Set points to r for each r in summaries, keeping only cases where the entries of r selected by 'policy' equals name and the entries of r selected by 'density' equals the traffic-density setting.
                r for r in summaries if r["policy"] == name and r["density"] == density  # Include r, r, summaries, the entries of r selected by 'policy' equals name and the entries of r selected by 'density' equals the traffic-density setting in the surrounding operation.
            ]  # Finish assembling the grouped values above.
            color = f"C{i}"  # Set color to f'C{i}'.
            for row, metric, scale in [(0, "crash_rate", 100), (1, "speed", 1)]:  # Process each row, metric, scale from 0, 'crash_rate', 100, 1, 'speed', 1.
                ax = axes[row, col]  # Set ax to the entries of axes selected by (row, col).
                ax.plot(  # Ax.plot applied to 20, 40, (the entries of r selected by metric times scale) for each r in points, '-o'.
                    [20, 40],  # Include 20, 40 in the surrounding operation.
                    [r[metric] * scale for r in points],  # Include (the entries of r selected by metric times scale) for each r in points in the surrounding operation.
                    "-o",  # Include '-o' in the surrounding operation.
                    color=color,  # Include color in the surrounding operation.
                    label=LABELS[name],  # Include the entries of LABELS selected by name in the surrounding operation.
                )  # Finish assembling ax.plot inputs.
                for seed in range(3):  # Process each seed from range applied to 3.
                    ax.plot(  # Ax.plot applied to 20, 40, (the entries of r['per seed'][seed] selected by metric times scale) for each r in points.
                        [20, 40],  # Include 20, 40 in the surrounding operation.
                        [r["per_seed"][seed][metric] * scale for r in points],  # Include (the entries of the entries of r['per seed'] selected by seed selected by metric times scale) for each r in points in the surrounding operation.
                        color=color,  # Include color in the surrounding operation.
                        alpha=0.25,  # Include 0.25 in the surrounding operation.
                        lw=0.8,  # Include 0.8 in the surrounding operation.
                    )  # Finish assembling ax.plot inputs.
                ax.grid(alpha=0.2)  # The result of ax.grid.
                ax.set_xticks([20, 40])  # Ax.set xticks applied to 20, 40.
                ax.set_xlabel("Training steps (thousands)")  # Ax.set xlabel applied to 'training steps (thousands)'.
        axes[0, col].set_title(f"Traffic density {density:g}")  # Axes[0, col].set title applied to f'traffic density {density:g}'.
        axes[0, col].set_ylim(0, 105)  # Axes[0, col].set ylim applied to 0, 105.
    axes[0, 0].set_ylabel("Episodes with collision (%)")  # Axes[0, 0].set ylabel applied to 'episodes with collision (%)'.
    axes[1, 0].set_ylabel("Mean episode speed (m/s)")  # Axes[1, 0].set ylabel applied to 'mean episode speed (m/s)'.
    axes[0, 0].legend(fontsize=8)  # The result of axes[0, 0].legend.
    fig.suptitle(  # Fig.suptitle applied to the report text.
        "More training, fixed exploration schedule; faint lines = individual training seeds"  # Include the report text in the surrounding operation.
    )  # Finish assembling fig.suptitle inputs.
    fig.tight_layout()  # The result of fig.tight layout.
    fig.savefig(viewer / "training_budget_budget.png", dpi=150)  # Save a figure for the local results page; this does not affect agent behavior.
    plt.close(fig)  # Plt.close applied to fig.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)  # Set fig, axes to plt.subplots applied to 1, 2.
    for ax, policy in zip(axes, ["mean", "tail25"]):  # Process each ax, policy from matching entries grouped across axes, 'mean', 'tail25'.
        for budget, color, offset in [(20000, "C0", -0.04), (40000, "C1", 0.04)]:  # Process each the environment-interaction budget, color, offset from 20000, 'C0', negative 0.04, 40000, 'C1', 0.04.
            rows = [  # Set rows to r for each r in calibration, keeping only cases where the entries of r selected by 'policy' equals policy and the entries of r selected by 'budget' equals the environment-interaction budget.
                r  # Include r in the surrounding operation.
                for r in calibration  # Include r, calibration in the surrounding operation.
                if r["policy"] == policy and r["budget"] == budget  # Include the entries of r selected by 'policy' equals policy and the entries of r selected by 'budget' equals the environment-interaction budget in the surrounding operation.
            ]  # Finish assembling the grouped values above.
            lo = np.array([r["mean_prediction_bias_bounds"][0] for r in rows])  # Set lo to the entries of the entries of r selected by 'mean_prediction_bias_bounds' selected by 0 for each r in rows as a numerical array/tensor for batch calculations.
            hi = np.array([r["mean_prediction_bias_bounds"][1] for r in rows])  # Set hi to the entries of the entries of r selected by 'mean_prediction_bias_bounds' selected by 1 for each r in rows as a numerical array/tensor for batch calculations.
            mid = (lo + hi) / 2  # Set mid to ((lo plus hi) divided by 2).
            ax.errorbar(  # Ax.errorbar applied to (1.0, 1.5, 2.0 as a numerical array/tensor for batch calculations plus offset), mid.
                np.array([1.0, 1.5, 2.0]) + offset,  # Include (1.0, 1.5, 2.0 as a numerical array/tensor for batch calculations plus offset) in the surrounding operation.
                mid,  # Include mid in the surrounding operation.
                yerr=[mid - lo, hi - mid],  # Include (mid minus lo), (hi minus mid) in the surrounding operation.
                fmt="o-",  # Include 'o-' in the surrounding operation.
                capsize=4,  # Include 4 in the surrounding operation.
                label=f"{budget//1000}k",  # Include f'{budget // 1000}k' in the surrounding operation.
                color=color,  # Include color in the surrounding operation.
            )  # Finish assembling ax.errorbar inputs.
        ax.axhline(0, color="black", lw=0.8)  # Ax.axhline applied to 0.
        ax.set_title(f"QR {policy} deployment")  # Ax.set title applied to f'qr {policy} deployment'.
        ax.set_xlabel("Traffic density")  # Ax.set xlabel applied to 'traffic density'.
        ax.set_xticks([1, 1.5, 2])  # Ax.set xticks applied to 1, 1.5, 2.
        ax.legend()  # The result of ax.legend.
        ax.grid(alpha=0.2)  # The result of ax.grid.
    axes[0].set_ylabel("Prediction minus discounted return")  # Axes[0].set ylabel applied to 'prediction minus discounted return'.
    fig.suptitle("Censoring bounds only; whiskers are not confidence intervals")  # Fig.suptitle applied to 'censoring bounds only; whiskers are not confidence intervals'.
    fig.tight_layout()  # The result of fig.tight layout.
    fig.savefig(viewer / "training_budget_bias.png", dpi=150)  # Save a figure for the local results page; this does not affect agent behavior.
    plt.close(fig)  # Plt.close applied to fig.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))  # Set fig, axes to plt.subplots applied to 1, 2.
    for ax, algo in zip(axes, ["dqn", "qrdqn"]):  # Process each ax, algo from matching entries grouped across axes, 'dqn', 'qrdqn'.
        for seed in range(3):  # Process each seed from range applied to 3.
            with (STAGE / f"{algo}_seed{seed}/training.csv").open() as f:  # Keep this temporary context active during the enclosed work, then clean it up.
                logs = list(csv.DictReader(f))  # Set logs to list applied to csv.DictReader applied to f.
            steps = np.array([int(r["step"]) for r in logs])  # Set steps to the entries of r selected by 'step' as a whole-number value for each r in logs as a numerical array/tensor for batch calculations.
            returns = np.array([float(r["return"]) for r in logs])  # Set returns to the entries of r selected by 'return' as a floating-point value for each r in logs as a numerical array/tensor for batch calculations.
            if len(returns) >= 100:  # Use this branch when the number of entries in returns is at least 100.
                ax.plot(  # Ax.plot applied to (the entries of steps selected by 99: divided by 1000), np.convolve applied to returns, (np.ones applied to 100 divided by 100).
                    steps[99:] / 1000,  # Include (the entries of steps selected by 99: divided by 1000) in the surrounding operation.
                    np.convolve(returns, np.ones(100) / 100, mode="valid"),  # Include np.convolve applied to returns, (np.ones applied to 100 divided by 100) in the surrounding operation.
                    label=f"Seed {seed}",  # Include f'Seed {seed}' in the surrounding operation.
                )  # Finish assembling ax.plot inputs.
        ax.axvline(20, color="black", ls="--", lw=1)  # Ax.axvline applied to 20.
        ax.set_title(algo.upper())  # Ax.set title applied to the result of algo.upper.
        ax.set_xlabel("Training steps (thousands)")  # Ax.set xlabel applied to 'training steps (thousands)'.
        ax.grid(alpha=0.2)  # The result of ax.grid.
        ax.legend()  # The result of ax.legend.
    axes[0].set_ylabel("Training return: trailing 100 episodes")  # Axes[0].set ylabel applied to 'training return: trailing 100 episodes'.
    fig.suptitle("Exploratory training returns include epsilon-greedy actions")  # Fig.suptitle applied to 'exploratory training returns include epsilon-greedy actions'.
    fig.tight_layout()  # The result of fig.tight layout.
    fig.savefig(viewer / "training_budget_learning.png", dpi=150)  # Save a figure for the local results page; this does not affect agent behavior.
    plt.close(fig)  # Plt.close applied to fig.

    def table(headers, rows):  # Build the HTML for a table of saved numbers.
        """Build the HTML for a table of saved numbers. Headers label columns; each
        row supplies values in the same order. This changes presentation, not the data.
        """
        return (  # Return (((('<div class="scroll"><table><thead><tr>' plus ''.join(('<th>' + html.escape(x) + '</th>' for x in headers))) plus '</tr></thead><tbody>') plus ''.join applied to '<tr>' + ''.join(('<td>' + str(x) + '</td>' for x in row)) + '</tr>' for each row in rows) plus '</tbody></table></div>').
            '<div class="scroll"><table><thead><tr>'  # Include (((('<div class="scroll"><table><thead><tr>' plus ''.join(('<th>' + html.escape(x) + '</th>' for x in headers))) plus '</tr></thead><tbody>') plus ''.join applied to '<tr>' + ''.join(('<td>' + str(x) + '</td>' for x in row)) + '</tr>' for each row in rows) plus '</tbody></table></div>') in the surrounding operation.
            + "".join("<th>" + html.escape(x) + "</th>" for x in headers)  # Include ''.join applied to (('<th>' plus html.escape(x)) plus '</th>') for each x in headers in the surrounding operation.
            + "</tr></thead><tbody>"  # Include '</tr></thead><tbody>' in the surrounding operation.
            + "".join(  # Include ''.join applied to (('<tr>' plus ''.join(('<td>' + str(x) + '</td>' for x in row))) plus '</tr>') for each row in rows in the surrounding operation.
                "<tr>" + "".join("<td>" + str(x) + "</td>" for x in row) + "</tr>"  # Include (('<tr>' plus ''.join applied to '<td>' + str(x) + '</td>' for each x in row) plus '</tr>') in the surrounding operation.
                for row in rows  # Include row, rows in the surrounding operation.
            )  # Finish assembling ''.join inputs.
            + "</tbody></table></div>"  # Include '</tbody></table></div>' in the surrounding operation.
        )  # Finish assembling the grouped values above.

    body = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>training budget experiment · Risk-sensitive driving</title><style>body{font:17px/1.6 system-ui;color:#18283d;background:#f4f7fb;margin:auto;max-width:1160px;padding:36px}h1{font-size:38px;line-height:1.2}section{background:white;padding:24px;margin:24px 0;border-radius:14px}img{width:100%;height:auto}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;text-align:left;border-bottom:1px solid #dde3eb}th{background:#eef3fa}a{color:#175bbc}.note{color:#47576b}</style></head><body><p>training budget experiment / CONTROLLED TRAINING BUDGET</p><h1>Does more experience improve risk-sensitive driving?</h1><p>Six training trajectories, two checkpoint budgets, and a fixed exploration schedule. This is a development study; confirmation scenarios remain reserved.</p><p><a href="../risk_analysis-report/index.html">risk analysis: interactive risk frontier, calibration and replays</a> · <a href="../training_budget/summary.json">Numerical results</a></p>"""  # Set body to the report text.
    normal = {(r["policy"], r["budget"]): r for r in summaries if r["density"] == 1.0}  # Set normal to r for each r in summaries, keeping only cases where the entries of r selected by 'density' equals 1.0.
    highlights = []  # Set highlights to .
    for name in LABELS:  # Process each name from LABELS.
        a, b = normal[name, 20000], normal[name, 40000]  # Set a, b to the entries of normal selected by (name, 20000), the entries of normal selected by (name, 40000).
        highlights.append(  # Add f"{labels[name]}: collisions {100 * a['crash rate']:.1f}% to {100 * b['crash rate']:.1f}%; speed {a['speed']:.2f} to {b['speed']:.2f} m/s." to highlights.
            f"{LABELS[name]}: collisions {100*a['crash_rate']:.1f}% to {100*b['crash_rate']:.1f}%; speed {a['speed']:.2f} to {b['speed']:.2f} m/s."  # Include f"{LABELS[name]}: collisions {100 * a['crash rate']:.1f}% to {100 * b['crash rate']:.1f}%; speed {a['speed']:.2f} to {b['speed']:.2f} m/s." in the surrounding operation.
        )  # Finish assembling highlights.append inputs.
    body += (  # Increase body using (('<section><h2>Normal-traffic findings</h2><ul>' plus ''.join applied to '<li>' + x + '</li>' for each x in highlights) plus the report text).
        "<section><h2>Normal-traffic findings</h2><ul>"  # Include (('<section><h2>Normal-traffic findings</h2><ul>' plus ''.join applied to '<li>' + x + '</li>' for each x in highlights) plus the report text) in the surrounding operation.
        + "".join("<li>" + x + "</li>" for x in highlights)  # Include ''.join applied to (('<li>' plus x) plus '</li>') for each x in highlights in the surrounding operation.
        + "</ul><p>Budget effects are paired within each trajectory. These development comparisons do not establish generalization to reserved scenarios or real driving.</p></section>"  # Include the report text in the surrounding operation.
    )  # Finish assembling the grouped values above.
    body += '<section><h2>Training budget and physical performance</h2><p>Networks are trained from scratch for 40,000 steps. Epsilon reaches .05 at step 12,001 in both budgets. Reward, traffic, architecture and optimizer are unchanged. Solid lines average three training seeds; faint lines show their paired checkpoints.</p><img src="training_budget_budget.png"></section>'  # Increase body using the report text.
    body += '<section><h2>Learning curves</h2><p>Training rewards include exploration and variable episode lengths; use frozen-policy evaluations below to judge performance.</p><img src="training_budget_learning.png"></section>'  # Increase body using the report text.
    body += (  # Increase body using ((the report text plus table applied to the listed settings or measurements in their recorded order, [LABELS[r['policy']], r['budget'], r['density'], f"{100 * r['crash rate']:.1f}%", f"{r['speed']:.2f}", f"{r['distance']:.0f}", f"{ for each r in summaries) plus '</section>').
        "<section><h2>All results</h2><p>30 shared development scenarios per density per model. Speed is the mean of episode-average speeds. Worst 10% means the three lowest undiscounted episode returns per seed, then averaged across seeds.</p>"  # Include ((the report text plus table applied to the listed settings or measurements in their recorded order, [LABELS[r['policy']], r['budget'], r['density'], f"{100 * r['crash rate']:.1f}%", f"{r['speed']:.2f}", f"{r['distance']:.0f}", f"{ for each r in summaries) plus '</section>') in the surrounding operation.
        + table(  # Include table applied to the listed settings or measurements in their recorded order, the listed settings or measurements in their recorded order for each r in summaries in the surrounding operation.
            [  # Include the listed settings or measurements in their recorded order in the surrounding operation.
                "Policy",  # Include 'Policy' in the surrounding operation.
                "Steps",  # Include 'Steps' in the surrounding operation.
                "Density",  # Include 'Density' in the surrounding operation.
                "Collision",  # Include 'Collision' in the surrounding operation.
                "Speed m/s",  # Include 'Speed m/s' in the surrounding operation.
                "Distance m",  # Include 'Distance m' in the surrounding operation.
                "Return",  # Include 'Return' in the surrounding operation.
                "Worst 10%",  # Include 'Worst 10%' in the surrounding operation.
            ],  # Finish assembling the grouped values above.
            [  # Include the listed settings or measurements in their recorded order for each r in summaries in the surrounding operation.
                [  # Include the listed settings or measurements in their recorded order in the surrounding operation.
                    LABELS[r["policy"]],  # Include the entries of LABELS selected by r['policy'] in the surrounding operation.
                    r["budget"],  # Include the entries of r selected by 'budget' in the surrounding operation.
                    r["density"],  # Include the entries of r selected by 'density' in the surrounding operation.
                    f"{100*r['crash_rate']:.1f}%",  # Include f"{100 * r['crash rate']:.1f}%" in the surrounding operation.
                    f"{r['speed']:.2f}",  # Include f"{r['speed']:.2f}" in the surrounding operation.
                    f"{r['distance']:.0f}",  # Include f"{r['distance']:.0f}" in the surrounding operation.
                    f"{r['mean_return']:.2f}",  # Include f"{r['mean return']:.2f}" in the surrounding operation.
                    f"{r['worst10']:.2f}",  # Include f"{r['worst10']:.2f}" in the surrounding operation.
                ]  # Finish assembling the grouped values above.
                for r in summaries  # Include r, summaries in the surrounding operation.
            ],  # Finish assembling table inputs.
        )  # Finish assembling table inputs.
        + "</section>"  # Include '</section>' in the surrounding operation.
    )  # Finish assembling the grouped values above.
    body += (  # Increase body using ((the report text plus table applied to the listed settings or measurements in their recorded order, [LABELS[r['policy']], r['density'], f"{r['crashed']['mean difference'] * 100:+.1f}", str([round(x * 100, 1) for x in r['crashed'][ for each r in comparisons) plus '</section>').
        "<section><h2>Paired budget effects: 40k minus 20k</h2><p>Negative collision differences favor 40k. Exploratory 95% intervals resample training seeds and shared scenarios as crossed factors, preserving checkpoint pairing (5,000 replicates). Three training seeds give limited uncertainty estimates. Repeated development evaluation is not a confirmatory test.</p>"  # Include ((the report text plus table applied to the listed settings or measurements in their recorded order, [LABELS[r['policy']], r['density'], f"{r['crashed']['mean difference'] * 100:+.1f}", str([round(x * 100, 1) for x in r['crashed'][ for each r in comparisons) plus '</section>') in the surrounding operation.
        + table(  # Include table applied to the listed settings or measurements in their recorded order, the listed settings or measurements in their recorded order for each r in comparisons in the surrounding operation.
            [  # Include the listed settings or measurements in their recorded order in the surrounding operation.
                "Policy",  # Include 'Policy' in the surrounding operation.
                "Density",  # Include 'Density' in the surrounding operation.
                "Collision change pp",  # Include 'Collision change pp' in the surrounding operation.
                "Exploratory interval pp",  # Include 'Exploratory interval pp' in the surrounding operation.
                "Per-seed change pp",  # Include 'Per-seed change pp' in the surrounding operation.
                "Speed change m/s",  # Include 'Speed change m/s' in the surrounding operation.
            ],  # Finish assembling the grouped values above.
            [  # Include the listed settings or measurements in their recorded order for each r in comparisons in the surrounding operation.
                [  # Include the listed settings or measurements in their recorded order in the surrounding operation.
                    LABELS[r["policy"]],  # Include the entries of LABELS selected by r['policy'] in the surrounding operation.
                    r["density"],  # Include the entries of r selected by 'density' in the surrounding operation.
                    f"{r['crashed']['mean_difference']*100:+.1f}",  # Include f"{r['crashed']['mean difference'] * 100:+.1f}" in the surrounding operation.
                    str(  # Include round applied to (x times 100), 1 for each x in the entries of the entries of r selected by 'crashed' selected by 'exploratory_95_percent_interval' as a text value in the surrounding operation.
                        [  # Include round applied to (x times 100), 1 for each x in the entries of the entries of r selected by 'crashed' selected by 'exploratory_95_percent_interval' in the surrounding operation.
                            round(x * 100, 1)  # Include round applied to (x times 100), 1 in the surrounding operation.
                            for x in r["crashed"]["exploratory_95_percent_interval"]  # Include x, the entries of the entries of r selected by 'crashed' selected by 'exploratory_95_percent_interval' in the surrounding operation.
                        ]  # Finish assembling str inputs.
                    ),  # Finish assembling str inputs.
                    str(  # Include round applied to (x times 100), 1 for each x in the entries of the entries of r selected by 'crashed' selected by 'per_training_seed_differences' as a text value in the surrounding operation.
                        [  # Include round applied to (x times 100), 1 for each x in the entries of the entries of r selected by 'crashed' selected by 'per_training_seed_differences' in the surrounding operation.
                            round(x * 100, 1)  # Include round applied to (x times 100), 1 in the surrounding operation.
                            for x in r["crashed"]["per_training_seed_differences"]  # Include x, the entries of the entries of r selected by 'crashed' selected by 'per_training_seed_differences' in the surrounding operation.
                        ]  # Finish assembling str inputs.
                    ),  # Finish assembling str inputs.
                    f"{r['mean_speed_m_s']['mean_difference']:+.2f}",  # Include f"{r['mean speed m s']['mean difference']:+.2f}" in the surrounding operation.
                ]  # Finish assembling the grouped values above.
                for r in comparisons  # Include r, comparisons in the surrounding operation.
            ],  # Finish assembling table inputs.
        )  # Finish assembling table inputs.
        + "</section>"  # Include '</section>' in the surrounding operation.
    )  # Finish assembling the grouped values above.
    body += (  # Increase body using ((the report text plus table applied to 'Density', 'Collision change pp', 'Exploratory interval pp', 'Per-seed change pp', 'Speed change m/s', [r['density'], f"{r['crashed']['mean difference'] * 100:+.1f}", str([round(x * 100, 1) for x in r['crashed']['exploratory 95 perce for each r in controls) plus '</section>').
        "<section><h2>Frozen speed control at 40k</h2><p>Lower quarter minus mean deployment with the old 25 m/s governor. This threshold was not reselected at 40k. Check residual speed differences before attributing collision changes to the risk rule.</p>"  # Include ((the report text plus table applied to 'Density', 'Collision change pp', 'Exploratory interval pp', 'Per-seed change pp', 'Speed change m/s', [r['density'], f"{r['crashed']['mean difference'] * 100:+.1f}", str([round(x * 100, 1) for x in r['crashed']['exploratory 95 perce for each r in controls) plus '</section>') in the surrounding operation.
        + table(  # Include table applied to 'Density', 'Collision change pp', 'Exploratory interval pp', 'Per-seed change pp', 'Speed change m/s', the entries of r selected by 'density', f"{r['crashed']['mean difference'] * 100:+.1f}", [round(x * 100, 1) for x in r['crashed']['exploratory 95 percent interval']] as a text value, [round(x * 100, 1) for x in r['crashed']['per training seed differences']] as a text value, f"{r['mean speed m s']['mean difference']:+.2f}" for each r in controls in the surrounding operation.
            [  # Include 'Density', 'Collision change pp', 'Exploratory interval pp', 'Per-seed change pp', 'Speed change m/s' in the surrounding operation.
                "Density",  # Include 'Density' in the surrounding operation.
                "Collision change pp",  # Include 'Collision change pp' in the surrounding operation.
                "Exploratory interval pp",  # Include 'Exploratory interval pp' in the surrounding operation.
                "Per-seed change pp",  # Include 'Per-seed change pp' in the surrounding operation.
                "Speed change m/s",  # Include 'Speed change m/s' in the surrounding operation.
            ],  # Finish assembling the grouped values above.
            [  # Include the entries of r selected by 'density', f"{r['crashed']['mean difference'] * 100:+.1f}", round(x * 100, 1) for each x in r['crashed']['exploratory 95 percent interval'] as a text value, round(x * 100, 1) for each x in r['crashed']['per training seed differences'] as a text value, f"{r['mean speed m s']['mean difference']:+.2f}" for each r in controls in the surrounding operation.
                [  # Include the entries of r selected by 'density', f"{r['crashed']['mean difference'] * 100:+.1f}", round applied to x * 100, 1 for each x in the entries of r['crashed'] selected by 'exploratory_95_percent_interval' as a text value, round applied to x * 100, 1 for each x in the entries of r['crashed'] selected by 'per_training_seed_differences' as a text value, f"{r['mean speed m s']['mean difference']:+.2f}" in the surrounding operation.
                    r["density"],  # Include the entries of r selected by 'density' in the surrounding operation.
                    f"{r['crashed']['mean_difference']*100:+.1f}",  # Include f"{r['crashed']['mean difference'] * 100:+.1f}" in the surrounding operation.
                    str(  # Include round applied to (x times 100), 1 for each x in the entries of the entries of r selected by 'crashed' selected by 'exploratory_95_percent_interval' as a text value in the surrounding operation.
                        [  # Include round applied to (x times 100), 1 for each x in the entries of the entries of r selected by 'crashed' selected by 'exploratory_95_percent_interval' in the surrounding operation.
                            round(x * 100, 1)  # Include round applied to (x times 100), 1 in the surrounding operation.
                            for x in r["crashed"]["exploratory_95_percent_interval"]  # Include x, the entries of the entries of r selected by 'crashed' selected by 'exploratory_95_percent_interval' in the surrounding operation.
                        ]  # Finish assembling str inputs.
                    ),  # Finish assembling str inputs.
                    str(  # Include round applied to (x times 100), 1 for each x in the entries of the entries of r selected by 'crashed' selected by 'per_training_seed_differences' as a text value in the surrounding operation.
                        [  # Include round applied to (x times 100), 1 for each x in the entries of the entries of r selected by 'crashed' selected by 'per_training_seed_differences' in the surrounding operation.
                            round(x * 100, 1)  # Include round applied to (x times 100), 1 in the surrounding operation.
                            for x in r["crashed"]["per_training_seed_differences"]  # Include x, the entries of the entries of r selected by 'crashed' selected by 'per_training_seed_differences' in the surrounding operation.
                        ]  # Finish assembling str inputs.
                    ),  # Finish assembling str inputs.
                    f"{r['mean_speed_m_s']['mean_difference']:+.2f}",  # Include f"{r['mean speed m s']['mean difference']:+.2f}" in the surrounding operation.
                ]  # Finish assembling the grouped values above.
                for r in controls  # Include r, controls in the surrounding operation.
            ],  # Finish assembling table inputs.
        )  # Finish assembling table inputs.
        + "</section>"  # Include '</section>' in the surrounding operation.
    )  # Finish assembling the grouped values above.
    body += (  # Increase body using ((the report text plus table applied to the listed settings or measurements in their recorded order, [r['policy'], r['budget'], r['density'], str([round(x, 3) for x in r['mean prediction bias bounds']]), f"{r['minimum mean absolute for each r in calibration) plus '</section>').
        '<section><h2>Return prediction diagnostics</h2><p>60 separate development scenarios per density per model. Successful rollouts right-censor the continuing return; whiskers bound missing continuation. These are not statistical confidence intervals. Lower-quarter deployment changes the continuation policy relative to mean-greedy training.</p><img src="training_budget_bias.png">'  # Include ((the report text plus table applied to the listed settings or measurements in their recorded order, [r['policy'], r['budget'], r['density'], str([round(x, 3) for x in r['mean prediction bias bounds']]), f"{r['minimum mean absolute for each r in calibration) plus '</section>') in the surrounding operation.
        + table(  # Include table applied to the listed settings or measurements in their recorded order, the listed settings or measurements in their recorded order for each r in calibration in the surrounding operation.
            [  # Include the listed settings or measurements in their recorded order in the surrounding operation.
                "Policy",  # Include 'Policy' in the surrounding operation.
                "Steps",  # Include 'Steps' in the surrounding operation.
                "Density",  # Include 'Density' in the surrounding operation.
                "Bias bounds",  # Include 'Bias bounds' in the surrounding operation.
                "Compatibility gap",  # Include 'Compatibility gap' in the surrounding operation.
                "Adjacent crossings",  # Include 'Adjacent crossings' in the surrounding operation.
            ],  # Finish assembling the grouped values above.
            [  # Include the listed settings or measurements in their recorded order for each r in calibration in the surrounding operation.
                [  # Include the listed settings or measurements in their recorded order in the surrounding operation.
                    r["policy"],  # Include the entries of r selected by 'policy' in the surrounding operation.
                    r["budget"],  # Include the entries of r selected by 'budget' in the surrounding operation.
                    r["density"],  # Include the entries of r selected by 'density' in the surrounding operation.
                    str([round(x, 3) for x in r["mean_prediction_bias_bounds"]]),  # Include round applied to x, 3 for each x in the entries of r selected by 'mean_prediction_bias_bounds' as a text value in the surrounding operation.
                    f"{r['minimum_mean_absolute_compatibility_gap']:.3f}",  # Include f"{r['minimum mean absolute compatibility gap']:.3f}" in the surrounding operation.
                    f"{100*r['quantile_crossing_fraction']:.1f}%",  # Include f"{100 * r['quantile crossing fraction']:.1f}%" in the surrounding operation.
                ]  # Finish assembling the grouped values above.
                for r in calibration  # Include r, calibration in the surrounding operation.
            ],  # Finish assembling table inputs.
        )  # Finish assembling table inputs.
        + "</section>"  # Include '</section>' in the surrounding operation.
    )  # Finish assembling the grouped values above.
    body += (  # Increase body using ((the report text plus sum applied to (r['prefix weights equal'] for r in prefix) as a text value) plus the report text).
        "<section><h2>Reproducibility and limits</h2><p>Exact original 20k online-weight matches: "  # Include ((the report text plus sum applied to (r['prefix weights equal'] for r in prefix) as a text value) plus the report text) in the surrounding operation.
        + str(sum(r["prefix_weights_equal"] for r in prefix))  # Include sum applied to the entries of r selected by 'prefix_weights_equal' for each r in prefix as a text value in the surrounding operation.
        + "/6. Checkpoints share training history and are not independent runs. Density randomization was deferred to isolate budget. Predicted quantiles are not safety probabilities; cautious selection has no global CVaR-optimality guarantee. HighwayEnv is a simplified simulator.</p><p>Run <code>python run_training_budget.py --jobs 3</code> and <code>python -m pytest -q</code>. Existing artifacts are validated before reuse. The original risk analysis viewer is preserved.</p></section></body></html>"  # Include the report text in the surrounding operation.
    )  # Finish assembling the grouped values above.
    (viewer / "index.html").write_text(body)  # (viewer / 'index.html').write text applied to body.
    print("training budget experiment report saved", flush=True)  # Print applied to 'training budget experiment report saved'.


if __name__ == "__main__":  # Run this workflow when launched from Terminal; importing its helpers must not launch experiments.
    report()  # The result of report.
