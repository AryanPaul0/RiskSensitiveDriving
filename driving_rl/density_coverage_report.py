"""Create the density coverage results page.

Read saved numbers, calculate clearly labeled summaries, draw figures, and write
an HTML page. This file presents evidence; it does not train or select models.
Long HTML strings are page templates. The reinforcement-learning calculations
live in agents.py and experiment.py, not inside the page layout.

Original implementation notes:
Reports for equal-budget traffic-density coverage experiments.
"""

import json  # Make json available for this file's calculations and workflow.
from pathlib import Path  # Make Path available for this file's calculations and workflow.
import shutil  # Make shutil available for this file's calculations and workflow.
import numpy as np  # Make numpy available for this file's calculations and workflow.
import matplotlib  # Make matplotlib available for this file's calculations and workflow.

matplotlib.use("Agg")  # Matplotlib.use applied to 'agg'.
import matplotlib.pyplot as plt  # Make matplotlib.pyplot available for this file's calculations and workflow.
from .analysis import paired_matrix, crossed_bootstrap, summarize_group  # Make paired_matrix, crossed_bootstrap, summarize_group available for this file's calculations and workflow.
from .calibration import summarize_rows  # Make summarize_rows available for this file's calculations and workflow.
from .experiment import save_json  # Make save_json available for this file's calculations and workflow.

ROOT = Path(__file__).resolve().parents[1]  # Set ROOT to the entries of Path( file ).resolve().parents selected by 1.
STAGE = ROOT / "results/density_coverage"  # Set STAGE to the path to 'results/density_coverage' inside ROOT.
POLICIES = {  # Set POLICIES to the named settings or measurements assembled below.
    "mean": "Mean selector",  # Record 'mean' as 'Mean selector'.
    "tail25": "Lower-quarter selector",  # Record 'tail25' as 'Lower-quarter selector'.
    "governed": "Mean + 25 m/s governor",  # Record 'governed' as 'Mean + 25 m/s governor'.
}  # Finish assembling the grouped values above.


def read(p):  # Read saved JSON data into ordinary Python dictionaries and lists.
    """Read saved JSON data into ordinary Python dictionaries and lists."""  # Describe this operation for readers and help().
    return json.loads(p.read_text())  # Return json.loads applied to the result of p.read text.


def load(regime, policy, block):  # Read the result files belonging to one comparison group.
    """Read the result files belonging to one comparison group. The group controls
    which saved model and decision rule are represented; it does not train a model.
    """
    return [  # Return read applied to (ROOT divided by f'results/training budget/eval qrdqn {s} 40000 {policy}.json') when regime == 'fixed' and block == 'main', otherwise (STAGE divided by f'eval {regime} {policy} seed{s} {block}.json') for each s in range applied to 3.
        read(  # Include read applied to (ROOT divided by f'results/training budget/eval qrdqn {s} 40000 {policy}.json') when regime equals 'fixed' and block equals 'main', otherwise (STAGE divided by f'eval {regime} {policy} seed{s} {block}.json') in the surrounding operation.
            ROOT / f"results/training_budget/eval_qrdqn_{s}_40000_{policy}.json"  # Include (ROOT divided by f'results/training budget/eval qrdqn {s} 40000 {policy}.json') when regime equals 'fixed' and block equals 'main', otherwise (STAGE divided by f'eval {regime} {policy} seed{s} {block}.json') in the surrounding operation.
            if regime == "fixed" and block == "main"  # Include regime equals 'fixed' and block equals 'main' in the surrounding operation.
            else STAGE / f"eval_{regime}_{policy}_seed{s}_{block}.json"  # Include (STAGE divided by f'eval {regime} {policy} seed{s} {block}.json') in the surrounding operation.
        )  # Finish assembling read inputs.
        for s in range(3)  # Include s, range applied to 3 in the surrounding operation.
    ]  # Finish assembling the grouped values above.


def table(headers, rows):  # Build the HTML for a table of saved numbers.
    """Build the HTML for a table of saved numbers. Headers label columns; each
    row supplies values in the same order. This changes presentation, not the data.
    """
    return (  # Return (((('<div class="scroll"><table><thead><tr>' plus ''.join(('<th>' + str(v) + '</th>' for v in headers))) plus '</tr></thead><tbody>') plus ''.join applied to '<tr>' + ''.join(('<td>' + str(v) + '</td>' for v in row)) + '</tr>' for each row in rows) plus '</tbody></table></div>').
        '<div class="scroll"><table><thead><tr>'  # Include (((('<div class="scroll"><table><thead><tr>' plus ''.join(('<th>' + str(v) + '</th>' for v in headers))) plus '</tr></thead><tbody>') plus ''.join applied to '<tr>' + ''.join(('<td>' + str(v) + '</td>' for v in row)) + '</tr>' for each row in rows) plus '</tbody></table></div>') in the surrounding operation.
        + "".join("<th>" + str(v) + "</th>" for v in headers)  # Include ''.join applied to (('<th>' plus str(v)) plus '</th>') for each v in headers in the surrounding operation.
        + "</tr></thead><tbody>"  # Include '</tr></thead><tbody>' in the surrounding operation.
        + "".join(  # Include ''.join applied to (('<tr>' plus ''.join(('<td>' + str(v) + '</td>' for v in row))) plus '</tr>') for each row in rows in the surrounding operation.
            "<tr>" + "".join("<td>" + str(v) + "</td>" for v in row) + "</tr>"  # Include (('<tr>' plus ''.join applied to '<td>' + str(v) + '</td>' for each v in row) plus '</tr>') in the surrounding operation.
            for row in rows  # Include row, rows in the surrounding operation.
        )  # Finish assembling ''.join inputs.
        + "</tbody></table></div>"  # Include '</tbody></table></div>' in the surrounding operation.
    )  # Finish assembling the grouped values above.


def report():  # Build graphs and an HTML page from saved result files.
    """Build graphs and an HTML page from saved result files.
    The arrays below hold measured results; plotting does not change the models.
    HTML strings describe page layout, while Python inserts the saved numbers.
    Keep primary comparisons separate from exploratory comparisons in the report.
    """
    viewer = ROOT / "results/report"  # Set viewer to the path to 'results/report' inside ROOT.
    backup = ROOT / "results/training_budget-report"  # Set backup to the path to 'results/training_budget-report' inside ROOT.
    if not backup.exists():  # Use this branch when not the result of backup.exists.
        shutil.copytree(viewer, backup)  # Shutil.copytree applied to viewer, backup.
    groups = {}  # Set groups to the named settings or measurements assembled below.
    contrasts = []  # Set contrasts to .
    calibration = []  # Set calibration to .
    exposure = []  # Set exposure to .
    controls = []  # Set controls to .
    for block, densities in [("main", [1.0, 1.5, 2.0]), ("transfer", [1.25, 2.5])]:  # Process each block, densities from 'main', 1.0, 1.5, 2.0, 'transfer', 1.25, 2.5.
        groups[block] = {}  # Set the entries of groups selected by block to the named settings or measurements assembled below.
        for regime in ["fixed", "mixed"]:  # Process each regime from 'fixed', 'mixed'.
            groups[block][regime] = {  # Set the entries of the entries of groups selected by block selected by regime to summarize group applied to load applied to regime, p, block for each p in POLICIES.
                p: summarize_group(load(regime, p, block)) for p in POLICIES  # Include p, summarize group applied to load applied to regime, p, block, p, POLICIES in the surrounding operation.
            }  # Finish assembling the grouped values above.
        for policy in POLICIES:  # Process each policy from POLICIES.
            for density in densities:  # Process each the traffic-density setting from densities.
                item = {"block": block, "policy": policy, "density": density}  # Set item to the named settings or measurements assembled below.
                for metric in ["crashed", "mean_speed_m_s", "distance_m", "return"]:  # Compare one measured quantity at a time, keeping the scenarios paired.
                    _, matrix = paired_matrix(  # Set the unused result, matrix to paired matrix applied to load applied to 'mixed', policy, block, load applied to 'fixed', policy, block, the traffic-density setting, metric.
                        load("mixed", policy, block),  # Include load applied to 'mixed', policy, block in the surrounding operation.
                        load("fixed", policy, block),  # Include load applied to 'fixed', policy, block in the surrounding operation.
                        density,  # Include the traffic-density setting in the surrounding operation.
                        metric,  # Include metric in the surrounding operation.
                    )  # Finish assembling paired matrix inputs.
                    item[metric] = crossed_bootstrap(matrix)  # Set the entries of item selected by metric to crossed bootstrap applied to matrix.
                contrasts.append(item)  # Add item to contrasts.
        for regime in ["fixed", "mixed"]:  # Process each regime from 'fixed', 'mixed'.
            for density in densities:  # Process each the traffic-density setting from densities.
                item = {"block": block, "regime": regime, "density": density}  # Set item to the named settings or measurements assembled below.
                for metric in ["crashed", "mean_speed_m_s"]:  # Compare one measured quantity at a time, keeping the scenarios paired.
                    _, matrix = paired_matrix(  # Set the unused result, matrix to paired matrix applied to load applied to regime, 'tail25', block, load applied to regime, 'governed', block, the traffic-density setting, metric.
                        load(regime, "tail25", block),  # Include load applied to regime, 'tail25', block in the surrounding operation.
                        load(regime, "governed", block),  # Include load applied to regime, 'governed', block in the surrounding operation.
                        density,  # Include the traffic-density setting in the surrounding operation.
                        metric,  # Include metric in the surrounding operation.
                    )  # Finish assembling paired matrix inputs.
                    item[metric] = crossed_bootstrap(matrix)  # Set the entries of item selected by metric to crossed bootstrap applied to matrix.
                controls.append(item)  # Add item to controls.
    for regime in ["fixed", "mixed"]:  # Process each regime from 'fixed', 'mixed'.
        folder = ROOT / "results/training_budget" if regime == "fixed" else STAGE  # Set folder to the path to 'results/training_budget' inside ROOT when regime equals 'fixed', otherwise STAGE.
        for policy in ["mean", "tail25"]:  # Process each policy from 'mean', 'tail25'.
            data = [  # Set data to read applied to (folder divided by f'calibration {policy} seed{s}.json') for each s in range applied to 3.
                read(folder / f"calibration_{policy}_seed{s}.json") for s in range(3)  # Include read applied to (folder divided by f'calibration {policy} seed{s}.json'), s, range applied to 3 in the surrounding operation.
            ]  # Finish assembling the grouped values above.
            for density in [1.0, 1.5, 2.0]:  # Process each the traffic-density setting from 1.0, 1.5, 2.0.
                rows = [  # Set rows to r for each d in data.
                    r for d in data for r in d["episodes"] if r["density"] == density  # Include r, d, data, r, the entries of d selected by 'episodes', the entries of r selected by 'density' equals the traffic-density setting in the surrounding operation.
                ]  # Finish assembling the grouped values above.
                calibration.append(  # Add the named settings or measurements assembled below to calibration.
                    {  # Include the named settings or measurements assembled below in the surrounding operation.
                        "regime": regime,  # Record 'regime' as regime.
                        "policy": policy,  # Record 'policy' as policy.
                        "density": density,  # Record 'density' as the traffic-density setting.
                        **summarize_rows(rows),  # Include summarize rows applied to rows in the surrounding operation.
                    }  # Finish assembling the grouped values above.
                )  # Finish assembling calibration.append inputs.
    for seed in range(3):  # Process each seed from range applied to 3.
        d = read(STAGE / f"qrdqn_seed{seed}/summary.json")  # Set d to read applied to (STAGE divided by f'qrdqn seed{seed}/summary.json').
        if sum(d["steps_by_density"].values()) != 40000:  # Use this branch when sum applied to the result of d['steps by density'].values differs from 40000.
            raise ValueError("Training exposure does not sum to budget")  # Reject this invalid case instead of continuing with misleading experiment results.
        exposure.append({"seed": seed, **d})  # Add the named settings or measurements assembled below to exposure.
    result = {  # Set result to the named settings or measurements assembled below.
        "design": read(STAGE / "design.json"),  # Record 'design' as read applied to the path to 'design.json' inside STAGE.
        "groups": groups,  # Record 'groups' as groups.
        "mixed_minus_fixed": contrasts,  # Record 'mixed_minus_fixed' as contrasts.
        "tail_minus_governed": controls,  # Record 'tail_minus_governed' as controls.
        "calibration": calibration,  # Record 'calibration' as calibration.
        "training_exposure": exposure,  # Record 'training_exposure' as exposure.
    }  # Finish assembling the grouped values above.
    save_json(STAGE / "summary.json", result)  # Save the following results/settings as json: the path to 'summary.json' inside stage, result.
    plt.rcParams.update(  # Plt.rcparams.update applied to the named settings or measurements assembled below.
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}  # Record 'font.size' as 10.
    )  # Finish assembling plt.rcParams.update inputs.
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))  # Set fig, axes to plt.subplots applied to 2, 3.
    for col, policy in enumerate(POLICIES):  # Process each col, policy from enumerate applied to POLICIES.
        for regime, color in [("fixed", "C0"), ("mixed", "C1")]:  # Process each regime, color from 'fixed', 'C0', 'mixed', 'C1'.
            rows = groups["main"][regime][policy]  # Set rows to the entries of the entries of the entries of groups selected by 'main' selected by regime selected by policy.
            x = [r["density"] for r in rows]  # Set x to the entries of r selected by 'density' for each r in rows.
            for row, metric, scale in [  # Process each row, metric, scale from 0, 'crash_rate', 100, 1, 'mean_speed_m_s', 1.
                (0, "crash_rate", 100),  # Include 0, 'crash_rate', 100 in the surrounding operation.
                (1, "mean_speed_m_s", 1),  # Include 1, 'mean_speed_m_s', 1 in the surrounding operation.
            ]:  # Finish assembling the grouped values above.
                ax = axes[row, col]  # Set ax to the entries of axes selected by (row, col).
                ax.plot(  # Ax.plot applied to x, (the entries of r selected by metric times scale) for each r in rows, '-o'.
                    x,  # Include x in the surrounding operation.
                    [r[metric] * scale for r in rows],  # Include (the entries of r selected by metric times scale) for each r in rows in the surrounding operation.
                    "-o",  # Include '-o' in the surrounding operation.
                    color=color,  # Include color in the surrounding operation.
                    label=regime,  # Include regime in the surrounding operation.
                )  # Finish assembling ax.plot inputs.
                for s in range(3):  # Process each s from range applied to 3.
                    ax.plot(  # Ax.plot applied to x, (the entries of r['per seed'][s] selected by metric times scale) for each r in rows.
                        x,  # Include x in the surrounding operation.
                        [r["per_seed"][s][metric] * scale for r in rows],  # Include (the entries of the entries of r['per seed'] selected by s selected by metric times scale) for each r in rows in the surrounding operation.
                        color=color,  # Include color in the surrounding operation.
                        lw=0.8,  # Include 0.8 in the surrounding operation.
                        alpha=0.25,  # Include 0.25 in the surrounding operation.
                    )  # Finish assembling ax.plot inputs.
                ax.grid(alpha=0.2)  # The result of ax.grid.
                ax.set_xticks(x)  # Ax.set xticks applied to x.
                ax.set_xlabel("Evaluation density")  # Ax.set xlabel applied to 'evaluation density'.
        axes[0, col].set_title(POLICIES[policy])  # Axes[0, col].set title applied to the entries of policies selected by policy.
        axes[0, col].set_ylim(0, 105)  # Axes[0, col].set ylim applied to 0, 105.
        axes[0, col].legend()  # The result of axes[0, col].legend.
    axes[0, 0].set_ylabel("Episodes with collision (%)")  # Axes[0, 0].set ylabel applied to 'episodes with collision (%)'.
    axes[1, 0].set_ylabel("Mean episode speed (m/s)")  # Axes[1, 0].set ylabel applied to 'mean episode speed (m/s)'.
    fig.suptitle("Equal 40k-step budgets; faint lines show individual training seeds")  # Fig.suptitle applied to the report text.
    fig.tight_layout()  # The result of fig.tight layout.
    fig.savefig(viewer / "density_coverage_performance.png", dpi=150)  # Save a figure for the local results page; this does not affect agent behavior.
    plt.close(fig)  # Plt.close applied to fig.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))  # Set fig, axes to plt.subplots applied to 1, 2.
    for ax, policy in zip(axes, ["mean", "tail25"]):  # Process each ax, policy from matching entries grouped across axes, 'mean', 'tail25'.
        for regime, color, offset in [("fixed", "C0", -0.035), ("mixed", "C1", 0.035)]:  # Process each regime, color, offset from 'fixed', 'C0', negative 0.035, 'mixed', 'C1', 0.035.
            rows = [  # Set rows to r for each r in calibration, keeping only cases where the entries of r selected by 'regime' equals regime and the entries of r selected by 'policy' equals policy.
                r  # Include r in the surrounding operation.
                for r in calibration  # Include r, calibration in the surrounding operation.
                if r["regime"] == regime and r["policy"] == policy  # Include the entries of r selected by 'regime' equals regime and the entries of r selected by 'policy' equals policy in the surrounding operation.
            ]  # Finish assembling the grouped values above.
            lo = np.array([r["mean_prediction_bias_bounds"][0] for r in rows])  # Set lo to the entries of the entries of r selected by 'mean_prediction_bias_bounds' selected by 0 for each r in rows as a numerical array/tensor for batch calculations.
            hi = np.array([r["mean_prediction_bias_bounds"][1] for r in rows])  # Set hi to the entries of the entries of r selected by 'mean_prediction_bias_bounds' selected by 1 for each r in rows as a numerical array/tensor for batch calculations.
            mid = (lo + hi) / 2  # Set mid to ((lo plus hi) divided by 2).
            ax.errorbar(  # Ax.errorbar applied to (1.0, 1.5, 2.0 as a numerical array/tensor for batch calculations plus offset), mid.
                np.array([1.0, 1.5, 2.0]) + offset,  # Include (1.0, 1.5, 2.0 as a numerical array/tensor for batch calculations plus offset) in the surrounding operation.
                mid,  # Include mid in the surrounding operation.
                yerr=[mid - lo, hi - mid],  # Include (mid minus lo), (hi minus mid) in the surrounding operation.
                fmt="o-",  # Include 'o-' in the surrounding operation.
                color=color,  # Include color in the surrounding operation.
                label=regime,  # Include regime in the surrounding operation.
                capsize=4,  # Include 4 in the surrounding operation.
            )  # Finish assembling ax.errorbar inputs.
        ax.axhline(0, color="black", lw=0.8)  # Ax.axhline applied to 0.
        ax.set_xticks([1, 1.5, 2])  # Ax.set xticks applied to 1, 1.5, 2.
        ax.set_xlabel("Evaluation density")  # Ax.set xlabel applied to 'evaluation density'.
        ax.set_title(POLICIES[policy])  # Ax.set title applied to the entries of policies selected by policy.
        ax.grid(alpha=0.2)  # The result of ax.grid.
        ax.legend()  # The result of ax.legend.
    axes[0].set_ylabel("Prediction minus discounted return")  # Axes[0].set ylabel applied to 'prediction minus discounted return'.
    fig.suptitle("Bias bounds account for censoring, not sampling uncertainty")  # Fig.suptitle applied to 'bias bounds account for censoring, not sampling uncertainty'.
    fig.tight_layout()  # The result of fig.tight layout.
    fig.savefig(viewer / "density_coverage_bias.png", dpi=150)  # Save a figure for the local results page; this does not affect agent behavior.
    plt.close(fig)  # Plt.close applied to fig.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # Set fig, axes to plt.subplots applied to 1, 2.
    for ax, key, title in zip(  # Process each ax, key, title from matching entries grouped across axes, 'completed_episodes_by_density', 'steps_by_density', 'Completed episodes', 'Environment transitions'.
        axes,  # Include axes in the surrounding operation.
        ["completed_episodes_by_density", "steps_by_density"],  # Include 'completed_episodes_by_density', 'steps_by_density' in the surrounding operation.
        ["Completed episodes", "Environment transitions"],  # Include 'Completed episodes', 'Environment transitions' in the surrounding operation.
    ):  # Finish assembling zip inputs.
        bottom = np.zeros(3)  # Set bottom to np.zeros applied to 3.
        for d in [1.0, 1.5, 2.0]:  # Process each d from 1.0, 1.5, 2.0.
            total = np.array([sum(r[key].values()) for r in exposure])  # Set total to sum applied to the result of r[key].values for each r in exposure as a numerical array/tensor for batch calculations.
            v = np.array([r[key][str(d)] for r in exposure]) / total * 100  # Set v to ((r[key][str(d)] for each r in exposure as a numerical array/tensor for batch calculations divided by total) times 100).
            ax.bar(range(3), v, bottom=bottom, label=f"Density {d:g}")  # Ax.bar applied to range applied to 3, v.
            bottom += v  # Increase bottom using v.
        ax.set_xticks(range(3), ["Seed 0", "Seed 1", "Seed 2"])  # Ax.set xticks applied to range applied to 3, 'seed 0', 'seed 1', 'seed 2'.
        ax.set_ylim(0, 100)  # Ax.set ylim applied to 0, 100.
        ax.set_title(title)  # Ax.set title applied to title.
        ax.set_ylabel("Share (%)")  # Ax.set ylabel applied to 'share (%)'.
    axes[1].legend(loc="upper left", bbox_to_anchor=(1, 1))  # The result of axes[1].legend.
    fig.suptitle("Uniform episode sampling does not produce equal transition shares")  # Fig.suptitle applied to the report text.
    fig.tight_layout()  # The result of fig.tight layout.
    fig.savefig(viewer / "density_coverage_exposure.png", dpi=150)  # Save a figure for the local results page; this does not affect agent behavior.
    plt.close(fig)  # Plt.close applied to fig.
    primary = next(  # Set primary to next applied to r for each r in contrasts, keeping only cases where r['block'] equals 'main' and r['policy'] equals 'tail25' and r['density'] equals 2.0.
        r  # Include r in the surrounding operation.
        for r in contrasts  # Include r, contrasts in the surrounding operation.
        if r["block"] == "main" and r["policy"] == "tail25" and r["density"] == 2.0  # Include the entries of r selected by 'block' equals 'main' and the entries of r selected by 'policy' equals 'tail25' and the entries of r selected by 'density' equals 2.0 in the surrounding operation.
    )  # Finish assembling next inputs.
    value = primary["crashed"]  # Set value to the entries of primary selected by 'crashed'.
    interval = [100 * x for x in value["exploratory_95_percent_interval"]]  # Set interval to (100 times x) for each x in the entries of value selected by 'exploratory_95_percent_interval'.
    primary_text = f"At density 2, the lower-quarter selector's mixed-minus-fixed collision difference is {100*value['mean_difference']:+.1f} percentage points; exploratory 95% interval [{interval[0]:+.1f}, {interval[1]:+.1f}]."  # Set primary text to f"At density 2, the lower-quarter selector's mixed-minus-fixed collision difference is {100 * value['mean difference']:+.1f} percentage points; exploratory 95% interval [{interval[.
    document = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Traffic-density coverage · Risk-sensitive driving</title><style>body{font:17px/1.6 system-ui;color:#18283d;background:#f4f7fb;margin:auto;max-width:1160px;padding:36px}h1{font-size:38px;line-height:1.2}section{background:white;padding:24px;margin:24px 0;border-radius:14px}img{width:100%;height:auto}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;text-align:left;border-bottom:1px solid #dde3eb}th{background:#eef3fa}a{color:#175bbc}summary{cursor:pointer}</style></head><body><p>DISTRIBUTIONAL RL / TRAFFIC COVERAGE</p><h1>Does broader training traffic improve driving?</h1><p>QR-DQN trained at density 1 versus a per-episode mixture of densities 1, 1.5 and 2. Each setting has three training seeds and 40,000 interactions per model. Rewards, architecture, optimizer and exploration schedule remain fixed.</p><p><a href="../training_budget-report/index.html">Training-budget comparison</a> · <a href="../risk_analysis-report/index.html">Interactive risk frontier and recorded driving</a> · <a href="../density_coverage/summary.json">Complete numerical results</a> · <a href="../density_coverage/design.json">Recorded experiment design</a></p>"""  # Set document to the report text.
    document += (  # Increase document using (('<section><h2>Predeclared primary comparison</h2><p>' plus primary text) plus the report text).
        "<section><h2>Predeclared primary comparison</h2><p>"  # Include (('<section><h2>Predeclared primary comparison</h2><p>' plus primary text) plus the report text) in the surrounding operation.
        + primary_text  # Include primary text in the surrounding operation.
        + "</p><p>Negative differences favor mixture training. The selectors and 25 m/s governor are unchanged. This is a development comparison with only three training seeds; all confirmation scenarios remain reserved.</p></section>"  # Include the report text in the surrounding operation.
    )  # Finish assembling the grouped values above.

    def metric(regime, policy, density, key):  # Look up one already-calculated metric for the requested group and density.
        """Look up one already-calculated metric for the requested group and density."""  # Describe this operation for readers and help().
        return next(  # Return the entries of next applied to r for each r in the entries of groups['main'][regime] selected by policy, keeping only cases where r['density'] equals the traffic-density setting selected by key.
            r for r in groups["main"][regime][policy] if r["density"] == density  # Include r, r, the entries of the entries of the entries of groups selected by 'main' selected by regime selected by policy, the entries of r selected by 'density' equals the traffic-density setting in the surrounding operation.
        )[key]  # Finish assembling next inputs.

    d = 1.5  # Set d to 1.5.
    coverage_text = f"At density 1.5, cautious-selector collisions change from {100*metric('fixed','tail25',d,'crash_rate'):.1f}% to {100*metric('mixed','tail25',d,'crash_rate'):.1f}%, while speed changes from {metric('fixed','tail25',d,'mean_speed_m_s'):.2f} to {metric('mixed','tail25',d,'mean_speed_m_s'):.2f} m/s."  # Set coverage text to f"At density 1.5, cautious-selector collisions change from {100 * metric('fixed', 'tail25', d, 'crash rate'):.1f}% to {100 * metric('mixed', 'tail25', d, 'crash rate'):.1f}%, while.
    governor_text = f"The normal-traffic governor result worsens: {100*metric('fixed','governed',1.,'crash_rate'):.1f}% to {100*metric('mixed','governed',1.,'crash_rate'):.1f}% collisions. Broader training is not an improvement for every deployment rule."  # Set governor text to f"The normal-traffic governor result worsens: {100 * metric('fixed', 'governed', 1.0, 'crash rate'):.1f}% to {100 * metric('mixed', 'governed', 1.0, 'crash rate'):.1f}% collisions..
    document += (  # Increase document using (((('<section><h2>Main tradeoffs</h2><p>' plus coverage text) plus '</p><p>') plus governor text) plus the report text).
        "<section><h2>Main tradeoffs</h2><p>"  # Include (((('<section><h2>Main tradeoffs</h2><p>' plus coverage text) plus '</p><p>') plus governor text) plus the report text) in the surrounding operation.
        + coverage_text  # Include coverage text in the surrounding operation.
        + "</p><p>"  # Include '</p><p>' in the surrounding operation.
        + governor_text  # Include governor text in the surrounding operation.
        + "</p><p>Fewer observed collisions come with slower driving. The same-network governor comparisons retain residual speed differences, and uncertainty intervals remain exploratory. This study isolates coverage within QR-DQN; it does not compare DQN and QR-DQN under identical randomized training.</p></section>"  # Include the report text in the surrounding operation.
    )  # Finish assembling the grouped values above.
    document += '<section><h2>Performance across covered densities</h2><p>Same evaluation scenario block for both training settings (10000–10029). Densities 1.5 and 2 were absent from fixed-density training and are included in mixed training. Faint curves show seed variation.</p><img src="density_coverage_performance.png"></section>'  # Increase document using the report text.
    for block, title in [  # Process each block, title from 'main', 'Covered traffic settings', 'transfer', 'Additional traffic settings'.
        ("main", "Covered traffic settings"),  # Include 'main', 'Covered traffic settings' in the surrounding operation.
        ("transfer", "Additional traffic settings"),  # Include 'transfer', 'Additional traffic settings' in the surrounding operation.
    ]:  # Finish assembling the grouped values above.
        document += "<section><h2>" + title + "</h2>"  # Increase document using (('<section><h2>' plus title) plus '</h2>').
        if block == "transfer":  # Use this branch when block equals 'transfer'.
            document += "<p>Densities 1.25 and 2.5 use a separate development scenario block, 16000–16029. Density 1.25 is an intermediate value absent from the sampled mixture; density 2.5 is beyond both training ranges. These rows use different scenarios from the covered-settings table.</p>"  # Increase document using the report text.
        document += (  # Increase document using (table applied to the listed settings or measurements in their recorded order, the listed settings or measurements in their recorded order for each regime in 'fixed', 'mixed' plus '</section>').
            table(  # Include (table applied to the listed settings or measurements in their recorded order, the listed settings or measurements in their recorded order for each regime in 'fixed', 'mixed' plus '</section>') in the surrounding operation.
                [  # Include the listed settings or measurements in their recorded order in the surrounding operation.
                    "Training",  # Include 'Training' in the surrounding operation.
                    "Selector",  # Include 'Selector' in the surrounding operation.
                    "Density",  # Include 'Density' in the surrounding operation.
                    "Collisions",  # Include 'Collisions' in the surrounding operation.
                    "Speed m/s",  # Include 'Speed m/s' in the surrounding operation.
                    "Distance m",  # Include 'Distance m' in the surrounding operation.
                    "Worst 10% return",  # Include 'Worst 10% return' in the surrounding operation.
                ],  # Finish assembling the grouped values above.
                [  # Include the listed settings or measurements in their recorded order for each regime in 'fixed', 'mixed' in the surrounding operation.
                    [  # Include the listed settings or measurements in their recorded order in the surrounding operation.
                        regime,  # Include regime in the surrounding operation.
                        POLICIES[p],  # Include the entries of POLICIES selected by p in the surrounding operation.
                        r["density"],  # Include the entries of r selected by 'density' in the surrounding operation.
                        f"{100*r['crash_rate']:.1f}%",  # Include f"{100 * r['crash rate']:.1f}%" in the surrounding operation.
                        f"{r['mean_speed_m_s']:.2f}",  # Include f"{r['mean speed m s']:.2f}" in the surrounding operation.
                        f"{r['mean_distance_m']:.0f}",  # Include f"{r['mean distance m']:.0f}" in the surrounding operation.
                        f"{r['worst_10_percent_mean_return']:.2f}",  # Include f"{r['worst 10 percent mean return']:.2f}" in the surrounding operation.
                    ]  # Finish assembling the grouped values above.
                    for regime in ["fixed", "mixed"]  # Include regime, 'fixed', 'mixed' in the surrounding operation.
                    for p in POLICIES  # Include p, POLICIES in the surrounding operation.
                    for r in groups[block][regime][p]  # Include r, the entries of the entries of the entries of groups selected by block selected by regime selected by p in the surrounding operation.
                ],  # Finish assembling table inputs.
            )  # Finish assembling table inputs.
            + "</section>"  # Include '</section>' in the surrounding operation.
        )  # Finish assembling the grouped values above.
    document += (  # Increase document using ((the report text plus table applied to the listed settings or measurements in their recorded order, [r['block'], POLICIES[r['policy']], r['density'], f"{100 * r['crashed']['mean difference']:+.1f}", str([round(100 * x, 1) for x in for each r in contrasts) plus '</section>').
        "<section><h2>Paired collision effects: mixed minus fixed</h2><p>Exploratory intervals use 5,000 crossed bootstrap replicates over training seed IDs and shared scenarios, preserving policy pairing. Matching seed IDs shares initialization and exploration streams, but not the simulated trajectories. Three seeds and repeated development evaluation limit inference.</p>"  # Include ((the report text plus table applied to the listed settings or measurements in their recorded order, [r['block'], POLICIES[r['policy']], r['density'], f"{100 * r['crashed']['mean difference']:+.1f}", str([round(100 * x, 1) for x in for each r in contrasts) plus '</section>') in the surrounding operation.
        + table(  # Include table applied to the listed settings or measurements in their recorded order, the listed settings or measurements in their recorded order for each r in contrasts in the surrounding operation.
            [  # Include the listed settings or measurements in their recorded order in the surrounding operation.
                "Block",  # Include 'Block' in the surrounding operation.
                "Selector",  # Include 'Selector' in the surrounding operation.
                "Density",  # Include 'Density' in the surrounding operation.
                "Change pp",  # Include 'Change pp' in the surrounding operation.
                "Exploratory interval pp",  # Include 'Exploratory interval pp' in the surrounding operation.
                "Per-seed changes pp",  # Include 'Per-seed changes pp' in the surrounding operation.
                "Speed change m/s",  # Include 'Speed change m/s' in the surrounding operation.
            ],  # Finish assembling the grouped values above.
            [  # Include the listed settings or measurements in their recorded order for each r in contrasts in the surrounding operation.
                [  # Include the listed settings or measurements in their recorded order in the surrounding operation.
                    r["block"],  # Include the entries of r selected by 'block' in the surrounding operation.
                    POLICIES[r["policy"]],  # Include the entries of POLICIES selected by r['policy'] in the surrounding operation.
                    r["density"],  # Include the entries of r selected by 'density' in the surrounding operation.
                    f"{100*r['crashed']['mean_difference']:+.1f}",  # Include f"{100 * r['crashed']['mean difference']:+.1f}" in the surrounding operation.
                    str(  # Include round applied to (100 times x), 1 for each x in the entries of the entries of r selected by 'crashed' selected by 'exploratory_95_percent_interval' as a text value in the surrounding operation.
                        [  # Include round applied to (100 times x), 1 for each x in the entries of the entries of r selected by 'crashed' selected by 'exploratory_95_percent_interval' in the surrounding operation.
                            round(100 * x, 1)  # Include round applied to (100 times x), 1 in the surrounding operation.
                            for x in r["crashed"]["exploratory_95_percent_interval"]  # Include x, the entries of the entries of r selected by 'crashed' selected by 'exploratory_95_percent_interval' in the surrounding operation.
                        ]  # Finish assembling str inputs.
                    ),  # Finish assembling str inputs.
                    str(  # Include round applied to (100 times x), 1 for each x in the entries of the entries of r selected by 'crashed' selected by 'per_training_seed_differences' as a text value in the surrounding operation.
                        [  # Include round applied to (100 times x), 1 for each x in the entries of the entries of r selected by 'crashed' selected by 'per_training_seed_differences' in the surrounding operation.
                            round(100 * x, 1)  # Include round applied to (100 times x), 1 in the surrounding operation.
                            for x in r["crashed"]["per_training_seed_differences"]  # Include x, the entries of the entries of r selected by 'crashed' selected by 'per_training_seed_differences' in the surrounding operation.
                        ]  # Finish assembling str inputs.
                    ),  # Finish assembling str inputs.
                    f"{r['mean_speed_m_s']['mean_difference']:+.2f}",  # Include f"{r['mean speed m s']['mean difference']:+.2f}" in the surrounding operation.
                ]  # Finish assembling the grouped values above.
                for r in contrasts  # Include r, contrasts in the surrounding operation.
            ],  # Finish assembling table inputs.
        )  # Finish assembling table inputs.
        + "</section>"  # Include '</section>' in the surrounding operation.
    )  # Finish assembling the grouped values above.
    document += (  # Increase document using ((the report text plus table applied to the listed settings or measurements in their recorded order, [r['block'], r['regime'], r['density'], f"{100 * r['crashed']['mean difference']:+.1f}", str([round(100 * x, 1) for x in r['crashe for each r in controls) plus '</section>').
        "<section><h2>Cautious decisions versus speed control</h2><p>Lower quarter minus the mean selector with the fixed 25 m/s governor, within the same networks. The governor threshold is not reselected for mixed training; check residual speed differences.</p>"  # Include ((the report text plus table applied to the listed settings or measurements in their recorded order, [r['block'], r['regime'], r['density'], f"{100 * r['crashed']['mean difference']:+.1f}", str([round(100 * x, 1) for x in r['crashe for each r in controls) plus '</section>') in the surrounding operation.
        + table(  # Include table applied to the listed settings or measurements in their recorded order, the listed settings or measurements in their recorded order for each r in controls in the surrounding operation.
            [  # Include the listed settings or measurements in their recorded order in the surrounding operation.
                "Block",  # Include 'Block' in the surrounding operation.
                "Training",  # Include 'Training' in the surrounding operation.
                "Density",  # Include 'Density' in the surrounding operation.
                "Collision change pp",  # Include 'Collision change pp' in the surrounding operation.
                "Exploratory interval pp",  # Include 'Exploratory interval pp' in the surrounding operation.
                "Speed change m/s",  # Include 'Speed change m/s' in the surrounding operation.
            ],  # Finish assembling the grouped values above.
            [  # Include the listed settings or measurements in their recorded order for each r in controls in the surrounding operation.
                [  # Include the listed settings or measurements in their recorded order in the surrounding operation.
                    r["block"],  # Include the entries of r selected by 'block' in the surrounding operation.
                    r["regime"],  # Include the entries of r selected by 'regime' in the surrounding operation.
                    r["density"],  # Include the entries of r selected by 'density' in the surrounding operation.
                    f"{100*r['crashed']['mean_difference']:+.1f}",  # Include f"{100 * r['crashed']['mean difference']:+.1f}" in the surrounding operation.
                    str(  # Include round applied to (100 times x), 1 for each x in the entries of the entries of r selected by 'crashed' selected by 'exploratory_95_percent_interval' as a text value in the surrounding operation.
                        [  # Include round applied to (100 times x), 1 for each x in the entries of the entries of r selected by 'crashed' selected by 'exploratory_95_percent_interval' in the surrounding operation.
                            round(100 * x, 1)  # Include round applied to (100 times x), 1 in the surrounding operation.
                            for x in r["crashed"]["exploratory_95_percent_interval"]  # Include x, the entries of the entries of r selected by 'crashed' selected by 'exploratory_95_percent_interval' in the surrounding operation.
                        ]  # Finish assembling str inputs.
                    ),  # Finish assembling str inputs.
                    f"{r['mean_speed_m_s']['mean_difference']:+.2f}",  # Include f"{r['mean speed m s']['mean difference']:+.2f}" in the surrounding operation.
                ]  # Finish assembling the grouped values above.
                for r in controls  # Include r, controls in the surrounding operation.
            ],  # Finish assembling table inputs.
        )  # Finish assembling table inputs.
        + "</section>"  # Include '</section>' in the surrounding operation.
    )  # Finish assembling the grouped values above.
    document += (  # Increase document using ((the report text plus table applied to the listed settings or measurements in their recorded order, [r['regime'], r['policy'], r['density'], str([round(x, 3) for x in r['mean prediction bias bounds']]), f"{r['minimum mean absolute for each r in calibration) plus '</section>').
        '<section><h2>What the networks predict</h2><p>Same separate calibration scenarios (12000–12059) for both settings. The 30-step time cutoff right-censors continuing returns. Bounds retain surviving episodes; they are not statistical confidence intervals. Lower-quarter deployment changes the continuation policy relative to training.</p><img src="density_coverage_bias.png">'  # Include ((the report text plus table applied to the listed settings or measurements in their recorded order, [r['regime'], r['policy'], r['density'], str([round(x, 3) for x in r['mean prediction bias bounds']]), f"{r['minimum mean absolute for each r in calibration) plus '</section>') in the surrounding operation.
        + table(  # Include table applied to the listed settings or measurements in their recorded order, the listed settings or measurements in their recorded order for each r in calibration in the surrounding operation.
            [  # Include the listed settings or measurements in their recorded order in the surrounding operation.
                "Training",  # Include 'Training' in the surrounding operation.
                "Selector",  # Include 'Selector' in the surrounding operation.
                "Density",  # Include 'Density' in the surrounding operation.
                "Bias bounds",  # Include 'Bias bounds' in the surrounding operation.
                "Compatibility gap",  # Include 'Compatibility gap' in the surrounding operation.
                "Adjacent crossings",  # Include 'Adjacent crossings' in the surrounding operation.
            ],  # Finish assembling the grouped values above.
            [  # Include the listed settings or measurements in their recorded order for each r in calibration in the surrounding operation.
                [  # Include the listed settings or measurements in their recorded order in the surrounding operation.
                    r["regime"],  # Include the entries of r selected by 'regime' in the surrounding operation.
                    r["policy"],  # Include the entries of r selected by 'policy' in the surrounding operation.
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
    document += '<section><h2>Same-scenario recorded driving</h2><p>Both lower-quarter policies use training seed 0, scenario seed 10000, density 2. The scenario was fixed before inspecting the new outcomes and is illustrative rather than a performance estimate.</p><div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px"><article><h3>Fixed-density training</h3><img src="../density_coverage/replay_fixed.gif"></article><article><h3>Mixed-density training</h3><img src="../density_coverage/replay_mixed.gif"></article></div></section>'  # Increase document using the report text.
    document += '<section><h2>Actual training exposure</h2><p>Density is sampled uniformly at episode reset with a separate seeded generator. Short dense-traffic episodes contribute fewer transitions. The density label is not added to the observation; the network still receives the same 40 inputs. Equal step budgets therefore do not imply equal episode counts or equal data by density.</p><img src="density_coverage_exposure.png"></section>'  # Increase document using the report text.
    document += "<section><h2>Reproduce and interpret</h2><p><code>python run_density_coverage.py --jobs 3</code> validates and reuses saved artifacts. <code>--stage evaluate</code> skips training; <code>--stage report</code> rebuilds the viewer. Run <code>python -m pytest -q</code> for checks.</p><p>Only traffic density changes during training. The simulator retains simplified vehicle interactions; the observation is partial. Quantile forecasts are not collision probabilities, and cautious deployment has no guaranteed safety or global CVaR optimality. Results concern sampled development scenarios and should not be presented as real-road validation.</p></section></body></html>"  # Increase document using the report text.
    (viewer / "index.html").write_text(document)  # (viewer / 'index.html').write text applied to document.
    print(primary_text, flush=True)  # Print applied to primary text.
    print("Density coverage report saved", flush=True)  # Print applied to 'density coverage report saved'.


if __name__ == "__main__":  # Run this workflow when launched from Terminal; importing its helpers must not launch experiments.
    report()  # The result of report.
