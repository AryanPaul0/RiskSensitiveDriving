"""Create the baseline controls results page.

Read saved numbers, calculate clearly labeled summaries, draw figures, and write
an HTML page. This file presents evidence; it does not train or select models.
Long HTML strings are page templates. The reinforcement-learning calculations
live in agents.py and experiment.py, not inside the page layout.

Original implementation notes:
Distributional and scalar baselines with equal traffic coverage and speed controls.
"""

from pathlib import Path  # Make Path available for this file's calculations and workflow.
import hashlib  # Make hashlib available for this file's calculations and workflow.
import json  # Make json available for this file's calculations and workflow.
import shutil  # Make shutil available for this file's calculations and workflow.
import numpy as np  # Make numpy available for this file's calculations and workflow.
import matplotlib  # Make matplotlib available for this file's calculations and workflow.

matplotlib.use("Agg")  # Matplotlib.use applied to 'agg'.
import matplotlib.pyplot as plt  # Make matplotlib.pyplot available for this file's calculations and workflow.
from .analysis import summarize_group, paired_matrix, crossed_bootstrap  # Make summarize_group, paired_matrix, crossed_bootstrap available for this file's calculations and workflow.
from .experiment import save_json  # Make save_json available for this file's calculations and workflow.
from .density_coverage_report import table  # Make table available for this file's calculations and workflow.

ROOT = Path(__file__).resolve().parents[1]  # Set ROOT to the entries of Path( file ).resolve().parents selected by 1.
STAGE = ROOT / "results/baseline_controls"  # Set STAGE to the path to 'results/baseline_controls' inside ROOT.
LABELS = {  # Set LABELS to the named settings or measurements assembled below.
    "fixed_dqn": "Fixed DQN",  # Record 'fixed_dqn' as 'Fixed DQN'.
    "mixed_dqn": "Mixed DQN",  # Record 'mixed_dqn' as 'Mixed DQN'.
    "mixed_qr_mean": "Mixed QR mean",  # Record 'mixed_qr_mean' as 'Mixed QR mean'.
    "mixed_qr_tail": "Mixed QR lower quarter",  # Record 'mixed_qr_tail' as 'Mixed QR lower quarter'.
    "matched_dqn": "Speed-targeted DQN",  # Record 'matched_dqn' as 'Speed-targeted DQN'.
    "matched_qr": "Speed-targeted QR mean",  # Record 'matched_qr' as 'Speed-targeted QR mean'.
}  # Finish assembling the grouped values above.


def read(p):  # Read saved JSON data into ordinary Python dictionaries and lists.
    """Read saved JSON data into ordinary Python dictionaries and lists."""  # Describe this operation for readers and help().
    return json.loads(p.read_text())  # Return json.loads applied to the result of p.read text.


def load(group, block, selections):  # Read the result files belonging to one comparison group.
    """Read the result files belonging to one comparison group. The group controls
    which saved model and decision rule are represented; it does not train a model.
    """
    paths = []  # Set paths to .
    for seed in range(3):  # Process each seed from range applied to 3.
        if group == "fixed_dqn":  # Use this branch when group equals 'fixed_dqn'.
            p = (  # Set p to (ROOT divided by f'results/training budget/eval dqn {seed} 40000 mean.json') when block equals 'main', otherwise (STAGE divided by f'eval dqn fixed mean seed{seed} transfer.json').
                ROOT / f"results/training_budget/eval_dqn_{seed}_40000_mean.json"  # Include (ROOT divided by f'results/training budget/eval dqn {seed} 40000 mean.json') when block equals 'main', otherwise (STAGE divided by f'eval dqn fixed mean seed{seed} transfer.json') in the surrounding operation.
                if block == "main"  # Include block equals 'main' in the surrounding operation.
                else STAGE / f"eval_dqn_fixed_mean_seed{seed}_transfer.json"  # Include (STAGE divided by f'eval dqn fixed mean seed{seed} transfer.json') in the surrounding operation.
            )  # Finish assembling the grouped values above.
        elif group == "mixed_dqn":  # Use this branch when group equals 'mixed_dqn'.
            p = STAGE / f"eval_dqn_mixed_mean_seed{seed}_{block}.json"  # Set p to (STAGE divided by f'eval dqn mixed mean seed{seed} {block}.json').
        elif group in ["mixed_qr_mean", "mixed_qr_tail"]:  # Use this branch when group belongs to 'mixed_qr_mean', 'mixed_qr_tail'.
            policy = "mean" if group == "mixed_qr_mean" else "tail25"  # Set policy to 'mean' when group equals 'mixed_qr_mean', otherwise 'tail25'.
            p = (  # Set p to (ROOT divided by f'results/density coverage/eval mixed {policy} seed{seed} {block}.json').
                ROOT  # Include (ROOT divided by f'results/density coverage/eval mixed {policy} seed{seed} {block}.json') in the surrounding operation.
                / f"results/density_coverage/eval_mixed_{policy}_seed{seed}_{block}.json"  # Include f'results/density coverage/eval mixed {policy} seed{seed} {block}.json' in the surrounding operation.
            )  # Finish assembling the grouped values above.
        else:  # Handle the alternative when the earlier condition (group belongs to 'mixed_qr_mean', 'mixed_qr_tail') is false.
            algo = "dqn" if group == "matched_dqn" else "qrdqn"  # Set algo to 'dqn' when group equals 'matched_dqn', otherwise 'qrdqn'.
            cap = selections[algo]["selected_cap_m_s"]  # Set cap to the entries of the entries of selections selected by algo selected by 'selected_cap_m_s'.
            token = f"cap{cap:g}".replace(".", "p")  # Set token to f'cap{cap:g}'.replace applied to '.', 'p'.
            p = STAGE / f"eval_{algo}_mixed_{token}_seed{seed}_{block}.json"  # Set p to (STAGE divided by f'eval {algo} mixed {token} seed{seed} {block}.json').
        paths.append(p)  # Add p to paths.
    return [read(p) for p in paths]  # Return read applied to p for each p in paths.


def report():  # Build graphs and an HTML page from saved result files.
    """Build graphs and an HTML page from saved result files.
    The arrays below hold measured results; plotting does not change the models.
    HTML strings describe page layout, while Python inserts the saved numbers.
    Keep primary comparisons separate from exploratory comparisons in the report.
    """
    viewer = ROOT / "results/report"  # Set viewer to the path to 'results/report' inside ROOT.
    backup = ROOT / "results/density_coverage-report"  # Set backup to the path to 'results/density_coverage-report' inside ROOT.
    if not backup.exists():  # Use this branch when not the result of backup.exists.
        shutil.copytree(viewer, backup)  # Shutil.copytree applied to viewer, backup.
    selections = {  # Set selections to read applied to (STAGE divided by f'{algo} governor selection.json') for each algo in 'dqn', 'qrdqn'.
        algo: read(STAGE / f"{algo}_governor_selection.json")  # Include algo, read applied to (STAGE divided by f'{algo} governor selection.json') in the surrounding operation.
        for algo in ["dqn", "qrdqn"]  # Include algo, 'dqn', 'qrdqn' in the surrounding operation.
    }  # Finish assembling the grouped values above.
    data = {}  # Set data to the named settings or measurements assembled below.
    groups = {}  # Set groups to the named settings or measurements assembled below.
    comparisons = []  # Set comparisons to .
    exposure = {}  # Set exposure to the named settings or measurements assembled below.
    mean_comparisons = []  # Set mean comparisons to .
    for block, densities in [("main", [1.0, 1.5, 2.0]), ("transfer", [1.25, 2.5])]:  # Process each block, densities from 'main', 1.0, 1.5, 2.0, 'transfer', 1.25, 2.5.
        data[block] = {g: load(g, block, selections) for g in LABELS}  # Set the entries of data selected by block to load applied to g, block, selections for each g in LABELS.
        groups[block] = {g: summarize_group(data[block][g]) for g in LABELS}  # Set the entries of groups selected by block to summarize group applied to the entries of the entries of data selected by block selected by g for each g in LABELS.
        for other in ["mixed_dqn", "mixed_qr_mean", "matched_dqn", "matched_qr"]:  # Process each other from 'mixed_dqn', 'mixed_qr_mean', 'matched_dqn', 'matched_qr'.
            for density in densities:  # Process each the traffic-density setting from densities.
                item = {"block": block, "reference": other, "density": density}  # Set item to the named settings or measurements assembled below.
                for metric in ["crashed", "mean_speed_m_s", "distance_m", "return"]:  # Compare one measured quantity at a time, keeping the scenarios paired.
                    _, matrix = paired_matrix(  # Set the unused result, matrix to paired matrix applied to the entries of the entries of data selected by block selected by 'mixed_qr_tail', the entries of the entries of data selected by block selected by other, the traffic-density setting, metric.
                        data[block]["mixed_qr_tail"],  # Include the entries of the entries of data selected by block selected by 'mixed_qr_tail' in the surrounding operation.
                        data[block][other],  # Include the entries of the entries of data selected by block selected by other in the surrounding operation.
                        density,  # Include the traffic-density setting in the surrounding operation.
                        metric,  # Include metric in the surrounding operation.
                    )  # Finish assembling paired matrix inputs.
                    item[metric] = crossed_bootstrap(matrix)  # Set the entries of item selected by metric to crossed bootstrap applied to matrix.
                comparisons.append(item)  # Add item to comparisons.
        for label, left, right in [  # Process each label, left, right from 'Mean selectors', 'mixed_qr_mean', 'mixed_dqn', 'Speed-targeted mean selectors', 'matched_qr', 'matched_dqn'.
            ("Mean selectors", "mixed_qr_mean", "mixed_dqn"),  # Include 'Mean selectors', 'mixed_qr_mean', 'mixed_dqn' in the surrounding operation.
            ("Speed-targeted mean selectors", "matched_qr", "matched_dqn"),  # Include 'Speed-targeted mean selectors', 'matched_qr', 'matched_dqn' in the surrounding operation.
        ]:  # Finish assembling the grouped values above.
            for density in densities:  # Process each the traffic-density setting from densities.
                item = {"block": block, "comparison": label, "density": density}  # Set item to the named settings or measurements assembled below.
                for metric in ["crashed", "mean_speed_m_s"]:  # Compare one measured quantity at a time, keeping the scenarios paired.
                    _, matrix = paired_matrix(  # Set the unused result, matrix to paired matrix applied to the entries of the entries of data selected by block selected by left, the entries of the entries of data selected by block selected by right, the traffic-density setting, metric.
                        data[block][left], data[block][right], density, metric  # Include the entries of the entries of data selected by block selected by left, the entries of the entries of data selected by block selected by right, the traffic-density setting, metric in the surrounding operation.
                    )  # Finish assembling paired matrix inputs.
                    item[metric] = crossed_bootstrap(matrix)  # Set the entries of item selected by metric to crossed bootstrap applied to matrix.
                mean_comparisons.append(item)  # Add item to mean comparisons.
    for algo in ["dqn", "qrdqn"]:  # Process each algo from 'dqn', 'qrdqn'.
        folder = STAGE if algo == "dqn" else ROOT / "results/density_coverage"  # Set folder to STAGE when algo equals 'dqn', otherwise the path to 'results/density_coverage' inside ROOT.
        exposure[algo] = [  # Set the entries of exposure selected by algo to the named settings or measurements assembled below for each s in range applied to 3.
            {"seed": s, **read(folder / f"{algo}_seed{s}/summary.json")}  # Record 'seed' as s.
            for s in range(3)  # Include s, range applied to 3 in the surrounding operation.
        ]  # Finish assembling the grouped values above.
        if any(sum(r["steps_by_density"].values()) != 40000 for r in exposure[algo]):  # Use this branch when any applied to sum applied to r['steps by density'].values() differs from 40000 for each r in the entries of exposure selected by algo.
            raise ValueError("Unequal interaction budgets")  # Reject this invalid case instead of continuing with misleading experiment results.
    save_json(  # Save the following results/settings as json: the path to 'summary.json' inside stage, the named settings or measurements assembled below.
        STAGE / "summary.json",  # Include the path to 'summary.json' inside STAGE in the surrounding operation.
        {  # Include the named settings or measurements assembled below in the surrounding operation.
            "design": read(STAGE / "design.json"),  # Record 'design' as read applied to the path to 'design.json' inside STAGE.
            "groups": groups,  # Record 'groups' as groups.
            "lower_quarter_minus_reference": comparisons,  # Record 'lower_quarter_minus_reference' as comparisons.
            "selections": selections,  # Record 'selections' as selections.
            "training_exposure": exposure,  # Record 'training_exposure' as exposure.
            "exploratory_mean_selector_comparisons": mean_comparisons,  # Record 'exploratory_mean_selector_comparisons' as mean comparisons.
        },  # Finish assembling the grouped values above.
    )  # Finish assembling save json inputs.
    plt.rcParams.update(  # Plt.rcparams.update applied to the named settings or measurements assembled below.
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}  # Record 'font.size' as 10.
    )  # Finish assembling plt.rcParams.update inputs.
    colors = {g: f"C{i}" for i, g in enumerate(LABELS)}  # Set colors to f'C{i}' for each i, g in enumerate applied to LABELS.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))  # Set fig, axes to plt.subplots applied to 1, 3.
    for group in LABELS:  # Process each group from LABELS.
        rows = groups["main"][group]  # Set rows to the entries of the entries of groups selected by 'main' selected by group.
        x = [r["density"] for r in rows]  # Set x to the entries of r selected by 'density' for each r in rows.
        for ax, metric, scale, title in zip(  # Process each ax, metric, scale, title from matching entries grouped across axes, 'crash_rate', 'mean_speed_m_s', 'mean_distance_m', 100, 1, 1, 'Episodes with collision (%)', 'Mean episode speed (m/s)', 'Mean distance (m)'.
            axes,  # Include axes in the surrounding operation.
            ["crash_rate", "mean_speed_m_s", "mean_distance_m"],  # Include 'crash_rate', 'mean_speed_m_s', 'mean_distance_m' in the surrounding operation.
            [100, 1, 1],  # Include 100, 1, 1 in the surrounding operation.
            [  # Include 'Episodes with collision (%)', 'Mean episode speed (m/s)', 'Mean distance (m)' in the surrounding operation.
                "Episodes with collision (%)",  # Include 'Episodes with collision (%)' in the surrounding operation.
                "Mean episode speed (m/s)",  # Include 'Mean episode speed (m/s)' in the surrounding operation.
                "Mean distance (m)",  # Include 'Mean distance (m)' in the surrounding operation.
            ],  # Finish assembling the grouped values above.
        ):  # Finish assembling zip inputs.
            ax.plot(  # Ax.plot applied to x, (the entries of r selected by metric times scale) for each r in rows, '-o'.
                x,  # Include x in the surrounding operation.
                [r[metric] * scale for r in rows],  # Include (the entries of r selected by metric times scale) for each r in rows in the surrounding operation.
                "-o",  # Include '-o' in the surrounding operation.
                color=colors[group],  # Include the entries of colors selected by group in the surrounding operation.
                label=LABELS[group],  # Include the entries of LABELS selected by group in the surrounding operation.
            )  # Finish assembling ax.plot inputs.
            ax.set_xlabel("Evaluation density")  # Ax.set xlabel applied to 'evaluation density'.
            ax.set_xticks(x)  # Ax.set xticks applied to x.
            ax.set_ylabel(title)  # Ax.set ylabel applied to title.
            ax.grid(alpha=0.2)  # The result of ax.grid.
    axes[0].set_ylim(0, 105)  # Axes[0].set ylim applied to 0, 105.
    axes[0].legend(fontsize=8)  # The result of axes[0].legend.
    fig.suptitle(  # Fig.suptitle applied to the report text.
        "40k interactions per model; identical configured traffic mixture for mixed learners"  # Include the report text in the surrounding operation.
    )  # Finish assembling fig.suptitle inputs.
    fig.tight_layout()  # The result of fig.tight layout.
    fig.savefig(viewer / "baseline_controls_performance.png", dpi=150)  # Save a figure for the local results page; this does not affect agent behavior.
    plt.close(fig)  # Plt.close applied to fig.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))  # Set fig, axes to plt.subplots applied to 1, 3.
    for ax, density in zip(axes, [1.0, 1.5, 2.0]):  # Process each ax, the traffic-density setting from matching entries grouped across axes, 1.0, 1.5, 2.0.
        for group in LABELS:  # Process each group from LABELS.
            row = next(r for r in groups["main"][group] if r["density"] == density)  # Set row to next applied to r for each r in the entries of the entries of groups selected by 'main' selected by group, keeping only cases where the entries of r selected by 'density' equals the traffic-density setting.
            for seed in row["per_seed"]:  # Process each seed from the entries of row selected by 'per_seed'.
                ax.scatter(  # Ax.scatter applied to the entries of seed selected by 'mean_speed_m_s', (100 times the entries of seed selected by 'crash_rate').
                    seed["mean_speed_m_s"],  # Include the entries of seed selected by 'mean_speed_m_s' in the surrounding operation.
                    100 * seed["crash_rate"],  # Include (100 times the entries of seed selected by 'crash_rate') in the surrounding operation.
                    color=colors[group],  # Include the entries of colors selected by group in the surrounding operation.
                    s=18,  # Include 18 in the surrounding operation.
                    alpha=0.35,  # Include 0.35 in the surrounding operation.
                )  # Finish assembling ax.scatter inputs.
            ax.scatter(  # Ax.scatter applied to the entries of row selected by 'mean_speed_m_s', (100 times the entries of row selected by 'crash_rate').
                row["mean_speed_m_s"],  # Include the entries of row selected by 'mean_speed_m_s' in the surrounding operation.
                100 * row["crash_rate"],  # Include (100 times the entries of row selected by 'crash_rate') in the surrounding operation.
                color=colors[group],  # Include the entries of colors selected by group in the surrounding operation.
                s=75,  # Include 75 in the surrounding operation.
                marker="D",  # Include 'D' in the surrounding operation.
                label=LABELS[group],  # Include the entries of LABELS selected by group in the surrounding operation.
            )  # Finish assembling ax.scatter inputs.
        ax.set_title(f"Density {density:g}")  # Ax.set title applied to f'density {density:g}'.
        ax.set_xlabel("Mean episode speed (m/s)")  # Ax.set xlabel applied to 'mean episode speed (m/s)'.
        ax.set_ylim(-3, 105)  # Ax.set ylim applied to negative 3, 105.
        ax.grid(alpha=0.2)  # The result of ax.grid.
    axes[0].set_ylabel("Episodes with collision (%)")  # Axes[0].set ylabel applied to 'episodes with collision (%)'.
    axes[0].legend(fontsize=7)  # The result of axes[0].legend.
    fig.suptitle(  # Fig.suptitle applied to the report text.
        "Speed/collision tradeoffs; diamonds = means, small dots = training seeds"  # Include the report text in the surrounding operation.
    )  # Finish assembling fig.suptitle inputs.
    fig.tight_layout()  # The result of fig.tight layout.
    fig.savefig(viewer / "baseline_controls_tradeoff.png", dpi=150)  # Save a figure for the local results page; this does not affect agent behavior.
    plt.close(fig)  # Plt.close applied to fig.
    document = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Scalar baselines and speed controls · Risk-sensitive driving</title><style>body{font:17px/1.6 system-ui;color:#18283d;background:#f4f7fb;margin:auto;max-width:1160px;padding:36px}h1{font-size:38px;line-height:1.2}section{background:white;padding:24px;margin:24px 0;border-radius:14px}img{width:100%;height:auto}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;text-align:left;border-bottom:1px solid #dde3eb}th{background:#eef3fa}a{color:#175bbc}</style></head><body><p>DEEP RL / STRONGER BASELINES</p><h1>Does cautious selection outperform a comparable scalar learner?</h1><p>DQN and QR-DQN now both train for 40,000 interactions on the same configured episode-density mixture: 1, 1.5 and 2. Rewards, hidden-layer sizes, optimizer settings and exploration schedules are held fixed. Experience remains policy-dependent; there are three training seeds per learner.</p><p><a href="../density_coverage-report/index.html">Coverage comparison, return diagnostics and replays</a> · <a href="../baseline_controls/summary.json">Numerical results</a> · <a href="../baseline_controls/design.json">Recorded design</a></p>"""  # Set document to the report text.
    headlines = []  # Set headlines to .
    for density in [1.0, 2.0]:  # Process each the traffic-density setting from 1.0, 2.0.
        contrast = next(  # Set contrast to next applied to r for each r in comparisons, keeping only cases where r['block'] equals 'main' and r['reference'] equals 'matched dqn' and r['density'] equals the traffic-density setting.
            r  # Include r in the surrounding operation.
            for r in comparisons  # Include r, comparisons in the surrounding operation.
            if r["block"] == "main"  # Include the entries of r selected by 'block' equals 'main' and the entries of r selected by 'reference' equals 'matched_dqn' and the entries of r selected by 'density' equals the traffic-density setting in the surrounding operation.
            and r["reference"] == "matched_dqn"  # Include the entries of r selected by 'reference' equals 'matched_dqn' in the surrounding operation.
            and r["density"] == density  # Include the entries of r selected by 'density' equals the traffic-density setting in the surrounding operation.
        )  # Finish assembling next inputs.
        diff = contrast["crashed"]  # Set diff to the entries of contrast selected by 'crashed'.
        interval = [round(100 * x, 1) for x in diff["exploratory_95_percent_interval"]]  # Set interval to round applied to (100 times x), 1 for each x in the entries of diff selected by 'exploratory_95_percent_interval'.
        headlines.append(  # Add f"at density {density:g}, cautious qr minus speed-targeted dqn is {100 * diff['mean difference']:+.1f} collision percentage points (exploratory interval {interval}), with a speed d to headlines.
            f"At density {density:g}, cautious QR minus speed-targeted DQN is {100*diff['mean_difference']:+.1f} collision percentage points (exploratory interval {interval}), with a speed difference of {contrast['mean_speed_m_s']['mean_difference']:+.2f} m/s."  # Include f"At density {density:g}, cautious QR minus speed-targeted DQN is {100 * diff['mean difference']:+.1f} collision percentage points (exploratory interval {interval}), with a speed d in the surrounding operation.
        )  # Finish assembling headlines.append inputs.
    document += (  # Increase document using (('<section><h2>Main control comparison</h2>' plus ''.join applied to '<p>' + line + '</p>' for each line in headlines) plus the report text).
        "<section><h2>Main control comparison</h2>"  # Include (('<section><h2>Main control comparison</h2>' plus ''.join applied to '<p>' + line + '</p>' for each line in headlines) plus the report text) in the surrounding operation.
        + "".join("<p>" + line + "</p>" for line in headlines)  # Include ''.join applied to (('<p>' plus line) plus '</p>') for each line in headlines in the surrounding operation.
        + "<p>The full seed results and residual speed differences below are needed to interpret these comparisons.</p></section>"  # Include the report text in the surrounding operation.
    )  # Finish assembling the grouped values above.
    document += '<section><h2>Equal-coverage comparison</h2><img src="baseline_controls_performance.png"><p>Mixed learners share the configured traffic distribution and step budget. The fixed-density DQN is retained to show its coverage effect. The cautious QR selector uses the lowest quarter of 32 outputs, without additional risk-level tuning.</p></section>'  # Increase document using the report text.
    document += (  # Increase document using ((the report text plus table applied to 'Learner', 'Selected cap m/s', 'Target speed m/s', 'Seed-0 residual m/s', [a, r['selected cap m s'], f"{r['target speed m s']:.3f}", f"{r['seed0 speed difference m s']:+.3f}"] for each (a, r) in selections.items()) plus '<img src="baseline_controls_tradeoff.png"></section>').
        "<section><h2>Seed-0 speed-targeted controls</h2><p>For each learner, select a cap using only seed-0 normal-traffic speed closeness to the cautious QR policy; then freeze it before evaluating seeds 1 and 2. The development threshold grid is 21–25 m/s. This does not match the full speed distribution or the action interventions.</p>"  # Include ((the report text plus table applied to 'Learner', 'Selected cap m/s', 'Target speed m/s', 'Seed-0 residual m/s', [a, r['selected cap m s'], f"{r['target speed m s']:.3f}", f"{r['seed0 speed difference m s']:+.3f}"] for each (a, r) in selections.items()) plus '<img src="baseline_controls_tradeoff.png"></section>') in the surrounding operation.
        + table(  # Include table applied to 'Learner', 'Selected cap m/s', 'Target speed m/s', 'Seed-0 residual m/s', a, the entries of r selected by 'selected_cap_m_s', f"{r['target speed m s']:.3f}", f"{r['seed0 speed difference m s']:+.3f}" for each a, r in the result of selections.items in the surrounding operation.
            ["Learner", "Selected cap m/s", "Target speed m/s", "Seed-0 residual m/s"],  # Include 'Learner', 'Selected cap m/s', 'Target speed m/s', 'Seed-0 residual m/s' in the surrounding operation.
            [  # Include a, the entries of r selected by 'selected_cap_m_s', f"{r['target speed m s']:.3f}", f"{r['seed0 speed difference m s']:+.3f}" for each a, r in the result of selections.items in the surrounding operation.
                [  # Include a, the entries of r selected by 'selected_cap_m_s', f"{r['target speed m s']:.3f}", f"{r['seed0 speed difference m s']:+.3f}" in the surrounding operation.
                    a,  # Include a in the surrounding operation.
                    r["selected_cap_m_s"],  # Include the entries of r selected by 'selected_cap_m_s' in the surrounding operation.
                    f"{r['target_speed_m_s']:.3f}",  # Include f"{r['target speed m s']:.3f}" in the surrounding operation.
                    f"{r['seed0_speed_difference_m_s']:+.3f}",  # Include f"{r['seed0 speed difference m s']:+.3f}" in the surrounding operation.
                ]  # Finish assembling the grouped values above.
                for a, r in selections.items()  # Include a, r, the result of selections.items in the surrounding operation.
            ],  # Finish assembling table inputs.
        )  # Finish assembling table inputs.
        + '<img src="baseline_controls_tradeoff.png"></section>'  # Include '<img src="baseline_controls_tradeoff.png"></section>' in the surrounding operation.
    )  # Finish assembling the grouped values above.
    for block, title in [  # Process each block, title from 'main', 'Primary traffic scenarios', 'transfer', 'Intermediate and beyond-training densities'.
        ("main", "Primary traffic scenarios"),  # Include 'main', 'Primary traffic scenarios' in the surrounding operation.
        ("transfer", "Intermediate and beyond-training densities"),  # Include 'transfer', 'Intermediate and beyond-training densities' in the surrounding operation.
    ]:  # Finish assembling the grouped values above.
        document += (  # Increase document using (((('<section><h2>' + title + '</h2><p>' plus 'Scenarios 10000–10029; densities 1, 1.5 and 2.' if block == 'main' else 'Separate development scenarios 16000–16029; density 1.25) plus '</p>') plus table applied to the listed settings or measurements in their recorded order, [LABELS[g], r['density'], f"{100 * r['crash rate']:.1f}%", f"{r['mean speed m s']:.2f}", f"{r['mean distance m']:.0f}", f"{r['mean for each g in LABELS) plus '</section>').
            "<section><h2>"  # Include (((('<section><h2>' + title + '</h2><p>' plus 'Scenarios 10000–10029; densities 1, 1.5 and 2.' if block == 'main' else 'Separate development scenarios 16000–16029; density 1.25) plus '</p>') plus table applied to the listed settings or measurements in their recorded order, [LABELS[g], r['density'], f"{100 * r['crash rate']:.1f}%", f"{r['mean speed m s']:.2f}", f"{r['mean distance m']:.0f}", f"{r['mean for each g in LABELS) plus '</section>') in the surrounding operation.
            + title  # Include title in the surrounding operation.
            + "</h2><p>"  # Include '</h2><p>' in the surrounding operation.
            + (  # Continue the calculation or input list started above.
                "Scenarios 10000–10029; densities 1, 1.5 and 2."  # Include 'Scenarios 10000–10029; densities 1, 1.5 and 2.' when block equals 'main', otherwise the report text in the surrounding operation.
                if block == "main"  # Include block equals 'main' in the surrounding operation.
                else "Separate development scenarios 16000–16029; density 1.25 is intermediate and density 2.5 is beyond both training ranges."  # Include the report text in the surrounding operation.
            )  # Finish assembling the grouped values above.
            + "</p>"  # Include '</p>' in the surrounding operation.
            + table(  # Include table applied to the listed settings or measurements in their recorded order, the listed settings or measurements in their recorded order for each g in LABELS in the surrounding operation.
                [  # Include the listed settings or measurements in their recorded order in the surrounding operation.
                    "Method",  # Include 'Method' in the surrounding operation.
                    "Density",  # Include 'Density' in the surrounding operation.
                    "Collisions",  # Include 'Collisions' in the surrounding operation.
                    "Speed m/s",  # Include 'Speed m/s' in the surrounding operation.
                    "Distance m",  # Include 'Distance m' in the surrounding operation.
                    "Mean return",  # Include 'Mean return' in the surrounding operation.
                    "Worst 10% return",  # Include 'Worst 10% return' in the surrounding operation.
                ],  # Finish assembling the grouped values above.
                [  # Include the listed settings or measurements in their recorded order for each g in LABELS in the surrounding operation.
                    [  # Include the listed settings or measurements in their recorded order in the surrounding operation.
                        LABELS[g],  # Include the entries of LABELS selected by g in the surrounding operation.
                        r["density"],  # Include the entries of r selected by 'density' in the surrounding operation.
                        f"{100*r['crash_rate']:.1f}%",  # Include f"{100 * r['crash rate']:.1f}%" in the surrounding operation.
                        f"{r['mean_speed_m_s']:.2f}",  # Include f"{r['mean speed m s']:.2f}" in the surrounding operation.
                        f"{r['mean_distance_m']:.0f}",  # Include f"{r['mean distance m']:.0f}" in the surrounding operation.
                        f"{r['mean_return']:.2f}",  # Include f"{r['mean return']:.2f}" in the surrounding operation.
                        f"{r['worst_10_percent_mean_return']:.2f}",  # Include f"{r['worst 10 percent mean return']:.2f}" in the surrounding operation.
                    ]  # Finish assembling the grouped values above.
                    for g in LABELS  # Include g, LABELS in the surrounding operation.
                    for r in groups[block][g]  # Include r, the entries of the entries of groups selected by block selected by g in the surrounding operation.
                ],  # Finish assembling table inputs.
            )  # Finish assembling table inputs.
            + "</section>"  # Include '</section>' in the surrounding operation.
        )  # Finish assembling the grouped values above.
    document += (  # Increase document using ((the report text plus table applied to the listed settings or measurements in their recorded order, [r['block'], LABELS[r['reference']], r['density'], f"{100 * r['crashed']['mean difference']:+.1f}", str([round(100 * x, 1) for x i for each r in comparisons) plus '</section>').
        "<section><h2>Cautious QR minus each reference</h2><p>Negative collision differences favor cautious QR. Exploratory 95% intervals use 5,000 crossed bootstrap replicates over model seed IDs and shared scenarios, preserving policy pairing. Speed-control intervals condition on the selected cap and do not adjust for its selection. Three seeds and repeated development use limit inference.</p>"  # Include ((the report text plus table applied to the listed settings or measurements in their recorded order, [r['block'], LABELS[r['reference']], r['density'], f"{100 * r['crashed']['mean difference']:+.1f}", str([round(100 * x, 1) for x i for each r in comparisons) plus '</section>') in the surrounding operation.
        + table(  # Include table applied to the listed settings or measurements in their recorded order, the listed settings or measurements in their recorded order for each r in comparisons in the surrounding operation.
            [  # Include the listed settings or measurements in their recorded order in the surrounding operation.
                "Block",  # Include 'Block' in the surrounding operation.
                "Reference",  # Include 'Reference' in the surrounding operation.
                "Density",  # Include 'Density' in the surrounding operation.
                "Collision change pp",  # Include 'Collision change pp' in the surrounding operation.
                "Exploratory interval pp",  # Include 'Exploratory interval pp' in the surrounding operation.
                "Speed change m/s",  # Include 'Speed change m/s' in the surrounding operation.
                "Per-seed collision change pp",  # Include 'Per-seed collision change pp' in the surrounding operation.
            ],  # Finish assembling the grouped values above.
            [  # Include the listed settings or measurements in their recorded order for each r in comparisons in the surrounding operation.
                [  # Include the listed settings or measurements in their recorded order in the surrounding operation.
                    r["block"],  # Include the entries of r selected by 'block' in the surrounding operation.
                    LABELS[r["reference"]],  # Include the entries of LABELS selected by r['reference'] in the surrounding operation.
                    r["density"],  # Include the entries of r selected by 'density' in the surrounding operation.
                    f"{100*r['crashed']['mean_difference']:+.1f}",  # Include f"{100 * r['crashed']['mean difference']:+.1f}" in the surrounding operation.
                    str(  # Include round applied to (100 times x), 1 for each x in the entries of the entries of r selected by 'crashed' selected by 'exploratory_95_percent_interval' as a text value in the surrounding operation.
                        [  # Include round applied to (100 times x), 1 for each x in the entries of the entries of r selected by 'crashed' selected by 'exploratory_95_percent_interval' in the surrounding operation.
                            round(100 * x, 1)  # Include round applied to (100 times x), 1 in the surrounding operation.
                            for x in r["crashed"]["exploratory_95_percent_interval"]  # Include x, the entries of the entries of r selected by 'crashed' selected by 'exploratory_95_percent_interval' in the surrounding operation.
                        ]  # Finish assembling str inputs.
                    ),  # Finish assembling str inputs.
                    f"{r['mean_speed_m_s']['mean_difference']:+.2f}",  # Include f"{r['mean speed m s']['mean difference']:+.2f}" in the surrounding operation.
                    str(  # Include round applied to (100 times x), 1 for each x in the entries of the entries of r selected by 'crashed' selected by 'per_training_seed_differences' as a text value in the surrounding operation.
                        [  # Include round applied to (100 times x), 1 for each x in the entries of the entries of r selected by 'crashed' selected by 'per_training_seed_differences' in the surrounding operation.
                            round(100 * x, 1)  # Include round applied to (100 times x), 1 in the surrounding operation.
                            for x in r["crashed"]["per_training_seed_differences"]  # Include x, the entries of the entries of r selected by 'crashed' selected by 'per_training_seed_differences' in the surrounding operation.
                        ]  # Finish assembling str inputs.
                    ),  # Finish assembling str inputs.
                ]  # Finish assembling the grouped values above.
                for r in comparisons  # Include r, comparisons in the surrounding operation.
            ],  # Finish assembling table inputs.
        )  # Finish assembling table inputs.
        + "</section>"  # Include '</section>' in the surrounding operation.
    )  # Finish assembling the grouped values above.
    residual = []  # Set residual to .
    for reference in ["matched_dqn", "matched_qr"]:  # Process each reference from 'matched_dqn', 'matched_qr'.
        target = next(r for r in groups["main"]["mixed_qr_tail"] if r["density"] == 1.0)  # Set target to next applied to r for each r in the entries of the entries of groups selected by 'main' selected by 'mixed_qr_tail', keeping only cases where the entries of r selected by 'density' equals 1.0.
        control = next(r for r in groups["main"][reference] if r["density"] == 1.0)  # Set control to next applied to r for each r in the entries of the entries of groups selected by 'main' selected by reference, keeping only cases where the entries of r selected by 'density' equals 1.0.
        for a, b in zip(target["per_seed"], control["per_seed"]):  # Process each a, b from matching entries grouped across the entries of target selected by 'per_seed', the entries of control selected by 'per_seed'.
            residual.append(  # Add the entries of labels selected by reference, the entries of a selected by 'training_seed', f"{a['mean speed m s'] - b['mean speed m s']:+.3f}", f"{100 * (a['crash rate'] - b['crash rate']):+.1f}" to residual.
                [  # Include the entries of LABELS selected by reference, the entries of a selected by 'training_seed', f"{a['mean speed m s'] - b['mean speed m s']:+.3f}", f"{100 * (a['crash rate'] - b['crash rate']):+.1f}" in the surrounding operation.
                    LABELS[reference],  # Include the entries of LABELS selected by reference in the surrounding operation.
                    a["training_seed"],  # Include the entries of a selected by 'training_seed' in the surrounding operation.
                    f"{a['mean_speed_m_s']-b['mean_speed_m_s']:+.3f}",  # Include f"{a['mean speed m s'] - b['mean speed m s']:+.3f}" in the surrounding operation.
                    f"{100*(a['crash_rate']-b['crash_rate']):+.1f}",  # Include f"{100 * (a['crash rate'] - b['crash rate']):+.1f}" in the surrounding operation.
                ]  # Finish assembling the grouped values above.
            )  # Finish assembling residual.append inputs.
    document += (  # Increase document using ((the report text plus table applied to the listed settings or measurements in their recorded order, [r['block'], r['comparison'], r['density'], f"{100 * r['crashed']['mean difference']:+.1f}", str([round(100 * x, 1) for x in r['cr for each r in mean comparisons) plus '</section>').
        "<section><h2>Exploratory comparison of the mean selectors</h2><p>QR mean minus scalar DQN, using the same training mixture. These secondary summaries were added after inspecting the primary results; no training or selection settings were changed. Distributional learning also changes the output head, loss and parameter count, so this does not isolate a single architectural cause.</p>"  # Include ((the report text plus table applied to the listed settings or measurements in their recorded order, [r['block'], r['comparison'], r['density'], f"{100 * r['crashed']['mean difference']:+.1f}", str([round(100 * x, 1) for x in r['cr for each r in mean comparisons) plus '</section>') in the surrounding operation.
        + table(  # Include table applied to the listed settings or measurements in their recorded order, the listed settings or measurements in their recorded order for each r in mean comparisons in the surrounding operation.
            [  # Include the listed settings or measurements in their recorded order in the surrounding operation.
                "Block",  # Include 'Block' in the surrounding operation.
                "Comparison",  # Include 'Comparison' in the surrounding operation.
                "Density",  # Include 'Density' in the surrounding operation.
                "Collision change pp",  # Include 'Collision change pp' in the surrounding operation.
                "Exploratory interval pp",  # Include 'Exploratory interval pp' in the surrounding operation.
                "Speed change m/s",  # Include 'Speed change m/s' in the surrounding operation.
            ],  # Finish assembling the grouped values above.
            [  # Include the listed settings or measurements in their recorded order for each r in mean comparisons in the surrounding operation.
                [  # Include the listed settings or measurements in their recorded order in the surrounding operation.
                    r["block"],  # Include the entries of r selected by 'block' in the surrounding operation.
                    r["comparison"],  # Include the entries of r selected by 'comparison' in the surrounding operation.
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
                for r in mean_comparisons  # Include r, mean comparisons in the surrounding operation.
            ],  # Finish assembling table inputs.
        )  # Finish assembling table inputs.
        + "</section>"  # Include '</section>' in the surrounding operation.
    )  # Finish assembling the grouped values above.
    document += (  # Increase document using ((the report text plus table applied to 'Reference', 'Training seed', 'Speed difference m/s', 'Collision difference pp', residual) plus '</section>').
        "<section><h2>Normal-traffic residuals by seed</h2><p>Values are cautious minus control. Aggregate matching can hide substantial differences by model.</p>"  # Include ((the report text plus table applied to 'Reference', 'Training seed', 'Speed difference m/s', 'Collision difference pp', residual) plus '</section>') in the surrounding operation.
        + table(  # Include table applied to 'Reference', 'Training seed', 'Speed difference m/s', 'Collision difference pp', residual in the surrounding operation.
            [  # Include 'Reference', 'Training seed', 'Speed difference m/s', 'Collision difference pp' in the surrounding operation.
                "Reference",  # Include 'Reference' in the surrounding operation.
                "Training seed",  # Include 'Training seed' in the surrounding operation.
                "Speed difference m/s",  # Include 'Speed difference m/s' in the surrounding operation.
                "Collision difference pp",  # Include 'Collision difference pp' in the surrounding operation.
            ],  # Finish assembling the grouped values above.
            residual,  # Include residual in the surrounding operation.
        )  # Finish assembling table inputs.
        + "</section>"  # Include '</section>' in the surrounding operation.
    )  # Finish assembling the grouped values above.
    document += (  # Increase document using ((the report text plus table applied to the listed settings or measurements in their recorded order, [algo, r['seed'], r['steps by density']['1.0'], r['steps by density']['1.5'], r['steps by density']['2.0'], r['episodes']] for each (algo, rows) in exposure.items()) plus '</section>').
        "<section><h2>Actual mixed-training exposure</h2><p>Density is sampled uniformly by episode, not by transition. Different policies generate different episode lengths and exposure shares despite equal total interaction budgets.</p>"  # Include ((the report text plus table applied to the listed settings or measurements in their recorded order, [algo, r['seed'], r['steps by density']['1.0'], r['steps by density']['1.5'], r['steps by density']['2.0'], r['episodes']] for each (algo, rows) in exposure.items()) plus '</section>') in the surrounding operation.
        + table(  # Include table applied to the listed settings or measurements in their recorded order, the listed settings or measurements in their recorded order for each algo, rows in the result of exposure.items in the surrounding operation.
            [  # Include the listed settings or measurements in their recorded order in the surrounding operation.
                "Learner",  # Include 'Learner' in the surrounding operation.
                "Seed",  # Include 'Seed' in the surrounding operation.
                "Density-1 steps",  # Include 'Density-1 steps' in the surrounding operation.
                "Density-1.5 steps",  # Include 'Density-1.5 steps' in the surrounding operation.
                "Density-2 steps",  # Include 'Density-2 steps' in the surrounding operation.
                "Completed episodes",  # Include 'Completed episodes' in the surrounding operation.
            ],  # Finish assembling the grouped values above.
            [  # Include the listed settings or measurements in their recorded order for each algo, rows in the result of exposure.items in the surrounding operation.
                [  # Include the listed settings or measurements in their recorded order in the surrounding operation.
                    algo,  # Include algo in the surrounding operation.
                    r["seed"],  # Include the entries of r selected by 'seed' in the surrounding operation.
                    r["steps_by_density"]["1.0"],  # Include the entries of the entries of r selected by 'steps_by_density' selected by '1.0' in the surrounding operation.
                    r["steps_by_density"]["1.5"],  # Include the entries of the entries of r selected by 'steps_by_density' selected by '1.5' in the surrounding operation.
                    r["steps_by_density"]["2.0"],  # Include the entries of the entries of r selected by 'steps_by_density' selected by '2.0' in the surrounding operation.
                    r["episodes"],  # Include the entries of r selected by 'episodes' in the surrounding operation.
                ]  # Finish assembling the grouped values above.
                for algo, rows in exposure.items()  # Include algo, rows, the result of exposure.items in the surrounding operation.
                for r in rows  # Include r, rows in the surrounding operation.
            ],  # Finish assembling table inputs.
        )  # Finish assembling table inputs.
        + "</section>"  # Include '</section>' in the surrounding operation.
    )  # Finish assembling the grouped values above.
    document += "<section><h2>Reproduce and interpret</h2><p><code>python run_baseline_controls.py --jobs 3</code> validates and reuses completed runs. <code>--stage evaluate</code> skips training; <code>--stage report</code> rebuilds this viewer. <code>python -m pytest -q</code> checks learning, sampling and speed-selection rules.</p><p>All scenarios remain development data; confirmation seeds 20000 onward are untouched. Reward and alpha are not retuned. A speed governor is a decision rule rather than a physical speed guarantee. Partial observations, simplified traffic and forecast error limit conclusions. Neither lower-tail selection nor favorable simulator results guarantee real-road safety.</p></section></body></html>"  # Increase document using the report text.
    (viewer / "index.html").write_text(document)  # (viewer / 'index.html').write text applied to document.
    print("Scalar baseline and speed-control report saved", flush=True)  # Print applied to 'scalar baseline and speed-control report saved'.


if __name__ == "__main__":  # Run this workflow when launched from Terminal; importing its helpers must not launch experiments.
    report()  # The result of report.
