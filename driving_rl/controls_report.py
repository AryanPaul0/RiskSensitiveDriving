"""Create the controls results page.

Read saved numbers, calculate clearly labeled summaries, draw figures, and write
an HTML page. This file presents evidence; it does not train or select models.
Long HTML strings are page templates. The reinforcement-learning calculations
live in agents.py and experiment.py, not inside the page layout.

Original implementation notes:
Publication-style figures plus a portable viewer for the control experiments.
"""

import html  # Make html available for this file's calculations and workflow.
import json  # Make json available for this file's calculations and workflow.
from pathlib import Path  # Make Path available for this file's calculations and workflow.

import numpy as np  # Make numpy available for this file's calculations and workflow.

from .diagnostics import reward_example  # Make reward_example available for this file's calculations and workflow.

COLORS = {  # Set COLORS to the named settings or measurements assembled below.
    "DQN": "#4889c9",  # Record 'DQN' as '#4889c9'.
    "QR-DQN mean": "#9563c2",  # Record 'QR-DQN mean' as '#9563c2'.
    "QR-DQN lower 25%": "#159c87",  # Record 'QR-DQN lower 25%' as '#159c87'.
    "Reward-tuned DQN": "#df9144",  # Record 'Reward-tuned DQN' as '#df9144'.
    "DQN + speed governor": "#d7677e",  # Record 'DQN + speed governor' as '#d7677e'.
}  # Finish assembling the grouped values above.


def create_report(root):  # Build graphs and an HTML page from saved result files.
    """Build graphs and an HTML page from saved result files.
    The arrays below hold measured results; plotting does not change the models.
    HTML strings describe page layout, while Python inserts the saved numbers.
    Keep primary comparisons separate from exploratory comparisons in the report.
    """
    import matplotlib  # Make matplotlib available for this file's calculations and workflow.

    matplotlib.use("Agg")  # Matplotlib.use applied to 'agg'.
    import matplotlib.pyplot as plt  # Make matplotlib.pyplot available for this file's calculations and workflow.

    root = Path(root)  # Set root to Path applied to root.
    data = json.loads((root / "controls/summary.json").read_text())  # Set data to json.loads applied to the result of (root / 'controls/summary.json').read text.
    groups = data["groups"]  # Set groups to the entries of data selected by 'groups'.
    output = root / "report"  # Set output to the path to 'report' inside root.
    output.mkdir(exist_ok=True)  # The result of output.mkdir.
    plt.rcParams.update(  # Plt.rcparams.update applied to the named settings or measurements assembled below.
        {  # Include the named settings or measurements assembled below in the surrounding operation.
            "figure.facecolor": "#f5f7fb",  # Record 'figure.facecolor' as '#f5f7fb'.
            "axes.facecolor": "white",  # Record 'axes.facecolor' as 'white'.
            "axes.spines.top": False,  # Record 'axes.spines.top' as False.
            "axes.spines.right": False,  # Record 'axes.spines.right' as False.
            "font.size": 11,  # Record 'font.size' as 11.
            "axes.titlesize": 13,  # Record 'axes.titlesize' as 13.
        }  # Finish assembling the grouped values above.
    )  # Finish assembling plt.rcParams.update inputs.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)  # Set fig, axes to plt.subplots applied to 1, 2.
    for label, rows in groups.items():  # Process each label, rows from the result of groups.items.
        densities = [row["density"] for row in rows]  # Set densities to the entries of row selected by 'density' for each row in rows.
        for axis, metric, multiplier in [  # Process each axis, metric, multiplier from the entries of axes selected by 0, 'crash_rate', 100, the entries of axes selected by 1, 'mean_speed_m_s', 1.
            (axes[0], "crash_rate", 100),  # Include the entries of axes selected by 0, 'crash_rate', 100 in the surrounding operation.
            (axes[1], "mean_speed_m_s", 1),  # Include the entries of axes selected by 1, 'mean_speed_m_s', 1 in the surrounding operation.
        ]:  # Finish assembling the grouped values above.
            means = [row[metric] * multiplier for row in rows]  # Set means to (the entries of row selected by metric times multiplier) for each row in rows.
            lows = [row[metric + "_seed_range"][0] * multiplier for row in rows]  # Set lows to (the entries of the entries of row selected by metric + '_seed_range' selected by 0 times multiplier) for each row in rows.
            highs = [row[metric + "_seed_range"][1] * multiplier for row in rows]  # Set highs to (the entries of the entries of row selected by metric + '_seed_range' selected by 1 times multiplier) for each row in rows.
            axis.plot(densities, means, "o-", color=COLORS[label], label=label)  # Axis.plot applied to densities, means, 'o-'.
            axis.fill_between(densities, lows, highs, color=COLORS[label], alpha=0.10)  # Axis.fill between applied to densities, lows, highs.
            for row in rows:  # Process each row from rows.
                axis.scatter(  # Axis.scatter applied to (the entries of row selected by 'density' times the number of entries in the entries of row selected by 'per_seed'), (the entries of item selected by metric times multiplier) for each item in the entries of row selected by 'per_seed'.
                    [row["density"]] * len(row["per_seed"]),  # Include (the entries of row selected by 'density' times the number of entries in the entries of row selected by 'per_seed') in the surrounding operation.
                    [item[metric] * multiplier for item in row["per_seed"]],  # Include (the entries of item selected by metric times multiplier) for each item in the entries of row selected by 'per_seed' in the surrounding operation.
                    s=20,  # Include 20 in the surrounding operation.
                    alpha=0.5,  # Include 0.5 in the surrounding operation.
                    color=COLORS[label],  # Include the entries of COLORS selected by label in the surrounding operation.
                )  # Finish assembling axis.scatter inputs.
    axes[0].set(  # The result of axes[0].set.
        title="Collision rate across traffic densities",  # Include 'Collision rate across traffic densities' in the surrounding operation.
        ylabel="Episodes ending in collision (%)",  # Include 'Episodes ending in collision (%)' in the surrounding operation.
        ylim=(-3, 103),  # Include negative 3, 103 in the surrounding operation.
    )  # Finish assembling axes[0].set inputs.
    axes[1].set(  # The result of axes[1].set.
        title="Speed across traffic densities", ylabel="Episode-averaged speed (m/s)"  # Include 'Speed across traffic densities', 'Episode-averaged speed (m/s)' in the surrounding operation.
    )  # Finish assembling axes[1].set inputs.
    for axis in axes:  # Process each axis from axes.
        axis.set_xlabel("Traffic density multiplier")  # Axis.set xlabel applied to 'traffic density multiplier'.
        axis.grid(alpha=0.15)  # The result of axis.grid.
        axis.legend(fontsize=8)  # The result of axis.legend.
    fig.savefig(output / "controls_replication.png", dpi=180)  # Save a figure for the local results page; this does not affect agent behavior.
    plt.close(fig)  # Plt.close applied to fig.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)  # Set fig, axes to plt.subplots applied to 1, 3.
    for axis, density in zip(axes, [1.0, 1.5, 2.0]):  # Process each axis, the traffic-density setting from matching entries grouped across axes, 1.0, 1.5, 2.0.
        for label, rows in groups.items():  # Process each label, rows from the result of groups.items.
            row = next(item for item in rows if item["density"] == density)  # Set row to next applied to item for each item in rows, keeping only cases where the entries of item selected by 'density' equals the traffic-density setting.
            axis.scatter(  # Axis.scatter applied to the entries of item selected by 'mean_speed_m_s' for each item in the entries of row selected by 'per_seed', (the entries of item selected by 'crash_rate' times 100) for each item in the entries of row selected by 'per_seed'.
                [item["mean_speed_m_s"] for item in row["per_seed"]],  # Include the entries of item selected by 'mean_speed_m_s' for each item in the entries of row selected by 'per_seed' in the surrounding operation.
                [item["crash_rate"] * 100 for item in row["per_seed"]],  # Include (the entries of item selected by 'crash_rate' times 100) for each item in the entries of row selected by 'per_seed' in the surrounding operation.
                color=COLORS[label],  # Include the entries of COLORS selected by label in the surrounding operation.
                s=32,  # Include 32 in the surrounding operation.
                alpha=0.5,  # Include 0.5 in the surrounding operation.
            )  # Finish assembling axis.scatter inputs.
            axis.scatter(  # Axis.scatter applied to the entries of row selected by 'mean_speed_m_s', (the entries of row selected by 'crash_rate' times 100).
                row["mean_speed_m_s"],  # Include the entries of row selected by 'mean_speed_m_s' in the surrounding operation.
                row["crash_rate"] * 100,  # Include (the entries of row selected by 'crash_rate' times 100) in the surrounding operation.
                color=COLORS[label],  # Include the entries of COLORS selected by label in the surrounding operation.
                s=100,  # Include 100 in the surrounding operation.
                marker="D",  # Include 'D' in the surrounding operation.
                label=label,  # Include label in the surrounding operation.
                edgecolors="white",  # Include 'white' in the surrounding operation.
                linewidths=1,  # Include 1 in the surrounding operation.
            )  # Finish assembling axis.scatter inputs.
        axis.set(  # The result of axis.set.
            title=f"Density {density:.1f}", xlabel="Mean speed (m/s)", ylim=(-3, 103)  # Include f'Density {density:.1f}', 'Mean speed (m/s)', negative 3, 103 in the surrounding operation.
        )  # Finish assembling axis.set inputs.
        axis.grid(alpha=0.15)  # The result of axis.grid.
    axes[0].set_ylabel("Collision rate (%)")  # Axes[0].set ylabel applied to 'collision rate (%)'.
    axes[-1].legend(fontsize=7, loc="lower right")  # The result of axes[-1].legend.
    fig.savefig(output / "controls_tradeoff.png", dpi=180)  # Save a figure for the local results page; this does not affect agent behavior.
    plt.close(fig)  # Plt.close applied to fig.
    contrasts = sorted(  # Set contrasts to sorted applied to the entries of row selected by 'left', the entries of row selected by 'right' for each row in the entries of data selected by 'paired_comparisons'.
        {(row["left"], row["right"]) for row in data["paired_comparisons"]}  # Include the entries of row selected by 'left', the entries of row selected by 'right' for each row in the entries of data selected by 'paired_comparisons' in the surrounding operation.
    )  # Finish assembling sorted inputs.
    fig, axis = plt.subplots(figsize=(10, 4.2), constrained_layout=True)  # Set fig, axis to the result of plt.subplots.
    for index, (left, right) in enumerate(contrasts):  # Process each index, left, right from enumerate applied to contrasts.
        selected = [  # Set the selected next-action return predictions to row for each row in the entries of data selected by 'paired_comparisons', keeping only cases where the entries of row selected by 'left' equals left and the entries of row selected by 'right' equals right.
            row  # Include row in the surrounding operation.
            for row in data["paired_comparisons"]  # Include row, the entries of data selected by 'paired_comparisons' in the surrounding operation.
            if row["left"] == left and row["right"] == right  # Include the entries of row selected by 'left' equals left and the entries of row selected by 'right' equals right in the surrounding operation.
        ]  # Finish assembling the grouped values above.
        x = (  # Set x to (the entries of row selected by 'density' for each row in the selected next-action return predictions as a numerical array/tensor for batch calculations plus ((index minus (len(contrasts) - 1 divided by 2)) times 0.025)).
            np.array([row["density"] for row in selected])  # Include (the entries of row selected by 'density' for each row in the selected next-action return predictions as a numerical array/tensor for batch calculations plus ((index minus (len(contrasts) - 1 divided by 2)) times 0.025)) in the surrounding operation.
            + (index - (len(contrasts) - 1) / 2) * 0.025  # Include ((index minus ((len(contrasts) minus 1) divided by 2)) times 0.025) in the surrounding operation.
        )  # Finish assembling the grouped values above.
        y = np.array([row["crashed"]["mean_difference"] * 100 for row in selected])  # Set y to (the entries of row['crashed'] selected by 'mean_difference' times 100) for each row in the selected next-action return predictions as a numerical array/tensor for batch calculations.
        intervals = (  # Set intervals to (the entries of row['crashed'] selected by 'exploratory_95_percent_interval' for each row in the selected next-action return predictions as a numerical array/tensor for batch calculations times 100).
            np.array(  # Include (the entries of row['crashed'] selected by 'exploratory_95_percent_interval' for each row in the selected next-action return predictions as a numerical array/tensor for batch calculations times 100) in the surrounding operation.
                [row["crashed"]["exploratory_95_percent_interval"] for row in selected]  # Include the entries of the entries of row selected by 'crashed' selected by 'exploratory_95_percent_interval' for each row in the selected next-action return predictions in the surrounding operation.
            )  # Finish assembling np.array inputs.
            * 100  # Include 100 in the surrounding operation.
        )  # Finish assembling the grouped values above.
        axis.errorbar(  # Axis.errorbar applied to x, y.
            x,  # Include x in the surrounding operation.
            y,  # Include y in the surrounding operation.
            yerr=np.maximum(0, np.stack([y - intervals[:, 0], intervals[:, 1] - y])),  # Include the elementwise larger values from 0, np.stack applied to (y minus intervals[:, 0]), (intervals[:, 1] minus y) in the surrounding operation.
            fmt="o-",  # Include 'o-' in the surrounding operation.
            capsize=5,  # Include 5 in the surrounding operation.
            label=f"Lower 25% minus {right}",  # Include f'Lower 25% minus {right}' in the surrounding operation.
        )  # Finish assembling axis.errorbar inputs.
    axis.axhline(0, color="#707d8f", linestyle="--", linewidth=1)  # Axis.axhline applied to 0.
    axis.set(  # The result of axis.set.
        title="Paired collision-rate differences · exploratory two-way bootstrap",  # Include the report text in the surrounding operation.
        xlabel="Traffic density multiplier",  # Include 'Traffic density multiplier' in the surrounding operation.
        ylabel="Collision-rate difference (percentage points)",  # Include 'Collision-rate difference (percentage points)' in the surrounding operation.
        xticks=[1.0, 1.5, 2.0],  # Include 1.0, 1.5, 2.0 in the surrounding operation.
    )  # Finish assembling axis.set inputs.
    axis.grid(alpha=0.15)  # The result of axis.grid.
    axis.legend(fontsize=9)  # The result of axis.legend.
    fig.savefig(output / "controls_paired.png", dpi=180)  # Save a figure for the local results page; this does not affect agent behavior.
    plt.close(fig)  # Plt.close applied to fig.
    diagnostics = []  # Set diagnostics to .
    for name, label in [  # Process each name, label from 'dqn', 'DQN', 'qrdqn_mean', 'QR-DQN mean', 'qrdqn_tail25', 'QR-DQN lower 25%'.
        ("dqn", "DQN"),  # Include 'dqn', 'DQN' in the surrounding operation.
        ("qrdqn_mean", "QR-DQN mean"),  # Include 'qrdqn_mean', 'QR-DQN mean' in the surrounding operation.
        ("qrdqn_tail25", "QR-DQN lower 25%"),  # Include 'qrdqn_tail25', 'QR-DQN lower 25%' in the surrounding operation.
    ]:  # Finish assembling the grouped values above.
        path = root / "controls" / f"diagnostic_{name}.json"  # Set path to (the path to 'controls' inside root divided by f'diagnostic {name}.json').
        if path.exists():  # Check existing data instead of silently replacing scientific results.
            diagnostics.append((label, json.loads(path.read_text())))  # Add label, json.loads applied to the result of path.read text to diagnostics.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)  # Set fig, axes to plt.subplots applied to 1, 2.
    action_colors = ["#4889c9", "#98a2b3", "#9563c2", "#df9144", "#159c87"]  # Set action colors to '#4889c9', '#98a2b3', '#9563c2', '#df9144', '#159c87'.
    starts = np.zeros(len(diagnostics))  # Set starts to np.zeros applied to the number of entries in diagnostics.
    for action, color in enumerate(action_colors):  # Process each action, color from enumerate applied to action colors.
        values = [  # Set values to (the entries of the entries of item['summary'][0] selected by 'action_fractions' selected by action times 100) for each the unused result, item in diagnostics.
            item["summary"][0]["action_fractions"][action] * 100  # Include (the entries of the entries of the entries of item['summary'] selected by 0 selected by 'action_fractions' selected by action times 100) in the surrounding operation.
            for _, item in diagnostics  # Include the unused result, item, diagnostics in the surrounding operation.
        ]  # Finish assembling the grouped values above.
        axes[0].barh(  # Axes[0].barh applied to label for each label, the unused result in diagnostics, values.
            [label for label, _ in diagnostics],  # Include label for each label, the unused result in diagnostics in the surrounding operation.
            values,  # Include values in the surrounding operation.
            left=starts,  # Include starts in the surrounding operation.
            color=color,  # Include color in the surrounding operation.
            label=["Left", "Idle", "Right", "Faster", "Slower"][action],  # Include the entries of 'Left', 'Idle', 'Right', 'Faster', 'Slower' selected by action in the surrounding operation.
        )  # Finish assembling axes[0].barh inputs.
        starts += values  # Increase starts using values.
    axes[0].set(  # The result of axes[0].set.
        title="Decision mix · separate normal-traffic diagnostic block",  # Include 'Decision mix · separate normal-traffic diagnostic block' in the surrounding operation.
        xlabel="Actions (%)",  # Include 'Actions (%)' in the surrounding operation.
        xlim=(0, 100),  # Include 0, 100 in the surrounding operation.
    )  # Finish assembling axes[0].set inputs.
    axes[0].legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))  # The result of axes[0].legend.
    example = reward_example()  # Set example to the result of reward example.
    axes[1].bar(  # Axes[1].bar applied to 'fast, then crash', 'slower, survive', the entries of example selected by 'fast_crash_discounted_return', the entries of example selected by 'slow_survival_discounted_return'.
        ["Fast, then crash", "Slower, survive"],  # Include 'Fast, then crash', 'Slower, survive' in the surrounding operation.
        [  # Include the entries of example selected by 'fast_crash_discounted_return', the entries of example selected by 'slow_survival_discounted_return' in the surrounding operation.
            example["fast_crash_discounted_return"],  # Include the entries of example selected by 'fast_crash_discounted_return' in the surrounding operation.
            example["slow_survival_discounted_return"],  # Include the entries of example selected by 'slow_survival_discounted_return' in the surrounding operation.
        ],  # Finish assembling the grouped values above.
        color=["#df9144", "#159c87"],  # Include '#df9144', '#159c87' in the surrounding operation.
    )  # Finish assembling axes[1].bar inputs.
    axes[1].set(  # The result of axes[1].set.
        title="The reward can favor a crash · illustrative paths",  # Include 'The reward can favor a crash · illustrative paths' in the surrounding operation.
        ylabel="Discounted return (gamma = 0.9)",  # Include 'Discounted return (gamma = 0.9)' in the surrounding operation.
    )  # Finish assembling axes[1].set inputs.
    fig.savefig(output / "controls_diagnostics.png", dpi=180)  # Save a figure for the local results page; this does not affect agent behavior.
    plt.close(fig)  # Plt.close applied to fig.
    selection = json.loads((root / "controls/baseline_selection.json").read_text())  # Set selection to json.loads applied to the result of (root / 'controls/baseline selection.json').read text.
    nominal = {label: rows[0] for label, rows in groups.items()}  # Set nominal to the entries of rows selected by 0 for each label, rows in the result of groups.items.
    mean = nominal["QR-DQN mean"]  # Set mean to the entries of nominal selected by 'QR-DQN mean'.
    cautious = nominal["QR-DQN lower 25%"]  # Set cautious to the entries of nominal selected by 'QR-DQN lower 25%'.
    reduction = (mean["crash_rate"] - cautious["crash_rate"]) * 100  # Set reduction to ((the entries of mean selected by 'crash_rate' minus the entries of cautious selected by 'crash_rate') times 100).
    slow = (1 - cautious["mean_speed_m_s"] / mean["mean_speed_m_s"]) * 100  # Set slow to ((1 minus (the entries of cautious selected by 'mean_speed_m_s' divided by the entries of mean selected by 'mean_speed_m_s')) times 100).
    per_seed_rows = []  # Set per seed rows to .
    for label, rows in groups.items():  # Process each label, rows from the result of groups.items.
        for row in rows:  # Process each row from rows.
            for item in row["per_seed"]:  # Process each item from the entries of row selected by 'per_seed'.
                per_seed_rows.append(  # Add f"<tr><td>{html.escape(label)}</td><td>{row['density']:.1f}</td><td>{item['training seed']}</td><td>{item['crash rate'] * 100:.1f}%</td><td>{item['mean speed m s']:.2f}</td><td>{it to per seed rows.
                    f'<tr><td>{html.escape(label)}</td><td>{row["density"]:.1f}</td><td>{item["training_seed"]}</td><td>{item["crash_rate"]*100:.1f}%</td><td>{item["mean_speed_m_s"]:.2f}</td><td>{item["mean_distance_m"]:.0f}</td><td>{item["mean_duration_s"]:.1f}</td></tr>'  # Include f"<tr><td>{html.escape(label)}</td><td>{row['density']:.1f}</td><td>{item['training seed']}</td><td>{item['crash rate'] * 100:.1f}%</td><td>{item['mean speed m s']:.2f}</td><td>{it in the surrounding operation.
                )  # Finish assembling per seed rows.append inputs.
    replay_names = [  # Set replay names to 'Mean-return QR-DQN', 'qrdqn_seed0_mean.gif', 'Cautious QR-DQN', 'qrdqn_seed0_tail25.gif', 'Reward-tuned DQN', f"dqn penalty{abs(int(selection['selected collision penalty']))} seed0.gif".
        ("Mean-return QR-DQN", "qrdqn_seed0_mean.gif"),  # Include 'Mean-return QR-DQN', 'qrdqn_seed0_mean.gif' in the surrounding operation.
        ("Cautious QR-DQN", "qrdqn_seed0_tail25.gif"),  # Include 'Cautious QR-DQN', 'qrdqn_seed0_tail25.gif' in the surrounding operation.
        (  # Include 'Reward-tuned DQN', f"dqn penalty{abs(int(selection['selected collision penalty']))} seed0.gif" in the surrounding operation.
            "Reward-tuned DQN",  # Include 'Reward-tuned DQN' in the surrounding operation.
            f'dqn_penalty{abs(int(selection["selected_collision_penalty"]))}_seed0.gif',  # Include f"dqn penalty{abs(int(selection['selected collision penalty']))} seed0.gif" in the surrounding operation.
        ),  # Finish assembling the grouped values above.
    ]  # Finish assembling the grouped values above.
    if selection["selected_collision_penalty"] == -2:  # Use this branch when the entries of selection selected by 'selected_collision_penalty' equals negative 2.
        replay_names[-1] = ("Reward-tuned DQN (same as standard)", "dqn_seed0_mean.gif")  # Set the entries of replay names selected by -1 to 'Reward-tuned DQN (same as standard)', 'dqn_seed0_mean.gif'.
    governor_text = ""  # Set governor text to ''.
    governor_path = root / "controls/governor_selection.json"  # Set governor path to the path to 'controls/governor_selection.json' inside root.
    if governor_path.exists():  # Use this branch when the result of governor path.exists.
        governor = json.loads(governor_path.read_text())  # Set governor to json.loads applied to the result of governor path.read text.
        cap = governor["selected_cap_m_s"]  # Set cap to the entries of governor selected by 'selected_cap_m_s'.
        replay_names.append(("DQN + speed governor", f"dqn_cap{cap}_seed0.gif"))  # Add 'dqn + speed governor', f'dqn cap{cap} seed0.gif' to replay names.
        governor_text = f'<p>The additional observation-based governor uses a <strong>{cap} m/s</strong> threshold, selected from 22, 24 and 26 m/s for seed-0 speed closeness. Its seed-0 mismatch is <strong>{governor["seed0_speed_difference_m_s"]:+.2f} m/s</strong>. This exploratory control was added after reward tuning failed to match speed. It uses only ego velocity from the same observation and adds no training.</p>'  # Set governor text to f"<p>The additional observation-based governor uses a <strong>{cap} m/s</strong> threshold, selected from 22, 24 and 26 m/s for seed-0 speed closeness. Its seed-0 mismatch is <stro.
    replays = "".join(  # Set replays to ''.join applied to f'<article><h3>{html.escape(label)}</h3><img src="../{name}" alt="{html.escape(label)} driving replay"></article>' for each label, name in replay names.
        f'<article><h3>{html.escape(label)}</h3><img src="../{name}" alt="{html.escape(label)} driving replay"></article>'  # Include f'<article><h3>{html.escape(label)}</h3><img src="../{name}" alt="{html.escape(label)} driving replay"></article>' in the surrounding operation.
        for label, name in replay_names  # Include label, name, replay names in the surrounding operation.
    )  # Finish assembling ''.join inputs.
    payload = json.dumps(groups).replace("<", "\\u003c")  # Set payload to json.dumps(groups).replace applied to '<', '\\u003c'.
    document = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Risk-aware driving · replicated experiment</title><style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#0d1423;color:#e9eef7;font:16px/1.65 system-ui,sans-serif}
main{max-width:1200px;margin:auto;padding:50px 28px}h1{font-size:clamp(30px,4vw,48px);line-height:1.13;max-width:1000px;letter-spacing:-.025em}
.eyebrow{color:#62dbbf;letter-spacing:.12em;font-size:12px;font-weight:750}.intro,small,footer{color:#acbad0}.intro{max-width:950px}
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:15px;margin:30px 0}.card{background:#172439;padding:20px;border-radius:12px}.number{font-size:32px;font-weight:750;color:#8be3cf}.caption{color:#acbad0;font-size:13px}
section{margin-top:44px}.figure{width:100%;border-radius:12px;background:white}h2{font-size:24px;letter-spacing:-.015em}p{max-width:1000px}
.notice{padding:18px 22px;background:#172439;border-left:3px solid #62dbbf;border-radius:6px}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;padding:12px;border-bottom:1px solid #283348;white-space:nowrap}th{color:#8ddfcd}tr.highlight{background:#13332f}
select{font:inherit;padding:8px 12px;border:1px solid #4a5b74;background:#172439;border-radius:7px;color:#e9eef7}label{margin-right:12px}a{color:#8ddfcd}summary{cursor:pointer;font-weight:650}details{margin-top:25px}article img{width:100%;border-radius:8px}article{margin:22px 0}h3{font-size:16px}footer{font-size:13px;margin-top:40px}@media(max-width:650px){.cards{grid-template-columns:1fr}main{padding:28px 16px}}
</style><main><div class="eyebrow">DEEP REINFORCEMENT LEARNING / DEVELOPMENT EXPERIMENT</div>
<h1>Can a cautious return selector make deep RL driving safer?</h1>
<p class="intro">Three independent training seeds, a stronger scalar baseline, and paired evaluations separate a promising first run from a repeatable result.</p>
<div class="cards"><div class="card"><div class="number">REDUCTION pp</div><div class="caption">fewer normal-traffic collisions vs QR-DQN mean</div></div><div class="card"><div class="number">SLOWER% slower</div><div class="caption">normal-traffic speed cost of the cautious selector</div></div><div class="card"><div class="number">3 training seeds</div><div class="caption">20,000 interactions per model · 30 scenarios per density</div></div></div>
<p class="notice"><strong>Development evidence.</strong> No final confirmation scenarios have been used. All learned methods are evaluated with the same reward. The scalar baseline was selected for speed closeness using development data; actual residual speed differences are reported. Three training seeds give limited evidence of robustness.</p>
<section><h2>Compare the policies</h2><label for="density">Traffic density</label><select id="density"><option value="1">1.0 · training density</option><option value="1.5">1.5 · denser traffic</option><option value="2">2.0 · stress test</option></select>
<div class="scroll"><table><thead><tr><th>Method</th><th>Crashes</th><th>Seed range</th><th>Speed m/s</th><th>Distance m</th><th>Duration s</th><th>Common return</th></tr></thead><tbody id="metrics"></tbody></table></div><p><small>Equal-weight averages across trained models. Ranges show the three training-seed estimates; they are not confidence intervals. Each model sees the same 30 scenario seeds at each density.</small></p>
<img class="figure" src="controls_replication.png" alt="Mean collision rate and speed with training seed ranges"><p><small>Lines: seed means. Dots: individual trained models. Shading: minimum to maximum across the three training seeds.</small></p></section>
<section><h2>Does the advantage survive stronger controls?</h2><p>Reward-tuned DQN uses a training collision penalty of <strong>PENALTY</strong>, selected only for proximity to the cautious seed-0 policy's normal-traffic speed. Its seed-0 speed mismatch is <strong>MISMATCH m/s</strong>. Selection used neither collision rate nor the other training seeds. This is speed targeting, not an exact speed match.</p>GOVERNOR
<img class="figure" src="controls_tradeoff.png" alt="Speed versus collisions for all seeds and densities"><p><small>Small dots: individual models. Diamonds: seed averages. Lower collision rate and higher speed are preferable; a faster policy and a safer policy can trade places without either dominating.</small></p>
<img class="figure" src="controls_paired.png" alt="Paired policy differences with exploratory intervals"><p><small>Negative differences favor the cautious selector on collisions. Intervals use 5,000 two-way bootstrap replicates over training seeds and shared scenario seeds while retaining policy pairing. With only three trained models, treat these intervals as exploratory.</small></p></section>
<section><h2>What changed in the behavior?</h2><img class="figure" src="controls_diagnostics.png" alt="Action mix and illustrative discounted reward paths">
<p>The action trace uses separate development scenarios (11000–11029). Gap and time-to-collision measurements use privileged simulator state for diagnosis only. They never enter the learned policy.</p>
<p>The reward illustration compares nine rewards of 0.4 followed by a -2 crash against thirty rewards of 0.12 without a crash. At gamma 0.9 the first path has higher discounted return. This demonstrates an objective-design issue; it does not establish the cause of a particular failure.</p></section>
<section><h2>Replay the same initial scenario</h2><p><small>Training seed 0, scenario seed 10000, normal traffic. These rollouts were selected by seed before inspecting outcomes.</small></p>REPLAYS</section>
<details><summary>Inspect every training-seed result</summary><div class="scroll"><table><thead><tr><th>Method</th><th>Density</th><th>Training seed</th><th>Crashes</th><th>Speed m/s</th><th>Distance m</th><th>Duration s</th></tr></thead><tbody>SEEDROWS</tbody></table></div></details>
<section><h2>Reproduce and inspect</h2><p><a href="../controls/summary.json">Raw aggregated results and paired comparisons</a> · <a href="../controls/baseline_selection.json">Recorded baseline selection</a> · <a href="../baseline-report/index.html">Preserved initial experiment report</a></p><p><code>python run_controls.py --jobs 2</code> reuses verified results and reproduces this experiment. Source code, configuration, model checkpoints, scenario results and diagnostic traces are included in the project.</p></section>
<footer>Lower-tail action selection remains a deployment heuristic, without a global CVaR optimality or safety guarantee. The network was trained for mean-greedy continuation; changing its selector changes the continuation policy. Inputs partially observe the simulator. Density changes vehicle spacing, not the entire driving domain. The full-state cautious rule from initial experiment is preserved separately because it has privileged information. Final confirmation seeds 20000 onward remain reserved.</footer>
<script>const groups=PAYLOAD;function render(){const d=Number(document.getElementById('density').value);const body=document.getElementById('metrics');body.replaceChildren();for(const [label,rows] of Object.entries(groups)){const r=rows.find(x=>x.density===d);const tr=document.createElement('tr');if(label==='QR-DQN lower 25%')tr.className='highlight';const values=[label,(100*r.crash_rate).toFixed(1)+'%',r.crash_rate_seed_range.map(x=>(100*x).toFixed(1)+'%').join(' – '),r.mean_speed_m_s.toFixed(2),r.mean_distance_m.toFixed(0),r.mean_duration_s.toFixed(1),r.mean_return.toFixed(2)];for(const value of values){const td=document.createElement('td');td.textContent=value;tr.appendChild(td)}body.appendChild(tr)}}document.getElementById('density').addEventListener('change',render);render();</script></main></html>"""
    replacements = {  # Set replacements to the named settings or measurements assembled below.
        "REDUCTION": f"{reduction:.1f}",  # Record 'REDUCTION' as f'{reduction:.1f}'.
        "SLOWER": f"{slow:.1f}",  # Record 'SLOWER' as f'{slow:.1f}'.
        "PENALTY": str(selection["selected_collision_penalty"]),  # Record 'PENALTY' as the entries of selection selected by 'selected_collision_penalty' as a text value.
        "MISMATCH": f'{selection["seed0_speed_difference_m_s"]:+.2f}',  # Record 'MISMATCH' as f"{selection['seed0 speed difference m s']:+.2f}".
        "REPLAYS": replays,  # Record 'REPLAYS' as replays.
        "SEEDROWS": "".join(per_seed_rows),  # Record 'SEEDROWS' as ''.join applied to per seed rows.
        "PAYLOAD": payload,  # Record 'PAYLOAD' as payload.
        "GOVERNOR": governor_text,  # Record 'GOVERNOR' as governor text.
    }  # Finish assembling the grouped values above.
    for key, value in replacements.items():  # Process each key, value from the result of replacements.items.
        document = document.replace(key, value)  # Set document to document.replace applied to key, value.
    (output / "index.html").write_text(document)  # (output / 'index.html').write text applied to document.
    print(f"Wrote {output / 'index.html'}")  # Print applied to f"wrote {output / 'index.html'}".
