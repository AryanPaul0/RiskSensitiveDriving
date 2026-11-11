"""Create the controls results page.

Read saved numbers, calculate clearly labeled summaries, draw figures, and write
an HTML page. This file presents evidence; it does not train or select models.
Long HTML strings are page templates. The reinforcement-learning calculations
live in agents.py and experiment.py, not inside the page layout.

Original implementation notes:
Publication-style figures plus a portable viewer for the control experiments.
"""

# import makes these modules available; "as" gives a module a shorter local name.
import html
# import makes these modules available; "as" gives a module a shorter local name.
import json
# from pathlib import brings the named tools into this file; parentheses allow several lines.
from pathlib import Path

# import makes these modules available; "as" gives a module a shorter local name.
import numpy as np

# from diagnostics import brings the named tools into this file; parentheses allow several lines.
from .diagnostics import reward_example

# = stores a result in COLORS; build a dictionary, which stores values under named keys.
COLORS = {
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "DQN": "#4889c9",
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "QR-DQN mean": "#9563c2",
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "QR-DQN lower 25%": "#159c87",
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "Reward-tuned DQN": "#df9144",
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "DQN + speed governor": "#d7677e",
# } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
}


# def defines create_report; parentheses list its inputs, and the colon starts its indented body.
def create_report(root):
    """Build graphs and an HTML page from saved result files.
    The arrays below hold measured results; plotting does not change the models.
    HTML strings describe page layout, while Python inserts the saved numbers.
    Keep primary comparisons separate from exploratory comparisons in the report.
    """
    # import makes these modules available; "as" gives a module a shorter local name.
    import matplotlib

    # Call matplotlib.use using parentheses to supply its inputs; this line does not store a separate return value.
    matplotlib.use("Agg")
    # import makes these modules available; "as" gives a module a shorter local name.
    import matplotlib.pyplot as plt

    # = stores a result in root; call Path using parentheses to supply its inputs.
    root = Path(root)
    # = stores a result in data; call json.loads using parentheses to supply its inputs.
    data = json.loads((root / "controls/summary.json").read_text())
    # = stores a result in groups; use square brackets to select part of data.
    groups = data["groups"]
    # = stores a result in output; apply / to these operands; numbers are divided, while Path objects join folder names.
    output = root / "report"
    # Call output.mkdir using parentheses to supply its inputs; this line does not store a separate return value.
    output.mkdir(exist_ok=True)
    # Call plt.rcparams.update using parentheses to supply its inputs; this line does not store a separate return value.
    plt.rcParams.update(
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        {
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "figure.facecolor": "#f5f7fb",
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "axes.facecolor": "white",
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "axes.spines.top": False,
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "axes.spines.right": False,
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "font.size": 11,
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "axes.titlesize": 13,
        # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
        }
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # = unpacks the right-hand result into (fig, axes), matching values to names in order.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    # for takes each item from groups.items() in turn; (label, rows) holds the current item.
    for label, rows in groups.items():
        # = stores a result in densities; collect or generate one result per item; the "for" part names the item being processed.
        densities = [row["density"] for row in rows]
        # for takes each item from [(axes[0], 'crash_rate', 100), (axes[1], 'mean_speed_m_s', 1)] in turn; (axis, metric, multiplier) holds the current item.
        for axis, metric, multiplier in [
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            (axes[0], "crash_rate", 100),
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            (axes[1], "mean_speed_m_s", 1),
        # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
        ]:
            # = stores a result in means; collect or generate one result per item; the "for" part names the item being processed.
            means = [row[metric] * multiplier for row in rows]
            # = stores a result in lows; collect or generate one result per item; the "for" part names the item being processed.
            lows = [row[metric + "_seed_range"][0] * multiplier for row in rows]
            # = stores a result in highs; collect or generate one result per item; the "for" part names the item being processed.
            highs = [row[metric + "_seed_range"][1] * multiplier for row in rows]
            # Call axis.plot using parentheses to supply its inputs; this line does not store a separate return value.
            axis.plot(densities, means, "o-", color=COLORS[label], label=label)
            # Call axis.fill_between using parentheses to supply its inputs; this line does not store a separate return value.
            axis.fill_between(densities, lows, highs, color=COLORS[label], alpha=0.10)
            # for takes each item from rows in turn; row holds the current item.
            for row in rows:
                # Call axis.scatter using parentheses to supply its inputs; this line does not store a separate return value.
                axis.scatter(
                    # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                    [row["density"]] * len(row["per_seed"]),
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    [item[metric] * multiplier for item in row["per_seed"]],
                    # Inside this continued call, name=value supplies an option by name rather than by its position.
                    s=20,
                    # Inside this continued call, name=value supplies an option by name rather than by its position.
                    alpha=0.5,
                    # Inside this continued call, name=value supplies an option by name rather than by its position.
                    color=COLORS[label],
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                )
    # Call axes[0].set using parentheses to supply its inputs; this line does not store a separate return value.
    axes[0].set(
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        title="Collision rate across traffic densities",
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        ylabel="Episodes ending in collision (%)",
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        ylim=(-3, 103),
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # Call axes[1].set using parentheses to supply its inputs; this line does not store a separate return value.
    axes[1].set(
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        title="Speed across traffic densities", ylabel="Episode-averaged speed (m/s)"
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # for takes each item from axes in turn; axis holds the current item.
    for axis in axes:
        # Call axis.set_xlabel using parentheses to supply its inputs; this line does not store a separate return value.
        axis.set_xlabel("Traffic density multiplier")
        # Call axis.grid using parentheses to supply its inputs; this line does not store a separate return value.
        axis.grid(alpha=0.15)
        # Call axis.legend using parentheses to supply its inputs; this line does not store a separate return value.
        axis.legend(fontsize=8)
    # Save a figure for the local results page; this does not affect agent behavior.
    # Call fig.savefig using parentheses to supply its inputs; this line does not store a separate return value.
    fig.savefig(output / "controls_replication.png", dpi=180)
    # Call plt.close using parentheses to supply its inputs; this line does not store a separate return value.
    plt.close(fig)
    # = unpacks the right-hand result into (fig, axes), matching values to names in order.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
    # for takes each item from zip(axes, [1.0, 1.5, 2.0]) in turn; (axis, density) holds the current item.
    for axis, density in zip(axes, [1.0, 1.5, 2.0]):
        # for takes each item from groups.items() in turn; (label, rows) holds the current item.
        for label, rows in groups.items():
            # = stores a result in row; call next using parentheses to supply its inputs.
            row = next(item for item in rows if item["density"] == density)
            # Call axis.scatter using parentheses to supply its inputs; this line does not store a separate return value.
            axis.scatter(
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                [item["mean_speed_m_s"] for item in row["per_seed"]],
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                [item["crash_rate"] * 100 for item in row["per_seed"]],
                # Inside this continued call, name=value supplies an option by name rather than by its position.
                color=COLORS[label],
                # Inside this continued call, name=value supplies an option by name rather than by its position.
                s=32,
                # Inside this continued call, name=value supplies an option by name rather than by its position.
                alpha=0.5,
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            )
            # Call axis.scatter using parentheses to supply its inputs; this line does not store a separate return value.
            axis.scatter(
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                row["mean_speed_m_s"],
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                row["crash_rate"] * 100,
                # Inside this continued call, name=value supplies an option by name rather than by its position.
                color=COLORS[label],
                # Inside this continued call, name=value supplies an option by name rather than by its position.
                s=100,
                # Inside this continued call, name=value supplies an option by name rather than by its position.
                marker="D",
                # Inside this continued call, name=value supplies an option by name rather than by its position.
                label=label,
                # Inside this continued call, name=value supplies an option by name rather than by its position.
                edgecolors="white",
                # Inside this continued call, name=value supplies an option by name rather than by its position.
                linewidths=1,
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            )
        # Call axis.set using parentheses to supply its inputs; this line does not store a separate return value.
        axis.set(
            # Inside this continued call, name=value supplies an option by name rather than by its position.
            title=f"Density {density:.1f}", xlabel="Mean speed (m/s)", ylim=(-3, 103)
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
        # Call axis.grid using parentheses to supply its inputs; this line does not store a separate return value.
        axis.grid(alpha=0.15)
    # Call axes[0].set_ylabel using parentheses to supply its inputs; this line does not store a separate return value.
    axes[0].set_ylabel("Collision rate (%)")
    # Call axes[-1].legend using parentheses to supply its inputs; this line does not store a separate return value.
    axes[-1].legend(fontsize=7, loc="lower right")
    # Save a figure for the local results page; this does not affect agent behavior.
    # Call fig.savefig using parentheses to supply its inputs; this line does not store a separate return value.
    fig.savefig(output / "controls_tradeoff.png", dpi=180)
    # Call plt.close using parentheses to supply its inputs; this line does not store a separate return value.
    plt.close(fig)
    # = stores a result in contrasts; call sorted using parentheses to supply its inputs.
    contrasts = sorted(
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        {(row["left"], row["right"]) for row in data["paired_comparisons"]}
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # = unpacks the right-hand result into (fig, axis), matching values to names in order.
    fig, axis = plt.subplots(figsize=(10, 4.2), constrained_layout=True)
    # for takes each item from enumerate(contrasts) in turn; (index, (left, right)) holds the current item.
    for index, (left, right) in enumerate(contrasts):
        # = stores a result in selected; collect or generate one result per item; the "for" part names the item being processed.
        selected = [
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            row
            # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
            for row in data["paired_comparisons"]
            # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
            if row["left"] == left and row["right"] == right
        # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
        ]
        # = stores a result in x; combine the operands with the displayed operator; parentheses control grouping.
        x = (
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            np.array([row["density"] for row in selected])
            # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
            + (index - (len(contrasts) - 1) / 2) * 0.025
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
        # = stores a result in y; call np.array using parentheses to supply its inputs.
        y = np.array([row["crashed"]["mean_difference"] * 100 for row in selected])
        # = stores a result in intervals; multiply the operands; NumPy and PyTorch arrays do this element by element.
        intervals = (
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            np.array(
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                [row["crashed"]["exploratory_95_percent_interval"] for row in selected]
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            )
            # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
            * 100
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
        # Call axis.errorbar using parentheses to supply its inputs; this line does not store a separate return value.
        axis.errorbar(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            x,
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            y,
            # Inside this continued call, name=value supplies an option by name rather than by its position.
            yerr=np.maximum(0, np.stack([y - intervals[:, 0], intervals[:, 1] - y])),
            # Inside this continued call, name=value supplies an option by name rather than by its position.
            fmt="o-",
            # Inside this continued call, name=value supplies an option by name rather than by its position.
            capsize=5,
            # Inside this continued call, name=value supplies an option by name rather than by its position.
            label=f"Lower 25% minus {right}",
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
    # Call axis.axhline using parentheses to supply its inputs; this line does not store a separate return value.
    axis.axhline(0, color="#707d8f", linestyle="--", linewidth=1)
    # Call axis.set using parentheses to supply its inputs; this line does not store a separate return value.
    axis.set(
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        title="Paired collision-rate differences · exploratory two-way bootstrap",
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        xlabel="Traffic density multiplier",
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        ylabel="Collision-rate difference (percentage points)",
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        xticks=[1.0, 1.5, 2.0],
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # Call axis.grid using parentheses to supply its inputs; this line does not store a separate return value.
    axis.grid(alpha=0.15)
    # Call axis.legend using parentheses to supply its inputs; this line does not store a separate return value.
    axis.legend(fontsize=9)
    # Save a figure for the local results page; this does not affect agent behavior.
    # Call fig.savefig using parentheses to supply its inputs; this line does not store a separate return value.
    fig.savefig(output / "controls_paired.png", dpi=180)
    # Call plt.close using parentheses to supply its inputs; this line does not store a separate return value.
    plt.close(fig)
    # = stores a result in diagnostics; build a list, keeping its items in order.
    diagnostics = []
    # for takes each item from [('dqn', 'DQN'), ('qrdqn_mean', 'QR-DQN mean'), ('qrdqn_tail25', 'QR-DQN lower 25%')] in turn; (name, label) holds the current item.
    for name, label in [
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        ("dqn", "DQN"),
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        ("qrdqn_mean", "QR-DQN mean"),
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        ("qrdqn_tail25", "QR-DQN lower 25%"),
    # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
    ]:
        # = stores a result in path; apply / to these operands; numbers are divided, while Path objects join folder names.
        path = root / "controls" / f"diagnostic_{name}.json"
        # Check existing data instead of silently replacing scientific results.
        # if checks path.exists(); only a true condition runs the indented block.
        if path.exists():
            # Call diagnostics.append using parentheses to supply its inputs; this line does not store a separate return value.
            diagnostics.append((label, json.loads(path.read_text())))
    # = unpacks the right-hand result into (fig, axes), matching values to names in order.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    # = stores a result in action_colors; build a list, keeping its items in order.
    action_colors = ["#4889c9", "#98a2b3", "#9563c2", "#df9144", "#159c87"]
    # = stores a result in starts; call np.zeros using parentheses to supply its inputs.
    starts = np.zeros(len(diagnostics))
    # for takes each item from enumerate(action_colors) in turn; (action, color) holds the current item.
    for action, color in enumerate(action_colors):
        # = stores a result in values; collect or generate one result per item; the "for" part names the item being processed.
        values = [
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            item["summary"][0]["action_fractions"][action] * 100
            # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
            for _, item in diagnostics
        # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
        ]
        # Call axes[0].barh using parentheses to supply its inputs; this line does not store a separate return value.
        axes[0].barh(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            [label for label, _ in diagnostics],
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            values,
            # Inside this continued call, name=value supplies an option by name rather than by its position.
            left=starts,
            # Inside this continued call, name=value supplies an option by name rather than by its position.
            color=color,
            # Inside this continued call, name=value supplies an option by name rather than by its position.
            label=["Left", "Idle", "Right", "Faster", "Slower"][action],
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
        # The combined operator updates starts using its previous value.
        starts += values
    # Call axes[0].set using parentheses to supply its inputs; this line does not store a separate return value.
    axes[0].set(
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        title="Decision mix · separate normal-traffic diagnostic block",
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        xlabel="Actions (%)",
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        xlim=(0, 100),
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # Call axes[0].legend using parentheses to supply its inputs; this line does not store a separate return value.
    axes[0].legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    # = stores a result in example; call reward_example using parentheses to supply its inputs.
    example = reward_example()
    # Call axes[1].bar using parentheses to supply its inputs; this line does not store a separate return value.
    axes[1].bar(
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        ["Fast, then crash", "Slower, survive"],
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        [
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            example["fast_crash_discounted_return"],
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            example["slow_survival_discounted_return"],
        # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
        ],
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        color=["#df9144", "#159c87"],
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # Call axes[1].set using parentheses to supply its inputs; this line does not store a separate return value.
    axes[1].set(
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        title="The reward can favor a crash · illustrative paths",
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        ylabel="Discounted return (gamma = 0.9)",
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # Save a figure for the local results page; this does not affect agent behavior.
    # Call fig.savefig using parentheses to supply its inputs; this line does not store a separate return value.
    fig.savefig(output / "controls_diagnostics.png", dpi=180)
    # Call plt.close using parentheses to supply its inputs; this line does not store a separate return value.
    plt.close(fig)
    # = stores a result in selection; call json.loads using parentheses to supply its inputs.
    selection = json.loads((root / "controls/baseline_selection.json").read_text())
    # = stores a result in nominal; collect or generate one result per item; the "for" part names the item being processed.
    nominal = {label: rows[0] for label, rows in groups.items()}
    # = stores a result in mean; use square brackets to select part of nominal.
    mean = nominal["QR-DQN mean"]
    # = stores a result in cautious; use square brackets to select part of nominal.
    cautious = nominal["QR-DQN lower 25%"]
    # = stores a result in reduction; multiply the operands; NumPy and PyTorch arrays do this element by element.
    reduction = (mean["crash_rate"] - cautious["crash_rate"]) * 100
    # = stores a result in slow; multiply the operands; NumPy and PyTorch arrays do this element by element.
    slow = (1 - cautious["mean_speed_m_s"] / mean["mean_speed_m_s"]) * 100
    # = stores a result in per_seed_rows; build a list, keeping its items in order.
    per_seed_rows = []
    # for takes each item from groups.items() in turn; (label, rows) holds the current item.
    for label, rows in groups.items():
        # for takes each item from rows in turn; row holds the current item.
        for row in rows:
            # for takes each item from row['per_seed'] in turn; item holds the current item.
            for item in row["per_seed"]:
                # Call per_seed_rows.append using parentheses to supply its inputs; this line does not store a separate return value.
                per_seed_rows.append(
                    # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                    f'<tr><td>{html.escape(label)}</td><td>{row["density"]:.1f}</td><td>{item["training_seed"]}</td><td>{item["crash_rate"]*100:.1f}%</td><td>{item["mean_speed_m_s"]:.2f}</td><td>{item["mean_distance_m"]:.0f}</td><td>{item["mean_duration_s"]:.1f}</td></tr>'
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                )
    # = stores a result in replay_names; build a list, keeping its items in order.
    replay_names = [
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        ("Mean-return QR-DQN", "qrdqn_seed0_mean.gif"),
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        ("Cautious QR-DQN", "qrdqn_seed0_tail25.gif"),
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        (
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "Reward-tuned DQN",
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            f'dqn_penalty{abs(int(selection["selected_collision_penalty"]))}_seed0.gif',
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        ),
    # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
    ]
    # if checks selection['selected_collision_penalty'] == -2; only a true condition runs the indented block.
    if selection["selected_collision_penalty"] == -2:
        # = stores a result in replay_names[-1]; group these values into a tuple, keeping their order.
        replay_names[-1] = ("Reward-tuned DQN (same as standard)", "dqn_seed0_mean.gif")
    # = stores a result in governor_text; use quoted text as data; Python does not execute the words inside it.
    governor_text = ""
    # = stores a result in governor_path; apply / to these operands; numbers are divided, while Path objects join folder names.
    governor_path = root / "controls/governor_selection.json"
    # if checks governor_path.exists(); only a true condition runs the indented block.
    if governor_path.exists():
        # = stores a result in governor; call json.loads using parentheses to supply its inputs.
        governor = json.loads(governor_path.read_text())
        # = stores a result in cap; use square brackets to select part of governor.
        cap = governor["selected_cap_m_s"]
        # Call replay_names.append using parentheses to supply its inputs; this line does not store a separate return value.
        replay_names.append(("DQN + speed governor", f"dqn_cap{cap}_seed0.gif"))
        # = stores a result in governor_text; evaluate the expression on the right before using its result.
        governor_text = f'<p>The additional observation-based governor uses a <strong>{cap} m/s</strong> threshold, selected from 22, 24 and 26 m/s for seed-0 speed closeness. Its seed-0 mismatch is <strong>{governor["seed0_speed_difference_m_s"]:+.2f} m/s</strong>. This exploratory control was added after reward tuning failed to match speed. It uses only ego velocity from the same observation and adds no training.</p>'
    # = stores a result in replays; call ''.join using parentheses to supply its inputs.
    replays = "".join(
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        f'<article><h3>{html.escape(label)}</h3><img src="../{name}" alt="{html.escape(label)} driving replay"></article>'
        # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
        for label, name in replay_names
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # = stores a result in payload; call json.dumps(groups).replace using parentheses to supply its inputs.
    payload = json.dumps(groups).replace("<", "\\u003c")
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
    # = stores a result in replacements; build a dictionary, which stores values under named keys.
    replacements = {
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "REDUCTION": f"{reduction:.1f}",
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "SLOWER": f"{slow:.1f}",
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "PENALTY": str(selection["selected_collision_penalty"]),
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "MISMATCH": f'{selection["seed0_speed_difference_m_s"]:+.2f}',
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "REPLAYS": replays,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "SEEDROWS": "".join(per_seed_rows),
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "PAYLOAD": payload,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "GOVERNOR": governor_text,
    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
    }
    # for takes each item from replacements.items() in turn; (key, value) holds the current item.
    for key, value in replacements.items():
        # = stores a result in document; call document.replace using parentheses to supply its inputs.
        document = document.replace(key, value)
    # Call (output / 'index.html').write_text using parentheses to supply its inputs; this line does not store a separate return value.
    (output / "index.html").write_text(document)
    # Call print using parentheses to supply its inputs; this line does not store a separate return value.
    print(f"Wrote {output / 'index.html'}")
