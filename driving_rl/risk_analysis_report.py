"""Create the risk analysis results page.

Read saved numbers, calculate clearly labeled summaries, draw figures, and write
an HTML page. This file presents evidence; it does not train or select models.
Long HTML strings are page templates. The reinforcement-learning calculations
live in agents.py and experiment.py, not inside the page layout.

Original implementation notes:
Scientific figures and an interactive viewer for risk analysis development results.
"""

# import makes these modules available; "as" gives a module a shorter local name.
import json
# from pathlib import brings the named tools into this file; parentheses allow several lines.
from pathlib import Path

# import makes these modules available; "as" gives a module a shorter local name.
import numpy as np

# = stores a result in COLORS; build a dictionary, which stores values under named keys.
COLORS = {
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "QR-DQN mean": "#9563c2",
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "QR-DQN lower 25%": "#159c87",
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "Same QR-DQN + governor": "#df9144",
# } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
}
# = stores a result in POLICIES; build a dictionary, which stores values under named keys.
POLICIES = {
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "mean": "QR-DQN mean",
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "tail25": "QR-DQN lower 25%",
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "governed": "Same QR-DQN + governor",
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
    data = json.loads((root / "risk_analysis/summary.json").read_text())
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
            "font.size": 10,
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "axes.titlesize": 12,
        # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
        }
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # = unpacks the right-hand result into (fig, axes), matching values to names in order.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    # for takes each item from data['groups'].items() in turn; (name, rows) holds the current item.
    for name, rows in data["groups"].items():
        # = stores a result in x; collect or generate one result per item; the "for" part names the item being processed.
        x = [row["density"] for row in rows]
        # for takes each item from [(axes[0], 'crash_rate', 100), (axes[1], 'mean_speed_m_s', 1)] in turn; (axis, metric, scale) holds the current item.
        for axis, metric, scale in [
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            (axes[0], "crash_rate", 100),
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            (axes[1], "mean_speed_m_s", 1),
        # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
        ]:
            # Call axis.plot using parentheses to supply its inputs; this line does not store a separate return value.
            axis.plot(
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                x,
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                [row[metric] * scale for row in rows],
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "o-",
                # Inside this continued call, name=value supplies an option by name rather than by its position.
                color=COLORS[name],
                # Inside this continued call, name=value supplies an option by name rather than by its position.
                label=name,
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            )
            # for takes each item from rows in turn; row holds the current item.
            for row in rows:
                # Call axis.scatter using parentheses to supply its inputs; this line does not store a separate return value.
                axis.scatter(
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    [row["density"]] * 3,
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    [seed[metric] * scale for seed in row["per_seed"]],
                    # Inside this continued call, name=value supplies an option by name rather than by its position.
                    color=COLORS[name],
                    # Inside this continued call, name=value supplies an option by name rather than by its position.
                    alpha=0.35,
                    # Inside this continued call, name=value supplies an option by name rather than by its position.
                    s=16,
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                )
    # = stores a result in comparisons; use square brackets to select part of data.
    comparisons = data["paired_comparisons"]
    # = stores a result in x; collect or generate one result per item; the "for" part names the item being processed.
    x = [row["density"] for row in comparisons]
    # = stores a result in y; call np.array using parentheses to supply its inputs.
    y = np.array([row["crashed"]["mean_difference"] * 100 for row in comparisons])
    # = stores a result in bounds; multiply the operands; NumPy and PyTorch arrays do this element by element.
    bounds = (
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        np.array(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            [row["crashed"]["exploratory_95_percent_interval"] for row in comparisons]
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        * 100
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # Call axes[2].errorbar using parentheses to supply its inputs; this line does not store a separate return value.
    axes[2].errorbar(
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        x,
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        y,
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        yerr=np.maximum(0, np.stack([y - bounds[:, 0], bounds[:, 1] - y])),
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        fmt="o-",
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        capsize=5,
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        color="#159c87",
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # Call axes[2].axhline using parentheses to supply its inputs; this line does not store a separate return value.
    axes[2].axhline(0, color="#7a8697", linestyle="--")
    # Call axes[0].set using parentheses to supply its inputs; this line does not store a separate return value.
    axes[0].set(
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        title="Same models, different deployment",
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        ylabel="Collision rate (%)",
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        ylim=(-3, 103),
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # Call axes[1].set using parentheses to supply its inputs; this line does not store a separate return value.
    axes[1].set(
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        title="Check the remaining speed mismatch", ylabel="Average speed (m/s)"
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # Call axes[2].set using parentheses to supply its inputs; this line does not store a separate return value.
    axes[2].set(
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        title="Cautious minus governed · paired",
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        ylabel="Collision difference (percentage points)",
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # for takes each item from axes in turn; axis holds the current item.
    for axis in axes:
        # Call axis.set_xlabel using parentheses to supply its inputs; this line does not store a separate return value.
        axis.set_xlabel("Traffic density")
        # Call axis.grid using parentheses to supply its inputs; this line does not store a separate return value.
        axis.grid(alpha=0.15)
    # Call axes[0].legend using parentheses to supply its inputs; this line does not store a separate return value.
    axes[0].legend(fontsize=7, loc="lower right")
    # Save a figure for the local results page; this does not affect agent behavior.
    # Call fig.savefig using parentheses to supply its inputs; this line does not store a separate return value.
    fig.savefig(output / "risk_analysis_control.png", dpi=180)
    # Call plt.close using parentheses to supply its inputs; this line does not store a separate return value.
    plt.close(fig)
    # = unpacks the right-hand result into (fig, axes), matching values to names in order.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True)
    # for takes each item from zip(axes, [1.0, 1.5, 2.0]) in turn; (axis, density) holds the current item.
    for axis, density in zip(axes, [1.0, 1.5, 2.0]):
        # = stores a result in rows; collect or generate one result per item; the "for" part names the item being processed.
        rows = [
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            next(row for row in item["summary"] if row["density"] == density)
            # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
            for item in data["risk_frontier"]
        # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
        ]
        # = stores a result in speeds; collect or generate one result per item; the "for" part names the item being processed.
        speeds = [row["mean_speed_m_s"] for row in rows]
        # = stores a result in crashes; collect or generate one result per item; the "for" part names the item being processed.
        crashes = [row["crash_rate"] * 100 for row in rows]
        # Call axis.plot using parentheses to supply its inputs; this line does not store a separate return value.
        axis.plot(
            # Inside this continued call, name=value supplies an option by name rather than by its position.
            speeds, crashes, "o-", color="#159c87", label="Quantile tail selector"
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
        # for takes each item from zip(data['risk_frontier'], speeds, crashes) in turn; (item, speed, crash) holds the current item.
        for item, speed, crash in zip(data["risk_frontier"], speeds, crashes):
            # Call axis.annotate using parentheses to supply its inputs; this line does not store a separate return value.
            axis.annotate(
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                f'α={item["alpha"]:g}',
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                (speed, crash),
                # Inside this continued call, name=value supplies an option by name rather than by its position.
                xytext=(4, -12),
                # Inside this continued call, name=value supplies an option by name rather than by its position.
                textcoords="offset points",
                # Inside this continued call, name=value supplies an option by name rather than by its position.
                fontsize=8,
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            )
        # = stores a result in control; call next using parentheses to supply its inputs.
        control = next(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            row
            # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
            for row in data["groups"]["Same QR-DQN + governor"]
            # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
            if row["density"] == density
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
        # Call axis.scatter using parentheses to supply its inputs; this line does not store a separate return value.
        axis.scatter(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            control["mean_speed_m_s"],
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            control["crash_rate"] * 100,
            # Inside this continued call, name=value supplies an option by name rather than by its position.
            color="#df9144",
            # Inside this continued call, name=value supplies an option by name rather than by its position.
            marker="*",
            # Inside this continued call, name=value supplies an option by name rather than by its position.
            s=160,
            # Inside this continued call, name=value supplies an option by name rather than by its position.
            label="Mean + fixed governor",
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
        # Call axis.set using parentheses to supply its inputs; this line does not store a separate return value.
        axis.set(
            # Inside this continued call, name=value supplies an option by name rather than by its position.
            title=f"Density {density:g}", xlabel="Mean speed (m/s)", ylim=(-3, 108)
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
        # Call axis.grid using parentheses to supply its inputs; this line does not store a separate return value.
        axis.grid(alpha=0.15)
    # Call axes[0].set_ylabel using parentheses to supply its inputs; this line does not store a separate return value.
    axes[0].set_ylabel("Collision rate (%)")
    # Call axes[0].legend using parentheses to supply its inputs; this line does not store a separate return value.
    axes[0].legend(fontsize=8)
    # Save a figure for the local results page; this does not affect agent behavior.
    # Call fig.savefig using parentheses to supply its inputs; this line does not store a separate return value.
    fig.savefig(output / "risk_analysis_frontier.png", dpi=180)
    # Call plt.close using parentheses to supply its inputs; this line does not store a separate return value.
    plt.close(fig)
    # = unpacks the right-hand result into (fig, axes), matching values to names in order.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    # for takes each item from zip(axes, [1.0, 1.5, 2.0]) in turn; (axis, density) holds the current item.
    for axis, density in zip(axes, [1.0, 1.5, 2.0]):
        # Call axis.plot using parentheses to supply its inputs; this line does not store a separate return value.
        axis.plot(
            # Inside this continued call, name=value supplies an option by name rather than by its position.
            [0, 1], [0, 1], "--", color="#7a8697", label="Nominal quantile fraction"
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
        # for takes each item from POLICIES.items() in turn; (policy, name) holds the current item.
        for policy, name in POLICIES.items():
            # = stores a result in row; call next using parentheses to supply its inputs.
            row = next(
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                row for row in data["calibration"][policy] if row["density"] == density
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            )
            # = stores a result in taus; use square brackets to select part of row.
            taus = row["taus"]
            # = unpacks the right-hand result into (low, high), matching values to names in order.
            low, high = np.array(row["coverage_lower"]), np.array(row["coverage_upper"])
            # Call axis.plot using parentheses to supply its inputs; this line does not store a separate return value.
            axis.plot(taus, (low + high) / 2, color=COLORS[name], label=name)
            # Call axis.fill_between using parentheses to supply its inputs; this line does not store a separate return value.
            axis.fill_between(taus, low, high, color=COLORS[name], alpha=0.15)
        # Call axis.set using parentheses to supply its inputs; this line does not store a separate return value.
        axis.set(
            # Inside this continued call, name=value supplies an option by name rather than by its position.
            title=f"Density {density:g}",
            # Inside this continued call, name=value supplies an option by name rather than by its position.
            xlabel="Predicted quantile fraction τ",
            # Inside this continued call, name=value supplies an option by name rather than by its position.
            xlim=(0, 1),
            # Inside this continued call, name=value supplies an option by name rather than by its position.
            ylim=(0, 1),
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
        # Call axis.grid using parentheses to supply its inputs; this line does not store a separate return value.
        axis.grid(alpha=0.15)
    # Call axes[0].set_ylabel using parentheses to supply its inputs; this line does not store a separate return value.
    axes[0].set_ylabel("Empirical quantile-compatibility bracket")
    # Call axes[0].legend using parentheses to supply its inputs; this line does not store a separate return value.
    axes[0].legend(fontsize=7, loc="lower right")
    # Save a figure for the local results page; this does not affect agent behavior.
    # Call fig.savefig using parentheses to supply its inputs; this line does not store a separate return value.
    fig.savefig(output / "risk_analysis_calibration.png", dpi=180)
    # Call plt.close using parentheses to supply its inputs; this line does not store a separate return value.
    plt.close(fig)
    # = unpacks the right-hand result into (fig, axes), matching values to names in order.
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), constrained_layout=True)
    # for takes each item from zip(axes, [1.0, 1.5, 2.0]) in turn; (axis, density) holds the current item.
    for axis, density in zip(axes, [1.0, 1.5, 2.0]):
        # for takes each item from enumerate(POLICIES.items()) in turn; (index, (policy, name)) holds the current item.
        for index, (policy, name) in enumerate(POLICIES.items()):
            # = stores a result in row; call next using parentheses to supply its inputs.
            row = next(
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                row for row in data["calibration"][policy] if row["density"] == density
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            )
            # = unpacks the right-hand result into (low, high), matching values to names in order.
            low, high = row["mean_prediction_bias_bounds"]
            # = stores a result in middle; apply / to these operands; numbers are divided, while Path objects join folder names.
            middle = (low + high) / 2
            # Call axis.errorbar using parentheses to supply its inputs; this line does not store a separate return value.
            axis.errorbar(
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                middle,
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                index,
                # Inside this continued call, name=value supplies an option by name rather than by its position.
                xerr=[[middle - low], [high - middle]],
                # Inside this continued call, name=value supplies an option by name rather than by its position.
                fmt="o",
                # Inside this continued call, name=value supplies an option by name rather than by its position.
                capsize=5,
                # Inside this continued call, name=value supplies an option by name rather than by its position.
                color=COLORS[name],
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            )
        # Call axis.axvline using parentheses to supply its inputs; this line does not store a separate return value.
        axis.axvline(0, color="#7a8697", linestyle="--")
        # Call axis.set using parentheses to supply its inputs; this line does not store a separate return value.
        axis.set(
            # Inside this continued call, name=value supplies an option by name rather than by its position.
            title=f"Density {density:g}",
            # Inside this continued call, name=value supplies an option by name rather than by its position.
            xlabel="Mean prediction minus realized return",
            # Inside this continued call, name=value supplies an option by name rather than by its position.
            yticks=[0, 1, 2],
            # Inside this continued call, name=value supplies an option by name rather than by its position.
            yticklabels=["Mean", "Lower 25%", "Governed"],
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
        # Call axis.grid using parentheses to supply its inputs; this line does not store a separate return value.
        axis.grid(alpha=0.15)
    # Save a figure for the local results page; this does not affect agent behavior.
    # Call fig.savefig using parentheses to supply its inputs; this line does not store a separate return value.
    fig.savefig(output / "risk_analysis_bias.png", dpi=180)
    # Call plt.close using parentheses to supply its inputs; this line does not store a separate return value.
    plt.close(fig)
    # = stores a result in calibration; collect or generate one result per item; the "for" part names the item being processed.
    calibration = {
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        policy: [
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            {
                # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                key: row[key]
                # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                for key in [
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    "density",
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    "mean_prediction_bias_bounds",
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    "censored_fraction",
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    "quantile_crossing_fraction",
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    "minimum_mean_absolute_compatibility_gap",
                # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
                ]
            # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
            }
            # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
            for row in rows
        # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
        ]
        # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
        for policy, rows in data["calibration"].items()
    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
    }
    # = stores a result in payload; build a dictionary, which stores values under named keys.
    payload = {
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "groups": data["groups"],
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "frontier": data["risk_frontier"],
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "calibration": calibration,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "failures": data["failure_diagnostics"],
    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
    }
    # = stores a result in nominal; use square brackets to select part of data['paired_comparisons'].
    nominal = data["paired_comparisons"][0]
    # = stores a result in document; call (Path(__file__).parent / 'report_risk_analysis.html').read_text using parentheses to supply its inputs.
    document = (Path(__file__).parent / "report_risk_analysis.html").read_text()
    # = stores a result in replacements; build a dictionary, which stores values under named keys.
    replacements = {
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "__PAYLOAD__": json.dumps(payload).replace("<", "\\u003c"),
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "__DELTA__": f'{-100*nominal["crashed"]["mean_difference"]:.1f}',
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "__SPEED__": f'{abs(nominal["mean_speed_m_s"]["mean_difference"]):.2f}',
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "__INTERVAL__": " to ".join(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            f"{100*value:.1f}"
            # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
            for value in nominal["crashed"]["exploratory_95_percent_interval"]
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        ),
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "__CAP__": f'{data["selection"]["selected_cap_m_s"]:g}',
    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
    }
    # for takes each item from replacements.items() in turn; (key, value) holds the current item.
    for key, value in replacements.items():
        # = stores a result in document; call document.replace using parentheses to supply its inputs.
        document = document.replace(key, value)
    # Call (output / 'index.html').write_text using parentheses to supply its inputs; this line does not store a separate return value.
    (output / "index.html").write_text(document)
    # Call print using parentheses to supply its inputs; this line does not store a separate return value.
    print(output / "index.html")
