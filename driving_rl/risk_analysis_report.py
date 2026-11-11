"""Create the risk analysis results page.

Read saved numbers, calculate clearly labeled summaries, draw figures, and write
an HTML page. This file presents evidence; it does not train or select models.
Long HTML strings are page templates. The reinforcement-learning calculations
live in agents.py and experiment.py, not inside the page layout.

Original implementation notes:
Scientific figures and an interactive viewer for risk analysis development results.
"""

import json  # Make json available for this file's calculations and workflow.
from pathlib import Path  # Make Path available for this file's calculations and workflow.

import numpy as np  # Make numpy available for this file's calculations and workflow.

COLORS = {  # Set COLORS to the named settings or measurements assembled below.
    "QR-DQN mean": "#9563c2",  # Record 'QR-DQN mean' as '#9563c2'.
    "QR-DQN lower 25%": "#159c87",  # Record 'QR-DQN lower 25%' as '#159c87'.
    "Same QR-DQN + governor": "#df9144",  # Record 'Same QR-DQN + governor' as '#df9144'.
}  # Finish assembling the grouped values above.
POLICIES = {  # Set POLICIES to the named settings or measurements assembled below.
    "mean": "QR-DQN mean",  # Record 'mean' as 'QR-DQN mean'.
    "tail25": "QR-DQN lower 25%",  # Record 'tail25' as 'QR-DQN lower 25%'.
    "governed": "Same QR-DQN + governor",  # Record 'governed' as 'Same QR-DQN + governor'.
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
    data = json.loads((root / "risk_analysis/summary.json").read_text())  # Set data to json.loads applied to the result of (root / 'risk analysis/summary.json').read text.
    output = root / "report"  # Set output to the path to 'report' inside root.
    output.mkdir(exist_ok=True)  # The result of output.mkdir.
    plt.rcParams.update(  # Plt.rcparams.update applied to the named settings or measurements assembled below.
        {  # Include the named settings or measurements assembled below in the surrounding operation.
            "figure.facecolor": "#f5f7fb",  # Record 'figure.facecolor' as '#f5f7fb'.
            "axes.facecolor": "white",  # Record 'axes.facecolor' as 'white'.
            "axes.spines.top": False,  # Record 'axes.spines.top' as False.
            "axes.spines.right": False,  # Record 'axes.spines.right' as False.
            "font.size": 10,  # Record 'font.size' as 10.
            "axes.titlesize": 12,  # Record 'axes.titlesize' as 12.
        }  # Finish assembling the grouped values above.
    )  # Finish assembling plt.rcParams.update inputs.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)  # Set fig, axes to plt.subplots applied to 1, 3.
    for name, rows in data["groups"].items():  # Process each name, rows from the result of data['groups'].items.
        x = [row["density"] for row in rows]  # Set x to the entries of row selected by 'density' for each row in rows.
        for axis, metric, scale in [  # Process each axis, metric, scale from the entries of axes selected by 0, 'crash_rate', 100, the entries of axes selected by 1, 'mean_speed_m_s', 1.
            (axes[0], "crash_rate", 100),  # Include the entries of axes selected by 0, 'crash_rate', 100 in the surrounding operation.
            (axes[1], "mean_speed_m_s", 1),  # Include the entries of axes selected by 1, 'mean_speed_m_s', 1 in the surrounding operation.
        ]:  # Finish assembling the grouped values above.
            axis.plot(  # Axis.plot applied to x, (the entries of row selected by metric times scale) for each row in rows, 'o-'.
                x,  # Include x in the surrounding operation.
                [row[metric] * scale for row in rows],  # Include (the entries of row selected by metric times scale) for each row in rows in the surrounding operation.
                "o-",  # Include 'o-' in the surrounding operation.
                color=COLORS[name],  # Include the entries of COLORS selected by name in the surrounding operation.
                label=name,  # Include name in the surrounding operation.
            )  # Finish assembling axis.plot inputs.
            for row in rows:  # Process each row from rows.
                axis.scatter(  # Axis.scatter applied to (the entries of row selected by 'density' times 3), (the entries of seed selected by metric times scale) for each seed in the entries of row selected by 'per_seed'.
                    [row["density"]] * 3,  # Include (the entries of row selected by 'density' times 3) in the surrounding operation.
                    [seed[metric] * scale for seed in row["per_seed"]],  # Include (the entries of seed selected by metric times scale) for each seed in the entries of row selected by 'per_seed' in the surrounding operation.
                    color=COLORS[name],  # Include the entries of COLORS selected by name in the surrounding operation.
                    alpha=0.35,  # Include 0.35 in the surrounding operation.
                    s=16,  # Include 16 in the surrounding operation.
                )  # Finish assembling axis.scatter inputs.
    comparisons = data["paired_comparisons"]  # Set comparisons to the entries of data selected by 'paired_comparisons'.
    x = [row["density"] for row in comparisons]  # Set x to the entries of row selected by 'density' for each row in comparisons.
    y = np.array([row["crashed"]["mean_difference"] * 100 for row in comparisons])  # Set y to (the entries of row['crashed'] selected by 'mean_difference' times 100) for each row in comparisons as a numerical array/tensor for batch calculations.
    bounds = (  # Set bounds to (the entries of row['crashed'] selected by 'exploratory_95_percent_interval' for each row in comparisons as a numerical array/tensor for batch calculations times 100).
        np.array(  # Include (the entries of row['crashed'] selected by 'exploratory_95_percent_interval' for each row in comparisons as a numerical array/tensor for batch calculations times 100) in the surrounding operation.
            [row["crashed"]["exploratory_95_percent_interval"] for row in comparisons]  # Include the entries of the entries of row selected by 'crashed' selected by 'exploratory_95_percent_interval' for each row in comparisons in the surrounding operation.
        )  # Finish assembling np.array inputs.
        * 100  # Include 100 in the surrounding operation.
    )  # Finish assembling the grouped values above.
    axes[2].errorbar(  # Axes[2].errorbar applied to x, y.
        x,  # Include x in the surrounding operation.
        y,  # Include y in the surrounding operation.
        yerr=np.maximum(0, np.stack([y - bounds[:, 0], bounds[:, 1] - y])),  # Include the elementwise larger values from 0, np.stack applied to (y minus bounds[:, 0]), (bounds[:, 1] minus y) in the surrounding operation.
        fmt="o-",  # Include 'o-' in the surrounding operation.
        capsize=5,  # Include 5 in the surrounding operation.
        color="#159c87",  # Include '#159c87' in the surrounding operation.
    )  # Finish assembling axes[2].errorbar inputs.
    axes[2].axhline(0, color="#7a8697", linestyle="--")  # Axes[2].axhline applied to 0.
    axes[0].set(  # The result of axes[0].set.
        title="Same models, different deployment",  # Include 'Same models, different deployment' in the surrounding operation.
        ylabel="Collision rate (%)",  # Include 'Collision rate (%)' in the surrounding operation.
        ylim=(-3, 103),  # Include negative 3, 103 in the surrounding operation.
    )  # Finish assembling axes[0].set inputs.
    axes[1].set(  # The result of axes[1].set.
        title="Check the remaining speed mismatch", ylabel="Average speed (m/s)"  # Include 'Check the remaining speed mismatch', 'Average speed (m/s)' in the surrounding operation.
    )  # Finish assembling axes[1].set inputs.
    axes[2].set(  # The result of axes[2].set.
        title="Cautious minus governed · paired",  # Include 'Cautious minus governed · paired' in the surrounding operation.
        ylabel="Collision difference (percentage points)",  # Include 'Collision difference (percentage points)' in the surrounding operation.
    )  # Finish assembling axes[2].set inputs.
    for axis in axes:  # Process each axis from axes.
        axis.set_xlabel("Traffic density")  # Axis.set xlabel applied to 'traffic density'.
        axis.grid(alpha=0.15)  # The result of axis.grid.
    axes[0].legend(fontsize=7, loc="lower right")  # The result of axes[0].legend.
    fig.savefig(output / "risk_analysis_control.png", dpi=180)  # Save a figure for the local results page; this does not affect agent behavior.
    plt.close(fig)  # Plt.close applied to fig.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True)  # Set fig, axes to plt.subplots applied to 1, 3.
    for axis, density in zip(axes, [1.0, 1.5, 2.0]):  # Process each axis, the traffic-density setting from matching entries grouped across axes, 1.0, 1.5, 2.0.
        rows = [  # Set rows to next applied to row for each row in the entries of item selected by 'summary', keeping only cases where row['density'] equals the traffic-density setting for each item in the entries of data selected by 'risk_frontier'.
            next(row for row in item["summary"] if row["density"] == density)  # Include next applied to row for each row in the entries of item selected by 'summary', keeping only cases where the entries of row selected by 'density' equals the traffic-density setting in the surrounding operation.
            for item in data["risk_frontier"]  # Include item, the entries of data selected by 'risk_frontier' in the surrounding operation.
        ]  # Finish assembling the grouped values above.
        speeds = [row["mean_speed_m_s"] for row in rows]  # Set speeds to the entries of row selected by 'mean_speed_m_s' for each row in rows.
        crashes = [row["crash_rate"] * 100 for row in rows]  # Set crashes to (the entries of row selected by 'crash_rate' times 100) for each row in rows.
        axis.plot(  # Axis.plot applied to speeds, crashes, 'o-'.
            speeds, crashes, "o-", color="#159c87", label="Quantile tail selector"  # Include speeds, crashes, 'o-', '#159c87', 'Quantile tail selector' in the surrounding operation.
        )  # Finish assembling axis.plot inputs.
        for item, speed, crash in zip(data["risk_frontier"], speeds, crashes):  # Process each item, speed, crash from matching entries grouped across the entries of data selected by 'risk_frontier', speeds, crashes.
            axis.annotate(  # Axis.annotate applied to f"α={item['alpha']:g}", speed, crash.
                f'α={item["alpha"]:g}',  # Include f"α={item['alpha']:g}" in the surrounding operation.
                (speed, crash),  # Include speed, crash in the surrounding operation.
                xytext=(4, -12),  # Include 4, negative 12 in the surrounding operation.
                textcoords="offset points",  # Include 'offset points' in the surrounding operation.
                fontsize=8,  # Include 8 in the surrounding operation.
            )  # Finish assembling axis.annotate inputs.
        control = next(  # Set control to next applied to row for each row in the entries of the entries of data selected by 'groups' selected by 'Same QR-DQN + governor', keeping only cases where the entries of row selected by 'density' equals the traffic-density setting.
            row  # Include row in the surrounding operation.
            for row in data["groups"]["Same QR-DQN + governor"]  # Include row, the entries of the entries of data selected by 'groups' selected by 'Same QR-DQN + governor' in the surrounding operation.
            if row["density"] == density  # Include the entries of row selected by 'density' equals the traffic-density setting in the surrounding operation.
        )  # Finish assembling next inputs.
        axis.scatter(  # Axis.scatter applied to the entries of control selected by 'mean_speed_m_s', (the entries of control selected by 'crash_rate' times 100).
            control["mean_speed_m_s"],  # Include the entries of control selected by 'mean_speed_m_s' in the surrounding operation.
            control["crash_rate"] * 100,  # Include (the entries of control selected by 'crash_rate' times 100) in the surrounding operation.
            color="#df9144",  # Include '#df9144' in the surrounding operation.
            marker="*",  # Include '*' in the surrounding operation.
            s=160,  # Include 160 in the surrounding operation.
            label="Mean + fixed governor",  # Include 'Mean + fixed governor' in the surrounding operation.
        )  # Finish assembling axis.scatter inputs.
        axis.set(  # The result of axis.set.
            title=f"Density {density:g}", xlabel="Mean speed (m/s)", ylim=(-3, 108)  # Include f'Density {density:g}', 'Mean speed (m/s)', negative 3, 108 in the surrounding operation.
        )  # Finish assembling axis.set inputs.
        axis.grid(alpha=0.15)  # The result of axis.grid.
    axes[0].set_ylabel("Collision rate (%)")  # Axes[0].set ylabel applied to 'collision rate (%)'.
    axes[0].legend(fontsize=8)  # The result of axes[0].legend.
    fig.savefig(output / "risk_analysis_frontier.png", dpi=180)  # Save a figure for the local results page; this does not affect agent behavior.
    plt.close(fig)  # Plt.close applied to fig.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)  # Set fig, axes to plt.subplots applied to 1, 3.
    for axis, density in zip(axes, [1.0, 1.5, 2.0]):  # Process each axis, the traffic-density setting from matching entries grouped across axes, 1.0, 1.5, 2.0.
        axis.plot(  # Axis.plot applied to 0, 1, 0, 1, '--'.
            [0, 1], [0, 1], "--", color="#7a8697", label="Nominal quantile fraction"  # Include 0, 1, 0, 1, '--', '#7a8697', 'Nominal quantile fraction' in the surrounding operation.
        )  # Finish assembling axis.plot inputs.
        for policy, name in POLICIES.items():  # Process each policy, name from the result of POLICIES.items.
            row = next(  # Set row to next applied to row for each row in the entries of the entries of data selected by 'calibration' selected by policy, keeping only cases where the entries of row selected by 'density' equals the traffic-density setting.
                row for row in data["calibration"][policy] if row["density"] == density  # Include row, row, the entries of the entries of data selected by 'calibration' selected by policy, the entries of row selected by 'density' equals the traffic-density setting in the surrounding operation.
            )  # Finish assembling next inputs.
            taus = row["taus"]  # Set the assigned quantile fractions to the entries of row selected by 'taus'.
            low, high = np.array(row["coverage_lower"]), np.array(row["coverage_upper"])  # Set low, high to the entries of row selected by 'coverage_lower' as a numerical array/tensor for batch calculations, the entries of row selected by 'coverage_upper' as a numerical array/tensor for batch calculations.
            axis.plot(taus, (low + high) / 2, color=COLORS[name], label=name)  # Axis.plot applied to the assigned quantile fractions, ((low plus high) divided by 2).
            axis.fill_between(taus, low, high, color=COLORS[name], alpha=0.15)  # Axis.fill between applied to the assigned quantile fractions, low, high.
        axis.set(  # The result of axis.set.
            title=f"Density {density:g}",  # Include f'Density {density:g}' in the surrounding operation.
            xlabel="Predicted quantile fraction τ",  # Include 'Predicted quantile fraction τ' in the surrounding operation.
            xlim=(0, 1),  # Include 0, 1 in the surrounding operation.
            ylim=(0, 1),  # Include 0, 1 in the surrounding operation.
        )  # Finish assembling axis.set inputs.
        axis.grid(alpha=0.15)  # The result of axis.grid.
    axes[0].set_ylabel("Empirical quantile-compatibility bracket")  # Axes[0].set ylabel applied to 'empirical quantile-compatibility bracket'.
    axes[0].legend(fontsize=7, loc="lower right")  # The result of axes[0].legend.
    fig.savefig(output / "risk_analysis_calibration.png", dpi=180)  # Save a figure for the local results page; this does not affect agent behavior.
    plt.close(fig)  # Plt.close applied to fig.
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), constrained_layout=True)  # Set fig, axes to plt.subplots applied to 1, 3.
    for axis, density in zip(axes, [1.0, 1.5, 2.0]):  # Process each axis, the traffic-density setting from matching entries grouped across axes, 1.0, 1.5, 2.0.
        for index, (policy, name) in enumerate(POLICIES.items()):  # Process each index, policy, name from enumerate applied to the result of POLICIES.items.
            row = next(  # Set row to next applied to row for each row in the entries of the entries of data selected by 'calibration' selected by policy, keeping only cases where the entries of row selected by 'density' equals the traffic-density setting.
                row for row in data["calibration"][policy] if row["density"] == density  # Include row, row, the entries of the entries of data selected by 'calibration' selected by policy, the entries of row selected by 'density' equals the traffic-density setting in the surrounding operation.
            )  # Finish assembling next inputs.
            low, high = row["mean_prediction_bias_bounds"]  # Set low, high to the entries of row selected by 'mean_prediction_bias_bounds'.
            middle = (low + high) / 2  # Set middle to ((low plus high) divided by 2).
            axis.errorbar(  # Axis.errorbar applied to middle, index.
                middle,  # Include middle in the surrounding operation.
                index,  # Include index in the surrounding operation.
                xerr=[[middle - low], [high - middle]],  # Include (middle minus low), (high minus middle) in the surrounding operation.
                fmt="o",  # Include 'o' in the surrounding operation.
                capsize=5,  # Include 5 in the surrounding operation.
                color=COLORS[name],  # Include the entries of COLORS selected by name in the surrounding operation.
            )  # Finish assembling axis.errorbar inputs.
        axis.axvline(0, color="#7a8697", linestyle="--")  # Axis.axvline applied to 0.
        axis.set(  # The result of axis.set.
            title=f"Density {density:g}",  # Include f'Density {density:g}' in the surrounding operation.
            xlabel="Mean prediction minus realized return",  # Include 'Mean prediction minus realized return' in the surrounding operation.
            yticks=[0, 1, 2],  # Include 0, 1, 2 in the surrounding operation.
            yticklabels=["Mean", "Lower 25%", "Governed"],  # Include 'Mean', 'Lower 25%', 'Governed' in the surrounding operation.
        )  # Finish assembling axis.set inputs.
        axis.grid(alpha=0.15)  # The result of axis.grid.
    fig.savefig(output / "risk_analysis_bias.png", dpi=180)  # Save a figure for the local results page; this does not affect agent behavior.
    plt.close(fig)  # Plt.close applied to fig.
    calibration = {  # Set calibration to the entries of row selected by key for each key in 'density', 'mean prediction bias bounds', 'censored fraction', 'quantile crossing fraction', 'minimum mean absolute compatibility gap' for each row in rows for each policy, rows in the result of data['calibration'].items.
        policy: [  # Include policy, the entries of row selected by key for each key in 'density', 'mean_prediction_bias_bounds', 'censored_fraction', 'quantile_crossing_fraction', 'minimum_mean_absolute_compatibility_gap' for each row in rows in the surrounding operation.
            {  # Include the entries of row selected by key for each key in 'density', 'mean_prediction_bias_bounds', 'censored_fraction', 'quantile_crossing_fraction', 'minimum_mean_absolute_compatibility_gap' in the surrounding operation.
                key: row[key]  # Include key, the entries of row selected by key in the surrounding operation.
                for key in [  # Include key, 'density', 'mean_prediction_bias_bounds', 'censored_fraction', 'quantile_crossing_fraction', 'minimum_mean_absolute_compatibility_gap' in the surrounding operation.
                    "density",  # Include 'density' in the surrounding operation.
                    "mean_prediction_bias_bounds",  # Include 'mean_prediction_bias_bounds' in the surrounding operation.
                    "censored_fraction",  # Include 'censored_fraction' in the surrounding operation.
                    "quantile_crossing_fraction",  # Include 'quantile_crossing_fraction' in the surrounding operation.
                    "minimum_mean_absolute_compatibility_gap",  # Include 'minimum_mean_absolute_compatibility_gap' in the surrounding operation.
                ]  # Finish assembling the grouped values above.
            }  # Finish assembling the grouped values above.
            for row in rows  # Include row, rows in the surrounding operation.
        ]  # Finish assembling the grouped values above.
        for policy, rows in data["calibration"].items()  # Include policy, rows, the result of data['calibration'].items in the surrounding operation.
    }  # Finish assembling the grouped values above.
    payload = {  # Set payload to the named settings or measurements assembled below.
        "groups": data["groups"],  # Record 'groups' as the entries of data selected by 'groups'.
        "frontier": data["risk_frontier"],  # Record 'frontier' as the entries of data selected by 'risk_frontier'.
        "calibration": calibration,  # Record 'calibration' as calibration.
        "failures": data["failure_diagnostics"],  # Record 'failures' as the entries of data selected by 'failure_diagnostics'.
    }  # Finish assembling the grouped values above.
    nominal = data["paired_comparisons"][0]  # Set nominal to the entries of the entries of data selected by 'paired_comparisons' selected by 0.
    document = (Path(__file__).parent / "report_risk_analysis.html").read_text()  # Set document to the result of (Path( file ).parent / 'report risk analysis.html').read text.
    replacements = {  # Set replacements to the named settings or measurements assembled below.
        "__PAYLOAD__": json.dumps(payload).replace("<", "\\u003c"),  # Record '__PAYLOAD__' as json.dumps(payload).replace applied to '<', '\\u003c'.
        "__DELTA__": f'{-100*nominal["crashed"]["mean_difference"]:.1f}',  # Record '__DELTA__' as f"{-100 * nominal['crashed']['mean difference']:.1f}".
        "__SPEED__": f'{abs(nominal["mean_speed_m_s"]["mean_difference"]):.2f}',  # Record '__SPEED__' as f"{abs(nominal['mean speed m s']['mean difference']):.2f}".
        "__INTERVAL__": " to ".join(  # Record '__INTERVAL__' as ' to '.join applied to f'{100 * value:.1f}' for each value in the entries of the entries of nominal selected by 'crashed' selected by 'exploratory_95_percent_interval'.
            f"{100*value:.1f}"  # Include f'{100 * value:.1f}' in the surrounding operation.
            for value in nominal["crashed"]["exploratory_95_percent_interval"]  # Include value, the entries of the entries of nominal selected by 'crashed' selected by 'exploratory_95_percent_interval' in the surrounding operation.
        ),  # Finish assembling ' to '.join inputs.
        "__CAP__": f'{data["selection"]["selected_cap_m_s"]:g}',  # Record '__CAP__' as f"{data['selection']['selected cap m s']:g}".
    }  # Finish assembling the grouped values above.
    for key, value in replacements.items():  # Process each key, value from the result of replacements.items.
        document = document.replace(key, value)  # Set document to document.replace applied to key, value.
    (output / "index.html").write_text(document)  # (output / 'index.html').write text applied to document.
    print(output / "index.html")  # Print applied to the path to 'index.html' inside output.
