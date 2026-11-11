"""Turn saved evaluation files into graphs and a local HTML report.

This is presentation code. It reads existing results and does not improve a
policy or run an extra experiment. Matplotlib draws graphs; HTML describes the page.

Original implementation notes:
Static figures and a portable HTML experiment viewer.
"""

import csv  # Make csv available for this file's calculations and workflow.
import html  # Make html available for this file's calculations and workflow.
import json  # Make json available for this file's calculations and workflow.
from pathlib import Path  # Make Path available for this file's calculations and workflow.

import numpy as np  # Make numpy available for this file's calculations and workflow.

from .metrics import wilson_interval  # Make wilson_interval available for this file's calculations and workflow.


def report(args):  # Build graphs and an HTML page from saved result files.
    """Build graphs and an HTML page from saved result files.
    The arrays below hold measured results; plotting does not change the models.
    HTML strings describe page layout, while Python inserts the saved numbers.
    Keep primary comparisons separate from exploratory comparisons in the report.
    """
    import matplotlib  # Make matplotlib available for this file's calculations and workflow.

    matplotlib.use("Agg")  # Matplotlib.use applied to 'agg'.
    import matplotlib.pyplot as plt  # Make matplotlib.pyplot available for this file's calculations and workflow.

    plt.rcParams.update(  # Plt.rcparams.update applied to the named settings or measurements assembled below.
        {  # Include the named settings or measurements assembled below in the surrounding operation.
            "figure.facecolor": "#f6f8fb",  # Record 'figure.facecolor' as '#f6f8fb'.
            "axes.facecolor": "white",  # Record 'axes.facecolor' as 'white'.
            "axes.spines.top": False,  # Record 'axes.spines.top' as False.
            "axes.spines.right": False,  # Record 'axes.spines.right' as False.
            "font.size": 11,  # Record 'font.size' as 11.
            "axes.titlesize": 13,  # Record 'axes.titlesize' as 13.
        }  # Finish assembling the grouped values above.
    )  # Finish assembling plt.rcParams.update inputs.
    root = Path(args.results)  # Set root to Path applied to args.results.
    output = Path(args.out)  # Set output to Path applied to args.out.
    output.mkdir(parents=True, exist_ok=True)  # The result of output.mkdir.
    evaluations = []  # Set evaluations to .
    for path in sorted(root.glob("eval_*.json")):  # Process each path from sorted applied to root.glob applied to 'eval_*.json'.
        data = json.loads(path.read_text())  # Set data to json.loads applied to the result of path.read text.
        if "provenance" in data:  # Use this branch when 'provenance' belongs to data.
            evaluations.append((path.stem.removeprefix("eval_"), data))  # Add path.stem.removeprefix applied to 'eval_', data to evaluations.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)  # Set fig, axes to plt.subplots applied to 1, 2.
    for path in sorted(root.glob("*/training.csv")):  # Process each path from sorted applied to root.glob applied to '*/training.csv'.
        with path.open() as file:  # Keep this temporary context active during the enclosed work, then clean it up.
            rows = list(csv.DictReader(file))  # Set rows to list applied to csv.DictReader applied to file.
        if not rows:  # Use this branch when not rows.
            continue  # Skip this case and move on to the next loop item.
        steps = np.array([float(row["step"]) for row in rows])  # Set steps to the entries of row selected by 'step' as a floating-point value for each row in rows as a numerical array/tensor for batch calculations.
        returns = np.array([float(row["return"]) for row in rows])  # Set returns to the entries of row selected by 'return' as a floating-point value for each row in rows as a numerical array/tensor for batch calculations.
        crashes = np.array([float(row["crashed"]) for row in rows])  # Set crashes to the entries of row selected by 'crashed' as a floating-point value for each row in rows as a numerical array/tensor for batch calculations.
        window = min(50, len(rows))  # Set window to min applied to 50, the number of entries in rows.
        kernel = np.ones(window) / window  # Set kernel to (np.ones applied to window divided by window).
        axes[0].plot(  # Axes[0].plot applied to the entries of steps selected by window - 1:, np.convolve applied to returns, kernel, 'valid'.
            steps[window - 1 :],  # Include the entries of steps selected by window - 1: in the surrounding operation.
            np.convolve(returns, kernel, "valid"),  # Include np.convolve applied to returns, kernel, 'valid' in the surrounding operation.
            label=path.parent.name,  # Include path.parent.name in the surrounding operation.
        )  # Finish assembling axes[0].plot inputs.
        axes[1].plot(  # Axes[1].plot applied to the entries of steps selected by window - 1:, np.convolve applied to crashes, kernel, 'valid'.
            steps[window - 1 :],  # Include the entries of steps selected by window - 1: in the surrounding operation.
            np.convolve(crashes, kernel, "valid"),  # Include np.convolve applied to crashes, kernel, 'valid' in the surrounding operation.
            label=path.parent.name,  # Include path.parent.name in the surrounding operation.
        )  # Finish assembling axes[1].plot inputs.
    for ax in axes:  # Process each ax from axes.
        ax.set_xlabel("Environment interactions")  # Ax.set xlabel applied to 'environment interactions'.
        ax.grid(alpha=0.18)  # The result of ax.grid.
        if ax.lines:  # Use this branch when ax.lines.
            ax.legend(fontsize=8)  # The result of ax.legend.
    axes[0].set(  # The result of axes[0].set.
        title="Training return · rolling 50 episodes", ylabel="Undiscounted return"  # Include 'Training return · rolling 50 episodes', 'Undiscounted return' in the surrounding operation.
    )  # Finish assembling axes[0].set inputs.
    axes[1].set(  # The result of axes[1].set.
        title="Exploratory training collisions",  # Include 'Exploratory training collisions' in the surrounding operation.
        ylabel="Collision fraction",  # Include 'Collision fraction' in the surrounding operation.
        ylim=(0, 1),  # Include 0, 1 in the surrounding operation.
    )  # Finish assembling axes[1].set inputs.
    fig.savefig(output / "learning_curves.png", dpi=160)  # Save a figure for the local results page; this does not affect agent behavior.
    plt.close(fig)  # Plt.close applied to fig.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)  # Set fig, axes to plt.subplots applied to 1, 3.
    for label, data in evaluations:  # Process each label, data from evaluations.
        summary = data["summary"]  # Set summary to the entries of data selected by 'summary'.
        density = [row["density"] for row in summary]  # Set the traffic-density setting to the entries of row selected by 'density' for each row in summary.
        rates = np.array([row["crash_rate"] for row in summary])  # Set rates to the entries of row selected by 'crash_rate' for each row in summary as a numerical array/tensor for batch calculations.
        intervals = np.array(  # Set intervals to wilson interval applied to round applied to row['crash rate'] * row['episodes'], the entries of row selected by 'episodes' for each row in summary as a numerical array/tensor for batch calculations.
            [  # Include wilson interval applied to round applied to (row['crash rate'] times row['episodes']), the entries of row selected by 'episodes' for each row in summary in the surrounding operation.
                wilson_interval(  # Include wilson interval applied to round applied to (the entries of row selected by 'crash_rate' times the entries of row selected by 'episodes'), the entries of row selected by 'episodes' in the surrounding operation.
                    round(row["crash_rate"] * row["episodes"]), row["episodes"]  # Include round applied to (the entries of row selected by 'crash_rate' times the entries of row selected by 'episodes'), the entries of row selected by 'episodes' in the surrounding operation.
                )  # Finish assembling wilson interval inputs.
                for row in summary  # Include row, summary in the surrounding operation.
            ]  # Finish assembling np.array inputs.
        )  # Finish assembling np.array inputs.
        axes[0].errorbar(  # Axes[0].errorbar applied to the traffic-density setting, rates.
            density,  # Include the traffic-density setting in the surrounding operation.
            rates,  # Include rates in the surrounding operation.
            yerr=np.maximum(  # Include the elementwise larger values from 0, np.stack applied to (rates minus intervals[:, 0]), (intervals[:, 1] minus rates) in the surrounding operation.
                0, np.stack([rates - intervals[:, 0], intervals[:, 1] - rates])  # Include 0, np.stack applied to (rates minus the entries of intervals selected by (:, 0)), (the entries of intervals selected by (:, 1) minus rates) in the surrounding operation.
            ),  # Finish assembling np.maximum inputs.
            fmt="o-",  # Include 'o-' in the surrounding operation.
            capsize=3,  # Include 3 in the surrounding operation.
            label=label,  # Include label in the surrounding operation.
        )  # Finish assembling axes[0].errorbar inputs.
        axes[1].plot(  # Axes[1].plot applied to the traffic-density setting, the entries of row selected by 'mean_speed_m_s' for each row in summary, 'o-'.
            density, [row["mean_speed_m_s"] for row in summary], "o-", label=label  # Include the traffic-density setting, the entries of row selected by 'mean_speed_m_s' for each row in summary, 'o-', label in the surrounding operation.
        )  # Finish assembling axes[1].plot inputs.
        axes[2].plot(  # Axes[2].plot applied to the entries of row selected by 'mean_speed_m_s' for each row in summary, the entries of row selected by 'crash_rate' for each row in summary, 'o-'.
            [row["mean_speed_m_s"] for row in summary],  # Include the entries of row selected by 'mean_speed_m_s' for each row in summary in the surrounding operation.
            [row["crash_rate"] for row in summary],  # Include the entries of row selected by 'crash_rate' for each row in summary in the surrounding operation.
            "o-",  # Include 'o-' in the surrounding operation.
            label=label,  # Include label in the surrounding operation.
        )  # Finish assembling axes[2].plot inputs.
    axes[0].set(  # The result of axes[0].set.
        title="Held-out collision rate",  # Include 'Held-out collision rate' in the surrounding operation.
        xlabel="Traffic density multiplier",  # Include 'Traffic density multiplier' in the surrounding operation.
        ylabel="Collision fraction",  # Include 'Collision fraction' in the surrounding operation.
        ylim=(-0.03, 1.03),  # Include negative 0.03, 1.03 in the surrounding operation.
    )  # Finish assembling axes[0].set inputs.
    axes[1].set(  # The result of axes[1].set.
        title="Held-out driving speed",  # Include 'Held-out driving speed' in the surrounding operation.
        xlabel="Traffic density multiplier",  # Include 'Traffic density multiplier' in the surrounding operation.
        ylabel="Mean speed (m/s)",  # Include 'Mean speed (m/s)' in the surrounding operation.
    )  # Finish assembling axes[1].set inputs.
    axes[2].set(  # The result of axes[2].set.
        title="Speed–collision tradeoff",  # Include 'Speed–collision tradeoff' in the surrounding operation.
        xlabel="Mean speed (m/s)",  # Include 'Mean speed (m/s)' in the surrounding operation.
        ylabel="Collision fraction",  # Include 'Collision fraction' in the surrounding operation.
        ylim=(-0.03, 1.03),  # Include negative 0.03, 1.03 in the surrounding operation.
    )  # Finish assembling axes[2].set inputs.
    for ax in axes:  # Process each ax from axes.
        ax.grid(alpha=0.18)  # The result of ax.grid.
        if ax.lines:  # Use this branch when ax.lines.
            ax.legend(fontsize=7)  # The result of ax.legend.
    fig.savefig(output / "evaluation.png", dpi=160)  # Save a figure for the local results page; this does not affect agent behavior.
    plt.close(fig)  # Plt.close applied to fig.
    table_rows = []  # Set table rows to .
    for label, data in evaluations:  # Process each label, data from evaluations.
        for row in data["summary"]:  # Process each row from the entries of data selected by 'summary'.
            values = [  # Set values to the listed settings or measurements in their recorded order.
                html.escape(label),  # Include html.escape applied to label in the surrounding operation.
                row["density"],  # Include the entries of row selected by 'density' in the surrounding operation.
                row["episodes"],  # Include the entries of row selected by 'episodes' in the surrounding operation.
                f'{100 * row["crash_rate"]:.1f}%',  # Include f"{100 * row['crash rate']:.1f}%" in the surrounding operation.
                f'{row["mean_speed_m_s"]:.1f}',  # Include f"{row['mean speed m s']:.1f}" in the surrounding operation.
                f'{row["mean_return"]:.2f}',  # Include f"{row['mean return']:.2f}" in the surrounding operation.
                f'{row["worst_10_percent_mean_return"]:.2f}',  # Include f"{row['worst 10 percent mean return']:.2f}" in the surrounding operation.
                f'{row["mean_distance_m"]:.0f}',  # Include f"{row['mean distance m']:.0f}" in the surrounding operation.
            ]  # Finish assembling the grouped values above.
            table_rows.append(  # Add (('<tr>' plus ''.join applied to (f'<td>{v}</td>' for v in values)) plus '</tr>') to table rows.
                "<tr>" + "".join(f"<td>{v}</td>" for v in values) + "</tr>"  # Include (('<tr>' plus ''.join applied to f'<td>{v}</td>' for each v in values) plus '</tr>') in the surrounding operation.
            )  # Finish assembling table rows.append inputs.
    gifs = sorted(root.glob("*.gif"))  # Set gifs to sorted applied to root.glob applied to '*.gif'.
    video_cards = "".join(  # Set video cards to ''.join applied to f'<article><h3>{html.escape(path.stem)}</h3><img src="../{html.escape(path.name)}" alt="Driving rollout"></article>' for each path in gifs.
        f'<article><h3>{html.escape(path.stem)}</h3><img src="../{html.escape(path.name)}" alt="Driving rollout"></article>'  # Include f'<article><h3>{html.escape(path.stem)}</h3><img src="../{html.escape(path.name)}" alt="Driving rollout"></article>' in the surrounding operation.
        for path in gifs  # Include path, gifs in the surrounding operation.
    )  # Finish assembling ''.join inputs.
    document = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Risk-aware driving · experiment lab</title><style>
body{margin:0;background:#0d1423;color:#e9eef7;font:16px/1.6 system-ui,sans-serif}
main{max-width:1180px;margin:auto;padding:52px 28px}h1{font-size:44px;line-height:1.15;max-width:850px}
.eyebrow{color:#62dbbf;letter-spacing:.12em;font-size:13px;font-weight:700}.intro{color:#acbad0;max-width:850px}
section{margin-top:42px}.figure{width:100%;background:white;border-radius:12px}table{border-collapse:collapse;width:100%;font-size:14px}
td,th{text-align:left;padding:12px;border-bottom:1px solid #283348}th{color:#8ddfcd}.scroll{overflow:auto}
.notice{padding:18px 22px;background:#172439;border-left:3px solid #62dbbf;border-radius:6px}
article{padding:12px 0}article img{width:100%;border-radius:10px}h3{font-size:16px}footer{color:#acbad0;margin-top:40px}
</style><main><div class="eyebrow">DEEP REINFORCEMENT LEARNING / initial experiment LAB</div>
<h1>How much speed should an agent trade for fewer crashes?</h1>
<p class="intro">A controlled comparison of DQN, distributional QR-DQN, and lower-tail action selection in simulated highway traffic.</p>
<p class="notice"><strong>Preliminary experiments.</strong> Individual training seeds and finite evaluation samples do not establish a reliable improvement. Lower-tail selection is a deployment heuristic; its return predictions were trained for mean-greedy control. The heuristic comparator uses privileged simulator state. All policies operate in a simplified simulator.</p>
<section><h2>Held-out evaluation</h2><img class="figure" src="evaluation.png" alt="Collision rate and speed by traffic density">
<div class="scroll"><table><thead><tr><th>Policy</th><th>Density</th><th>Episodes</th><th>Crashes</th><th>Speed m/s</th><th>Return</th><th>Worst 10%</th><th>Distance m</th></tr></thead><tbody>"""
    document += "".join(table_rows) + "</tbody></table></div></section>"  # Increase document using (''.join applied to table rows plus '</tbody></table></div></section>').
    document += '<section><h2>Learning dynamics</h2><p>These training curves include exploratory actions; assess deployed policies using the held-out evaluation above.</p><img class="figure" src="learning_curves.png" alt="Training curves"></section>'  # Increase document using the report text.
    document += (  # Increase document using ((the report text plus video cards) plus '</section>').
        "<section><h2>Rollout replays</h2><p>Each replay uses the first evaluation seed, selected before seeing outcomes.</p>"  # Include ((the report text plus video cards) plus '</section>') in the surrounding operation.
        + video_cards  # Include video cards in the surrounding operation.
        + "</section>"  # Include '</section>' in the surrounding operation.
    )  # Finish assembling the grouped values above.
    document += "<footer>Collision error bars are 95% Wilson intervals across evaluation episodes, conditional on the evaluated policy. They do not measure training-seed uncertainty. Units: speed in meters/second; distance is traveled path length. Worst 10% is the empirical lower-tail mean of undiscounted episode returns. Episode averages give each episode equal weight. Raw evaluation JSON includes per-episode results, seeds, checkpoint hashes, and package versions.</footer></main></html>"  # Increase document using the report text.
    (output / "index.html").write_text(document)  # (output / 'index.html').write text applied to document.
    print(f"Wrote {output / 'index.html'}")  # Print applied to f"wrote {output / 'index.html'}".
