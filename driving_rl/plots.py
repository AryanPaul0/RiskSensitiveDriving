"""Turn saved evaluation files into graphs and a local HTML report.

This is presentation code. It reads existing results and does not improve a
policy or run an extra experiment. Matplotlib draws graphs; HTML describes the page.

Original implementation notes:
Static figures and a portable HTML experiment viewer.
"""

# import makes these modules available; "as" gives a module a shorter local name.
import csv
# import makes these modules available; "as" gives a module a shorter local name.
import html
# import makes these modules available; "as" gives a module a shorter local name.
import json
# from pathlib import brings the named tools into this file; parentheses allow several lines.
from pathlib import Path

# import makes these modules available; "as" gives a module a shorter local name.
import numpy as np

# from metrics import brings the named tools into this file; parentheses allow several lines.
from .metrics import wilson_interval


# def defines report; parentheses list its inputs, and the colon starts its indented body.
def report(args):
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

    # Call plt.rcparams.update using parentheses to supply its inputs; this line does not store a separate return value.
    plt.rcParams.update(
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        {
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "figure.facecolor": "#f6f8fb",
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
    # = stores a result in root; call Path using parentheses to supply its inputs.
    root = Path(args.results)
    # = stores a result in output; call Path using parentheses to supply its inputs.
    output = Path(args.out)
    # Call output.mkdir using parentheses to supply its inputs; this line does not store a separate return value.
    output.mkdir(parents=True, exist_ok=True)
    # = stores a result in evaluations; build a list, keeping its items in order.
    evaluations = []
    # for takes each item from sorted(root.glob('eval_*.json')) in turn; path holds the current item.
    for path in sorted(root.glob("eval_*.json")):
        # = stores a result in data; call json.loads using parentheses to supply its inputs.
        data = json.loads(path.read_text())
        # if checks 'provenance' in data; only a true condition runs the indented block.
        if "provenance" in data:
            # Call evaluations.append using parentheses to supply its inputs; this line does not store a separate return value.
            evaluations.append((path.stem.removeprefix("eval_"), data))
    # = unpacks the right-hand result into (fig, axes), matching values to names in order.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    # for takes each item from sorted(root.glob('*/training.csv')) in turn; path holds the current item.
    for path in sorted(root.glob("*/training.csv")):
        # with enters a temporary context and exits it afterward, even if an error interrupts the block.
        with path.open() as file:
            # = stores a result in rows; call list using parentheses to supply its inputs.
            rows = list(csv.DictReader(file))
        # if checks not rows; only a true condition runs the indented block.
        if not rows:
            # continue skips the rest of this loop iteration and moves to its next item.
            continue
        # = stores a result in steps; call np.array using parentheses to supply its inputs.
        steps = np.array([float(row["step"]) for row in rows])
        # = stores a result in returns; call np.array using parentheses to supply its inputs.
        returns = np.array([float(row["return"]) for row in rows])
        # = stores a result in crashes; call np.array using parentheses to supply its inputs.
        crashes = np.array([float(row["crashed"]) for row in rows])
        # = stores a result in window; call min using parentheses to supply its inputs.
        window = min(50, len(rows))
        # = stores a result in kernel; apply / to these operands; numbers are divided, while Path objects join folder names.
        kernel = np.ones(window) / window
        # Call axes[0].plot using parentheses to supply its inputs; this line does not store a separate return value.
        axes[0].plot(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            steps[window - 1 :],
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            np.convolve(returns, kernel, "valid"),
            # Inside this continued call, name=value supplies an option by name rather than by its position.
            label=path.parent.name,
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
        # Call axes[1].plot using parentheses to supply its inputs; this line does not store a separate return value.
        axes[1].plot(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            steps[window - 1 :],
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            np.convolve(crashes, kernel, "valid"),
            # Inside this continued call, name=value supplies an option by name rather than by its position.
            label=path.parent.name,
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
    # for takes each item from axes in turn; ax holds the current item.
    for ax in axes:
        # Call ax.set_xlabel using parentheses to supply its inputs; this line does not store a separate return value.
        ax.set_xlabel("Environment interactions")
        # Call ax.grid using parentheses to supply its inputs; this line does not store a separate return value.
        ax.grid(alpha=0.18)
        # if checks ax.lines; only a true condition runs the indented block.
        if ax.lines:
            # Call ax.legend using parentheses to supply its inputs; this line does not store a separate return value.
            ax.legend(fontsize=8)
    # Call axes[0].set using parentheses to supply its inputs; this line does not store a separate return value.
    axes[0].set(
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        title="Training return · rolling 50 episodes", ylabel="Undiscounted return"
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # Call axes[1].set using parentheses to supply its inputs; this line does not store a separate return value.
    axes[1].set(
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        title="Exploratory training collisions",
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        ylabel="Collision fraction",
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        ylim=(0, 1),
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # Save a figure for the local results page; this does not affect agent behavior.
    # Call fig.savefig using parentheses to supply its inputs; this line does not store a separate return value.
    fig.savefig(output / "learning_curves.png", dpi=160)
    # Call plt.close using parentheses to supply its inputs; this line does not store a separate return value.
    plt.close(fig)
    # = unpacks the right-hand result into (fig, axes), matching values to names in order.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    # for takes each item from evaluations in turn; (label, data) holds the current item.
    for label, data in evaluations:
        # = stores a result in summary; use square brackets to select part of data.
        summary = data["summary"]
        # = stores a result in density; collect or generate one result per item; the "for" part names the item being processed.
        density = [row["density"] for row in summary]
        # = stores a result in rates; call np.array using parentheses to supply its inputs.
        rates = np.array([row["crash_rate"] for row in summary])
        # = stores a result in intervals; call np.array using parentheses to supply its inputs.
        intervals = np.array(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            [
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                wilson_interval(
                    # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                    round(row["crash_rate"] * row["episodes"]), row["episodes"]
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                )
                # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                for row in summary
            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
            ]
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
        # Call axes[0].errorbar using parentheses to supply its inputs; this line does not store a separate return value.
        axes[0].errorbar(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            density,
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            rates,
            # Inside this continued call, name=value supplies an option by name rather than by its position.
            yerr=np.maximum(
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                0, np.stack([rates - intervals[:, 0], intervals[:, 1] - rates])
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            ),
            # Inside this continued call, name=value supplies an option by name rather than by its position.
            fmt="o-",
            # Inside this continued call, name=value supplies an option by name rather than by its position.
            capsize=3,
            # Inside this continued call, name=value supplies an option by name rather than by its position.
            label=label,
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
        # Call axes[1].plot using parentheses to supply its inputs; this line does not store a separate return value.
        axes[1].plot(
            # Inside this continued call, name=value supplies an option by name rather than by its position.
            density, [row["mean_speed_m_s"] for row in summary], "o-", label=label
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
        # Call axes[2].plot using parentheses to supply its inputs; this line does not store a separate return value.
        axes[2].plot(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            [row["mean_speed_m_s"] for row in summary],
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            [row["crash_rate"] for row in summary],
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "o-",
            # Inside this continued call, name=value supplies an option by name rather than by its position.
            label=label,
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
    # Call axes[0].set using parentheses to supply its inputs; this line does not store a separate return value.
    axes[0].set(
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        title="Held-out collision rate",
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        xlabel="Traffic density multiplier",
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        ylabel="Collision fraction",
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        ylim=(-0.03, 1.03),
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # Call axes[1].set using parentheses to supply its inputs; this line does not store a separate return value.
    axes[1].set(
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        title="Held-out driving speed",
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        xlabel="Traffic density multiplier",
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        ylabel="Mean speed (m/s)",
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # Call axes[2].set using parentheses to supply its inputs; this line does not store a separate return value.
    axes[2].set(
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        title="Speed–collision tradeoff",
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        xlabel="Mean speed (m/s)",
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        ylabel="Collision fraction",
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        ylim=(-0.03, 1.03),
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # for takes each item from axes in turn; ax holds the current item.
    for ax in axes:
        # Call ax.grid using parentheses to supply its inputs; this line does not store a separate return value.
        ax.grid(alpha=0.18)
        # if checks ax.lines; only a true condition runs the indented block.
        if ax.lines:
            # Call ax.legend using parentheses to supply its inputs; this line does not store a separate return value.
            ax.legend(fontsize=7)
    # Save a figure for the local results page; this does not affect agent behavior.
    # Call fig.savefig using parentheses to supply its inputs; this line does not store a separate return value.
    fig.savefig(output / "evaluation.png", dpi=160)
    # Call plt.close using parentheses to supply its inputs; this line does not store a separate return value.
    plt.close(fig)
    # = stores a result in table_rows; build a list, keeping its items in order.
    table_rows = []
    # for takes each item from evaluations in turn; (label, data) holds the current item.
    for label, data in evaluations:
        # for takes each item from data['summary'] in turn; row holds the current item.
        for row in data["summary"]:
            # = stores a result in values; build a list, keeping its items in order.
            values = [
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                html.escape(label),
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                row["density"],
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                row["episodes"],
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                f'{100 * row["crash_rate"]:.1f}%',
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                f'{row["mean_speed_m_s"]:.1f}',
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                f'{row["mean_return"]:.2f}',
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                f'{row["worst_10_percent_mean_return"]:.2f}',
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                f'{row["mean_distance_m"]:.0f}',
            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
            ]
            # Call table_rows.append using parentheses to supply its inputs; this line does not store a separate return value.
            table_rows.append(
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                "<tr>" + "".join(f"<td>{v}</td>" for v in values) + "</tr>"
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            )
    # = stores a result in gifs; call sorted using parentheses to supply its inputs.
    gifs = sorted(root.glob("*.gif"))
    # = stores a result in video_cards; call ''.join using parentheses to supply its inputs.
    video_cards = "".join(
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        f'<article><h3>{html.escape(path.stem)}</h3><img src="../{html.escape(path.name)}" alt="Driving rollout"></article>'
        # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
        for path in gifs
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
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
    # The combined operator updates document using its previous value.
    document += "".join(table_rows) + "</tbody></table></div></section>"
    # The combined operator updates document using its previous value.
    document += '<section><h2>Learning dynamics</h2><p>These training curves include exploratory actions; assess deployed policies using the held-out evaluation above.</p><img class="figure" src="learning_curves.png" alt="Training curves"></section>'
    # The combined operator updates document using its previous value.
    document += (
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "<section><h2>Rollout replays</h2><p>Each replay uses the first evaluation seed, selected before seeing outcomes.</p>"
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + video_cards
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + "</section>"
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # The combined operator updates document using its previous value.
    document += "<footer>Collision error bars are 95% Wilson intervals across evaluation episodes, conditional on the evaluated policy. They do not measure training-seed uncertainty. Units: speed in meters/second; distance is traveled path length. Worst 10% is the empirical lower-tail mean of undiscounted episode returns. Episode averages give each episode equal weight. Raw evaluation JSON includes per-episode results, seeds, checkpoint hashes, and package versions.</footer></main></html>"
    # Call (output / 'index.html').write_text using parentheses to supply its inputs; this line does not store a separate return value.
    (output / "index.html").write_text(document)
    # Call print using parentheses to supply its inputs; this line does not store a separate return value.
    print(f"Wrote {output / 'index.html'}")
