"""Create the density coverage results page.

Read saved numbers, calculate clearly labeled summaries, draw figures, and write
an HTML page. This file presents evidence; it does not train or select models.
Long HTML strings are page templates. The reinforcement-learning calculations
live in agents.py and experiment.py, not inside the page layout.

Original implementation notes:
Reports for equal-budget traffic-density coverage experiments.
"""

# import makes these modules available; "as" gives a module a shorter local name.
import json
# from pathlib import brings the named tools into this file; parentheses allow several lines.
from pathlib import Path
# import makes these modules available; "as" gives a module a shorter local name.
import shutil
# import makes these modules available; "as" gives a module a shorter local name.
import numpy as np
# import makes these modules available; "as" gives a module a shorter local name.
import matplotlib

# Call matplotlib.use using parentheses to supply its inputs; this line does not store a separate return value.
matplotlib.use("Agg")
# import makes these modules available; "as" gives a module a shorter local name.
import matplotlib.pyplot as plt
# from analysis import brings the named tools into this file; parentheses allow several lines.
from .analysis import paired_matrix, crossed_bootstrap, summarize_group
# from calibration import brings the named tools into this file; parentheses allow several lines.
from .calibration import summarize_rows
# from experiment import brings the named tools into this file; parentheses allow several lines.
from .experiment import save_json

# = stores a result in ROOT; use square brackets to select part of Path(__file__).resolve().parents.
ROOT = Path(__file__).resolve().parents[1]
# = stores a result in STAGE; apply / to these operands; numbers are divided, while Path objects join folder names.
STAGE = ROOT / "results/density_coverage"
# = stores a result in POLICIES; build a dictionary, which stores values under named keys.
POLICIES = {
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "mean": "Mean selector",
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "tail25": "Lower-quarter selector",
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "governed": "Mean + 25 m/s governor",
# } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
}


# def defines read; parentheses list its inputs, and the colon starts its indented body.
def read(p):
    # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
    """Read saved JSON data into ordinary Python dictionaries and lists."""
    # return ends this function and sends its result back to the code that called it.
    return json.loads(p.read_text())


# def defines load; parentheses list its inputs, and the colon starts its indented body.
def load(regime, policy, block):
    """Read the result files belonging to one comparison group. The group controls
    which saved model and decision rule are represented; it does not train a model.
    """
    # return ends this function and sends its result back to the code that called it.
    return [
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        read(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            ROOT / f"results/training_budget/eval_qrdqn_{s}_40000_{policy}.json"
            # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
            if regime == "fixed" and block == "main"
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            else STAGE / f"eval_{regime}_{policy}_seed{s}_{block}.json"
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
        # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
        for s in range(3)
    # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
    ]


# def defines table; parentheses list its inputs, and the colon starts its indented body.
def table(headers, rows):
    """Build the HTML for a table of saved numbers. Headers label columns; each
    row supplies values in the same order. This changes presentation, not the data.
    """
    # return ends this function and sends its result back to the code that called it.
    return (
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        '<div class="scroll"><table><thead><tr>'
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + "".join("<th>" + str(v) + "</th>" for v in headers)
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + "</tr></thead><tbody>"
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + "".join(
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            "<tr>" + "".join("<td>" + str(v) + "</td>" for v in row) + "</tr>"
            # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
            for row in rows
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + "</tbody></table></div>"
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )


# def defines report; parentheses list its inputs, and the colon starts its indented body.
def report():
    """Build graphs and an HTML page from saved result files.
    The arrays below hold measured results; plotting does not change the models.
    HTML strings describe page layout, while Python inserts the saved numbers.
    Keep primary comparisons separate from exploratory comparisons in the report.
    """
    # = stores a result in viewer; apply / to these operands; numbers are divided, while Path objects join folder names.
    viewer = ROOT / "results/report"
    # = stores a result in backup; apply / to these operands; numbers are divided, while Path objects join folder names.
    backup = ROOT / "results/training_budget-report"
    # if checks not backup.exists(); only a true condition runs the indented block.
    if not backup.exists():
        # Call shutil.copytree using parentheses to supply its inputs; this line does not store a separate return value.
        shutil.copytree(viewer, backup)
    # = stores a result in groups; build a dictionary, which stores values under named keys.
    groups = {}
    # = stores a result in contrasts; build a list, keeping its items in order.
    contrasts = []
    # = stores a result in calibration; build a list, keeping its items in order.
    calibration = []
    # = stores a result in exposure; build a list, keeping its items in order.
    exposure = []
    # = stores a result in controls; build a list, keeping its items in order.
    controls = []
    # for takes each item from [('main', [1.0, 1.5, 2.0]), ('transfer', [1.25, 2.5])] in turn; (block, densities) holds the current item.
    for block, densities in [("main", [1.0, 1.5, 2.0]), ("transfer", [1.25, 2.5])]:
        # = stores a result in groups[block]; build a dictionary, which stores values under named keys.
        groups[block] = {}
        # for takes each item from ['fixed', 'mixed'] in turn; regime holds the current item.
        for regime in ["fixed", "mixed"]:
            # = stores a result in groups[block][regime]; collect or generate one result per item; the "for" part names the item being processed.
            groups[block][regime] = {
                # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                p: summarize_group(load(regime, p, block)) for p in POLICIES
            # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
            }
        # for takes each item from POLICIES in turn; policy holds the current item.
        for policy in POLICIES:
            # for takes each item from densities in turn; density holds the current item.
            for density in densities:
                # = stores a result in item; build a dictionary, which stores values under named keys.
                item = {"block": block, "policy": policy, "density": density}
                # Compare one measured quantity at a time, keeping the scenarios paired.
                # for takes each item from ['crashed', 'mean_speed_m_s', 'distance_m', 'return'] in turn; metric holds the current item.
                for metric in ["crashed", "mean_speed_m_s", "distance_m", "return"]:
                    # = unpacks the right-hand result into (_, matrix), matching values to names in order.
                    _, matrix = paired_matrix(
                        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                        load("mixed", policy, block),
                        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                        load("fixed", policy, block),
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        density,
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        metric,
                    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                    )
                    # = stores a result in item[metric]; call crossed_bootstrap using parentheses to supply its inputs.
                    item[metric] = crossed_bootstrap(matrix)
                # Call contrasts.append using parentheses to supply its inputs; this line does not store a separate return value.
                contrasts.append(item)
        # for takes each item from ['fixed', 'mixed'] in turn; regime holds the current item.
        for regime in ["fixed", "mixed"]:
            # for takes each item from densities in turn; density holds the current item.
            for density in densities:
                # = stores a result in item; build a dictionary, which stores values under named keys.
                item = {"block": block, "regime": regime, "density": density}
                # Compare one measured quantity at a time, keeping the scenarios paired.
                # for takes each item from ['crashed', 'mean_speed_m_s'] in turn; metric holds the current item.
                for metric in ["crashed", "mean_speed_m_s"]:
                    # = unpacks the right-hand result into (_, matrix), matching values to names in order.
                    _, matrix = paired_matrix(
                        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                        load(regime, "tail25", block),
                        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                        load(regime, "governed", block),
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        density,
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        metric,
                    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                    )
                    # = stores a result in item[metric]; call crossed_bootstrap using parentheses to supply its inputs.
                    item[metric] = crossed_bootstrap(matrix)
                # Call controls.append using parentheses to supply its inputs; this line does not store a separate return value.
                controls.append(item)
    # for takes each item from ['fixed', 'mixed'] in turn; regime holds the current item.
    for regime in ["fixed", "mixed"]:
        # = stores a result in folder; choose the value before "if" when its condition is true, otherwise the value after "else".
        folder = ROOT / "results/training_budget" if regime == "fixed" else STAGE
        # for takes each item from ['mean', 'tail25'] in turn; policy holds the current item.
        for policy in ["mean", "tail25"]:
            # = stores a result in data; collect or generate one result per item; the "for" part names the item being processed.
            data = [
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                read(folder / f"calibration_{policy}_seed{s}.json") for s in range(3)
            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
            ]
            # for takes each item from [1.0, 1.5, 2.0] in turn; density holds the current item.
            for density in [1.0, 1.5, 2.0]:
                # = stores a result in rows; collect or generate one result per item; the "for" part names the item being processed.
                rows = [
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    r for d in data for r in d["episodes"] if r["density"] == density
                # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
                ]
                # Call calibration.append using parentheses to supply its inputs; this line does not store a separate return value.
                calibration.append(
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    {
                        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                        "regime": regime,
                        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                        "policy": policy,
                        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                        "density": density,
                        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
                        **summarize_rows(rows),
                    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
                    }
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                )
    # for takes each item from range(3) in turn; seed holds the current item.
    for seed in range(3):
        # = stores a result in d; call read using parentheses to supply its inputs.
        d = read(STAGE / f"qrdqn_seed{seed}/summary.json")
        # if checks sum(d['steps_by_density'].values()) != 40000; only a true condition runs the indented block.
        if sum(d["steps_by_density"].values()) != 40000:
            # raise signals an error and stops this normal path, so invalid data are not silently accepted.
            raise ValueError("Training exposure does not sum to budget")
        # Call exposure.append using parentheses to supply its inputs; this line does not store a separate return value.
        exposure.append({"seed": seed, **d})
    # = stores a result in result; build a dictionary, which stores values under named keys.
    result = {
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "design": read(STAGE / "design.json"),
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "groups": groups,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "mixed_minus_fixed": contrasts,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "tail_minus_governed": controls,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "calibration": calibration,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "training_exposure": exposure,
    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
    }
    # Call save_json using parentheses to supply its inputs; this line does not store a separate return value.
    save_json(STAGE / "summary.json", result)
    # Call plt.rcparams.update using parentheses to supply its inputs; this line does not store a separate return value.
    plt.rcParams.update(
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # = unpacks the right-hand result into (fig, axes), matching values to names in order.
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    # for takes each item from enumerate(POLICIES) in turn; (col, policy) holds the current item.
    for col, policy in enumerate(POLICIES):
        # for takes each item from [('fixed', 'C0'), ('mixed', 'C1')] in turn; (regime, color) holds the current item.
        for regime, color in [("fixed", "C0"), ("mixed", "C1")]:
            # = stores a result in rows; use square brackets to select part of groups['main'][regime].
            rows = groups["main"][regime][policy]
            # = stores a result in x; collect or generate one result per item; the "for" part names the item being processed.
            x = [r["density"] for r in rows]
            # for takes each item from [(0, 'crash_rate', 100), (1, 'mean_speed_m_s', 1)] in turn; (row, metric, scale) holds the current item.
            for row, metric, scale in [
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                (0, "crash_rate", 100),
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                (1, "mean_speed_m_s", 1),
            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
            ]:
                # = stores a result in ax; use square brackets to select part of axes.
                ax = axes[row, col]
                # Call ax.plot using parentheses to supply its inputs; this line does not store a separate return value.
                ax.plot(
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    x,
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    [r[metric] * scale for r in rows],
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    "-o",
                    # Inside this continued call, name=value supplies an option by name rather than by its position.
                    color=color,
                    # Inside this continued call, name=value supplies an option by name rather than by its position.
                    label=regime,
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                )
                # for takes each item from range(3) in turn; s holds the current item.
                for s in range(3):
                    # Call ax.plot using parentheses to supply its inputs; this line does not store a separate return value.
                    ax.plot(
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        x,
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        [r["per_seed"][s][metric] * scale for r in rows],
                        # Inside this continued call, name=value supplies an option by name rather than by its position.
                        color=color,
                        # Inside this continued call, name=value supplies an option by name rather than by its position.
                        lw=0.8,
                        # Inside this continued call, name=value supplies an option by name rather than by its position.
                        alpha=0.25,
                    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                    )
                # Call ax.grid using parentheses to supply its inputs; this line does not store a separate return value.
                ax.grid(alpha=0.2)
                # Call ax.set_xticks using parentheses to supply its inputs; this line does not store a separate return value.
                ax.set_xticks(x)
                # Call ax.set_xlabel using parentheses to supply its inputs; this line does not store a separate return value.
                ax.set_xlabel("Evaluation density")
        # Call axes[0, col].set_title using parentheses to supply its inputs; this line does not store a separate return value.
        axes[0, col].set_title(POLICIES[policy])
        # Call axes[0, col].set_ylim using parentheses to supply its inputs; this line does not store a separate return value.
        axes[0, col].set_ylim(0, 105)
        # Call axes[0, col].legend using parentheses to supply its inputs; this line does not store a separate return value.
        axes[0, col].legend()
    # Call axes[0, 0].set_ylabel using parentheses to supply its inputs; this line does not store a separate return value.
    axes[0, 0].set_ylabel("Episodes with collision (%)")
    # Call axes[1, 0].set_ylabel using parentheses to supply its inputs; this line does not store a separate return value.
    axes[1, 0].set_ylabel("Mean episode speed (m/s)")
    # Call fig.suptitle using parentheses to supply its inputs; this line does not store a separate return value.
    fig.suptitle("Equal 40k-step budgets; faint lines show individual training seeds")
    # Call fig.tight_layout using parentheses to supply its inputs; this line does not store a separate return value.
    fig.tight_layout()
    # Save a figure for the local results page; this does not affect agent behavior.
    # Call fig.savefig using parentheses to supply its inputs; this line does not store a separate return value.
    fig.savefig(viewer / "density_coverage_performance.png", dpi=150)
    # Call plt.close using parentheses to supply its inputs; this line does not store a separate return value.
    plt.close(fig)
    # = unpacks the right-hand result into (fig, axes), matching values to names in order.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    # for takes each item from zip(axes, ['mean', 'tail25']) in turn; (ax, policy) holds the current item.
    for ax, policy in zip(axes, ["mean", "tail25"]):
        # for takes each item from [('fixed', 'C0', -0.035), ('mixed', 'C1', 0.035)] in turn; (regime, color, offset) holds the current item.
        for regime, color, offset in [("fixed", "C0", -0.035), ("mixed", "C1", 0.035)]:
            # = stores a result in rows; collect or generate one result per item; the "for" part names the item being processed.
            rows = [
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                r
                # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                for r in calibration
                # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                if r["regime"] == regime and r["policy"] == policy
            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
            ]
            # = stores a result in lo; call np.array using parentheses to supply its inputs.
            lo = np.array([r["mean_prediction_bias_bounds"][0] for r in rows])
            # = stores a result in hi; call np.array using parentheses to supply its inputs.
            hi = np.array([r["mean_prediction_bias_bounds"][1] for r in rows])
            # = stores a result in mid; apply / to these operands; numbers are divided, while Path objects join folder names.
            mid = (lo + hi) / 2
            # Call ax.errorbar using parentheses to supply its inputs; this line does not store a separate return value.
            ax.errorbar(
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                np.array([1.0, 1.5, 2.0]) + offset,
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                mid,
                # Inside this continued call, name=value supplies an option by name rather than by its position.
                yerr=[mid - lo, hi - mid],
                # Inside this continued call, name=value supplies an option by name rather than by its position.
                fmt="o-",
                # Inside this continued call, name=value supplies an option by name rather than by its position.
                color=color,
                # Inside this continued call, name=value supplies an option by name rather than by its position.
                label=regime,
                # Inside this continued call, name=value supplies an option by name rather than by its position.
                capsize=4,
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            )
        # Call ax.axhline using parentheses to supply its inputs; this line does not store a separate return value.
        ax.axhline(0, color="black", lw=0.8)
        # Call ax.set_xticks using parentheses to supply its inputs; this line does not store a separate return value.
        ax.set_xticks([1, 1.5, 2])
        # Call ax.set_xlabel using parentheses to supply its inputs; this line does not store a separate return value.
        ax.set_xlabel("Evaluation density")
        # Call ax.set_title using parentheses to supply its inputs; this line does not store a separate return value.
        ax.set_title(POLICIES[policy])
        # Call ax.grid using parentheses to supply its inputs; this line does not store a separate return value.
        ax.grid(alpha=0.2)
        # Call ax.legend using parentheses to supply its inputs; this line does not store a separate return value.
        ax.legend()
    # Call axes[0].set_ylabel using parentheses to supply its inputs; this line does not store a separate return value.
    axes[0].set_ylabel("Prediction minus discounted return")
    # Call fig.suptitle using parentheses to supply its inputs; this line does not store a separate return value.
    fig.suptitle("Bias bounds account for censoring, not sampling uncertainty")
    # Call fig.tight_layout using parentheses to supply its inputs; this line does not store a separate return value.
    fig.tight_layout()
    # Save a figure for the local results page; this does not affect agent behavior.
    # Call fig.savefig using parentheses to supply its inputs; this line does not store a separate return value.
    fig.savefig(viewer / "density_coverage_bias.png", dpi=150)
    # Call plt.close using parentheses to supply its inputs; this line does not store a separate return value.
    plt.close(fig)
    # = unpacks the right-hand result into (fig, axes), matching values to names in order.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # for takes each item from zip(axes, ['completed_episodes_by_density', 'steps_by_density'], ['Completed episodes', 'Environm... in turn; (ax, key, title) holds the current item.
    for ax, key, title in zip(
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        axes,
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        ["completed_episodes_by_density", "steps_by_density"],
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        ["Completed episodes", "Environment transitions"],
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    ):
        # = stores a result in bottom; call np.zeros using parentheses to supply its inputs.
        bottom = np.zeros(3)
        # for takes each item from [1.0, 1.5, 2.0] in turn; d holds the current item.
        for d in [1.0, 1.5, 2.0]:
            # = stores a result in total; call np.array using parentheses to supply its inputs.
            total = np.array([sum(r[key].values()) for r in exposure])
            # = stores a result in v; multiply the operands; NumPy and PyTorch arrays do this element by element.
            v = np.array([r[key][str(d)] for r in exposure]) / total * 100
            # Call ax.bar using parentheses to supply its inputs; this line does not store a separate return value.
            ax.bar(range(3), v, bottom=bottom, label=f"Density {d:g}")
            # The combined operator updates bottom using its previous value.
            bottom += v
        # Call ax.set_xticks using parentheses to supply its inputs; this line does not store a separate return value.
        ax.set_xticks(range(3), ["Seed 0", "Seed 1", "Seed 2"])
        # Call ax.set_ylim using parentheses to supply its inputs; this line does not store a separate return value.
        ax.set_ylim(0, 100)
        # Call ax.set_title using parentheses to supply its inputs; this line does not store a separate return value.
        ax.set_title(title)
        # Call ax.set_ylabel using parentheses to supply its inputs; this line does not store a separate return value.
        ax.set_ylabel("Share (%)")
    # Call axes[1].legend using parentheses to supply its inputs; this line does not store a separate return value.
    axes[1].legend(loc="upper left", bbox_to_anchor=(1, 1))
    # Call fig.suptitle using parentheses to supply its inputs; this line does not store a separate return value.
    fig.suptitle("Uniform episode sampling does not produce equal transition shares")
    # Call fig.tight_layout using parentheses to supply its inputs; this line does not store a separate return value.
    fig.tight_layout()
    # Save a figure for the local results page; this does not affect agent behavior.
    # Call fig.savefig using parentheses to supply its inputs; this line does not store a separate return value.
    fig.savefig(viewer / "density_coverage_exposure.png", dpi=150)
    # Call plt.close using parentheses to supply its inputs; this line does not store a separate return value.
    plt.close(fig)
    # = stores a result in primary; call next using parentheses to supply its inputs.
    primary = next(
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        r
        # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
        for r in contrasts
        # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
        if r["block"] == "main" and r["policy"] == "tail25" and r["density"] == 2.0
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # = stores a result in value; use square brackets to select part of primary.
    value = primary["crashed"]
    # = stores a result in interval; collect or generate one result per item; the "for" part names the item being processed.
    interval = [100 * x for x in value["exploratory_95_percent_interval"]]
    # = stores a result in primary_text; evaluate the expression on the right before using its result.
    primary_text = f"At density 2, the lower-quarter selector's mixed-minus-fixed collision difference is {100*value['mean_difference']:+.1f} percentage points; exploratory 95% interval [{interval[0]:+.1f}, {interval[1]:+.1f}]."
    # = stores a result in document; use quoted text as data; Python does not execute the words inside it.
    document = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Traffic-density coverage · Risk-sensitive driving</title><style>body{font:17px/1.6 system-ui;color:#18283d;background:#f4f7fb;margin:auto;max-width:1160px;padding:36px}h1{font-size:38px;line-height:1.2}section{background:white;padding:24px;margin:24px 0;border-radius:14px}img{width:100%;height:auto}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;text-align:left;border-bottom:1px solid #dde3eb}th{background:#eef3fa}a{color:#175bbc}summary{cursor:pointer}</style></head><body><p>DISTRIBUTIONAL RL / TRAFFIC COVERAGE</p><h1>Does broader training traffic improve driving?</h1><p>QR-DQN trained at density 1 versus a per-episode mixture of densities 1, 1.5 and 2. Each setting has three training seeds and 40,000 interactions per model. Rewards, architecture, optimizer and exploration schedule remain fixed.</p><p><a href="../training_budget-report/index.html">Training-budget comparison</a> · <a href="../risk_analysis-report/index.html">Interactive risk frontier and recorded driving</a> · <a href="../density_coverage/summary.json">Complete numerical results</a> · <a href="../density_coverage/design.json">Recorded experiment design</a></p>"""
    # The combined operator updates document using its previous value.
    document += (
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "<section><h2>Predeclared primary comparison</h2><p>"
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + primary_text
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + "</p><p>Negative differences favor mixture training. The selectors and 25 m/s governor are unchanged. This is a development comparison with only three training seeds; all confirmation scenarios remain reserved.</p></section>"
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )

    # def defines metric; parentheses list its inputs, and the colon starts its indented body.
    def metric(regime, policy, density, key):
        # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
        """Look up one already-calculated metric for the requested group and density."""
        # return ends this function and sends its result back to the code that called it.
        return next(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            r for r in groups["main"][regime][policy] if r["density"] == density
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )[key]

    # = stores a result in d; use the literal value 1.5.
    d = 1.5
    # = stores a result in coverage_text; evaluate the expression on the right before using its result.
    coverage_text = f"At density 1.5, cautious-selector collisions change from {100*metric('fixed','tail25',d,'crash_rate'):.1f}% to {100*metric('mixed','tail25',d,'crash_rate'):.1f}%, while speed changes from {metric('fixed','tail25',d,'mean_speed_m_s'):.2f} to {metric('mixed','tail25',d,'mean_speed_m_s'):.2f} m/s."
    # = stores a result in governor_text; evaluate the expression on the right before using its result.
    governor_text = f"The normal-traffic governor result worsens: {100*metric('fixed','governed',1.,'crash_rate'):.1f}% to {100*metric('mixed','governed',1.,'crash_rate'):.1f}% collisions. Broader training is not an improvement for every deployment rule."
    # The combined operator updates document using its previous value.
    document += (
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "<section><h2>Main tradeoffs</h2><p>"
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + coverage_text
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + "</p><p>"
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + governor_text
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + "</p><p>Fewer observed collisions come with slower driving. The same-network governor comparisons retain residual speed differences, and uncertainty intervals remain exploratory. This study isolates coverage within QR-DQN; it does not compare DQN and QR-DQN under identical randomized training.</p></section>"
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # The combined operator updates document using its previous value.
    document += '<section><h2>Performance across covered densities</h2><p>Same evaluation scenario block for both training settings (10000–10029). Densities 1.5 and 2 were absent from fixed-density training and are included in mixed training. Faint curves show seed variation.</p><img src="density_coverage_performance.png"></section>'
    # for takes each item from [('main', 'Covered traffic settings'), ('transfer', 'Additional traffic settings')] in turn; (block, title) holds the current item.
    for block, title in [
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        ("main", "Covered traffic settings"),
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        ("transfer", "Additional traffic settings"),
    # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
    ]:
        # The combined operator updates document using its previous value.
        document += "<section><h2>" + title + "</h2>"
        # if checks block == 'transfer'; only a true condition runs the indented block.
        if block == "transfer":
            # The combined operator updates document using its previous value.
            document += "<p>Densities 1.25 and 2.5 use a separate development scenario block, 16000–16029. Density 1.25 is an intermediate value absent from the sampled mixture; density 2.5 is beyond both training ranges. These rows use different scenarios from the covered-settings table.</p>"
        # The combined operator updates document using its previous value.
        document += (
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            table(
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                [
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    "Training",
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    "Selector",
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    "Density",
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    "Collisions",
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    "Speed m/s",
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    "Distance m",
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    "Worst 10% return",
                # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
                ],
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                [
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    [
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        regime,
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        POLICIES[p],
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        r["density"],
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        f"{100*r['crash_rate']:.1f}%",
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        f"{r['mean_speed_m_s']:.2f}",
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        f"{r['mean_distance_m']:.0f}",
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        f"{r['worst_10_percent_mean_return']:.2f}",
                    # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
                    ]
                    # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                    for regime in ["fixed", "mixed"]
                    # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                    for p in POLICIES
                    # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                    for r in groups[block][regime][p]
                # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
                ],
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            )
            # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
            + "</section>"
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
    # The combined operator updates document using its previous value.
    document += (
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "<section><h2>Paired collision effects: mixed minus fixed</h2><p>Exploratory intervals use 5,000 crossed bootstrap replicates over training seed IDs and shared scenarios, preserving policy pairing. Matching seed IDs shares initialization and exploration streams, but not the simulated trajectories. Three seeds and repeated development evaluation limit inference.</p>"
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + table(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            [
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Block",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Selector",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Density",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Change pp",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Exploratory interval pp",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Per-seed changes pp",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Speed change m/s",
            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
            ],
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            [
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                [
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    r["block"],
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    POLICIES[r["policy"]],
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    r["density"],
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    f"{100*r['crashed']['mean_difference']:+.1f}",
                    # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                    str(
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        [
                            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                            round(100 * x, 1)
                            # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                            for x in r["crashed"]["exploratory_95_percent_interval"]
                        # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
                        ]
                    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                    ),
                    # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                    str(
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        [
                            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                            round(100 * x, 1)
                            # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                            for x in r["crashed"]["per_training_seed_differences"]
                        # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
                        ]
                    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                    ),
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    f"{r['mean_speed_m_s']['mean_difference']:+.2f}",
                # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
                ]
                # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                for r in contrasts
            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
            ],
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + "</section>"
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # The combined operator updates document using its previous value.
    document += (
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "<section><h2>Cautious decisions versus speed control</h2><p>Lower quarter minus the mean selector with the fixed 25 m/s governor, within the same networks. The governor threshold is not reselected for mixed training; check residual speed differences.</p>"
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + table(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            [
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Block",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Training",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Density",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Collision change pp",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Exploratory interval pp",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Speed change m/s",
            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
            ],
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            [
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                [
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    r["block"],
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    r["regime"],
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    r["density"],
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    f"{100*r['crashed']['mean_difference']:+.1f}",
                    # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                    str(
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        [
                            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                            round(100 * x, 1)
                            # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                            for x in r["crashed"]["exploratory_95_percent_interval"]
                        # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
                        ]
                    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                    ),
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    f"{r['mean_speed_m_s']['mean_difference']:+.2f}",
                # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
                ]
                # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                for r in controls
            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
            ],
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + "</section>"
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # The combined operator updates document using its previous value.
    document += (
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        '<section><h2>What the networks predict</h2><p>Same separate calibration scenarios (12000–12059) for both settings. The 30-step time cutoff right-censors continuing returns. Bounds retain surviving episodes; they are not statistical confidence intervals. Lower-quarter deployment changes the continuation policy relative to training.</p><img src="density_coverage_bias.png">'
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + table(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            [
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Training",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Selector",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Density",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Bias bounds",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Compatibility gap",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Adjacent crossings",
            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
            ],
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            [
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                [
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    r["regime"],
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    r["policy"],
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    r["density"],
                    # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                    str([round(x, 3) for x in r["mean_prediction_bias_bounds"]]),
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    f"{r['minimum_mean_absolute_compatibility_gap']:.3f}",
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    f"{100*r['quantile_crossing_fraction']:.1f}%",
                # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
                ]
                # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                for r in calibration
            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
            ],
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + "</section>"
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # The combined operator updates document using its previous value.
    document += '<section><h2>Same-scenario recorded driving</h2><p>Both lower-quarter policies use training seed 0, scenario seed 10000, density 2. The scenario was fixed before inspecting the new outcomes and is illustrative rather than a performance estimate.</p><div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px"><article><h3>Fixed-density training</h3><img src="../density_coverage/replay_fixed.gif"></article><article><h3>Mixed-density training</h3><img src="../density_coverage/replay_mixed.gif"></article></div></section>'
    # The combined operator updates document using its previous value.
    document += '<section><h2>Actual training exposure</h2><p>Density is sampled uniformly at episode reset with a separate seeded generator. Short dense-traffic episodes contribute fewer transitions. The density label is not added to the observation; the network still receives the same 40 inputs. Equal step budgets therefore do not imply equal episode counts or equal data by density.</p><img src="density_coverage_exposure.png"></section>'
    # The combined operator updates document using its previous value.
    document += "<section><h2>Reproduce and interpret</h2><p><code>python run_density_coverage.py --jobs 3</code> validates and reuses saved artifacts. <code>--stage evaluate</code> skips training; <code>--stage report</code> rebuilds the viewer. Run <code>python -m pytest -q</code> for checks.</p><p>Only traffic density changes during training. The simulator retains simplified vehicle interactions; the observation is partial. Quantile forecasts are not collision probabilities, and cautious deployment has no guaranteed safety or global CVaR optimality. Results concern sampled development scenarios and should not be presented as real-road validation.</p></section></body></html>"
    # Call (viewer / 'index.html').write_text using parentheses to supply its inputs; this line does not store a separate return value.
    (viewer / "index.html").write_text(document)
    # Call print using parentheses to supply its inputs; this line does not store a separate return value.
    print(primary_text, flush=True)
    # Call print using parentheses to supply its inputs; this line does not store a separate return value.
    print("Density coverage report saved", flush=True)


# Run the entry point only when this file is executed, not when it is imported.
# __name__ equals "__main__" when this file runs directly; this guard avoids running the script merely because another file imports it.
if __name__ == "__main__":
    # Call report using parentheses to supply its inputs; this line does not store a separate return value.
    report()
