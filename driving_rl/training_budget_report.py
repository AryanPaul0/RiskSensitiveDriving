"""Create the training budget results page.

Read saved numbers, calculate clearly labeled summaries, draw figures, and write
an HTML page. This file presents evidence; it does not train or select models.
Long HTML strings are page templates. The reinforcement-learning calculations
live in agents.py and experiment.py, not inside the page layout.

Original implementation notes:
Scientific report for the predeclared training-budget comparison.
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
from .analysis import paired_matrix, crossed_bootstrap
# from calibration import brings the named tools into this file; parentheses allow several lines.
from .calibration import summarize_rows
# from experiment import brings the named tools into this file; parentheses allow several lines.
from .experiment import save_json

# = stores a result in ROOT; use square brackets to select part of Path(__file__).resolve().parents.
ROOT = Path(__file__).resolve().parents[1]
# = stores a result in STAGE; apply / to these operands; numbers are divided, while Path objects join folder names.
STAGE = ROOT / "results/training_budget"
# = stores a result in LABELS; build a dictionary, which stores values under named keys.
LABELS = {
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "dqn_mean": "DQN",
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "qrdqn_mean": "QR mean",
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "qrdqn_tail25": "QR lower 25%",
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "qrdqn_governed": "QR + fixed governor",
# } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
}


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
    backup = ROOT / "results/risk_analysis-report"
    # if checks not backup.exists(); only a true condition runs the indented block.
    if not backup.exists():
        # Call shutil.copytree using parentheses to supply its inputs; this line does not store a separate return value.
        shutil.copytree(viewer, backup)
    # = stores a result in groups; build a dictionary, which stores values under named keys.
    groups = {}
    # = stores a result in summaries; build a list, keeping its items in order.
    summaries = []
    # for takes each item from LABELS in turn; name holds the current item.
    for name in LABELS:
        # = unpacks the right-hand result into (algo, policy), matching values to names in order.
        algo, policy = name.split("_", 1)
        # for takes each item from [20000, 40000] in turn; budget holds the current item.
        for budget in [20000, 40000]:
            # = stores a result in data; collect or generate one result per item; the "for" part names the item being processed.
            data = [
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                json.loads(
                    # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                    (STAGE / f"eval_{algo}_{s}_{budget}_{policy}.json").read_text()
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                )
                # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                for s in range(3)
            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
            ]
            # = stores a result in groups[name, budget]; evaluate the expression on the right before using its result.
            groups[name, budget] = data
            # for takes each item from [1.0, 1.5, 2.0] in turn; density holds the current item.
            for density in [1.0, 1.5, 2.0]:
                # = stores a result in per_seed; build a list, keeping its items in order.
                per_seed = []
                # for takes each item from data in turn; d holds the current item.
                for d in data:
                    # = stores a result in rows; collect or generate one result per item; the "for" part names the item being processed.
                    rows = [r for r in d["episodes"] if r["density"] == density]
                    # = stores a result in returns; call sorted using parentheses to supply its inputs.
                    returns = sorted(r["return"] for r in rows)
                    # Call per_seed.append using parentheses to supply its inputs; this line does not store a separate return value.
                    per_seed.append(
                        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                        dict(
                            # Inside this continued call, name=value supplies an option by name rather than by its position.
                            seed=d["provenance"]["training_seed"],
                            # Inside this continued call, name=value supplies an option by name rather than by its position.
                            crash_rate=np.mean([r["crashed"] for r in rows]),
                            # Inside this continued call, name=value supplies an option by name rather than by its position.
                            speed=np.mean([r["mean_speed_m_s"] for r in rows]),
                            # Inside this continued call, name=value supplies an option by name rather than by its position.
                            distance=np.mean([r["distance_m"] for r in rows]),
                            # Inside this continued call, name=value supplies an option by name rather than by its position.
                            mean_return=np.mean(returns),
                            # Inside this continued call, name=value supplies an option by name rather than by its position.
                            worst10=np.mean(returns[:3]),
                        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                        )
                    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                    )
                # Call summaries.append using parentheses to supply its inputs; this line does not store a separate return value.
                summaries.append(
                    # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                    dict(
                        # Inside this continued call, name=value supplies an option by name rather than by its position.
                        policy=name,
                        # Inside this continued call, name=value supplies an option by name rather than by its position.
                        budget=budget,
                        # Inside this continued call, name=value supplies an option by name rather than by its position.
                        density=density,
                        # Inside this continued call, name=value supplies an option by name rather than by its position.
                        per_seed=per_seed,
                        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
                        **{
                            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                            k: float(np.mean([r[k] for r in per_seed]))
                            # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                            for k in [
                                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                                "crash_rate",
                                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                                "speed",
                                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                                "distance",
                                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                                "mean_return",
                                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                                "worst10",
                            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
                            ]
                        # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
                        },
                    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                    )
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                )
    # = stores a result in comparisons; build a list, keeping its items in order.
    comparisons = []
    # for takes each item from LABELS in turn; name holds the current item.
    for name in LABELS:
        # for takes each item from [1.0, 1.5, 2.0] in turn; density holds the current item.
        for density in [1.0, 1.5, 2.0]:
            # = stores a result in row; call dict using parentheses to supply its inputs.
            row = dict(policy=name, density=density)
            # Compare one measured quantity at a time, keeping the scenarios paired.
            # for takes each item from ['crashed', 'mean_speed_m_s', 'distance_m', 'return'] in turn; metric holds the current item.
            for metric in ["crashed", "mean_speed_m_s", "distance_m", "return"]:
                # = unpacks the right-hand result into (_, matrix), matching values to names in order.
                _, matrix = paired_matrix(
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    groups[name, 40000], groups[name, 20000], density, metric
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                )
                # = stores a result in row[metric]; call crossed_bootstrap using parentheses to supply its inputs.
                row[metric] = crossed_bootstrap(matrix)
            # Call comparisons.append using parentheses to supply its inputs; this line does not store a separate return value.
            comparisons.append(row)
    # = stores a result in controls; build a list, keeping its items in order.
    controls = []
    # for takes each item from [1.0, 1.5, 2.0] in turn; density holds the current item.
    for density in [1.0, 1.5, 2.0]:
        # = stores a result in item; call dict using parentheses to supply its inputs.
        item = dict(density=density)
        # Compare one measured quantity at a time, keeping the scenarios paired.
        # for takes each item from ['crashed', 'mean_speed_m_s'] in turn; metric holds the current item.
        for metric in ["crashed", "mean_speed_m_s"]:
            # = unpacks the right-hand result into (_, matrix), matching values to names in order.
            _, matrix = paired_matrix(
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                groups["qrdqn_tail25", 40000],
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                groups["qrdqn_governed", 40000],
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
    # = stores a result in calibration; build a list, keeping its items in order.
    calibration = []
    # for takes each item from ['mean', 'tail25'] in turn; policy holds the current item.
    for policy in ["mean", "tail25"]:
        # for takes each item from [20000, 40000] in turn; budget holds the current item.
        for budget in [20000, 40000]:
            # = stores a result in folder; choose the value before "if" when its condition is true, otherwise the value after "else".
            folder = ROOT / "results/risk_analysis" if budget == 20000 else STAGE
            # = stores a result in data; collect or generate one result per item; the "for" part names the item being processed.
            data = [
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                json.loads((folder / f"calibration_{policy}_seed{s}.json").read_text())
                # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                for s in range(3)
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
                    # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                    dict(
                        # Inside this continued call, name=value supplies an option by name rather than by its position.
                        policy=policy,
                        # Inside this continued call, name=value supplies an option by name rather than by its position.
                        budget=budget,
                        # Inside this continued call, name=value supplies an option by name rather than by its position.
                        density=density,
                        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
                        **summarize_rows(rows),
                    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                    )
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                )
    # = stores a result in prefix; call json.loads using parentheses to supply its inputs.
    prefix = json.loads((STAGE / "prefix_verification.json").read_text())
    # = stores a result in reproduction; build a list, keeping its items in order.
    reproduction = []
    # for takes each item from ['dqn', 'qrdqn'] in turn; algo holds the current item.
    for algo in ["dqn", "qrdqn"]:
        # for takes each item from range(3) in turn; seed holds the current item.
        for seed in range(3):
            # for takes each item from ['mean'] if algo == 'dqn' else ['mean', 'tail25'] in turn; policy holds the current item.
            for policy in (["mean"] if algo == "dqn" else ["mean", "tail25"]):
                # = stores a result in original; call json.loads using parentheses to supply its inputs.
                original = json.loads(
                    # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                    (ROOT / f"results/eval_{algo}_seed{seed}_{policy}.json").read_text()
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                )
                # = stores a result in fresh; use square brackets to select part of groups[f'{algo}_{policy}', 20000].
                fresh = groups[f"{algo}_{policy}", 20000][seed]
                # = stores a result in equal; evaluate the expression on the right before using its result.
                equal = original["episodes"] == fresh["episodes"]
                # Call reproduction.append using parentheses to supply its inputs; this line does not store a separate return value.
                reproduction.append(
                    # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                    dict(
                        # Inside this continued call, name=value supplies an option by name rather than by its position.
                        algorithm=algo,
                        # Inside this continued call, name=value supplies an option by name rather than by its position.
                        seed=seed,
                        # Inside this continued call, name=value supplies an option by name rather than by its position.
                        policy=policy,
                        # Inside this continued call, name=value supplies an option by name rather than by its position.
                        exact_episode_match=equal,
                    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                    )
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                )
                # if checks not equal; only a true condition runs the indented block.
                if not equal:
                    # raise signals an error and stops this normal path, so invalid data are not silently accepted.
                    raise ValueError("20k episode reproduction failed")
    # Call save_json using parentheses to supply its inputs; this line does not store a separate return value.
    save_json(STAGE / "evaluation_reproduction.json", reproduction)
    # = stores a result in result; call dict using parentheses to supply its inputs.
    result = dict(
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        evaluation_reproduction=reproduction,
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        groups=summaries,
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        budget_differences=comparisons,
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        same_network_control=controls,
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        calibration=calibration,
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        prefix_verification=prefix,
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        confirmation_seeds_used=False,
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # Call save_json using parentheses to supply its inputs; this line does not store a separate return value.
    save_json(STAGE / "summary.json", result)
    # Call plt.rcparams.update using parentheses to supply its inputs; this line does not store a separate return value.
    plt.rcParams.update(
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # = unpacks the right-hand result into (fig, axes), matching values to names in order.
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    # for takes each item from enumerate([1.0, 1.5, 2.0]) in turn; (col, density) holds the current item.
    for col, density in enumerate([1.0, 1.5, 2.0]):
        # for takes each item from enumerate(LABELS) in turn; (i, name) holds the current item.
        for i, name in enumerate(LABELS):
            # = stores a result in points; collect or generate one result per item; the "for" part names the item being processed.
            points = [
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                r for r in summaries if r["policy"] == name and r["density"] == density
            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
            ]
            # = stores a result in color; evaluate the expression on the right before using its result.
            color = f"C{i}"
            # for takes each item from [(0, 'crash_rate', 100), (1, 'speed', 1)] in turn; (row, metric, scale) holds the current item.
            for row, metric, scale in [(0, "crash_rate", 100), (1, "speed", 1)]:
                # = stores a result in ax; use square brackets to select part of axes.
                ax = axes[row, col]
                # Call ax.plot using parentheses to supply its inputs; this line does not store a separate return value.
                ax.plot(
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    [20, 40],
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    [r[metric] * scale for r in points],
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    "-o",
                    # Inside this continued call, name=value supplies an option by name rather than by its position.
                    color=color,
                    # Inside this continued call, name=value supplies an option by name rather than by its position.
                    label=LABELS[name],
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                )
                # for takes each item from range(3) in turn; seed holds the current item.
                for seed in range(3):
                    # Call ax.plot using parentheses to supply its inputs; this line does not store a separate return value.
                    ax.plot(
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        [20, 40],
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        [r["per_seed"][seed][metric] * scale for r in points],
                        # Inside this continued call, name=value supplies an option by name rather than by its position.
                        color=color,
                        # Inside this continued call, name=value supplies an option by name rather than by its position.
                        alpha=0.25,
                        # Inside this continued call, name=value supplies an option by name rather than by its position.
                        lw=0.8,
                    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                    )
                # Call ax.grid using parentheses to supply its inputs; this line does not store a separate return value.
                ax.grid(alpha=0.2)
                # Call ax.set_xticks using parentheses to supply its inputs; this line does not store a separate return value.
                ax.set_xticks([20, 40])
                # Call ax.set_xlabel using parentheses to supply its inputs; this line does not store a separate return value.
                ax.set_xlabel("Training steps (thousands)")
        # Call axes[0, col].set_title using parentheses to supply its inputs; this line does not store a separate return value.
        axes[0, col].set_title(f"Traffic density {density:g}")
        # Call axes[0, col].set_ylim using parentheses to supply its inputs; this line does not store a separate return value.
        axes[0, col].set_ylim(0, 105)
    # Call axes[0, 0].set_ylabel using parentheses to supply its inputs; this line does not store a separate return value.
    axes[0, 0].set_ylabel("Episodes with collision (%)")
    # Call axes[1, 0].set_ylabel using parentheses to supply its inputs; this line does not store a separate return value.
    axes[1, 0].set_ylabel("Mean episode speed (m/s)")
    # Call axes[0, 0].legend using parentheses to supply its inputs; this line does not store a separate return value.
    axes[0, 0].legend(fontsize=8)
    # Call fig.suptitle using parentheses to supply its inputs; this line does not store a separate return value.
    fig.suptitle(
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "More training, fixed exploration schedule; faint lines = individual training seeds"
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # Call fig.tight_layout using parentheses to supply its inputs; this line does not store a separate return value.
    fig.tight_layout()
    # Save a figure for the local results page; this does not affect agent behavior.
    # Call fig.savefig using parentheses to supply its inputs; this line does not store a separate return value.
    fig.savefig(viewer / "training_budget_budget.png", dpi=150)
    # Call plt.close using parentheses to supply its inputs; this line does not store a separate return value.
    plt.close(fig)
    # = unpacks the right-hand result into (fig, axes), matching values to names in order.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    # for takes each item from zip(axes, ['mean', 'tail25']) in turn; (ax, policy) holds the current item.
    for ax, policy in zip(axes, ["mean", "tail25"]):
        # for takes each item from [(20000, 'C0', -0.04), (40000, 'C1', 0.04)] in turn; (budget, color, offset) holds the current item.
        for budget, color, offset in [(20000, "C0", -0.04), (40000, "C1", 0.04)]:
            # = stores a result in rows; collect or generate one result per item; the "for" part names the item being processed.
            rows = [
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                r
                # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                for r in calibration
                # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                if r["policy"] == policy and r["budget"] == budget
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
                capsize=4,
                # Inside this continued call, name=value supplies an option by name rather than by its position.
                label=f"{budget//1000}k",
                # Inside this continued call, name=value supplies an option by name rather than by its position.
                color=color,
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            )
        # Call ax.axhline using parentheses to supply its inputs; this line does not store a separate return value.
        ax.axhline(0, color="black", lw=0.8)
        # Call ax.set_title using parentheses to supply its inputs; this line does not store a separate return value.
        ax.set_title(f"QR {policy} deployment")
        # Call ax.set_xlabel using parentheses to supply its inputs; this line does not store a separate return value.
        ax.set_xlabel("Traffic density")
        # Call ax.set_xticks using parentheses to supply its inputs; this line does not store a separate return value.
        ax.set_xticks([1, 1.5, 2])
        # Call ax.legend using parentheses to supply its inputs; this line does not store a separate return value.
        ax.legend()
        # Call ax.grid using parentheses to supply its inputs; this line does not store a separate return value.
        ax.grid(alpha=0.2)
    # Call axes[0].set_ylabel using parentheses to supply its inputs; this line does not store a separate return value.
    axes[0].set_ylabel("Prediction minus discounted return")
    # Call fig.suptitle using parentheses to supply its inputs; this line does not store a separate return value.
    fig.suptitle("Censoring bounds only; whiskers are not confidence intervals")
    # Call fig.tight_layout using parentheses to supply its inputs; this line does not store a separate return value.
    fig.tight_layout()
    # Save a figure for the local results page; this does not affect agent behavior.
    # Call fig.savefig using parentheses to supply its inputs; this line does not store a separate return value.
    fig.savefig(viewer / "training_budget_bias.png", dpi=150)
    # Call plt.close using parentheses to supply its inputs; this line does not store a separate return value.
    plt.close(fig)
    # = unpacks the right-hand result into (fig, axes), matching values to names in order.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # for takes each item from zip(axes, ['dqn', 'qrdqn']) in turn; (ax, algo) holds the current item.
    for ax, algo in zip(axes, ["dqn", "qrdqn"]):
        # for takes each item from range(3) in turn; seed holds the current item.
        for seed in range(3):
            # with enters a temporary context and exits it afterward, even if an error interrupts the block.
            with (STAGE / f"{algo}_seed{seed}/training.csv").open() as f:
                # = stores a result in logs; call list using parentheses to supply its inputs.
                logs = list(csv.DictReader(f))
            # = stores a result in steps; call np.array using parentheses to supply its inputs.
            steps = np.array([int(r["step"]) for r in logs])
            # = stores a result in returns; call np.array using parentheses to supply its inputs.
            returns = np.array([float(r["return"]) for r in logs])
            # if checks len(returns) >= 100; only a true condition runs the indented block.
            if len(returns) >= 100:
                # Call ax.plot using parentheses to supply its inputs; this line does not store a separate return value.
                ax.plot(
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    steps[99:] / 1000,
                    # Inside this continued call, name=value supplies an option by name rather than by its position.
                    np.convolve(returns, np.ones(100) / 100, mode="valid"),
                    # Inside this continued call, name=value supplies an option by name rather than by its position.
                    label=f"Seed {seed}",
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                )
        # Call ax.axvline using parentheses to supply its inputs; this line does not store a separate return value.
        ax.axvline(20, color="black", ls="--", lw=1)
        # Call ax.set_title using parentheses to supply its inputs; this line does not store a separate return value.
        ax.set_title(algo.upper())
        # Call ax.set_xlabel using parentheses to supply its inputs; this line does not store a separate return value.
        ax.set_xlabel("Training steps (thousands)")
        # Call ax.grid using parentheses to supply its inputs; this line does not store a separate return value.
        ax.grid(alpha=0.2)
        # Call ax.legend using parentheses to supply its inputs; this line does not store a separate return value.
        ax.legend()
    # Call axes[0].set_ylabel using parentheses to supply its inputs; this line does not store a separate return value.
    axes[0].set_ylabel("Training return: trailing 100 episodes")
    # Call fig.suptitle using parentheses to supply its inputs; this line does not store a separate return value.
    fig.suptitle("Exploratory training returns include epsilon-greedy actions")
    # Call fig.tight_layout using parentheses to supply its inputs; this line does not store a separate return value.
    fig.tight_layout()
    # Save a figure for the local results page; this does not affect agent behavior.
    # Call fig.savefig using parentheses to supply its inputs; this line does not store a separate return value.
    fig.savefig(viewer / "training_budget_learning.png", dpi=150)
    # Call plt.close using parentheses to supply its inputs; this line does not store a separate return value.
    plt.close(fig)

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
            + "".join("<th>" + html.escape(x) + "</th>" for x in headers)
            # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
            + "</tr></thead><tbody>"
            # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
            + "".join(
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                "<tr>" + "".join("<td>" + str(x) + "</td>" for x in row) + "</tr>"
                # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                for row in rows
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            )
            # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
            + "</tbody></table></div>"
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )

    # = stores a result in body; use quoted text as data; Python does not execute the words inside it.
    body = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>training budget experiment · Risk-sensitive driving</title><style>body{font:17px/1.6 system-ui;color:#18283d;background:#f4f7fb;margin:auto;max-width:1160px;padding:36px}h1{font-size:38px;line-height:1.2}section{background:white;padding:24px;margin:24px 0;border-radius:14px}img{width:100%;height:auto}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;text-align:left;border-bottom:1px solid #dde3eb}th{background:#eef3fa}a{color:#175bbc}.note{color:#47576b}</style></head><body><p>training budget experiment / CONTROLLED TRAINING BUDGET</p><h1>Does more experience improve risk-sensitive driving?</h1><p>Six training trajectories, two checkpoint budgets, and a fixed exploration schedule. This is a development study; confirmation scenarios remain reserved.</p><p><a href="../risk_analysis-report/index.html">risk analysis: interactive risk frontier, calibration and replays</a> · <a href="../training_budget/summary.json">Numerical results</a></p>"""
    # = stores a result in normal; collect or generate one result per item; the "for" part names the item being processed.
    normal = {(r["policy"], r["budget"]): r for r in summaries if r["density"] == 1.0}
    # = stores a result in highlights; build a list, keeping its items in order.
    highlights = []
    # for takes each item from LABELS in turn; name holds the current item.
    for name in LABELS:
        # = unpacks the right-hand result into (a, b), matching values to names in order.
        a, b = normal[name, 20000], normal[name, 40000]
        # Call highlights.append using parentheses to supply its inputs; this line does not store a separate return value.
        highlights.append(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            f"{LABELS[name]}: collisions {100*a['crash_rate']:.1f}% to {100*b['crash_rate']:.1f}%; speed {a['speed']:.2f} to {b['speed']:.2f} m/s."
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
    # The combined operator updates body using its previous value.
    body += (
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "<section><h2>Normal-traffic findings</h2><ul>"
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + "".join("<li>" + x + "</li>" for x in highlights)
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + "</ul><p>Budget effects are paired within each trajectory. These development comparisons do not establish generalization to reserved scenarios or real driving.</p></section>"
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # The combined operator updates body using its previous value.
    body += '<section><h2>Training budget and physical performance</h2><p>Networks are trained from scratch for 40,000 steps. Epsilon reaches .05 at step 12,001 in both budgets. Reward, traffic, architecture and optimizer are unchanged. Solid lines average three training seeds; faint lines show their paired checkpoints.</p><img src="training_budget_budget.png"></section>'
    # The combined operator updates body using its previous value.
    body += '<section><h2>Learning curves</h2><p>Training rewards include exploration and variable episode lengths; use frozen-policy evaluations below to judge performance.</p><img src="training_budget_learning.png"></section>'
    # The combined operator updates body using its previous value.
    body += (
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "<section><h2>All results</h2><p>30 shared development scenarios per density per model. Speed is the mean of episode-average speeds. Worst 10% means the three lowest undiscounted episode returns per seed, then averaged across seeds.</p>"
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + table(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            [
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Policy",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Steps",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Density",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Collision",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Speed m/s",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Distance m",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Return",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Worst 10%",
            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
            ],
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            [
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                [
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    LABELS[r["policy"]],
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    r["budget"],
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    r["density"],
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    f"{100*r['crash_rate']:.1f}%",
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    f"{r['speed']:.2f}",
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    f"{r['distance']:.0f}",
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    f"{r['mean_return']:.2f}",
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    f"{r['worst10']:.2f}",
                # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
                ]
                # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                for r in summaries
            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
            ],
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + "</section>"
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # The combined operator updates body using its previous value.
    body += (
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "<section><h2>Paired budget effects: 40k minus 20k</h2><p>Negative collision differences favor 40k. Exploratory 95% intervals resample training seeds and shared scenarios as crossed factors, preserving checkpoint pairing (5,000 replicates). Three training seeds give limited uncertainty estimates. Repeated development evaluation is not a confirmatory test.</p>"
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + table(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            [
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Policy",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Density",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Collision change pp",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Exploratory interval pp",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Per-seed change pp",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Speed change m/s",
            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
            ],
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            [
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                [
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    LABELS[r["policy"]],
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    r["density"],
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    f"{r['crashed']['mean_difference']*100:+.1f}",
                    # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                    str(
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        [
                            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                            round(x * 100, 1)
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
                            round(x * 100, 1)
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
                for r in comparisons
            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
            ],
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + "</section>"
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # The combined operator updates body using its previous value.
    body += (
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "<section><h2>Frozen speed control at 40k</h2><p>Lower quarter minus mean deployment with the old 25 m/s governor. This threshold was not reselected at 40k. Check residual speed differences before attributing collision changes to the risk rule.</p>"
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + table(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            [
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Density",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Collision change pp",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Exploratory interval pp",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Per-seed change pp",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Speed change m/s",
            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
            ],
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            [
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                [
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    r["density"],
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    f"{r['crashed']['mean_difference']*100:+.1f}",
                    # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                    str(
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        [
                            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                            round(x * 100, 1)
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
                            round(x * 100, 1)
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
                for r in controls
            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
            ],
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + "</section>"
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # The combined operator updates body using its previous value.
    body += (
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        '<section><h2>Return prediction diagnostics</h2><p>60 separate development scenarios per density per model. Successful rollouts right-censor the continuing return; whiskers bound missing continuation. These are not statistical confidence intervals. Lower-quarter deployment changes the continuation policy relative to mean-greedy training.</p><img src="training_budget_bias.png">'
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + table(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            [
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Policy",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Steps",
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
                    r["policy"],
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    r["budget"],
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
    # The combined operator updates body using its previous value.
    body += (
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "<section><h2>Reproducibility and limits</h2><p>Exact original 20k online-weight matches: "
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + str(sum(r["prefix_weights_equal"] for r in prefix))
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + "/6. Checkpoints share training history and are not independent runs. Density randomization was deferred to isolate budget. Predicted quantiles are not safety probabilities; cautious selection has no global CVaR-optimality guarantee. HighwayEnv is a simplified simulator.</p><p>Run <code>python run_training_budget.py --jobs 3</code> and <code>python -m pytest -q</code>. Existing artifacts are validated before reuse. The original risk analysis viewer is preserved.</p></section></body></html>"
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # Call (viewer / 'index.html').write_text using parentheses to supply its inputs; this line does not store a separate return value.
    (viewer / "index.html").write_text(body)
    # Call print using parentheses to supply its inputs; this line does not store a separate return value.
    print("training budget experiment report saved", flush=True)


# Run the entry point only when this file is executed, not when it is imported.
# __name__ equals "__main__" when this file runs directly; this guard avoids running the script merely because another file imports it.
if __name__ == "__main__":
    # Call report using parentheses to supply its inputs; this line does not store a separate return value.
    report()
