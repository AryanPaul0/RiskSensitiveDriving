"""Create the baseline controls results page.

Read saved numbers, calculate clearly labeled summaries, draw figures, and write
an HTML page. This file presents evidence; it does not train or select models.
Long HTML strings are page templates. The reinforcement-learning calculations
live in agents.py and experiment.py, not inside the page layout.

Original implementation notes:
Distributional and scalar baselines with equal traffic coverage and speed controls.
"""

# from pathlib import brings the named tools into this file; parentheses allow several lines.
from pathlib import Path
# import makes these modules available; "as" gives a module a shorter local name.
import hashlib
# import makes these modules available; "as" gives a module a shorter local name.
import json
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
from .analysis import summarize_group, paired_matrix, crossed_bootstrap
# from experiment import brings the named tools into this file; parentheses allow several lines.
from .experiment import save_json
# from density_coverage_report import brings the named tools into this file; parentheses allow several lines.
from .density_coverage_report import table

# = stores a result in ROOT; use square brackets to select part of Path(__file__).resolve().parents.
ROOT = Path(__file__).resolve().parents[1]
# = stores a result in STAGE; apply / to these operands; numbers are divided, while Path objects join folder names.
STAGE = ROOT / "results/baseline_controls"
# = stores a result in LABELS; build a dictionary, which stores values under named keys.
LABELS = {
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "fixed_dqn": "Fixed DQN",
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "mixed_dqn": "Mixed DQN",
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "mixed_qr_mean": "Mixed QR mean",
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "mixed_qr_tail": "Mixed QR lower quarter",
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "matched_dqn": "Speed-targeted DQN",
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "matched_qr": "Speed-targeted QR mean",
# } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
}


# def defines read; parentheses list its inputs, and the colon starts its indented body.
def read(p):
    # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
    """Read saved JSON data into ordinary Python dictionaries and lists."""
    # return ends this function and sends its result back to the code that called it.
    return json.loads(p.read_text())


# def defines load; parentheses list its inputs, and the colon starts its indented body.
def load(group, block, selections):
    """Read the result files belonging to one comparison group. The group controls
    which saved model and decision rule are represented; it does not train a model.
    """
    # = stores a result in paths; build a list, keeping its items in order.
    paths = []
    # for takes each item from range(3) in turn; seed holds the current item.
    for seed in range(3):
        # if checks group == 'fixed_dqn'; only a true condition runs the indented block.
        if group == "fixed_dqn":
            # = stores a result in p; choose the value before "if" when its condition is true, otherwise the value after "else".
            p = (
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                ROOT / f"results/training_budget/eval_dqn_{seed}_40000_mean.json"
                # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                if block == "main"
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                else STAGE / f"eval_dqn_fixed_mean_seed{seed}_transfer.json"
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            )
        # if checks group == 'mixed_dqn'; only a true condition runs the indented block.
        elif group == "mixed_dqn":
            # = stores a result in p; apply / to these operands; numbers are divided, while Path objects join folder names.
            p = STAGE / f"eval_dqn_mixed_mean_seed{seed}_{block}.json"
        # if checks group in ['mixed_qr_mean', 'mixed_qr_tail']; only a true condition runs the indented block.
        elif group in ["mixed_qr_mean", "mixed_qr_tail"]:
            # = stores a result in policy; choose the value before "if" when its condition is true, otherwise the value after "else".
            policy = "mean" if group == "mixed_qr_mean" else "tail25"
            # = stores a result in p; apply / to these operands; numbers are divided, while Path objects join folder names.
            p = (
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                ROOT
                # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
                / f"results/density_coverage/eval_mixed_{policy}_seed{seed}_{block}.json"
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            )
        # else provides the alternative branch; after a loop it instead means the loop finished without break.
        else:
            # = stores a result in algo; choose the value before "if" when its condition is true, otherwise the value after "else".
            algo = "dqn" if group == "matched_dqn" else "qrdqn"
            # = stores a result in cap; use square brackets to select part of selections[algo].
            cap = selections[algo]["selected_cap_m_s"]
            # = stores a result in token; call f'cap{cap:g}'.replace using parentheses to supply its inputs.
            token = f"cap{cap:g}".replace(".", "p")
            # = stores a result in p; apply / to these operands; numbers are divided, while Path objects join folder names.
            p = STAGE / f"eval_{algo}_mixed_{token}_seed{seed}_{block}.json"
        # Call paths.append using parentheses to supply its inputs; this line does not store a separate return value.
        paths.append(p)
    # return ends this function and sends its result back to the code that called it.
    return [read(p) for p in paths]


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
    backup = ROOT / "results/density_coverage-report"
    # if checks not backup.exists(); only a true condition runs the indented block.
    if not backup.exists():
        # Call shutil.copytree using parentheses to supply its inputs; this line does not store a separate return value.
        shutil.copytree(viewer, backup)
    # = stores a result in selections; collect or generate one result per item; the "for" part names the item being processed.
    selections = {
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        algo: read(STAGE / f"{algo}_governor_selection.json")
        # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
        for algo in ["dqn", "qrdqn"]
    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
    }
    # = stores a result in data; build a dictionary, which stores values under named keys.
    data = {}
    # = stores a result in groups; build a dictionary, which stores values under named keys.
    groups = {}
    # = stores a result in comparisons; build a list, keeping its items in order.
    comparisons = []
    # = stores a result in exposure; build a dictionary, which stores values under named keys.
    exposure = {}
    # = stores a result in mean_comparisons; build a list, keeping its items in order.
    mean_comparisons = []
    # for takes each item from [('main', [1.0, 1.5, 2.0]), ('transfer', [1.25, 2.5])] in turn; (block, densities) holds the current item.
    for block, densities in [("main", [1.0, 1.5, 2.0]), ("transfer", [1.25, 2.5])]:
        # = stores a result in data[block]; collect or generate one result per item; the "for" part names the item being processed.
        data[block] = {g: load(g, block, selections) for g in LABELS}
        # = stores a result in groups[block]; collect or generate one result per item; the "for" part names the item being processed.
        groups[block] = {g: summarize_group(data[block][g]) for g in LABELS}
        # for takes each item from ['mixed_dqn', 'mixed_qr_mean', 'matched_dqn', 'matched_qr'] in turn; other holds the current item.
        for other in ["mixed_dqn", "mixed_qr_mean", "matched_dqn", "matched_qr"]:
            # for takes each item from densities in turn; density holds the current item.
            for density in densities:
                # = stores a result in item; build a dictionary, which stores values under named keys.
                item = {"block": block, "reference": other, "density": density}
                # Compare one measured quantity at a time, keeping the scenarios paired.
                # for takes each item from ['crashed', 'mean_speed_m_s', 'distance_m', 'return'] in turn; metric holds the current item.
                for metric in ["crashed", "mean_speed_m_s", "distance_m", "return"]:
                    # = unpacks the right-hand result into (_, matrix), matching values to names in order.
                    _, matrix = paired_matrix(
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        data[block]["mixed_qr_tail"],
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        data[block][other],
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        density,
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        metric,
                    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                    )
                    # = stores a result in item[metric]; call crossed_bootstrap using parentheses to supply its inputs.
                    item[metric] = crossed_bootstrap(matrix)
                # Call comparisons.append using parentheses to supply its inputs; this line does not store a separate return value.
                comparisons.append(item)
        # for takes each item from [('Mean selectors', 'mixed_qr_mean', 'mixed_dqn'), ('Speed-targeted mean selectors', 'matched_qr'... in turn; (label, left, right) holds the current item.
        for label, left, right in [
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            ("Mean selectors", "mixed_qr_mean", "mixed_dqn"),
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            ("Speed-targeted mean selectors", "matched_qr", "matched_dqn"),
        # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
        ]:
            # for takes each item from densities in turn; density holds the current item.
            for density in densities:
                # = stores a result in item; build a dictionary, which stores values under named keys.
                item = {"block": block, "comparison": label, "density": density}
                # Compare one measured quantity at a time, keeping the scenarios paired.
                # for takes each item from ['crashed', 'mean_speed_m_s'] in turn; metric holds the current item.
                for metric in ["crashed", "mean_speed_m_s"]:
                    # = unpacks the right-hand result into (_, matrix), matching values to names in order.
                    _, matrix = paired_matrix(
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        data[block][left], data[block][right], density, metric
                    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                    )
                    # = stores a result in item[metric]; call crossed_bootstrap using parentheses to supply its inputs.
                    item[metric] = crossed_bootstrap(matrix)
                # Call mean_comparisons.append using parentheses to supply its inputs; this line does not store a separate return value.
                mean_comparisons.append(item)
    # for takes each item from ['dqn', 'qrdqn'] in turn; algo holds the current item.
    for algo in ["dqn", "qrdqn"]:
        # = stores a result in folder; choose the value before "if" when its condition is true, otherwise the value after "else".
        folder = STAGE if algo == "dqn" else ROOT / "results/density_coverage"
        # = stores a result in exposure[algo]; collect or generate one result per item; the "for" part names the item being processed.
        exposure[algo] = [
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            {"seed": s, **read(folder / f"{algo}_seed{s}/summary.json")}
            # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
            for s in range(3)
        # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
        ]
        # if checks any((sum(r['steps_by_density'].values()) != 40000 for r in exposure[algo])); only a true condition runs the indented block.
        if any(sum(r["steps_by_density"].values()) != 40000 for r in exposure[algo]):
            # raise signals an error and stops this normal path, so invalid data are not silently accepted.
            raise ValueError("Unequal interaction budgets")
    # Call save_json using parentheses to supply its inputs; this line does not store a separate return value.
    save_json(
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        STAGE / "summary.json",
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        {
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "design": read(STAGE / "design.json"),
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "groups": groups,
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "lower_quarter_minus_reference": comparisons,
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "selections": selections,
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "training_exposure": exposure,
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "exploratory_mean_selector_comparisons": mean_comparisons,
        # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
        },
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # Call plt.rcparams.update using parentheses to supply its inputs; this line does not store a separate return value.
    plt.rcParams.update(
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # = stores a result in colors; collect or generate one result per item; the "for" part names the item being processed.
    colors = {g: f"C{i}" for i, g in enumerate(LABELS)}
    # = unpacks the right-hand result into (fig, axes), matching values to names in order.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    # for takes each item from LABELS in turn; group holds the current item.
    for group in LABELS:
        # = stores a result in rows; use square brackets to select part of groups['main'].
        rows = groups["main"][group]
        # = stores a result in x; collect or generate one result per item; the "for" part names the item being processed.
        x = [r["density"] for r in rows]
        # for takes each item from zip(axes, ['crash_rate', 'mean_speed_m_s', 'mean_distance_m'], [100, 1, 1], ['Episodes with colli... in turn; (ax, metric, scale, title) holds the current item.
        for ax, metric, scale, title in zip(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            axes,
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            ["crash_rate", "mean_speed_m_s", "mean_distance_m"],
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            [100, 1, 1],
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            [
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Episodes with collision (%)",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Mean episode speed (m/s)",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Mean distance (m)",
            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
            ],
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        ):
            # Call ax.plot using parentheses to supply its inputs; this line does not store a separate return value.
            ax.plot(
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                x,
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                [r[metric] * scale for r in rows],
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "-o",
                # Inside this continued call, name=value supplies an option by name rather than by its position.
                color=colors[group],
                # Inside this continued call, name=value supplies an option by name rather than by its position.
                label=LABELS[group],
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            )
            # Call ax.set_xlabel using parentheses to supply its inputs; this line does not store a separate return value.
            ax.set_xlabel("Evaluation density")
            # Call ax.set_xticks using parentheses to supply its inputs; this line does not store a separate return value.
            ax.set_xticks(x)
            # Call ax.set_ylabel using parentheses to supply its inputs; this line does not store a separate return value.
            ax.set_ylabel(title)
            # Call ax.grid using parentheses to supply its inputs; this line does not store a separate return value.
            ax.grid(alpha=0.2)
    # Call axes[0].set_ylim using parentheses to supply its inputs; this line does not store a separate return value.
    axes[0].set_ylim(0, 105)
    # Call axes[0].legend using parentheses to supply its inputs; this line does not store a separate return value.
    axes[0].legend(fontsize=8)
    # Call fig.suptitle using parentheses to supply its inputs; this line does not store a separate return value.
    fig.suptitle(
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "40k interactions per model; identical configured traffic mixture for mixed learners"
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # Call fig.tight_layout using parentheses to supply its inputs; this line does not store a separate return value.
    fig.tight_layout()
    # Save a figure for the local results page; this does not affect agent behavior.
    # Call fig.savefig using parentheses to supply its inputs; this line does not store a separate return value.
    fig.savefig(viewer / "baseline_controls_performance.png", dpi=150)
    # Call plt.close using parentheses to supply its inputs; this line does not store a separate return value.
    plt.close(fig)
    # = unpacks the right-hand result into (fig, axes), matching values to names in order.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    # for takes each item from zip(axes, [1.0, 1.5, 2.0]) in turn; (ax, density) holds the current item.
    for ax, density in zip(axes, [1.0, 1.5, 2.0]):
        # for takes each item from LABELS in turn; group holds the current item.
        for group in LABELS:
            # = stores a result in row; call next using parentheses to supply its inputs.
            row = next(r for r in groups["main"][group] if r["density"] == density)
            # for takes each item from row['per_seed'] in turn; seed holds the current item.
            for seed in row["per_seed"]:
                # Call ax.scatter using parentheses to supply its inputs; this line does not store a separate return value.
                ax.scatter(
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    seed["mean_speed_m_s"],
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    100 * seed["crash_rate"],
                    # Inside this continued call, name=value supplies an option by name rather than by its position.
                    color=colors[group],
                    # Inside this continued call, name=value supplies an option by name rather than by its position.
                    s=18,
                    # Inside this continued call, name=value supplies an option by name rather than by its position.
                    alpha=0.35,
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                )
            # Call ax.scatter using parentheses to supply its inputs; this line does not store a separate return value.
            ax.scatter(
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                row["mean_speed_m_s"],
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                100 * row["crash_rate"],
                # Inside this continued call, name=value supplies an option by name rather than by its position.
                color=colors[group],
                # Inside this continued call, name=value supplies an option by name rather than by its position.
                s=75,
                # Inside this continued call, name=value supplies an option by name rather than by its position.
                marker="D",
                # Inside this continued call, name=value supplies an option by name rather than by its position.
                label=LABELS[group],
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            )
        # Call ax.set_title using parentheses to supply its inputs; this line does not store a separate return value.
        ax.set_title(f"Density {density:g}")
        # Call ax.set_xlabel using parentheses to supply its inputs; this line does not store a separate return value.
        ax.set_xlabel("Mean episode speed (m/s)")
        # Call ax.set_ylim using parentheses to supply its inputs; this line does not store a separate return value.
        ax.set_ylim(-3, 105)
        # Call ax.grid using parentheses to supply its inputs; this line does not store a separate return value.
        ax.grid(alpha=0.2)
    # Call axes[0].set_ylabel using parentheses to supply its inputs; this line does not store a separate return value.
    axes[0].set_ylabel("Episodes with collision (%)")
    # Call axes[0].legend using parentheses to supply its inputs; this line does not store a separate return value.
    axes[0].legend(fontsize=7)
    # Call fig.suptitle using parentheses to supply its inputs; this line does not store a separate return value.
    fig.suptitle(
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "Speed/collision tradeoffs; diamonds = means, small dots = training seeds"
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # Call fig.tight_layout using parentheses to supply its inputs; this line does not store a separate return value.
    fig.tight_layout()
    # Save a figure for the local results page; this does not affect agent behavior.
    # Call fig.savefig using parentheses to supply its inputs; this line does not store a separate return value.
    fig.savefig(viewer / "baseline_controls_tradeoff.png", dpi=150)
    # Call plt.close using parentheses to supply its inputs; this line does not store a separate return value.
    plt.close(fig)
    # = stores a result in document; use quoted text as data; Python does not execute the words inside it.
    document = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Scalar baselines and speed controls · Risk-sensitive driving</title><style>body{font:17px/1.6 system-ui;color:#18283d;background:#f4f7fb;margin:auto;max-width:1160px;padding:36px}h1{font-size:38px;line-height:1.2}section{background:white;padding:24px;margin:24px 0;border-radius:14px}img{width:100%;height:auto}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;text-align:left;border-bottom:1px solid #dde3eb}th{background:#eef3fa}a{color:#175bbc}</style></head><body><p>DEEP RL / STRONGER BASELINES</p><h1>Does cautious selection outperform a comparable scalar learner?</h1><p>DQN and QR-DQN now both train for 40,000 interactions on the same configured episode-density mixture: 1, 1.5 and 2. Rewards, hidden-layer sizes, optimizer settings and exploration schedules are held fixed. Experience remains policy-dependent; there are three training seeds per learner.</p><p><a href="../density_coverage-report/index.html">Coverage comparison, return diagnostics and replays</a> · <a href="../baseline_controls/summary.json">Numerical results</a> · <a href="../baseline_controls/design.json">Recorded design</a></p>"""
    # = stores a result in headlines; build a list, keeping its items in order.
    headlines = []
    # for takes each item from [1.0, 2.0] in turn; density holds the current item.
    for density in [1.0, 2.0]:
        # = stores a result in contrast; call next using parentheses to supply its inputs.
        contrast = next(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            r
            # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
            for r in comparisons
            # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
            if r["block"] == "main"
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            and r["reference"] == "matched_dqn"
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            and r["density"] == density
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
        # = stores a result in diff; use square brackets to select part of contrast.
        diff = contrast["crashed"]
        # = stores a result in interval; collect or generate one result per item; the "for" part names the item being processed.
        interval = [round(100 * x, 1) for x in diff["exploratory_95_percent_interval"]]
        # Call headlines.append using parentheses to supply its inputs; this line does not store a separate return value.
        headlines.append(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            f"At density {density:g}, cautious QR minus speed-targeted DQN is {100*diff['mean_difference']:+.1f} collision percentage points (exploratory interval {interval}), with a speed difference of {contrast['mean_speed_m_s']['mean_difference']:+.2f} m/s."
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
    # The combined operator updates document using its previous value.
    document += (
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "<section><h2>Main control comparison</h2>"
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + "".join("<p>" + line + "</p>" for line in headlines)
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + "<p>The full seed results and residual speed differences below are needed to interpret these comparisons.</p></section>"
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # The combined operator updates document using its previous value.
    document += '<section><h2>Equal-coverage comparison</h2><img src="baseline_controls_performance.png"><p>Mixed learners share the configured traffic distribution and step budget. The fixed-density DQN is retained to show its coverage effect. The cautious QR selector uses the lowest quarter of 32 outputs, without additional risk-level tuning.</p></section>'
    # The combined operator updates document using its previous value.
    document += (
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "<section><h2>Seed-0 speed-targeted controls</h2><p>For each learner, select a cap using only seed-0 normal-traffic speed closeness to the cautious QR policy; then freeze it before evaluating seeds 1 and 2. The development threshold grid is 21–25 m/s. This does not match the full speed distribution or the action interventions.</p>"
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + table(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            ["Learner", "Selected cap m/s", "Target speed m/s", "Seed-0 residual m/s"],
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            [
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                [
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    a,
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    r["selected_cap_m_s"],
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    f"{r['target_speed_m_s']:.3f}",
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    f"{r['seed0_speed_difference_m_s']:+.3f}",
                # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
                ]
                # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                for a, r in selections.items()
            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
            ],
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + '<img src="baseline_controls_tradeoff.png"></section>'
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # for takes each item from [('main', 'Primary traffic scenarios'), ('transfer', 'Intermediate and beyond-training densities')] in turn; (block, title) holds the current item.
    for block, title in [
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        ("main", "Primary traffic scenarios"),
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        ("transfer", "Intermediate and beyond-training densities"),
    # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
    ]:
        # The combined operator updates document using its previous value.
        document += (
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "<section><h2>"
            # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
            + title
            # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
            + "</h2><p>"
            # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
            + (
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Scenarios 10000–10029; densities 1, 1.5 and 2."
                # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                if block == "main"
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                else "Separate development scenarios 16000–16029; density 1.25 is intermediate and density 2.5 is beyond both training ranges."
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            )
            # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
            + "</p>"
            # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
            + table(
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                [
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    "Method",
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    "Density",
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    "Collisions",
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    "Speed m/s",
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    "Distance m",
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    "Mean return",
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    "Worst 10% return",
                # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
                ],
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                [
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    [
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        LABELS[g],
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        r["density"],
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        f"{100*r['crash_rate']:.1f}%",
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        f"{r['mean_speed_m_s']:.2f}",
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        f"{r['mean_distance_m']:.0f}",
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        f"{r['mean_return']:.2f}",
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        f"{r['worst_10_percent_mean_return']:.2f}",
                    # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
                    ]
                    # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                    for g in LABELS
                    # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                    for r in groups[block][g]
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
        "<section><h2>Cautious QR minus each reference</h2><p>Negative collision differences favor cautious QR. Exploratory 95% intervals use 5,000 crossed bootstrap replicates over model seed IDs and shared scenarios, preserving policy pairing. Speed-control intervals condition on the selected cap and do not adjust for its selection. Three seeds and repeated development use limit inference.</p>"
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + table(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            [
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Block",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Reference",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Density",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Collision change pp",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Exploratory interval pp",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Speed change m/s",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Per-seed collision change pp",
            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
            ],
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            [
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                [
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    r["block"],
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    LABELS[r["reference"]],
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
    # = stores a result in residual; build a list, keeping its items in order.
    residual = []
    # for takes each item from ['matched_dqn', 'matched_qr'] in turn; reference holds the current item.
    for reference in ["matched_dqn", "matched_qr"]:
        # = stores a result in target; call next using parentheses to supply its inputs.
        target = next(r for r in groups["main"]["mixed_qr_tail"] if r["density"] == 1.0)
        # = stores a result in control; call next using parentheses to supply its inputs.
        control = next(r for r in groups["main"][reference] if r["density"] == 1.0)
        # for takes each item from zip(target['per_seed'], control['per_seed']) in turn; (a, b) holds the current item.
        for a, b in zip(target["per_seed"], control["per_seed"]):
            # Call residual.append using parentheses to supply its inputs; this line does not store a separate return value.
            residual.append(
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                [
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    LABELS[reference],
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    a["training_seed"],
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    f"{a['mean_speed_m_s']-b['mean_speed_m_s']:+.3f}",
                    # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                    f"{100*(a['crash_rate']-b['crash_rate']):+.1f}",
                # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
                ]
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            )
    # The combined operator updates document using its previous value.
    document += (
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "<section><h2>Exploratory comparison of the mean selectors</h2><p>QR mean minus scalar DQN, using the same training mixture. These secondary summaries were added after inspecting the primary results; no training or selection settings were changed. Distributional learning also changes the output head, loss and parameter count, so this does not isolate a single architectural cause.</p>"
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + table(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            [
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Block",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Comparison",
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
                    r["comparison"],
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
                for r in mean_comparisons
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
        "<section><h2>Normal-traffic residuals by seed</h2><p>Values are cautious minus control. Aggregate matching can hide substantial differences by model.</p>"
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + table(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            [
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Reference",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Training seed",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Speed difference m/s",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Collision difference pp",
            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
            ],
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            residual,
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + "</section>"
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # The combined operator updates document using its previous value.
    document += (
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "<section><h2>Actual mixed-training exposure</h2><p>Density is sampled uniformly by episode, not by transition. Different policies generate different episode lengths and exposure shares despite equal total interaction budgets.</p>"
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + table(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            [
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Learner",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Seed",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Density-1 steps",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Density-1.5 steps",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Density-2 steps",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Completed episodes",
            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
            ],
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            [
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                [
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    algo,
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    r["seed"],
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    r["steps_by_density"]["1.0"],
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    r["steps_by_density"]["1.5"],
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    r["steps_by_density"]["2.0"],
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    r["episodes"],
                # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
                ]
                # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                for algo, rows in exposure.items()
                # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                for r in rows
            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
            ],
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + "</section>"
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # The combined operator updates document using its previous value.
    document += "<section><h2>Reproduce and interpret</h2><p><code>python run_baseline_controls.py --jobs 3</code> validates and reuses completed runs. <code>--stage evaluate</code> skips training; <code>--stage report</code> rebuilds this viewer. <code>python -m pytest -q</code> checks learning, sampling and speed-selection rules.</p><p>All scenarios remain development data; confirmation seeds 20000 onward are untouched. Reward and alpha are not retuned. A speed governor is a decision rule rather than a physical speed guarantee. Partial observations, simplified traffic and forecast error limit conclusions. Neither lower-tail selection nor favorable simulator results guarantee real-road safety.</p></section></body></html>"
    # Call (viewer / 'index.html').write_text using parentheses to supply its inputs; this line does not store a separate return value.
    (viewer / "index.html").write_text(document)
    # Call print using parentheses to supply its inputs; this line does not store a separate return value.
    print("Scalar baseline and speed-control report saved", flush=True)


# Run the entry point only when this file is executed, not when it is imported.
# __name__ equals "__main__" when this file runs directly; this guard avoids running the script merely because another file imports it.
if __name__ == "__main__":
    # Call report using parentheses to supply its inputs; this line does not store a separate return value.
    report()
