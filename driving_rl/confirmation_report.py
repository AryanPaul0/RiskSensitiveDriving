"""Create the confirmation results page.

Read saved numbers, calculate clearly labeled summaries, draw figures, and write
an HTML page. This file presents evidence; it does not train or select models.
Long HTML strings are page templates. The reinforcement-learning calculations
live in agents.py and experiment.py, not inside the page layout.

Original implementation notes:
Locked-policy confirmation with conditional scenario-cluster uncertainty.
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
from .analysis import (
    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
    summarize_group,
    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
    paired_matrix,
    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
    crossed_bootstrap,
    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
    scenario_bootstrap_fixed_models,
# ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
)
# from density_coverage_report import brings the named tools into this file; parentheses allow several lines.
from .density_coverage_report import table
# from experiment import brings the named tools into this file; parentheses allow several lines.
from .experiment import save_json

# = stores a result in ROOT; use square brackets to select part of Path(__file__).resolve().parents.
ROOT = Path(__file__).resolve().parents[1]
# = stores a result in STAGE; apply / to these operands; numbers are divided, while Path objects join folder names.
STAGE = ROOT / "results/confirmation"
# = stores a result in LABELS; build a dictionary, which stores values under named keys.
LABELS = {
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "qr_tail": "QR lower quarter",
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "dqn_governed": "DQN + frozen governor",
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "qr_governed": "QR mean + frozen governor",
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "dqn_mean": "DQN mean",
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "qr_mean": "QR mean",
# } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
}


# def defines read; parentheses list its inputs, and the colon starts its indented body.
def read(p):
    # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
    """Read saved JSON data into ordinary Python dictionaries and lists."""
    # return ends this function and sends its result back to the code that called it.
    return json.loads(p.read_text())


# def defines development; parentheses list its inputs, and the colon starts its indented body.
def development(policy, density):
    """Read the earlier development result for a policy and density.
    It is shown for comparison, not used to retune the confirmation policy.
    """
    # = stores a result in rows; build a list, keeping its items in order.
    rows = []
    # for takes each item from range(3) in turn; s holds the current item.
    for s in range(3):
        # = stores a result in block; choose the value before "if" when its condition is true, otherwise the value after "else".
        block = "main" if density != 2.5 else "transfer"
        # if checks policy == 'qr_tail'; only a true condition runs the indented block.
        if policy == "qr_tail":
            # = stores a result in p; apply / to these operands; numbers are divided, while Path objects join folder names.
            p = (
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                ROOT
                # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
                / f"results/density_coverage/eval_mixed_tail25_seed{s}_{block}.json"
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            )
        # if checks policy == 'qr_mean'; only a true condition runs the indented block.
        elif policy == "qr_mean":
            # = stores a result in p; apply / to these operands; numbers are divided, while Path objects join folder names.
            p = ROOT / f"results/density_coverage/eval_mixed_mean_seed{s}_{block}.json"
        # else provides the alternative branch; after a loop it instead means the loop finished without break.
        else:
            # = stores a result in algo; choose the value before "if" when its condition is true, otherwise the value after "else".
            algo = "qrdqn" if policy == "qr_governed" else "dqn"
            # = stores a result in token; choose the value before "if" when its condition is true, otherwise the value after "else".
            token = "mean" if policy == "dqn_mean" else "cap25"
            # = stores a result in p; apply / to these operands; numbers are divided, while Path objects join folder names.
            p = (
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                ROOT
                # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
                / f"results/baseline_controls/eval_{algo}_mixed_{token}_seed{s}_{block}.json"
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            )
        # Call rows.append using parentheses to supply its inputs; this line does not store a separate return value.
        rows.append(next(r for r in read(p)["summary"] if r["density"] == density))
    # return ends this function and sends its result back to the code that called it.
    return {
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "collision_fraction": float(np.mean([r["crash_rate"] for r in rows])),
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "speed_m_s": float(np.mean([r["mean_speed_m_s"] for r in rows])),
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
    backup = ROOT / "results/baseline_controls-report"
    # if checks not backup.exists(); only a true condition runs the indented block.
    if not backup.exists():
        # Call shutil.copytree using parentheses to supply its inputs; this line does not store a separate return value.
        shutil.copytree(viewer, backup)
    # = stores a result in design; call read using parentheses to supply its inputs.
    design = read(STAGE / "design.json")
    # for takes each item from design['checkpoint_hashes'].values() in turn; configs holds the current item.
    for configs in design["checkpoint_hashes"].values():
        # for takes each item from configs.values() in turn; item holds the current item.
        for item in configs.values():
            # if checks hashlib.sha256((ROOT / item['path']).read_bytes()).hexdigest() != item['sha256']; only a true condition runs the indented block.
            if (
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest()
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                != item["sha256"]
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            ):
                # raise signals an error and stops this normal path, so invalid data are not silently accepted.
                raise ValueError("Frozen model changed")
    # = stores a result in data; collect or generate one result per item; the "for" part names the item being processed.
    data = {
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        p: [read(STAGE / f"eval_{p}_seed{s}.json") for s in range(3)] for p in LABELS
    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
    }
    # = stores a result in groups; collect or generate one result per item; the "for" part names the item being processed.
    groups = {p: summarize_group(d) for p, d in data.items()}
    # = stores a result in comparisons; build a list, keeping its items in order.
    comparisons = []
    # = stores a result in dev; build a list, keeping its items in order.
    dev = []
    # for takes each item from ['dqn_governed', 'qr_governed', 'dqn_mean', 'qr_mean'] in turn; reference holds the current item.
    for reference in ["dqn_governed", "qr_governed", "dqn_mean", "qr_mean"]:
        # for takes each item from design['densities'] in turn; density holds the current item.
        for density in design["densities"]:
            # = stores a result in item; build a dictionary, which stores values under named keys.
            item = {
                # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                "reference": reference,
                # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                "density": density,
                # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                "primary": reference == "dqn_governed" and density == 1.0,
            # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
            }
            # Compare one measured quantity at a time, keeping the scenarios paired.
            # for takes each item from ['crashed', 'mean_speed_m_s', 'distance_m', 'return'] in turn; metric holds the current item.
            for metric in ["crashed", "mean_speed_m_s", "distance_m", "return"]:
                # = unpacks the right-hand result into (_, matrix), matching values to names in order.
                _, matrix = paired_matrix(
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    data["qr_tail"], data[reference], density, metric
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                )
                # = stores a result in item[metric]; build a dictionary, which stores values under named keys.
                item[metric] = {
                    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                    "fixed_models": scenario_bootstrap_fixed_models(matrix),
                    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                    "exploratory_new_training": crossed_bootstrap(matrix),
                # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
                }
            # Call comparisons.append using parentheses to supply its inputs; this line does not store a separate return value.
            comparisons.append(item)
    # for takes each item from LABELS in turn; policy holds the current item.
    for policy in LABELS:
        # for takes each item from design['densities'] in turn; density holds the current item.
        for density in design["densities"]:
            # = stores a result in current; call next using parentheses to supply its inputs.
            current = next(r for r in groups[policy] if r["density"] == density)
            # Call dev.append using parentheses to supply its inputs; this line does not store a separate return value.
            dev.append(
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                {
                    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                    "policy": policy,
                    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                    "density": density,
                    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                    "development": development(policy, density),
                    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                    "confirmation": {
                        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                        "collision_fraction": current["crash_rate"],
                        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                        "speed_m_s": current["mean_speed_m_s"],
                    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
                    },
                # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
                }
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            )
    # = stores a result in primary; call next using parentheses to supply its inputs.
    primary = next(r for r in comparisons if r["primary"])
    # = stores a result in criterion; evaluate the expression on the right before using its result.
    criterion = (
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        primary["crashed"]["fixed_models"]["conditional_95_percent_interval"][1] < 0
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # = stores a result in result; build a dictionary, which stores values under named keys.
    result = {
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "design": design,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "groups": groups,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "contrasts": comparisons,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "primary_criterion_met": criterion,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "primary": primary,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "development_comparison": dev,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "interpretation": "Primary inference is conditional on six selected trained networks and frozen deployment settings. Residual speeds/action interventions remain; no claim of speed-independent benefit, new-training robustness or real-road safety. Other contrasts are secondary/exploratory.",
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "consumed_confirmation_seeds": [20000, 20099],
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "future_untouched_seeds_begin": 20100,
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
    # = stores a result in colors; build a dictionary, which stores values under named keys.
    colors = {
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "qr_tail": "C3",
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "dqn_governed": "C4",
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "qr_governed": "C5",
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "dqn_mean": "C1",
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "qr_mean": "C2",
    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
    }
    # = unpacks the right-hand result into (fig, axes), matching values to names in order.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    # for takes each item from LABELS in turn; p holds the current item.
    for p in LABELS:
        # = stores a result in rows; use square brackets to select part of groups.
        rows = groups[p]
        # = stores a result in x; collect or generate one result per item; the "for" part names the item being processed.
        x = [r["density"] for r in rows]
        # for takes each item from zip(axes, ['crash_rate', 'mean_speed_m_s', 'mean_distance_m'], [100, 1, 1], ['Episodes with colli... in turn; (ax, key, scale, ylabel) holds the current item.
        for ax, key, scale, ylabel in zip(
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
                [r[key] * scale for r in rows],
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "-o",
                # Inside this continued call, name=value supplies an option by name rather than by its position.
                color=colors[p],
                # Inside this continued call, name=value supplies an option by name rather than by its position.
                label=LABELS[p],
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            )
            # Call ax.set_xticks using parentheses to supply its inputs; this line does not store a separate return value.
            ax.set_xticks(x)
            # Call ax.set_xlabel using parentheses to supply its inputs; this line does not store a separate return value.
            ax.set_xlabel("Traffic density")
            # Call ax.set_ylabel using parentheses to supply its inputs; this line does not store a separate return value.
            ax.set_ylabel(ylabel)
            # Call ax.grid using parentheses to supply its inputs; this line does not store a separate return value.
            ax.grid(alpha=0.2)
    # Call axes[0].set_ylim using parentheses to supply its inputs; this line does not store a separate return value.
    axes[0].set_ylim(0, 105)
    # Call axes[0].legend using parentheses to supply its inputs; this line does not store a separate return value.
    axes[0].legend(fontsize=8)
    # Call fig.suptitle using parentheses to supply its inputs; this line does not store a separate return value.
    fig.suptitle(
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "Untouched confirmation scenarios: 100 scenario IDs, three fixed models per learner"
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # Call fig.tight_layout using parentheses to supply its inputs; this line does not store a separate return value.
    fig.tight_layout()
    # Save a figure for the local results page; this does not affect agent behavior.
    # Call fig.savefig using parentheses to supply its inputs; this line does not store a separate return value.
    fig.savefig(viewer / "confirmation_performance.png", dpi=150)
    # Call plt.close using parentheses to supply its inputs; this line does not store a separate return value.
    plt.close(fig)
    # = stores a result in rows; collect or generate one result per item; the "for" part names the item being processed.
    rows = [r for r in comparisons if r["density"] == 1.0]
    # = unpacks the right-hand result into (fig, axes), matching values to names in order.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # for takes each item from enumerate(rows) in turn; (index, r) holds the current item.
    for index, r in enumerate(rows):
        # for takes each item from zip(axes, ['crashed', 'mean_speed_m_s'], [100, 1]) in turn; (ax, metric, scale) holds the current item.
        for ax, metric, scale in zip(axes, ["crashed", "mean_speed_m_s"], [100, 1]):
            # for takes each item from [('fixed_models', -0.1, 'C0', 'Fixed-model scenario interval'), ('exploratory_new_training', 0.1,... in turn; (scope, offset, color, label) holds the current item.
            for scope, offset, color, label in [
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                ("fixed_models", -0.10, "C0", "Fixed-model scenario interval"),
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                (
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    "exploratory_new_training",
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    0.10,
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    "#888888",
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    "Exploratory training/scenario interval",
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                ),
            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
            ]:
                # = stores a result in v; use square brackets to select part of r[metric].
                v = r[metric][scope]
                # = stores a result in key; choose the value before "if" when its condition is true, otherwise the value after "else".
                key = (
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    "conditional_95_percent_interval"
                    # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                    if scope == "fixed_models"
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    else "exploratory_95_percent_interval"
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                )
                # = unpacks the right-hand result into (lo, hi), matching values to names in order.
                lo, hi = np.array(v[key]) * scale
                # = stores a result in mean; multiply the operands; NumPy and PyTorch arrays do this element by element.
                mean = v["mean_difference"] * scale
                # Call ax.errorbar using parentheses to supply its inputs; this line does not store a separate return value.
                ax.errorbar(
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    mean,
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    index + offset,
                    # Inside this continued call, name=value supplies an option by name rather than by its position.
                    xerr=[[max(0.0, mean - lo)], [max(0.0, hi - mean)]],
                    # Inside this continued call, name=value supplies an option by name rather than by its position.
                    fmt="o",
                    # Inside this continued call, name=value supplies an option by name rather than by its position.
                    color=color,
                    # Inside this continued call, name=value supplies an option by name rather than by its position.
                    capsize=4,
                    # Inside this continued call, name=value supplies an option by name rather than by its position.
                    label=label if index == 0 else None,
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                )
    # for takes each item from zip(axes, ['Collision difference (percentage points)', 'Speed difference (m/s)']) in turn; (ax, label) holds the current item.
    for ax, label in zip(
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        axes, ["Collision difference (percentage points)", "Speed difference (m/s)"]
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    ):
        # Call ax.axvline using parentheses to supply its inputs; this line does not store a separate return value.
        ax.axvline(0, color="black", lw=0.8)
        # Call ax.set_xlabel using parentheses to supply its inputs; this line does not store a separate return value.
        ax.set_xlabel(label)
        # Call ax.set_yticks using parentheses to supply its inputs; this line does not store a separate return value.
        ax.set_yticks(
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            range(4),
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            [LABELS[r["reference"]] + (" *" if r["primary"] else "") for r in rows],
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
        # Call ax.grid using parentheses to supply its inputs; this line does not store a separate return value.
        ax.grid(alpha=0.2)
    # = unpacks the right-hand result into (handles, legend_labels), matching values to names in order.
    handles, legend_labels = axes[0].get_legend_handles_labels()
    # Call fig.legend using parentheses to supply its inputs; this line does not store a separate return value.
    fig.legend(handles, legend_labels, fontsize=8, loc="lower center", ncol=2)
    # Call fig.suptitle using parentheses to supply its inputs; this line does not store a separate return value.
    fig.suptitle(
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "Normal traffic: cautious QR minus each reference; * = the single primary comparison"
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # Call fig.tight_layout using parentheses to supply its inputs; this line does not store a separate return value.
    fig.tight_layout(rect=[0, 0.09, 1, 0.95])
    # Save a figure for the local results page; this does not affect agent behavior.
    # Call fig.savefig using parentheses to supply its inputs; this line does not store a separate return value.
    fig.savefig(viewer / "confirmation_intervals.png", dpi=150)
    # Call plt.close using parentheses to supply its inputs; this line does not store a separate return value.
    plt.close(fig)
    # = stores a result in primary_stats; use square brackets to select part of primary['crashed'].
    primary_stats = primary["crashed"]["fixed_models"]
    # = stores a result in interval; use square brackets to select part of primary_stats.
    interval = primary_stats["conditional_95_percent_interval"]
    # = stores a result in speed; use square brackets to select part of primary['mean_speed_m_s']['fixed_models'].
    speed = primary["mean_speed_m_s"]["fixed_models"]["mean_difference"]
    # = stores a result in status; choose the value before "if" when its condition is true, otherwise the value after "else".
    status = "met" if criterion else "not met"
    # = stores a result in document; use quoted text as data; Python does not execute the words inside it.
    document = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Confirmation study · Risk-sensitive driving</title><style>body{font:17px/1.6 system-ui;color:#18283d;background:#f4f7fb;margin:auto;max-width:1160px;padding:36px}h1{font-size:38px;line-height:1.2}section{background:white;padding:24px;margin:24px 0;border-radius:14px}img{width:100%;height:auto}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;text-align:left;border-bottom:1px solid #dde3eb}th{background:#eef3fa}a{color:#175bbc}summary{cursor:pointer}</style></head><body><p>DEEP RL / LOCKED CONFIRMATION STUDY</p><h1>Do the development findings survive fresh scenarios?</h1><p>Six frozen mixed-density networks, five deployment rules, and no changes to training, checkpoints, risk level or speed thresholds after inspecting the reserved results. Scenario IDs 20000–20099 were verified unused before the design was locked.</p><p><a href="../confirmation/design.json">Locked design and model hashes</a> · <a href="../confirmation/preflight.json">Preflight audit</a> · <a href="../confirmation/summary.json">Full results</a> · <a href="../baseline_controls-report/index.html">Archived development comparisons</a> · <a href="../density_coverage-report/index.html">Training coverage and prediction diagnostics</a></p>"""
    # The combined operator updates document using its previous value.
    document += (
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "<section><h2>Single primary comparison: normal-traffic collisions</h2><p>The predeclared criterion is <strong>"
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + status
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + "</strong>: cautious QR minus DQN with its frozen 25 m/s governor is "
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + f"{100*primary_stats['mean_difference']:+.1f}"
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + " percentage points, with an approximate conditional 95% interval of "
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + str([round(100 * v, 1) for v in interval])
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + " pp.</p><p>The average-speed difference is "
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + f"{speed:+.2f}"
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + " m/s. This is a comparison of the locked policies; a collision advantage does not establish a speed-independent effect.</p><p>The criterion requires the upper endpoint to be strictly below zero. It concerns new scenarios for this selected set of networks, not new training runs.</p></section>"
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # = stores a result in normal_tail; call next using parentheses to supply its inputs.
    normal_tail = next(r for r in groups["qr_tail"] if r["density"] == 1.0)
    # = stores a result in normal_dqn; call next using parentheses to supply its inputs.
    normal_dqn = next(r for r in groups["dqn_governed"] if r["density"] == 1.0)
    # = stores a result in raw_text; evaluate the expression on the right before using its result.
    raw_text = f"Cautious QR: {100*normal_tail['crash_rate']:.1f}% collisions at {normal_tail['mean_speed_m_s']:.2f} m/s. Controlled DQN: {100*normal_dqn['crash_rate']:.1f}% at {normal_dqn['mean_speed_m_s']:.2f} m/s."
    # = stores a result in seed_text; combine the operands with the displayed operator; parentheses control grouping.
    seed_text = (
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "Per-model collision differences (QR minus controlled DQN) are "
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + str(
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            [round(100 * v, 1) for v in primary_stats["per_training_seed_differences"]]
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + " percentage points. The interval permits effects in either direction; equivalence was not tested."
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # The combined operator updates document using its previous value.
    document += (
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "<section><h2>Reading the primary result</h2><p>"
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + raw_text
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + "</p><p>"
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + seed_text
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + "</p></section>"
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # The combined operator updates document using its previous value.
    document += (
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        '<section><h2>Frozen-policy performance</h2><p>Each policy/density row contains 300 rollouts: three training models evaluated on the same 100 fresh scenario IDs. Those 300 observations are not treated as 300 independent scenario samples.</p><img src="confirmation_performance.png">'
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + table(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            [
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Policy",
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
                    LABELS[p],
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
                for p in LABELS
                # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                for r in groups[p]
            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
            ],
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + "</section>"
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # The combined operator updates document using its previous value.
    document += '<section><h2>Two scopes of uncertainty</h2><p>Blue intervals resample scenario IDs while keeping all trained networks and policy pairings together. They estimate uncertainty over fresh scenarios conditional on the selected models. Gray intervals also resample training seed IDs; with only three model pairs, these remain exploratory. Only the starred collision contrast is primary; other effects are descriptive secondary comparisons without family-wise adjustment.</p><img src="confirmation_intervals.png"></section>'
    # The combined operator updates document using its previous value.
    document += (
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "<section><h2>Cautious QR minus each reference</h2>"
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + table(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            [
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Role",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Reference",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Density",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Collision change pp",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Conditional scenario interval pp",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Exploratory crossed interval pp",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Speed change m/s",
            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
            ],
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            [
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                [
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    "Primary" if r["primary"] else "Secondary",
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    LABELS[r["reference"]],
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    r["density"],
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    f"{100*r['crashed']['fixed_models']['mean_difference']:+.1f}",
                    # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                    str(
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        [
                            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                            round(100 * v, 1)
                            # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                            for v in r["crashed"]["fixed_models"][
                                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                                "conditional_95_percent_interval"
                            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
                            ]
                        # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
                        ]
                    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                    ),
                    # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                    str(
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        [
                            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                            round(100 * v, 1)
                            # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                            for v in r["crashed"]["exploratory_new_training"][
                                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                                "exploratory_95_percent_interval"
                            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
                            ]
                        # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
                        ]
                    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                    ),
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    f"{r['mean_speed_m_s']['fixed_models']['mean_difference']:+.2f}",
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
    # The combined operator updates document using its previous value.
    document += (
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "<section><h2>Development versus confirmation</h2><p>Development used 30 scenarios per density/model and supported tuning and selection. Confirmation uses 100 fresh scenarios per density/model. The two blocks differ in initial traffic; these figures are shown without changing any policy.</p>"
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + table(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            [
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Policy",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Density",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Development collisions",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Confirmation collisions",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Development speed",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Confirmation speed",
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
                    f"{100*r['development']['collision_fraction']:.1f}%",
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    f"{100*r['confirmation']['collision_fraction']:.1f}%",
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    f"{r['development']['speed_m_s']:.2f}",
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    f"{r['confirmation']['speed_m_s']:.2f}",
                # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
                ]
                # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                for r in dev
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
        "<section><h2>Model variation</h2><details><summary>View every training seed</summary>"
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + table(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            [
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Policy",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Density",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Training seed",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Collisions",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Speed m/s",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Distance m",
            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
            ],
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            [
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                [
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    LABELS[p],
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    r["density"],
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    seed["training_seed"],
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    f"{100*seed['crash_rate']:.1f}%",
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    f"{seed['mean_speed_m_s']:.2f}",
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    f"{seed['mean_distance_m']:.0f}",
                # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
                ]
                # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                for p in LABELS
                # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                for r in groups[p]
                # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                for seed in r["per_seed"]
            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
            ],
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + "</details></section>"
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # The combined operator updates document using its previous value.
    document += (
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        '<section><h2>First reserved scenario, selected before outcomes</h2><p>Training seed 0, scenario 20000, density 1. This is one illustrative case, not evidence of typical performance.</p><div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px">'
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + "".join(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "<article><h3>"
            # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
            + LABELS[p]
            # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
            + '</h3><img alt="Locked policy driving replay" src="../confirmation/replay_'
            # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
            + p
            # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
            + '.gif"></article>'
            # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
            for p in ["qr_tail", "dqn_governed", "qr_governed"]
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + "</div></section>"
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # The combined operator updates document using its previous value.
    document += "<section><h2>Reproduce and retain the holdout boundary</h2><p><code>python run_confirmation.py --jobs 3</code> validates and reuses saved outputs. <code>--stage report</code> regenerates the report. Models and selection rules remain locked by hashes and recorded settings.</p><p>Seeds 20000–20099 are now consumed confirmation data for this locked study. Any further model tuning must use development blocks, and a later confirmation needs a new untouched block (20100 onward). Existing development viewers are historical snapshots from before this confirmation.</p><p>Observed speeds, action interventions, partial observations, simulator simplifications, forecast error and only three training seeds limit interpretation. Return quantiles are not collision probabilities. This experiment does not establish a globally CVaR-optimal policy or real-road safety.</p></section></body></html>"
    # Call (viewer / 'index.html').write_text using parentheses to supply its inputs; this line does not store a separate return value.
    (viewer / "index.html").write_text(document)
    # Call print using parentheses to supply its inputs; this line does not store a separate return value.
    print(
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        f'Primary criterion {status}: {100*primary_stats["mean_difference"]:+.1f} pp; conditional interval {[round(100*v,1) for v in interval]}; speed {speed:+.2f} m/s',
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        flush=True,
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # Call print using parentheses to supply its inputs; this line does not store a separate return value.
    print("Confirmation report saved", flush=True)


# Run the entry point only when this file is executed, not when it is imported.
# __name__ equals "__main__" when this file runs directly; this guard avoids running the script merely because another file imports it.
if __name__ == "__main__":
    # Call report using parentheses to supply its inputs; this line does not store a separate return value.
    report()
