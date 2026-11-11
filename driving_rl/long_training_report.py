"""Explain the 100K study using simple tables, plots, and paired comparisons.

Nothing in this file trains a model or chooses a better-looking checkpoint.
The two budgets face the same scenarios, so we subtract matching outcomes.
"""

# import makes these modules available; "as" gives a module a shorter local name.
import json
# from pathlib import brings the named tools into this file; parentheses allow several lines.
from pathlib import Path
# import makes these modules available; "as" gives a module a shorter local name.
import shutil

# import makes these modules available; "as" gives a module a shorter local name.
import matplotlib

# Call matplotlib.use using parentheses to supply its inputs; this line does not store a separate return value.
matplotlib.use("Agg")
# import makes these modules available; "as" gives a module a shorter local name.
import matplotlib.pyplot as plt

# from analysis import brings the named tools into this file; parentheses allow several lines.
from .analysis import summarize_group, paired_matrix, scenario_bootstrap_fixed_models
# from density_coverage_report import brings the named tools into this file; parentheses allow several lines.
from .density_coverage_report import table
# from experiment import brings the named tools into this file; parentheses allow several lines.
from .experiment import save_json

# = stores a result in ROOT; use square brackets to select part of Path(__file__).resolve().parents.
ROOT = Path(__file__).resolve().parents[1]
# = stores a result in OUTPUT; apply / to these operands; numbers are divided, while Path objects join folder names.
OUTPUT = ROOT / "results/long_training_evaluation"
# = stores a result in LABELS; build a dictionary, which stores values under named keys.
LABELS = {
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "qr_tail": "QR lower quarter",
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "dqn_governed": "DQN + governor",
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "qr_governed": "QR mean + governor",
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "dqn_mean": "DQN mean",
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "qr_mean": "QR mean",
# } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
}


# def defines read; parentheses list its inputs, and the colon starts its indented body.
def read(path):
    # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
    """Turn a saved JSON file into Python lists and dictionaries."""
    # return ends this function and sends its result back to the code that called it.
    return json.loads(path.read_text())


# def defines compare; parentheses list its inputs, and the colon starts its indented body.
def compare(left, right, density):
    """Subtract matching cases, then estimate uncertainty over fresh scenarios.

    The trained models stay fixed. Resample scenario IDs, not individual cells,
    because the same starting traffic is shared across the models.
    """
    # = stores a result in result; build a dictionary, which stores values under named keys.
    result = {}
    # Compare one quantity at a time, using matching model and scenario IDs.
    # for takes each item from ['crashed', 'mean_speed_m_s', 'distance_m', 'return'] in turn; metric holds the current item.
    for metric in ["crashed", "mean_speed_m_s", "distance_m", "return"]:
        # = unpacks the right-hand result into (seeds, differences), matching values to names in order.
        seeds, differences = paired_matrix(left, right, density, metric)
        # = stores a result in result[metric]; call scenario_bootstrap_fixed_models using parentheses to supply its inputs.
        result[metric] = scenario_bootstrap_fixed_models(differences)
    # return ends this function and sends its result back to the code that called it.
    return result


# def defines load_and_compare_results; parentheses list its inputs, and the colon starts its indented body.
def load_and_compare_results():
    # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
    """Read matching cases, calculate differences, and save the full numerical summary."""
    # = stores a result in design; call read using parentheses to supply its inputs.
    design = read(OUTPUT / "design.json")
    # = stores a result in data; build a dictionary, which stores values under named keys.
    data = {}
    # = stores a result in groups; build a dictionary, which stores values under named keys.
    groups = {}
    # = stores a result in budget_effects; build a list, keeping its items in order.
    budget_effects = []
    # = stores a result in policy_effects; build a list, keeping its items in order.
    policy_effects = []
    # Each budget gets its own results. Keep 40K and 100K separate until the paired comparison.
    # for takes each item from [40000, 100000] in turn; budget holds the current item.
    for budget in [40000, 100000]:
        # = stores a result in data[budget]; build a dictionary, which stores values under named keys.
        data[budget] = {}
        # = stores a result in groups[budget]; build a dictionary, which stores values under named keys.
        groups[budget] = {}
        # A policy is a decision rule. Several rules can share the very same QR-DQN network.
        # for takes each item from LABELS in turn; policy holds the current item.
        for policy in LABELS:
            # = stores a result in runs; build a list, keeping its items in order.
            runs = []
            # A training seed identifies one independently trained model, not a new traffic scenario.
            # for takes each item from range(3) in turn; seed holds the current item.
            for seed in range(3):
                # = stores a result in path; apply / to these operands; numbers are divided, while Path objects join folder names.
                path = OUTPUT / f"eval_{budget}_{policy}_seed{seed}.json"
                # Call runs.append using parentheses to supply its inputs; this line does not store a separate return value.
                runs.append(read(path))
            # = stores a result in data[budget][policy]; evaluate the expression on the right before using its result.
            data[budget][policy] = runs
            # = stores a result in groups[budget][policy]; call summarize_group using parentheses to supply its inputs.
            groups[budget][policy] = summarize_group(runs)
    # Compare budgets using matching scenario/model IDs, rather than comparing
    # the old 20000 block with the new 20100 block.
    # for takes each item from LABELS in turn; policy holds the current item.
    for policy in LABELS:
        # for takes each item from design['densities'] in turn; density holds the current item.
        for density in design["densities"]:
            # = stores a result in effect; call compare using parentheses to supply its inputs.
            effect = compare(data[100000][policy], data[40000][policy], density)
            # Call budget_effects.append using parentheses to supply its inputs; this line does not store a separate return value.
            budget_effects.append({"policy": policy, "density": density, **effect})
    # for takes each item from ['dqn_governed', 'qr_governed', 'dqn_mean', 'qr_mean'] in turn; reference holds the current item.
    for reference in ["dqn_governed", "qr_governed", "dqn_mean", "qr_mean"]:
        # for takes each item from design['densities'] in turn; density holds the current item.
        for density in design["densities"]:
            # = stores a result in effect; call compare using parentheses to supply its inputs.
            effect = compare(data[100000]["qr_tail"], data[100000][reference], density)
            # Call policy_effects.append using parentheses to supply its inputs; this line does not store a separate return value.
            policy_effects.append(
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                {"reference": reference, "density": density, **effect}
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            )
    # Find the one comparison named in the recorded design.
    # = stores a result in primary; use the literal value None.
    primary = None
    # for takes each item from policy_effects in turn; comparison holds the current item.
    for comparison in policy_effects:
        # if checks comparison['reference'] == 'dqn_governed' and comparison['density'] == 1.0; only a true condition runs the indented block.
        if comparison["reference"] == "dqn_governed" and comparison["density"] == 1.0:
            # = stores a result in primary; evaluate the expression on the right before using its result.
            primary = comparison
            # break leaves the nearest loop immediately; execution continues after that loop.
            break
    # if checks primary is None; only a true condition runs the indented block.
    if primary is None:
        # raise signals an error and stops this normal path, so invalid data are not silently accepted.
        raise StopIteration
    # = stores a result in upper; use square brackets to select part of primary['crashed']['conditional_95_percent_interval'].
    upper = primary["crashed"]["conditional_95_percent_interval"][1]
    # Use the rule recorded before evaluating the fresh scenarios; do not move the goalposts afterward.
    # = stores a result in criterion_met; evaluate the expression on the right before using its result.
    criterion_met = upper < 0
    # = stores a result in summary; build a dictionary, which stores values under named keys.
    summary = {
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "design": design,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "groups": groups,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "budget_effects": budget_effects,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "policy_effects_100k": policy_effects,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "primary": primary,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "primary_criterion_met": criterion_met,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "prefix_verification": read(OUTPUT / "prefix_verification.json"),
    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
    }
    # JSON object keys become text when saved; this is normal and intentional.
    # Call save_json using parentheses to supply its inputs; this line does not store a separate return value.
    save_json(OUTPUT / "summary.json", summary)
    # return ends this function and sends its result back to the code that called it.
    return summary


# def defines draw_budget_comparison; parentheses list its inputs, and the colon starts its indented body.
def draw_budget_comparison(groups):
    # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
    """Plot the measured outcomes at both budgets. This is display code, not training."""
    # = stores a result in viewer; apply / to these operands; numbers are divided, while Path objects join folder names.
    viewer = ROOT / "results/report"
    # Call plt.rcparams.update using parentheses to supply its inputs; this line does not store a separate return value.
    plt.rcParams.update(
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # A figure is the whole image; axes are its individual plotting panels.
    # = unpacks the right-hand result into (figure, axes), matching values to names in order.
    figure, axes = plt.subplots(2, 3, figsize=(14, 8))
    # for takes each item from enumerate(['qr_tail', 'dqn_governed', 'qr_governed']) in turn; (column, policy) holds the current item.
    for column, policy in enumerate(["qr_tail", "dqn_governed", "qr_governed"]):
        # for takes each item from [(40000, 'C0'), (100000, 'C1')] in turn; (budget, color) holds the current item.
        for budget, color in [(40000, "C0"), (100000, "C1")]:
            # = stores a result in rows; use square brackets to select part of groups[budget].
            rows = groups[budget][policy]
            # = stores a result in densities; build a list, keeping its items in order.
            densities = []
            # for takes each item from rows in turn; row holds the current item.
            for row in rows:
                # Call densities.append using parentheses to supply its inputs; this line does not store a separate return value.
                densities.append(row["density"])
            # for takes each item from [(0, 'crash_rate', 100), (1, 'mean_speed_m_s', 1)] in turn; (row_number, metric, scale) holds the current item.
            for row_number, metric, scale in [
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                (0, "crash_rate", 100),
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                (1, "mean_speed_m_s", 1),
            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
            ]:
                # = stores a result in axis; use square brackets to select part of axes.
                axis = axes[row_number, column]
                # = stores a result in values; build a list, keeping its items in order.
                values = []
                # for takes each item from rows in turn; row holds the current item.
                for row in rows:
                    # Call values.append using parentheses to supply its inputs; this line does not store a separate return value.
                    values.append(row[metric] * scale)
                # Call axis.plot using parentheses to supply its inputs; this line does not store a separate return value.
                axis.plot(
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    densities,
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    values,
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    "-o",
                    # Inside this continued call, name=value supplies an option by name rather than by its position.
                    color=color,
                    # Inside this continued call, name=value supplies an option by name rather than by its position.
                    label=f"{budget // 1000}K steps",
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                )
                # Call axis.set_xticks using parentheses to supply its inputs; this line does not store a separate return value.
                axis.set_xticks(densities)
                # Call axis.set_xlabel using parentheses to supply its inputs; this line does not store a separate return value.
                axis.set_xlabel("Traffic density")
                # Call axis.grid using parentheses to supply its inputs; this line does not store a separate return value.
                axis.grid(alpha=0.2)
        # Call axes[0, column].set_title using parentheses to supply its inputs; this line does not store a separate return value.
        axes[0, column].set_title(LABELS[policy])
        # Call axes[0, column].set_ylim using parentheses to supply its inputs; this line does not store a separate return value.
        axes[0, column].set_ylim(0, 105)
        # Call axes[0, column].legend using parentheses to supply its inputs; this line does not store a separate return value.
        axes[0, column].legend()
    # Call axes[0, 0].set_ylabel using parentheses to supply its inputs; this line does not store a separate return value.
    axes[0, 0].set_ylabel("Episodes with collision (%)")
    # Call axes[1, 0].set_ylabel using parentheses to supply its inputs; this line does not store a separate return value.
    axes[1, 0].set_ylabel("Mean episode speed (m/s)")
    # Call figure.suptitle using parentheses to supply its inputs; this line does not store a separate return value.
    figure.suptitle(
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "Same fresh scenarios, different training budgets; three models per learner"
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # Call figure.tight_layout using parentheses to supply its inputs; this line does not store a separate return value.
    figure.tight_layout()
    # Save the graph for the local report. Drawing it does not change any measured result.
    # Call figure.savefig using parentheses to supply its inputs; this line does not store a separate return value.
    figure.savefig(viewer / "long_training_comparison.png", dpi=150)
    # Call plt.close using parentheses to supply its inputs; this line does not store a separate return value.
    plt.close(figure)


# def defines build_budget_tables; parentheses list its inputs, and the colon starts its indented body.
def build_budget_tables(groups):
    # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
    """List the measured outcomes at each budget. This table does not choose a winner."""
    # = stores a result in page; use quoted text as data; Python does not execute the words inside it.
    page = ""
    # Each budget gets its own results. Keep 40K and 100K separate until the paired comparison.
    # for takes each item from [40000, 100000] in turn; budget holds the current item.
    for budget in [40000, 100000]:
        # = stores a result in rows; build a list, keeping its items in order.
        rows = []
        # A policy is a decision rule. Several rules can share the very same QR-DQN network.
        # for takes each item from LABELS in turn; policy holds the current item.
        for policy in LABELS:
            # for takes each item from groups[budget][policy] in turn; result holds the current item.
            for result in groups[budget][policy]:
                # Call rows.append using parentheses to supply its inputs; this line does not store a separate return value.
                rows.append(
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    [
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        LABELS[policy],
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        result["density"],
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        f"{100 * result['crash_rate']:.1f}%",
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        f"{result['mean_speed_m_s']:.2f}",
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        f"{result['mean_distance_m']:.0f}",
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        f"{result['mean_return']:.2f}",
                    # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
                    ]
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                )
        # The combined operator updates page using its previous value.
        page += f"<section><h2>{budget // 1000}K results on the new block</h2>"
        # The combined operator updates page using its previous value.
        page += (
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            table(
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
                # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
                ],
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                rows,
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            )
            # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
            + "</section>"
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
    # return ends this function and sends its result back to the code that called it.
    return page


# def defines build_budget_effect_table; parentheses list its inputs, and the colon starts its indented body.
def build_budget_effect_table(budget_effects):
    # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
    """Show 100K minus 40K. Negative collision differences favor the longer budget."""
    # = stores a result in page; use quoted text as data; Python does not execute the words inside it.
    page = ""
    # Negative collision changes favor the longer training budget. Show speed
    # too: a reduced collision count might accompany slower driving.
    # = stores a result in rows; build a list, keeping its items in order.
    rows = []
    # for takes each item from budget_effects in turn; result holds the current item.
    for result in budget_effects:
        # = stores a result in crash; use square brackets to select part of result.
        crash = result["crashed"]
        # Convert this comparison interval into percentage points for the results table.
        # = stores a result in budget_interval_percent; build a list, keeping its items in order.
        budget_interval_percent = []
        # for takes each item from crash['conditional_95_percent_interval'] in turn; x holds the current item.
        for x in crash["conditional_95_percent_interval"]:
            # Call budget_interval_percent.append using parentheses to supply its inputs; this line does not store a separate return value.
            budget_interval_percent.append(round(x * 100, 1))
        # Show each model pair too: a good average can hide differences between training seeds.
        # = stores a result in seed_changes_percent; build a list, keeping its items in order.
        seed_changes_percent = []
        # for takes each item from crash['per_training_seed_differences'] in turn; x holds the current item.
        for x in crash["per_training_seed_differences"]:
            # Call seed_changes_percent.append using parentheses to supply its inputs; this line does not store a separate return value.
            seed_changes_percent.append(round(x * 100, 1))
        # Call rows.append using parentheses to supply its inputs; this line does not store a separate return value.
        rows.append(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            [
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                LABELS[result["policy"]],
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                result["density"],
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                f"{100 * crash['mean_difference']:+.1f}",
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                str(budget_interval_percent),
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                f"{result['mean_speed_m_s']['mean_difference']:+.2f}",
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                str(seed_changes_percent),
            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
            ]
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
    # The combined operator updates page using its previous value.
    page += "<section><h2>100K minus 40K on matching cases</h2><p>These budget comparisons are secondary. Intervals are conditional on the fixed model set and are not adjusted for multiple comparisons.</p>"
    # The combined operator updates page using its previous value.
    page += (
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        table(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            [
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Policy",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Density",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Collision change pp",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Scenario interval pp",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Speed change m/s",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Per-model collision changes pp",
            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
            ],
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            rows,
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + "</section>"
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # return ends this function and sends its result back to the code that called it.
    return page


# def defines build_policy_effect_table; parentheses list its inputs, and the colon starts its indented body.
def build_policy_effect_table(policy_effects):
    # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
    """Show cautious QR minus each reference, keeping speed differences visible."""
    # = stores a result in page; use quoted text as data; Python does not execute the words inside it.
    page = ""
    # = stores a result in rows; build a list, keeping its items in order.
    rows = []
    # for takes each item from policy_effects in turn; result holds the current item.
    for result in policy_effects:
        # = stores a result in crash; use square brackets to select part of result.
        crash = result["crashed"]
        # = stores a result in policy_interval_percent; build a list, keeping its items in order.
        policy_interval_percent = []
        # for takes each item from crash['conditional_95_percent_interval'] in turn; x holds the current item.
        for x in crash["conditional_95_percent_interval"]:
            # Call policy_interval_percent.append using parentheses to supply its inputs; this line does not store a separate return value.
            policy_interval_percent.append(round(x * 100, 1))
        # Call rows.append using parentheses to supply its inputs; this line does not store a separate return value.
        rows.append(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            [
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                LABELS[result["reference"]],
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                result["density"],
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                f"{100 * crash['mean_difference']:+.1f}",
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                str(policy_interval_percent),
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                f"{result['mean_speed_m_s']['mean_difference']:+.2f}",
            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
            ]
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
    # The combined operator updates page using its previous value.
    page += "<section><h2>100K cautious QR minus each reference</h2>"
    # The combined operator updates page using its previous value.
    page += (
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        table(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            [
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Reference",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Density",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Collision change pp",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Scenario interval pp",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Speed change m/s",
            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
            ],
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            rows,
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + "</section>"
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # return ends this function and sends its result back to the code that called it.
    return page


# def defines write_report_page; parentheses list its inputs, and the colon starts its indented body.
def write_report_page(summary):
    # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
    """Build tables from the saved comparisons. The original primary criterion stays unchanged."""
    # = stores a result in viewer; apply / to these operands; numbers are divided, while Path objects join folder names.
    viewer = ROOT / "results/report"
    # = stores a result in groups; use square brackets to select part of summary.
    groups = summary["groups"]
    # = stores a result in budget_effects; use square brackets to select part of summary.
    budget_effects = summary["budget_effects"]
    # = stores a result in policy_effects; use square brackets to select part of summary.
    policy_effects = summary["policy_effects_100k"]
    # = stores a result in primary; use square brackets to select part of summary.
    primary = summary["primary"]
    # Use the rule recorded before evaluating the fresh scenarios; do not move the goalposts afterward.
    # = stores a result in criterion_met; use square brackets to select part of summary.
    criterion_met = summary["primary_criterion_met"]
    # = stores a result in interval; use square brackets to select part of primary['crashed'].
    interval = primary["crashed"]["conditional_95_percent_interval"]
    # = stores a result in difference; multiply the operands; NumPy and PyTorch arrays do this element by element.
    difference = primary["crashed"]["mean_difference"] * 100
    # = stores a result in speed; use square brackets to select part of primary['mean_speed_m_s'].
    speed = primary["mean_speed_m_s"]["mean_difference"]
    # = stores a result in status; choose the value before "if" when its condition is true, otherwise the value after "else".
    status = "met" if criterion_met else "not met"
    # The long string below is page layout, not reinforcement learning. Later lines insert the measured numbers.
    # = stores a result in page; use quoted text as data; Python does not execute the words inside it.
    page = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Longer training · Risk-sensitive driving</title><style>body{font:17px/1.6 system-ui;color:#18283d;background:#f4f7fb;margin:auto;max-width:1160px;padding:36px}h1{font-size:38px;line-height:1.2}section{background:white;padding:24px;margin:24px 0;border-radius:14px}img{width:100%;height:auto}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;text-align:left;border-bottom:1px solid #dde3eb}th{background:#eef3fa}a{color:#175bbc}</style></head><body><p>DEEP RL / LONGER TRAINING</p><h1>Did 100,000 training steps help?</h1><p>Six completed 100K runs are evaluated alongside their 40K versions on exactly the same fresh scenarios. No new training, risk-level selection, governor tuning or checkpoint selection happens in this study.</p><p><a href="../long_training_evaluation/design.json">Locked design</a> · <a href="../long_training_evaluation/summary.json">Full numerical results</a> · <a href="../confirmation_40k-report/index.html">Earlier confirmation study</a></p>"""
    # The combined operator updates page using its previous value.
    page += "<section><h2>Primary 100K comparison</h2>"
    # Convert fractions such as 0.05 into percentage points such as 5.0 for the reader.
    # = stores a result in primary_interval_percent; build a list, keeping its items in order.
    primary_interval_percent = []
    # for takes each item from interval in turn; x holds the current item.
    for x in interval:
        # Call primary_interval_percent.append using parentheses to supply its inputs; this line does not store a separate return value.
        primary_interval_percent.append(round(x * 100, 1))
    # The combined operator updates page using its previous value.
    page += f"<p>The predeclared normal-traffic collision criterion is <strong>{status}</strong>. Cautious QR minus governed DQN: {difference:+.1f} percentage points; conditional 95% interval {primary_interval_percent} pp. Speed difference: {speed:+.2f} m/s.</p>"
    # The combined operator updates page using its previous value.
    page += "<p>The interval estimates scenario uncertainty for these fixed models. It does not establish equivalence, speed-independent benefit, or reliability across new training runs.</p></section>"
    # The combined operator updates page using its previous value.
    page += (
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "<section><h2>Interpreting the speed control</h2><p>The 25 m/s governor was frozen before this study. Its resulting average speed need not match a newly trained network. Here cautious QR differs from controlled DQN by "
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + f"{speed:+.2f}"
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + " m/s in normal traffic. This comparison is therefore not evidence that a risk rule alone caused a collision difference.</p><p>A primary criterion that is not met means the study has not established the specified advantage. It does not establish statistical equivalence.</p></section>"
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # The combined operator updates page using its previous value.
    page += '<section><h2>Paired budget comparison</h2><p>Both budgets use seeds 20100–20199: 100 scenario IDs and three trained models per policy. Those 300 rollouts are not treated as 300 independent starting scenarios. All six 40K network prefixes exactly match the previous trained models.</p><img src="long_training_comparison.png"></section>'
    # The combined operator updates page using its previous value.
    page += build_budget_tables(groups)
    # The combined operator updates page using its previous value.
    page += build_budget_effect_table(budget_effects)
    # The combined operator updates page using its previous value.
    page += build_policy_effect_table(policy_effects)
    # The combined operator updates page using its previous value.
    page += "<section><h2>Reproduce and interpret</h2><p><code>python run_long_training_evaluation.py --jobs 4</code> validates and reuses saved evaluations. <code>--report-only</code> rebuilds this page.</p><p>The primary comparison is governed DQN versus cautious QR at density 1 with 100K models. All other contrasts are secondary/exploratory. Policies share the configured training-density mixture, but their experiences differ. Only three training seeds are available.</p><p>Seeds 20100–20199 are now consumed evaluation data, alongside the earlier 20000–20099 block. Any later tuning requires a new untouched confirmation block beginning at 20200. Partial observations, prediction error and simplified traffic limit conclusions; no real-road safety guarantee is established.</p></section></body></html>"
    # Call (viewer / 'index.html').write_text using parentheses to supply its inputs; this line does not store a separate return value.
    (viewer / "index.html").write_text(page)
    # Call print using parentheses to supply its inputs; this line does not store a separate return value.
    print(
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        f"100K primary criterion {status}: {difference:+.1f} pp; interval {primary_interval_percent}; speed {speed:+.2f} m/s",
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        flush=True,
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )


# def defines report; parentheses list its inputs, and the colon starts its indented body.
def report():
    # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
    """Keep the earlier page, then read, compare, draw, and write."""
    # = stores a result in viewer; apply / to these operands; numbers are divided, while Path objects join folder names.
    viewer = ROOT / "results/report"
    # = stores a result in backup; apply / to these operands; numbers are divided, while Path objects join folder names.
    backup = ROOT / "results/confirmation_40k-report"
    # if checks not backup.exists(); only a true condition runs the indented block.
    if not backup.exists():
        # Call shutil.copytree using parentheses to supply its inputs; this line does not store a separate return value.
        shutil.copytree(viewer, backup)
    # = stores a result in summary; call load_and_compare_results using parentheses to supply its inputs.
    summary = load_and_compare_results()
    # Call draw_budget_comparison using parentheses to supply its inputs; this line does not store a separate return value.
    draw_budget_comparison(summary["groups"])
    # Call write_report_page using parentheses to supply its inputs; this line does not store a separate return value.
    write_report_page(summary)


# __name__ equals "__main__" when this file runs directly; this guard avoids running the script merely because another file imports it.
if __name__ == "__main__":
    # Call report using parentheses to supply its inputs; this line does not store a separate return value.
    report()
