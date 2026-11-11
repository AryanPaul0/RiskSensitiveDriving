"""Explain forecast accuracy without confusing it with driving performance.

All results here are development diagnostics. No model is trained or selected.
"""

# from pathlib import brings the named tools into this file; parentheses allow several lines.
from pathlib import Path
# import makes these modules available; "as" gives a module a shorter local name.
import json
# import makes these modules available; "as" gives a module a shorter local name.
import numpy as np
# import makes these modules available; "as" gives a module a shorter local name.
import matplotlib

# Call matplotlib.use using parentheses to supply its inputs; this line does not store a separate return value.
matplotlib.use("Agg")
# import makes these modules available; "as" gives a module a shorter local name.
import matplotlib.pyplot as plt
# from forecast_metrics import brings the named tools into this file; parentheses allow several lines.
from .forecast_metrics import summarize_predictions
# from density_coverage_report import brings the named tools into this file; parentheses allow several lines.
from .density_coverage_report import table
# from experiment import brings the named tools into this file; parentheses allow several lines.
from .experiment import save_json

# = stores a result in ROOT; use square brackets to select part of Path(__file__).resolve().parents.
ROOT = Path(__file__).resolve().parents[1]
# = stores a result in OUTPUT; apply / to these operands; numbers are divided, while Path objects join folder names.
OUTPUT = ROOT / "results/prediction_audit"
# = stores a result in LABELS; build a dictionary, which stores values under named keys.
LABELS = {"dqn_mean": "DQN mean", "qr_mean": "QR mean", "qr_tail": "QR lower quarter"}
# = stores a result in DENSITIES; build a list, keeping its items in order.
DENSITIES = [1.0, 1.5, 2.0, 2.5]


# def defines find_summary; parentheses list its inputs, and the colon starts its indented body.
def find_summary(summaries, budget, policy, density):
    """Look through saved summaries until we find the requested group.

    Check all three labels: budget, policy, and density. A missing group is
    still an error, not a made-up zero result.
    """
    # for takes each item from summaries in turn; summary holds the current item.
    for summary in summaries:
        # if checks summary['budget'] == budget; only a true condition runs the indented block.
        if summary["budget"] == budget:
            # if checks summary['policy'] == policy; only a true condition runs the indented block.
            if summary["policy"] == policy:
                # if checks summary['density'] == density; only a true condition runs the indented block.
                if summary["density"] == density:
                    # return ends this function and sends its result back to the code that called it.
                    return summary
    # raise signals an error and stops this normal path, so invalid data are not silently accepted.
    raise StopIteration


# def defines load_and_summarize_results; parentheses list its inputs, and the colon starts its indented body.
def load_and_summarize_results():
    # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
    """Read the saved rollouts and keep each policy/budget/density group separate."""
    # = stores a result in summaries; build a list, keeping its items in order.
    summaries = []
    # Each budget gets its own results. Keep 40K and 100K separate until the paired comparison.
    # for takes each item from [40000, 100000] in turn; budget holds the current item.
    for budget in [40000, 100000]:
        # A policy is a decision rule. Several rules can share the very same QR-DQN network.
        # for takes each item from LABELS in turn; policy holds the current item.
        for policy in LABELS:
            # = stores a result in all_rows; build a list, keeping its items in order.
            all_rows = []
            # A training seed identifies one independently trained model, not a new traffic scenario.
            # for takes each item from range(3) in turn; seed holds the current item.
            for seed in range(3):
                # = stores a result in path; apply / to these operands; numbers are divided, while Path objects join folder names.
                path = OUTPUT / f"audit_{budget}_{policy}_seed{seed}.json"
                # = stores a result in data; call json.loads using parentheses to supply its inputs.
                data = json.loads(path.read_text())
                # Call all_rows.extend using parentheses to supply its inputs; this line does not store a separate return value.
                all_rows.extend(data["episodes"])
            # Traffic density changes vehicle spacing. It does not add more network inputs.
            # for takes each item from DENSITIES in turn; density holds the current item.
            for density in DENSITIES:
                # = stores a result in rows; build a list, keeping its items in order.
                rows = []
                # for takes each item from all_rows in turn; row holds the current item.
                for row in all_rows:
                    # if checks row['density'] == density; only a true condition runs the indented block.
                    if row["density"] == density:
                        # Call rows.append using parentheses to supply its inputs; this line does not store a separate return value.
                        rows.append(row)
                # Call summaries.append using parentheses to supply its inputs; this line does not store a separate return value.
                summaries.append(
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    {
                        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                        "budget": budget,
                        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                        "policy": policy,
                        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                        "density": density,
                        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
                        **summarize_predictions(rows),
                    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
                    }
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                )
    # Call save_json using parentheses to supply its inputs; this line does not store a separate return value.
    save_json(
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        OUTPUT / "summary.json",
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        {
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "groups": summaries,
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "scope": "Initial chosen-action predictions on development scenarios. Error ranges account for cutoff continuation, not sampling uncertainty. Policy changes also change the outcomes being predicted.",
        # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
        },
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # return ends this function and sends its result back to the code that called it.
    return summaries


# def defines draw_error_comparison; parentheses list its inputs, and the colon starts its indented body.
def draw_error_comparison(summaries):
    # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
    """Draw signed and absolute errors. Whiskers bound missing future rewards, not sample noise."""
    # Call plt.rcparams.update using parentheses to supply its inputs; this line does not store a separate return value.
    plt.rcParams.update(
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # A figure is the whole image; axes are its individual plotting panels.
    # = unpacks the right-hand result into (figure, axes), matching values to names in order.
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), sharey="row")
    # for takes each item from enumerate(LABELS) in turn; (column, policy) holds the current item.
    for column, policy in enumerate(LABELS):
        # for takes each item from [(40000, 'C0', -0.03), (100000, 'C1', 0.03)] in turn; (budget, color, offset) holds the current item.
        for budget, color, offset in [(40000, "C0", -0.03), (100000, "C1", 0.03)]:
            # = stores a result in rows; build a list, keeping its items in order.
            rows = []
            # for takes each item from summaries in turn; summary holds the current item.
            for summary in summaries:
                # if checks summary['budget'] == budget and summary['policy'] == policy; only a true condition runs the indented block.
                if summary["budget"] == budget and summary["policy"] == policy:
                    # Call rows.append using parentheses to supply its inputs; this line does not store a separate return value.
                    rows.append(summary)
            # for takes each item from [(0, 'bias_bounds'), (1, 'absolute_error_bounds')] in turn; (row_number, metric) holds the current item.
            for row_number, metric in [
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                (0, "bias_bounds"),
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                (1, "absolute_error_bounds"),
            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
            ]:
                # = stores a result in lower_error_limits; build a list, keeping its items in order.
                lower_error_limits = []
                # for takes each item from rows in turn; row holds the current item.
                for row in rows:
                    # Call lower_error_limits.append using parentheses to supply its inputs; this line does not store a separate return value.
                    lower_error_limits.append(row[metric][0])
                # = stores a result in lows; call np.array using parentheses to supply its inputs.
                lows = np.array(lower_error_limits)
                # = stores a result in upper_error_limits; build a list, keeping its items in order.
                upper_error_limits = []
                # for takes each item from rows in turn; row holds the current item.
                for row in rows:
                    # Call upper_error_limits.append using parentheses to supply its inputs; this line does not store a separate return value.
                    upper_error_limits.append(row[metric][1])
                # = stores a result in highs; call np.array using parentheses to supply its inputs.
                highs = np.array(upper_error_limits)
                # = stores a result in centers; apply / to these operands; numbers are divided, while Path objects join folder names.
                centers = (lows + highs) / 2
                # = stores a result in axis; use square brackets to select part of axes.
                axis = axes[row_number, column]
                # Whiskers here describe a range. Read the plot label: they are not automatically statistical confidence intervals.
                # Call axis.errorbar using parentheses to supply its inputs; this line does not store a separate return value.
                axis.errorbar(
                    # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                    np.array(DENSITIES) + offset,
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    centers,
                    # Inside this continued call, name=value supplies an option by name rather than by its position.
                    yerr=[centers - lows, highs - centers],
                    # Inside this continued call, name=value supplies an option by name rather than by its position.
                    fmt="o-",
                    # Inside this continued call, name=value supplies an option by name rather than by its position.
                    color=color,
                    # Inside this continued call, name=value supplies an option by name rather than by its position.
                    capsize=4,
                    # Inside this continued call, name=value supplies an option by name rather than by its position.
                    label=f"{budget // 1000}K",
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                )
                # Call axis.axhline using parentheses to supply its inputs; this line does not store a separate return value.
                axis.axhline(0, color="black", linewidth=0.7)
                # Call axis.set_xticks using parentheses to supply its inputs; this line does not store a separate return value.
                axis.set_xticks(DENSITIES)
                # Call axis.set_xlabel using parentheses to supply its inputs; this line does not store a separate return value.
                axis.set_xlabel("Traffic density")
                # Call axis.grid using parentheses to supply its inputs; this line does not store a separate return value.
                axis.grid(alpha=0.2)
        # Call axes[0, column].set_title using parentheses to supply its inputs; this line does not store a separate return value.
        axes[0, column].set_title(LABELS[policy])
        # Call axes[0, column].legend using parentheses to supply its inputs; this line does not store a separate return value.
        axes[0, column].legend()
    # Call axes[0, 0].set_ylabel using parentheses to supply its inputs; this line does not store a separate return value.
    axes[0, 0].set_ylabel("Prediction minus return")
    # Call axes[1, 0].set_ylabel using parentheses to supply its inputs; this line does not store a separate return value.
    axes[1, 0].set_ylabel("Mean absolute prediction error")
    # Call figure.suptitle using parentheses to supply its inputs; this line does not store a separate return value.
    figure.suptitle(
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "Missing-continuation bounds only; whiskers are NOT sampling confidence intervals"
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # Call figure.tight_layout using parentheses to supply its inputs; this line does not store a separate return value.
    figure.tight_layout()
    # Save the graph for the local report. Drawing it does not change any measured result.
    # Call figure.savefig using parentheses to supply its inputs; this line does not store a separate return value.
    figure.savefig(OUTPUT / "forecast_errors.png", dpi=150)
    # Call plt.close using parentheses to supply its inputs; this line does not store a separate return value.
    plt.close(figure)


# def defines draw_quantile_checks; parentheses list its inputs, and the colon starts its indented body.
def draw_quantile_checks(summaries):
    # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
    """Draw QR distribution checks. A DQN scalar is never drawn as a quantile distribution."""
    # A quantile curve is meaningful for QR-DQN, not for DQN's scalar mean.
    # A figure is the whole image; axes are its individual plotting panels.
    # = unpacks the right-hand result into (figure, axes), matching values to names in order.
    figure, axes = plt.subplots(2, 4, figsize=(15, 8), sharex=True, sharey=True)
    # for takes each item from enumerate(['qr_mean', 'qr_tail']) in turn; (row_number, policy) holds the current item.
    for row_number, policy in enumerate(["qr_mean", "qr_tail"]):
        # for takes each item from enumerate(DENSITIES) in turn; (column, density) holds the current item.
        for column, density in enumerate(DENSITIES):
            # = stores a result in axis; use square brackets to select part of axes.
            axis = axes[row_number, column]
            # for takes each item from [(40000, 'C0'), (100000, 'C1')] in turn; (budget, color) holds the current item.
            for budget, color in [(40000, "C0"), (100000, "C1")]:
                # = stores a result in item; call find_summary using parentheses to supply its inputs.
                item = find_summary(summaries, budget, policy, density)
                # = stores a result in check; use square brackets to select part of item.
                check = item["quantile_checks"]
                # Call axis.fill_between using parentheses to supply its inputs; this line does not store a separate return value.
                axis.fill_between(
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    check["taus"],
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    check["coverage_lower"],
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    check["coverage_upper"],
                    # Inside this continued call, name=value supplies an option by name rather than by its position.
                    color=color,
                    # Inside this continued call, name=value supplies an option by name rather than by its position.
                    alpha=0.25,
                    # Inside this continued call, name=value supplies an option by name rather than by its position.
                    label=f"{budget // 1000}K",
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                )
                # Call axis.plot using parentheses to supply its inputs; this line does not store a separate return value.
                axis.plot(
                    # Inside this continued call, name=value supplies an option by name rather than by its position.
                    check["taus"], check["coverage_lower"], color=color, linewidth=0.8
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                )
                # Call axis.plot using parentheses to supply its inputs; this line does not store a separate return value.
                axis.plot(
                    # Inside this continued call, name=value supplies an option by name rather than by its position.
                    check["taus"], check["coverage_upper"], color=color, linewidth=0.8
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                )
            # Call axis.plot using parentheses to supply its inputs; this line does not store a separate return value.
            axis.plot([0, 1], [0, 1], "--", color="black", linewidth=0.8)
            # Call axis.set_title using parentheses to supply its inputs; this line does not store a separate return value.
            axis.set_title(f"{LABELS[policy]}, density {density:g}")
            # Call axis.set_xlabel using parentheses to supply its inputs; this line does not store a separate return value.
            axis.set_xlabel("Assigned quantile fraction")
            # Call axis.grid using parentheses to supply its inputs; this line does not store a separate return value.
            axis.grid(alpha=0.2)
    # Call axes[0, 0].set_ylabel using parentheses to supply its inputs; this line does not store a separate return value.
    axes[0, 0].set_ylabel("Empirical compatibility bracket")
    # Call axes[1, 0].set_ylabel using parentheses to supply its inputs; this line does not store a separate return value.
    axes[1, 0].set_ylabel("Empirical compatibility bracket")
    # Call axes[0, 0].legend using parentheses to supply its inputs; this line does not store a separate return value.
    axes[0, 0].legend()
    # Call figure.suptitle using parentheses to supply its inputs; this line does not store a separate return value.
    figure.suptitle(
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "Bands account for ties and cutoff censoring; not statistical confidence bands"
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # Call figure.tight_layout using parentheses to supply its inputs; this line does not store a separate return value.
    figure.tight_layout()
    # Save the graph for the local report. Drawing it does not change any measured result.
    # Call figure.savefig using parentheses to supply its inputs; this line does not store a separate return value.
    figure.savefig(OUTPUT / "quantile_checks.png", dpi=150)
    # Call plt.close using parentheses to supply its inputs; this line does not store a separate return value.
    plt.close(figure)


# def defines write_report_page; parentheses list its inputs, and the colon starts its indented body.
def write_report_page(summaries):
    # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
    """Turn the measured numbers into a page. No model is changed or selected here."""
    # The long string below is page layout, not reinforcement learning. Later lines insert the measured numbers.
    # = stores a result in page; use quoted text as data; Python does not execute the words inside it.
    page = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Prediction audit</title><style>body{font:17px/1.6 system-ui;background:#f2f5fa;color:#172a40;max-width:1160px;margin:auto;padding:32px}section{background:white;padding:24px;border-radius:14px;margin:24px 0}img{width:100%}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;text-align:left;border-bottom:1px solid #dce4ef}a{color:#165bb5}</style></head><body><p>DEEP RL / DEVELOPMENT PREDICTION AUDIT</p><h1>Did longer training improve the predictions?</h1><p><a href="../report/index.html">Interactive decision viewer</a> · <a href="../long_training-report/index.html">Frozen driving-performance study</a> · <a href="summary.json">Raw summaries</a> · <a href="design.json">Recorded diagnostic design</a></p><section><h2>What is being checked?</h2><p>40K and 100K models predict the return of their initial chosen action, then drive under the stated rule. Each policy/budget/density uses three models and 60 shared development scenarios per model. Seeds 19000–19059 are development data. No further confirmation seeds are consumed.</p><p>Different policies can choose different actions and create different outcomes. This is an audit of forecasts under their deployed behavior, not a fixed-target test isolating predictor quality. QR-tail deployment also differs from the mean-based continuation rule used in training.</p></section><section><h2>Mean prediction error</h2><p>Positive signed error means optimism. Absolute errors avoid cancellation between overestimates and underestimates. A crash gives an exact return; the 30-second recording cutoff leaves a bounded unknown future. Whiskers bound that missing continuation, not statistical uncertainty.</p><img src="forecast_errors.png"></section>"""
    # Summarize two representative diagnostics without changing any policy.
    # = stores a result in old_dense; call find_summary using parentheses to supply its inputs.
    old_dense = find_summary(summaries, 40000, "qr_tail", 2.0)
    # = stores a result in new_dense; call find_summary using parentheses to supply its inputs.
    new_dense = find_summary(summaries, 100000, "qr_tail", 2.0)
    # = stores a result in old_normal; call find_summary using parentheses to supply its inputs.
    old_normal = find_summary(summaries, 40000, "qr_mean", 1.0)
    # = stores a result in new_normal; call find_summary using parentheses to supply its inputs.
    new_normal = find_summary(summaries, 100000, "qr_mean", 1.0)
    # = stores a result in old_dense_limits; build a list, keeping its items in order.
    old_dense_limits = []
    # for takes each item from old_dense['absolute_error_bounds'] in turn; x holds the current item.
    for x in old_dense["absolute_error_bounds"]:
        # Call old_dense_limits.append using parentheses to supply its inputs; this line does not store a separate return value.
        old_dense_limits.append(round(x, 3))
    # = stores a result in new_dense_limits; build a list, keeping its items in order.
    new_dense_limits = []
    # for takes each item from new_dense['absolute_error_bounds'] in turn; x holds the current item.
    for x in new_dense["absolute_error_bounds"]:
        # Call new_dense_limits.append using parentheses to supply its inputs; this line does not store a separate return value.
        new_dense_limits.append(round(x, 3))
    # = stores a result in dense_text; combine the operands with the displayed operator; parentheses control grouping.
    dense_text = (
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "At density 2, cautious QR absolute-error bounds change from "
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + str(old_dense_limits)
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + " at 40K to "
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + str(new_dense_limits)
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + " at 100K."
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # = stores a result in normal_text; evaluate the expression on the right before using its result.
    normal_text = f"In normal traffic, QR mean's empirical compatibility gap decreases from {old_normal['quantile_checks']['minimum_mean_absolute_compatibility_gap']:.3f} to {new_normal['quantile_checks']['minimum_mean_absolute_compatibility_gap']:.3f}."
    # The combined operator updates page using its previous value.
    page += (
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "<section><h2>Main observation</h2><p>Longer training does not uniformly improve prediction quality. "
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + dense_text
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + " "
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + normal_text
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        + "</p><p>These are descriptive results on this diagnostic block. Better driving performance and more accurate predictions are separate properties. Missing-continuation bounds do not provide a population confidence interval, and changing a policy also changes what it experiences.</p></section>"
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # = stores a result in rows; build a list, keeping its items in order.
    rows = []
    # for takes each item from summaries in turn; result holds the current item.
    for result in summaries:
        # = stores a result in printed_bias_limits; build a list, keeping its items in order.
        printed_bias_limits = []
        # for takes each item from result['bias_bounds'] in turn; x holds the current item.
        for x in result["bias_bounds"]:
            # Call printed_bias_limits.append using parentheses to supply its inputs; this line does not store a separate return value.
            printed_bias_limits.append(round(x, 3))
        # = stores a result in printed_absolute_limits; build a list, keeping its items in order.
        printed_absolute_limits = []
        # for takes each item from result['absolute_error_bounds'] in turn; x holds the current item.
        for x in result["absolute_error_bounds"]:
            # Call printed_absolute_limits.append using parentheses to supply its inputs; this line does not store a separate return value.
            printed_absolute_limits.append(round(x, 3))
        # Call rows.append using parentheses to supply its inputs; this line does not store a separate return value.
        rows.append(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            [
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                LABELS[result["policy"]],
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                result["budget"],
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                result["density"],
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                str(printed_bias_limits),
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                str(printed_absolute_limits),
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                f"{100 * result['censored_fraction']:.1f}%",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                f"{100 * result['crash_rate']:.1f}%",
            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
            ]
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
    # The combined operator updates page using its previous value.
    page += "<section><h2>All empirical error bounds</h2>"
    # The combined operator updates page using its previous value.
    page += (
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        table(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            [
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Policy",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Steps",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Density",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Signed bias bounds",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Absolute error bounds",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Censored",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Collisions",
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
    # The combined operator updates page using its previous value.
    page += '<section><h2>QR-DQN distribution checks</h2><p>The dashed line is the assigned quantile fraction. Bands show which empirical coverage values are compatible with ties and unobserved continuation. They are not sampling confidence bands. Raw quantile heads are not sorted here, so crossings remain visible. Good marginal agreement does not establish accurate conditional forecasts in every state.</p><img src="quantile_checks.png">'
    # = stores a result in rows; build a list, keeping its items in order.
    rows = []
    # for takes each item from summaries in turn; result holds the current item.
    for result in summaries:
        # if checks 'quantile_checks' in result; only a true condition runs the indented block.
        if "quantile_checks" in result:
            # = stores a result in check; use square brackets to select part of result.
            check = result["quantile_checks"]
            # Call rows.append using parentheses to supply its inputs; this line does not store a separate return value.
            rows.append(
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                [
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    LABELS[result["policy"]],
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    result["budget"],
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    result["density"],
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    f"{check['minimum_mean_absolute_compatibility_gap']:.3f}",
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    f"{100 * result['quantile_crossing_fraction']:.1f}%",
                # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
                ]
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            )
    # The combined operator updates page using its previous value.
    page += (
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        table(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            [
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Policy",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Steps",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Density",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Minimum compatibility gap",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Adjacent crossings",
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
    # The combined operator updates page using its previous value.
    page += "<section><h2>Limits and reproduction</h2><p>DQN supplies one mean estimate, so no quantile-distribution test is invented for it. Small mean bias can hide large individual errors. These comparisons do not establish why a policy crashed or guarantee safe driving.</p><p><code>python run_prediction_audit.py --jobs 4</code> validates and reuses saved data. <code>--report-only</code> rebuilds this page. No model training, risk-level selection, or speed-cap tuning occurs.</p></section></body></html>"
    # Call (output / 'index.html').write_text using parentheses to supply its inputs; this line does not store a separate return value.
    (OUTPUT / "index.html").write_text(page)
    # Call print using parentheses to supply its inputs; this line does not store a separate return value.
    print("Prediction audit report saved.", flush=True)


# def defines report; parentheses list its inputs, and the colon starts its indented body.
def report():
    # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
    """Four short steps: read, draw two figures, then write the page."""
    # = stores a result in summaries; call load_and_summarize_results using parentheses to supply its inputs.
    summaries = load_and_summarize_results()
    # Call draw_error_comparison using parentheses to supply its inputs; this line does not store a separate return value.
    draw_error_comparison(summaries)
    # Call draw_quantile_checks using parentheses to supply its inputs; this line does not store a separate return value.
    draw_quantile_checks(summaries)
    # Call write_report_page using parentheses to supply its inputs; this line does not store a separate return value.
    write_report_page(summaries)


# __name__ equals "__main__" when this file runs directly; this guard avoids running the script merely because another file imports it.
if __name__ == "__main__":
    # Call report using parentheses to supply its inputs; this line does not store a separate return value.
    report()
