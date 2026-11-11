"""Explain forecast accuracy without confusing it with driving performance.

All results here are development diagnostics. No model is trained or selected.
"""

from pathlib import Path  # Make Path available for this file's calculations and workflow.
import json  # Make json available for this file's calculations and workflow.
import numpy as np  # Make numpy available for this file's calculations and workflow.
import matplotlib  # Make matplotlib available for this file's calculations and workflow.

matplotlib.use("Agg")  # Matplotlib.use applied to 'agg'.
import matplotlib.pyplot as plt  # Make matplotlib.pyplot available for this file's calculations and workflow.
from .forecast_metrics import summarize_predictions  # Make summarize_predictions available for this file's calculations and workflow.
from .density_coverage_report import table  # Make table available for this file's calculations and workflow.
from .experiment import save_json  # Make save_json available for this file's calculations and workflow.

ROOT = Path(__file__).resolve().parents[1]  # Set ROOT to the entries of Path( file ).resolve().parents selected by 1.
OUTPUT = ROOT / "results/prediction_audit"  # Set OUTPUT to the path to 'results/prediction_audit' inside ROOT.
LABELS = {"dqn_mean": "DQN mean", "qr_mean": "QR mean", "qr_tail": "QR lower quarter"}  # Set LABELS to the named settings or measurements assembled below.
DENSITIES = [1.0, 1.5, 2.0, 2.5]  # Set DENSITIES to 1.0, 1.5, 2.0, 2.5.


def find_summary(summaries, budget, policy, density):  # Look through saved summaries until we find the requested group.
    """Look through saved summaries until we find the requested group.

    Check all three labels: budget, policy, and density. A missing group is
    still an error, not a made-up zero result.
    """
    for summary in summaries:  # Process each summary from summaries.
        if summary["budget"] == budget:  # Use this branch when the entries of summary selected by 'budget' equals the environment-interaction budget.
            if summary["policy"] == policy:  # Use this branch when the entries of summary selected by 'policy' equals policy.
                if summary["density"] == density:  # Use this branch when the entries of summary selected by 'density' equals the traffic-density setting.
                    return summary  # Return summary.
    raise StopIteration  # Reject this invalid case instead of continuing with misleading experiment results.


def load_and_summarize_results():  # Read the saved rollouts and keep each policy/budget/density group separate.
    """Read the saved rollouts and keep each policy/budget/density group separate."""  # Describe this operation for readers and help().
    summaries = []  # Set summaries to .
    for budget in [40000, 100000]:  # Each budget gets its own results. Keep 40K and 100K separate until the paired comparison.
        for policy in LABELS:  # A policy is a decision rule. Several rules can share the very same QR-DQN network.
            all_rows = []  # Set all rows to .
            for seed in range(3):  # A training seed identifies one independently trained model, not a new traffic scenario.
                path = OUTPUT / f"audit_{budget}_{policy}_seed{seed}.json"  # Set path to (OUTPUT divided by f'audit {budget} {policy} seed{seed}.json').
                data = json.loads(path.read_text())  # Set data to json.loads applied to the result of path.read text.
                all_rows.extend(data["episodes"])  # Add all entries from the entries of data selected by 'episodes' to all rows.
            for density in DENSITIES:  # Traffic density changes vehicle spacing. It does not add more network inputs.
                rows = []  # Set rows to .
                for row in all_rows:  # Process each row from all rows.
                    if row["density"] == density:  # Use this branch when the entries of row selected by 'density' equals the traffic-density setting.
                        rows.append(row)  # Add row to rows.
                summaries.append(  # Add the named settings or measurements assembled below to summaries.
                    {  # Include the named settings or measurements assembled below in the surrounding operation.
                        "budget": budget,  # Record 'budget' as the environment-interaction budget.
                        "policy": policy,  # Record 'policy' as policy.
                        "density": density,  # Record 'density' as the traffic-density setting.
                        **summarize_predictions(rows),  # Include summarize predictions applied to rows in the surrounding operation.
                    }  # Finish assembling the grouped values above.
                )  # Finish assembling summaries.append inputs.
    save_json(  # Save the following results/settings as json: the path to 'summary.json' inside output, the named settings or measurements assembled below.
        OUTPUT / "summary.json",  # Include the path to 'summary.json' inside OUTPUT in the surrounding operation.
        {  # Include the named settings or measurements assembled below in the surrounding operation.
            "groups": summaries,  # Record 'groups' as summaries.
            "scope": "Initial chosen-action predictions on development scenarios. Error ranges account for cutoff continuation, not sampling uncertainty. Policy changes also change the outcomes being predicted.",  # Record 'scope' as the report text.
        },  # Finish assembling the grouped values above.
    )  # Finish assembling save json inputs.
    return summaries  # Return summaries.


def draw_error_comparison(summaries):  # Draw signed and absolute errors.
    """Draw signed and absolute errors. Whiskers bound missing future rewards, not sample noise."""  # Describe this operation for readers and help().
    plt.rcParams.update(  # Plt.rcparams.update applied to the named settings or measurements assembled below.
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}  # Record 'font.size' as 10.
    )  # Finish assembling plt.rcParams.update inputs.
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), sharey="row")  # A figure is the whole image; axes are its individual plotting panels.
    for column, policy in enumerate(LABELS):  # Process each column, policy from enumerate applied to LABELS.
        for budget, color, offset in [(40000, "C0", -0.03), (100000, "C1", 0.03)]:  # Process each the environment-interaction budget, color, offset from 40000, 'C0', negative 0.03, 100000, 'C1', 0.03.
            rows = []  # Set rows to .
            for summary in summaries:  # Process each summary from summaries.
                if summary["budget"] == budget and summary["policy"] == policy:  # Use this branch when the entries of summary selected by 'budget' equals the environment-interaction budget and the entries of summary selected by 'policy' equals policy.
                    rows.append(summary)  # Add summary to rows.
            for row_number, metric in [  # Process each row number, metric from 0, 'bias_bounds', 1, 'absolute_error_bounds'.
                (0, "bias_bounds"),  # Include 0, 'bias_bounds' in the surrounding operation.
                (1, "absolute_error_bounds"),  # Include 1, 'absolute_error_bounds' in the surrounding operation.
            ]:  # Finish assembling the grouped values above.
                lower_error_limits = []  # Set lower error limits to .
                for row in rows:  # Process each row from rows.
                    lower_error_limits.append(row[metric][0])  # Add the entries of the entries of row selected by metric selected by 0 to lower error limits.
                lows = np.array(lower_error_limits)  # Set lows to lower error limits as a numerical array/tensor for batch calculations.
                upper_error_limits = []  # Set upper error limits to .
                for row in rows:  # Process each row from rows.
                    upper_error_limits.append(row[metric][1])  # Add the entries of the entries of row selected by metric selected by 1 to upper error limits.
                highs = np.array(upper_error_limits)  # Set highs to upper error limits as a numerical array/tensor for batch calculations.
                centers = (lows + highs) / 2  # Set centers to ((lows plus highs) divided by 2).
                axis = axes[row_number, column]  # Set axis to the entries of axes selected by (row_number, column).
                axis.errorbar(  # Whiskers here describe a range. Read the plot label: they are not automatically statistical confidence intervals.
                    np.array(DENSITIES) + offset,  # Include (DENSITIES as a numerical array/tensor for batch calculations plus offset) in the surrounding operation.
                    centers,  # Include centers in the surrounding operation.
                    yerr=[centers - lows, highs - centers],  # Include (centers minus lows), (highs minus centers) in the surrounding operation.
                    fmt="o-",  # Include 'o-' in the surrounding operation.
                    color=color,  # Include color in the surrounding operation.
                    capsize=4,  # Include 4 in the surrounding operation.
                    label=f"{budget // 1000}K",  # Include f'{budget // 1000}K' in the surrounding operation.
                )  # Finish assembling axis.errorbar inputs.
                axis.axhline(0, color="black", linewidth=0.7)  # Axis.axhline applied to 0.
                axis.set_xticks(DENSITIES)  # Axis.set xticks applied to densities.
                axis.set_xlabel("Traffic density")  # Axis.set xlabel applied to 'traffic density'.
                axis.grid(alpha=0.2)  # The result of axis.grid.
        axes[0, column].set_title(LABELS[policy])  # Axes[0, column].set title applied to the entries of labels selected by policy.
        axes[0, column].legend()  # The result of axes[0, column].legend.
    axes[0, 0].set_ylabel("Prediction minus return")  # Axes[0, 0].set ylabel applied to 'prediction minus return'.
    axes[1, 0].set_ylabel("Mean absolute prediction error")  # Axes[1, 0].set ylabel applied to 'mean absolute prediction error'.
    figure.suptitle(  # Figure.suptitle applied to the report text.
        "Missing-continuation bounds only; whiskers are NOT sampling confidence intervals"  # Include the report text in the surrounding operation.
    )  # Finish assembling figure.suptitle inputs.
    figure.tight_layout()  # The result of figure.tight layout.
    figure.savefig(OUTPUT / "forecast_errors.png", dpi=150)  # Save the graph for the local report. Drawing it does not change any measured result.
    plt.close(figure)  # Plt.close applied to figure.


def draw_quantile_checks(summaries):  # Draw QR distribution checks.
    """Draw QR distribution checks. A DQN scalar is never drawn as a quantile distribution."""  # Describe this operation for readers and help().
    figure, axes = plt.subplots(2, 4, figsize=(15, 8), sharex=True, sharey=True)  # A quantile curve is meaningful for QR-DQN, not for DQN's scalar mean. A figure is the whole image; axes are its individual plotting panels.
    for row_number, policy in enumerate(["qr_mean", "qr_tail"]):  # Process each row number, policy from enumerate applied to 'qr_mean', 'qr_tail'.
        for column, density in enumerate(DENSITIES):  # Process each column, the traffic-density setting from enumerate applied to DENSITIES.
            axis = axes[row_number, column]  # Set axis to the entries of axes selected by (row_number, column).
            for budget, color in [(40000, "C0"), (100000, "C1")]:  # Process each the environment-interaction budget, color from 40000, 'C0', 100000, 'C1'.
                item = find_summary(summaries, budget, policy, density)  # Set item to find summary applied to summaries, the environment-interaction budget, policy, the traffic-density setting.
                check = item["quantile_checks"]  # Set check to the entries of item selected by 'quantile_checks'.
                axis.fill_between(  # Axis.fill between applied to the entries of check selected by 'taus', the entries of check selected by 'coverage_lower', the entries of check selected by 'coverage_upper'.
                    check["taus"],  # Include the entries of check selected by 'taus' in the surrounding operation.
                    check["coverage_lower"],  # Include the entries of check selected by 'coverage_lower' in the surrounding operation.
                    check["coverage_upper"],  # Include the entries of check selected by 'coverage_upper' in the surrounding operation.
                    color=color,  # Include color in the surrounding operation.
                    alpha=0.25,  # Include 0.25 in the surrounding operation.
                    label=f"{budget // 1000}K",  # Include f'{budget // 1000}K' in the surrounding operation.
                )  # Finish assembling axis.fill between inputs.
                axis.plot(  # Axis.plot applied to the entries of check selected by 'taus', the entries of check selected by 'coverage_lower'.
                    check["taus"], check["coverage_lower"], color=color, linewidth=0.8  # Include the entries of check selected by 'taus', the entries of check selected by 'coverage_lower', color, 0.8 in the surrounding operation.
                )  # Finish assembling axis.plot inputs.
                axis.plot(  # Axis.plot applied to the entries of check selected by 'taus', the entries of check selected by 'coverage_upper'.
                    check["taus"], check["coverage_upper"], color=color, linewidth=0.8  # Include the entries of check selected by 'taus', the entries of check selected by 'coverage_upper', color, 0.8 in the surrounding operation.
                )  # Finish assembling axis.plot inputs.
            axis.plot([0, 1], [0, 1], "--", color="black", linewidth=0.8)  # Axis.plot applied to 0, 1, 0, 1, '--'.
            axis.set_title(f"{LABELS[policy]}, density {density:g}")  # Axis.set title applied to f'{labels[policy]}, density {density:g}'.
            axis.set_xlabel("Assigned quantile fraction")  # Axis.set xlabel applied to 'assigned quantile fraction'.
            axis.grid(alpha=0.2)  # The result of axis.grid.
    axes[0, 0].set_ylabel("Empirical compatibility bracket")  # Axes[0, 0].set ylabel applied to 'empirical compatibility bracket'.
    axes[1, 0].set_ylabel("Empirical compatibility bracket")  # Axes[1, 0].set ylabel applied to 'empirical compatibility bracket'.
    axes[0, 0].legend()  # The result of axes[0, 0].legend.
    figure.suptitle(  # Figure.suptitle applied to the report text.
        "Bands account for ties and cutoff censoring; not statistical confidence bands"  # Include the report text in the surrounding operation.
    )  # Finish assembling figure.suptitle inputs.
    figure.tight_layout()  # The result of figure.tight layout.
    figure.savefig(OUTPUT / "quantile_checks.png", dpi=150)  # Save the graph for the local report. Drawing it does not change any measured result.
    plt.close(figure)  # Plt.close applied to figure.


def write_report_page(summaries):  # Turn the measured numbers into a page.
    """Turn the measured numbers into a page. No model is changed or selected here."""  # Describe this operation for readers and help().
    page = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Prediction audit</title><style>body{font:17px/1.6 system-ui;background:#f2f5fa;color:#172a40;max-width:1160px;margin:auto;padding:32px}section{background:white;padding:24px;border-radius:14px;margin:24px 0}img{width:100%}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;text-align:left;border-bottom:1px solid #dce4ef}a{color:#165bb5}</style></head><body><p>DEEP RL / DEVELOPMENT PREDICTION AUDIT</p><h1>Did longer training improve the predictions?</h1><p><a href="../report/index.html">Interactive decision viewer</a> · <a href="../long_training-report/index.html">Frozen driving-performance study</a> · <a href="summary.json">Raw summaries</a> · <a href="design.json">Recorded diagnostic design</a></p><section><h2>What is being checked?</h2><p>40K and 100K models predict the return of their initial chosen action, then drive under the stated rule. Each policy/budget/density uses three models and 60 shared development scenarios per model. Seeds 19000–19059 are development data. No further confirmation seeds are consumed.</p><p>Different policies can choose different actions and create different outcomes. This is an audit of forecasts under their deployed behavior, not a fixed-target test isolating predictor quality. QR-tail deployment also differs from the mean-based continuation rule used in training.</p></section><section><h2>Mean prediction error</h2><p>Positive signed error means optimism. Absolute errors avoid cancellation between overestimates and underestimates. A crash gives an exact return; the 30-second recording cutoff leaves a bounded unknown future. Whiskers bound that missing continuation, not statistical uncertainty.</p><img src="forecast_errors.png"></section>"""  # The long string below is page layout, not reinforcement learning. Later lines insert the measured numbers.
    old_dense = find_summary(summaries, 40000, "qr_tail", 2.0)  # Summarize two representative diagnostics without changing any policy.
    new_dense = find_summary(summaries, 100000, "qr_tail", 2.0)  # Set new dense to find summary applied to summaries, 100000, 'qr_tail', 2.0.
    old_normal = find_summary(summaries, 40000, "qr_mean", 1.0)  # Set old normal to find summary applied to summaries, 40000, 'qr_mean', 1.0.
    new_normal = find_summary(summaries, 100000, "qr_mean", 1.0)  # Set new normal to find summary applied to summaries, 100000, 'qr_mean', 1.0.
    old_dense_limits = []  # Set old dense limits to .
    for x in old_dense["absolute_error_bounds"]:  # Process each x from the entries of old dense selected by 'absolute_error_bounds'.
        old_dense_limits.append(round(x, 3))  # Add round applied to x, 3 to old dense limits.
    new_dense_limits = []  # Set new dense limits to .
    for x in new_dense["absolute_error_bounds"]:  # Process each x from the entries of new dense selected by 'absolute_error_bounds'.
        new_dense_limits.append(round(x, 3))  # Add round applied to x, 3 to new dense limits.
    dense_text = (  # Set dense text to (((('At density 2, cautious QR absolute-error bounds change from ' plus str(old dense limits)) plus ' at 40K to ') plus new dense limits as a text value) plus ' at 100K.').
        "At density 2, cautious QR absolute-error bounds change from "  # Include (((('At density 2, cautious QR absolute-error bounds change from ' plus str(old dense limits)) plus ' at 40K to ') plus new dense limits as a text value) plus ' at 100K.') in the surrounding operation.
        + str(old_dense_limits)  # Include old dense limits as a text value in the surrounding operation.
        + " at 40K to "  # Include ' at 40K to ' in the surrounding operation.
        + str(new_dense_limits)  # Include new dense limits as a text value in the surrounding operation.
        + " at 100K."  # Include ' at 100K.' in the surrounding operation.
    )  # Finish assembling the grouped values above.
    normal_text = f"In normal traffic, QR mean's empirical compatibility gap decreases from {old_normal['quantile_checks']['minimum_mean_absolute_compatibility_gap']:.3f} to {new_normal['quantile_checks']['minimum_mean_absolute_compatibility_gap']:.3f}."  # Set normal text to f"In normal traffic, QR mean's empirical compatibility gap decreases from {old normal['quantile checks']['minimum mean absolute compatibility gap']:.3f} to {new normal['quantile ch.
    page += (  # Increase page using (((('<section><h2>Main observation</h2><p>Longer training does not uniformly improve prediction quality. ' plus dense text) plus ' ') plus normal text) plus the report text).
        "<section><h2>Main observation</h2><p>Longer training does not uniformly improve prediction quality. "  # Include (((('<section><h2>Main observation</h2><p>Longer training does not uniformly improve prediction quality. ' plus dense text) plus ' ') plus normal text) plus the report text) in the surrounding operation.
        + dense_text  # Include dense text in the surrounding operation.
        + " "  # Include ' ' in the surrounding operation.
        + normal_text  # Include normal text in the surrounding operation.
        + "</p><p>These are descriptive results on this diagnostic block. Better driving performance and more accurate predictions are separate properties. Missing-continuation bounds do not provide a population confidence interval, and changing a policy also changes what it experiences.</p></section>"  # Include the report text in the surrounding operation.
    )  # Finish assembling the grouped values above.
    rows = []  # Set rows to .
    for result in summaries:  # Process each result from summaries.
        printed_bias_limits = []  # Set printed bias limits to .
        for x in result["bias_bounds"]:  # Process each x from the entries of result selected by 'bias_bounds'.
            printed_bias_limits.append(round(x, 3))  # Add round applied to x, 3 to printed bias limits.
        printed_absolute_limits = []  # Set printed absolute limits to .
        for x in result["absolute_error_bounds"]:  # Process each x from the entries of result selected by 'absolute_error_bounds'.
            printed_absolute_limits.append(round(x, 3))  # Add round applied to x, 3 to printed absolute limits.
        rows.append(  # Add the listed settings or measurements in their recorded order to rows.
            [  # Include the listed settings or measurements in their recorded order in the surrounding operation.
                LABELS[result["policy"]],  # Include the entries of LABELS selected by result['policy'] in the surrounding operation.
                result["budget"],  # Include the entries of result selected by 'budget' in the surrounding operation.
                result["density"],  # Include the entries of result selected by 'density' in the surrounding operation.
                str(printed_bias_limits),  # Include printed bias limits as a text value in the surrounding operation.
                str(printed_absolute_limits),  # Include printed absolute limits as a text value in the surrounding operation.
                f"{100 * result['censored_fraction']:.1f}%",  # Include f"{100 * result['censored fraction']:.1f}%" in the surrounding operation.
                f"{100 * result['crash_rate']:.1f}%",  # Include f"{100 * result['crash rate']:.1f}%" in the surrounding operation.
            ]  # Finish assembling the grouped values above.
        )  # Finish assembling rows.append inputs.
    page += "<section><h2>All empirical error bounds</h2>"  # Increase page using '<section><h2>All empirical error bounds</h2>'.
    page += (  # Increase page using (table applied to the listed settings or measurements in their recorded order, rows plus '</section>').
        table(  # Include (table applied to the listed settings or measurements in their recorded order, rows plus '</section>') in the surrounding operation.
            [  # Include the listed settings or measurements in their recorded order in the surrounding operation.
                "Policy",  # Include 'Policy' in the surrounding operation.
                "Steps",  # Include 'Steps' in the surrounding operation.
                "Density",  # Include 'Density' in the surrounding operation.
                "Signed bias bounds",  # Include 'Signed bias bounds' in the surrounding operation.
                "Absolute error bounds",  # Include 'Absolute error bounds' in the surrounding operation.
                "Censored",  # Include 'Censored' in the surrounding operation.
                "Collisions",  # Include 'Collisions' in the surrounding operation.
            ],  # Finish assembling the grouped values above.
            rows,  # Include rows in the surrounding operation.
        )  # Finish assembling table inputs.
        + "</section>"  # Include '</section>' in the surrounding operation.
    )  # Finish assembling the grouped values above.
    page += '<section><h2>QR-DQN distribution checks</h2><p>The dashed line is the assigned quantile fraction. Bands show which empirical coverage values are compatible with ties and unobserved continuation. They are not sampling confidence bands. Raw quantile heads are not sorted here, so crossings remain visible. Good marginal agreement does not establish accurate conditional forecasts in every state.</p><img src="quantile_checks.png">'  # Increase page using the report text.
    rows = []  # Set rows to .
    for result in summaries:  # Process each result from summaries.
        if "quantile_checks" in result:  # Use this branch when 'quantile_checks' belongs to result.
            check = result["quantile_checks"]  # Set check to the entries of result selected by 'quantile_checks'.
            rows.append(  # Add the entries of labels selected by result['policy'], the entries of result selected by 'budget', the entries of result selected by 'density', f"{check['minimum mean absolute compatibility gap']:.3f}", f"{100 * result['quantile crossing fraction']:.1f}%" to rows.
                [  # Include the entries of LABELS selected by result['policy'], the entries of result selected by 'budget', the entries of result selected by 'density', f"{check['minimum mean absolute compatibility gap']:.3f}", f"{100 * result['quantile crossing fraction']:.1f}%" in the surrounding operation.
                    LABELS[result["policy"]],  # Include the entries of LABELS selected by result['policy'] in the surrounding operation.
                    result["budget"],  # Include the entries of result selected by 'budget' in the surrounding operation.
                    result["density"],  # Include the entries of result selected by 'density' in the surrounding operation.
                    f"{check['minimum_mean_absolute_compatibility_gap']:.3f}",  # Include f"{check['minimum mean absolute compatibility gap']:.3f}" in the surrounding operation.
                    f"{100 * result['quantile_crossing_fraction']:.1f}%",  # Include f"{100 * result['quantile crossing fraction']:.1f}%" in the surrounding operation.
                ]  # Finish assembling the grouped values above.
            )  # Finish assembling rows.append inputs.
    page += (  # Increase page using (table applied to 'Policy', 'Steps', 'Density', 'Minimum compatibility gap', 'Adjacent crossings', rows plus '</section>').
        table(  # Include (table applied to 'Policy', 'Steps', 'Density', 'Minimum compatibility gap', 'Adjacent crossings', rows plus '</section>') in the surrounding operation.
            [  # Include 'Policy', 'Steps', 'Density', 'Minimum compatibility gap', 'Adjacent crossings' in the surrounding operation.
                "Policy",  # Include 'Policy' in the surrounding operation.
                "Steps",  # Include 'Steps' in the surrounding operation.
                "Density",  # Include 'Density' in the surrounding operation.
                "Minimum compatibility gap",  # Include 'Minimum compatibility gap' in the surrounding operation.
                "Adjacent crossings",  # Include 'Adjacent crossings' in the surrounding operation.
            ],  # Finish assembling the grouped values above.
            rows,  # Include rows in the surrounding operation.
        )  # Finish assembling table inputs.
        + "</section>"  # Include '</section>' in the surrounding operation.
    )  # Finish assembling the grouped values above.
    page += "<section><h2>Limits and reproduction</h2><p>DQN supplies one mean estimate, so no quantile-distribution test is invented for it. Small mean bias can hide large individual errors. These comparisons do not establish why a policy crashed or guarantee safe driving.</p><p><code>python run_prediction_audit.py --jobs 4</code> validates and reuses saved data. <code>--report-only</code> rebuilds this page. No model training, risk-level selection, or speed-cap tuning occurs.</p></section></body></html>"  # Increase page using the report text.
    (OUTPUT / "index.html").write_text(page)  # (output / 'index.html').write text applied to page.
    print("Prediction audit report saved.", flush=True)  # Print applied to 'prediction audit report saved.'.


def report():  # Four short steps: read, draw two figures, then write the page.
    """Four short steps: read, draw two figures, then write the page."""  # Describe this operation for readers and help().
    summaries = load_and_summarize_results()  # Set summaries to the result of load and summarize results.
    draw_error_comparison(summaries)  # Draw error comparison applied to summaries.
    draw_quantile_checks(summaries)  # Draw quantile checks applied to summaries.
    write_report_page(summaries)  # Write report page applied to summaries.


if __name__ == "__main__":  # Run this workflow when launched from Terminal; importing its helpers must not launch experiments.
    report()  # The result of report.
