"""Explain the 100K study using simple tables, plots, and paired comparisons.

Nothing in this file trains a model or chooses a better-looking checkpoint.
The two budgets face the same scenarios, so we subtract matching outcomes.
"""

import json  # Make json available for this file's calculations and workflow.
from pathlib import Path  # Make Path available for this file's calculations and workflow.
import shutil  # Make shutil available for this file's calculations and workflow.

import matplotlib  # Make matplotlib available for this file's calculations and workflow.

matplotlib.use("Agg")  # Matplotlib.use applied to 'agg'.
import matplotlib.pyplot as plt  # Make matplotlib.pyplot available for this file's calculations and workflow.

from .analysis import summarize_group, paired_matrix, scenario_bootstrap_fixed_models  # Make summarize_group, paired_matrix, scenario_bootstrap_fixed_models available for this file's calculations and workflow.
from .density_coverage_report import table  # Make table available for this file's calculations and workflow.
from .experiment import save_json  # Make save_json available for this file's calculations and workflow.

ROOT = Path(__file__).resolve().parents[1]  # Set ROOT to the entries of Path( file ).resolve().parents selected by 1.
OUTPUT = ROOT / "results/long_training_evaluation"  # Set OUTPUT to the path to 'results/long_training_evaluation' inside ROOT.
LABELS = {  # Set LABELS to the named settings or measurements assembled below.
    "qr_tail": "QR lower quarter",  # Record 'qr_tail' as 'QR lower quarter'.
    "dqn_governed": "DQN + governor",  # Record 'dqn_governed' as 'DQN + governor'.
    "qr_governed": "QR mean + governor",  # Record 'qr_governed' as 'QR mean + governor'.
    "dqn_mean": "DQN mean",  # Record 'dqn_mean' as 'DQN mean'.
    "qr_mean": "QR mean",  # Record 'qr_mean' as 'QR mean'.
}  # Finish assembling the grouped values above.


def read(path):  # Turn a saved JSON file into Python lists and dictionaries.
    """Turn a saved JSON file into Python lists and dictionaries."""  # Describe this operation for readers and help().
    return json.loads(path.read_text())  # Return json.loads applied to the result of path.read text.


def compare(left, right, density):  # Subtract matching cases, then estimate uncertainty over fresh scenarios.
    """Subtract matching cases, then estimate uncertainty over fresh scenarios.

    The trained models stay fixed. Resample scenario IDs, not individual cells,
    because the same starting traffic is shared across the models.
    """
    result = {}  # Set result to the named settings or measurements assembled below.
    for metric in ["crashed", "mean_speed_m_s", "distance_m", "return"]:  # Compare one quantity at a time, using matching model and scenario IDs.
        seeds, differences = paired_matrix(left, right, density, metric)  # Set seeds, differences to paired matrix applied to left, right, the traffic-density setting, metric.
        result[metric] = scenario_bootstrap_fixed_models(differences)  # Set the entries of result selected by metric to scenario bootstrap fixed models applied to differences.
    return result  # Return result.


def load_and_compare_results():  # Read matching cases, calculate differences, and save the full numerical summary.
    """Read matching cases, calculate differences, and save the full numerical summary."""  # Describe this operation for readers and help().
    design = read(OUTPUT / "design.json")  # Set design to read applied to the path to 'design.json' inside OUTPUT.
    data = {}  # Set data to the named settings or measurements assembled below.
    groups = {}  # Set groups to the named settings or measurements assembled below.
    budget_effects = []  # Set budget effects to .
    policy_effects = []  # Set policy effects to .
    for budget in [40000, 100000]:  # Each budget gets its own results. Keep 40K and 100K separate until the paired comparison.
        data[budget] = {}  # Set the entries of data selected by budget to the named settings or measurements assembled below.
        groups[budget] = {}  # Set the entries of groups selected by budget to the named settings or measurements assembled below.
        for policy in LABELS:  # A policy is a decision rule. Several rules can share the very same QR-DQN network.
            runs = []  # Set runs to .
            for seed in range(3):  # A training seed identifies one independently trained model, not a new traffic scenario.
                path = OUTPUT / f"eval_{budget}_{policy}_seed{seed}.json"  # Set path to (OUTPUT divided by f'eval {budget} {policy} seed{seed}.json').
                runs.append(read(path))  # Add read applied to path to runs.
            data[budget][policy] = runs  # Set the entries of the entries of data selected by budget selected by policy to runs.
            groups[budget][policy] = summarize_group(runs)  # Set the entries of the entries of groups selected by budget selected by policy to summarize group applied to runs.
    for policy in LABELS:  # Compare budgets using matching scenario/model IDs, rather than comparing the old 20000 block with the new 20100 block.
        for density in design["densities"]:  # Process each the traffic-density setting from the entries of design selected by 'densities'.
            effect = compare(data[100000][policy], data[40000][policy], density)  # Set effect to compare applied to the entries of the entries of data selected by 100000 selected by policy, the entries of the entries of data selected by 40000 selected by policy, the traffic-density setting.
            budget_effects.append({"policy": policy, "density": density, **effect})  # Add the named settings or measurements assembled below to budget effects.
    for reference in ["dqn_governed", "qr_governed", "dqn_mean", "qr_mean"]:  # Process each reference from 'dqn_governed', 'qr_governed', 'dqn_mean', 'qr_mean'.
        for density in design["densities"]:  # Process each the traffic-density setting from the entries of design selected by 'densities'.
            effect = compare(data[100000]["qr_tail"], data[100000][reference], density)  # Set effect to compare applied to the entries of the entries of data selected by 100000 selected by 'qr_tail', the entries of the entries of data selected by 100000 selected by reference, the traffic-density setting.
            policy_effects.append(  # Add the named settings or measurements assembled below to policy effects.
                {"reference": reference, "density": density, **effect}  # Record 'reference' as reference.
            )  # Finish assembling policy effects.append inputs.
    primary = None  # Find the one comparison named in the recorded design.
    for comparison in policy_effects:  # Process each comparison from policy effects.
        if comparison["reference"] == "dqn_governed" and comparison["density"] == 1.0:  # Use this branch when the entries of comparison selected by 'reference' equals 'dqn_governed' and the entries of comparison selected by 'density' equals 1.0.
            primary = comparison  # Set primary to comparison.
            break  # Stop this loop now that its stopping condition has been reached.
    if primary is None:  # Use this branch when primary is None.
        raise StopIteration  # Reject this invalid case instead of continuing with misleading experiment results.
    upper = primary["crashed"]["conditional_95_percent_interval"][1]  # Set upper to the entries of the entries of the entries of primary selected by 'crashed' selected by 'conditional_95_percent_interval' selected by 1.
    criterion_met = upper < 0  # Use the rule recorded before evaluating the fresh scenarios; do not move the goalposts afterward.
    summary = {  # Set summary to the named settings or measurements assembled below.
        "design": design,  # Record 'design' as design.
        "groups": groups,  # Record 'groups' as groups.
        "budget_effects": budget_effects,  # Record 'budget_effects' as budget effects.
        "policy_effects_100k": policy_effects,  # Record 'policy_effects_100k' as policy effects.
        "primary": primary,  # Record 'primary' as primary.
        "primary_criterion_met": criterion_met,  # Record 'primary_criterion_met' as criterion met.
        "prefix_verification": read(OUTPUT / "prefix_verification.json"),  # Record 'prefix_verification' as read applied to the path to 'prefix_verification.json' inside OUTPUT.
    }  # Finish assembling the grouped values above.
    save_json(OUTPUT / "summary.json", summary)  # JSON object keys become text when saved; this is normal and intentional.
    return summary  # Return summary.


def draw_budget_comparison(groups):  # Plot the measured outcomes at both budgets.
    """Plot the measured outcomes at both budgets. This is display code, not training."""  # Describe this operation for readers and help().
    viewer = ROOT / "results/report"  # Set viewer to the path to 'results/report' inside ROOT.
    plt.rcParams.update(  # Plt.rcparams.update applied to the named settings or measurements assembled below.
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}  # Record 'font.size' as 10.
    )  # Finish assembling plt.rcParams.update inputs.
    figure, axes = plt.subplots(2, 3, figsize=(14, 8))  # A figure is the whole image; axes are its individual plotting panels.
    for column, policy in enumerate(["qr_tail", "dqn_governed", "qr_governed"]):  # Process each column, policy from enumerate applied to 'qr_tail', 'dqn_governed', 'qr_governed'.
        for budget, color in [(40000, "C0"), (100000, "C1")]:  # Process each the environment-interaction budget, color from 40000, 'C0', 100000, 'C1'.
            rows = groups[budget][policy]  # Set rows to the entries of the entries of groups selected by budget selected by policy.
            densities = []  # Set densities to .
            for row in rows:  # Process each row from rows.
                densities.append(row["density"])  # Add the entries of row selected by 'density' to densities.
            for row_number, metric, scale in [  # Process each row number, metric, scale from 0, 'crash_rate', 100, 1, 'mean_speed_m_s', 1.
                (0, "crash_rate", 100),  # Include 0, 'crash_rate', 100 in the surrounding operation.
                (1, "mean_speed_m_s", 1),  # Include 1, 'mean_speed_m_s', 1 in the surrounding operation.
            ]:  # Finish assembling the grouped values above.
                axis = axes[row_number, column]  # Set axis to the entries of axes selected by (row_number, column).
                values = []  # Set values to .
                for row in rows:  # Process each row from rows.
                    values.append(row[metric] * scale)  # Add (the entries of row selected by metric times scale) to values.
                axis.plot(  # Axis.plot applied to densities, values, '-o'.
                    densities,  # Include densities in the surrounding operation.
                    values,  # Include values in the surrounding operation.
                    "-o",  # Include '-o' in the surrounding operation.
                    color=color,  # Include color in the surrounding operation.
                    label=f"{budget // 1000}K steps",  # Include f'{budget // 1000}K steps' in the surrounding operation.
                )  # Finish assembling axis.plot inputs.
                axis.set_xticks(densities)  # Axis.set xticks applied to densities.
                axis.set_xlabel("Traffic density")  # Axis.set xlabel applied to 'traffic density'.
                axis.grid(alpha=0.2)  # The result of axis.grid.
        axes[0, column].set_title(LABELS[policy])  # Axes[0, column].set title applied to the entries of labels selected by policy.
        axes[0, column].set_ylim(0, 105)  # Axes[0, column].set ylim applied to 0, 105.
        axes[0, column].legend()  # The result of axes[0, column].legend.
    axes[0, 0].set_ylabel("Episodes with collision (%)")  # Axes[0, 0].set ylabel applied to 'episodes with collision (%)'.
    axes[1, 0].set_ylabel("Mean episode speed (m/s)")  # Axes[1, 0].set ylabel applied to 'mean episode speed (m/s)'.
    figure.suptitle(  # Figure.suptitle applied to the report text.
        "Same fresh scenarios, different training budgets; three models per learner"  # Include the report text in the surrounding operation.
    )  # Finish assembling figure.suptitle inputs.
    figure.tight_layout()  # The result of figure.tight layout.
    figure.savefig(viewer / "long_training_comparison.png", dpi=150)  # Save the graph for the local report. Drawing it does not change any measured result.
    plt.close(figure)  # Plt.close applied to figure.


def build_budget_tables(groups):  # List the measured outcomes at each budget.
    """List the measured outcomes at each budget. This table does not choose a winner."""  # Describe this operation for readers and help().
    page = ""  # Set page to ''.
    for budget in [40000, 100000]:  # Each budget gets its own results. Keep 40K and 100K separate until the paired comparison.
        rows = []  # Set rows to .
        for policy in LABELS:  # A policy is a decision rule. Several rules can share the very same QR-DQN network.
            for result in groups[budget][policy]:  # Process each result from the entries of the entries of groups selected by budget selected by policy.
                rows.append(  # Add the listed settings or measurements in their recorded order to rows.
                    [  # Include the listed settings or measurements in their recorded order in the surrounding operation.
                        LABELS[policy],  # Include the entries of LABELS selected by policy in the surrounding operation.
                        result["density"],  # Include the entries of result selected by 'density' in the surrounding operation.
                        f"{100 * result['crash_rate']:.1f}%",  # Include f"{100 * result['crash rate']:.1f}%" in the surrounding operation.
                        f"{result['mean_speed_m_s']:.2f}",  # Include f"{result['mean speed m s']:.2f}" in the surrounding operation.
                        f"{result['mean_distance_m']:.0f}",  # Include f"{result['mean distance m']:.0f}" in the surrounding operation.
                        f"{result['mean_return']:.2f}",  # Include f"{result['mean return']:.2f}" in the surrounding operation.
                    ]  # Finish assembling the grouped values above.
                )  # Finish assembling rows.append inputs.
        page += f"<section><h2>{budget // 1000}K results on the new block</h2>"  # Increase page using f'<section><h2>{budget // 1000}K results on the new block</h2>'.
        page += (  # Increase page using (table applied to the listed settings or measurements in their recorded order, rows plus '</section>').
            table(  # Include (table applied to the listed settings or measurements in their recorded order, rows plus '</section>') in the surrounding operation.
                [  # Include the listed settings or measurements in their recorded order in the surrounding operation.
                    "Policy",  # Include 'Policy' in the surrounding operation.
                    "Density",  # Include 'Density' in the surrounding operation.
                    "Collisions",  # Include 'Collisions' in the surrounding operation.
                    "Speed m/s",  # Include 'Speed m/s' in the surrounding operation.
                    "Distance m",  # Include 'Distance m' in the surrounding operation.
                    "Mean return",  # Include 'Mean return' in the surrounding operation.
                ],  # Finish assembling the grouped values above.
                rows,  # Include rows in the surrounding operation.
            )  # Finish assembling table inputs.
            + "</section>"  # Include '</section>' in the surrounding operation.
        )  # Finish assembling the grouped values above.
    return page  # Return page.


def build_budget_effect_table(budget_effects):  # Show 100K minus 40K.
    """Show 100K minus 40K. Negative collision differences favor the longer budget."""  # Describe this operation for readers and help().
    page = ""  # Set page to ''.
    rows = []  # Negative collision changes favor the longer training budget. Show speed too: a reduced collision count might accompany slower driving.
    for result in budget_effects:  # Process each result from budget effects.
        crash = result["crashed"]  # Set crash to the entries of result selected by 'crashed'.
        budget_interval_percent = []  # Convert this comparison interval into percentage points for the results table.
        for x in crash["conditional_95_percent_interval"]:  # Process each x from the entries of crash selected by 'conditional_95_percent_interval'.
            budget_interval_percent.append(round(x * 100, 1))  # Add round applied to (x times 100), 1 to budget interval percent.
        seed_changes_percent = []  # Show each model pair too: a good average can hide differences between training seeds.
        for x in crash["per_training_seed_differences"]:  # Process each x from the entries of crash selected by 'per_training_seed_differences'.
            seed_changes_percent.append(round(x * 100, 1))  # Add round applied to (x times 100), 1 to seed changes percent.
        rows.append(  # Add the listed settings or measurements in their recorded order to rows.
            [  # Include the listed settings or measurements in their recorded order in the surrounding operation.
                LABELS[result["policy"]],  # Include the entries of LABELS selected by result['policy'] in the surrounding operation.
                result["density"],  # Include the entries of result selected by 'density' in the surrounding operation.
                f"{100 * crash['mean_difference']:+.1f}",  # Include f"{100 * crash['mean difference']:+.1f}" in the surrounding operation.
                str(budget_interval_percent),  # Include budget interval percent as a text value in the surrounding operation.
                f"{result['mean_speed_m_s']['mean_difference']:+.2f}",  # Include f"{result['mean speed m s']['mean difference']:+.2f}" in the surrounding operation.
                str(seed_changes_percent),  # Include seed changes percent as a text value in the surrounding operation.
            ]  # Finish assembling the grouped values above.
        )  # Finish assembling rows.append inputs.
    page += "<section><h2>100K minus 40K on matching cases</h2><p>These budget comparisons are secondary. Intervals are conditional on the fixed model set and are not adjusted for multiple comparisons.</p>"  # Increase page using the report text.
    page += (  # Increase page using (table applied to the listed settings or measurements in their recorded order, rows plus '</section>').
        table(  # Include (table applied to the listed settings or measurements in their recorded order, rows plus '</section>') in the surrounding operation.
            [  # Include the listed settings or measurements in their recorded order in the surrounding operation.
                "Policy",  # Include 'Policy' in the surrounding operation.
                "Density",  # Include 'Density' in the surrounding operation.
                "Collision change pp",  # Include 'Collision change pp' in the surrounding operation.
                "Scenario interval pp",  # Include 'Scenario interval pp' in the surrounding operation.
                "Speed change m/s",  # Include 'Speed change m/s' in the surrounding operation.
                "Per-model collision changes pp",  # Include 'Per-model collision changes pp' in the surrounding operation.
            ],  # Finish assembling the grouped values above.
            rows,  # Include rows in the surrounding operation.
        )  # Finish assembling table inputs.
        + "</section>"  # Include '</section>' in the surrounding operation.
    )  # Finish assembling the grouped values above.
    return page  # Return page.


def build_policy_effect_table(policy_effects):  # Show cautious QR minus each reference, keeping speed differences visible.
    """Show cautious QR minus each reference, keeping speed differences visible."""  # Describe this operation for readers and help().
    page = ""  # Set page to ''.
    rows = []  # Set rows to .
    for result in policy_effects:  # Process each result from policy effects.
        crash = result["crashed"]  # Set crash to the entries of result selected by 'crashed'.
        policy_interval_percent = []  # Set policy interval percent to .
        for x in crash["conditional_95_percent_interval"]:  # Process each x from the entries of crash selected by 'conditional_95_percent_interval'.
            policy_interval_percent.append(round(x * 100, 1))  # Add round applied to (x times 100), 1 to policy interval percent.
        rows.append(  # Add the entries of labels selected by result['reference'], the entries of result selected by 'density', f"{100 * crash['mean difference']:+.1f}", policy interval percent as a text value, f"{result['mean speed m s']['mean difference']:+.2f}" to rows.
            [  # Include the entries of LABELS selected by result['reference'], the entries of result selected by 'density', f"{100 * crash['mean difference']:+.1f}", policy interval percent as a text value, f"{result['mean speed m s']['mean difference']:+.2f}" in the surrounding operation.
                LABELS[result["reference"]],  # Include the entries of LABELS selected by result['reference'] in the surrounding operation.
                result["density"],  # Include the entries of result selected by 'density' in the surrounding operation.
                f"{100 * crash['mean_difference']:+.1f}",  # Include f"{100 * crash['mean difference']:+.1f}" in the surrounding operation.
                str(policy_interval_percent),  # Include policy interval percent as a text value in the surrounding operation.
                f"{result['mean_speed_m_s']['mean_difference']:+.2f}",  # Include f"{result['mean speed m s']['mean difference']:+.2f}" in the surrounding operation.
            ]  # Finish assembling the grouped values above.
        )  # Finish assembling rows.append inputs.
    page += "<section><h2>100K cautious QR minus each reference</h2>"  # Increase page using '<section><h2>100K cautious QR minus each reference</h2>'.
    page += (  # Increase page using (table applied to 'Reference', 'Density', 'Collision change pp', 'Scenario interval pp', 'Speed change m/s', rows plus '</section>').
        table(  # Include (table applied to 'Reference', 'Density', 'Collision change pp', 'Scenario interval pp', 'Speed change m/s', rows plus '</section>') in the surrounding operation.
            [  # Include 'Reference', 'Density', 'Collision change pp', 'Scenario interval pp', 'Speed change m/s' in the surrounding operation.
                "Reference",  # Include 'Reference' in the surrounding operation.
                "Density",  # Include 'Density' in the surrounding operation.
                "Collision change pp",  # Include 'Collision change pp' in the surrounding operation.
                "Scenario interval pp",  # Include 'Scenario interval pp' in the surrounding operation.
                "Speed change m/s",  # Include 'Speed change m/s' in the surrounding operation.
            ],  # Finish assembling the grouped values above.
            rows,  # Include rows in the surrounding operation.
        )  # Finish assembling table inputs.
        + "</section>"  # Include '</section>' in the surrounding operation.
    )  # Finish assembling the grouped values above.
    return page  # Return page.


def write_report_page(summary):  # Build tables from the saved comparisons.
    """Build tables from the saved comparisons. The original primary criterion stays unchanged."""  # Describe this operation for readers and help().
    viewer = ROOT / "results/report"  # Set viewer to the path to 'results/report' inside ROOT.
    groups = summary["groups"]  # Set groups to the entries of summary selected by 'groups'.
    budget_effects = summary["budget_effects"]  # Set budget effects to the entries of summary selected by 'budget_effects'.
    policy_effects = summary["policy_effects_100k"]  # Set policy effects to the entries of summary selected by 'policy_effects_100k'.
    primary = summary["primary"]  # Set primary to the entries of summary selected by 'primary'.
    criterion_met = summary["primary_criterion_met"]  # Use the rule recorded before evaluating the fresh scenarios; do not move the goalposts afterward.
    interval = primary["crashed"]["conditional_95_percent_interval"]  # Set interval to the entries of the entries of primary selected by 'crashed' selected by 'conditional_95_percent_interval'.
    difference = primary["crashed"]["mean_difference"] * 100  # Set difference to (the entries of the entries of primary selected by 'crashed' selected by 'mean_difference' times 100).
    speed = primary["mean_speed_m_s"]["mean_difference"]  # Set speed to the entries of the entries of primary selected by 'mean_speed_m_s' selected by 'mean_difference'.
    status = "met" if criterion_met else "not met"  # Set status to 'met' when criterion met, otherwise 'not met'.
    page = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Longer training · Risk-sensitive driving</title><style>body{font:17px/1.6 system-ui;color:#18283d;background:#f4f7fb;margin:auto;max-width:1160px;padding:36px}h1{font-size:38px;line-height:1.2}section{background:white;padding:24px;margin:24px 0;border-radius:14px}img{width:100%;height:auto}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;text-align:left;border-bottom:1px solid #dde3eb}th{background:#eef3fa}a{color:#175bbc}</style></head><body><p>DEEP RL / LONGER TRAINING</p><h1>Did 100,000 training steps help?</h1><p>Six completed 100K runs are evaluated alongside their 40K versions on exactly the same fresh scenarios. No new training, risk-level selection, governor tuning or checkpoint selection happens in this study.</p><p><a href="../long_training_evaluation/design.json">Locked design</a> · <a href="../long_training_evaluation/summary.json">Full numerical results</a> · <a href="../confirmation_40k-report/index.html">Earlier confirmation study</a></p>"""  # The long string below is page layout, not reinforcement learning. Later lines insert the measured numbers.
    page += "<section><h2>Primary 100K comparison</h2>"  # Increase page using '<section><h2>Primary 100K comparison</h2>'.
    primary_interval_percent = []  # Convert fractions such as 0.05 into percentage points such as 5.0 for the reader.
    for x in interval:  # Process each x from interval.
        primary_interval_percent.append(round(x * 100, 1))  # Add round applied to (x times 100), 1 to primary interval percent.
    page += f"<p>The predeclared normal-traffic collision criterion is <strong>{status}</strong>. Cautious QR minus governed DQN: {difference:+.1f} percentage points; conditional 95% interval {primary_interval_percent} pp. Speed difference: {speed:+.2f} m/s.</p>"  # Increase page using f'<p>The predeclared normal-traffic collision criterion is <strong>{status}</strong>. Cautious QR minus governed DQN: {difference:+.1f} percentage points; conditional 95% interval .
    page += "<p>The interval estimates scenario uncertainty for these fixed models. It does not establish equivalence, speed-independent benefit, or reliability across new training runs.</p></section>"  # Increase page using the report text.
    page += (  # Increase page using ((the report text plus f'{speed:+.2f}') plus the report text).
        "<section><h2>Interpreting the speed control</h2><p>The 25 m/s governor was frozen before this study. Its resulting average speed need not match a newly trained network. Here cautious QR differs from controlled DQN by "  # Include ((the report text plus f'{speed:+.2f}') plus the report text) in the surrounding operation.
        + f"{speed:+.2f}"  # Include f'{speed:+.2f}' in the surrounding operation.
        + " m/s in normal traffic. This comparison is therefore not evidence that a risk rule alone caused a collision difference.</p><p>A primary criterion that is not met means the study has not established the specified advantage. It does not establish statistical equivalence.</p></section>"  # Include the report text in the surrounding operation.
    )  # Finish assembling the grouped values above.
    page += '<section><h2>Paired budget comparison</h2><p>Both budgets use seeds 20100–20199: 100 scenario IDs and three trained models per policy. Those 300 rollouts are not treated as 300 independent starting scenarios. All six 40K network prefixes exactly match the previous trained models.</p><img src="long_training_comparison.png"></section>'  # Increase page using the report text.
    page += build_budget_tables(groups)  # Increase page using build budget tables applied to groups.
    page += build_budget_effect_table(budget_effects)  # Increase page using build budget effect table applied to budget effects.
    page += build_policy_effect_table(policy_effects)  # Increase page using build policy effect table applied to policy effects.
    page += "<section><h2>Reproduce and interpret</h2><p><code>python run_long_training_evaluation.py --jobs 4</code> validates and reuses saved evaluations. <code>--report-only</code> rebuilds this page.</p><p>The primary comparison is governed DQN versus cautious QR at density 1 with 100K models. All other contrasts are secondary/exploratory. Policies share the configured training-density mixture, but their experiences differ. Only three training seeds are available.</p><p>Seeds 20100–20199 are now consumed evaluation data, alongside the earlier 20000–20099 block. Any later tuning requires a new untouched confirmation block beginning at 20200. Partial observations, prediction error and simplified traffic limit conclusions; no real-road safety guarantee is established.</p></section></body></html>"  # Increase page using the report text.
    (viewer / "index.html").write_text(page)  # (viewer / 'index.html').write text applied to page.
    print(  # Print applied to f'100k primary criterion {status}: {difference:+.1f} pp; interval {primary interval percent}; speed {speed:+.2f} m/s'.
        f"100K primary criterion {status}: {difference:+.1f} pp; interval {primary_interval_percent}; speed {speed:+.2f} m/s",  # Include f'100K primary criterion {status}: {difference:+.1f} pp; interval {primary interval percent}; speed {speed:+.2f} m/s' in the surrounding operation.
        flush=True,  # Include True in the surrounding operation.
    )  # Finish assembling print inputs.


def report():  # Keep the earlier page, then read, compare, draw, and write.
    """Keep the earlier page, then read, compare, draw, and write."""  # Describe this operation for readers and help().
    viewer = ROOT / "results/report"  # Set viewer to the path to 'results/report' inside ROOT.
    backup = ROOT / "results/confirmation_40k-report"  # Set backup to the path to 'results/confirmation_40k-report' inside ROOT.
    if not backup.exists():  # Use this branch when not the result of backup.exists.
        shutil.copytree(viewer, backup)  # Shutil.copytree applied to viewer, backup.
    summary = load_and_compare_results()  # Set summary to the result of load and compare results.
    draw_budget_comparison(summary["groups"])  # Draw budget comparison applied to the entries of summary selected by 'groups'.
    write_report_page(summary)  # Write report page applied to summary.


if __name__ == "__main__":  # Run this workflow when launched from Terminal; importing its helpers must not launch experiments.
    report()  # The result of report.
