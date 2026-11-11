"""Combine the risk-selector experiments and their prediction checks.

The same saved QR-DQN networks are reused with different action-selection rules.
This holds the network fixed while we compare decisions, but it does not make
the rules mathematically optimal or guarantee safety.

Original implementation notes:
Same-checkpoint comparisons, exploratory risk frontier and calibration.
"""

import json  # Make json available for this file's calculations and workflow.
from pathlib import Path  # Make Path available for this file's calculations and workflow.

import numpy as np  # Make numpy available for this file's calculations and workflow.

from .analysis import crossed_bootstrap, paired_matrix, summarize_group  # Make crossed_bootstrap, paired_matrix, summarize_group available for this file's calculations and workflow.
from .calibration import summarize_rows  # Make summarize_rows available for this file's calculations and workflow.
from .experiment import save_json  # Make save_json available for this file's calculations and workflow.


def analyze(root):  # Combine the saved experiments into comparison tables. Importing here avoids a runner/analysis import cycle at module startup.
    """Combine the saved experiments into comparison tables. Match scenarios and
    model seeds before calculating differences, and keep development limitations visible.
    """
    from run_risk_analysis import ALPHAS, eval_path  # Make ALPHAS, eval_path available for this file's calculations and workflow.

    root = Path(root)  # Set root to Path applied to root.
    stage = root / "risk_analysis"  # Set stage to the path to 'risk_analysis' inside root.
    selection = json.loads((stage / "governor_selection.json").read_text())  # Set selection to json.loads applied to the result of (stage / 'governor selection.json').read text.
    cap = selection["selected_cap_m_s"]  # Set cap to the entries of selection selected by 'selected_cap_m_s'.
    load = lambda path: json.loads(path.read_text())  # Set load to lambda path: json.loads(path.read text()).
    definitions = {  # Set definitions to the named settings or measurements assembled below.
        "QR-DQN mean": (1.0, None),  # Record 'QR-DQN mean' as 1.0, None.
        "QR-DQN lower 25%": (0.25, None),  # Record 'QR-DQN lower 25%' as 0.25, None.
        "Same QR-DQN + governor": (1.0, cap),  # Record 'Same QR-DQN + governor' as 1.0, cap.
    }  # Finish assembling the grouped values above.
    groups = {  # Set groups to load applied to eval path applied to seed, the fraction of low-return predictions used to score actions, speed cap for each seed in 0, 1, 2 for each name, the fraction of low-return predictions used to score actions, speed cap in the result of definitions.items.
        name: [load(eval_path(seed, alpha, speed_cap)) for seed in [0, 1, 2]]  # Include name, load applied to eval path applied to seed, the fraction of low-return predictions used to score actions, speed cap for each seed in 0, 1, 2 in the surrounding operation.
        for name, (alpha, speed_cap) in definitions.items()  # Include name, the fraction of low-return predictions used to score actions, speed cap, the result of definitions.items in the surrounding operation.
    }  # Finish assembling the grouped values above.
    paired = []  # Set paired to .
    for density in [1.0, 1.5, 2.0]:  # Process each the traffic-density setting from 1.0, 1.5, 2.0.
        output = {  # Set output to the named settings or measurements assembled below.
            "density": density,  # Record 'density' as the traffic-density setting.
            "left": "QR-DQN lower 25%",  # Record 'left' as 'QR-DQN lower 25%'.
            "right": "Same QR-DQN + governor",  # Record 'right' as 'Same QR-DQN + governor'.
            "sign": "left minus right",  # Record 'sign' as 'left minus right'.
        }  # Finish assembling the grouped values above.
        for metric in ["crashed", "mean_speed_m_s", "distance_m", "return"]:  # Compare one measured quantity at a time, keeping the scenarios paired.
            seeds, matrix = paired_matrix(  # Set seeds, matrix to paired matrix applied to the entries of groups selected by output['left'], the entries of groups selected by output['right'], the traffic-density setting, metric.
                groups[output["left"]], groups[output["right"]], density, metric  # Include the entries of groups selected by output['left'], the entries of groups selected by output['right'], the traffic-density setting, metric in the surrounding operation.
            )  # Finish assembling paired matrix inputs.
            output[metric] = crossed_bootstrap(matrix)  # Set the entries of output selected by metric to crossed bootstrap applied to matrix.
            output[metric]["training_seed_ids"] = seeds  # Set the entries of the entries of output selected by metric selected by 'training_seed_ids' to seeds.
        paired.append(output)  # Add output to paired.
    frontier = [  # Set frontier to the named settings or measurements assembled below for each the fraction of low-return predictions used to score actions in ALPHAS.
        {  # Include the named settings or measurements assembled below in the surrounding operation.
            "alpha": alpha,  # Record 'alpha' as the fraction of low-return predictions used to score actions.
            "summary": summarize_group(  # Record 'summary' as summarize group applied to load applied to eval path applied to seed, the fraction of low-return predictions used to score actions for each seed in 0, 1, 2.
                [load(eval_path(seed, alpha)) for seed in [0, 1, 2]]  # Include load applied to eval path applied to seed, the fraction of low-return predictions used to score actions for each seed in 0, 1, 2 in the surrounding operation.
            ),  # Finish assembling summarize group inputs.
        }  # Finish assembling the grouped values above.
        for alpha in ALPHAS  # Include the fraction of low-return predictions used to score actions, ALPHAS in the surrounding operation.
    ]  # Finish assembling the grouped values above.
    calibration = {}  # Set calibration to the named settings or measurements assembled below.
    failures = {}  # Set failures to the named settings or measurements assembled below.
    for policy in ["mean", "tail25", "governed"]:  # Process each policy from 'mean', 'tail25', 'governed'.
        datasets = [  # Set datasets to load applied to (stage divided by f'calibration {policy} seed{seed}.json') for each seed in 0, 1, 2.
            load(stage / f"calibration_{policy}_seed{seed}.json") for seed in [0, 1, 2]  # Include load applied to (stage divided by f'calibration {policy} seed{seed}.json'), seed, 0, 1, 2 in the surrounding operation.
        ]  # Finish assembling the grouped values above.
        for seed, data in enumerate(datasets):  # Process each seed, data from enumerate applied to datasets.
            p = data["provenance"]  # Set p to the entries of data selected by 'provenance'.
            if (  # Use this branch when the entries of p selected by 'training_seed' differs from seed or the entries of p selected by 'seed_start' differs from 12000 or the entries of p selected by 'episodes_per_density' differs from 60 or the entries of p selected by 'training_steps' differs from 20000 or the entries of p selected by 'alpha' differs from 0.25 when policy equals 'tail25', otherwise 1.0 or the entries of p selected by 'speed_cap' differs from cap when policy equals 'governed', otherwise None.
                p["training_seed"] != seed  # Include the entries of p selected by 'training_seed' differs from seed or the entries of p selected by 'seed_start' differs from 12000 or the entries of p selected by 'episodes_per_density' differs from 60 or the entries of p selected by 'training_steps' differs from 20000 or the entries of p selected by 'alpha' differs from 0.25 when policy equals 'tail25', otherwise 1.0 or the entries of p selected by 'speed_cap' differs from cap when policy equals 'governed', otherwise None in the surrounding operation.
                or p["seed_start"] != 12000  # Include the entries of p selected by 'seed_start' differs from 12000 in the surrounding operation.
                or p["episodes_per_density"] != 60  # Include the entries of p selected by 'episodes_per_density' differs from 60 in the surrounding operation.
                or p["training_steps"] != 20000  # Include the entries of p selected by 'training_steps' differs from 20000 in the surrounding operation.
                or p["alpha"] != (0.25 if policy == "tail25" else 1.0)  # Include the entries of p selected by 'alpha' differs from 0.25 when policy equals 'tail25', otherwise 1.0 in the surrounding operation.
                or p["speed_cap"] != (cap if policy == "governed" else None)  # Include the entries of p selected by 'speed_cap' differs from cap when policy equals 'governed', otherwise None in the surrounding operation.
            ):  # Finish assembling the grouped values above.
                raise ValueError(  # Reject this invalid case instead of continuing with misleading experiment results.
                    "Calibration provenance does not match risk analysis protocol"  # Include 'Calibration provenance does not match risk analysis protocol' in the surrounding operation.
                )  # Finish assembling ValueError inputs.
        calibration[policy], failures[policy] = [], []  # Set the entries of calibration selected by policy, the entries of failures selected by policy to , .
        for density in [1.0, 1.5, 2.0]:  # Process each the traffic-density setting from 1.0, 1.5, 2.0.
            rows = [  # Set rows to row for each data in datasets.
                row  # Include row in the surrounding operation.
                for data in datasets  # Include data, datasets in the surrounding operation.
                for row in data["episodes"]  # Include row, the entries of data selected by 'episodes' in the surrounding operation.
                if row["density"] == density  # Include the entries of row selected by 'density' equals the traffic-density setting in the surrounding operation.
            ]  # Finish assembling the grouped values above.
            metrics = summarize_rows(rows)  # Set metrics to summarize rows applied to rows.
            metrics["per_seed"] = [  # Set the entries of metrics selected by 'per_seed' to the named settings or measurements assembled below for each seed, data in enumerate applied to datasets.
                {  # Include the named settings or measurements assembled below in the surrounding operation.
                    "training_seed": seed,  # Record 'training_seed' as seed.
                    **summarize_rows(  # Include summarize rows applied to row for each row in the entries of data selected by 'episodes', keeping only cases where the entries of row selected by 'density' equals the traffic-density setting in the surrounding operation.
                        [row for row in data["episodes"] if row["density"] == density]  # Include row for each row in the entries of data selected by 'episodes', keeping only cases where the entries of row selected by 'density' equals the traffic-density setting in the surrounding operation.
                    ),  # Finish assembling summarize rows inputs.
                }  # Finish assembling the grouped values above.
                for seed, data in enumerate(datasets)  # Include seed, data, enumerate applied to datasets in the surrounding operation.
            ]  # Finish assembling the grouped values above.
            calibration[policy].append({"density": density, **metrics})  # Add the named settings or measurements assembled below to the entries of calibration selected by policy.
            crashed = [row for row in rows if row["crashed"]]  # Set crashed to row for each row in rows, keeping only cases where the entries of row selected by 'crashed'.
            proxy_counts = {  # Set proxy counts to sum applied to row['terminal nearest neighbor']['position proxy'] equals name for each row in crashed, keeping only cases where the entries of row selected by 'terminal_nearest_neighbor' for each name in 'front', 'rear', 'side'.
                name: sum(  # Include name, sum applied to the entries of row['terminal nearest neighbor'] selected by 'position_proxy' equals name for each row in crashed, keeping only cases where the entries of row selected by 'terminal_nearest_neighbor' in the surrounding operation.
                    row["terminal_nearest_neighbor"]["position_proxy"] == name  # Include the entries of the entries of row selected by 'terminal_nearest_neighbor' selected by 'position_proxy' equals name in the surrounding operation.
                    for row in crashed  # Include row, crashed in the surrounding operation.
                    if row["terminal_nearest_neighbor"]  # Include the entries of row selected by 'terminal_nearest_neighbor' in the surrounding operation.
                )  # Finish assembling sum inputs.
                for name in ["front", "rear", "side"]  # Include name, 'front', 'rear', 'side' in the surrounding operation.
            }  # Finish assembling the grouped values above.
            failures[policy].append(  # Add the named settings or measurements assembled below to the entries of failures selected by policy.
                {  # Include the named settings or measurements assembled below in the surrounding operation.
                    "density": density,  # Record 'density' as the traffic-density setting.
                    "episodes": len(rows),  # Record 'episodes' as the number of entries in rows.
                    "crashes": len(crashed),  # Record 'crashes' as the number of entries in crashed.
                    "early_crashes_within_3_seconds": sum(  # Record 'early_crashes_within_3_seconds' as sum applied to the entries of row selected by 'length' is at most 3 for each row in crashed.
                        row["length"] <= 3 for row in crashed  # Include the entries of row selected by 'length' is at most 3, row, crashed in the surrounding operation.
                    ),  # Finish assembling sum inputs.
                    "recent_lateral_request_given_crash": metrics[  # Record 'recent_lateral_request_given_crash' as the entries of metrics selected by 'recent_lateral_request_given_crash'.
                        "recent_lateral_request_given_crash"  # Include 'recent_lateral_request_given_crash' in the surrounding operation.
                    ],  # Finish assembling the grouped values above.
                    "recent_lane_transition_given_crash": metrics[  # Record 'recent_lane_transition_given_crash' as the entries of metrics selected by 'recent_lane_transition_given_crash'.
                        "recent_lane_transition_given_crash"  # Include 'recent_lane_transition_given_crash' in the surrounding operation.
                    ],  # Finish assembling the grouped values above.
                    "terminal_nearest_neighbor_proxy_counts": proxy_counts,  # Record 'terminal_nearest_neighbor_proxy_counts' as proxy counts.
                    "all_observation_slots_filled_near_crash_fraction": (  # Record 'all_observation_slots_filled_near_crash_fraction' as the average of row['recent states'][-1]['observed vehicle slots'] == 8 for each row in crashed as a floating-point value when crashed, otherwise None.
                        float(  # Include the average of row['recent states'][-1]['observed vehicle slots'] == 8 for each row in crashed as a floating-point value when crashed, otherwise None in the surrounding operation.
                            np.mean(  # Include the average of the entries of row['recent states'][-1] selected by 'observed_vehicle_slots' equals 8 for each row in crashed in the surrounding operation.
                                [  # Include the entries of the entries of row['recent states'] selected by -1 selected by 'observed_vehicle_slots' equals 8 for each row in crashed in the surrounding operation.
                                    row["recent_states"][-1]["observed_vehicle_slots"]  # Include the entries of the entries of the entries of row selected by 'recent_states' selected by -1 selected by 'observed_vehicle_slots' equals 8 in the surrounding operation.
                                    == 8  # Include 8 in the surrounding operation.
                                    for row in crashed  # Include row, crashed in the surrounding operation.
                                ]  # Finish assembling np.mean inputs.
                            )  # Finish assembling np.mean inputs.
                        )  # Finish assembling float inputs.
                        if crashed  # Include crashed in the surrounding operation.
                        else None  # Include None in the surrounding operation.
                    ),  # Finish assembling the grouped values above.
                }  # Finish assembling the grouped values above.
            )  # Finish assembling failures[policy].append inputs.
    result = {  # Set result to the named settings or measurements assembled below.
        "status": "Exploratory development; final confirmation seeds remain unused",  # Record 'status' as 'Exploratory development; final confirmation seeds remain unused'.
        "selection": selection,  # Record 'selection' as selection.
        "groups": {name: summarize_group(data) for name, data in groups.items()},  # Record 'groups' as summarize group applied to data for each name, data in the result of groups.items.
        "paired_comparisons": paired,  # Record 'paired_comparisons' as paired.
        "risk_frontier": frontier,  # Record 'risk_frontier' as frontier.
        "calibration": calibration,  # Record 'calibration' as calibration.
        "failure_diagnostics": failures,  # Record 'failure_diagnostics' as failures.
        "interpretation": {  # Record 'interpretation' as the named settings or measurements assembled below.
            "paired_intervals": "Exploratory two-way bootstrap, conditional on development choices; only three training seeds.",  # Record 'paired_intervals' as the report text.
            "calibration": "Marginal empirical compatibility brackets account for censoring and ties, not sampling uncertainty. Initial-state samples; changed-policy continuations differ from training.",  # Record 'calibration' as the report text.
            "failures": "Descriptive recent-action and geometry proxies, not identified crash causes or proof of unavoidable collisions.",  # Record 'failures' as the report text.
        },  # Finish assembling the grouped values above.
    }  # Finish assembling the grouped values above.
    save_json(stage / "summary.json", result)  # Save the following results/settings as json: the path to 'summary.json' inside stage, result.
    return result  # Return result.
