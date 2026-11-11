"""Combine the risk-selector experiments and their prediction checks.

The same saved QR-DQN networks are reused with different action-selection rules.
This holds the network fixed while we compare decisions, but it does not make
the rules mathematically optimal or guarantee safety.

Original implementation notes:
Same-checkpoint comparisons, exploratory risk frontier and calibration.
"""

# import makes these modules available; "as" gives a module a shorter local name.
import json
# from pathlib import brings the named tools into this file; parentheses allow several lines.
from pathlib import Path

# import makes these modules available; "as" gives a module a shorter local name.
import numpy as np

# from analysis import brings the named tools into this file; parentheses allow several lines.
from .analysis import crossed_bootstrap, paired_matrix, summarize_group
# from calibration import brings the named tools into this file; parentheses allow several lines.
from .calibration import summarize_rows
# from experiment import brings the named tools into this file; parentheses allow several lines.
from .experiment import save_json


# def defines analyze; parentheses list its inputs, and the colon starts its indented body.
def analyze(root):
    # Importing here avoids a runner/analysis import cycle at module startup.
    """Combine the saved experiments into comparison tables. Match scenarios and
    model seeds before calculating differences, and keep development limitations visible.
    """
    # from run_risk_analysis import brings the named tools into this file; parentheses allow several lines.
    from run_risk_analysis import ALPHAS, eval_path

    # = stores a result in root; call Path using parentheses to supply its inputs.
    root = Path(root)
    # = stores a result in stage; apply / to these operands; numbers are divided, while Path objects join folder names.
    stage = root / "risk_analysis"
    # = stores a result in selection; call json.loads using parentheses to supply its inputs.
    selection = json.loads((stage / "governor_selection.json").read_text())
    # = stores a result in cap; use square brackets to select part of selection.
    cap = selection["selected_cap_m_s"]
    # = stores a result in load; evaluate the expression on the right before using its result.
    load = lambda path: json.loads(path.read_text())
    # = stores a result in definitions; build a dictionary, which stores values under named keys.
    definitions = {
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "QR-DQN mean": (1.0, None),
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "QR-DQN lower 25%": (0.25, None),
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "Same QR-DQN + governor": (1.0, cap),
    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
    }
    # = stores a result in groups; collect or generate one result per item; the "for" part names the item being processed.
    groups = {
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        name: [load(eval_path(seed, alpha, speed_cap)) for seed in [0, 1, 2]]
        # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
        for name, (alpha, speed_cap) in definitions.items()
    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
    }
    # = stores a result in paired; build a list, keeping its items in order.
    paired = []
    # for takes each item from [1.0, 1.5, 2.0] in turn; density holds the current item.
    for density in [1.0, 1.5, 2.0]:
        # = stores a result in output; build a dictionary, which stores values under named keys.
        output = {
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "density": density,
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "left": "QR-DQN lower 25%",
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "right": "Same QR-DQN + governor",
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "sign": "left minus right",
        # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
        }
        # Compare one measured quantity at a time, keeping the scenarios paired.
        # for takes each item from ['crashed', 'mean_speed_m_s', 'distance_m', 'return'] in turn; metric holds the current item.
        for metric in ["crashed", "mean_speed_m_s", "distance_m", "return"]:
            # = unpacks the right-hand result into (seeds, matrix), matching values to names in order.
            seeds, matrix = paired_matrix(
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                groups[output["left"]], groups[output["right"]], density, metric
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            )
            # = stores a result in output[metric]; call crossed_bootstrap using parentheses to supply its inputs.
            output[metric] = crossed_bootstrap(matrix)
            # = stores a result in output[metric]['training_seed_ids']; evaluate the expression on the right before using its result.
            output[metric]["training_seed_ids"] = seeds
        # Call paired.append using parentheses to supply its inputs; this line does not store a separate return value.
        paired.append(output)
    # = stores a result in frontier; collect or generate one result per item; the "for" part names the item being processed.
    frontier = [
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        {
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "alpha": alpha,
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "summary": summarize_group(
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                [load(eval_path(seed, alpha)) for seed in [0, 1, 2]]
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            ),
        # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
        }
        # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
        for alpha in ALPHAS
    # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
    ]
    # = stores a result in calibration; build a dictionary, which stores values under named keys.
    calibration = {}
    # = stores a result in failures; build a dictionary, which stores values under named keys.
    failures = {}
    # for takes each item from ['mean', 'tail25', 'governed'] in turn; policy holds the current item.
    for policy in ["mean", "tail25", "governed"]:
        # = stores a result in datasets; collect or generate one result per item; the "for" part names the item being processed.
        datasets = [
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            load(stage / f"calibration_{policy}_seed{seed}.json") for seed in [0, 1, 2]
        # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
        ]
        # for takes each item from enumerate(datasets) in turn; (seed, data) holds the current item.
        for seed, data in enumerate(datasets):
            # = stores a result in p; use square brackets to select part of data.
            p = data["provenance"]
            # if checks p['training_seed'] != seed or p['seed_start'] != 12000 or p['episodes_per_density'] != 60 or (p['...; only a true condition runs the indented block.
            if (
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                p["training_seed"] != seed
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                or p["seed_start"] != 12000
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                or p["episodes_per_density"] != 60
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                or p["training_steps"] != 20000
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                or p["alpha"] != (0.25 if policy == "tail25" else 1.0)
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                or p["speed_cap"] != (cap if policy == "governed" else None)
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            ):
                # raise signals an error and stops this normal path, so invalid data are not silently accepted.
                raise ValueError(
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    "Calibration provenance does not match risk analysis protocol"
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                )
        # = unpacks the right-hand result into (calibration[policy], failures[policy]), matching values to names in order.
        calibration[policy], failures[policy] = [], []
        # for takes each item from [1.0, 1.5, 2.0] in turn; density holds the current item.
        for density in [1.0, 1.5, 2.0]:
            # = stores a result in rows; collect or generate one result per item; the "for" part names the item being processed.
            rows = [
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                row
                # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                for data in datasets
                # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                for row in data["episodes"]
                # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                if row["density"] == density
            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
            ]
            # = stores a result in metrics; call summarize_rows using parentheses to supply its inputs.
            metrics = summarize_rows(rows)
            # = stores a result in metrics['per_seed']; collect or generate one result per item; the "for" part names the item being processed.
            metrics["per_seed"] = [
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                {
                    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                    "training_seed": seed,
                    # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
                    **summarize_rows(
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        [row for row in data["episodes"] if row["density"] == density]
                    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                    ),
                # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
                }
                # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                for seed, data in enumerate(datasets)
            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
            ]
            # Call calibration[policy].append using parentheses to supply its inputs; this line does not store a separate return value.
            calibration[policy].append({"density": density, **metrics})
            # = stores a result in crashed; collect or generate one result per item; the "for" part names the item being processed.
            crashed = [row for row in rows if row["crashed"]]
            # = stores a result in proxy_counts; collect or generate one result per item; the "for" part names the item being processed.
            proxy_counts = {
                # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                name: sum(
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    row["terminal_nearest_neighbor"]["position_proxy"] == name
                    # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                    for row in crashed
                    # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                    if row["terminal_nearest_neighbor"]
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                )
                # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                for name in ["front", "rear", "side"]
            # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
            }
            # Call failures[policy].append using parentheses to supply its inputs; this line does not store a separate return value.
            failures[policy].append(
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                {
                    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                    "density": density,
                    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                    "episodes": len(rows),
                    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                    "crashes": len(crashed),
                    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                    "early_crashes_within_3_seconds": sum(
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        row["length"] <= 3 for row in crashed
                    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                    ),
                    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                    "recent_lateral_request_given_crash": metrics[
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        "recent_lateral_request_given_crash"
                    # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
                    ],
                    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                    "recent_lane_transition_given_crash": metrics[
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        "recent_lane_transition_given_crash"
                    # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
                    ],
                    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                    "terminal_nearest_neighbor_proxy_counts": proxy_counts,
                    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                    "all_observation_slots_filled_near_crash_fraction": (
                        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                        float(
                            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                            np.mean(
                                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                                [
                                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                                    row["recent_states"][-1]["observed_vehicle_slots"]
                                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                                    == 8
                                    # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                                    for row in crashed
                                # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
                                ]
                            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                            )
                        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                        )
                        # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                        if crashed
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        else None
                    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                    ),
                # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
                }
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            )
    # = stores a result in result; build a dictionary, which stores values under named keys.
    result = {
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "status": "Exploratory development; final confirmation seeds remain unused",
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "selection": selection,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "groups": {name: summarize_group(data) for name, data in groups.items()},
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "paired_comparisons": paired,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "risk_frontier": frontier,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "calibration": calibration,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "failure_diagnostics": failures,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "interpretation": {
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "paired_intervals": "Exploratory two-way bootstrap, conditional on development choices; only three training seeds.",
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "calibration": "Marginal empirical compatibility brackets account for censoring and ties, not sampling uncertainty. Initial-state samples; changed-policy continuations differ from training.",
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "failures": "Descriptive recent-action and geometry proxies, not identified crash causes or proof of unavoidable collisions.",
        # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
        },
    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
    }
    # Call save_json using parentheses to supply its inputs; this line does not store a separate return value.
    save_json(stage / "summary.json", result)
    # return ends this function and sends its result back to the code that called it.
    return result
