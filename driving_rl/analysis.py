"""Compare saved results without training any new agents.

Always compare matching training seeds and matching traffic scenarios. Otherwise
a policy might look better simply because it faced easier traffic. A bootstrap
interval estimates uncertainty by repeatedly sampling the observed cases again.
Scenario uncertainty and uncertainty from training different models are different
questions; the functions below keep those questions separate.

Original implementation notes:
Seed-aware summaries and paired comparisons for the fixed control experiments design.
"""

import argparse  # Make argparse available for this file's calculations and workflow.
import json  # Make json available for this file's calculations and workflow.
from pathlib import Path  # Make Path available for this file's calculations and workflow.

import numpy as np  # Make numpy available for this file's calculations and workflow.

from .experiment import save_json  # Make save_json available for this file's calculations and workflow.


def paired_matrix(left, right, density, metric):  # Line up the same traffic scenarios and training seeds before subtracting results.
    """Line up the same traffic scenarios and training seeds before subtracting results.
    Rows represent model seeds and columns represent scenario seeds. Reject missing
    or duplicated cases, which could otherwise make the comparison unfair.

    Original implementation notes:
    Align by training seed AND scenario seed; reject missing or duplicate data.
    """

    def index(evaluations):  # Build a lookup by model seed and scenario seed, rejecting duplicate cases.
        """Build a lookup by model seed and scenario seed, rejecting duplicate cases."""  # Describe this operation for readers and help().
        indexed = {}  # Set indexed to the named settings or measurements assembled below.
        for data in evaluations:  # Process each data from evaluations.
            training_seed = data["provenance"]["training_seed"]  # Set the independently trained model ID to the entries of the entries of data selected by 'provenance' selected by 'training_seed'.
            for row in data["episodes"]:  # Process each row from the entries of data selected by 'episodes'.
                if row["density"] != density:  # Use this branch when the entries of row selected by 'density' differs from the traffic-density setting.
                    continue  # Skip this case and move on to the next loop item.
                key = (training_seed, row["seed"])  # Set key to the independently trained model ID, the entries of row selected by 'seed'.
                if key in indexed:  # Use this branch when key belongs to indexed.
                    raise ValueError(f"Duplicate paired scenario: {key}")  # Reject this invalid case instead of continuing with misleading experiment results.
                indexed[key] = float(row[metric])  # Set the entries of indexed selected by key to the entries of row selected by metric as a floating-point value.
        return indexed  # Return indexed.

    first, second = index(left), index(right)  # Set first, second to index applied to left, index applied to right.
    if not first or set(first) != set(second):  # Use this branch when not first or set applied to first differs from set applied to second.
        raise ValueError("Paired comparison requires identical training/scenario keys")  # Reject this invalid case instead of continuing with misleading experiment results.
    seeds = sorted({key[0] for key in first})  # Set seeds to sorted applied to the entries of key selected by 0 for each key in first.
    scenarios = sorted({key[1] for key in first})  # Set scenarios to sorted applied to the entries of key selected by 1 for each key in first.
    if len(first) != len(seeds) * len(scenarios):  # Use this branch when the number of entries in first differs from (the number of entries in seeds times the number of entries in scenarios).
        raise ValueError(  # Reject this invalid case instead of continuing with misleading experiment results.
            "Paired data must form a complete training-seed by scenario matrix"  # Include the report text in the surrounding operation.
        )  # Finish assembling ValueError inputs.
    return seeds, np.array(  # Return seeds, first[seed, scenario] - second[seed, scenario] for each the starting-traffic seed in scenarios for each seed in seeds as a numerical array/tensor for batch calculations.
        [  # Build [training model, scenario] differences. Each cell subtracts policies facing matching initial traffic; positive is left minus right.
            [first[seed, scenario] - second[seed, scenario] for scenario in scenarios]  # Include (the entries of first selected by (seed, scenario) minus the entries of second selected by (seed, scenario)) for each the starting-traffic seed in scenarios in the surrounding operation.
            for seed in seeds  # Include seed, seeds in the surrounding operation.
        ]  # Finish assembling the grouped values above.
    )  # Finish assembling the grouped values above.


def crossed_bootstrap(differences, replicates=5000, seed=2026):  # Estimate variation from both training seeds and traffic scenarios.
    """Estimate variation from both training seeds and traffic scenarios.
    Sample rows and columns independently, keeping the left/right policy comparison
    paired. With only three training seeds, treat the interval as exploratory.

    Original implementation notes:
    Resample the two crossed factors independently; retain policy pairing.

    Rows are independently trained models, columns are shared scenario seeds.
    This avoids pretending that repeated scenarios are independent training runs.
    With only three training seeds the resulting interval remains exploratory.
    """
    values = np.asarray(differences, dtype=float)  # Set values to differences as a numerical array/tensor for batch calculations.
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():  # Use this branch when values.ndim differs from 2 or not values.size or not the result of np.isfinite(values).all.
        raise ValueError("Expected a nonempty finite 2D difference matrix")  # Reject this invalid case instead of continuing with misleading experiment results.
    rng = np.random.default_rng(seed)  # Set rng to a separate reproducible random generator seeded with seed.
    rows = rng.integers(values.shape[0], size=(replicates, values.shape[0]))  # Resample model rows independently of scenario columns, while keeping each policy difference paired.
    columns = rng.integers(values.shape[1], size=(replicates, values.shape[1]))  # A shared scenario stays the same column across model rows in a bootstrap sample.
    samples = values[rows[:, :, None], columns[:, None, :]].mean(axis=(1, 2))  # Set samples to the average of the entries of values selected by (rows[:, :, None], columns[:, None, :]) along axis (1, 2).
    low, high = np.quantile(samples, [0.025, 0.975])  # rows is [replicate, sampled model]; columns is [replicate, sampled scenario]. Adding None axes makes a rectangular grid per replicate. Paired indexing retrieves that grid, then mean over axes 1 and 2 yields one effect estimate. The middle 95% of the resampled estimates forms an approximate uncertainty interval.
    return {  # Return the named settings or measurements assembled below.
        "mean_difference": float(values.mean()),  # Record 'mean_difference' as the average of values as a floating-point value.
        "exploratory_95_percent_interval": [float(low), float(high)],  # Record 'exploratory_95_percent_interval' as low as a floating-point value, high as a floating-point value.
        "per_training_seed_differences": values.mean(axis=1).tolist(),  # Record 'per_training_seed_differences' as the average of values along axis 1 as a list that can be saved in JSON.
        "training_seeds": values.shape[0],  # Record 'training_seeds' as the entries of values.shape selected by 0.
        "scenarios_per_seed": values.shape[1],  # Record 'scenarios_per_seed' as the entries of values.shape selected by 1.
        "bootstrap_replicates": replicates,  # Record 'bootstrap_replicates' as replicates.
        "bootstrap_rng_seed": seed,  # Record 'bootstrap_rng_seed' as seed.
    }  # Finish assembling the grouped values above.


def choose_speed_targeted_baseline(candidates, target_speed):  # Choose a collision penalty by speed closeness, not by collision performance.
    """Choose a collision penalty by speed closeness, not by collision performance.
    If speeds tie, prefer the smaller penalty magnitude. This is a development-data
    selection rule; the same data cannot then be treated as untouched confirmation.

    Original implementation notes:
    Choose by speed closeness only; ties prefer the smaller penalty magnitude.
    """
    if not candidates:  # Use this branch when not candidates.
        raise ValueError("No baseline candidates")  # Reject this invalid case instead of continuing with misleading experiment results.
    return min(  # Return min applied to candidates.
        candidates,  # Include candidates in the surrounding operation.
        key=lambda item: (  # Include lambda item: (abs(item['speed m s'] - target speed), abs(item['collision penalty'])) in the surrounding operation.
            abs(item["speed_m_s"] - target_speed),  # Include abs applied to (the entries of item selected by 'speed_m_s' minus target speed) in the surrounding operation.
            abs(item["collision_penalty"]),  # Include abs applied to the entries of item selected by 'collision_penalty' in the surrounding operation.
        ),  # Finish assembling the grouped values above.
    )  # Finish assembling min inputs.


def summarize_group(evaluations):  # Carry out the summarize group part of this module's workflow.
    """Carry out the summarize group part of this module's workflow.
    See the calling function for the settings and saved files used here.
    """
    if not evaluations:  # Use this branch when not evaluations.
        return []  # Return .
    densities = sorted({row["density"] for row in evaluations[0]["summary"]})  # Set densities to sorted applied to the entries of row selected by 'density' for each row in the entries of the entries of evaluations selected by 0 selected by 'summary'.
    output = []  # Set output to .
    for density in densities:  # Process each the traffic-density setting from densities.
        seed_rows = []  # Set seed rows to .
        for data in evaluations:  # Process each data from evaluations.
            rows = [row for row in data["episodes"] if row["density"] == density]  # Set rows to row for each row in the entries of data selected by 'episodes', keeping only cases where the entries of row selected by 'density' equals the traffic-density setting.
            if not rows:  # Use this branch when not rows.
                raise ValueError("Each trained model needs every requested density")  # Reject this invalid case instead of continuing with misleading experiment results.
            returns = np.array([row["return"] for row in rows])  # Set returns to the entries of row selected by 'return' for each row in rows as a numerical array/tensor for batch calculations.
            tail_count = max(1, int(np.ceil(0.1 * len(rows))))  # Set tail count to max applied to 1, np.ceil applied to (0.1 times len(rows)) as a whole-number value.
            distance = sum(row["distance_m"] for row in rows)  # This tail averages the worst realized EPISODE scores. It is a reporting statistic, distinct from averaging a network's lowest predicted quantiles to choose an action inside one observation.
            seed_rows.append(  # Add the named settings or measurements assembled below to seed rows.
                {  # Include the named settings or measurements assembled below in the surrounding operation.
                    "training_seed": data["provenance"]["training_seed"],  # Record 'training_seed' as the entries of the entries of data selected by 'provenance' selected by 'training_seed'.
                    "episodes": len(rows),  # Record 'episodes' as the number of entries in rows.
                    "crash_rate": float(np.mean([row["crashed"] for row in rows])),  # Record 'crash_rate' as the average of the entries of row selected by 'crashed' for each row in rows as a floating-point value.
                    "mean_speed_m_s": float(  # Record 'mean_speed_m_s' as the average of the entries of row selected by 'mean_speed_m_s' for each row in rows as a floating-point value.
                        np.mean([row["mean_speed_m_s"] for row in rows])  # Include the average of the entries of row selected by 'mean_speed_m_s' for each row in rows in the surrounding operation.
                    ),  # Finish assembling float inputs.
                    "mean_return": float(returns.mean()),  # Record 'mean_return' as the average of returns as a floating-point value.
                    "worst_10_percent_mean_return": float(  # Record 'worst_10_percent_mean_return' as the average of the entries of the values sorted from lowest to highest in returns selected by :tail_count as a floating-point value.
                        np.sort(returns)[:tail_count].mean()  # Include the average of the entries of the values sorted from lowest to highest in returns selected by :tail_count in the surrounding operation.
                    ),  # Finish assembling float inputs.
                    "mean_distance_m": float(distance / len(rows)),  # Record 'mean_distance_m' as (distance divided by the number of entries in rows) as a floating-point value.
                    "mean_duration_s": float(np.mean([row["length"] for row in rows])),  # Record 'mean_duration_s' as the average of the entries of row selected by 'length' for each row in rows as a floating-point value.
                    "collisions_per_1000m": (  # Record 'collisions_per_1000m' as ((sum applied to (row['crashed'] for row in rows) divided by distance) times 1000) when distance, otherwise None.
                        sum(row["crashed"] for row in rows) / distance * 1000  # Include ((sum applied to (row['crashed'] for row in rows) divided by distance) times 1000) when distance, otherwise None in the surrounding operation.
                        if distance  # Include distance in the surrounding operation.
                        else None  # Include None in the surrounding operation.
                    ),  # Finish assembling the grouped values above.
                }  # Finish assembling the grouped values above.
            )  # Finish assembling seed rows.append inputs.
        aggregate = {  # Set aggregate to the named settings or measurements assembled below.
            "density": density,  # Record 'density' as the traffic-density setting.
            "training_seeds": len(seed_rows),  # Record 'training_seeds' as the number of entries in seed rows.
            "total_evaluation_episodes": sum(row["episodes"] for row in seed_rows),  # Record 'total_evaluation_episodes' as sum applied to the entries of row selected by 'episodes' for each row in seed rows.
            "per_seed": seed_rows,  # Record 'per_seed' as seed rows.
        }  # Finish assembling the grouped values above.
        for metric in [  # Compare one measured quantity at a time, keeping the scenarios paired.
            "crash_rate",  # Include 'crash_rate' in the surrounding operation.
            "mean_speed_m_s",  # Include 'mean_speed_m_s' in the surrounding operation.
            "mean_return",  # Include 'mean_return' in the surrounding operation.
            "worst_10_percent_mean_return",  # Include 'worst_10_percent_mean_return' in the surrounding operation.
            "mean_distance_m",  # Include 'mean_distance_m' in the surrounding operation.
            "mean_duration_s",  # Include 'mean_duration_s' in the surrounding operation.
        ]:  # Finish assembling the grouped values above.
            values = [row[metric] for row in seed_rows]  # Set values to the entries of row selected by metric for each row in seed rows.
            aggregate[metric] = float(np.mean(values))  # Set the entries of aggregate selected by metric to the average of values as a floating-point value.
            aggregate[metric + "_seed_range"] = [float(min(values)), float(max(values))]  # Set the entries of aggregate selected by metric + '_seed_range' to min applied to values as a floating-point value, max applied to values as a floating-point value.
        output.append(aggregate)  # Add aggregate to output.
    return output  # Return output.


def load_main_groups(root, seeds=(0, 1, 2)):  # Read the saved evaluation groups used in the main comparison.
    """Read the saved evaluation groups used in the main comparison."""  # Describe this operation for readers and help().
    groups = {}  # Set groups to the named settings or measurements assembled below.
    definitions = {  # Set definitions to the named settings or measurements assembled below.
        "DQN": "dqn_seed{seed}_mean",  # Record 'DQN' as 'dqn_seed{seed}_mean'.
        "QR-DQN mean": "qrdqn_seed{seed}_mean",  # Record 'QR-DQN mean' as 'qrdqn_seed{seed}_mean'.
        "QR-DQN lower 25%": "qrdqn_seed{seed}_tail25",  # Record 'QR-DQN lower 25%' as 'qrdqn_seed{seed}_tail25'.
    }  # Finish assembling the grouped values above.
    for label, pattern in definitions.items():  # Process each label, pattern from the result of definitions.items.
        groups[label] = [  # Set the entries of groups selected by label to json.loads applied to the result of (root / f'eval {pattern.format(seed=seed)}.json').read text for each seed in seeds.
            json.loads((root / f"eval_{pattern.format(seed=seed)}.json").read_text())  # Include json.loads applied to the result of (root / f'eval {pattern.format(seed=seed)}.json').read text in the surrounding operation.
            for seed in seeds  # Include seed, seeds in the surrounding operation.
        ]  # Finish assembling the grouped values above.
    selection_path = root / "controls" / "baseline_selection.json"  # Set selection path to the path to 'baseline_selection.json' inside the path to 'controls' inside root.
    if selection_path.exists():  # Use this branch when the result of selection path.exists.
        selection = json.loads(selection_path.read_text())  # Set selection to json.loads applied to the result of selection path.read text.
        penalty = selection["selected_collision_penalty"]  # Set penalty to the entries of selection selected by 'selected_collision_penalty'.
        if penalty == -2:  # Use this branch when penalty equals negative 2.
            groups["Reward-tuned DQN"] = groups["DQN"]  # Set the entries of groups selected by 'Reward-tuned DQN' to the entries of groups selected by 'DQN'.
        else:  # Handle the alternative when the earlier condition (penalty equals negative 2) is false.
            groups["Reward-tuned DQN"] = [  # Set the entries of groups selected by 'Reward-tuned DQN' to json.loads applied to the result of (root / 'controls' / f'eval dqn penalty{abs(int(penalty))} seed{seed}.json').read text for each seed in seeds.
                json.loads(  # Include json.loads applied to the result of (root / 'controls' / f'eval dqn penalty{abs(int(penalty))} seed{seed}.json').read text in the surrounding operation.
                    (  # Include the result of (root / 'controls' / f'eval dqn penalty{abs(int(penalty))} seed{seed}.json').read text in the surrounding operation.
                        root  # Include (the path to 'controls' inside root divided by f'eval dqn penalty{abs(int(penalty))} seed{seed}.json') in the surrounding operation.
                        / "controls"  # Include 'controls' in the surrounding operation.
                        / f"eval_dqn_penalty{abs(int(penalty))}_seed{seed}.json"  # Include f'eval dqn penalty{abs(int(penalty))} seed{seed}.json' in the surrounding operation.
                    ).read_text()  # Finish assembling (root / 'controls' / f'eval dqn penalty{abs(int(penalty))} seed{seed}.json').read text inputs.
                )  # Finish assembling json.loads inputs.
                for seed in seeds  # Include seed, seeds in the surrounding operation.
            ]  # Finish assembling the grouped values above.
    governor_path = root / "controls" / "governor_selection.json"  # Set governor path to the path to 'governor_selection.json' inside the path to 'controls' inside root.
    if governor_path.exists():  # Use this branch when the result of governor path.exists.
        cap = json.loads(governor_path.read_text())["selected_cap_m_s"]  # Set cap to the entries of json.loads applied to the result of governor path.read text selected by 'selected_cap_m_s'.
        groups["DQN + speed governor"] = [  # Set the entries of groups selected by 'DQN + speed governor' to json.loads applied to the result of (root / 'controls' / f'eval dqn cap{cap} seed{seed}.json').read text for each seed in seeds.
            json.loads(  # Include json.loads applied to the result of (root / 'controls' / f'eval dqn cap{cap} seed{seed}.json').read text in the surrounding operation.
                (root / "controls" / f"eval_dqn_cap{cap}_seed{seed}.json").read_text()  # Include the result of (root / 'controls' / f'eval dqn cap{cap} seed{seed}.json').read text in the surrounding operation.
            )  # Finish assembling json.loads inputs.
            for seed in seeds  # Include seed, seeds in the surrounding operation.
        ]  # Finish assembling the grouped values above.
    for label, evaluations in groups.items():  # All physical comparisons use the same evaluation task and reward, even when the training reward was changed for the scalar baseline.
        observed_seeds = [item["provenance"]["training_seed"] for item in evaluations]  # Set observed seeds to the entries of the entries of item selected by 'provenance' selected by 'training_seed' for each item in evaluations.
        if sorted(observed_seeds) != sorted(seeds):  # Use this branch when sorted applied to observed seeds differs from sorted applied to seeds.
            raise ValueError(f"Unexpected or duplicate training seeds in {label}")  # Reject this invalid case instead of continuing with misleading experiment results.
        for item in evaluations:  # Process each item from evaluations.
            provenance = item["provenance"]  # Set provenance to the entries of item selected by 'provenance'.
            if (  # Use this branch when the entries of provenance selected by 'training_steps' differs from 20000 or the entries of provenance selected by 'seed_start' differs from 10000 or the entries of provenance selected by 'episodes_per_density' differs from 30 or provenance.get applied to 'evaluation_collision_penalty', negative 2.0 differs from negative 2.0.
                provenance["training_steps"] != 20000  # Include the entries of provenance selected by 'training_steps' differs from 20000 or the entries of provenance selected by 'seed_start' differs from 10000 or the entries of provenance selected by 'episodes_per_density' differs from 30 or provenance.get applied to 'evaluation_collision_penalty', negative 2.0 differs from negative 2.0 in the surrounding operation.
                or provenance["seed_start"] != 10000  # Include the entries of provenance selected by 'seed_start' differs from 10000 in the surrounding operation.
                or provenance["episodes_per_density"] != 30  # Include the entries of provenance selected by 'episodes_per_density' differs from 30 in the surrounding operation.
                or provenance.get("evaluation_collision_penalty", -2.0) != -2.0  # Include provenance.get applied to 'evaluation_collision_penalty', negative 2.0 differs from negative 2.0 in the surrounding operation.
            ):  # Finish assembling the grouped values above.
                raise ValueError(  # Reject this invalid case instead of continuing with misleading experiment results.
                    f"Evaluation does not match fixed control experiments design: {label}"  # Include f'Evaluation does not match fixed control experiments design: {label}' in the surrounding operation.
                )  # Finish assembling ValueError inputs.
            expected_alpha = 0.25 if label == "QR-DQN lower 25%" else 1.0  # Set expected alpha to 0.25 when label equals 'QR-DQN lower 25%', otherwise 1.0.
            expected_algorithm = "qrdqn" if label.startswith("QR-DQN") else "dqn"  # Set expected algorithm to 'qrdqn' when label.startswith applied to 'QR-DQN', otherwise 'dqn'.
            if (  # Use this branch when the entries of provenance selected by 'alpha' differs from expected alpha or the entries of provenance selected by 'algorithm' differs from expected algorithm.
                provenance["alpha"] != expected_alpha  # Include the entries of provenance selected by 'alpha' differs from expected alpha or the entries of provenance selected by 'algorithm' differs from expected algorithm in the surrounding operation.
                or provenance["algorithm"] != expected_algorithm  # Include the entries of provenance selected by 'algorithm' differs from expected algorithm in the surrounding operation.
            ):  # Finish assembling the grouped values above.
                raise ValueError(f"Incorrect policy identity in {label}")  # Reject this invalid case instead of continuing with misleading experiment results.
            expected_cap = (  # Set expected cap to the entries of json.loads applied to the result of governor path.read text selected by 'selected_cap_m_s' when label equals 'DQN + speed governor', otherwise None.
                json.loads(governor_path.read_text())["selected_cap_m_s"]  # Include the entries of json.loads applied to the result of governor path.read text selected by 'selected_cap_m_s' when label equals 'DQN + speed governor', otherwise None in the surrounding operation.
                if label == "DQN + speed governor"  # Include label equals 'DQN + speed governor' in the surrounding operation.
                else None  # Include None in the surrounding operation.
            )  # Finish assembling the grouped values above.
            if provenance.get("speed_governor_cap_m_s") != expected_cap:  # Use this branch when provenance.get applied to 'speed_governor_cap_m_s' differs from expected cap.
                raise ValueError(f"Incorrect governor setting in {label}")  # Reject this invalid case instead of continuing with misleading experiment results.
    return groups  # Return groups.


def analyze(root):  # Combine the saved experiments into comparison tables.
    """Combine the saved experiments into comparison tables. Match scenarios and
    model seeds before calculating differences, and keep development limitations visible.
    """
    groups = load_main_groups(root)  # Set groups to load main groups applied to root.
    comparisons = []  # Set comparisons to .
    contrasts = [("QR-DQN lower 25%", "QR-DQN mean")]  # Set contrasts to 'QR-DQN lower 25%', 'QR-DQN mean'.
    for control in ["Reward-tuned DQN", "DQN + speed governor"]:  # Process each control from 'Reward-tuned DQN', 'DQN + speed governor'.
        if control in groups:  # Use this branch when control belongs to groups.
            contrasts.append(("QR-DQN lower 25%", control))  # Add 'qr-dqn lower 25%', control to contrasts.
    for left, right in contrasts:  # Process each left, right from contrasts.
        for density in [1.0, 1.5, 2.0]:  # Process each the traffic-density setting from 1.0, 1.5, 2.0.
            result = {  # Set result to the named settings or measurements assembled below.
                "left": left,  # Record 'left' as left.
                "right": right,  # Record 'right' as right.
                "density": density,  # Record 'density' as the traffic-density setting.
                "sign": "left minus right",  # Record 'sign' as 'left minus right'.
            }  # Finish assembling the grouped values above.
            for metric in ["crashed", "mean_speed_m_s", "distance_m", "return"]:  # Compare one measured quantity at a time, keeping the scenarios paired.
                seeds, differences = paired_matrix(  # Set seeds, differences to paired matrix applied to the entries of groups selected by left, the entries of groups selected by right, the traffic-density setting, metric.
                    groups[left], groups[right], density, metric  # Include the entries of groups selected by left, the entries of groups selected by right, the traffic-density setting, metric in the surrounding operation.
                )  # Finish assembling paired matrix inputs.
                result[metric] = crossed_bootstrap(differences)  # Set the entries of result selected by metric to crossed bootstrap applied to differences.
                result[metric]["training_seed_ids"] = seeds  # Set the entries of the entries of result selected by metric selected by 'training_seed_ids' to seeds.
            comparisons.append(result)  # Add result to comparisons.
    output = {  # Set output to the named settings or measurements assembled below.
        "status": "Development experiment; final confirmation scenarios remain reserved",  # Record 'status' as the report text.
        "groups": {label: summarize_group(values) for label, values in groups.items()},  # Record 'groups' as summarize group applied to values for each label, values in the result of groups.items.
        "paired_comparisons": comparisons,  # Record 'paired_comparisons' as comparisons.
        "uncertainty_note": "Equal-weight training-seed means and seed ranges. Paired differences use an exploratory two-way bootstrap over the three training seeds and the shared scenario seeds. Three training seeds provide limited evidence of robustness.",  # Record 'uncertainty_note' as the report text.
    }  # Finish assembling the grouped values above.
    save_json(root / "controls" / "summary.json", output)  # Save the following results/settings as json: the path to 'summary.json' inside the path to 'controls' inside root, output.
    return output  # Return output.


def main():  # Read Terminal options, check settings, and run the requested workflow.
    """Read Terminal options, check settings, and run the requested workflow.
    Existing result files are checked before reuse so a rerun cannot silently mix
    different experiments. The main guard below prevents running jobs on import.
    """
    parser = argparse.ArgumentParser(description=__doc__)  # Set parser to the result of argparse.ArgumentParser.
    parser.add_argument("--results", default="results")  # Parser.add argument applied to '--results'.
    args = parser.parse_args()  # Set args to the result of parser.parse args.
    result = analyze(Path(args.results))  # Set result to analyze applied to Path applied to args.results.
    for label, rows in result["groups"].items():  # Process each label, rows from the result of result['groups'].items.
        print(  # Print applied to label, the entries of row selected by 'density', round applied to row['crash rate'], 3, round applied to row['mean speed m s'], 2 for each row in rows.
            label,  # Include label in the surrounding operation.
            [  # Include the entries of row selected by 'density', round applied to the entries of row selected by 'crash_rate', 3, round applied to the entries of row selected by 'mean_speed_m_s', 2 for each row in rows in the surrounding operation.
                (  # Include the entries of row selected by 'density', round applied to the entries of row selected by 'crash_rate', 3, round applied to the entries of row selected by 'mean_speed_m_s', 2 in the surrounding operation.
                    row["density"],  # Include the entries of row selected by 'density' in the surrounding operation.
                    round(row["crash_rate"], 3),  # Include round applied to the entries of row selected by 'crash_rate', 3 in the surrounding operation.
                    round(row["mean_speed_m_s"], 2),  # Include round applied to the entries of row selected by 'mean_speed_m_s', 2 in the surrounding operation.
                )  # Finish assembling the grouped values above.
                for row in rows  # Include row, rows in the surrounding operation.
            ],  # Finish assembling print inputs.
        )  # Finish assembling print inputs.


if __name__ == "__main__":  # Run this workflow when launched from Terminal; importing its helpers must not launch experiments.
    main()  # The result of main.


def choose_speed_matched_control(candidates, target_speed):  # Choose the speed threshold whose observed speed is closest to the target.
    """Choose the speed threshold whose observed speed is closest to the target.
    If two thresholds are equally close, use the lower one. Collision results are
    deliberately excluded from this choice so we do not select a flattering outcome.

    Original implementation notes:
    Use speed closeness only; tie-break by lower cap, never collision rate.
    """
    if not candidates or not np.isfinite(target_speed) or target_speed <= 0:  # Use this branch when not candidates or not a true/false check for finite numerical values in target speed or target speed is at most 0.
        raise ValueError("Require candidates and a finite positive target speed")  # Reject this invalid case instead of continuing with misleading experiment results.
    if any(  # Use this branch when any applied to not np.isfinite(r['speed m s']) or not np.isfinite(r['cap m s']) or r['cap m s'] is at most 0 for each r in candidates.
        not np.isfinite(r["speed_m_s"])  # Include not a true/false check for finite numerical values in the entries of r selected by 'speed_m_s' or not a true/false check for finite numerical values in the entries of r selected by 'cap_m_s' or the entries of r selected by 'cap_m_s' is at most 0 in the surrounding operation.
        or not np.isfinite(r["cap_m_s"])  # Include not a true/false check for finite numerical values in the entries of r selected by 'cap_m_s' in the surrounding operation.
        or r["cap_m_s"] <= 0  # Include the entries of r selected by 'cap_m_s' is at most 0 in the surrounding operation.
        for r in candidates  # Include r, candidates in the surrounding operation.
    ):  # Finish assembling any inputs.
        raise ValueError("Require finite speeds and positive finite caps")  # Reject this invalid case instead of continuing with misleading experiment results.
    return min(  # Return min applied to candidates.
        candidates, key=lambda r: (abs(r["speed_m_s"] - target_speed), r["cap_m_s"])  # Include candidates, lambda r: (abs(r['speed m s'] - target speed), r['cap m s']) in the surrounding operation.
    )  # Finish assembling min inputs.


def scenario_bootstrap_fixed_models(values, replicates=10000, seed=20261111):  # Estimate scenario uncertainty while keeping the trained models fixed.
    """Estimate scenario uncertainty while keeping the trained models fixed.
    Average the fixed models within each scenario, then resample those scenario
    averages. Do not treat three models on the same scenario as three independent
    scenarios. This interval does not describe training a completely new network.

    Original implementation notes:
    Resample scenario columns, retaining all fixed models and policy pairing.

    Rows are the locked trained models, not new training draws. A scenario is
    the unit of resampling because its initial traffic is shared across models.
    Apply this to a paired-difference matrix for a conditional policy contrast.
    """
    matrix = np.asarray(values, dtype=float)  # Set matrix to values as a numerical array/tensor for batch calculations.
    if matrix.ndim != 2 or not matrix.size or not np.isfinite(matrix).all():  # Use this branch when matrix.ndim differs from 2 or not matrix.size or not the result of np.isfinite(matrix).all.
        raise ValueError("Expected a finite nonempty model-by-scenario matrix")  # Reject this invalid case instead of continuing with misleading experiment results.
    if not isinstance(replicates, int) or replicates < 1:  # Use this branch when not isinstance applied to replicates, int or replicates is below 1.
        raise ValueError("Replicates must be a positive integer")  # Reject this invalid case instead of continuing with misleading experiment results.
    by_scenario = matrix.mean(axis=0)  # For fixed-model inference, each scenario contributes one average across the locked model pairs.
    rng = np.random.default_rng(seed)  # Set rng to a separate reproducible random generator seeded with seed.
    indices = rng.integers(len(by_scenario), size=(replicates, len(by_scenario)))  # Resample whole scenarios, not individual model/scenario cells.
    samples = by_scenario[indices].mean(axis=1)  # Set samples to the average of the entries of by scenario selected by indices along axis 1.
    low, high = np.quantile(samples, [0.025, 0.975])  # indices is a table: each row is one resampled collection of scenario IDs. Indexing with that table retrieves their effects; axis=1 averages each collection into one bootstrap estimate. Repeated IDs are intended. The middle 95% of the resampled estimates forms an approximate uncertainty interval.
    return {  # Return the named settings or measurements assembled below.
        "mean_difference": float(matrix.mean()),  # Record 'mean_difference' as the average of matrix as a floating-point value.
        "conditional_95_percent_interval": [float(low), float(high)],  # Record 'conditional_95_percent_interval' as low as a floating-point value, high as a floating-point value.
        "per_training_seed_differences": matrix.mean(axis=1).tolist(),  # Record 'per_training_seed_differences' as the average of matrix along axis 1 as a list that can be saved in JSON.
        "fixed_model_pairs": matrix.shape[0],  # Record 'fixed_model_pairs' as the entries of matrix.shape selected by 0.
        "scenario_clusters": matrix.shape[1],  # Record 'scenario_clusters' as the entries of matrix.shape selected by 1.
        "bootstrap_replicates": replicates,  # Record 'bootstrap_replicates' as replicates.
        "bootstrap_rng_seed": seed,  # Record 'bootstrap_rng_seed' as seed.
        "scope": "Fresh scenarios conditional on the locked trained models; no new-training generalization.",  # Record 'scope' as the report text.
    }  # Finish assembling the grouped values above.
