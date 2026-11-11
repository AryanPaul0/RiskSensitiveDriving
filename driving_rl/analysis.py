"""Compare saved results without training any new agents.

Always compare matching training seeds and matching traffic scenarios. Otherwise
a policy might look better simply because it faced easier traffic. A bootstrap
interval estimates uncertainty by repeatedly sampling the observed cases again.
Scenario uncertainty and uncertainty from training different models are different
questions; the functions below keep those questions separate.

Original implementation notes:
Seed-aware summaries and paired comparisons for the fixed control experiments design.
"""

# import makes these modules available; "as" gives a module a shorter local name.
import argparse
# import makes these modules available; "as" gives a module a shorter local name.
import json
# from pathlib import brings the named tools into this file; parentheses allow several lines.
from pathlib import Path

# import makes these modules available; "as" gives a module a shorter local name.
import numpy as np

# from experiment import brings the named tools into this file; parentheses allow several lines.
from .experiment import save_json


# def defines paired_matrix; parentheses list its inputs, and the colon starts its indented body.
def paired_matrix(left, right, density, metric):
    """Line up the same traffic scenarios and training seeds before subtracting results.
    Rows represent model seeds and columns represent scenario seeds. Reject missing
    or duplicated cases, which could otherwise make the comparison unfair.

    Original implementation notes:
    Align by training seed AND scenario seed; reject missing or duplicate data.
    """

    # def defines index; parentheses list its inputs, and the colon starts its indented body.
    def index(evaluations):
        # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
        """Build a lookup by model seed and scenario seed, rejecting duplicate cases."""
        # = stores a result in indexed; build a dictionary, which stores values under named keys.
        indexed = {}
        # for takes each item from evaluations in turn; data holds the current item.
        for data in evaluations:
            # = stores a result in training_seed; use square brackets to select part of data['provenance'].
            training_seed = data["provenance"]["training_seed"]
            # for takes each item from data['episodes'] in turn; row holds the current item.
            for row in data["episodes"]:
                # if checks row['density'] != density; only a true condition runs the indented block.
                if row["density"] != density:
                    # continue skips the rest of this loop iteration and moves to its next item.
                    continue
                # = stores a result in key; group these values into a tuple, keeping their order.
                key = (training_seed, row["seed"])
                # if checks key in indexed; only a true condition runs the indented block.
                if key in indexed:
                    # raise signals an error and stops this normal path, so invalid data are not silently accepted.
                    raise ValueError(f"Duplicate paired scenario: {key}")
                # = stores a result in indexed[key]; call float using parentheses to supply its inputs.
                indexed[key] = float(row[metric])
        # return ends this function and sends its result back to the code that called it.
        return indexed

    # = unpacks the right-hand result into (first, second), matching values to names in order.
    first, second = index(left), index(right)
    # if checks not first or set(first) != set(second); only a true condition runs the indented block.
    if not first or set(first) != set(second):
        # raise signals an error and stops this normal path, so invalid data are not silently accepted.
        raise ValueError("Paired comparison requires identical training/scenario keys")
    # = stores a result in seeds; call sorted using parentheses to supply its inputs.
    seeds = sorted({key[0] for key in first})
    # = stores a result in scenarios; call sorted using parentheses to supply its inputs.
    scenarios = sorted({key[1] for key in first})
    # if checks len(first) != len(seeds) * len(scenarios); only a true condition runs the indented block.
    if len(first) != len(seeds) * len(scenarios):
        # raise signals an error and stops this normal path, so invalid data are not silently accepted.
        raise ValueError(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "Paired data must form a complete training-seed by scenario matrix"
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
    # return ends this function and sends its result back to the code that called it.
    return seeds, np.array(
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        [
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            [first[seed, scenario] - second[seed, scenario] for scenario in scenarios]
            # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
            for seed in seeds
        # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
        ]
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )


# def defines crossed_bootstrap; parentheses list its inputs, and the colon starts its indented body.
def crossed_bootstrap(differences, replicates=5000, seed=2026):
    """Estimate variation from both training seeds and traffic scenarios.
    Sample rows and columns independently, keeping the left/right policy comparison
    paired. With only three training seeds, treat the interval as exploratory.

    Original implementation notes:
    Resample the two crossed factors independently; retain policy pairing.

    Rows are independently trained models, columns are shared scenario seeds.
    This avoids pretending that repeated scenarios are independent training runs.
    With only three training seeds the resulting interval remains exploratory.
    """
    # = stores a result in values; call np.asarray using parentheses to supply its inputs.
    values = np.asarray(differences, dtype=float)
    # if checks values.ndim != 2 or not values.size or (not np.isfinite(values).all()); only a true condition runs the indented block.
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        # raise signals an error and stops this normal path, so invalid data are not silently accepted.
        raise ValueError("Expected a nonempty finite 2D difference matrix")
    # = stores a result in rng; call np.random.default_rng using parentheses to supply its inputs.
    rng = np.random.default_rng(seed)
    # Resample model rows independently of scenario columns, while keeping each policy difference paired.
    # = stores a result in rows; call rng.integers using parentheses to supply its inputs.
    rows = rng.integers(values.shape[0], size=(replicates, values.shape[0]))
    # A shared scenario stays the same column across model rows in a bootstrap sample.
    # = stores a result in columns; call rng.integers using parentheses to supply its inputs.
    columns = rng.integers(values.shape[1], size=(replicates, values.shape[1]))
    # = stores a result in samples; call values[rows[:, :, None], columns[:, None, :]].mean using parentheses to supply its inputs.
    samples = values[rows[:, :, None], columns[:, None, :]].mean(axis=(1, 2))
    # The middle 95% of the resampled estimates forms an approximate uncertainty interval.
    # = unpacks the right-hand result into (low, high), matching values to names in order.
    low, high = np.quantile(samples, [0.025, 0.975])
    # return ends this function and sends its result back to the code that called it.
    return {
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "mean_difference": float(values.mean()),
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "exploratory_95_percent_interval": [float(low), float(high)],
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "per_training_seed_differences": values.mean(axis=1).tolist(),
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "training_seeds": values.shape[0],
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "scenarios_per_seed": values.shape[1],
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "bootstrap_replicates": replicates,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "bootstrap_rng_seed": seed,
    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
    }


# def defines choose_speed_targeted_baseline; parentheses list its inputs, and the colon starts its indented body.
def choose_speed_targeted_baseline(candidates, target_speed):
    """Choose a collision penalty by speed closeness, not by collision performance.
    If speeds tie, prefer the smaller penalty magnitude. This is a development-data
    selection rule; the same data cannot then be treated as untouched confirmation.

    Original implementation notes:
    Choose by speed closeness only; ties prefer the smaller penalty magnitude.
    """
    # if checks not candidates; only a true condition runs the indented block.
    if not candidates:
        # raise signals an error and stops this normal path, so invalid data are not silently accepted.
        raise ValueError("No baseline candidates")
    # return ends this function and sends its result back to the code that called it.
    return min(
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        candidates,
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        key=lambda item: (
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            abs(item["speed_m_s"] - target_speed),
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            abs(item["collision_penalty"]),
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        ),
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )


# def defines summarize_group; parentheses list its inputs, and the colon starts its indented body.
def summarize_group(evaluations):
    """Carry out the summarize group part of this module's workflow.
    See the calling function for the settings and saved files used here.
    """
    # if checks not evaluations; only a true condition runs the indented block.
    if not evaluations:
        # return ends this function and sends its result back to the code that called it.
        return []
    # = stores a result in densities; call sorted using parentheses to supply its inputs.
    densities = sorted({row["density"] for row in evaluations[0]["summary"]})
    # = stores a result in output; build a list, keeping its items in order.
    output = []
    # for takes each item from densities in turn; density holds the current item.
    for density in densities:
        # = stores a result in seed_rows; build a list, keeping its items in order.
        seed_rows = []
        # for takes each item from evaluations in turn; data holds the current item.
        for data in evaluations:
            # = stores a result in rows; collect or generate one result per item; the "for" part names the item being processed.
            rows = [row for row in data["episodes"] if row["density"] == density]
            # if checks not rows; only a true condition runs the indented block.
            if not rows:
                # raise signals an error and stops this normal path, so invalid data are not silently accepted.
                raise ValueError("Each trained model needs every requested density")
            # = stores a result in returns; call np.array using parentheses to supply its inputs.
            returns = np.array([row["return"] for row in rows])
            # = stores a result in tail_count; call max using parentheses to supply its inputs.
            tail_count = max(1, int(np.ceil(0.1 * len(rows))))
            # = stores a result in distance; call sum using parentheses to supply its inputs.
            distance = sum(row["distance_m"] for row in rows)
            # Call seed_rows.append using parentheses to supply its inputs; this line does not store a separate return value.
            seed_rows.append(
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                {
                    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                    "training_seed": data["provenance"]["training_seed"],
                    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                    "episodes": len(rows),
                    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                    "crash_rate": float(np.mean([row["crashed"] for row in rows])),
                    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                    "mean_speed_m_s": float(
                        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                        np.mean([row["mean_speed_m_s"] for row in rows])
                    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                    ),
                    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                    "mean_return": float(returns.mean()),
                    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                    "worst_10_percent_mean_return": float(
                        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                        np.sort(returns)[:tail_count].mean()
                    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                    ),
                    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                    "mean_distance_m": float(distance / len(rows)),
                    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                    "mean_duration_s": float(np.mean([row["length"] for row in rows])),
                    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                    "collisions_per_1000m": (
                        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                        sum(row["crashed"] for row in rows) / distance * 1000
                        # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                        if distance
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        else None
                    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                    ),
                # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
                }
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            )
        # = stores a result in aggregate; build a dictionary, which stores values under named keys.
        aggregate = {
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "density": density,
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "training_seeds": len(seed_rows),
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "total_evaluation_episodes": sum(row["episodes"] for row in seed_rows),
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "per_seed": seed_rows,
        # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
        }
        # Compare one measured quantity at a time, keeping the scenarios paired.
        # for takes each item from ['crash_rate', 'mean_speed_m_s', 'mean_return', 'worst_10_percent_mean_return', 'mean_distance_m'... in turn; metric holds the current item.
        for metric in [
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "crash_rate",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "mean_speed_m_s",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "mean_return",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "worst_10_percent_mean_return",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "mean_distance_m",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "mean_duration_s",
        # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
        ]:
            # = stores a result in values; collect or generate one result per item; the "for" part names the item being processed.
            values = [row[metric] for row in seed_rows]
            # = stores a result in aggregate[metric]; call float using parentheses to supply its inputs.
            aggregate[metric] = float(np.mean(values))
            # = stores a result in aggregate[metric + '_seed_range']; build a list, keeping its items in order.
            aggregate[metric + "_seed_range"] = [float(min(values)), float(max(values))]
        # Call output.append using parentheses to supply its inputs; this line does not store a separate return value.
        output.append(aggregate)
    # return ends this function and sends its result back to the code that called it.
    return output


# def defines load_main_groups; parentheses list its inputs, and the colon starts its indented body.
def load_main_groups(root, seeds=(0, 1, 2)):
    # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
    """Read the saved evaluation groups used in the main comparison."""
    # = stores a result in groups; build a dictionary, which stores values under named keys.
    groups = {}
    # = stores a result in definitions; build a dictionary, which stores values under named keys.
    definitions = {
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "DQN": "dqn_seed{seed}_mean",
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "QR-DQN mean": "qrdqn_seed{seed}_mean",
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "QR-DQN lower 25%": "qrdqn_seed{seed}_tail25",
    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
    }
    # for takes each item from definitions.items() in turn; (label, pattern) holds the current item.
    for label, pattern in definitions.items():
        # = stores a result in groups[label]; collect or generate one result per item; the "for" part names the item being processed.
        groups[label] = [
            # Inside this continued call, name=value supplies an option by name rather than by its position.
            json.loads((root / f"eval_{pattern.format(seed=seed)}.json").read_text())
            # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
            for seed in seeds
        # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
        ]
    # = stores a result in selection_path; apply / to these operands; numbers are divided, while Path objects join folder names.
    selection_path = root / "controls" / "baseline_selection.json"
    # if checks selection_path.exists(); only a true condition runs the indented block.
    if selection_path.exists():
        # = stores a result in selection; call json.loads using parentheses to supply its inputs.
        selection = json.loads(selection_path.read_text())
        # = stores a result in penalty; use square brackets to select part of selection.
        penalty = selection["selected_collision_penalty"]
        # if checks penalty == -2; only a true condition runs the indented block.
        if penalty == -2:
            # = stores a result in groups['Reward-tuned DQN']; use square brackets to select part of groups.
            groups["Reward-tuned DQN"] = groups["DQN"]
        # else provides the alternative branch; after a loop it instead means the loop finished without break.
        else:
            # = stores a result in groups['Reward-tuned DQN']; collect or generate one result per item; the "for" part names the item being processed.
            groups["Reward-tuned DQN"] = [
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                json.loads(
                    # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                    (
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        root
                        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
                        / "controls"
                        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
                        / f"eval_dqn_penalty{abs(int(penalty))}_seed{seed}.json"
                    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                    ).read_text()
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                )
                # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                for seed in seeds
            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
            ]
    # = stores a result in governor_path; apply / to these operands; numbers are divided, while Path objects join folder names.
    governor_path = root / "controls" / "governor_selection.json"
    # if checks governor_path.exists(); only a true condition runs the indented block.
    if governor_path.exists():
        # = stores a result in cap; use square brackets to select part of json.loads(governor_path.read_text()).
        cap = json.loads(governor_path.read_text())["selected_cap_m_s"]
        # = stores a result in groups['DQN + speed governor']; collect or generate one result per item; the "for" part names the item being processed.
        groups["DQN + speed governor"] = [
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            json.loads(
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                (root / "controls" / f"eval_dqn_cap{cap}_seed{seed}.json").read_text()
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            )
            # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
            for seed in seeds
        # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
        ]
    # All physical comparisons use the same evaluation task and reward, even
    # when the training reward was changed for the scalar baseline.
    # for takes each item from groups.items() in turn; (label, evaluations) holds the current item.
    for label, evaluations in groups.items():
        # = stores a result in observed_seeds; collect or generate one result per item; the "for" part names the item being processed.
        observed_seeds = [item["provenance"]["training_seed"] for item in evaluations]
        # if checks sorted(observed_seeds) != sorted(seeds); only a true condition runs the indented block.
        if sorted(observed_seeds) != sorted(seeds):
            # raise signals an error and stops this normal path, so invalid data are not silently accepted.
            raise ValueError(f"Unexpected or duplicate training seeds in {label}")
        # for takes each item from evaluations in turn; item holds the current item.
        for item in evaluations:
            # = stores a result in provenance; use square brackets to select part of item.
            provenance = item["provenance"]
            # if checks provenance['training_steps'] != 20000 or provenance['seed_start'] != 10000 or provenance['episode...; only a true condition runs the indented block.
            if (
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                provenance["training_steps"] != 20000
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                or provenance["seed_start"] != 10000
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                or provenance["episodes_per_density"] != 30
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                or provenance.get("evaluation_collision_penalty", -2.0) != -2.0
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            ):
                # raise signals an error and stops this normal path, so invalid data are not silently accepted.
                raise ValueError(
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    f"Evaluation does not match fixed control experiments design: {label}"
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                )
            # = stores a result in expected_alpha; choose the value before "if" when its condition is true, otherwise the value after "else".
            expected_alpha = 0.25 if label == "QR-DQN lower 25%" else 1.0
            # = stores a result in expected_algorithm; choose the value before "if" when its condition is true, otherwise the value after "else".
            expected_algorithm = "qrdqn" if label.startswith("QR-DQN") else "dqn"
            # if checks provenance['alpha'] != expected_alpha or provenance['algorithm'] != expected_algorithm; only a true condition runs the indented block.
            if (
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                provenance["alpha"] != expected_alpha
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                or provenance["algorithm"] != expected_algorithm
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            ):
                # raise signals an error and stops this normal path, so invalid data are not silently accepted.
                raise ValueError(f"Incorrect policy identity in {label}")
            # = stores a result in expected_cap; choose the value before "if" when its condition is true, otherwise the value after "else".
            expected_cap = (
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                json.loads(governor_path.read_text())["selected_cap_m_s"]
                # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                if label == "DQN + speed governor"
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                else None
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            )
            # if checks provenance.get('speed_governor_cap_m_s') != expected_cap; only a true condition runs the indented block.
            if provenance.get("speed_governor_cap_m_s") != expected_cap:
                # raise signals an error and stops this normal path, so invalid data are not silently accepted.
                raise ValueError(f"Incorrect governor setting in {label}")
    # return ends this function and sends its result back to the code that called it.
    return groups


# def defines analyze; parentheses list its inputs, and the colon starts its indented body.
def analyze(root):
    """Combine the saved experiments into comparison tables. Match scenarios and
    model seeds before calculating differences, and keep development limitations visible.
    """
    # = stores a result in groups; call load_main_groups using parentheses to supply its inputs.
    groups = load_main_groups(root)
    # = stores a result in comparisons; build a list, keeping its items in order.
    comparisons = []
    # = stores a result in contrasts; build a list, keeping its items in order.
    contrasts = [("QR-DQN lower 25%", "QR-DQN mean")]
    # for takes each item from ['Reward-tuned DQN', 'DQN + speed governor'] in turn; control holds the current item.
    for control in ["Reward-tuned DQN", "DQN + speed governor"]:
        # if checks control in groups; only a true condition runs the indented block.
        if control in groups:
            # Call contrasts.append using parentheses to supply its inputs; this line does not store a separate return value.
            contrasts.append(("QR-DQN lower 25%", control))
    # for takes each item from contrasts in turn; (left, right) holds the current item.
    for left, right in contrasts:
        # for takes each item from [1.0, 1.5, 2.0] in turn; density holds the current item.
        for density in [1.0, 1.5, 2.0]:
            # = stores a result in result; build a dictionary, which stores values under named keys.
            result = {
                # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                "left": left,
                # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                "right": right,
                # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                "density": density,
                # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                "sign": "left minus right",
            # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
            }
            # Compare one measured quantity at a time, keeping the scenarios paired.
            # for takes each item from ['crashed', 'mean_speed_m_s', 'distance_m', 'return'] in turn; metric holds the current item.
            for metric in ["crashed", "mean_speed_m_s", "distance_m", "return"]:
                # = unpacks the right-hand result into (seeds, differences), matching values to names in order.
                seeds, differences = paired_matrix(
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    groups[left], groups[right], density, metric
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                )
                # = stores a result in result[metric]; call crossed_bootstrap using parentheses to supply its inputs.
                result[metric] = crossed_bootstrap(differences)
                # = stores a result in result[metric]['training_seed_ids']; evaluate the expression on the right before using its result.
                result[metric]["training_seed_ids"] = seeds
            # Call comparisons.append using parentheses to supply its inputs; this line does not store a separate return value.
            comparisons.append(result)
    # = stores a result in output; build a dictionary, which stores values under named keys.
    output = {
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "status": "Development experiment; final confirmation scenarios remain reserved",
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "groups": {label: summarize_group(values) for label, values in groups.items()},
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "paired_comparisons": comparisons,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "uncertainty_note": "Equal-weight training-seed means and seed ranges. Paired differences use an exploratory two-way bootstrap over the three training seeds and the shared scenario seeds. Three training seeds provide limited evidence of robustness.",
    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
    }
    # Call save_json using parentheses to supply its inputs; this line does not store a separate return value.
    save_json(root / "controls" / "summary.json", output)
    # return ends this function and sends its result back to the code that called it.
    return output


# def defines main; parentheses list its inputs, and the colon starts its indented body.
def main():
    """Read Terminal options, check settings, and run the requested workflow.
    Existing result files are checked before reuse so a rerun cannot silently mix
    different experiments. The main guard below prevents running jobs on import.
    """
    # = stores a result in parser; call argparse.ArgumentParser using parentheses to supply its inputs.
    parser = argparse.ArgumentParser(description=__doc__)
    # Call parser.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    parser.add_argument("--results", default="results")
    # = stores a result in args; call parser.parse_args using parentheses to supply its inputs.
    args = parser.parse_args()
    # = stores a result in result; call analyze using parentheses to supply its inputs.
    result = analyze(Path(args.results))
    # for takes each item from result['groups'].items() in turn; (label, rows) holds the current item.
    for label, rows in result["groups"].items():
        # Call print using parentheses to supply its inputs; this line does not store a separate return value.
        print(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            label,
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            [
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                (
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    row["density"],
                    # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                    round(row["crash_rate"], 3),
                    # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                    round(row["mean_speed_m_s"], 2),
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                )
                # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                for row in rows
            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
            ],
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )


# Run the entry point only when this file is executed, not when it is imported.
# __name__ equals "__main__" when this file runs directly; this guard avoids running the script merely because another file imports it.
if __name__ == "__main__":
    # Call main using parentheses to supply its inputs; this line does not store a separate return value.
    main()


# def defines choose_speed_matched_control; parentheses list its inputs, and the colon starts its indented body.
def choose_speed_matched_control(candidates, target_speed):
    """Choose the speed threshold whose observed speed is closest to the target.
    If two thresholds are equally close, use the lower one. Collision results are
    deliberately excluded from this choice so we do not select a flattering outcome.

    Original implementation notes:
    Use speed closeness only; tie-break by lower cap, never collision rate.
    """
    # if checks not candidates or not np.isfinite(target_speed) or target_speed <= 0; only a true condition runs the indented block.
    if not candidates or not np.isfinite(target_speed) or target_speed <= 0:
        # raise signals an error and stops this normal path, so invalid data are not silently accepted.
        raise ValueError("Require candidates and a finite positive target speed")
    # if checks any((not np.isfinite(r['speed_m_s']) or not np.isfinite(r['cap_m_s']) or r['cap_m_s'] <= 0 for r ...; only a true condition runs the indented block.
    if any(
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        not np.isfinite(r["speed_m_s"])
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        or not np.isfinite(r["cap_m_s"])
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        or r["cap_m_s"] <= 0
        # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
        for r in candidates
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    ):
        # raise signals an error and stops this normal path, so invalid data are not silently accepted.
        raise ValueError("Require finite speeds and positive finite caps")
    # return ends this function and sends its result back to the code that called it.
    return min(
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        candidates, key=lambda r: (abs(r["speed_m_s"] - target_speed), r["cap_m_s"])
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )


# def defines scenario_bootstrap_fixed_models; parentheses list its inputs, and the colon starts its indented body.
def scenario_bootstrap_fixed_models(values, replicates=10000, seed=20261111):
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
    # = stores a result in matrix; call np.asarray using parentheses to supply its inputs.
    matrix = np.asarray(values, dtype=float)
    # if checks matrix.ndim != 2 or not matrix.size or (not np.isfinite(matrix).all()); only a true condition runs the indented block.
    if matrix.ndim != 2 or not matrix.size or not np.isfinite(matrix).all():
        # raise signals an error and stops this normal path, so invalid data are not silently accepted.
        raise ValueError("Expected a finite nonempty model-by-scenario matrix")
    # if checks not isinstance(replicates, int) or replicates < 1; only a true condition runs the indented block.
    if not isinstance(replicates, int) or replicates < 1:
        # raise signals an error and stops this normal path, so invalid data are not silently accepted.
        raise ValueError("Replicates must be a positive integer")
    # For fixed-model inference, each scenario contributes one average across the locked model pairs.
    # = stores a result in by_scenario; call matrix.mean using parentheses to supply its inputs.
    by_scenario = matrix.mean(axis=0)
    # = stores a result in rng; call np.random.default_rng using parentheses to supply its inputs.
    rng = np.random.default_rng(seed)
    # Resample whole scenarios, not individual model/scenario cells.
    # = stores a result in indices; call rng.integers using parentheses to supply its inputs.
    indices = rng.integers(len(by_scenario), size=(replicates, len(by_scenario)))
    # = stores a result in samples; call by_scenario[indices].mean using parentheses to supply its inputs.
    samples = by_scenario[indices].mean(axis=1)
    # The middle 95% of the resampled estimates forms an approximate uncertainty interval.
    # = unpacks the right-hand result into (low, high), matching values to names in order.
    low, high = np.quantile(samples, [0.025, 0.975])
    # return ends this function and sends its result back to the code that called it.
    return {
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "mean_difference": float(matrix.mean()),
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "conditional_95_percent_interval": [float(low), float(high)],
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "per_training_seed_differences": matrix.mean(axis=1).tolist(),
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "fixed_model_pairs": matrix.shape[0],
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "scenario_clusters": matrix.shape[1],
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "bootstrap_replicates": replicates,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "bootstrap_rng_seed": seed,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "scope": "Fresh scenarios conditional on the locked trained models; no new-training generalization.",
    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
    }
