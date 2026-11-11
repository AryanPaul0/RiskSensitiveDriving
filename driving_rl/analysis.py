"""Seed-aware summaries and paired comparisons for the fixed control experiments design."""
import argparse
import json
from pathlib import Path

import numpy as np

from .experiment import save_json


def paired_matrix(left, right, density, metric):
    """Align by training seed AND scenario seed; reject missing or duplicate data."""
    def index(evaluations):
        indexed = {}
        for data in evaluations:
            training_seed = data["provenance"]["training_seed"]
            for row in data["episodes"]:
                if row["density"] != density:
                    continue
                key = (training_seed, row["seed"])
                if key in indexed:
                    raise ValueError(f"Duplicate paired scenario: {key}")
                indexed[key] = float(row[metric])
        return indexed
    first, second = index(left), index(right)
    if not first or set(first) != set(second):
        raise ValueError("Paired comparison requires identical training/scenario keys")
    seeds = sorted({key[0] for key in first})
    scenarios = sorted({key[1] for key in first})
    if len(first) != len(seeds) * len(scenarios):
        raise ValueError("Paired data must form a complete training-seed by scenario matrix")
    return seeds, np.array([[first[seed, scenario] - second[seed, scenario]
                             for scenario in scenarios] for seed in seeds])


def crossed_bootstrap(differences, replicates=5000, seed=2026):
    """Resample the two crossed factors independently; retain policy pairing.

    Rows are independently trained models, columns are shared scenario seeds.
    This avoids pretending that repeated scenarios are independent training runs.
    With only three training seeds the resulting interval remains exploratory.
    """
    values = np.asarray(differences, dtype=float)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError("Expected a nonempty finite 2D difference matrix")
    rng = np.random.default_rng(seed)
    rows = rng.integers(values.shape[0], size=(replicates, values.shape[0]))
    columns = rng.integers(values.shape[1], size=(replicates, values.shape[1]))
    samples = values[rows[:, :, None], columns[:, None, :]].mean(axis=(1, 2))
    low, high = np.quantile(samples, [.025, .975])
    return {"mean_difference": float(values.mean()),
            "exploratory_95_percent_interval": [float(low), float(high)],
            "per_training_seed_differences": values.mean(axis=1).tolist(),
            "training_seeds": values.shape[0], "scenarios_per_seed": values.shape[1],
            "bootstrap_replicates": replicates, "bootstrap_rng_seed": seed}


def choose_speed_targeted_baseline(candidates, target_speed):
    """Choose by speed closeness only; ties prefer the smaller penalty magnitude."""
    if not candidates:
        raise ValueError("No baseline candidates")
    return min(candidates, key=lambda item: (abs(item["speed_m_s"] - target_speed),
                                            abs(item["collision_penalty"])))


def summarize_group(evaluations):
    if not evaluations:
        return []
    densities = sorted({row["density"] for row in evaluations[0]["summary"]})
    output = []
    for density in densities:
        seed_rows = []
        for data in evaluations:
            rows = [row for row in data["episodes"] if row["density"] == density]
            if not rows:
                raise ValueError("Each trained model needs every requested density")
            returns = np.array([row["return"] for row in rows])
            tail_count = max(1, int(np.ceil(.1 * len(rows))))
            distance = sum(row["distance_m"] for row in rows)
            seed_rows.append({"training_seed": data["provenance"]["training_seed"],
                              "episodes": len(rows),
                              "crash_rate": float(np.mean([row["crashed"] for row in rows])),
                              "mean_speed_m_s": float(np.mean([row["mean_speed_m_s"] for row in rows])),
                              "mean_return": float(returns.mean()),
                              "worst_10_percent_mean_return": float(np.sort(returns)[:tail_count].mean()),
                              "mean_distance_m": float(distance / len(rows)),
                              "mean_duration_s": float(np.mean([row["length"] for row in rows])),
                              "collisions_per_1000m": sum(row["crashed"] for row in rows) / distance * 1000 if distance else None})
        aggregate = {"density": density, "training_seeds": len(seed_rows),
                     "total_evaluation_episodes": sum(row["episodes"] for row in seed_rows),
                     "per_seed": seed_rows}
        for metric in ["crash_rate", "mean_speed_m_s", "mean_return",
                       "worst_10_percent_mean_return", "mean_distance_m", "mean_duration_s"]:
            values = [row[metric] for row in seed_rows]
            aggregate[metric] = float(np.mean(values))
            aggregate[metric + "_seed_range"] = [float(min(values)), float(max(values))]
        output.append(aggregate)
    return output


def load_main_groups(root, seeds=(0, 1, 2)):
    groups = {}
    definitions = {
        "DQN": "dqn_seed{seed}_mean",
        "QR-DQN mean": "qrdqn_seed{seed}_mean",
        "QR-DQN lower 25%": "qrdqn_seed{seed}_tail25",
    }
    for label, pattern in definitions.items():
        groups[label] = [json.loads((root / f"eval_{pattern.format(seed=seed)}.json").read_text())
                         for seed in seeds]
    selection_path = root / "controls" / "baseline_selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
        penalty = selection["selected_collision_penalty"]
        if penalty == -2:
            groups["Reward-tuned DQN"] = groups["DQN"]
        else:
            groups["Reward-tuned DQN"] = [json.loads((root / "controls" / f"eval_dqn_penalty{abs(int(penalty))}_seed{seed}.json").read_text()) for seed in seeds]
    governor_path = root / "controls" / "governor_selection.json"
    if governor_path.exists():
        cap = json.loads(governor_path.read_text())["selected_cap_m_s"]
        groups["DQN + speed governor"] = [json.loads((root / "controls" / f"eval_dqn_cap{cap}_seed{seed}.json").read_text()) for seed in seeds]
    # All physical comparisons use the same evaluation task and reward, even
    # when the training reward was changed for the scalar baseline.
    for label, evaluations in groups.items():
        observed_seeds = [item["provenance"]["training_seed"] for item in evaluations]
        if sorted(observed_seeds) != sorted(seeds):
            raise ValueError(f"Unexpected or duplicate training seeds in {label}")
        for item in evaluations:
            provenance = item["provenance"]
            if (provenance["training_steps"] != 20000 or provenance["seed_start"] != 10000
                    or provenance["episodes_per_density"] != 30
                    or provenance.get("evaluation_collision_penalty", -2.) != -2.):
                raise ValueError(f"Evaluation does not match fixed control experiments design: {label}")
            expected_alpha = .25 if label == "QR-DQN lower 25%" else 1.
            expected_algorithm = "qrdqn" if label.startswith("QR-DQN") else "dqn"
            if provenance["alpha"] != expected_alpha or provenance["algorithm"] != expected_algorithm:
                raise ValueError(f"Incorrect policy identity in {label}")
            expected_cap = (json.loads(governor_path.read_text())["selected_cap_m_s"]
                            if label == "DQN + speed governor" else None)
            if provenance.get("speed_governor_cap_m_s") != expected_cap:
                raise ValueError(f"Incorrect governor setting in {label}")
    return groups


def analyze(root):
    groups = load_main_groups(root)
    comparisons = []
    contrasts = [("QR-DQN lower 25%", "QR-DQN mean")]
    for control in ["Reward-tuned DQN", "DQN + speed governor"]:
        if control in groups:
            contrasts.append(("QR-DQN lower 25%", control))
    for left, right in contrasts:
        for density in [1., 1.5, 2.]:
            result = {"left": left, "right": right, "density": density,
                      "sign": "left minus right"}
            for metric in ["crashed", "mean_speed_m_s", "distance_m", "return"]:
                seeds, differences = paired_matrix(groups[left], groups[right], density, metric)
                result[metric] = crossed_bootstrap(differences)
                result[metric]["training_seed_ids"] = seeds
            comparisons.append(result)
    output = {"status": "Development experiment; final confirmation scenarios remain reserved",
              "groups": {label: summarize_group(values) for label, values in groups.items()},
              "paired_comparisons": comparisons,
              "uncertainty_note": "Equal-weight training-seed means and seed ranges. Paired differences use an exploratory two-way bootstrap over the three training seeds and the shared scenario seeds. Three training seeds provide limited evidence of robustness."}
    save_json(root / "controls" / "summary.json", output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    result = analyze(Path(args.results))
    for label, rows in result["groups"].items():
        print(label, [(row["density"], round(row["crash_rate"], 3),
                       round(row["mean_speed_m_s"], 2)) for row in rows])


if __name__ == "__main__":
    main()


def choose_speed_matched_control(candidates, target_speed):
    """Use speed closeness only; tie-break by lower cap, never collision rate."""
    if not candidates or not np.isfinite(target_speed) or target_speed <= 0:
        raise ValueError('Require candidates and a finite positive target speed')
    if any(not np.isfinite(r['speed_m_s']) or not np.isfinite(r['cap_m_s'])
           or r['cap_m_s'] <= 0 for r in candidates):
        raise ValueError('Require finite speeds and positive finite caps')
    return min(candidates, key=lambda r:(abs(r['speed_m_s']-target_speed),r['cap_m_s']))
