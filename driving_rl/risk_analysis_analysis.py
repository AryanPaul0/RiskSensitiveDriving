"""Combine the risk-selector experiments and their prediction checks.

The same saved QR-DQN networks are reused with different action-selection rules.
This holds the network fixed while we compare decisions, but it does not make
the rules mathematically optimal or guarantee safety.

Original implementation notes:
Same-checkpoint comparisons, exploratory risk frontier and calibration.
"""

import json
from pathlib import Path

import numpy as np

from .analysis import crossed_bootstrap, paired_matrix, summarize_group
from .calibration import summarize_rows
from .experiment import save_json


def analyze(root):
    # Importing here avoids a runner/analysis import cycle at module startup.
    """Combine the saved experiments into comparison tables. Match scenarios and
    model seeds before calculating differences, and keep development limitations visible.
    """
    from run_risk_analysis import ALPHAS, eval_path

    root = Path(root)
    stage = root / "risk_analysis"
    selection = json.loads((stage / "governor_selection.json").read_text())
    cap = selection["selected_cap_m_s"]
    load = lambda path: json.loads(path.read_text())
    definitions = {
        "QR-DQN mean": (1.0, None),
        "QR-DQN lower 25%": (0.25, None),
        "Same QR-DQN + governor": (1.0, cap),
    }
    groups = {
        name: [load(eval_path(seed, alpha, speed_cap)) for seed in [0, 1, 2]]
        for name, (alpha, speed_cap) in definitions.items()
    }
    paired = []
    for density in [1.0, 1.5, 2.0]:
        output = {
            "density": density,
            "left": "QR-DQN lower 25%",
            "right": "Same QR-DQN + governor",
            "sign": "left minus right",
        }
        # Compare one measured quantity at a time, keeping the scenarios paired.
        for metric in ["crashed", "mean_speed_m_s", "distance_m", "return"]:
            seeds, matrix = paired_matrix(
                groups[output["left"]], groups[output["right"]], density, metric
            )
            output[metric] = crossed_bootstrap(matrix)
            output[metric]["training_seed_ids"] = seeds
        paired.append(output)
    frontier = [
        {
            "alpha": alpha,
            "summary": summarize_group(
                [load(eval_path(seed, alpha)) for seed in [0, 1, 2]]
            ),
        }
        for alpha in ALPHAS
    ]
    calibration = {}
    failures = {}
    for policy in ["mean", "tail25", "governed"]:
        datasets = [
            load(stage / f"calibration_{policy}_seed{seed}.json") for seed in [0, 1, 2]
        ]
        for seed, data in enumerate(datasets):
            p = data["provenance"]
            if (
                p["training_seed"] != seed
                or p["seed_start"] != 12000
                or p["episodes_per_density"] != 60
                or p["training_steps"] != 20000
                or p["alpha"] != (0.25 if policy == "tail25" else 1.0)
                or p["speed_cap"] != (cap if policy == "governed" else None)
            ):
                raise ValueError(
                    "Calibration provenance does not match risk analysis protocol"
                )
        calibration[policy], failures[policy] = [], []
        for density in [1.0, 1.5, 2.0]:
            rows = [
                row
                for data in datasets
                for row in data["episodes"]
                if row["density"] == density
            ]
            metrics = summarize_rows(rows)
            metrics["per_seed"] = [
                {
                    "training_seed": seed,
                    **summarize_rows(
                        [row for row in data["episodes"] if row["density"] == density]
                    ),
                }
                for seed, data in enumerate(datasets)
            ]
            calibration[policy].append({"density": density, **metrics})
            crashed = [row for row in rows if row["crashed"]]
            proxy_counts = {
                name: sum(
                    row["terminal_nearest_neighbor"]["position_proxy"] == name
                    for row in crashed
                    if row["terminal_nearest_neighbor"]
                )
                for name in ["front", "rear", "side"]
            }
            failures[policy].append(
                {
                    "density": density,
                    "episodes": len(rows),
                    "crashes": len(crashed),
                    "early_crashes_within_3_seconds": sum(
                        row["length"] <= 3 for row in crashed
                    ),
                    "recent_lateral_request_given_crash": metrics[
                        "recent_lateral_request_given_crash"
                    ],
                    "recent_lane_transition_given_crash": metrics[
                        "recent_lane_transition_given_crash"
                    ],
                    "terminal_nearest_neighbor_proxy_counts": proxy_counts,
                    "all_observation_slots_filled_near_crash_fraction": (
                        float(
                            np.mean(
                                [
                                    row["recent_states"][-1]["observed_vehicle_slots"]
                                    == 8
                                    for row in crashed
                                ]
                            )
                        )
                        if crashed
                        else None
                    ),
                }
            )
    result = {
        "status": "Exploratory development; final confirmation seeds remain unused",
        "selection": selection,
        "groups": {name: summarize_group(data) for name, data in groups.items()},
        "paired_comparisons": paired,
        "risk_frontier": frontier,
        "calibration": calibration,
        "failure_diagnostics": failures,
        "interpretation": {
            "paired_intervals": "Exploratory two-way bootstrap, conditional on development choices; only three training seeds.",
            "calibration": "Marginal empirical compatibility brackets account for censoring and ties, not sampling uncertainty. Initial-state samples; changed-policy continuations differ from training.",
            "failures": "Descriptive recent-action and geometry proxies, not identified crash causes or proof of unavoidable collisions.",
        },
    }
    save_json(stage / "summary.json", result)
    return result
