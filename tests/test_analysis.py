"""Tests for analysis.

Tests use deliberately small examples so mistakes in learning, data pairing or
evaluation show up before spending time on long training runs.
"""

import pytest
import numpy as np

from driving_rl.analysis import (
    choose_speed_targeted_baseline,
    crossed_bootstrap,
    paired_matrix,
    summarize_group,
)


def evaluation(seed, cases):
    """Build a small controlled example for the tests below."""
    return {
        "provenance": {"training_seed": seed},
        "episodes": [
            {"seed": scenario, "density": 1.0, "crashed": crashed}
            for scenario, crashed in cases
        ],
    }


def test_pairing_uses_training_and_scenario_keys_independent_of_order():
    """Check that pairing uses training and scenario keys independent of order.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    left = [evaluation(1, [(11, 1), (10, 0)]), evaluation(0, [(10, 1), (11, 0)])]
    right = [evaluation(0, [(11, 1), (10, 0)]), evaluation(1, [(10, 1), (11, 0)])]
    seeds, matrix = paired_matrix(left, right, 1.0, "crashed")
    assert seeds == [0, 1]
    np.testing.assert_array_equal(matrix, [[1.0, -1.0], [-1.0, 1.0]])


def test_pairing_rejects_missing_or_duplicate_scenarios():
    """Check that pairing rejects missing or duplicate scenarios.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    first = [evaluation(0, [(10, 1), (11, 0)])]
    # An invalid input is supposed to fail here; accepting it would hide a bug.
    with pytest.raises(ValueError, match="identical"):
        paired_matrix(first, [evaluation(0, [(10, 1)])], 1.0, "crashed")
    # An invalid input is supposed to fail here; accepting it would hide a bug.
    with pytest.raises(ValueError, match="Duplicate"):
        paired_matrix(first + first, first, 1.0, "crashed")


def test_bootstrap_keeps_training_seed_uncertainty_with_many_episodes():
    # Within each trained model all episodes agree, but models disagree.
    # Treating 200 episodes as independent training replications is misleading.
    """Check that bootstrap keeps training seed uncertainty with many episodes.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    matrix = np.array([[0.0] * 100, [1.0] * 100])
    output = crossed_bootstrap(matrix, replicates=1000)
    assert output["mean_difference"] == 0.5
    assert output["exploratory_95_percent_interval"] == [0.0, 1.0]
    assert output["training_seeds"] == 2


def test_bootstrap_is_reproducible_and_retains_pairing():
    """Check that bootstrap is reproducible and retains pairing.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    matrix = np.full((3, 30), -0.25)
    first = crossed_bootstrap(matrix, replicates=100)
    second = crossed_bootstrap(matrix, replicates=100)
    assert first == second
    assert first["exploratory_95_percent_interval"] == [-0.25, -0.25]


def test_baseline_selection_uses_only_speed_and_predeclared_tie_rule():
    """Check that baseline selection uses only speed and predeclared tie rule.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    candidates = [
        {"collision_penalty": -8, "speed_m_s": 22.0, "crash_rate": 0.0},
        {"collision_penalty": -4, "speed_m_s": 24.0, "crash_rate": 1.0},
    ]
    assert choose_speed_targeted_baseline(candidates, 23.0)["collision_penalty"] == -4
    assert choose_speed_targeted_baseline(candidates, 22.1)["collision_penalty"] == -8


def test_speed_matching_never_uses_collision_rate():
    """Check that speed matching never uses collision rate.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    from driving_rl.analysis import choose_speed_matched_control

    candidates = [
        {"cap_m_s": 23.0, "speed_m_s": 22.5, "crash_rate": 1.0},
        {"cap_m_s": 22.0, "speed_m_s": 21.0, "crash_rate": 0.0},
    ]
    assert choose_speed_matched_control(candidates, 22.5) is candidates[0]


def test_speed_matching_ties_prefer_lower_cap():
    """Check that speed matching ties prefer lower cap.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    from driving_rl.analysis import choose_speed_matched_control

    candidates = [
        {"cap_m_s": 24.0, "speed_m_s": 23.0},
        {"cap_m_s": 22.0, "speed_m_s": 21.0},
    ]
    assert choose_speed_matched_control(candidates, 22.0)["cap_m_s"] == 22.0


def test_speed_matching_rejects_missing_and_nonfinite_values():
    """Check that speed matching rejects missing and nonfinite values.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    from driving_rl.analysis import choose_speed_matched_control

    # An invalid input is supposed to fail here; accepting it would hide a bug.
    with pytest.raises(ValueError):
        choose_speed_matched_control([], 22.0)
    # An invalid input is supposed to fail here; accepting it would hide a bug.
    with pytest.raises(ValueError):
        choose_speed_matched_control(
            [{"cap_m_s": 23.0, "speed_m_s": float("nan")}], 22.0
        )


def test_scenario_bootstrap_retains_fixed_model_pairing():
    """Check that scenario bootstrap retains fixed model pairing.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    from driving_rl.analysis import scenario_bootstrap_fixed_models

    # Every scenario averages to zero across the two fixed models. Flattening
    # model/scenario observations would incorrectly introduce sampling variance.
    r = scenario_bootstrap_fixed_models(
        [[1.0, -1.0, 1.0, -1.0], [-1.0, 1.0, -1.0, 1.0]], replicates=1000
    )
    assert r["conditional_95_percent_interval"] == [0.0, 0.0]
    assert r["scenario_clusters"] == 4 and r["fixed_model_pairs"] == 2


def test_scenario_bootstrap_constant_contrast_is_exact():
    """Check that scenario bootstrap constant contrast is exact.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    from driving_rl.analysis import scenario_bootstrap_fixed_models

    r = scenario_bootstrap_fixed_models(np.full((3, 100), -0.25), replicates=1000)
    assert r["mean_difference"] == -0.25
    assert r["conditional_95_percent_interval"] == [-0.25, -0.25]


def test_scenario_bootstrap_reproducible_without_changing_global_rng():
    """Check that scenario bootstrap reproducible without changing global rng.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    from driving_rl.analysis import scenario_bootstrap_fixed_models

    matrix = np.array([[0.0, -1.0, 0.0, 1.0], [1.0, 0.0, -1.0, 0.0]])
    a = scenario_bootstrap_fixed_models(matrix, replicates=1000, seed=72)
    np.random.seed(91)
    b = scenario_bootstrap_fixed_models(matrix, replicates=1000, seed=72)
    assert a == b
    assert np.random.random() == np.random.RandomState(91).random_sample()


def test_scenario_bootstrap_rejects_invalid_input():
    """Check that scenario bootstrap rejects invalid input.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    from driving_rl.analysis import scenario_bootstrap_fixed_models

    for value in [[], [1.0, 2.0], [[np.nan]], [[np.inf]]]:
        # An invalid input is supposed to fail here; accepting it would hide a bug.
        with pytest.raises(ValueError):
            scenario_bootstrap_fixed_models(value)
    # An invalid input is supposed to fail here; accepting it would hide a bug.
    with pytest.raises(ValueError):
        scenario_bootstrap_fixed_models([[0.0]], replicates=0)
