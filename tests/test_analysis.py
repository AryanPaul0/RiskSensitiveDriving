import pytest
import numpy as np

from driving_rl.analysis import (choose_speed_targeted_baseline, crossed_bootstrap,
                                 paired_matrix, summarize_group)


def evaluation(seed, cases):
    return {"provenance": {"training_seed": seed},
            "episodes": [{"seed": scenario, "density": 1., "crashed": crashed}
                         for scenario, crashed in cases]}


def test_pairing_uses_training_and_scenario_keys_independent_of_order():
    left = [evaluation(1, [(11, 1), (10, 0)]), evaluation(0, [(10, 1), (11, 0)])]
    right = [evaluation(0, [(11, 1), (10, 0)]), evaluation(1, [(10, 1), (11, 0)])]
    seeds, matrix = paired_matrix(left, right, 1., "crashed")
    assert seeds == [0, 1]
    np.testing.assert_array_equal(matrix, [[1., -1.], [-1., 1.]])


def test_pairing_rejects_missing_or_duplicate_scenarios():
    first = [evaluation(0, [(10, 1), (11, 0)])]
    with pytest.raises(ValueError, match="identical"):
        paired_matrix(first, [evaluation(0, [(10, 1)])], 1., "crashed")
    with pytest.raises(ValueError, match="Duplicate"):
        paired_matrix(first + first, first, 1., "crashed")


def test_bootstrap_keeps_training_seed_uncertainty_with_many_episodes():
    # Within each trained model all episodes agree, but models disagree.
    # Treating 200 episodes as independent training replications is misleading.
    matrix = np.array([[0.] * 100, [1.] * 100])
    output = crossed_bootstrap(matrix, replicates=1000)
    assert output["mean_difference"] == .5
    assert output["exploratory_95_percent_interval"] == [0., 1.]
    assert output["training_seeds"] == 2


def test_bootstrap_is_reproducible_and_retains_pairing():
    matrix = np.full((3, 30), -.25)
    first = crossed_bootstrap(matrix, replicates=100)
    second = crossed_bootstrap(matrix, replicates=100)
    assert first == second
    assert first["exploratory_95_percent_interval"] == [-.25, -.25]


def test_baseline_selection_uses_only_speed_and_predeclared_tie_rule():
    candidates = [{"collision_penalty": -8, "speed_m_s": 22., "crash_rate": 0.},
                  {"collision_penalty": -4, "speed_m_s": 24., "crash_rate": 1.}]
    assert choose_speed_targeted_baseline(candidates, 23.)["collision_penalty"] == -4
    assert choose_speed_targeted_baseline(candidates, 22.1)["collision_penalty"] == -8
