"""Tests for analysis.

Tests use deliberately small examples so mistakes in learning, data pairing or
evaluation show up before spending time on long training runs.
"""

import pytest  # Make pytest available for this file's calculations and workflow.
import numpy as np  # Make numpy available for this file's calculations and workflow.

from driving_rl.analysis import (  # Make choose_speed_targeted_baseline, crossed_bootstrap, paired_matrix, summarize_group available for this file's calculations and workflow.
    choose_speed_targeted_baseline,  # Continue the calculation or input list started above.
    crossed_bootstrap,  # Continue the calculation or input list started above.
    paired_matrix,  # Continue the calculation or input list started above.
    summarize_group,  # Continue the calculation or input list started above.
)  # Finish assembling the grouped values above.


def evaluation(seed, cases):  # Build a small controlled example for the tests below.
    """Build a small controlled example for the tests below."""  # Describe this operation for readers and help().
    return {  # Return the named settings or measurements assembled below.
        "provenance": {"training_seed": seed},  # Record 'provenance' as the named settings or measurements assembled below.
        "episodes": [  # Record 'episodes' as the named settings or measurements assembled below for each the starting-traffic seed, crashed in cases.
            {"seed": scenario, "density": 1.0, "crashed": crashed}  # Record 'seed' as the starting-traffic seed.
            for scenario, crashed in cases  # Include the starting-traffic seed, crashed, cases in the surrounding operation.
        ],  # Finish assembling the grouped values above.
    }  # Finish assembling the grouped values above.


def test_pairing_uses_training_and_scenario_keys_independent_of_order():  # Check that pairing uses training and scenario keys independent of order.
    """Check that pairing uses training and scenario keys independent of order.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    left = [evaluation(1, [(11, 1), (10, 0)]), evaluation(0, [(10, 1), (11, 0)])]  # Set left to evaluation applied to 1, 11, 1, 10, 0, evaluation applied to 0, 10, 1, 11, 0.
    right = [evaluation(0, [(11, 1), (10, 0)]), evaluation(1, [(10, 1), (11, 0)])]  # Set right to evaluation applied to 0, 11, 1, 10, 0, evaluation applied to 1, 10, 1, 11, 0.
    seeds, matrix = paired_matrix(left, right, 1.0, "crashed")  # Set seeds, matrix to paired matrix applied to left, right, 1.0, 'crashed'.
    assert seeds == [0, 1]  # Check that seeds equals 0, 1; fail this test if it is not true.
    np.testing.assert_array_equal(matrix, [[1.0, -1.0], [-1.0, 1.0]])  # Np.testing.assert array equal applied to matrix, 1.0, negative 1.0, negative 1.0, 1.0.


def test_pairing_rejects_missing_or_duplicate_scenarios():  # Check that pairing rejects missing or duplicate scenarios.
    """Check that pairing rejects missing or duplicate scenarios.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    first = [evaluation(0, [(10, 1), (11, 0)])]  # Set first to evaluation applied to 0, 10, 1, 11, 0.
    with pytest.raises(ValueError, match="identical"):  # An invalid input is supposed to fail here; accepting it would hide a bug.
        paired_matrix(first, [evaluation(0, [(10, 1)])], 1.0, "crashed")  # Paired matrix applied to first, evaluation applied to 0, (10, 1), 1.0, 'crashed'.
    with pytest.raises(ValueError, match="Duplicate"):  # An invalid input is supposed to fail here; accepting it would hide a bug.
        paired_matrix(first + first, first, 1.0, "crashed")  # Paired matrix applied to (first plus first), first, 1.0, 'crashed'.


def test_bootstrap_keeps_training_seed_uncertainty_with_many_episodes():  # Check that bootstrap keeps training seed uncertainty with many episodes. Within each trained model all episodes agree, but models disagree. Treating 200 episodes as independent training replications is misleading.
    """Check that bootstrap keeps training seed uncertainty with many episodes.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    matrix = np.array([[0.0] * 100, [1.0] * 100])  # Set matrix to (0.0 times 100), (1.0 times 100) as a numerical array/tensor for batch calculations.
    output = crossed_bootstrap(matrix, replicates=1000)  # Set output to crossed bootstrap applied to matrix.
    assert output["mean_difference"] == 0.5  # Check that the entries of output selected by 'mean_difference' equals 0.5; fail this test if it is not true.
    assert output["exploratory_95_percent_interval"] == [0.0, 1.0]  # Check that the entries of output selected by 'exploratory_95_percent_interval' equals 0.0, 1.0; fail this test if it is not true.
    assert output["training_seeds"] == 2  # Check that the entries of output selected by 'training_seeds' equals 2; fail this test if it is not true.


def test_bootstrap_is_reproducible_and_retains_pairing():  # Check that bootstrap is reproducible and retains pairing.
    """Check that bootstrap is reproducible and retains pairing.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    matrix = np.full((3, 30), -0.25)  # Set matrix to np.full applied to 3, 30, negative 0.25.
    first = crossed_bootstrap(matrix, replicates=100)  # Set first to crossed bootstrap applied to matrix.
    second = crossed_bootstrap(matrix, replicates=100)  # Set second to crossed bootstrap applied to matrix.
    assert first == second  # Check that first equals second; fail this test if it is not true.
    assert first["exploratory_95_percent_interval"] == [-0.25, -0.25]  # Check that the entries of first selected by 'exploratory_95_percent_interval' equals negative 0.25, negative 0.25; fail this test if it is not true.


def test_baseline_selection_uses_only_speed_and_predeclared_tie_rule():  # Check that baseline selection uses only speed and predeclared tie rule.
    """Check that baseline selection uses only speed and predeclared tie rule.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    candidates = [  # Set candidates to the named settings or measurements assembled below, the named settings or measurements assembled below.
        {"collision_penalty": -8, "speed_m_s": 22.0, "crash_rate": 0.0},  # Record 'collision_penalty' as negative 8.
        {"collision_penalty": -4, "speed_m_s": 24.0, "crash_rate": 1.0},  # Record 'collision_penalty' as negative 4.
    ]  # Finish assembling the grouped values above.
    assert choose_speed_targeted_baseline(candidates, 23.0)["collision_penalty"] == -4  # Check that the entries of choose speed targeted baseline applied to candidates, 23.0 selected by 'collision_penalty' equals negative 4; fail this test if it is not true.
    assert choose_speed_targeted_baseline(candidates, 22.1)["collision_penalty"] == -8  # Check that the entries of choose speed targeted baseline applied to candidates, 22.1 selected by 'collision_penalty' equals negative 8; fail this test if it is not true.


def test_speed_matching_never_uses_collision_rate():  # Check that speed matching never uses collision rate.
    """Check that speed matching never uses collision rate.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    from driving_rl.analysis import choose_speed_matched_control  # Make choose_speed_matched_control available for this file's calculations and workflow.

    candidates = [  # Set candidates to the named settings or measurements assembled below, the named settings or measurements assembled below.
        {"cap_m_s": 23.0, "speed_m_s": 22.5, "crash_rate": 1.0},  # Record 'cap_m_s' as 23.0.
        {"cap_m_s": 22.0, "speed_m_s": 21.0, "crash_rate": 0.0},  # Record 'cap_m_s' as 22.0.
    ]  # Finish assembling the grouped values above.
    assert choose_speed_matched_control(candidates, 22.5) is candidates[0]  # Check that choose speed matched control applied to candidates, 22.5 is the entries of candidates selected by 0; fail this test if it is not true.


def test_speed_matching_ties_prefer_lower_cap():  # Check that speed matching ties prefer lower cap.
    """Check that speed matching ties prefer lower cap.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    from driving_rl.analysis import choose_speed_matched_control  # Make choose_speed_matched_control available for this file's calculations and workflow.

    candidates = [  # Set candidates to the named settings or measurements assembled below, the named settings or measurements assembled below.
        {"cap_m_s": 24.0, "speed_m_s": 23.0},  # Record 'cap_m_s' as 24.0.
        {"cap_m_s": 22.0, "speed_m_s": 21.0},  # Record 'cap_m_s' as 22.0.
    ]  # Finish assembling the grouped values above.
    assert choose_speed_matched_control(candidates, 22.0)["cap_m_s"] == 22.0  # Check that the entries of choose speed matched control applied to candidates, 22.0 selected by 'cap_m_s' equals 22.0; fail this test if it is not true.


def test_speed_matching_rejects_missing_and_nonfinite_values():  # Check that speed matching rejects missing and nonfinite values.
    """Check that speed matching rejects missing and nonfinite values.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    from driving_rl.analysis import choose_speed_matched_control  # Make choose_speed_matched_control available for this file's calculations and workflow.

    with pytest.raises(ValueError):  # An invalid input is supposed to fail here; accepting it would hide a bug.
        choose_speed_matched_control([], 22.0)  # Choose speed matched control applied to , 22.0.
    with pytest.raises(ValueError):  # An invalid input is supposed to fail here; accepting it would hide a bug.
        choose_speed_matched_control(  # Choose speed matched control applied to the named settings or measurements assembled below, 22.0.
            [{"cap_m_s": 23.0, "speed_m_s": float("nan")}], 22.0  # Record 'cap_m_s' as 23.0.
        )  # Finish assembling choose speed matched control inputs.


def test_scenario_bootstrap_retains_fixed_model_pairing():  # Check that scenario bootstrap retains fixed model pairing.
    """Check that scenario bootstrap retains fixed model pairing.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    from driving_rl.analysis import scenario_bootstrap_fixed_models  # Make scenario_bootstrap_fixed_models available for this file's calculations and workflow.

    r = scenario_bootstrap_fixed_models(  # Every scenario averages to zero across the two fixed models. Flattening model/scenario observations would incorrectly introduce sampling variance.
        [[1.0, -1.0, 1.0, -1.0], [-1.0, 1.0, -1.0, 1.0]], replicates=1000  # Include 1.0, negative 1.0, 1.0, negative 1.0, negative 1.0, 1.0, negative 1.0, 1.0, 1000 in the surrounding operation.
    )  # Finish assembling scenario bootstrap fixed models inputs.
    assert r["conditional_95_percent_interval"] == [0.0, 0.0]  # Check that the entries of r selected by 'conditional_95_percent_interval' equals 0.0, 0.0; fail this test if it is not true.
    assert r["scenario_clusters"] == 4 and r["fixed_model_pairs"] == 2  # Check that the entries of r selected by 'scenario_clusters' equals 4 and the entries of r selected by 'fixed_model_pairs' equals 2; fail this test if it is not true.


def test_scenario_bootstrap_constant_contrast_is_exact():  # Check that scenario bootstrap constant contrast is exact.
    """Check that scenario bootstrap constant contrast is exact.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    from driving_rl.analysis import scenario_bootstrap_fixed_models  # Make scenario_bootstrap_fixed_models available for this file's calculations and workflow.

    r = scenario_bootstrap_fixed_models(np.full((3, 100), -0.25), replicates=1000)  # Set r to scenario bootstrap fixed models applied to np.full applied to 3, 100, negative 0.25.
    assert r["mean_difference"] == -0.25  # Check that the entries of r selected by 'mean_difference' equals negative 0.25; fail this test if it is not true.
    assert r["conditional_95_percent_interval"] == [-0.25, -0.25]  # Check that the entries of r selected by 'conditional_95_percent_interval' equals negative 0.25, negative 0.25; fail this test if it is not true.


def test_scenario_bootstrap_reproducible_without_changing_global_rng():  # Check that scenario bootstrap reproducible without changing global rng.
    """Check that scenario bootstrap reproducible without changing global rng.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    from driving_rl.analysis import scenario_bootstrap_fixed_models  # Make scenario_bootstrap_fixed_models available for this file's calculations and workflow.

    matrix = np.array([[0.0, -1.0, 0.0, 1.0], [1.0, 0.0, -1.0, 0.0]])  # Set matrix to 0.0, negative 1.0, 0.0, 1.0, 1.0, 0.0, negative 1.0, 0.0 as a numerical array/tensor for batch calculations.
    a = scenario_bootstrap_fixed_models(matrix, replicates=1000, seed=72)  # Set a to scenario bootstrap fixed models applied to matrix.
    np.random.seed(91)  # Np.random.seed applied to 91.
    b = scenario_bootstrap_fixed_models(matrix, replicates=1000, seed=72)  # Set b to scenario bootstrap fixed models applied to matrix.
    assert a == b  # Check that a equals b; fail this test if it is not true.
    assert np.random.random() == np.random.RandomState(91).random_sample()  # Check that the result of np.random.random equals the result of np.random.RandomState(91).random sample; fail this test if it is not true.


def test_scenario_bootstrap_rejects_invalid_input():  # Check that scenario bootstrap rejects invalid input.
    """Check that scenario bootstrap rejects invalid input.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    from driving_rl.analysis import scenario_bootstrap_fixed_models  # Make scenario_bootstrap_fixed_models available for this file's calculations and workflow.

    for value in [[], [1.0, 2.0], [[np.nan]], [[np.inf]]]:  # Process each value from , 1.0, 2.0, np.nan, np.inf.
        with pytest.raises(ValueError):  # An invalid input is supposed to fail here; accepting it would hide a bug.
            scenario_bootstrap_fixed_models(value)  # Scenario bootstrap fixed models applied to value.
    with pytest.raises(ValueError):  # An invalid input is supposed to fail here; accepting it would hide a bug.
        scenario_bootstrap_fixed_models([[0.0]], replicates=0)  # Scenario bootstrap fixed models applied to 0.0.
