"""Tests for analysis.

Tests use deliberately small examples so mistakes in learning, data pairing or
evaluation show up before spending time on long training runs.
"""

# import makes these modules available; "as" gives a module a shorter local name.
import pytest
# import makes these modules available; "as" gives a module a shorter local name.
import numpy as np

# from driving_rl.analysis import brings the named tools into this file; parentheses allow several lines.
from driving_rl.analysis import (
    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
    choose_speed_targeted_baseline,
    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
    crossed_bootstrap,
    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
    paired_matrix,
    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
    summarize_group,
# ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
)


# def defines evaluation; parentheses list its inputs, and the colon starts its indented body.
def evaluation(seed, cases):
    # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
    """Build a small controlled example for the tests below."""
    # return ends this function and sends its result back to the code that called it.
    return {
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "provenance": {"training_seed": seed},
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "episodes": [
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            {"seed": scenario, "density": 1.0, "crashed": crashed}
            # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
            for scenario, crashed in cases
        # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
        ],
    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
    }


# def defines test_pairing_uses_training_and_scenario_keys_independent_of_order; parentheses list its inputs, and the colon starts its indented body.
def test_pairing_uses_training_and_scenario_keys_independent_of_order():
    """Check that pairing uses training and scenario keys independent of order.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    # = stores a result in left; build a list, keeping its items in order.
    left = [evaluation(1, [(11, 1), (10, 0)]), evaluation(0, [(10, 1), (11, 0)])]
    # = stores a result in right; build a list, keeping its items in order.
    right = [evaluation(0, [(11, 1), (10, 0)]), evaluation(1, [(10, 1), (11, 0)])]
    # = unpacks the right-hand result into (seeds, matrix), matching values to names in order.
    seeds, matrix = paired_matrix(left, right, 1.0, "crashed")
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert seeds == [0, 1]
    # Call np.testing.assert_array_equal using parentheses to supply its inputs; this line does not store a separate return value.
    np.testing.assert_array_equal(matrix, [[1.0, -1.0], [-1.0, 1.0]])


# def defines test_pairing_rejects_missing_or_duplicate_scenarios; parentheses list its inputs, and the colon starts its indented body.
def test_pairing_rejects_missing_or_duplicate_scenarios():
    """Check that pairing rejects missing or duplicate scenarios.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    # = stores a result in first; build a list, keeping its items in order.
    first = [evaluation(0, [(10, 1), (11, 0)])]
    # An invalid input is supposed to fail here; accepting it would hide a bug.
    # with enters a temporary context and exits it afterward, even if an error interrupts the block.
    with pytest.raises(ValueError, match="identical"):
        # Call paired_matrix using parentheses to supply its inputs; this line does not store a separate return value.
        paired_matrix(first, [evaluation(0, [(10, 1)])], 1.0, "crashed")
    # An invalid input is supposed to fail here; accepting it would hide a bug.
    # with enters a temporary context and exits it afterward, even if an error interrupts the block.
    with pytest.raises(ValueError, match="Duplicate"):
        # Call paired_matrix using parentheses to supply its inputs; this line does not store a separate return value.
        paired_matrix(first + first, first, 1.0, "crashed")


# def defines test_bootstrap_keeps_training_seed_uncertainty_with_many_episodes; parentheses list its inputs, and the colon starts its indented body.
def test_bootstrap_keeps_training_seed_uncertainty_with_many_episodes():
    # Within each trained model all episodes agree, but models disagree.
    # Treating 200 episodes as independent training replications is misleading.
    """Check that bootstrap keeps training seed uncertainty with many episodes.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    # = stores a result in matrix; call np.array using parentheses to supply its inputs.
    matrix = np.array([[0.0] * 100, [1.0] * 100])
    # = stores a result in output; call crossed_bootstrap using parentheses to supply its inputs.
    output = crossed_bootstrap(matrix, replicates=1000)
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert output["mean_difference"] == 0.5
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert output["exploratory_95_percent_interval"] == [0.0, 1.0]
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert output["training_seeds"] == 2


# def defines test_bootstrap_is_reproducible_and_retains_pairing; parentheses list its inputs, and the colon starts its indented body.
def test_bootstrap_is_reproducible_and_retains_pairing():
    """Check that bootstrap is reproducible and retains pairing.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    # = stores a result in matrix; call np.full using parentheses to supply its inputs.
    matrix = np.full((3, 30), -0.25)
    # = stores a result in first; call crossed_bootstrap using parentheses to supply its inputs.
    first = crossed_bootstrap(matrix, replicates=100)
    # = stores a result in second; call crossed_bootstrap using parentheses to supply its inputs.
    second = crossed_bootstrap(matrix, replicates=100)
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert first == second
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert first["exploratory_95_percent_interval"] == [-0.25, -0.25]


# def defines test_baseline_selection_uses_only_speed_and_predeclared_tie_rule; parentheses list its inputs, and the colon starts its indented body.
def test_baseline_selection_uses_only_speed_and_predeclared_tie_rule():
    """Check that baseline selection uses only speed and predeclared tie rule.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    # = stores a result in candidates; build a list, keeping its items in order.
    candidates = [
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        {"collision_penalty": -8, "speed_m_s": 22.0, "crash_rate": 0.0},
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        {"collision_penalty": -4, "speed_m_s": 24.0, "crash_rate": 1.0},
    # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
    ]
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert choose_speed_targeted_baseline(candidates, 23.0)["collision_penalty"] == -4
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert choose_speed_targeted_baseline(candidates, 22.1)["collision_penalty"] == -8


# def defines test_speed_matching_never_uses_collision_rate; parentheses list its inputs, and the colon starts its indented body.
def test_speed_matching_never_uses_collision_rate():
    """Check that speed matching never uses collision rate.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    # from driving_rl.analysis import brings the named tools into this file; parentheses allow several lines.
    from driving_rl.analysis import choose_speed_matched_control

    # = stores a result in candidates; build a list, keeping its items in order.
    candidates = [
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        {"cap_m_s": 23.0, "speed_m_s": 22.5, "crash_rate": 1.0},
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        {"cap_m_s": 22.0, "speed_m_s": 21.0, "crash_rate": 0.0},
    # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
    ]
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert choose_speed_matched_control(candidates, 22.5) is candidates[0]


# def defines test_speed_matching_ties_prefer_lower_cap; parentheses list its inputs, and the colon starts its indented body.
def test_speed_matching_ties_prefer_lower_cap():
    """Check that speed matching ties prefer lower cap.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    # from driving_rl.analysis import brings the named tools into this file; parentheses allow several lines.
    from driving_rl.analysis import choose_speed_matched_control

    # = stores a result in candidates; build a list, keeping its items in order.
    candidates = [
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        {"cap_m_s": 24.0, "speed_m_s": 23.0},
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        {"cap_m_s": 22.0, "speed_m_s": 21.0},
    # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
    ]
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert choose_speed_matched_control(candidates, 22.0)["cap_m_s"] == 22.0


# def defines test_speed_matching_rejects_missing_and_nonfinite_values; parentheses list its inputs, and the colon starts its indented body.
def test_speed_matching_rejects_missing_and_nonfinite_values():
    """Check that speed matching rejects missing and nonfinite values.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    # from driving_rl.analysis import brings the named tools into this file; parentheses allow several lines.
    from driving_rl.analysis import choose_speed_matched_control

    # An invalid input is supposed to fail here; accepting it would hide a bug.
    # with enters a temporary context and exits it afterward, even if an error interrupts the block.
    with pytest.raises(ValueError):
        # Call choose_speed_matched_control using parentheses to supply its inputs; this line does not store a separate return value.
        choose_speed_matched_control([], 22.0)
    # An invalid input is supposed to fail here; accepting it would hide a bug.
    # with enters a temporary context and exits it afterward, even if an error interrupts the block.
    with pytest.raises(ValueError):
        # Call choose_speed_matched_control using parentheses to supply its inputs; this line does not store a separate return value.
        choose_speed_matched_control(
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            [{"cap_m_s": 23.0, "speed_m_s": float("nan")}], 22.0
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )


# def defines test_scenario_bootstrap_retains_fixed_model_pairing; parentheses list its inputs, and the colon starts its indented body.
def test_scenario_bootstrap_retains_fixed_model_pairing():
    """Check that scenario bootstrap retains fixed model pairing.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    # from driving_rl.analysis import brings the named tools into this file; parentheses allow several lines.
    from driving_rl.analysis import scenario_bootstrap_fixed_models

    # Every scenario averages to zero across the two fixed models. Flattening
    # model/scenario observations would incorrectly introduce sampling variance.
    # = stores a result in r; call scenario_bootstrap_fixed_models using parentheses to supply its inputs.
    r = scenario_bootstrap_fixed_models(
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        [[1.0, -1.0, 1.0, -1.0], [-1.0, 1.0, -1.0, 1.0]], replicates=1000
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert r["conditional_95_percent_interval"] == [0.0, 0.0]
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert r["scenario_clusters"] == 4 and r["fixed_model_pairs"] == 2


# def defines test_scenario_bootstrap_constant_contrast_is_exact; parentheses list its inputs, and the colon starts its indented body.
def test_scenario_bootstrap_constant_contrast_is_exact():
    """Check that scenario bootstrap constant contrast is exact.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    # from driving_rl.analysis import brings the named tools into this file; parentheses allow several lines.
    from driving_rl.analysis import scenario_bootstrap_fixed_models

    # = stores a result in r; call scenario_bootstrap_fixed_models using parentheses to supply its inputs.
    r = scenario_bootstrap_fixed_models(np.full((3, 100), -0.25), replicates=1000)
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert r["mean_difference"] == -0.25
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert r["conditional_95_percent_interval"] == [-0.25, -0.25]


# def defines test_scenario_bootstrap_reproducible_without_changing_global_rng; parentheses list its inputs, and the colon starts its indented body.
def test_scenario_bootstrap_reproducible_without_changing_global_rng():
    """Check that scenario bootstrap reproducible without changing global rng.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    # from driving_rl.analysis import brings the named tools into this file; parentheses allow several lines.
    from driving_rl.analysis import scenario_bootstrap_fixed_models

    # = stores a result in matrix; call np.array using parentheses to supply its inputs.
    matrix = np.array([[0.0, -1.0, 0.0, 1.0], [1.0, 0.0, -1.0, 0.0]])
    # = stores a result in a; call scenario_bootstrap_fixed_models using parentheses to supply its inputs.
    a = scenario_bootstrap_fixed_models(matrix, replicates=1000, seed=72)
    # Call np.random.seed using parentheses to supply its inputs; this line does not store a separate return value.
    np.random.seed(91)
    # = stores a result in b; call scenario_bootstrap_fixed_models using parentheses to supply its inputs.
    b = scenario_bootstrap_fixed_models(matrix, replicates=1000, seed=72)
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert a == b
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert np.random.random() == np.random.RandomState(91).random_sample()


# def defines test_scenario_bootstrap_rejects_invalid_input; parentheses list its inputs, and the colon starts its indented body.
def test_scenario_bootstrap_rejects_invalid_input():
    """Check that scenario bootstrap rejects invalid input.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    # from driving_rl.analysis import brings the named tools into this file; parentheses allow several lines.
    from driving_rl.analysis import scenario_bootstrap_fixed_models

    # for takes each item from [[], [1.0, 2.0], [[np.nan]], [[np.inf]]] in turn; value holds the current item.
    for value in [[], [1.0, 2.0], [[np.nan]], [[np.inf]]]:
        # An invalid input is supposed to fail here; accepting it would hide a bug.
        # with enters a temporary context and exits it afterward, even if an error interrupts the block.
        with pytest.raises(ValueError):
            # Call scenario_bootstrap_fixed_models using parentheses to supply its inputs; this line does not store a separate return value.
            scenario_bootstrap_fixed_models(value)
    # An invalid input is supposed to fail here; accepting it would hide a bug.
    # with enters a temporary context and exits it afterward, even if an error interrupts the block.
    with pytest.raises(ValueError):
        # Call scenario_bootstrap_fixed_models using parentheses to supply its inputs; this line does not store a separate return value.
        scenario_bootstrap_fixed_models([[0.0]], replicates=0)
