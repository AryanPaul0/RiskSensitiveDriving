# This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
"""Prediction checks must not invent an exact return or a scalar distribution."""

# import makes these modules available; "as" gives a module a shorter local name.
import pytest
# from driving_rl.forecast_metrics import brings the named tools into this file; parentheses allow several lines.
from driving_rl.forecast_metrics import error_bounds, summarize_predictions


# def defines test_exact_terminal_error_and_censored_error_range; parentheses list its inputs, and the colon starts its indented body.
def test_exact_terminal_error_and_censored_error_range():
    # = stores a result in exact; call error_bounds using parentheses to supply its inputs.
    exact = error_bounds(
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        {"mean_prediction": 3.0, "return_lower": 2.0, "return_upper": 2.0}
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert exact == {
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "bias_lower": 1.0,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "bias_upper": 1.0,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "absolute_lower": 1.0,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "absolute_upper": 1.0,
    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
    }
    # = stores a result in censored; call error_bounds using parentheses to supply its inputs.
    censored = error_bounds(
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        {"mean_prediction": 2.0, "return_lower": 1.0, "return_upper": 3.0}
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert censored == {
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "bias_lower": -1.0,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "bias_upper": 1.0,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "absolute_lower": 0.0,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "absolute_upper": 1.0,
    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
    }


# def defines test_scalar_mean_never_becomes_a_quantile_distribution; parentheses list its inputs, and the colon starts its indented body.
def test_scalar_mean_never_becomes_a_quantile_distribution():
    # = stores a result in row; build a dictionary, which stores values under named keys.
    row = {
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "mean_prediction": 3.0,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "return_lower": 2.0,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "return_upper": 2.0,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "quantiles": None,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "censored": False,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "crashed": True,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "mean_speed_m_s": 20.0,
    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
    }
    # = stores a result in result; call summarize_predictions using parentheses to supply its inputs.
    result = summarize_predictions([row])
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert result["absolute_error_bounds"] == [1.0, 1.0]
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert "quantile_checks" not in result


# def defines test_quantile_ties_and_invalid_intervals_are_handled; parentheses list its inputs, and the colon starts its indented body.
def test_quantile_ties_and_invalid_intervals_are_handled():
    # = stores a result in row; build a dictionary, which stores values under named keys.
    row = {
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "mean_prediction": 2.0,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "return_lower": 2.0,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "return_upper": 2.0,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "quantiles": [2.0] * 32,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "censored": False,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "crashed": False,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "mean_speed_m_s": 20.0,
    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
    }
    # = stores a result in result; call summarize_predictions using parentheses to supply its inputs.
    result = summarize_predictions([row])
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert result["quantile_checks"]["minimum_mean_absolute_compatibility_gap"] == 0.0
    # with enters a temporary context and exits it afterward, even if an error interrupts the block.
    with pytest.raises(ValueError):
        # Call error_bounds using parentheses to supply its inputs; this line does not store a separate return value.
        error_bounds({"mean_prediction": 2.0, "return_lower": 3.0, "return_upper": 1.0})
