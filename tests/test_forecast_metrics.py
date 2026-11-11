"""Prediction checks must not invent an exact return or a scalar distribution."""  # Describe this operation for readers and help().

import pytest  # Make pytest available for this file's calculations and workflow.
from driving_rl.forecast_metrics import error_bounds, summarize_predictions  # Make error_bounds, summarize_predictions available for this file's calculations and workflow.


def test_exact_terminal_error_and_censored_error_range():  # Define the test exact terminal error and censored error range operation used by this workflow.
    exact = error_bounds(  # Set exact to error bounds applied to the named settings or measurements assembled below.
        {"mean_prediction": 3.0, "return_lower": 2.0, "return_upper": 2.0}  # Record 'mean_prediction' as 3.0.
    )  # Finish assembling error bounds inputs.
    assert exact == {  # Check that exact equals the named settings or measurements assembled below; fail this test if it is not true.
        "bias_lower": 1.0,  # Record 'bias_lower' as 1.0.
        "bias_upper": 1.0,  # Record 'bias_upper' as 1.0.
        "absolute_lower": 1.0,  # Record 'absolute_lower' as 1.0.
        "absolute_upper": 1.0,  # Record 'absolute_upper' as 1.0.
    }  # Finish assembling the grouped values above.
    censored = error_bounds(  # Set censored to error bounds applied to the named settings or measurements assembled below.
        {"mean_prediction": 2.0, "return_lower": 1.0, "return_upper": 3.0}  # Record 'mean_prediction' as 2.0.
    )  # Finish assembling error bounds inputs.
    assert censored == {  # Check that censored equals the named settings or measurements assembled below; fail this test if it is not true.
        "bias_lower": -1.0,  # Record 'bias_lower' as negative 1.0.
        "bias_upper": 1.0,  # Record 'bias_upper' as 1.0.
        "absolute_lower": 0.0,  # Record 'absolute_lower' as 0.0.
        "absolute_upper": 1.0,  # Record 'absolute_upper' as 1.0.
    }  # Finish assembling the grouped values above.


def test_scalar_mean_never_becomes_a_quantile_distribution():  # Define the test scalar mean never becomes a quantile distribution operation used by this workflow.
    row = {  # Set row to the named settings or measurements assembled below.
        "mean_prediction": 3.0,  # Record 'mean_prediction' as 3.0.
        "return_lower": 2.0,  # Record 'return_lower' as 2.0.
        "return_upper": 2.0,  # Record 'return_upper' as 2.0.
        "quantiles": None,  # Record 'quantiles' as None.
        "censored": False,  # Record 'censored' as False.
        "crashed": True,  # Record 'crashed' as True.
        "mean_speed_m_s": 20.0,  # Record 'mean_speed_m_s' as 20.0.
    }  # Finish assembling the grouped values above.
    result = summarize_predictions([row])  # Set result to summarize predictions applied to row.
    assert result["absolute_error_bounds"] == [1.0, 1.0]  # Check that the entries of result selected by 'absolute_error_bounds' equals 1.0, 1.0; fail this test if it is not true.
    assert "quantile_checks" not in result  # Check that 'quantile_checks' does not belong to result; fail this test if it is not true.


def test_quantile_ties_and_invalid_intervals_are_handled():  # Define the test quantile ties and invalid intervals are handled operation used by this workflow.
    row = {  # Set row to the named settings or measurements assembled below.
        "mean_prediction": 2.0,  # Record 'mean_prediction' as 2.0.
        "return_lower": 2.0,  # Record 'return_lower' as 2.0.
        "return_upper": 2.0,  # Record 'return_upper' as 2.0.
        "quantiles": [2.0] * 32,  # Record 'quantiles' as (2.0 times 32).
        "censored": False,  # Record 'censored' as False.
        "crashed": False,  # Record 'crashed' as False.
        "mean_speed_m_s": 20.0,  # Record 'mean_speed_m_s' as 20.0.
    }  # Finish assembling the grouped values above.
    result = summarize_predictions([row])  # Set result to summarize predictions applied to row.
    assert result["quantile_checks"]["minimum_mean_absolute_compatibility_gap"] == 0.0  # Check that the entries of the entries of result selected by 'quantile_checks' selected by 'minimum_mean_absolute_compatibility_gap' equals 0.0; fail this test if it is not true.
    with pytest.raises(ValueError):  # Keep this temporary context active during the enclosed work, then clean it up.
        error_bounds({"mean_prediction": 2.0, "return_lower": 3.0, "return_upper": 1.0})  # Error bounds applied to the named settings or measurements assembled below.
