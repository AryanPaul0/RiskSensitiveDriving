"""Prediction checks must not invent an exact return or a scalar distribution."""

import pytest
from driving_rl.forecast_metrics import error_bounds, summarize_predictions


def test_exact_terminal_error_and_censored_error_range():
    exact = error_bounds(
        {"mean_prediction": 3.0, "return_lower": 2.0, "return_upper": 2.0}
    )
    assert exact == {
        "bias_lower": 1.0,
        "bias_upper": 1.0,
        "absolute_lower": 1.0,
        "absolute_upper": 1.0,
    }
    censored = error_bounds(
        {"mean_prediction": 2.0, "return_lower": 1.0, "return_upper": 3.0}
    )
    assert censored == {
        "bias_lower": -1.0,
        "bias_upper": 1.0,
        "absolute_lower": 0.0,
        "absolute_upper": 1.0,
    }


def test_scalar_mean_never_becomes_a_quantile_distribution():
    row = {
        "mean_prediction": 3.0,
        "return_lower": 2.0,
        "return_upper": 2.0,
        "quantiles": None,
        "censored": False,
        "crashed": True,
        "mean_speed_m_s": 20.0,
    }
    result = summarize_predictions([row])
    assert result["absolute_error_bounds"] == [1.0, 1.0]
    assert "quantile_checks" not in result


def test_quantile_ties_and_invalid_intervals_are_handled():
    row = {
        "mean_prediction": 2.0,
        "return_lower": 2.0,
        "return_upper": 2.0,
        "quantiles": [2.0] * 32,
        "censored": False,
        "crashed": False,
        "mean_speed_m_s": 20.0,
    }
    result = summarize_predictions([row])
    assert result["quantile_checks"]["minimum_mean_absolute_compatibility_gap"] == 0.0
    with pytest.raises(ValueError):
        error_bounds({"mean_prediction": 2.0, "return_lower": 3.0, "return_upper": 1.0})
