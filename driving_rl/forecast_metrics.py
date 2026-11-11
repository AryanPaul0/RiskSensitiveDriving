"""Small calculations for checking predicted returns.

An episode ending in a crash has an exact observed return. A recording that
stops at the time limit has a range. These functions keep that distinction.
"""

import numpy as np
from .calibration import quantile_compatibility


def error_bounds(row):
    """Find all possible signed and absolute errors for one recorded prediction.

    Signed error means prediction minus actual return. If the prediction lies
    inside the observed return range, its smallest possible absolute error is 0.
    """
    prediction = float(row["mean_prediction"])
    low = float(row["return_lower"])
    high = float(row["return_upper"])
    if not np.isfinite([prediction, low, high]).all() or low > high:
        raise ValueError("Expected a finite prediction and an ordered return range")
    return {
        "bias_lower": prediction - high,
        "bias_upper": prediction - low,
        "absolute_lower": max(0.0, low - prediction, prediction - high),
        "absolute_upper": max(abs(prediction - low), abs(prediction - high)),
    }


def summarize_predictions(rows):
    """Average the error bounds without discarding surviving episodes.

    Signed errors can cancel, so also show absolute errors. These are bounds
    from missing continuation, not statistical confidence intervals.
    """
    if not rows:
        raise ValueError("Need at least one recorded prediction")
    errors = []
    for row in rows:
        errors.append(error_bounds(row))
    result = {
        "episodes": len(rows),
        "mean_prediction": float(np.mean([r["mean_prediction"] for r in rows])),
        "bias_bounds": [
            float(np.mean([e["bias_lower"] for e in errors])),
            float(np.mean([e["bias_upper"] for e in errors])),
        ],
        "absolute_error_bounds": [
            float(np.mean([e["absolute_lower"] for e in errors])),
            float(np.mean([e["absolute_upper"] for e in errors])),
        ],
        "censored_fraction": float(np.mean([r["censored"] for r in rows])),
        "crash_rate": float(np.mean([r["crashed"] for r in rows])),
        "mean_speed_m_s": float(np.mean([r["mean_speed_m_s"] for r in rows])),
    }
    available = [r["quantiles"] is not None for r in rows]
    if any(available):
        if not all(available):
            raise ValueError("Do not combine scalar and quantile predictors")
        values = np.array([r["quantiles"] for r in rows])
        lows = np.array([r["return_lower"] for r in rows])
        highs = np.array([r["return_upper"] for r in rows])
        result["quantile_checks"] = quantile_compatibility(values, lows, highs)
        result["quantile_crossing_fraction"] = float(
            np.mean(np.diff(values, axis=1) < 0)
        )
    return result
