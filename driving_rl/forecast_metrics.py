"""Check predictions using the rewards that were actually recorded.

Think of a prediction as a guess about the final score. If a car crashes, we
know its complete score. If recording stops while it is still driving, we only
know a range. We must not pretend the unknown part is zero.
"""

import numpy as np
from .calibration import quantile_compatibility


def error_bounds(row):
    """Work out the smallest and largest possible error for one prediction.

    Signed error = prediction minus actual return. A positive number means the
    network was optimistic. Absolute error ignores the sign of the mistake.
    """
    # This is the mean future-return estimate for the selected action, not the next reward or a crash probability.
    prediction = float(row["mean_prediction"])
    lowest_possible_return = float(row["return_lower"])
    highest_possible_return = float(row["return_upper"])

    # Reject impossible inputs instead of silently producing misleading numbers.
    numbers = [prediction, lowest_possible_return, highest_possible_return]
    if not np.isfinite(numbers).all():
        raise ValueError("Expected a finite prediction and an ordered return range")
    if lowest_possible_return > highest_possible_return:
        raise ValueError("Expected a finite prediction and an ordered return range")

    # Subtracting the largest possible return gives the smallest signed error.
    smallest_signed_error = prediction - highest_possible_return
    largest_signed_error = prediction - lowest_possible_return

    # A prediction inside the return range might be exactly right. Otherwise,
    # its minimum absolute error is the distance to the nearest end of the range.
    smallest_absolute_error = max(
        0.0,
        lowest_possible_return - prediction,
        prediction - highest_possible_return,
    )
    largest_absolute_error = max(
        abs(prediction - lowest_possible_return),
        abs(prediction - highest_possible_return),
    )
    return {
        "bias_lower": smallest_signed_error,
        "bias_upper": largest_signed_error,
        "absolute_lower": smallest_absolute_error,
        "absolute_upper": largest_absolute_error,
    }


def average_column(rows, column_name):
    """Collect one column, then average it in the same order as before.

    For example, average_column(rows, 'mean_prediction') averages every saved
    prediction. NumPy still does the average, so rounding behavior stays the same.
    """
    values = []
    for row in rows:
        values.append(row[column_name])
    return float(np.mean(values))


def add_quantile_checks(result, rows):
    """Add distribution checks only when every row really contains quantiles.

    DQN supplies a single mean estimate. Calling that a quantile distribution
    would make the analysis look more informative than it actually is.
    """
    quantile_availability = []
    for row in rows:
        quantile_availability.append(row["quantiles"] is not None)
    if not any(quantile_availability):
        return
    if not all(quantile_availability):
        raise ValueError("Do not combine scalar and quantile predictors")

    predictions = []
    return_lower_limits = []
    return_upper_limits = []
    for row in rows:
        predictions.append(row["quantiles"])
        return_lower_limits.append(row["return_lower"])
        return_upper_limits.append(row["return_upper"])

    # Rows are recorded episodes; columns are the network's quantile outputs.
    # Keep their original order so we do not hide predictions that cross.
    predictions = np.array(predictions)
    lower_limits = np.array(return_lower_limits)
    upper_limits = np.array(return_upper_limits)
    result["quantile_checks"] = quantile_compatibility(
        predictions, lower_limits, upper_limits
    )
    adjacent_changes = np.diff(predictions, axis=1)
    result["quantile_crossing_fraction"] = float(np.mean(adjacent_changes < 0))


def summarize_predictions(rows):
    """Build one summary without removing the cars that survived the recording.

    Do not rely on average signed error alone: a +2 mistake and a -2 mistake
    cancel even though both predictions were wrong. Show absolute error too.
    These ranges account for missing continuation, not sampling uncertainty.
    """
    if not rows:
        raise ValueError("Need at least one recorded prediction")
    errors = []
    for row in rows:
        errors.append(error_bounds(row))
    result = {
        "episodes": len(rows),
        "mean_prediction": average_column(rows, "mean_prediction"),
        "bias_bounds": [
            average_column(errors, "bias_lower"),
            average_column(errors, "bias_upper"),
        ],
        "absolute_error_bounds": [
            average_column(errors, "absolute_lower"),
            average_column(errors, "absolute_upper"),
        ],
        "censored_fraction": average_column(rows, "censored"),
        "crash_rate": average_column(rows, "crashed"),
        "mean_speed_m_s": average_column(rows, "mean_speed_m_s"),
    }
    add_quantile_checks(result, rows)
    return result
