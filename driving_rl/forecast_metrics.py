"""Check predictions using the rewards that were actually recorded.

Think of a prediction as a guess about the final score. If a car crashes, we
know its complete score. If recording stops while it is still driving, we only
know a range. We must not pretend the unknown part is zero.
"""

import numpy as np  # Make numpy available for this file's calculations and workflow.
from .calibration import quantile_compatibility  # Make quantile_compatibility available for this file's calculations and workflow.


def error_bounds(row):  # Work out the smallest and largest possible error for one prediction.
    """Work out the smallest and largest possible error for one prediction.

    Signed error = prediction minus actual return. A positive number means the
    network was optimistic. Absolute error ignores the sign of the mistake.
    """
    prediction = float(row["mean_prediction"])  # This is the mean future-return estimate for the selected action, not the next reward or a crash probability.
    lowest_possible_return = float(row["return_lower"])  # Set lowest possible return to the entries of row selected by 'return_lower' as a floating-point value.
    highest_possible_return = float(row["return_upper"])  # Set highest possible return to the entries of row selected by 'return_upper' as a floating-point value.

    numbers = [prediction, lowest_possible_return, highest_possible_return]  # Reject impossible inputs instead of silently producing misleading numbers.
    if not np.isfinite(numbers).all():  # Use this branch when not the result of np.isfinite(numbers).all.
        raise ValueError("Expected a finite prediction and an ordered return range")  # Reject this invalid case instead of continuing with misleading experiment results.
    if lowest_possible_return > highest_possible_return:  # Use this branch when lowest possible return is above highest possible return.
        raise ValueError("Expected a finite prediction and an ordered return range")  # Reject this invalid case instead of continuing with misleading experiment results.

    smallest_signed_error = prediction - highest_possible_return  # Subtracting the largest possible return gives the smallest signed error.
    largest_signed_error = prediction - lowest_possible_return  # Set largest signed error to (prediction minus lowest possible return).

    smallest_absolute_error = max(  # Example: prediction=3 and possible return=[1,2] gives signed error [1,2]. The model was optimistic by at least 1; the cutoff prevents knowing exactly how optimistic. This is not a confidence interval. A prediction inside the return range might be exactly right. Otherwise, its minimum absolute error is the distance to the nearest end of the range.
        0.0,  # Include 0.0 in the surrounding operation.
        lowest_possible_return - prediction,  # Include (lowest possible return minus prediction) in the surrounding operation.
        prediction - highest_possible_return,  # Include (prediction minus highest possible return) in the surrounding operation.
    )  # Finish assembling max inputs.
    largest_absolute_error = max(  # Set largest absolute error to max applied to abs applied to (prediction minus lowest possible return), abs applied to (prediction minus highest possible return).
        abs(prediction - lowest_possible_return),  # Include abs applied to (prediction minus lowest possible return) in the surrounding operation.
        abs(prediction - highest_possible_return),  # Include abs applied to (prediction minus highest possible return) in the surrounding operation.
    )  # Finish assembling max inputs.
    return {  # Return the named settings or measurements assembled below.
        "bias_lower": smallest_signed_error,  # Record 'bias_lower' as smallest signed error.
        "bias_upper": largest_signed_error,  # Record 'bias_upper' as largest signed error.
        "absolute_lower": smallest_absolute_error,  # Record 'absolute_lower' as smallest absolute error.
        "absolute_upper": largest_absolute_error,  # Record 'absolute_upper' as largest absolute error.
    }  # Finish assembling the grouped values above.


def average_column(rows, column_name):  # Collect one column, then average it in the same order as before.
    """Collect one column, then average it in the same order as before.

    For example, average_column(rows, 'mean_prediction') averages every saved
    prediction. NumPy still does the average, so rounding behavior stays the same.
    """
    values = []  # Set values to .
    for row in rows:  # Process each row from rows.
        values.append(row[column_name])  # Add the entries of row selected by column_name to values.
    return float(np.mean(values))  # Return the average of values as a floating-point value.


def add_quantile_checks(result, rows):  # Add distribution checks only when every row really contains quantiles.
    """Add distribution checks only when every row really contains quantiles.

    DQN supplies a single mean estimate. Calling that a quantile distribution
    would make the analysis look more informative than it actually is.
    """
    quantile_availability = []  # Set quantile availability to .
    for row in rows:  # Process each row from rows.
        quantile_availability.append(row["quantiles"] is not None)  # Add the entries of row selected by 'quantiles' is not none to quantile availability.
    if not any(quantile_availability):  # Use this branch when not any applied to quantile availability.
        return  # Finish this operation without producing another result.
    if not all(quantile_availability):  # Use this branch when not all applied to quantile availability.
        raise ValueError("Do not combine scalar and quantile predictors")  # Reject this invalid case instead of continuing with misleading experiment results.

    predictions = []  # Set predictions to .
    return_lower_limits = []  # Set return lower limits to .
    return_upper_limits = []  # Set return upper limits to .
    for row in rows:  # Process each row from rows.
        predictions.append(row["quantiles"])  # Add the entries of row selected by 'quantiles' to predictions.
        return_lower_limits.append(row["return_lower"])  # Add the entries of row selected by 'return_lower' to return lower limits.
        return_upper_limits.append(row["return_upper"])  # Add the entries of row selected by 'return_upper' to return upper limits.

    predictions = np.array(predictions)  # Rows are recorded episodes; columns are the network's quantile outputs. Keep their original order so we do not hide predictions that cross.
    lower_limits = np.array(return_lower_limits)  # Set lower limits to return lower limits as a numerical array/tensor for batch calculations.
    upper_limits = np.array(return_upper_limits)  # Set upper limits to return upper limits as a numerical array/tensor for batch calculations.
    result["quantile_checks"] = quantile_compatibility(  # Set the entries of result selected by 'quantile_checks' to quantile compatibility applied to predictions, lower limits, upper limits.
        predictions, lower_limits, upper_limits  # Include predictions, lower limits, upper limits in the surrounding operation.
    )  # Finish assembling quantile compatibility inputs.
    adjacent_changes = np.diff(predictions, axis=1)  # Set adjacent changes to the differences between adjacent entries of predictions.
    result["quantile_crossing_fraction"] = float(np.mean(adjacent_changes < 0))  # diff subtracts neighboring heads within each episode. Negative means a higher assigned quantile predicted a lower return than its neighbor. mean of these True/False flags is the fraction of adjacent pairs crossing.


def summarize_predictions(rows):  # Build one summary without removing the cars that survived the recording.
    """Build one summary without removing the cars that survived the recording.

    Do not rely on average signed error alone: a +2 mistake and a -2 mistake
    cancel even though both predictions were wrong. Show absolute error too.
    These ranges account for missing continuation, not sampling uncertainty.
    """
    if not rows:  # Use this branch when not rows.
        raise ValueError("Need at least one recorded prediction")  # Reject this invalid case instead of continuing with misleading experiment results.
    errors = []  # Set errors to .
    for row in rows:  # Process each row from rows.
        errors.append(error_bounds(row))  # Add error bounds applied to row to errors.
    result = {  # Set result to the named settings or measurements assembled below.
        "episodes": len(rows),  # Record 'episodes' as the number of entries in rows.
        "mean_prediction": average_column(rows, "mean_prediction"),  # Record 'mean_prediction' as average column applied to rows, 'mean_prediction'.
        "bias_bounds": [  # Record 'bias_bounds' as average column applied to errors, 'bias_lower', average column applied to errors, 'bias_upper'.
            average_column(errors, "bias_lower"),  # Include average column applied to errors, 'bias_lower' in the surrounding operation.
            average_column(errors, "bias_upper"),  # Include average column applied to errors, 'bias_upper' in the surrounding operation.
        ],  # Finish assembling the grouped values above.
        "absolute_error_bounds": [  # Record 'absolute_error_bounds' as average column applied to errors, 'absolute_lower', average column applied to errors, 'absolute_upper'.
            average_column(errors, "absolute_lower"),  # Include average column applied to errors, 'absolute_lower' in the surrounding operation.
            average_column(errors, "absolute_upper"),  # Include average column applied to errors, 'absolute_upper' in the surrounding operation.
        ],  # Finish assembling the grouped values above.
        "censored_fraction": average_column(rows, "censored"),  # Record 'censored_fraction' as average column applied to rows, 'censored'.
        "crash_rate": average_column(rows, "crashed"),  # Record 'crash_rate' as average column applied to rows, 'crashed'.
        "mean_speed_m_s": average_column(rows, "mean_speed_m_s"),  # Record 'mean_speed_m_s' as average column applied to rows, 'mean_speed_m_s'.
    }  # Finish assembling the grouped values above.
    add_quantile_checks(result, rows)  # Add quantile checks applied to result, rows.
    return result  # Return result.
