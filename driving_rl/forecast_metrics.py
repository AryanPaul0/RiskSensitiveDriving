"""Check predictions using the rewards that were actually recorded.

Think of a prediction as a guess about the final score. If a car crashes, we
know its complete score. If recording stops while it is still driving, we only
know a range. We must not pretend the unknown part is zero.
"""

# import makes these modules available; "as" gives a module a shorter local name.
import numpy as np
# from calibration import brings the named tools into this file; parentheses allow several lines.
from .calibration import quantile_compatibility


# def defines error_bounds; parentheses list its inputs, and the colon starts its indented body.
def error_bounds(row):
    """Work out the smallest and largest possible error for one prediction.

    Signed error = prediction minus actual return. A positive number means the
    network was optimistic. Absolute error ignores the sign of the mistake.
    """
    # This is the mean future-return estimate for the selected action, not the next reward or a crash probability.
    # = stores a result in prediction; call float using parentheses to supply its inputs.
    prediction = float(row["mean_prediction"])
    # = stores a result in lowest_possible_return; call float using parentheses to supply its inputs.
    lowest_possible_return = float(row["return_lower"])
    # = stores a result in highest_possible_return; call float using parentheses to supply its inputs.
    highest_possible_return = float(row["return_upper"])

    # Reject impossible inputs instead of silently producing misleading numbers.
    # = stores a result in numbers; build a list, keeping its items in order.
    numbers = [prediction, lowest_possible_return, highest_possible_return]
    # if checks not np.isfinite(numbers).all(); only a true condition runs the indented block.
    if not np.isfinite(numbers).all():
        # raise signals an error and stops this normal path, so invalid data are not silently accepted.
        raise ValueError("Expected a finite prediction and an ordered return range")
    # if checks lowest_possible_return > highest_possible_return; only a true condition runs the indented block.
    if lowest_possible_return > highest_possible_return:
        # raise signals an error and stops this normal path, so invalid data are not silently accepted.
        raise ValueError("Expected a finite prediction and an ordered return range")

    # Subtracting the largest possible return gives the smallest signed error.
    # = stores a result in smallest_signed_error; combine the operands with the displayed operator; parentheses control grouping.
    smallest_signed_error = prediction - highest_possible_return
    # = stores a result in largest_signed_error; combine the operands with the displayed operator; parentheses control grouping.
    largest_signed_error = prediction - lowest_possible_return

    # A prediction inside the return range might be exactly right. Otherwise,
    # its minimum absolute error is the distance to the nearest end of the range.
    # = stores a result in smallest_absolute_error; call max using parentheses to supply its inputs.
    smallest_absolute_error = max(
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        0.0,
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        lowest_possible_return - prediction,
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        prediction - highest_possible_return,
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # = stores a result in largest_absolute_error; call max using parentheses to supply its inputs.
    largest_absolute_error = max(
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        abs(prediction - lowest_possible_return),
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        abs(prediction - highest_possible_return),
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # return ends this function and sends its result back to the code that called it.
    return {
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "bias_lower": smallest_signed_error,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "bias_upper": largest_signed_error,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "absolute_lower": smallest_absolute_error,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "absolute_upper": largest_absolute_error,
    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
    }


# def defines average_column; parentheses list its inputs, and the colon starts its indented body.
def average_column(rows, column_name):
    """Collect one column, then average it in the same order as before.

    For example, average_column(rows, 'mean_prediction') averages every saved
    prediction. NumPy still does the average, so rounding behavior stays the same.
    """
    # = stores a result in values; build a list, keeping its items in order.
    values = []
    # for takes each item from rows in turn; row holds the current item.
    for row in rows:
        # Call values.append using parentheses to supply its inputs; this line does not store a separate return value.
        values.append(row[column_name])
    # return ends this function and sends its result back to the code that called it.
    return float(np.mean(values))


# def defines add_quantile_checks; parentheses list its inputs, and the colon starts its indented body.
def add_quantile_checks(result, rows):
    """Add distribution checks only when every row really contains quantiles.

    DQN supplies a single mean estimate. Calling that a quantile distribution
    would make the analysis look more informative than it actually is.
    """
    # = stores a result in quantile_availability; build a list, keeping its items in order.
    quantile_availability = []
    # for takes each item from rows in turn; row holds the current item.
    for row in rows:
        # Call quantile_availability.append using parentheses to supply its inputs; this line does not store a separate return value.
        quantile_availability.append(row["quantiles"] is not None)
    # if checks not any(quantile_availability); only a true condition runs the indented block.
    if not any(quantile_availability):
        # return ends this function and sends its result back to the code that called it.
        return
    # if checks not all(quantile_availability); only a true condition runs the indented block.
    if not all(quantile_availability):
        # raise signals an error and stops this normal path, so invalid data are not silently accepted.
        raise ValueError("Do not combine scalar and quantile predictors")

    # = stores a result in predictions; build a list, keeping its items in order.
    predictions = []
    # = stores a result in return_lower_limits; build a list, keeping its items in order.
    return_lower_limits = []
    # = stores a result in return_upper_limits; build a list, keeping its items in order.
    return_upper_limits = []
    # for takes each item from rows in turn; row holds the current item.
    for row in rows:
        # Call predictions.append using parentheses to supply its inputs; this line does not store a separate return value.
        predictions.append(row["quantiles"])
        # Call return_lower_limits.append using parentheses to supply its inputs; this line does not store a separate return value.
        return_lower_limits.append(row["return_lower"])
        # Call return_upper_limits.append using parentheses to supply its inputs; this line does not store a separate return value.
        return_upper_limits.append(row["return_upper"])

    # Rows are recorded episodes; columns are the network's quantile outputs.
    # Keep their original order so we do not hide predictions that cross.
    # = stores a result in predictions; call np.array using parentheses to supply its inputs.
    predictions = np.array(predictions)
    # = stores a result in lower_limits; call np.array using parentheses to supply its inputs.
    lower_limits = np.array(return_lower_limits)
    # = stores a result in upper_limits; call np.array using parentheses to supply its inputs.
    upper_limits = np.array(return_upper_limits)
    # = stores a result in result['quantile_checks']; call quantile_compatibility using parentheses to supply its inputs.
    result["quantile_checks"] = quantile_compatibility(
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        predictions, lower_limits, upper_limits
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # = stores a result in adjacent_changes; call np.diff using parentheses to supply its inputs.
    adjacent_changes = np.diff(predictions, axis=1)
    # = stores a result in result['quantile_crossing_fraction']; call float using parentheses to supply its inputs.
    result["quantile_crossing_fraction"] = float(np.mean(adjacent_changes < 0))


# def defines summarize_predictions; parentheses list its inputs, and the colon starts its indented body.
def summarize_predictions(rows):
    """Build one summary without removing the cars that survived the recording.

    Do not rely on average signed error alone: a +2 mistake and a -2 mistake
    cancel even though both predictions were wrong. Show absolute error too.
    These ranges account for missing continuation, not sampling uncertainty.
    """
    # if checks not rows; only a true condition runs the indented block.
    if not rows:
        # raise signals an error and stops this normal path, so invalid data are not silently accepted.
        raise ValueError("Need at least one recorded prediction")
    # = stores a result in errors; build a list, keeping its items in order.
    errors = []
    # for takes each item from rows in turn; row holds the current item.
    for row in rows:
        # Call errors.append using parentheses to supply its inputs; this line does not store a separate return value.
        errors.append(error_bounds(row))
    # = stores a result in result; build a dictionary, which stores values under named keys.
    result = {
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "episodes": len(rows),
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "mean_prediction": average_column(rows, "mean_prediction"),
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "bias_bounds": [
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            average_column(errors, "bias_lower"),
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            average_column(errors, "bias_upper"),
        # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
        ],
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "absolute_error_bounds": [
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            average_column(errors, "absolute_lower"),
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            average_column(errors, "absolute_upper"),
        # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
        ],
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "censored_fraction": average_column(rows, "censored"),
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "crash_rate": average_column(rows, "crashed"),
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "mean_speed_m_s": average_column(rows, "mean_speed_m_s"),
    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
    }
    # Call add_quantile_checks using parentheses to supply its inputs; this line does not store a separate return value.
    add_quantile_checks(result, rows)
    # return ends this function and sends its result back to the code that called it.
    return result
