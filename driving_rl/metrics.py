"""Estimate uncertainty in a fraction, such as the fraction of episodes with crashes.

A fraction from a small sample is not an exact probability. The Wilson interval
gives a reasonable uncertainty range, including when every recorded case has
the same outcome. It assumes independent trials and does not account for
variation from training different networks.

Original implementation notes:
Simple uncertainty summaries for independent evaluation episodes.
"""

# import makes these modules available; "as" gives a module a shorter local name.
import math


# def defines wilson_interval; parentheses list its inputs, and the colon starts its indented body.
def wilson_interval(successes, trials, z=1.959963984540054):
    """Estimate an interval for an observed fraction such as crashes / episodes.
    The interval stays meaningful at 0% and 100%, unlike a simple symmetric error bar.

    Original implementation notes:
    95% binomial interval by default; conditional on the evaluated policy.

    This interval measures scenario variability, not training-seed variability.
    """
    # if checks trials <= 0 or not 0 <= successes <= trials; only a true condition runs the indented block.
    if trials <= 0 or not 0 <= successes <= trials:
        # raise signals an error and stops this normal path, so invalid data are not silently accepted.
        raise ValueError("Require trials > 0 and 0 <= successes <= trials")
    # = stores a result in proportion; apply / to these operands; numbers are divided, while Path objects join folder names.
    proportion = successes / trials
    # = stores a result in denominator; combine the operands with the displayed operator; parentheses control grouping.
    denominator = 1 + z * z / trials
    # = stores a result in center; apply / to these operands; numbers are divided, while Path objects join folder names.
    center = (proportion + z * z / (2 * trials)) / denominator
    # = stores a result in half; apply / to these operands; numbers are divided, while Path objects join folder names.
    half = (
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        z
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        * math.sqrt(
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        / denominator
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # return ends this function and sends its result back to the code that called it.
    return max(0.0, center - half), min(1.0, center + half)
