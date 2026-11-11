"""Simple uncertainty summaries for independent evaluation episodes."""
import math


def wilson_interval(successes, trials, z=1.959963984540054):
    """95% binomial interval by default; conditional on the evaluated policy.

    This interval measures scenario variability, not training-seed variability.
    """
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("Require trials > 0 and 0 <= successes <= trials")
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)) / denominator
    return max(0., center - half), min(1., center + half)
