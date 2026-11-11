"""Estimate uncertainty in a fraction, such as the fraction of episodes with crashes.

A fraction from a small sample is not an exact probability. The Wilson interval
gives a reasonable uncertainty range, including when every recorded case has
the same outcome. It assumes independent trials and does not account for
variation from training different networks.

Original implementation notes:
Simple uncertainty summaries for independent evaluation episodes.
"""

import math


def wilson_interval(successes, trials, z=1.959963984540054):
    """Estimate an interval for an observed fraction such as crashes / episodes.
    The interval stays meaningful at 0% and 100%, unlike a simple symmetric error bar.

    Original implementation notes:
    95% binomial interval by default; conditional on the evaluated policy.

    This interval measures scenario variability, not training-seed variability.
    """
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("Require trials > 0 and 0 <= successes <= trials")
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    half = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - half), min(1.0, center + half)
