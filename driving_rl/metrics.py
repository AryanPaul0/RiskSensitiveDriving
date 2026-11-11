"""Estimate uncertainty in a fraction, such as the fraction of episodes with crashes.

A fraction from a small sample is not an exact probability. The Wilson interval
gives a reasonable uncertainty range, including when every recorded case has
the same outcome. It assumes independent trials and does not account for
variation from training different networks.

Original implementation notes:
Simple uncertainty summaries for independent evaluation episodes.
"""

import math  # Make math available for this file's calculations and workflow.


def wilson_interval(successes, trials, z=1.959963984540054):  # Estimate an interval for an observed fraction such as crashes / episodes.
    """Estimate an interval for an observed fraction such as crashes / episodes.
    The interval stays meaningful at 0% and 100%, unlike a simple symmetric error bar.

    Original implementation notes:
    95% binomial interval by default; conditional on the evaluated policy.

    This interval measures scenario variability, not training-seed variability.
    """
    if trials <= 0 or not 0 <= successes <= trials:  # Use this branch when trials is at most 0 or not 0 is at most successes is at most trials.
        raise ValueError("Require trials > 0 and 0 <= successes <= trials")  # Reject this invalid case instead of continuing with misleading experiment results.
    proportion = successes / trials  # Set proportion to (successes divided by trials).
    denominator = 1 + z * z / trials  # Here a "success" may mean a crash: it is simply the event being counted. z is about 1.96 for a 95% interval. The Wilson adjustment below prevents an observed 0 crashes from being reported as zero uncertainty.
    center = (proportion + z * z / (2 * trials)) / denominator  # Set center to ((proportion plus ((z times z) divided by (2 times trials))) divided by denominator).
    half = (  # Set half to ((z times math.sqrt applied to (proportion * (1 - proportion) / trials plus z * z / (4 * trials * trials))) divided by denominator).
        z  # Include ((z times math.sqrt applied to (proportion * (1 - proportion) / trials plus z * z / (4 * trials * trials))) divided by denominator) in the surrounding operation.
        * math.sqrt(  # Include math.sqrt applied to (((proportion times 1 - proportion) divided by trials) plus ((z times z) divided by (4 * trials times trials))) in the surrounding operation.
            proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)  # Include (((proportion times (1 minus proportion)) divided by trials) plus ((z times z) divided by ((4 times trials) times trials))) in the surrounding operation.
        )  # Finish assembling math.sqrt inputs.
        / denominator  # Include denominator in the surrounding operation.
    )  # Finish assembling the grouped values above.
    return max(0.0, center - half), min(1.0, center + half)  # Keep probability endpoints inside [0,1]. This applies to independent episode outcomes; paired model/scenario comparisons use the bootstrap functions instead of pretending all pooled episodes are independent.
