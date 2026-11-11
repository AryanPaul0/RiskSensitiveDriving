"""Tests for density coverage.

Tests use deliberately small examples so mistakes in learning, data pairing or
evaluation show up before spending time on long training runs.
"""

# import makes these modules available; "as" gives a module a shorter local name.
import numpy as np
# import makes these modules available; "as" gives a module a shorter local name.
import pytest
# from driving_rl.envs import brings the named tools into this file; parentheses allow several lines.
from driving_rl.envs import make_env, reset_training_episode


# def defines test_single_density_reproduces_original_environment_trajectory; parentheses list its inputs, and the colon starts its indented body.
def test_single_density_reproduces_original_environment_trajectory():
    """Check that single density reproduces original environment trajectory.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    # = unpacks the right-hand result into (a, b), matching values to names in order.
    a, b = make_env(), make_env()
    # try starts a block whose errors or cleanup are handled by the following except/finally blocks.
    try:
        # = unpacks the right-hand result into (x, _), matching values to names in order.
        x, _ = a.reset(seed=72)
        # = unpacks the right-hand result into (y, _, density), matching values to names in order.
        y, _, density = reset_training_episode(b, np.random.default_rng(8), [1.0], 72)
        # Call np.testing.assert_array_equal using parentheses to supply its inputs; this line does not store a separate return value.
        np.testing.assert_array_equal(x, y)
        # assert checks this condition; a false result raises AssertionError and fails this check.
        assert density == 1.0
        # for takes each item from [1, 3, 0, 1, 4] in turn; action holds the current item.
        for action in [1, 3, 0, 1, 4]:
            # = unpacks the right-hand result into (x, r, t, u, _), matching values to names in order.
            x, r, t, u, _ = a.step(action)
            # = unpacks the right-hand result into (y, s, v, w, _), matching values to names in order.
            y, s, v, w, _ = b.step(action)
            # Call np.testing.assert_array_equal using parentheses to supply its inputs; this line does not store a separate return value.
            np.testing.assert_array_equal(x, y)
            # assert checks this condition; a false result raises AssertionError and fails this check.
            assert (r, t, u) == (s, v, w)
            # if checks t or u; only a true condition runs the indented block.
            if t or u:
                # break leaves the nearest loop immediately; execution continues after that loop.
                break
    # finally always runs this cleanup block when the try block ends, including after an error or return.
    finally:
        # Call a.close using parentheses to supply its inputs; this line does not store a separate return value.
        a.close()
        # Call b.close using parentheses to supply its inputs; this line does not store a separate return value.
        b.close()


# def defines test_density_is_applied_before_reset_and_changes_traffic_spacing; parentheses list its inputs, and the colon starts its indented body.
def test_density_is_applied_before_reset_and_changes_traffic_spacing():
    """Check that density is applied before reset and changes traffic spacing.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    # = stores a result in env; call make_env using parentheses to supply its inputs.
    env = make_env()
    # try starts a block whose errors or cleanup are handled by the following except/finally blocks.
    try:
        # = stores a result in spans; build a list, keeping its items in order.
        spans = []
        # for takes each item from [1.0, 2.0] in turn; density holds the current item.
        for density in [1.0, 2.0]:
            # = unpacks the right-hand result into (obs, _, selected), matching values to names in order.
            obs, _, selected = reset_training_episode(
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                env, np.random.default_rng(2), [density], 47
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            )
            # assert checks this condition; a false result raises AssertionError and fails this check.
            assert (
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                selected == density
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                and env.unwrapped.config["vehicles_density"] == density
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            )
            # assert checks this condition; a false result raises AssertionError and fails this check.
            assert obs.shape == (8, 5)
            # Call spans.append using parentheses to supply its inputs; this line does not store a separate return value.
            spans.append(np.ptp([v.position[0] for v in env.unwrapped.road.vehicles]))
        # assert checks this condition; a false result raises AssertionError and fails this check.
        assert spans[1] < spans[0]
        # An invalid input is supposed to fail here; accepting it would hide a bug.
        # with enters a temporary context and exits it afterward, even if an error interrupts the block.
        with pytest.raises(ValueError):
            # Call reset_training_episode using parentheses to supply its inputs; this line does not store a separate return value.
            reset_training_episode(env, np.random.default_rng(2), [float("nan")])
    # finally always runs this cleanup block when the try block ends, including after an error or return.
    finally:
        # Call env.close using parentheses to supply its inputs; this line does not store a separate return value.
        env.close()


# def defines test_mixture_schedule_is_reproducible_and_does_not_consume_action_rng; parentheses list its inputs, and the colon starts its indented body.
def test_mixture_schedule_is_reproducible_and_does_not_consume_action_rng():
    """Check that mixture schedule is reproducible and does not consume action rng.
    This test protects an assumption that the experiment or its interpretation relies on.
    """

    # class defines the Fake object type; names in parentheses are parent classes to inherit from.
    class Fake:
        # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
        """Build a small controlled example for the tests below."""

        # def defines __init__; parentheses list its inputs, and the colon starts its indented body.
        def __init__(self):
            # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
            """Build a small controlled example for the tests below."""
            # = stores a result in self.unwrapped; evaluate the expression on the right before using its result.
            self.unwrapped = self
            # = stores a result in self.config; build a dictionary, which stores values under named keys.
            self.config = {"vehicles_density": 1.0}

        # def defines configure; parentheses list its inputs, and the colon starts its indented body.
        def configure(self, c):
            # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
            """Build a small controlled example for the tests below."""
            # Call self.config.update using parentheses to supply its inputs; this line does not store a separate return value.
            self.config.update(c)

        # def defines reset; parentheses list its inputs, and the colon starts its indented body.
        def reset(self, seed=None):
            # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
            """Build a small controlled example for the tests below."""
            # return ends this function and sends its result back to the code that called it.
            return np.zeros((8, 5)), {}

    # = stores a result in sequences; build a list, keeping its items in order.
    sequences = []
    # for takes each item from range(2) in turn; _ holds the current item.
    for _ in range(2):
        # = stores a result in action_rng; call np.random.default_rng using parentheses to supply its inputs.
        action_rng = np.random.default_rng(19)
        # = stores a result in density_rng; call np.random.default_rng using parentheses to supply its inputs.
        density_rng = np.random.default_rng(np.random.SeedSequence([19, 8675309]))
        # = stores a result in env; call Fake using parentheses to supply its inputs.
        env = Fake()
        # Call sequences.append using parentheses to supply its inputs; this line does not store a separate return value.
        sequences.append(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            [
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                reset_training_episode(env, density_rng, [1.0, 1.5, 2.0])[2]
                # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                for _ in range(100)
            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
            ]
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
        # Call np.testing.assert_array_equal using parentheses to supply its inputs; this line does not store a separate return value.
        np.testing.assert_array_equal(
            # Inside this continued call, name=value supplies an option by name rather than by its position.
            action_rng.integers(100, size=10),
            # Inside this continued call, name=value supplies an option by name rather than by its position.
            np.random.default_rng(19).integers(100, size=10),
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert sequences[0] == sequences[1] and set(sequences[0]) == {1.0, 1.5, 2.0}
