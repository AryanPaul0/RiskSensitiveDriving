"""Tests for density coverage.

Tests use deliberately small examples so mistakes in learning, data pairing or
evaluation show up before spending time on long training runs.
"""

import numpy as np
import pytest
from driving_rl.envs import make_env, reset_training_episode


def test_single_density_reproduces_original_environment_trajectory():
    """Check that single density reproduces original environment trajectory.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    a, b = make_env(), make_env()
    try:
        x, _ = a.reset(seed=72)
        y, _, density = reset_training_episode(b, np.random.default_rng(8), [1.0], 72)
        np.testing.assert_array_equal(x, y)
        assert density == 1.0
        for action in [1, 3, 0, 1, 4]:
            x, r, t, u, _ = a.step(action)
            y, s, v, w, _ = b.step(action)
            np.testing.assert_array_equal(x, y)
            assert (r, t, u) == (s, v, w)
            if t or u:
                break
    finally:
        a.close()
        b.close()


def test_density_is_applied_before_reset_and_changes_traffic_spacing():
    """Check that density is applied before reset and changes traffic spacing.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    env = make_env()
    try:
        spans = []
        for density in [1.0, 2.0]:
            obs, _, selected = reset_training_episode(
                env, np.random.default_rng(2), [density], 47
            )
            assert (
                selected == density
                and env.unwrapped.config["vehicles_density"] == density
            )
            assert obs.shape == (8, 5)
            spans.append(np.ptp([v.position[0] for v in env.unwrapped.road.vehicles]))
        assert spans[1] < spans[0]
        # An invalid input is supposed to fail here; accepting it would hide a bug.
        with pytest.raises(ValueError):
            reset_training_episode(env, np.random.default_rng(2), [float("nan")])
    finally:
        env.close()


def test_mixture_schedule_is_reproducible_and_does_not_consume_action_rng():
    """Check that mixture schedule is reproducible and does not consume action rng.
    This test protects an assumption that the experiment or its interpretation relies on.
    """

    class Fake:
        """Build a small controlled example for the tests below."""

        def __init__(self):
            """Build a small controlled example for the tests below."""
            self.unwrapped = self
            self.config = {"vehicles_density": 1.0}

        def configure(self, c):
            """Build a small controlled example for the tests below."""
            self.config.update(c)

        def reset(self, seed=None):
            """Build a small controlled example for the tests below."""
            return np.zeros((8, 5)), {}

    sequences = []
    for _ in range(2):
        action_rng = np.random.default_rng(19)
        density_rng = np.random.default_rng(np.random.SeedSequence([19, 8675309]))
        env = Fake()
        sequences.append(
            [
                reset_training_episode(env, density_rng, [1.0, 1.5, 2.0])[2]
                for _ in range(100)
            ]
        )
        np.testing.assert_array_equal(
            action_rng.integers(100, size=10),
            np.random.default_rng(19).integers(100, size=10),
        )
    assert sequences[0] == sequences[1] and set(sequences[0]) == {1.0, 1.5, 2.0}
