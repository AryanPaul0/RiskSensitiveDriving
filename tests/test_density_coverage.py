"""Tests for density coverage.

Tests use deliberately small examples so mistakes in learning, data pairing or
evaluation show up before spending time on long training runs.
"""

import numpy as np  # Make numpy available for this file's calculations and workflow.
import pytest  # Make pytest available for this file's calculations and workflow.
from driving_rl.envs import make_env, reset_training_episode  # Make make_env, reset_training_episode available for this file's calculations and workflow.


def test_single_density_reproduces_original_environment_trajectory():  # Check that single density reproduces original environment trajectory.
    """Check that single density reproduces original environment trajectory.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    a, b = make_env(), make_env()  # Set a, b to a highway simulator configured with, a highway simulator configured with.
    try:  # Run this work with the error handling or resource cleanup specified below.
        x, _ = a.reset(seed=72)  # Set x, the unused result to the result of a.reset.
        y, _, density = reset_training_episode(b, np.random.default_rng(8), [1.0], 72)  # Set y, the unused result, the traffic-density setting to a fresh driving episode and sampled density from b, a separate reproducible random generator seeded with 8, 1.0, 72.
        np.testing.assert_array_equal(x, y)  # Np.testing.assert array equal applied to x, y.
        assert density == 1.0  # Check that the traffic-density setting equals 1.0; fail this test if it is not true.
        for action in [1, 3, 0, 1, 4]:  # Process each action from 1, 3, 0, 1, 4.
            x, r, t, u, _ = a.step(action)  # Set x, r, t, u, the unused result to a.step applied to action.
            y, s, v, w, _ = b.step(action)  # Set y, s, v, w, the unused result to b.step applied to action.
            np.testing.assert_array_equal(x, y)  # Np.testing.assert array equal applied to x, y.
            assert (r, t, u) == (s, v, w)  # Check that r, t, u equals s, v, w; fail this test if it is not true.
            if t or u:  # Use this branch when t or u.
                break  # Stop this loop now that its stopping condition has been reached.
    finally:  # Always release resources here, including after an error or interruption.
        a.close()  # The result of a.close.
        b.close()  # The result of b.close.


def test_density_is_applied_before_reset_and_changes_traffic_spacing():  # Check that density is applied before reset and changes traffic spacing.
    """Check that density is applied before reset and changes traffic spacing.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    env = make_env()  # Set env to a highway simulator configured with.
    try:  # Run this work with the error handling or resource cleanup specified below.
        spans = []  # Set spans to .
        for density in [1.0, 2.0]:  # Process each the traffic-density setting from 1.0, 2.0.
            obs, _, selected = reset_training_episode(  # Set obs, the unused result, the selected next-action return predictions to a fresh driving episode and sampled density from env, a separate reproducible random generator seeded with 2, the traffic-density setting, 47.
                env, np.random.default_rng(2), [density], 47  # Include env, a separate reproducible random generator seeded with 2, the traffic-density setting, 47 in the surrounding operation.
            )  # Finish assembling reset training episode inputs.
            assert (  # Check that the selected next-action return predictions equals the traffic-density setting and the entries of env.unwrapped.config selected by 'vehicles_density' equals the traffic-density setting; fail this test if it is not true.
                selected == density  # Include the selected next-action return predictions equals the traffic-density setting and the entries of env.unwrapped.config selected by 'vehicles_density' equals the traffic-density setting in the surrounding operation.
                and env.unwrapped.config["vehicles_density"] == density  # Include the entries of env.unwrapped.config selected by 'vehicles_density' equals the traffic-density setting in the surrounding operation.
            )  # Finish assembling the grouped values above.
            assert obs.shape == (8, 5)  # Check that obs.shape equals 8, 5; fail this test if it is not true.
            spans.append(np.ptp([v.position[0] for v in env.unwrapped.road.vehicles]))  # Add np.ptp applied to the entries of v.position selected by 0 for each v in env.unwrapped.road.vehicles to spans.
        assert spans[1] < spans[0]  # Check that the entries of spans selected by 1 is below the entries of spans selected by 0; fail this test if it is not true.
        with pytest.raises(ValueError):  # An invalid input is supposed to fail here; accepting it would hide a bug.
            reset_training_episode(env, np.random.default_rng(2), [float("nan")])  # A fresh driving episode and sampled density from env, a separate reproducible random generator seeded with 2, 'nan' as a floating-point value.
    finally:  # Always release resources here, including after an error or interruption.
        env.close()  # The result of env.close.


def test_mixture_schedule_is_reproducible_and_does_not_consume_action_rng():  # Check that mixture schedule is reproducible and does not consume action rng.
    """Check that mixture schedule is reproducible and does not consume action rng.
    This test protects an assumption that the experiment or its interpretation relies on.
    """

    class Fake:  # Group the state and operations for Fake.
        """Build a small controlled example for the tests below."""  # Describe this operation for readers and help().

        def __init__(self):  # Build a small controlled example for the tests below.
            """Build a small controlled example for the tests below."""  # Describe this operation for readers and help().
            self.unwrapped = self  # Set this learner's unwrapped to this object.
            self.config = {"vehicles_density": 1.0}  # Set this learner's config to the named settings or measurements assembled below.

        def configure(self, c):  # Build a small controlled example for the tests below.
            """Build a small controlled example for the tests below."""  # Describe this operation for readers and help().
            self.config.update(c)  # This learner's config.update applied to c.

        def reset(self, seed=None):  # Build a small controlled example for the tests below.
            """Build a small controlled example for the tests below."""  # Describe this operation for readers and help().
            return np.zeros((8, 5)), {}  # Return np.zeros applied to 8, 5, the named settings or measurements assembled below.

    sequences = []  # Set sequences to .
    for _ in range(2):  # Process each the unused result from range applied to 2.
        action_rng = np.random.default_rng(19)  # Set action rng to a separate reproducible random generator seeded with 19.
        density_rng = np.random.default_rng(np.random.SeedSequence([19, 8675309]))  # Set density rng to a separate reproducible random generator seeded with np.random.SeedSequence applied to 19, 8675309.
        env = Fake()  # Set env to the result of Fake.
        sequences.append(  # Add the entries of a fresh driving episode and sampled density from env, density rng, [1.0, 1.5, 2.0] selected by 2 for each the unused result in range applied to 100 to sequences.
            [  # Include the entries of a fresh driving episode and sampled density from env, density rng, 1.0, 1.5, 2.0 selected by 2 for each the unused result in range applied to 100 in the surrounding operation.
                reset_training_episode(env, density_rng, [1.0, 1.5, 2.0])[2]  # Include the entries of a fresh driving episode and sampled density from env, density rng, 1.0, 1.5, 2.0 selected by 2 in the surrounding operation.
                for _ in range(100)  # Include the unused result, range applied to 100 in the surrounding operation.
            ]  # Finish assembling sequences.append inputs.
        )  # Finish assembling sequences.append inputs.
        np.testing.assert_array_equal(  # Np.testing.assert array equal applied to action rng.integers applied to 100, np.random.default rng(19).integers applied to 100.
            action_rng.integers(100, size=10),  # Include action rng.integers applied to 100 in the surrounding operation.
            np.random.default_rng(19).integers(100, size=10),  # Include np.random.default rng(19).integers applied to 100 in the surrounding operation.
        )  # Finish assembling np.testing.assert array equal inputs.
    assert sequences[0] == sequences[1] and set(sequences[0]) == {1.0, 1.5, 2.0}  # Check that the entries of sequences selected by 0 equals the entries of sequences selected by 1 and set applied to the entries of sequences selected by 0 equals 1.0, 1.5, 2.0; fail this test if it is not true.
