"""Tests for calibration.

Tests use deliberately small examples so mistakes in learning, data pairing or
evaluation show up before spending time on long training runs.
"""

import numpy as np  # Make numpy available for this file's calculations and workflow.
import pytest  # Make pytest available for this file's calculations and workflow.

from driving_rl.calibration import discounted_return_bounds, quantile_compatibility  # Make discounted_return_bounds, quantile_compatibility available for this file's calculations and workflow.


def test_terminal_return_is_exact_but_cutoff_retains_continuation():  # Check that terminal return is exact but cutoff retains continuation.
    """Check that terminal return is exact but cutoff retains continuation.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    low, high = discounted_return_bounds([0.4, -2.0], 0.9, True)  # Set low, high to discounted return bounds applied to 0.4, negative 2.0, 0.9, True.
    assert low == high == pytest.approx(-1.4)  # Check that low equals high equals pytest.approx applied to negative 1.4; fail this test if it is not true.
    low, high = discounted_return_bounds([0.4], 0.9, False)  # Set low, high to discounted return bounds applied to 0.4, 0.9, False.
    assert low == pytest.approx(0.4 - 0.9 * 2)  # Check that low equals pytest.approx applied to (0.4 minus (0.9 times 2)); fail this test if it is not true.
    assert high == pytest.approx(0.4 + 0.9 * 4)  # Check that high equals pytest.approx applied to (0.4 plus (0.9 times 4)); fail this test if it is not true.


def test_time_limit_bounds_contain_concrete_safe_and_crash_continuations():  # Check that time limit bounds contain concrete safe and crash continuations.
    """Check that time limit bounds contain concrete safe and crash continuations.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    rewards = [0.12] * 30  # Set the observed rewards to (0.12 times 30).
    low, high = discounted_return_bounds(rewards, 0.9, False)  # Set low, high to discounted return bounds applied to the observed rewards, 0.9, False.
    prefix = sum(0.9**i * value for i, value in enumerate(rewards))  # Set prefix to sum applied to ((0.9 to the power i) times value) for each i, value in enumerate applied to the observed rewards.
    crash = prefix + 0.9**30 * -2  # Set crash to (prefix plus ((0.9 to the power 30) times negative 2)).
    safe = prefix + 0.9**30 * 4  # Set safe to (prefix plus ((0.9 to the power 30) times 4)).
    assert low <= crash <= high  # Check that low is at most crash is at most high; fail this test if it is not true.
    assert low <= safe <= high  # Check that low is at most safe is at most high; fail this test if it is not true.
    assert high - low == pytest.approx(6 * 0.9**30)  # Check that (high minus low) equals pytest.approx applied to (6 times (0.9 to the power 30)); fail this test if it is not true.


def test_point_mass_ties_are_not_falsely_declared_miscalibrated():  # Check that point mass ties are not falsely declared miscalibrated.
    """Check that point mass ties are not falsely declared miscalibrated.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    result = quantile_compatibility(np.full((20, 4), 2.0), [2.0] * 20, [2.0] * 20)  # Set result to quantile compatibility applied to np.full applied to 20, 4, 2.0, (2.0 times 20), (2.0 times 20).
    assert result["coverage_lower"] == [0.0] * 4  # Check that the entries of result selected by 'coverage_lower' equals (0.0 times 4); fail this test if it is not true.
    assert result["coverage_upper"] == [1.0] * 4  # Check that the entries of result selected by 'coverage_upper' equals (1.0 times 4); fail this test if it is not true.
    assert result["minimum_mean_absolute_compatibility_gap"] == 0  # Check that the entries of result selected by 'minimum_mean_absolute_compatibility_gap' equals 0; fail this test if it is not true.


def test_censoring_widens_compatibility_without_imputing_success_returns():  # Check that censoring widens compatibility without imputing success returns.
    """Check that censoring widens compatibility without imputing success returns.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    result = quantile_compatibility([[0.0, 2.0, 4.0]], [1.0], [3.0])  # Set result to quantile compatibility applied to 0.0, 2.0, 4.0, 1.0, 3.0.
    assert result["coverage_lower"] == [0.0, 0.0, 1.0]  # Check that the entries of result selected by 'coverage_lower' equals 0.0, 0.0, 1.0; fail this test if it is not true.
    assert result["coverage_upper"] == [0.0, 1.0, 1.0]  # Check that the entries of result selected by 'coverage_upper' equals 0.0, 1.0, 1.0; fail this test if it is not true.
    assert result["minimum_mean_absolute_compatibility_gap"] > 0  # Check that the entries of result selected by 'minimum_mean_absolute_compatibility_gap' is above 0; fail this test if it is not true.


def test_invalid_return_intervals_are_rejected():  # Check that invalid return intervals are rejected.
    """Check that invalid return intervals are rejected.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    with pytest.raises(ValueError):  # An invalid input is supposed to fail here; accepting it would hide a bug.
        quantile_compatibility([[1.0, 2.0]], [3.0], [1.0])  # Quantile compatibility applied to 1.0, 2.0, 3.0, 1.0.
    with pytest.raises(ValueError):  # An invalid input is supposed to fail here; accepting it would hide a bug.
        quantile_compatibility([[1.0, 2.0]], [float("nan")], [3.0])  # Quantile compatibility applied to 1.0, 2.0, 'nan' as a floating-point value, 3.0.
    with pytest.raises(ValueError):  # An invalid input is supposed to fail here; accepting it would hide a bug.
        quantile_compatibility([[1.0, 2.0]], [[0.0]], [[3.0]])  # Quantile compatibility applied to 1.0, 2.0, 0.0, 3.0.


def test_calibration_collector_retains_survivors_and_provenance(tmp_path, monkeypatch):  # Check that calibration collector retains survivors and provenance. Force IDLE in a traffic-free real simulator to exercise censoring without depending on a successful trained policy or random collision outcomes.
    """Check that calibration collector retains survivors and provenance.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    from driving_rl.agents import Agent  # Make Agent available for this file's calculations and workflow.
    from driving_rl.experiment import save_checkpoint  # Make save_checkpoint available for this file's calculations and workflow.
    from driving_rl.calibration import main  # Make main available for this file's calculations and workflow.
    import json  # Make json available for this file's calculations and workflow.
    import torch  # Make torch available for this file's calculations and workflow.

    agent = Agent(40, 5, "qrdqn")  # Set agent to Agent applied to 40, 5, 'qrdqn'.
    for parameter in agent.online.parameters():  # Process each parameter from the result of agent.online.parameters.
        with torch.no_grad():  # Treat predictions in this block as fixed learning targets or inference results; do not backpropagate through them.
            parameter.zero_()  # The result of parameter.zero .
    with torch.no_grad():  # Treat predictions in this block as fixed learning targets or inference results; do not backpropagate through them.
        agent.online.network[-1].bias[32:64] = 1.0  # Set the entries of agent.online.network[-1].bias selected by 32:64 to 1.0.
    from driving_rl.envs import make_env as real_make_env  # Make make_env available for this file's calculations and workflow.

    def traffic_free_env(*args, **kwargs):  # Build a small controlled example for the tests below.
        """Build a small controlled example for the tests below."""  # Describe this operation for readers and help().
        env = real_make_env(*args, **kwargs)  # Set env to real make env applied to *args.
        env.unwrapped.config["vehicles_count"] = 0  # Set the entries of env.unwrapped.config selected by 'vehicles_count' to 0.
        return env  # Return env.

    monkeypatch.setattr("driving_rl.calibration.make_env", traffic_free_env)  # Monkeypatch.setattr applied to 'driving_rl.calibration.make_env', traffic free env.
    config = {  # Set config to the named settings or measurements assembled below.
        "observation_dim": 40,  # Record 'observation_dim' as 40.
        "n_actions": 5,  # Record 'n_actions' as 5.
        "algorithm": "qrdqn",  # Record 'algorithm' as 'qrdqn'.
        "quantiles": 32,  # Record 'quantiles' as 32.
        "learning_rate": 0.0005,  # Record 'learning_rate' as 0.0005.
        "gamma": 0.9,  # Record 'gamma' as 0.9.
        "seed": 0,  # Record 'seed' as 0.
    }  # Finish assembling the grouped values above.
    checkpoint = tmp_path / "test.pt"  # Set checkpoint to the path to 'test.pt' inside tmp path.
    save_checkpoint(checkpoint, agent, config, 20000)  # Save checkpoint applied to checkpoint, agent, config, 20000.
    output = tmp_path / "calibration.json"  # Set output to the path to 'calibration.json' inside tmp path.
    monkeypatch.setattr(  # Monkeypatch.setattr applied to 'sys.argv', the listed settings or measurements in their recorded order.
        "sys.argv",  # Include 'sys.argv' in the surrounding operation.
        [  # Include the listed settings or measurements in their recorded order in the surrounding operation.
            "calibration",  # Include 'calibration' in the surrounding operation.
            "--checkpoint",  # Include '--checkpoint' in the surrounding operation.
            str(checkpoint),  # Include checkpoint as a text value in the surrounding operation.
            "--episodes",  # Include '--episodes' in the surrounding operation.
            "2",  # Include '2' in the surrounding operation.
            "--densities",  # Include '--densities' in the surrounding operation.
            "1",  # Include '1' in the surrounding operation.
            "--out",  # Include '--out' in the surrounding operation.
            str(output),  # Include output as a text value in the surrounding operation.
        ],  # Finish assembling the grouped values above.
    )  # Finish assembling monkeypatch.setattr inputs.
    main()  # The result of main.
    result = json.loads(output.read_text())  # Set result to json.loads applied to the result of output.read text.
    assert len(result["episodes"]) == 2  # Check that the number of entries in the entries of result selected by 'episodes' equals 2; fail this test if it is not true.
    assert result["provenance"]["seed_start"] == 12000  # Check that the entries of the entries of result selected by 'provenance' selected by 'seed_start' equals 12000; fail this test if it is not true.
    for row in result["episodes"]:  # Process each row from the entries of result selected by 'episodes'.
        assert row["return_lower"] <= row["return_upper"]  # Check that the entries of row selected by 'return_lower' is at most the entries of row selected by 'return_upper'; fail this test if it is not true.
        assert len(row["initial_quantiles"]) == 32  # Check that the number of entries in the entries of row selected by 'initial_quantiles' equals 32; fail this test if it is not true.
        assert row["length"] == len(row["rewards"])  # Check that the entries of row selected by 'length' equals the number of entries in the entries of row selected by 'rewards'; fail this test if it is not true.
        assert row["censored"] and row["length"] == 30  # Check that the entries of row selected by 'censored' and the entries of row selected by 'length' equals 30; fail this test if it is not true.
        assert row["return_lower"] <= 2.0 <= row["return_upper"]  # IDLE at 25 m/s yields reward .2; the continuing return is 2.
