"""Tests for calibration.

Tests use deliberately small examples so mistakes in learning, data pairing or
evaluation show up before spending time on long training runs.
"""

# import makes these modules available; "as" gives a module a shorter local name.
import numpy as np
# import makes these modules available; "as" gives a module a shorter local name.
import pytest

# from driving_rl.calibration import brings the named tools into this file; parentheses allow several lines.
from driving_rl.calibration import discounted_return_bounds, quantile_compatibility


# def defines test_terminal_return_is_exact_but_cutoff_retains_continuation; parentheses list its inputs, and the colon starts its indented body.
def test_terminal_return_is_exact_but_cutoff_retains_continuation():
    """Check that terminal return is exact but cutoff retains continuation.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    # = unpacks the right-hand result into (low, high), matching values to names in order.
    low, high = discounted_return_bounds([0.4, -2.0], 0.9, True)
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert low == high == pytest.approx(-1.4)
    # = unpacks the right-hand result into (low, high), matching values to names in order.
    low, high = discounted_return_bounds([0.4], 0.9, False)
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert low == pytest.approx(0.4 - 0.9 * 2)
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert high == pytest.approx(0.4 + 0.9 * 4)


# def defines test_time_limit_bounds_contain_concrete_safe_and_crash_continuations; parentheses list its inputs, and the colon starts its indented body.
def test_time_limit_bounds_contain_concrete_safe_and_crash_continuations():
    """Check that time limit bounds contain concrete safe and crash continuations.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    # = stores a result in rewards; multiply the operands; NumPy and PyTorch arrays do this element by element.
    rewards = [0.12] * 30
    # = unpacks the right-hand result into (low, high), matching values to names in order.
    low, high = discounted_return_bounds(rewards, 0.9, False)
    # = stores a result in prefix; call sum using parentheses to supply its inputs.
    prefix = sum(0.9**i * value for i, value in enumerate(rewards))
    # = stores a result in crash; combine the operands with the displayed operator; parentheses control grouping.
    crash = prefix + 0.9**30 * -2
    # = stores a result in safe; combine the operands with the displayed operator; parentheses control grouping.
    safe = prefix + 0.9**30 * 4
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert low <= crash <= high
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert low <= safe <= high
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert high - low == pytest.approx(6 * 0.9**30)


# def defines test_point_mass_ties_are_not_falsely_declared_miscalibrated; parentheses list its inputs, and the colon starts its indented body.
def test_point_mass_ties_are_not_falsely_declared_miscalibrated():
    """Check that point mass ties are not falsely declared miscalibrated.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    # = stores a result in result; call quantile_compatibility using parentheses to supply its inputs.
    result = quantile_compatibility(np.full((20, 4), 2.0), [2.0] * 20, [2.0] * 20)
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert result["coverage_lower"] == [0.0] * 4
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert result["coverage_upper"] == [1.0] * 4
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert result["minimum_mean_absolute_compatibility_gap"] == 0


# def defines test_censoring_widens_compatibility_without_imputing_success_returns; parentheses list its inputs, and the colon starts its indented body.
def test_censoring_widens_compatibility_without_imputing_success_returns():
    """Check that censoring widens compatibility without imputing success returns.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    # = stores a result in result; call quantile_compatibility using parentheses to supply its inputs.
    result = quantile_compatibility([[0.0, 2.0, 4.0]], [1.0], [3.0])
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert result["coverage_lower"] == [0.0, 0.0, 1.0]
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert result["coverage_upper"] == [0.0, 1.0, 1.0]
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert result["minimum_mean_absolute_compatibility_gap"] > 0


# def defines test_invalid_return_intervals_are_rejected; parentheses list its inputs, and the colon starts its indented body.
def test_invalid_return_intervals_are_rejected():
    """Check that invalid return intervals are rejected.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    # An invalid input is supposed to fail here; accepting it would hide a bug.
    # with enters a temporary context and exits it afterward, even if an error interrupts the block.
    with pytest.raises(ValueError):
        # Call quantile_compatibility using parentheses to supply its inputs; this line does not store a separate return value.
        quantile_compatibility([[1.0, 2.0]], [3.0], [1.0])
    # An invalid input is supposed to fail here; accepting it would hide a bug.
    # with enters a temporary context and exits it afterward, even if an error interrupts the block.
    with pytest.raises(ValueError):
        # Call quantile_compatibility using parentheses to supply its inputs; this line does not store a separate return value.
        quantile_compatibility([[1.0, 2.0]], [float("nan")], [3.0])
    # An invalid input is supposed to fail here; accepting it would hide a bug.
    # with enters a temporary context and exits it afterward, even if an error interrupts the block.
    with pytest.raises(ValueError):
        # Call quantile_compatibility using parentheses to supply its inputs; this line does not store a separate return value.
        quantile_compatibility([[1.0, 2.0]], [[0.0]], [[3.0]])


# def defines test_calibration_collector_retains_survivors_and_provenance; parentheses list its inputs, and the colon starts its indented body.
def test_calibration_collector_retains_survivors_and_provenance(tmp_path, monkeypatch):
    # Force IDLE in a traffic-free real simulator to exercise censoring without
    # depending on a successful trained policy or random collision outcomes.
    """Check that calibration collector retains survivors and provenance.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    # from driving_rl.agents import brings the named tools into this file; parentheses allow several lines.
    from driving_rl.agents import Agent
    # from driving_rl.experiment import brings the named tools into this file; parentheses allow several lines.
    from driving_rl.experiment import save_checkpoint
    # from driving_rl.calibration import brings the named tools into this file; parentheses allow several lines.
    from driving_rl.calibration import main
    # import makes these modules available; "as" gives a module a shorter local name.
    import json
    # import makes these modules available; "as" gives a module a shorter local name.
    import torch

    # = stores a result in agent; call Agent using parentheses to supply its inputs.
    agent = Agent(40, 5, "qrdqn")
    # for takes each item from agent.online.parameters() in turn; parameter holds the current item.
    for parameter in agent.online.parameters():
        # with temporarily disables gradient tracking for this block; these predictions are fixed answers, not a second training path.
        with torch.no_grad():
            # Call parameter.zero_ using parentheses to supply its inputs; this line does not store a separate return value.
            parameter.zero_()
    # with temporarily disables gradient tracking for this block; these predictions are fixed answers, not a second training path.
    with torch.no_grad():
        # = stores a result in agent.online.network[-1].bias[32:64]; use the literal value 1.0.
        agent.online.network[-1].bias[32:64] = 1.0
    # from driving_rl.envs import brings the named tools into this file; parentheses allow several lines.
    from driving_rl.envs import make_env as real_make_env

    # def defines traffic_free_env; parentheses list its inputs, and the colon starts its indented body.
    def traffic_free_env(*args, **kwargs):
        # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
        """Build a small controlled example for the tests below."""
        # = stores a result in env; call real_make_env using parentheses to supply its inputs.
        env = real_make_env(*args, **kwargs)
        # = stores a result in env.unwrapped.config['vehicles_count']; use the literal value 0.
        env.unwrapped.config["vehicles_count"] = 0
        # return ends this function and sends its result back to the code that called it.
        return env

    # Call monkeypatch.setattr using parentheses to supply its inputs; this line does not store a separate return value.
    monkeypatch.setattr("driving_rl.calibration.make_env", traffic_free_env)
    # = stores a result in config; build a dictionary, which stores values under named keys.
    config = {
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "observation_dim": 40,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "n_actions": 5,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "algorithm": "qrdqn",
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "quantiles": 32,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "learning_rate": 0.0005,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "gamma": 0.9,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "seed": 0,
    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
    }
    # = stores a result in checkpoint; apply / to these operands; numbers are divided, while Path objects join folder names.
    checkpoint = tmp_path / "test.pt"
    # Call save_checkpoint using parentheses to supply its inputs; this line does not store a separate return value.
    save_checkpoint(checkpoint, agent, config, 20000)
    # = stores a result in output; apply / to these operands; numbers are divided, while Path objects join folder names.
    output = tmp_path / "calibration.json"
    # Call monkeypatch.setattr using parentheses to supply its inputs; this line does not store a separate return value.
    monkeypatch.setattr(
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "sys.argv",
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        [
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "calibration",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "--checkpoint",
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            str(checkpoint),
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "--episodes",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "2",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "--densities",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "1",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "--out",
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            str(output),
        # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
        ],
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # Call main using parentheses to supply its inputs; this line does not store a separate return value.
    main()
    # = stores a result in result; call json.loads using parentheses to supply its inputs.
    result = json.loads(output.read_text())
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert len(result["episodes"]) == 2
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert result["provenance"]["seed_start"] == 12000
    # for takes each item from result['episodes'] in turn; row holds the current item.
    for row in result["episodes"]:
        # assert checks this condition; a false result raises AssertionError and fails this check.
        assert row["return_lower"] <= row["return_upper"]
        # assert checks this condition; a false result raises AssertionError and fails this check.
        assert len(row["initial_quantiles"]) == 32
        # assert checks this condition; a false result raises AssertionError and fails this check.
        assert row["length"] == len(row["rewards"])
        # assert checks this condition; a false result raises AssertionError and fails this check.
        assert row["censored"] and row["length"] == 30
        # IDLE at 25 m/s yields reward .2; the continuing return is 2.
        # assert checks this condition; a false result raises AssertionError and fails this check.
        assert row["return_lower"] <= 2.0 <= row["return_upper"]
