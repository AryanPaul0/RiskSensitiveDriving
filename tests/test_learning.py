"""Tests for learning.

Tests use deliberately small examples so mistakes in learning, data pairing or
evaluation show up before spending time on long training runs.
"""

# import makes these modules available; "as" gives a module a shorter local name.
import numpy as np
# import makes these modules available; "as" gives a module a shorter local name.
import pytest
# import makes these modules available; "as" gives a module a shorter local name.
import torch

# from driving_rl.agents import brings the named tools into this file; parentheses allow several lines.
from driving_rl.agents import (
    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
    Agent,
    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
    ReplayBuffer,
    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
    action_scores,
    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
    bellman_targets,
    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
    quantile_huber_loss,
# ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
)
# from driving_rl.envs import brings the named tools into this file; parentheses allow several lines.
from driving_rl.envs import flatten, make_env, seed_everything


# def defines test_terminal_target_and_time_limit_bootstrap; parentheses list its inputs, and the colon starts its indented body.
def test_terminal_target_and_time_limit_bootstrap():
    """Check that terminal target and time limit bootstrap.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    # = stores a result in rewards; call torch.tensor using parentheses to supply its inputs.
    rewards = torch.tensor([2.0, 2.0])
    # First transition is a crash; second is a time-limit truncation.
    # = stores a result in terminated; call torch.tensor using parentheses to supply its inputs.
    terminated = torch.tensor([1.0, 0.0])
    # = stores a result in target; call bellman_targets using parentheses to supply its inputs.
    target = bellman_targets(rewards, terminated, torch.tensor([10.0, 10.0]), 0.9)
    # Call torch.testing.assert_close using parentheses to supply its inputs; this line does not store a separate return value.
    torch.testing.assert_close(target, torch.tensor([2.0, 11.0]))


# def defines test_lower_tail_changes_decision_and_handles_quantile_crossing; parentheses list its inputs, and the colon starts its indented body.
def test_lower_tail_changes_decision_and_handles_quantile_crossing():
    """Check that lower tail changes decision and handles quantile crossing.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    # = stores a result in values; call torch.tensor using parentheses to supply its inputs.
    values = torch.tensor([[[10.0, -4.0, 10.0, 10.0], [2.0, 2.0, 2.0, 2.0]]])
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert action_scores(values).argmax(1).item() == 0
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert action_scores(values, 0.25).argmax(1).item() == 1
    # An invalid input is supposed to fail here; accepting it would hide a bug.
    # with enters a temporary context and exits it afterward, even if an error interrupts the block.
    with pytest.raises(ValueError):
        # Call action_scores using parentheses to supply its inputs; this line does not store a separate return value.
        action_scores(values, 0)


# def defines test_quantile_loss_matches_hand_computation; parentheses list its inputs, and the colon starts its indented body.
def test_quantile_loss_matches_hand_computation():
    # Taus .25,.75; deltas +1,-1; Huber=.5; each asymmetric weight=.25.
    """Check that quantile loss matches hand computation.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    # = stores a result in predicted; call torch.tensor using parentheses to supply its inputs.
    predicted = torch.tensor([[0.0, 2.0]], requires_grad=True)
    # = stores a result in target; call torch.tensor using parentheses to supply its inputs.
    target = torch.tensor([[1.0, 1.0]])
    # = stores a result in loss; call quantile_huber_loss using parentheses to supply its inputs.
    loss = quantile_huber_loss(predicted, target)
    # Call torch.testing.assert_close using parentheses to supply its inputs; this line does not store a separate return value.
    torch.testing.assert_close(loss, torch.tensor(0.25))
    # backward() calculates gradients: how changing each trainable weight would change this loss.
    loss.backward()
    # Call torch.testing.assert_close using parentheses to supply its inputs; this line does not store a separate return value.
    torch.testing.assert_close(predicted.grad, torch.tensor([[-0.25, 0.25]]))


# @ applies this decorator to the next function, adding the behavior supplied by that helper.
@pytest.mark.parametrize("algorithm", ["dqn", "qrdqn"])
# def defines test_update_improves_terminal_reward_prediction_and_freezes_target; parentheses list its inputs, and the colon starts its indented body.
def test_update_improves_terminal_reward_prediction_and_freezes_target(algorithm):
    """Check that update improves terminal reward prediction and freezes target.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    # Call seed_everything using parentheses to supply its inputs; this line does not store a separate return value.
    seed_everything(3)
    # = stores a result in agent; call Agent using parentheses to supply its inputs.
    agent = Agent(4, 2, algorithm, learning_rate=0.005)
    # = stores a result in observations; call torch.zeros using parentheses to supply its inputs.
    observations = torch.zeros(16, 4)
    # = stores a result in batch; group these values into a tuple, keeping their order.
    batch = (
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        observations,
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        torch.zeros(16, dtype=torch.long),
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        torch.ones(16),
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        observations,
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        torch.ones(16),
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # = stores a result in target_before; collect or generate one result per item; the "for" part names the item being processed.
    target_before = {
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        key: value.clone() for key, value in agent.target.state_dict().items()
    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
    }
    # = stores a result in before; call abs using parentheses to supply its inputs.
    before = abs(agent.online(observations)[0, 0].mean().item() - 1)
    # for takes each item from range(50) in turn; _ holds the current item.
    for _ in range(50):
        # = unpacks the right-hand result into (loss, gradient), matching values to names in order.
        loss, gradient = agent.update(batch)
        # assert checks this condition; a false result raises AssertionError and fails this check.
        assert np.isfinite(loss) and np.isfinite(gradient)
    # = stores a result in after; call abs using parentheses to supply its inputs.
    after = abs(agent.online(observations)[0, 0].mean().item() - 1)
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert after < before * 0.3
    # for takes each item from agent.target.state_dict().items() in turn; (key, value) holds the current item.
    for key, value in agent.target.state_dict().items():
        # Call torch.testing.assert_close using parentheses to supply its inputs; this line does not store a separate return value.
        torch.testing.assert_close(value, target_before[key])
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert all(parameter.grad is None for parameter in agent.target.parameters())


# def defines test_seeded_environment_and_replay_ownership; parentheses list its inputs, and the colon starts its indented body.
def test_seeded_environment_and_replay_ownership():
    """Check that seeded environment and replay ownership.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    # = stores a result in env; call make_env using parentheses to supply its inputs.
    env = make_env()
    # = unpacks the right-hand result into (first, _), matching values to names in order.
    first, _ = env.reset(seed=42)
    # = unpacks the right-hand result into (second, _), matching values to names in order.
    second, _ = env.reset(seed=42)
    # Call np.testing.assert_array_equal using parentheses to supply its inputs; this line does not store a separate return value.
    np.testing.assert_array_equal(first, second)
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert flatten(first).shape == (40,)
    # = stores a result in buffer; call ReplayBuffer using parentheses to supply its inputs.
    buffer = ReplayBuffer(4)
    # = stores a result in observation; call flatten using parentheses to supply its inputs.
    observation = flatten(first)
    # Call buffer.add using parentheses to supply its inputs; this line does not store a separate return value.
    buffer.add(observation, 1, 1.0, observation, False)
    # = stores a result in observation[:]; use the literal value 999.
    observation[:] = 999
    # = stores a result in sampled; call buffer.sample using parentheses to supply its inputs.
    sampled = buffer.sample(1, np.random.default_rng(0))
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert not (sampled[0] == 999).any()
    # Call env.close using parentheses to supply its inputs; this line does not store a separate return value.
    env.close()


# def defines test_cli_training_checkpoint_and_evaluation_roundtrip; parentheses list its inputs, and the colon starts its indented body.
def test_cli_training_checkpoint_and_evaluation_roundtrip(tmp_path, monkeypatch):
    """Check that cli training checkpoint and evaluation roundtrip.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    # import makes these modules available; "as" gives a module a shorter local name.
    import json
    # from driving_rl.cli import brings the named tools into this file; parentheses allow several lines.
    from driving_rl.cli import main

    # = stores a result in run; apply / to these operands; numbers are divided, while Path objects join folder names.
    run = tmp_path / "run"
    # Call monkeypatch.setattr using parentheses to supply its inputs; this line does not store a separate return value.
    monkeypatch.setattr(
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "sys.argv",
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        [
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "driving-rl",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "train",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "--algorithm",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "qrdqn",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "--steps",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "8",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "--learning-starts",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "4",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "--batch-size",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "4",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "--checkpoint-interval",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "4",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "--collision-penalty",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "-8",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "--out",
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            str(run),
        # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
        ],
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # Call main using parentheses to supply its inputs; this line does not store a separate return value.
    main()
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert json.loads((run / "summary.json").read_text())["training_complete"]
    # = stores a result in result; apply / to these operands; numbers are divided, while Path objects join folder names.
    result = tmp_path / "evaluation.json"
    # Call monkeypatch.setattr using parentheses to supply its inputs; this line does not store a separate return value.
    monkeypatch.setattr(
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "sys.argv",
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        [
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "driving-rl",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "evaluate",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "--checkpoint",
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            str(run / "final.pt"),
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
            str(result),
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "--gif",
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            str(tmp_path / "rollout.gif"),
        # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
        ],
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # Call main using parentheses to supply its inputs; this line does not store a separate return value.
    main()
    # = stores a result in data; call json.loads using parentheses to supply its inputs.
    data = json.loads(result.read_text())
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert data["provenance"]["training_steps"] == 8
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert data["provenance"]["training_collision_penalty"] == -8
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert data["provenance"]["evaluation_collision_penalty"] == -2
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert len(data["episodes"]) == 2
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert (tmp_path / "rollout.gif").is_file()
    # from PIL import brings the named tools into this file; parentheses allow several lines.
    from PIL import Image

    # = stores a result in image; call Image.open using parentheses to supply its inputs.
    image = Image.open(tmp_path / "rollout.gif")
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert image.n_frames >= 2
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert np.asarray(image.convert("RGB")).std() > 5


# def defines test_binomial_interval_keeps_uncertainty_at_extremes; parentheses list its inputs, and the colon starts its indented body.
def test_binomial_interval_keeps_uncertainty_at_extremes():
    """Check that binomial interval keeps uncertainty at extremes.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    # from driving_rl.metrics import brings the named tools into this file; parentheses allow several lines.
    from driving_rl.metrics import wilson_interval

    # = unpacks the right-hand result into (low, high), matching values to names in order.
    low, high = wilson_interval(0, 30)
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert low == pytest.approx(0.0, abs=1e-12)
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert 0.11 < high < 0.12
    # = unpacks the right-hand result into (mirrored_low, mirrored_high), matching values to names in order.
    mirrored_low, mirrored_high = wilson_interval(30, 30)
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert mirrored_low == pytest.approx(1 - high)
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert mirrored_high == pytest.approx(1.0)


# def defines test_speed_governor_decodes_observation_and_controls_acceleration; parentheses list its inputs, and the colon starts its indented body.
def test_speed_governor_decodes_observation_and_controls_acceleration():
    """Check that speed governor decodes observation and controls acceleration.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    # from driving_rl.envs import brings the named tools into this file; parentheses allow several lines.
    from driving_rl.envs import govern_action, observed_speed

    # = stores a result in env; call make_env using parentheses to supply its inputs.
    env = make_env()
    # = unpacks the right-hand result into (observation, _), matching values to names in order.
    observation, _ = env.reset(seed=42)
    # Simulator speed is consulted only to validate the decoding in this test.
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert observed_speed(env, observation) == pytest.approx(
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        env.unwrapped.vehicle.speed, abs=1e-4
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # = stores a result in actions; use a dot to access actions_indexes on env.unwrapped.action_type.
    actions = env.unwrapped.action_type.actions_indexes
    # = stores a result in speed; call observed_speed using parentheses to supply its inputs.
    speed = observed_speed(env, observation)
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert (
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        govern_action(env, observation, actions["FASTER"], speed - 1)
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        == actions["SLOWER"]
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert (
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        govern_action(env, observation, actions["FASTER"], speed + 1) == actions["IDLE"]
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert (
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        govern_action(env, observation, actions["LANE_LEFT"], speed + 4)
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        == actions["LANE_LEFT"]
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # Call env.close using parentheses to supply its inputs; this line does not store a separate return value.
    env.close()


# def defines test_absolute_exploration_is_invariant_to_training_budget; parentheses list its inputs, and the colon starts its indented body.
def test_absolute_exploration_is_invariant_to_training_budget():
    """Check that absolute exploration is invariant to training budget.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    # from driving_rl.experiment import brings the named tools into this file; parentheses allow several lines.
    from driving_rl.experiment import exploration_epsilon

    # for takes each item from [1, 5000, 12001, 20000] in turn; step holds the current item.
    for step in [1, 5000, 12001, 20000]:
        # = stores a result in short; call exploration_epsilon using parentheses to supply its inputs.
        short = exploration_epsilon(step, 20000)
        # = stores a result in long; call exploration_epsilon using parentheses to supply its inputs.
        long = exploration_epsilon(step, 40000, decay_steps=12000)
        # assert checks this condition; a false result raises AssertionError and fails this check.
        assert short == pytest.approx(long)
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert exploration_epsilon(1, 40000, decay_steps=12000) == 1.0
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert exploration_epsilon(12001, 40000, decay_steps=12000) == pytest.approx(0.05)
