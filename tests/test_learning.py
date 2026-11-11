import numpy as np
import pytest
import torch

from driving_rl.agents import (Agent, ReplayBuffer, action_scores,
                               bellman_targets, quantile_huber_loss)
from driving_rl.envs import flatten, make_env, seed_everything


def test_terminal_target_and_time_limit_bootstrap():
    rewards = torch.tensor([2., 2.])
    # First transition is a crash; second is a time-limit truncation.
    terminated = torch.tensor([1., 0.])
    target = bellman_targets(rewards, terminated, torch.tensor([10., 10.]), .9)
    torch.testing.assert_close(target, torch.tensor([2., 11.]))


def test_lower_tail_changes_decision_and_handles_quantile_crossing():
    values = torch.tensor([[[10., -4., 10., 10.], [2., 2., 2., 2.]]])
    assert action_scores(values).argmax(1).item() == 0
    assert action_scores(values, .25).argmax(1).item() == 1
    with pytest.raises(ValueError):
        action_scores(values, 0)


def test_quantile_loss_matches_hand_computation():
    # Taus .25,.75; deltas +1,-1; Huber=.5; each asymmetric weight=.25.
    predicted = torch.tensor([[0., 2.]], requires_grad=True)
    target = torch.tensor([[1., 1.]])
    loss = quantile_huber_loss(predicted, target)
    torch.testing.assert_close(loss, torch.tensor(.25))
    loss.backward()
    torch.testing.assert_close(predicted.grad, torch.tensor([[-.25, .25]]))


@pytest.mark.parametrize("algorithm", ["dqn", "qrdqn"])
def test_update_improves_terminal_reward_prediction_and_freezes_target(algorithm):
    seed_everything(3)
    agent = Agent(4, 2, algorithm, learning_rate=.005)
    observations = torch.zeros(16, 4)
    batch = (observations, torch.zeros(16, dtype=torch.long), torch.ones(16),
             observations, torch.ones(16))
    target_before = {key: value.clone() for key, value in agent.target.state_dict().items()}
    before = abs(agent.online(observations)[0, 0].mean().item() - 1)
    for _ in range(50):
        loss, gradient = agent.update(batch)
        assert np.isfinite(loss) and np.isfinite(gradient)
    after = abs(agent.online(observations)[0, 0].mean().item() - 1)
    assert after < before * .3
    for key, value in agent.target.state_dict().items():
        torch.testing.assert_close(value, target_before[key])
    assert all(parameter.grad is None for parameter in agent.target.parameters())


def test_seeded_environment_and_replay_ownership():
    env = make_env()
    first, _ = env.reset(seed=42)
    second, _ = env.reset(seed=42)
    np.testing.assert_array_equal(first, second)
    assert flatten(first).shape == (40,)
    buffer = ReplayBuffer(4)
    observation = flatten(first)
    buffer.add(observation, 1, 1., observation, False)
    observation[:] = 999
    sampled = buffer.sample(1, np.random.default_rng(0))
    assert not (sampled[0] == 999).any()
    env.close()


def test_cli_training_checkpoint_and_evaluation_roundtrip(tmp_path, monkeypatch):
    import json
    from driving_rl.cli import main
    run = tmp_path / "run"
    monkeypatch.setattr("sys.argv", ["driving-rl", "train", "--algorithm", "qrdqn",
                        "--steps", "8", "--learning-starts", "4", "--batch-size", "4",
                        "--checkpoint-interval", "4", "--collision-penalty", "-8", "--out", str(run)])
    main()
    assert json.loads((run / "summary.json").read_text())["training_complete"]
    result = tmp_path / "evaluation.json"
    monkeypatch.setattr("sys.argv", ["driving-rl", "evaluate", "--checkpoint", str(run / "final.pt"),
                        "--episodes", "2", "--densities", "1", "--out", str(result),
                        "--gif", str(tmp_path / "rollout.gif")])
    main()
    data = json.loads(result.read_text())
    assert data["provenance"]["training_steps"] == 8
    assert data["provenance"]["training_collision_penalty"] == -8
    assert data["provenance"]["evaluation_collision_penalty"] == -2
    assert len(data["episodes"]) == 2
    assert (tmp_path / "rollout.gif").is_file()
    from PIL import Image
    image = Image.open(tmp_path / "rollout.gif")
    assert image.n_frames >= 2
    assert np.asarray(image.convert("RGB")).std() > 5


def test_binomial_interval_keeps_uncertainty_at_extremes():
    from driving_rl.metrics import wilson_interval
    low, high = wilson_interval(0, 30)
    assert low == pytest.approx(0., abs=1e-12)
    assert .11 < high < .12
    mirrored_low, mirrored_high = wilson_interval(30, 30)
    assert mirrored_low == pytest.approx(1 - high)
    assert mirrored_high == pytest.approx(1.)


def test_speed_governor_decodes_observation_and_controls_acceleration():
    from driving_rl.envs import govern_action, observed_speed
    env = make_env()
    observation, _ = env.reset(seed=42)
    # Simulator speed is consulted only to validate the decoding in this test.
    assert observed_speed(env, observation) == pytest.approx(env.unwrapped.vehicle.speed, abs=1e-4)
    actions = env.unwrapped.action_type.actions_indexes
    speed = observed_speed(env, observation)
    assert govern_action(env, observation, actions["FASTER"], speed-1) == actions["SLOWER"]
    assert govern_action(env, observation, actions["FASTER"], speed+1) == actions["IDLE"]
    assert govern_action(env, observation, actions["LANE_LEFT"], speed+4) == actions["LANE_LEFT"]
    env.close()


def test_absolute_exploration_is_invariant_to_training_budget():
    from driving_rl.experiment import exploration_epsilon
    for step in [1, 5000, 12001, 20000]:
        short = exploration_epsilon(step, 20000)
        long = exploration_epsilon(step, 40000, decay_steps=12000)
        assert short == pytest.approx(long)
    assert exploration_epsilon(1, 40000, decay_steps=12000) == 1.
    assert exploration_epsilon(12001, 40000, decay_steps=12000) == pytest.approx(.05)
