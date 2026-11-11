"""Tests for learning.

Tests use deliberately small examples so mistakes in learning, data pairing or
evaluation show up before spending time on long training runs.
"""

import numpy as np  # Make numpy available for this file's calculations and workflow.
import pytest  # Make pytest available for this file's calculations and workflow.
import torch  # Make torch available for this file's calculations and workflow.

from driving_rl.agents import (  # Make Agent, ReplayBuffer, action_scores, bellman_targets, quantile_huber_loss available for this file's calculations and workflow.
    Agent,  # Continue the calculation or input list started above.
    ReplayBuffer,  # Continue the calculation or input list started above.
    action_scores,  # Continue the calculation or input list started above.
    bellman_targets,  # Continue the calculation or input list started above.
    quantile_huber_loss,  # Continue the calculation or input list started above.
)  # Finish assembling the grouped values above.
from driving_rl.envs import flatten, make_env, seed_everything  # Make flatten, make_env, seed_everything available for this file's calculations and workflow.


def test_terminal_target_and_time_limit_bootstrap():  # Check that terminal target and time limit bootstrap.
    """Check that terminal target and time limit bootstrap.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    rewards = torch.tensor([2.0, 2.0])  # Set the observed rewards to 2.0, 2.0 as a numerical array/tensor for batch calculations.
    terminated = torch.tensor([1.0, 0.0])  # First transition is a crash; second is a time-limit truncation.
    target = bellman_targets(rewards, terminated, torch.tensor([10.0, 10.0]), 0.9)  # Set target to bellman targets applied to the observed rewards, the true-terminal flags, 10.0, 10.0 as a numerical array/tensor for batch calculations, 0.9.
    torch.testing.assert_close(target, torch.tensor([2.0, 11.0]))  # Torch.testing.assert close applied to target, 2.0, 11.0 as a numerical array/tensor for batch calculations.


def test_lower_tail_changes_decision_and_handles_quantile_crossing():  # Check that lower tail changes decision and handles quantile crossing.
    """Check that lower tail changes decision and handles quantile crossing.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    values = torch.tensor([[[10.0, -4.0, 10.0, 10.0], [2.0, 2.0, 2.0, 2.0]]])  # Set values to 10.0, -4.0, 10.0, 10.0, 2.0, 2.0, 2.0, 2.0 as a numerical array/tensor for batch calculations.
    assert action_scores(values).argmax(1).item() == 0  # Check that the index of the highest score in the mean or lower-tail action scores computed from values extracted as a plain scalar equals 0; fail this test if it is not true.
    assert action_scores(values, 0.25).argmax(1).item() == 1  # Check that the index of the highest score in the mean or lower-tail action scores computed from values, 0.25 extracted as a plain scalar equals 1; fail this test if it is not true.
    with pytest.raises(ValueError):  # An invalid input is supposed to fail here; accepting it would hide a bug.
        action_scores(values, 0)  # The mean or lower-tail action scores computed from values, 0.


def test_quantile_loss_matches_hand_computation():  # Check that quantile loss matches hand computation. Taus .25,.75; deltas +1,-1; Huber=.5; each asymmetric weight=.25.
    """Check that quantile loss matches hand computation.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    predicted = torch.tensor([[0.0, 2.0]], requires_grad=True)  # Set the recorded actions' predicted return quantiles to 0.0, 2.0 as a numerical array/tensor for batch calculations.
    target = torch.tensor([[1.0, 1.0]])  # Set target to 1.0, 1.0 as a numerical array/tensor for batch calculations.
    loss = quantile_huber_loss(predicted, target)  # Set loss to quantile huber loss applied to the recorded actions' predicted return quantiles, target.
    torch.testing.assert_close(loss, torch.tensor(0.25))  # Torch.testing.assert close applied to loss, 0.25 as a numerical array/tensor for batch calculations.
    loss.backward()  # Calculate how each online-network weight affects prediction error; these gradients guide learning.
    torch.testing.assert_close(predicted.grad, torch.tensor([[-0.25, 0.25]]))  # Torch.testing.assert close applied to predicted.grad, -0.25, 0.25 as a numerical array/tensor for batch calculations.


@pytest.mark.parametrize("algorithm", ["dqn", "qrdqn"])  # Include pytest.mark.parametrize applied to 'algorithm', 'dqn', 'qrdqn' in the surrounding operation.
def test_update_improves_terminal_reward_prediction_and_freezes_target(algorithm):  # Check that update improves terminal reward prediction and freezes target.
    """Check that update improves terminal reward prediction and freezes target.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    seed_everything(3)  # Initialize repeatable random streams using 3.
    agent = Agent(4, 2, algorithm, learning_rate=0.005)  # Set agent to Agent applied to 4, 2, algorithm.
    observations = torch.zeros(16, 4)  # Set observations to torch.zeros applied to 16, 4.
    batch = (  # Set batch to observations, torch.zeros applied to 16, torch.ones applied to 16, observations, torch.ones applied to 16.
        observations,  # Include observations in the surrounding operation.
        torch.zeros(16, dtype=torch.long),  # Include torch.zeros applied to 16 in the surrounding operation.
        torch.ones(16),  # Include torch.ones applied to 16 in the surrounding operation.
        observations,  # Include observations in the surrounding operation.
        torch.ones(16),  # Include torch.ones applied to 16 in the surrounding operation.
    )  # Finish assembling the grouped values above.
    target_before = {  # Set target before to the result of value.clone for each key, value in the result of agent.target.state dict().items.
        key: value.clone() for key, value in agent.target.state_dict().items()  # Include key, the result of value.clone, key, value, the result of agent.target.state dict().items in the surrounding operation.
    }  # Finish assembling the grouped values above.
    before = abs(agent.online(observations)[0, 0].mean().item() - 1)  # Set before to abs applied to (the average of agent.online(observations)[0, 0] extracted as a plain scalar minus 1).
    for _ in range(50):  # Process each the unused result from range applied to 50.
        loss, gradient = agent.update(batch)  # Set loss, gradient to agent.update applied to batch.
        assert np.isfinite(loss) and np.isfinite(gradient)  # Check that a true/false check for finite numerical values in loss and a true/false check for finite numerical values in gradient; fail this test if it is not true.
    after = abs(agent.online(observations)[0, 0].mean().item() - 1)  # Set after to abs applied to (the average of agent.online(observations)[0, 0] extracted as a plain scalar minus 1).
    assert after < before * 0.3  # Check that after is below (before times 0.3); fail this test if it is not true.
    for key, value in agent.target.state_dict().items():  # Process each key, value from the result of agent.target.state dict().items.
        torch.testing.assert_close(value, target_before[key])  # Torch.testing.assert close applied to value, the entries of target before selected by key.
    assert all(parameter.grad is None for parameter in agent.target.parameters())  # Check that all applied to parameter.grad is None for each parameter in the result of agent.target.parameters; fail this test if it is not true.


def test_seeded_environment_and_replay_ownership():  # Check that seeded environment and replay ownership.
    """Check that seeded environment and replay ownership.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    env = make_env()  # Set env to a highway simulator configured with.
    first, _ = env.reset(seed=42)  # Set first, the unused result to the result of env.reset.
    second, _ = env.reset(seed=42)  # Set second, the unused result to the result of env.reset.
    np.testing.assert_array_equal(first, second)  # Np.testing.assert array equal applied to first, second.
    assert flatten(first).shape == (40,)  # Check that flatten(first).shape equals 40; fail this test if it is not true.
    buffer = ReplayBuffer(4)  # Set buffer to ReplayBuffer applied to 4.
    observation = flatten(first)  # Set the currently observed traffic features to the 40 network inputs flattened from first.
    buffer.add(observation, 1, 1.0, observation, False)  # Buffer.add applied to the currently observed traffic features, 1, 1.0, the currently observed traffic features, false.
    observation[:] = 999  # Set the entries of the currently observed traffic features selected by : to 999.
    sampled = buffer.sample(1, np.random.default_rng(0))  # Set sampled to buffer.sample applied to 1, a separate reproducible random generator seeded with 0.
    assert not (sampled[0] == 999).any()  # Check that not the result of (sampled[0] == 999).any; fail this test if it is not true.
    env.close()  # The result of env.close.


def test_cli_training_checkpoint_and_evaluation_roundtrip(tmp_path, monkeypatch):  # Check that cli training checkpoint and evaluation roundtrip.
    """Check that cli training checkpoint and evaluation roundtrip.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    import json  # Make json available for this file's calculations and workflow.
    from driving_rl.cli import main  # Make main available for this file's calculations and workflow.

    run = tmp_path / "run"  # Set run to the path to 'run' inside tmp path.
    monkeypatch.setattr(  # Monkeypatch.setattr applied to 'sys.argv', the listed settings or measurements in their recorded order.
        "sys.argv",  # Include 'sys.argv' in the surrounding operation.
        [  # Include the listed settings or measurements in their recorded order in the surrounding operation.
            "driving-rl",  # Include 'driving-rl' in the surrounding operation.
            "train",  # Include 'train' in the surrounding operation.
            "--algorithm",  # Include '--algorithm' in the surrounding operation.
            "qrdqn",  # Include 'qrdqn' in the surrounding operation.
            "--steps",  # Include '--steps' in the surrounding operation.
            "8",  # Include '8' in the surrounding operation.
            "--learning-starts",  # Include '--learning-starts' in the surrounding operation.
            "4",  # Include '4' in the surrounding operation.
            "--batch-size",  # Include '--batch-size' in the surrounding operation.
            "4",  # Include '4' in the surrounding operation.
            "--checkpoint-interval",  # Include '--checkpoint-interval' in the surrounding operation.
            "4",  # Include '4' in the surrounding operation.
            "--collision-penalty",  # Include '--collision-penalty' in the surrounding operation.
            "-8",  # Include '-8' in the surrounding operation.
            "--out",  # Include '--out' in the surrounding operation.
            str(run),  # Include run as a text value in the surrounding operation.
        ],  # Finish assembling the grouped values above.
    )  # Finish assembling monkeypatch.setattr inputs.
    main()  # The result of main.
    assert json.loads((run / "summary.json").read_text())["training_complete"]  # Check that the entries of json.loads applied to the result of (run / 'summary.json').read text selected by 'training_complete'; fail this test if it is not true.
    result = tmp_path / "evaluation.json"  # Set result to the path to 'evaluation.json' inside tmp path.
    monkeypatch.setattr(  # Monkeypatch.setattr applied to 'sys.argv', the listed settings or measurements in their recorded order.
        "sys.argv",  # Include 'sys.argv' in the surrounding operation.
        [  # Include the listed settings or measurements in their recorded order in the surrounding operation.
            "driving-rl",  # Include 'driving-rl' in the surrounding operation.
            "evaluate",  # Include 'evaluate' in the surrounding operation.
            "--checkpoint",  # Include '--checkpoint' in the surrounding operation.
            str(run / "final.pt"),  # Include the path to 'final.pt' inside run as a text value in the surrounding operation.
            "--episodes",  # Include '--episodes' in the surrounding operation.
            "2",  # Include '2' in the surrounding operation.
            "--densities",  # Include '--densities' in the surrounding operation.
            "1",  # Include '1' in the surrounding operation.
            "--out",  # Include '--out' in the surrounding operation.
            str(result),  # Include result as a text value in the surrounding operation.
            "--gif",  # Include '--gif' in the surrounding operation.
            str(tmp_path / "rollout.gif"),  # Include the path to 'rollout.gif' inside tmp path as a text value in the surrounding operation.
        ],  # Finish assembling the grouped values above.
    )  # Finish assembling monkeypatch.setattr inputs.
    main()  # The result of main.
    data = json.loads(result.read_text())  # Set data to json.loads applied to the result of result.read text.
    assert data["provenance"]["training_steps"] == 8  # Check that the entries of the entries of data selected by 'provenance' selected by 'training_steps' equals 8; fail this test if it is not true.
    assert data["provenance"]["training_collision_penalty"] == -8  # Check that the entries of the entries of data selected by 'provenance' selected by 'training_collision_penalty' equals negative 8; fail this test if it is not true.
    assert data["provenance"]["evaluation_collision_penalty"] == -2  # Check that the entries of the entries of data selected by 'provenance' selected by 'evaluation_collision_penalty' equals negative 2; fail this test if it is not true.
    assert len(data["episodes"]) == 2  # Check that the number of entries in the entries of data selected by 'episodes' equals 2; fail this test if it is not true.
    assert (tmp_path / "rollout.gif").is_file()  # Check that the result of (tmp path / 'rollout.gif').is file; fail this test if it is not true.
    from PIL import Image  # Make Image available for this file's calculations and workflow.

    image = Image.open(tmp_path / "rollout.gif")  # Set image to Image.open applied to the path to 'rollout.gif' inside tmp path.
    assert image.n_frames >= 2  # Check that image.n frames is at least 2; fail this test if it is not true.
    assert np.asarray(image.convert("RGB")).std() > 5  # Check that the result of np.asarray(image.convert('RGB')).std is above 5; fail this test if it is not true.


def test_binomial_interval_keeps_uncertainty_at_extremes():  # Check that binomial interval keeps uncertainty at extremes.
    """Check that binomial interval keeps uncertainty at extremes.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    from driving_rl.metrics import wilson_interval  # Make wilson_interval available for this file's calculations and workflow.

    low, high = wilson_interval(0, 30)  # Set low, high to wilson interval applied to 0, 30.
    assert low == pytest.approx(0.0, abs=1e-12)  # Check that low equals pytest.approx applied to 0.0; fail this test if it is not true.
    assert 0.11 < high < 0.12  # Check that 0.11 is below high is below 0.12; fail this test if it is not true.
    mirrored_low, mirrored_high = wilson_interval(30, 30)  # Set mirrored low, mirrored high to wilson interval applied to 30, 30.
    assert mirrored_low == pytest.approx(1 - high)  # Check that mirrored low equals pytest.approx applied to (1 minus high); fail this test if it is not true.
    assert mirrored_high == pytest.approx(1.0)  # Check that mirrored high equals pytest.approx applied to 1.0; fail this test if it is not true.


def test_speed_governor_decodes_observation_and_controls_acceleration():  # Check that speed governor decodes observation and controls acceleration.
    """Check that speed governor decodes observation and controls acceleration.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    from driving_rl.envs import govern_action, observed_speed  # Make govern_action, observed_speed available for this file's calculations and workflow.

    env = make_env()  # Set env to a highway simulator configured with.
    observation, _ = env.reset(seed=42)  # Set the currently observed traffic features, the unused result to the result of env.reset.
    assert observed_speed(env, observation) == pytest.approx(  # Simulator speed is consulted only to validate the decoding in this test.
        env.unwrapped.vehicle.speed, abs=1e-4  # Include env.unwrapped.vehicle.speed, 0.0001 in the surrounding operation.
    )  # Finish assembling pytest.approx inputs.
    actions = env.unwrapped.action_type.actions_indexes  # Set the recorded action indices to env.unwrapped.action type.actions indexes.
    speed = observed_speed(env, observation)  # Set speed to observed speed applied to env, the currently observed traffic features.
    assert (  # Check that govern action applied to env, the currently observed traffic features, the entries of the recorded action indices selected by 'FASTER', (speed minus 1) equals the entries of the recorded action indices selected by 'SLOWER'; fail this test if it is not true.
        govern_action(env, observation, actions["FASTER"], speed - 1)  # Include govern action applied to env, the currently observed traffic features, the entries of the recorded action indices selected by 'FASTER', (speed minus 1) equals the entries of the recorded action indices selected by 'SLOWER' in the surrounding operation.
        == actions["SLOWER"]  # Include the entries of the recorded action indices selected by 'SLOWER' in the surrounding operation.
    )  # Finish assembling the grouped values above.
    assert (  # Check that govern action applied to env, the currently observed traffic features, the entries of the recorded action indices selected by 'FASTER', (speed plus 1) equals the entries of the recorded action indices selected by 'IDLE'; fail this test if it is not true.
        govern_action(env, observation, actions["FASTER"], speed + 1) == actions["IDLE"]  # Include govern action applied to env, the currently observed traffic features, the entries of the recorded action indices selected by 'FASTER', (speed plus 1) equals the entries of the recorded action indices selected by 'IDLE' in the surrounding operation.
    )  # Finish assembling the grouped values above.
    assert (  # Check that govern action applied to env, the currently observed traffic features, the entries of the recorded action indices selected by 'LANE_LEFT', (speed plus 4) equals the entries of the recorded action indices selected by 'LANE_LEFT'; fail this test if it is not true.
        govern_action(env, observation, actions["LANE_LEFT"], speed + 4)  # Include govern action applied to env, the currently observed traffic features, the entries of the recorded action indices selected by 'LANE_LEFT', (speed plus 4) equals the entries of the recorded action indices selected by 'LANE_LEFT' in the surrounding operation.
        == actions["LANE_LEFT"]  # Include the entries of the recorded action indices selected by 'LANE_LEFT' in the surrounding operation.
    )  # Finish assembling the grouped values above.
    env.close()  # The result of env.close.


def test_absolute_exploration_is_invariant_to_training_budget():  # Check that absolute exploration is invariant to training budget.
    """Check that absolute exploration is invariant to training budget.
    This test protects an assumption that the experiment or its interpretation relies on.
    """
    from driving_rl.experiment import exploration_epsilon  # Make exploration_epsilon available for this file's calculations and workflow.

    for step in [1, 5000, 12001, 20000]:  # Process each step from 1, 5000, 12001, 20000.
        short = exploration_epsilon(step, 20000)  # Set short to the random-action probability scheduled from step, 20000.
        long = exploration_epsilon(step, 40000, decay_steps=12000)  # Set long to the random-action probability scheduled from step, 40000.
        assert short == pytest.approx(long)  # Check that short equals pytest.approx applied to long; fail this test if it is not true.
    assert exploration_epsilon(1, 40000, decay_steps=12000) == 1.0  # Check that the random-action probability scheduled from 1, 40000 equals 1.0; fail this test if it is not true.
    assert exploration_epsilon(12001, 40000, decay_steps=12000) == pytest.approx(0.05)  # Check that the random-action probability scheduled from 12001, 40000 equals pytest.approx applied to 0.05; fail this test if it is not true.
