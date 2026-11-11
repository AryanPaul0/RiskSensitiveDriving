"""The viewer must explain real decisions, not change the driving trajectory."""

from pathlib import Path
import numpy as np
import torch
from driving_rl.agents import Agent, action_scores
from driving_rl.envs import make_env, flatten, seed_everything
import run_decision_viewer as viewer


def test_recorded_predictions_actions_and_frames_match_real_policy(
    tmp_path, monkeypatch
):
    """Use an untrained network: test recording behavior without training anything."""
    monkeypatch.setattr(viewer, "OUTPUT", tmp_path)
    (tmp_path / "frames").mkdir()
    seed_everything(6)
    agent = Agent(40, 5, "qrdqn")
    recording = viewer.record_episode(agent, "qr_tail", 1.0, 18000)
    env = make_env(density=1.0)
    observation, _ = env.reset(seed=18000)
    try:
        for state in recording["states"][:-1]:
            np.testing.assert_array_equal(state["observation"], observation)
            action = agent.act(flatten(observation), 0.25)
            assert state["action"] == action == state["proposed_action"]
            values = torch.tensor(state["predictions"])
            assert int(action_scores(values, 0.25).argmax()) == action
            assert (tmp_path / state["frame"]).is_file()
            observation, reward, terminal, cutoff, _ = env.step(action)
            assert state["reward_after_action"] == reward
        assert terminal == recording["terminated"]
        assert cutoff == recording["truncated"]
        assert recording["states"][-1]["action"] is None
        assert recording["states"][-1]["predictions"] is None
        assert len(recording["states"]) == len(recording["rewards"]) + 1
    finally:
        env.close()


def test_scalar_agent_records_one_value_per_action(tmp_path, monkeypatch):
    """Do not silently turn a DQN mean into 32 supposed quantiles."""
    monkeypatch.setattr(viewer, "OUTPUT", tmp_path)
    (tmp_path / "frames").mkdir()
    seed_everything(4)
    agent = Agent(40, 5, "dqn")
    recording = viewer.record_episode(agent, "dqn_governed", 2.0, 18001)
    for state in recording["states"][:-1]:
        assert np.asarray(state["predictions"]).shape == (5, 1)
        if state["governor_override"]:
            assert state["action"] != state["proposed_action"]
