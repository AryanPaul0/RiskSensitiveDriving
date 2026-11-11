"""The viewer must explain real decisions, not change the driving trajectory."""  # Describe this operation for readers and help().

from pathlib import Path  # Make Path available for this file's calculations and workflow.
import numpy as np  # Make numpy available for this file's calculations and workflow.
import torch  # Make torch available for this file's calculations and workflow.
from driving_rl.agents import Agent, action_scores  # Make Agent, action_scores available for this file's calculations and workflow.
from driving_rl.envs import make_env, flatten, seed_everything  # Make make_env, flatten, seed_everything available for this file's calculations and workflow.
import run_decision_viewer as viewer  # Make run_decision_viewer available for this file's calculations and workflow.


def test_recorded_predictions_actions_and_frames_match_real_policy(  # Use an untrained network: test recording behavior without training anything.
    tmp_path, monkeypatch  # Accept tmp path, monkeypatch for this operation.
):  # Begin the test recorded predictions actions and frames match real policy procedure with the inputs listed above.
    """Use an untrained network: test recording behavior without training anything."""  # Describe this operation for readers and help().
    monkeypatch.setattr(viewer, "OUTPUT", tmp_path)  # Monkeypatch.setattr applied to viewer, 'output', tmp path.
    (tmp_path / "frames").mkdir()  # The result of (tmp path / 'frames').mkdir.
    seed_everything(6)  # Initialize repeatable random streams using 6.
    agent = Agent(40, 5, "qrdqn")  # Set agent to Agent applied to 40, 5, 'qrdqn'.
    recording = viewer.record_episode(agent, "qr_tail", 1.0, 18000)  # Set recording to viewer.record episode applied to agent, 'qr_tail', 1.0, 18000.
    env = make_env(density=1.0)  # Set env to a highway simulator configured with.
    observation, _ = env.reset(seed=18000)  # Set the currently observed traffic features, the unused result to the result of env.reset.
    try:  # Run this work with the error handling or resource cleanup specified below.
        for state in recording["states"][:-1]:  # Process each state from the entries of the entries of recording selected by 'states' selected by :-1.
            np.testing.assert_array_equal(state["observation"], observation)  # Np.testing.assert array equal applied to the entries of state selected by 'observation', the currently observed traffic features.
            action = agent.act(flatten(observation), 0.25)  # Set action to agent.act applied to the 40 network inputs flattened from the currently observed traffic features, 0.25.
            assert state["action"] == action == state["proposed_action"]  # Check that the entries of state selected by 'action' equals action equals the entries of state selected by 'proposed_action'; fail this test if it is not true.
            values = torch.tensor(state["predictions"])  # Set values to the entries of state selected by 'predictions' as a numerical array/tensor for batch calculations.
            assert int(action_scores(values, 0.25).argmax()) == action  # Check that the index of the highest score in the mean or lower-tail action scores computed from values, 0.25 as a whole-number value equals action; fail this test if it is not true.
            assert (tmp_path / state["frame"]).is_file()  # Check that the result of (tmp path / state['frame']).is file; fail this test if it is not true.
            observation, reward, terminal, cutoff, _ = env.step(action)  # Set the currently observed traffic features, reward, terminal, cutoff, the unused result to env.step applied to action.
            assert state["reward_after_action"] == reward  # Check that the entries of state selected by 'reward_after_action' equals reward; fail this test if it is not true.
        assert terminal == recording["terminated"]  # Check that terminal equals the entries of recording selected by 'terminated'; fail this test if it is not true.
        assert cutoff == recording["truncated"]  # Check that cutoff equals the entries of recording selected by 'truncated'; fail this test if it is not true.
        assert recording["states"][-1]["action"] is None  # Check that the entries of the entries of the entries of recording selected by 'states' selected by -1 selected by 'action' is None; fail this test if it is not true.
        assert recording["states"][-1]["predictions"] is None  # Check that the entries of the entries of the entries of recording selected by 'states' selected by -1 selected by 'predictions' is None; fail this test if it is not true.
        assert len(recording["states"]) == len(recording["rewards"]) + 1  # Check that the number of entries in the entries of recording selected by 'states' equals (the number of entries in the entries of recording selected by 'rewards' plus 1); fail this test if it is not true.
    finally:  # Always release resources here, including after an error or interruption.
        env.close()  # The result of env.close.


def test_scalar_agent_records_one_value_per_action(tmp_path, monkeypatch):  # Do not silently turn a DQN mean into 32 supposed quantiles.
    """Do not silently turn a DQN mean into 32 supposed quantiles."""  # Describe this operation for readers and help().
    monkeypatch.setattr(viewer, "OUTPUT", tmp_path)  # Monkeypatch.setattr applied to viewer, 'output', tmp path.
    (tmp_path / "frames").mkdir()  # The result of (tmp path / 'frames').mkdir.
    seed_everything(4)  # Initialize repeatable random streams using 4.
    agent = Agent(40, 5, "dqn")  # Set agent to Agent applied to 40, 5, 'dqn'.
    recording = viewer.record_episode(agent, "dqn_governed", 2.0, 18001)  # Set recording to viewer.record episode applied to agent, 'dqn_governed', 2.0, 18001.
    for state in recording["states"][:-1]:  # Process each state from the entries of the entries of recording selected by 'states' selected by :-1.
        assert np.asarray(state["predictions"]).shape == (5, 1)  # Check that np.asarray(state['predictions']).shape equals 5, 1; fail this test if it is not true.
        if state["governor_override"]:  # Use this branch when the entries of state selected by 'governor_override'.
            assert state["action"] != state["proposed_action"]  # Check that the entries of state selected by 'action' differs from the entries of state selected by 'proposed_action'; fail this test if it is not true.
