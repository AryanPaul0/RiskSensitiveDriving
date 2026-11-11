# This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
"""The viewer must explain real decisions, not change the driving trajectory."""

# from pathlib import brings the named tools into this file; parentheses allow several lines.
from pathlib import Path
# import makes these modules available; "as" gives a module a shorter local name.
import numpy as np
# import makes these modules available; "as" gives a module a shorter local name.
import torch
# from driving_rl.agents import brings the named tools into this file; parentheses allow several lines.
from driving_rl.agents import Agent, action_scores
# from driving_rl.envs import brings the named tools into this file; parentheses allow several lines.
from driving_rl.envs import make_env, flatten, seed_everything
# import makes these modules available; "as" gives a module a shorter local name.
import run_decision_viewer as viewer


# def defines test_recorded_predictions_actions_and_frames_match_real_policy; parentheses list its inputs, and the colon starts its indented body.
def test_recorded_predictions_actions_and_frames_match_real_policy(
    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
    tmp_path, monkeypatch
# ) ends the parameter list; the colon starts the function body, which must be indented.
):
    # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
    """Use an untrained network: test recording behavior without training anything."""
    # Call monkeypatch.setattr using parentheses to supply its inputs; this line does not store a separate return value.
    monkeypatch.setattr(viewer, "OUTPUT", tmp_path)
    # Call (tmp_path / 'frames').mkdir using parentheses to supply its inputs; this line does not store a separate return value.
    (tmp_path / "frames").mkdir()
    # Call seed_everything using parentheses to supply its inputs; this line does not store a separate return value.
    seed_everything(6)
    # = stores a result in agent; call Agent using parentheses to supply its inputs.
    agent = Agent(40, 5, "qrdqn")
    # = stores a result in recording; call viewer.record_episode using parentheses to supply its inputs.
    recording = viewer.record_episode(agent, "qr_tail", 1.0, 18000)
    # = stores a result in env; call make_env using parentheses to supply its inputs.
    env = make_env(density=1.0)
    # = unpacks the right-hand result into (observation, _), matching values to names in order.
    observation, _ = env.reset(seed=18000)
    # try starts a block whose errors or cleanup are handled by the following except/finally blocks.
    try:
        # for takes each item from recording['states'][:-1] in turn; state holds the current item.
        for state in recording["states"][:-1]:
            # Call np.testing.assert_array_equal using parentheses to supply its inputs; this line does not store a separate return value.
            np.testing.assert_array_equal(state["observation"], observation)
            # = stores a result in action; call agent.act using parentheses to supply its inputs.
            action = agent.act(flatten(observation), 0.25)
            # assert checks this condition; a false result raises AssertionError and fails this check.
            assert state["action"] == action == state["proposed_action"]
            # = stores a result in values; call torch.tensor using parentheses to supply its inputs.
            values = torch.tensor(state["predictions"])
            # assert checks this condition; a false result raises AssertionError and fails this check.
            assert int(action_scores(values, 0.25).argmax()) == action
            # assert checks this condition; a false result raises AssertionError and fails this check.
            assert (tmp_path / state["frame"]).is_file()
            # = unpacks the right-hand result into (observation, reward, terminal, cutoff, _), matching values to names in order.
            observation, reward, terminal, cutoff, _ = env.step(action)
            # assert checks this condition; a false result raises AssertionError and fails this check.
            assert state["reward_after_action"] == reward
        # assert checks this condition; a false result raises AssertionError and fails this check.
        assert terminal == recording["terminated"]
        # assert checks this condition; a false result raises AssertionError and fails this check.
        assert cutoff == recording["truncated"]
        # assert checks this condition; a false result raises AssertionError and fails this check.
        assert recording["states"][-1]["action"] is None
        # assert checks this condition; a false result raises AssertionError and fails this check.
        assert recording["states"][-1]["predictions"] is None
        # assert checks this condition; a false result raises AssertionError and fails this check.
        assert len(recording["states"]) == len(recording["rewards"]) + 1
    # finally always runs this cleanup block when the try block ends, including after an error or return.
    finally:
        # Call env.close using parentheses to supply its inputs; this line does not store a separate return value.
        env.close()


# def defines test_scalar_agent_records_one_value_per_action; parentheses list its inputs, and the colon starts its indented body.
def test_scalar_agent_records_one_value_per_action(tmp_path, monkeypatch):
    # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
    """Do not silently turn a DQN mean into 32 supposed quantiles."""
    # Call monkeypatch.setattr using parentheses to supply its inputs; this line does not store a separate return value.
    monkeypatch.setattr(viewer, "OUTPUT", tmp_path)
    # Call (tmp_path / 'frames').mkdir using parentheses to supply its inputs; this line does not store a separate return value.
    (tmp_path / "frames").mkdir()
    # Call seed_everything using parentheses to supply its inputs; this line does not store a separate return value.
    seed_everything(4)
    # = stores a result in agent; call Agent using parentheses to supply its inputs.
    agent = Agent(40, 5, "dqn")
    # = stores a result in recording; call viewer.record_episode using parentheses to supply its inputs.
    recording = viewer.record_episode(agent, "dqn_governed", 2.0, 18001)
    # for takes each item from recording['states'][:-1] in turn; state holds the current item.
    for state in recording["states"][:-1]:
        # assert checks this condition; a false result raises AssertionError and fails this check.
        assert np.asarray(state["predictions"]).shape == (5, 1)
        # if checks state['governor_override']; only a true condition runs the indented block.
        if state["governor_override"]:
            # assert checks this condition; a false result raises AssertionError and fails this check.
            assert state["action"] != state["proposed_action"]
