"""Record development-only driving examples and build the decision viewer.

This script does not train, tune a policy, or use new confirmation scenarios.
Each recording saves what the network saw, predicted, and actually did.
"""

# from pathlib import brings the named tools into this file; parentheses allow several lines.
from pathlib import Path
# import makes these modules available; "as" gives a module a shorter local name.
import hashlib
# import makes these modules available; "as" gives a module a shorter local name.
import json
# import makes these modules available; "as" gives a module a shorter local name.
import shutil

# import makes these modules available; "as" gives a module a shorter local name.
import numpy as np
# from PIL import brings the named tools into this file; parentheses allow several lines.
from PIL import Image
# import makes these modules available; "as" gives a module a shorter local name.
import torch

# from driving_rl.agents import brings the named tools into this file; parentheses allow several lines.
from driving_rl.agents import action_scores
# from driving_rl.envs import brings the named tools into this file; parentheses allow several lines.
from driving_rl.envs import (
    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
    make_env,
    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
    flatten,
    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
    govern_action,
    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
    render_frame,
    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
    seed_everything,
# ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
)
# from driving_rl.experiment import brings the named tools into this file; parentheses allow several lines.
from driving_rl.experiment import load_agent, save_json
# from driving_rl.diagnostics import brings the named tools into this file; parentheses allow several lines.
from driving_rl.diagnostics import front_state

# = stores a result in ROOT; use a dot to access parent on Path(__file__).resolve().
ROOT = Path(__file__).resolve().parent
# = stores a result in OUTPUT; apply / to these operands; numbers are divided, while Path objects join folder names.
OUTPUT = ROOT / "results/decision_viewer"
# = stores a result in POLICIES; build a dictionary, which stores values under named keys.
POLICIES = {
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "qr_tail": ("qrdqn", 0.25, None),
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "qr_mean": ("qrdqn", 1.0, None),
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "qr_governed": ("qrdqn", 1.0, 25.0),
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "dqn_governed": ("dqn", 1.0, 25.0),
# } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
}


# def defines model_path; parentheses list its inputs, and the colon starts its indented body.
def model_path(budget, policy, seed):
    # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
    """Several decision rules share the same learned QR network."""
    # = stores a result in algorithm; use square brackets to select part of POLICIES[policy].
    algorithm = POLICIES[policy][0]
    # if checks budget == 100000; only a true condition runs the indented block.
    if budget == 100000:
        # = stores a result in folder; use quoted text as data; Python does not execute the words inside it.
        folder = "long_training_100k"
    # if checks algorithm == 'qrdqn'; only a true condition runs the indented block.
    elif algorithm == "qrdqn":
        # = stores a result in folder; use quoted text as data; Python does not execute the words inside it.
        folder = "density_coverage"
    # else provides the alternative branch; after a loop it instead means the loop finished without break.
    else:
        # = stores a result in folder; use quoted text as data; Python does not execute the words inside it.
        folder = "baseline_controls"
    # return ends this function and sends its result back to the code that called it.
    return ROOT / f"results/{folder}/{algorithm}_seed{seed}/final.pt"


# def defines save_frame; parentheses list its inputs, and the colon starts its indented body.
def save_frame(env):
    # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
    """Reuse identical pictures rather than saving repeated initial frames."""
    # = stores a result in pixels; call render_frame using parentheses to supply its inputs.
    pixels = render_frame(env)
    # = stores a result in name; combine the operands with the displayed operator; parentheses control grouping.
    name = hashlib.sha256(pixels.tobytes()).hexdigest() + ".png"
    # = stores a result in path; apply / to these operands; numbers are divided, while Path objects join folder names.
    path = OUTPUT / "frames" / name
    # if checks not path.exists(); only a true condition runs the indented block.
    if not path.exists():
        # Call image.fromarray(pixels).save using parentheses to supply its inputs; this line does not store a separate return value.
        Image.fromarray(pixels).save(path)
    # return ends this function and sends its result back to the code that called it.
    return "frames/" + name


# def defines record_episode; parentheses list its inputs, and the colon starts its indented body.
def record_episode(agent, policy, density, scenario):
    """Keep pre-action predictions paired with the action they really produced.

    Other policies soon reach different states. Their recordings are not
    counterfactual outcomes from the currently selected policy's later state.
    """
    # = unpacks the right-hand result into (algorithm, alpha, cap), matching values to names in order.
    algorithm, alpha, cap = POLICIES[policy]
    # = stores a result in env; call make_env using parentheses to supply its inputs.
    env = make_env(density=density, render=True)
    # = unpacks the right-hand result into (observation, _), matching values to names in order.
    observation, _ = env.reset(seed=scenario)
    # = stores a result in states; build a list, keeping its items in order.
    states = []
    # = stores a result in rewards; build a list, keeping its items in order.
    rewards = []
    # try starts a block whose errors or cleanup are handled by the following except/finally blocks.
    try:
        # while repeats the indented block as long as True is true.
        while True:
            # We are asking the saved network for predictions. No learning or weight update happens here.
            # with temporarily disables gradient tracking for this block; these predictions are fixed answers, not a second training path.
            with torch.no_grad():
                # = stores a result in inputs; call torch.as_tensor(flatten(observation)).unsqueeze using parentheses to supply its inputs.
                inputs = torch.as_tensor(flatten(observation)).unsqueeze(0)
                # = stores a result in predictions; call agent.online using parentheses to supply its inputs.
                predictions = agent.online(inputs)
                # = stores a result in proposed; call int using parentheses to supply its inputs.
                proposed = int(action_scores(predictions, alpha).argmax().item())
            # = stores a result in action; evaluate the expression on the right before using its result.
            action = proposed
            # The governor is an optional action override. Its threshold is not a guaranteed physical speed limit.
            # if checks cap is not None; only a true condition runs the indented block.
            if cap is not None:
                # = stores a result in action; call govern_action using parentheses to supply its inputs.
                action = govern_action(env, observation, proposed, cap)
            # = stores a result in state; build a dictionary, which stores values under named keys.
            state = {
                # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                "time": len(rewards),
                # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                "frame": save_frame(env),
                # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                "predictions": predictions[0].tolist(),
                # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                "proposed_action": proposed,
                # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                "action": action,
                # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                "governor_override": action != proposed,
                # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                "lane": int(env.unwrapped.vehicle.lane_index[2]),
                # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                "observation": np.asarray(observation).tolist(),
                # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
                **front_state(env),
            # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
            }
            # = unpacks the right-hand result into (observation, reward, terminated, truncated, _), matching values to names in order.
            observation, reward, terminated, truncated, _ = env.step(action)
            # = stores a result in state['reward_after_action']; call float using parentheses to supply its inputs.
            state["reward_after_action"] = float(reward)
            # Call states.append using parentheses to supply its inputs; this line does not store a separate return value.
            states.append(state)
            # Call rewards.append using parentheses to supply its inputs; this line does not store a separate return value.
            rewards.append(float(reward))
            # if checks terminated or truncated; only a true condition runs the indented block.
            if terminated or truncated:
                # The final picture has no issued action or predicted decision.
                # Call states.append using parentheses to supply its inputs; this line does not store a separate return value.
                states.append(
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    {
                        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                        "time": len(rewards),
                        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                        "frame": save_frame(env),
                        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                        "action": None,
                        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                        "predictions": None,
                        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                        "lane": int(env.unwrapped.vehicle.lane_index[2]),
                        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
                        **front_state(env),
                    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
                    }
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                )
                # return ends this function and sends its result back to the code that called it.
                return {
                    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                    "states": states,
                    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                    "rewards": rewards,
                    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                    "crashed": bool(env.unwrapped.vehicle.crashed),
                    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                    "terminated": bool(terminated),
                    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                    "truncated": bool(truncated),
                # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
                }
    # finally always runs this cleanup block when the try block ends, including after an error or return.
    finally:
        # Call env.close using parentheses to supply its inputs; this line does not store a separate return value.
        env.close()


# def defines main; parentheses list its inputs, and the colon starts its indented body.
def main():
    # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
    """Record a fixed grid of examples, then make a self-contained local page."""
    # Call seed_everything using parentheses to supply its inputs; this line does not store a separate return value.
    seed_everything(0)
    # Call output.mkdir using parentheses to supply its inputs; this line does not store a separate return value.
    OUTPUT.mkdir(exist_ok=True)
    # Call (output / 'frames').mkdir using parentheses to supply its inputs; this line does not store a separate return value.
    (OUTPUT / "frames").mkdir(exist_ok=True)
    # = stores a result in design; build a dictionary, which stores values under named keys.
    design = {
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "budgets": [40000, 100000],
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "training_seeds": [0, 1, 2],
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "densities": [1.0, 2.0, 2.5],
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "scenario_seeds": [18000, 18001],
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "policy_names": list(POLICIES),
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "purpose": "Illustrative development examples chosen before outcomes; not performance evidence or policy tuning.",
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "confirmation_seeds_used": False,
    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
    }
    # = stores a result in design_path; apply / to these operands; numbers are divided, while Path objects join folder names.
    design_path = OUTPUT / "design.json"
    # if checks design_path.exists() and json.loads(design_path.read_text()) != design; only a true condition runs the indented block.
    if design_path.exists() and json.loads(design_path.read_text()) != design:
        # raise signals an error and stops this normal path, so invalid data are not silently accepted.
        raise ValueError("The recorded example grid must not silently change")
    # Call save_json using parentheses to supply its inputs; this line does not store a separate return value.
    save_json(design_path, design)
    # = stores a result in records; build a list, keeping its items in order.
    records = []
    # Each budget gets its own results. Keep 40K and 100K separate until the paired comparison.
    # for takes each item from design['budgets'] in turn; budget holds the current item.
    for budget in design["budgets"]:
        # for takes each item from design['training_seeds'] in turn; training_seed holds the current item.
        for training_seed in design["training_seeds"]:
            # A policy is a decision rule. Several rules can share the very same QR-DQN network.
            # for takes each item from POLICIES in turn; policy holds the current item.
            for policy in POLICIES:
                # = stores a result in checkpoint; call model_path using parentheses to supply its inputs.
                checkpoint = model_path(budget, policy, training_seed)
                # = stores a result in model_hash; call hashlib.sha256(checkpoint.read_bytes()).hexdigest using parentheses to supply its inputs.
                model_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                # = unpacks the right-hand result into (agent, _), matching values to names in order.
                agent, _ = load_agent(checkpoint)
                # Traffic density changes vehicle spacing. It does not add more network inputs.
                # for takes each item from design['densities'] in turn; density holds the current item.
                for density in design["densities"]:
                    # A scenario seed fixes the starting traffic. Reuse the same IDs for fair policy comparisons.
                    # for takes each item from design['scenario_seeds'] in turn; scenario holds the current item.
                    for scenario in design["scenario_seeds"]:
                        # = stores a result in key; evaluate the expression on the right before using its result.
                        key = (
                            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                            f"{budget}_{training_seed}_{policy}_{density:g}_{scenario}"
                        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                        )
                        # = stores a result in path; apply / to these operands; numbers are divided, while Path objects join folder names.
                        path = OUTPUT / (key + ".json")
                        # A saved filename is not enough: the checks below verify the contents belong to this experiment.
                        # if checks path.exists(); only a true condition runs the indented block.
                        if path.exists():
                            # = stores a result in record; call json.loads using parentheses to supply its inputs.
                            record = json.loads(path.read_text())
                            # if checks record['model_hash'] != model_hash or record['key'] != key or record['budget'] != budget or (reco...; only a true condition runs the indented block.
                            if (
                                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                                record["model_hash"] != model_hash
                                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                                or record["key"] != key
                                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                                or record["budget"] != budget
                                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                                or record["training_seed"] != training_seed
                                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                                or record["policy"] != policy
                                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                                or record["density"] != density
                                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                                or record["scenario"] != scenario
                            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                            ):
                                # raise signals an error and stops this normal path, so invalid data are not silently accepted.
                                raise ValueError("A recording belongs to another model")
                        # else provides the alternative branch; after a loop it instead means the loop finished without break.
                        else:
                            # = stores a result in record; build a dictionary, which stores values under named keys.
                            record = {
                                # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                                "key": key,
                                # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                                "budget": budget,
                                # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                                "training_seed": training_seed,
                                # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                                "policy": policy,
                                # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                                "density": density,
                                # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                                "scenario": scenario,
                                # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                                "model_hash": model_hash,
                                # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
                                **record_episode(agent, policy, density, scenario),
                            # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
                            }
                            # Call save_json using parentheses to supply its inputs; this line does not store a separate return value.
                            save_json(path, record)
                        # for takes each item from record['states'] in turn; state holds the current item.
                        for state in record["states"]:
                            # if checks not (OUTPUT / state['frame']).exists(); only a true condition runs the indented block.
                            if not (OUTPUT / state["frame"]).exists():
                                # raise signals an error and stops this normal path, so invalid data are not silently accepted.
                                raise ValueError(
                                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                                    "A cached recording is missing a frame"
                                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                                )
                        # Call records.append using parentheses to supply its inputs; this line does not store a separate return value.
                        records.append(record)
                # Call print using parentheses to supply its inputs; this line does not store a separate return value.
                print(f"Recorded {budget} {training_seed} {policy}", flush=True)
    # Call save_json using parentheses to supply its inputs; this line does not store a separate return value.
    save_json(OUTPUT / "recordings.json", records)
    # = stores a result in viewer; apply / to these operands; numbers are divided, while Path objects join folder names.
    viewer = ROOT / "results/report"
    # = stores a result in backup; apply / to these operands; numbers are divided, while Path objects join folder names.
    backup = ROOT / "results/long_training-report"
    # if checks not backup.exists(); only a true condition runs the indented block.
    if not backup.exists():
        # Call shutil.copytree using parentheses to supply its inputs; this line does not store a separate return value.
        shutil.copytree(viewer, backup)
    # = stores a result in template; call (ROOT / 'driving_rl/decision_viewer.html').read_text using parentheses to supply its inputs.
    template = (ROOT / "driving_rl/decision_viewer.html").read_text()
    # = stores a result in payload; call json.dumps(records).replace using parentheses to supply its inputs.
    payload = json.dumps(records).replace("<", "\\u003c")
    # = stores a result in evidence; call (ROOT / 'results/long_training_evaluation/summary.json').read_text().replace using parentheses to supply its inputs.
    evidence = (
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        (ROOT / "results/long_training_evaluation/summary.json")
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        .read_text()
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        .replace("<", "\\u003c")
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # The long string below is page layout, not reinforcement learning. Later lines insert the measured numbers.
    # = stores a result in page; call template.replace('__RECORDINGS__', payload).replace using parentheses to supply its inputs.
    page = template.replace("__RECORDINGS__", payload).replace("__EVIDENCE__", evidence)
    # Call (viewer / 'index.html').write_text using parentheses to supply its inputs; this line does not store a separate return value.
    (viewer / "index.html").write_text(page)
    # Call print using parentheses to supply its inputs; this line does not store a separate return value.
    print(f"Viewer saved with {len(records)} development recordings.", flush=True)


# __name__ equals "__main__" when this file runs directly; this guard avoids running the script merely because another file imports it.
if __name__ == "__main__":
    # Call main using parentheses to supply its inputs; this line does not store a separate return value.
    main()
