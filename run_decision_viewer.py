"""Record development-only driving examples and build the decision viewer.

This script does not train, tune a policy, or use new confirmation scenarios.
Each recording saves what the network saw, predicted, and actually did.
"""

from pathlib import Path  # Make Path available for this file's calculations and workflow.
import hashlib  # Make hashlib available for this file's calculations and workflow.
import json  # Make json available for this file's calculations and workflow.
import shutil  # Make shutil available for this file's calculations and workflow.

import numpy as np  # Make numpy available for this file's calculations and workflow.
from PIL import Image  # Make Image available for this file's calculations and workflow.
import torch  # Make torch available for this file's calculations and workflow.

from driving_rl.agents import action_scores  # Make action_scores available for this file's calculations and workflow.
from driving_rl.envs import (  # Make make_env, flatten, govern_action, render_frame, seed_everything available for this file's calculations and workflow.
    make_env,  # Continue the calculation or input list started above.
    flatten,  # Continue the calculation or input list started above.
    govern_action,  # Continue the calculation or input list started above.
    render_frame,  # Continue the calculation or input list started above.
    seed_everything,  # Continue the calculation or input list started above.
)  # Finish assembling the grouped values above.
from driving_rl.experiment import load_agent, save_json  # Make load_agent, save_json available for this file's calculations and workflow.
from driving_rl.diagnostics import front_state  # Make front_state available for this file's calculations and workflow.

ROOT = Path(__file__).resolve().parent  # Set ROOT to Path( file ).resolve().parent.
OUTPUT = ROOT / "results/decision_viewer"  # Set OUTPUT to the path to 'results/decision_viewer' inside ROOT.
POLICIES = {  # Set POLICIES to the named settings or measurements assembled below.
    "qr_tail": ("qrdqn", 0.25, None),  # Record 'qr_tail' as 'qrdqn', 0.25, None.
    "qr_mean": ("qrdqn", 1.0, None),  # Record 'qr_mean' as 'qrdqn', 1.0, None.
    "qr_governed": ("qrdqn", 1.0, 25.0),  # Record 'qr_governed' as 'qrdqn', 1.0, 25.0.
    "dqn_governed": ("dqn", 1.0, 25.0),  # Record 'dqn_governed' as 'dqn', 1.0, 25.0.
}  # Finish assembling the grouped values above.


def model_path(budget, policy, seed):  # Several decision rules share the same learned QR network.
    """Several decision rules share the same learned QR network."""  # Describe this operation for readers and help().
    algorithm = POLICIES[policy][0]  # Set algorithm to the entries of the entries of POLICIES selected by policy selected by 0.
    if budget == 100000:  # Use this branch when the environment-interaction budget equals 100000.
        folder = "long_training_100k"  # Set folder to 'long_training_100k'.
    elif algorithm == "qrdqn":  # Use this branch when algorithm equals 'qrdqn'.
        folder = "density_coverage"  # Set folder to 'density_coverage'.
    else:  # Handle the alternative when the earlier condition (algorithm equals 'qrdqn') is false.
        folder = "baseline_controls"  # Set folder to 'baseline_controls'.
    return ROOT / f"results/{folder}/{algorithm}_seed{seed}/final.pt"  # Return (ROOT divided by f'results/{folder}/{algorithm} seed{seed}/final.pt').


def save_frame(env):  # Reuse identical pictures rather than saving repeated initial frames.
    """Reuse identical pictures rather than saving repeated initial frames."""  # Describe this operation for readers and help().
    pixels = render_frame(env)  # Set pixels to render frame applied to env.
    name = hashlib.sha256(pixels.tobytes()).hexdigest() + ".png"  # Set name to (the result of hashlib.sha256(pixels.tobytes()).hexdigest plus '.png').
    path = OUTPUT / "frames" / name  # Set path to (the path to 'frames' inside OUTPUT divided by name).
    if not path.exists():  # Use this branch when not the result of path.exists.
        Image.fromarray(pixels).save(path)  # Image.fromarray(pixels).save applied to path.
    return "frames/" + name  # Return ('frames/' plus name).


def record_episode(agent, policy, density, scenario):  # Keep pre-action predictions paired with the action they really produced.
    """Keep pre-action predictions paired with the action they really produced.

    Other policies soon reach different states. Their recordings are not
    counterfactual outcomes from the currently selected policy's later state.
    """
    algorithm, alpha, cap = POLICIES[policy]  # Set algorithm, the fraction of low-return predictions used to score actions, cap to the entries of POLICIES selected by policy.
    env = make_env(density=density, render=True)  # Set env to a highway simulator configured with.
    observation, _ = env.reset(seed=scenario)  # Set the currently observed traffic features, the unused result to the result of env.reset.
    states = []  # Set states to .
    rewards = []  # Set the observed rewards to .
    try:  # Run this work with the error handling or resource cleanup specified below.
        while True:  # Keep processing while True.
            with torch.no_grad():  # Treat predictions in this block as fixed learning targets or inference results; do not backpropagate through them.
                inputs = torch.as_tensor(flatten(observation)).unsqueeze(0)  # Set inputs to torch.as tensor(flatten(observation)).unsqueeze applied to 0.
                predictions = agent.online(inputs)  # Set predictions to agent.online applied to inputs.
                proposed = int(action_scores(predictions, alpha).argmax().item())  # The viewer records all five actions' predictions, not just the winner. Later we can compare hypothetical scores in this exact observation without rerunning the simulator.
            action = proposed  # There is only one observation here, so argmax over the whole [1,5] score array yields the same action as dim=1.
            if cap is not None:  # The governor is an optional action override. Its threshold is not a guaranteed physical speed limit.
                action = govern_action(env, observation, proposed, cap)  # Set action to govern action applied to env, the currently observed traffic features, proposed, cap.
            state = {  # Set state to the named settings or measurements assembled below.
                "time": len(rewards),  # Record 'time' as the number of entries in the observed rewards.
                "frame": save_frame(env),  # Record 'frame' as save frame applied to env.
                "predictions": predictions[0].tolist(),  # Record 'predictions' as the entries of predictions selected by 0 as a list that can be saved in JSON.
                "proposed_action": proposed,  # Record 'proposed_action' as proposed.
                "action": action,  # Record 'action' as action.
                "governor_override": action != proposed,  # Record 'governor_override' as action differs from proposed.
                "lane": int(env.unwrapped.vehicle.lane_index[2]),  # Record 'lane' as the entries of env.unwrapped.vehicle.lane index selected by 2 as a whole-number value.
                "observation": np.asarray(observation).tolist(),  # Record 'observation' as the currently observed traffic features as a numerical array/tensor for batch calculations as a list that can be saved in JSON.
                **front_state(env),  # Include front state applied to env in the surrounding operation.
            }  # Finish assembling the grouped values above.
            observation, reward, terminated, truncated, _ = env.step(action)  # Set the currently observed traffic features, reward, the true-terminal flags, the recording-cutoff flags, the unused result to env.step applied to action.
            state["reward_after_action"] = float(reward)  # Set the entries of state selected by 'reward_after_action' to reward as a floating-point value.
            states.append(state)  # Add state to states.
            rewards.append(float(reward))  # Add reward as a floating-point value to the observed rewards.
            if terminated or truncated:  # Use this branch when the true-terminal flags or the recording-cutoff flags.
                states.append(  # The final picture has no issued action or predicted decision.
                    {  # Include the named settings or measurements assembled below in the surrounding operation.
                        "time": len(rewards),  # Record 'time' as the number of entries in the observed rewards.
                        "frame": save_frame(env),  # Record 'frame' as save frame applied to env.
                        "action": None,  # Record 'action' as None.
                        "predictions": None,  # Record 'predictions' as None.
                        "lane": int(env.unwrapped.vehicle.lane_index[2]),  # Record 'lane' as the entries of env.unwrapped.vehicle.lane index selected by 2 as a whole-number value.
                        **front_state(env),  # Include front state applied to env in the surrounding operation.
                    }  # Finish assembling the grouped values above.
                )  # Finish assembling states.append inputs.
                return {  # Return the named settings or measurements assembled below.
                    "states": states,  # Record 'states' as states.
                    "rewards": rewards,  # Record 'rewards' as the observed rewards.
                    "crashed": bool(env.unwrapped.vehicle.crashed),  # Record 'crashed' as env.unwrapped.vehicle.crashed as a true/false value.
                    "terminated": bool(terminated),  # Record 'terminated' as the true-terminal flags as a true/false value.
                    "truncated": bool(truncated),  # Record 'truncated' as the recording-cutoff flags as a true/false value.
                }  # Finish assembling the grouped values above.
    finally:  # Always release resources here, including after an error or interruption.
        env.close()  # The result of env.close.


def main():  # Record a fixed grid of examples, then make a self-contained local page.
    """Record a fixed grid of examples, then make a self-contained local page."""  # Describe this operation for readers and help().
    seed_everything(0)  # Initialize repeatable random streams using 0.
    OUTPUT.mkdir(exist_ok=True)  # The result of output.mkdir.
    (OUTPUT / "frames").mkdir(exist_ok=True)  # The result of (output / 'frames').mkdir.
    design = {  # Set design to the named settings or measurements assembled below.
        "budgets": [40000, 100000],  # Record 'budgets' as 40000, 100000.
        "training_seeds": [0, 1, 2],  # Record 'training_seeds' as 0, 1, 2.
        "densities": [1.0, 2.0, 2.5],  # Record 'densities' as 1.0, 2.0, 2.5.
        "scenario_seeds": [18000, 18001],  # Record 'scenario_seeds' as 18000, 18001.
        "policy_names": list(POLICIES),  # Record 'policy_names' as list applied to POLICIES.
        "purpose": "Illustrative development examples chosen before outcomes; not performance evidence or policy tuning.",  # Record 'purpose' as the report text.
        "confirmation_seeds_used": False,  # Record 'confirmation_seeds_used' as False.
    }  # Finish assembling the grouped values above.
    design_path = OUTPUT / "design.json"  # Set design path to the path to 'design.json' inside OUTPUT.
    if design_path.exists() and json.loads(design_path.read_text()) != design:  # Use this branch when the result of design path.exists and json.loads applied to the result of design path.read text differs from design.
        raise ValueError("The recorded example grid must not silently change")  # Reject this invalid case instead of continuing with misleading experiment results.
    save_json(design_path, design)  # Save the following results/settings as json: design path, design.
    records = []  # Set records to .
    for budget in design["budgets"]:  # Each budget gets its own results. Keep 40K and 100K separate until the paired comparison.
        for training_seed in design["training_seeds"]:  # Process each the independently trained model ID from the entries of design selected by 'training_seeds'.
            for policy in POLICIES:  # A policy is a decision rule. Several rules can share the very same QR-DQN network.
                checkpoint = model_path(budget, policy, training_seed)  # Set checkpoint to model path applied to the environment-interaction budget, policy, the independently trained model ID.
                model_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()  # Set model hash to the result of hashlib.sha256(checkpoint.read bytes()).hexdigest.
                agent, _ = load_agent(checkpoint)  # Set agent, the unused result to the frozen learner and checkpoint loaded from checkpoint.
                for density in design["densities"]:  # Traffic density changes vehicle spacing. It does not add more network inputs.
                    for scenario in design["scenario_seeds"]:  # A scenario seed fixes the starting traffic. Reuse the same IDs for fair policy comparisons.
                        key = (  # Set key to f'{budget} {training seed} {policy} {density:g} {scenario}'.
                            f"{budget}_{training_seed}_{policy}_{density:g}_{scenario}"  # Include f'{budget} {training seed} {policy} {density:g} {scenario}' in the surrounding operation.
                        )  # Finish assembling the grouped values above.
                        path = OUTPUT / (key + ".json")  # Set path to (OUTPUT divided by (key plus '.json')).
                        if path.exists():  # A saved filename is not enough: the checks below verify the contents belong to this experiment.
                            record = json.loads(path.read_text())  # Set record to json.loads applied to the result of path.read text.
                            if (  # Use this branch when the entries of record selected by 'model_hash' differs from model hash or the entries of record selected by 'key' differs from key or the entries of record selected by 'budget' differs from the environment-interaction budget or the entries of record selected by 'training_seed' differs from the independently trained model ID or the entries of record selected by 'policy' differs from policy or the entries of record selected by 'density' differs from the traffic-density setting or the entries of record selected by 'scenario' differs from the starting-traffic seed.
                                record["model_hash"] != model_hash  # Include the entries of record selected by 'model_hash' differs from model hash or the entries of record selected by 'key' differs from key or the entries of record selected by 'budget' differs from the environment-interaction budget or the entries of record selected by 'training_seed' differs from the independently trained model ID or the entries of record selected by 'policy' differs from policy or the entries of record selected by 'density' differs from the traffic-density setting or the entries of record selected by 'scenario' differs from the starting-traffic seed in the surrounding operation.
                                or record["key"] != key  # Include the entries of record selected by 'key' differs from key in the surrounding operation.
                                or record["budget"] != budget  # Include the entries of record selected by 'budget' differs from the environment-interaction budget in the surrounding operation.
                                or record["training_seed"] != training_seed  # Include the entries of record selected by 'training_seed' differs from the independently trained model ID in the surrounding operation.
                                or record["policy"] != policy  # Include the entries of record selected by 'policy' differs from policy in the surrounding operation.
                                or record["density"] != density  # Include the entries of record selected by 'density' differs from the traffic-density setting in the surrounding operation.
                                or record["scenario"] != scenario  # Include the entries of record selected by 'scenario' differs from the starting-traffic seed in the surrounding operation.
                            ):  # Finish assembling the grouped values above.
                                raise ValueError("A recording belongs to another model")  # Reject this invalid case instead of continuing with misleading experiment results.
                        else:  # Handle the alternative when the earlier condition (the result of path.exists) is false.
                            record = {  # Set record to the named settings or measurements assembled below.
                                "key": key,  # Record 'key' as key.
                                "budget": budget,  # Record 'budget' as the environment-interaction budget.
                                "training_seed": training_seed,  # Record 'training_seed' as the independently trained model ID.
                                "policy": policy,  # Record 'policy' as policy.
                                "density": density,  # Record 'density' as the traffic-density setting.
                                "scenario": scenario,  # Record 'scenario' as the starting-traffic seed.
                                "model_hash": model_hash,  # Record 'model_hash' as model hash.
                                **record_episode(agent, policy, density, scenario),  # Include record episode applied to agent, policy, the traffic-density setting, the starting-traffic seed in the surrounding operation.
                            }  # Finish assembling the grouped values above.
                            save_json(path, record)  # Save the following results/settings as json: path, record.
                        for state in record["states"]:  # Process each state from the entries of record selected by 'states'.
                            if not (OUTPUT / state["frame"]).exists():  # Use this branch when not the result of (OUTPUT / state['frame']).exists.
                                raise ValueError(  # Reject this invalid case instead of continuing with misleading experiment results.
                                    "A cached recording is missing a frame"  # Include 'A cached recording is missing a frame' in the surrounding operation.
                                )  # Finish assembling ValueError inputs.
                        records.append(record)  # Add record to records.
                print(f"Recorded {budget} {training_seed} {policy}", flush=True)  # Print applied to f'recorded {budget} {training seed} {policy}'.
    save_json(OUTPUT / "recordings.json", records)  # Save the following results/settings as json: the path to 'recordings.json' inside output, records.
    viewer = ROOT / "results/report"  # Set viewer to the path to 'results/report' inside ROOT.
    backup = ROOT / "results/long_training-report"  # Set backup to the path to 'results/long_training-report' inside ROOT.
    if not backup.exists():  # Use this branch when not the result of backup.exists.
        shutil.copytree(viewer, backup)  # Shutil.copytree applied to viewer, backup.
    template = (ROOT / "driving_rl/decision_viewer.html").read_text()  # Set template to the result of (ROOT / 'driving rl/decision viewer.html').read text.
    payload = json.dumps(records).replace("<", "\\u003c")  # Set payload to json.dumps(records).replace applied to '<', '\\u003c'.
    evidence = (  # Set evidence to (ROOT / 'results/long training evaluation/summary.json').read text().replace applied to '<', '\\u003c'.
        (ROOT / "results/long_training_evaluation/summary.json")  # Include (ROOT / 'results/long training evaluation/summary.json').read text().replace applied to '<', '\\u003c' in the surrounding operation.
        .read_text()  # Continue the calculation or input list started above.
        .replace("<", "\\u003c")  # Include '<', '\\u003c' in the surrounding operation.
    )  # Finish assembling the grouped values above.
    page = template.replace("__RECORDINGS__", payload).replace("__EVIDENCE__", evidence)  # The long string below is page layout, not reinforcement learning. Later lines insert the measured numbers.
    (viewer / "index.html").write_text(page)  # (viewer / 'index.html').write text applied to page.
    print(f"Viewer saved with {len(records)} development recordings.", flush=True)  # Print applied to f'viewer saved with {len(records)} development recordings.'.


if __name__ == "__main__":  # Run this workflow when launched from Terminal; importing its helpers must not launch experiments.
    main()  # The result of main.
