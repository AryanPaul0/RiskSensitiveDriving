"""Record development-only driving examples and build the decision viewer.

This script does not train, tune a policy, or use new confirmation scenarios.
Each recording saves what the network saw, predicted, and actually did.
"""

from pathlib import Path
import hashlib
import json
import shutil

import numpy as np
from PIL import Image
import torch

from driving_rl.agents import action_scores
from driving_rl.envs import (
    make_env,
    flatten,
    govern_action,
    render_frame,
    seed_everything,
)
from driving_rl.experiment import load_agent, save_json
from driving_rl.diagnostics import front_state

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results/decision_viewer"
POLICIES = {
    "qr_tail": ("qrdqn", 0.25, None),
    "qr_mean": ("qrdqn", 1.0, None),
    "qr_governed": ("qrdqn", 1.0, 25.0),
    "dqn_governed": ("dqn", 1.0, 25.0),
}


def model_path(budget, policy, seed):
    """Several decision rules share the same learned QR network."""
    algorithm = POLICIES[policy][0]
    if budget == 100000:
        folder = "long_training_100k"
    elif algorithm == "qrdqn":
        folder = "density_coverage"
    else:
        folder = "baseline_controls"
    return ROOT / f"results/{folder}/{algorithm}_seed{seed}/final.pt"


def save_frame(env):
    """Reuse identical pictures rather than saving repeated initial frames."""
    pixels = render_frame(env)
    name = hashlib.sha256(pixels.tobytes()).hexdigest() + ".png"
    path = OUTPUT / "frames" / name
    if not path.exists():
        Image.fromarray(pixels).save(path)
    return "frames/" + name


def record_episode(agent, policy, density, scenario):
    """Keep pre-action predictions paired with the action they really produced.

    Other policies soon reach different states. Their recordings are not
    counterfactual outcomes from the currently selected policy's later state.
    """
    algorithm, alpha, cap = POLICIES[policy]
    env = make_env(density=density, render=True)
    observation, _ = env.reset(seed=scenario)
    states = []
    rewards = []
    try:
        while True:
            # We are asking the saved network for predictions. No learning or weight update happens here.
            with torch.no_grad():
                inputs = torch.as_tensor(flatten(observation)).unsqueeze(0)
                predictions = agent.online(inputs)
                proposed = int(action_scores(predictions, alpha).argmax().item())
            action = proposed
            # The governor is an optional action override. Its threshold is not a guaranteed physical speed limit.
            if cap is not None:
                action = govern_action(env, observation, proposed, cap)
            state = {
                "time": len(rewards),
                "frame": save_frame(env),
                "predictions": predictions[0].tolist(),
                "proposed_action": proposed,
                "action": action,
                "governor_override": action != proposed,
                "lane": int(env.unwrapped.vehicle.lane_index[2]),
                "observation": np.asarray(observation).tolist(),
                **front_state(env),
            }
            observation, reward, terminated, truncated, _ = env.step(action)
            state["reward_after_action"] = float(reward)
            states.append(state)
            rewards.append(float(reward))
            if terminated or truncated:
                # The final picture has no issued action or predicted decision.
                states.append(
                    {
                        "time": len(rewards),
                        "frame": save_frame(env),
                        "action": None,
                        "predictions": None,
                        "lane": int(env.unwrapped.vehicle.lane_index[2]),
                        **front_state(env),
                    }
                )
                return {
                    "states": states,
                    "rewards": rewards,
                    "crashed": bool(env.unwrapped.vehicle.crashed),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
    finally:
        env.close()


def main():
    """Record a fixed grid of examples, then make a self-contained local page."""
    seed_everything(0)
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "frames").mkdir(exist_ok=True)
    design = {
        "budgets": [40000, 100000],
        "training_seeds": [0, 1, 2],
        "densities": [1.0, 2.0, 2.5],
        "scenario_seeds": [18000, 18001],
        "policy_names": list(POLICIES),
        "purpose": "Illustrative development examples chosen before outcomes; not performance evidence or policy tuning.",
        "confirmation_seeds_used": False,
    }
    design_path = OUTPUT / "design.json"
    if design_path.exists() and json.loads(design_path.read_text()) != design:
        raise ValueError("The recorded example grid must not silently change")
    save_json(design_path, design)
    records = []
    # Each budget gets its own results. Keep 40K and 100K separate until the paired comparison.
    for budget in design["budgets"]:
        for training_seed in design["training_seeds"]:
            # A policy is a decision rule. Several rules can share the very same QR-DQN network.
            for policy in POLICIES:
                checkpoint = model_path(budget, policy, training_seed)
                model_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                agent, _ = load_agent(checkpoint)
                # Traffic density changes vehicle spacing. It does not add more network inputs.
                for density in design["densities"]:
                    # A scenario seed fixes the starting traffic. Reuse the same IDs for fair policy comparisons.
                    for scenario in design["scenario_seeds"]:
                        key = (
                            f"{budget}_{training_seed}_{policy}_{density:g}_{scenario}"
                        )
                        path = OUTPUT / (key + ".json")
                        # A saved filename is not enough: the checks below verify the contents belong to this experiment.
                        if path.exists():
                            record = json.loads(path.read_text())
                            if (
                                record["model_hash"] != model_hash
                                or record["key"] != key
                                or record["budget"] != budget
                                or record["training_seed"] != training_seed
                                or record["policy"] != policy
                                or record["density"] != density
                                or record["scenario"] != scenario
                            ):
                                raise ValueError("A recording belongs to another model")
                        else:
                            record = {
                                "key": key,
                                "budget": budget,
                                "training_seed": training_seed,
                                "policy": policy,
                                "density": density,
                                "scenario": scenario,
                                "model_hash": model_hash,
                                **record_episode(agent, policy, density, scenario),
                            }
                            save_json(path, record)
                        for state in record["states"]:
                            if not (OUTPUT / state["frame"]).exists():
                                raise ValueError(
                                    "A cached recording is missing a frame"
                                )
                        records.append(record)
                print(f"Recorded {budget} {training_seed} {policy}", flush=True)
    save_json(OUTPUT / "recordings.json", records)
    viewer = ROOT / "results/report"
    backup = ROOT / "results/long_training-report"
    if not backup.exists():
        shutil.copytree(viewer, backup)
    template = (ROOT / "driving_rl/decision_viewer.html").read_text()
    payload = json.dumps(records).replace("<", "\\u003c")
    evidence = (
        (ROOT / "results/long_training_evaluation/summary.json")
        .read_text()
        .replace("<", "\\u003c")
    )
    # The long string below is page layout, not reinforcement learning. Later lines insert the measured numbers.
    page = template.replace("__RECORDINGS__", payload).replace("__EVIDENCE__", evidence)
    (viewer / "index.html").write_text(page)
    print(f"Viewer saved with {len(records)} development recordings.", flush=True)


if __name__ == "__main__":
    main()
