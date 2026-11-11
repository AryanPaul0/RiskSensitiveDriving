"""Inspect policy behavior using privileged state only for measurement."""
import argparse
from pathlib import Path

import numpy as np
import torch

from .agents import action_scores
from .envs import flatten, make_env, seed_everything
from .experiment import load_agent, save_json, source_hashes


def front_state(env):
    vehicle = env.unwrapped.vehicle
    front, _ = env.unwrapped.road.neighbour_vehicles(vehicle)
    if front is None:
        return {"speed_m_s": float(vehicle.speed), "gap_m": None, "ttc_seconds": None}
    # lane_distance_to is center-to-center along the lane. Subtract half-lengths
    # for a bumper-gap approximation. This is not a full collision predictor.
    gap = float(vehicle.lane_distance_to(front) - (vehicle.LENGTH + front.LENGTH) / 2)
    closing_speed = float(vehicle.speed - front.speed)
    ttc = max(0., gap) / closing_speed if closing_speed > .1 else None
    return {"speed_m_s": float(vehicle.speed), "gap_m": gap, "ttc_seconds": ttc}


def reward_example(gamma=.9):
    fast_rewards = [.4] * 9 + [-2.]
    slow_rewards = [.12] * 30
    discounted = lambda rewards: sum(gamma ** t * value for t, value in enumerate(rewards))
    return {"gamma": gamma,
            "description": "Illustration only: nine high-speed .4 rewards then a -2 crash, versus thirty .12 rewards (about 23 m/s) with no crash.",
            "fast_crash_discounted_return": discounted(fast_rewards),
            "slow_survival_discounted_return": discounted(slow_rewards),
            "fast_crash_undiscounted_return": sum(fast_rewards),
            "slow_survival_undiscounted_return": sum(slow_rewards),
            "caveat": "Stylized rewards, not counterfactual trajectories from the simulator; this does not prove why a learned policy crashed."}


def diagnose(args):
    if Path(args.out).exists():
        raise ValueError(f"Refusing to overwrite diagnostic: {args.out}")
    if args.seed_start < 10000 or args.seed_start + args.episodes > 20000:
        raise ValueError("Diagnostics must use the development block [10000, 20000).")
    seed_everything(0)
    agent, checkpoint = load_agent(args.checkpoint)
    rows = []
    for density in args.densities:
        env = make_env(density=density)
        for index in range(args.episodes):
            seed = args.seed_start + index
            observation, _ = env.reset(seed=seed)
            steps = []
            while True:
                tensor = torch.as_tensor(flatten(observation)).unsqueeze(0)
                with torch.no_grad():
                    values = agent.online(tensor)
                    means = action_scores(values)[0].tolist()
                    tails = action_scores(values, .25)[0].tolist()
                    scores = action_scores(values, args.alpha)[0]
                action = int(scores.argmax().item())
                record = {"step": len(steps), "action": action,
                          "lane": int(env.unwrapped.vehicle.lane_index[2]),
                          **front_state(env), "mean_scores": means, "tail25_scores": tails,
                          "selectors_disagree": int(np.argmax(means) != np.argmax(tails))}
                observation, reward, terminated, truncated, _ = env.step(action)
                record["reward"] = float(reward)
                steps.append(record)
                if terminated or truncated:
                    rows.append({"density": density, "seed": seed,
                                 "crashed": bool(env.unwrapped.vehicle.crashed),
                                 "length": len(steps), "steps": steps})
                    break
        env.close()
    summary = []
    for density in args.densities:
        selected = [row for row in rows if row["density"] == density]
        steps = [step for row in selected for step in row["steps"]]
        first = [row["steps"][0] for row in selected]
        counts = np.bincount([step["action"] for step in steps], minlength=5)
        def fraction(items, condition):
            return sum(condition(item) for item in items) / len(items)
        summary.append({"density": density, "episodes": len(selected),
                        "crash_rate": fraction(selected, lambda row: row["crashed"]),
                        "mean_length": float(np.mean([row["length"] for row in selected])),
                        "crash_within_3_seconds_fraction": fraction(selected, lambda row: row["crashed"] and row["length"] <= 3),
                        "action_fractions": (counts / counts.sum()).tolist(),
                        "selector_disagreement_fraction": fraction(steps, lambda row: row["selectors_disagree"]),
                        "initial_gap_under_10m_fraction": fraction(first, lambda row: row["gap_m"] is not None and row["gap_m"] < 10),
                        "initial_ttc_under_3s_fraction": fraction(first, lambda row: row["ttc_seconds"] is not None and row["ttc_seconds"] < 3)})
    save_json(args.out, {"checkpoint": str(args.checkpoint), "training_seed": checkpoint["config"]["seed"],
                         "alpha": args.alpha, "source_hashes": source_hashes(),
                         "action_names": ["LEFT", "IDLE", "RIGHT", "FASTER", "SLOWER"],
                         "reward_example": reward_example(), "summary": summary, "episodes": rows})
    print(args.out, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--alpha", type=float, default=1.)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--densities", type=float, nargs="+", default=[1., 1.5, 2.])
    parser.add_argument("--seed-start", type=int, default=11000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.episodes < 1 or not 0 < args.alpha <= 1:
        parser.error("Require episodes > 0 and alpha in (0,1].")
    diagnose(args)


if __name__ == "__main__":
    main()
