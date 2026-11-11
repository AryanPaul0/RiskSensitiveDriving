"""Look inside recorded decisions to understand behavior.

Extra simulator information, such as the true distance to another car, is used
for measurements only. It is not secretly given to the learned policy. A small
gap or a recent lane change can describe a crash without proving its cause.

Original implementation notes:
Inspect policy behavior using privileged state only for measurement.
"""

import argparse  # Make argparse available for this file's calculations and workflow.
from pathlib import Path  # Make Path available for this file's calculations and workflow.

import numpy as np  # Make numpy available for this file's calculations and workflow.
import torch  # Make torch available for this file's calculations and workflow.

from .agents import action_scores  # Make action_scores available for this file's calculations and workflow.
from .envs import flatten, make_env, seed_everything  # Make flatten, make_env, seed_everything available for this file's calculations and workflow.
from .experiment import load_agent, save_json, source_hashes  # Make load_agent, save_json, source_hashes available for this file's calculations and workflow.


def front_state(env):  # Measure the nearest car ahead and an approximate time to catch it.
    """Measure the nearest car ahead and an approximate time to catch it.
    Divide the bumper gap by the closing speed only when the ego car is gaining
    on it. These measurements describe the situation; they do not prove crash causes.
    """
    vehicle = env.unwrapped.vehicle  # Set vehicle to env.unwrapped.vehicle.
    front, _ = env.unwrapped.road.neighbour_vehicles(vehicle)  # Set front, the unused result to env.unwrapped.road.neighbour vehicles applied to vehicle.
    if front is None:  # Use this branch when front is None.
        return {"speed_m_s": float(vehicle.speed), "gap_m": None, "ttc_seconds": None}  # Return the named settings or measurements assembled below.
    gap = float(vehicle.lane_distance_to(front) - (vehicle.LENGTH + front.LENGTH) / 2)  # lane_distance_to is center-to-center along the lane. Subtract half-lengths for a bumper-gap approximation. This is not a full collision predictor.
    closing_speed = float(vehicle.speed - front.speed)  # Set closing speed to (vehicle.speed minus front.speed) as a floating-point value.
    ttc = max(0.0, gap) / closing_speed if closing_speed > 0.1 else None  # Set ttc to (max applied to 0.0, gap divided by closing speed) when closing speed is above 0.1, otherwise None.
    return {"speed_m_s": float(vehicle.speed), "gap_m": gap, "ttc_seconds": ttc}  # Return the named settings or measurements assembled below.


def reward_example(gamma=0.9):  # Compare two made-up reward sequences to explain the discount factor.
    """Compare two made-up reward sequences to explain the discount factor.
    This illustrates the reward design. It is not a simulator experiment or proof
    of why a particular learned policy chose an action.
    """
    fast_rewards = [0.4] * 9 + [-2.0]  # Set fast rewards to ((0.4 times 9) plus negative 2.0).
    slow_rewards = [0.12] * 30  # Set slow rewards to (0.12 times 30).
    discounted = lambda rewards: sum(  # Set discounted to lambda rewards: sum((gamma ** t * value for t, value in enumerate(rewards))).
        gamma**t * value for t, value in enumerate(rewards)  # Include ((the future-reward discount to the power t) times value), t, value, enumerate applied to the observed rewards in the surrounding operation.
    )  # Finish assembling sum inputs.
    return {  # Return the named settings or measurements assembled below.
        "gamma": gamma,  # Record 'gamma' as the future-reward discount.
        "description": "Illustration only: nine high-speed .4 rewards then a -2 crash, versus thirty .12 rewards (about 23 m/s) with no crash.",  # Record 'description' as the report text.
        "fast_crash_discounted_return": discounted(fast_rewards),  # Record 'fast_crash_discounted_return' as discounted applied to fast rewards.
        "slow_survival_discounted_return": discounted(slow_rewards),  # Record 'slow_survival_discounted_return' as discounted applied to slow rewards.
        "fast_crash_undiscounted_return": sum(fast_rewards),  # Record 'fast_crash_undiscounted_return' as sum applied to fast rewards.
        "slow_survival_undiscounted_return": sum(slow_rewards),  # Record 'slow_survival_undiscounted_return' as sum applied to slow rewards.
        "caveat": "Stylized rewards, not counterfactual trajectories from the simulator; this does not prove why a learned policy crashed.",  # Record 'caveat' as the report text.
    }  # Finish assembling the grouped values above.


def diagnose(args):  # Record decisions and nearby-traffic measurements to inspect behavior.
    """Record decisions and nearby-traffic measurements to inspect behavior.
    Measurements can use simulator state, but policy actions still use their normal inputs.
    """
    if Path(args.out).exists():  # Use this branch when the result of Path(args.out).exists.
        raise ValueError(f"Refusing to overwrite diagnostic: {args.out}")  # Reject this invalid case instead of continuing with misleading experiment results.
    if args.seed_start < 10000 or args.seed_start + args.episodes > 20000:  # Use this branch when args.seed start is below 10000 or (args.seed start plus args.episodes) is above 20000.
        raise ValueError("Diagnostics must use the development block [10000, 20000).")  # Reject this invalid case instead of continuing with misleading experiment results.
    seed_everything(0)  # Initialize repeatable random streams using 0.
    agent, checkpoint = load_agent(args.checkpoint)  # Set agent, checkpoint to the frozen learner and checkpoint loaded from args.checkpoint.
    rows = []  # Set rows to .
    for density in args.densities:  # Process each the traffic-density setting from args.densities.
        env = make_env(density=density)  # Set env to a highway simulator configured with.
        for index in range(args.episodes):  # Process each index from range applied to args.episodes.
            seed = args.seed_start + index  # Set seed to (args.seed start plus index).
            observation, _ = env.reset(seed=seed)  # Set the currently observed traffic features, the unused result to the result of env.reset.
            steps = []  # Set steps to .
            while True:  # Keep processing while True.
                tensor = torch.as_tensor(flatten(observation)).unsqueeze(0)  # Set tensor to torch.as tensor(flatten(observation)).unsqueeze applied to 0.
                with torch.no_grad():  # Treat predictions in this block as fixed learning targets or inference results; do not backpropagate through them.
                    values = agent.online(tensor)  # Set values to agent.online applied to tensor.
                    means = action_scores(values)[0].tolist()  # Set means to the entries of the mean or lower-tail action scores computed from values selected by 0 as a list that can be saved in JSON.
                    tails = action_scores(values, 0.25)[0].tolist()  # Set tails to the entries of the mean or lower-tail action scores computed from values, 0.25 selected by 0 as a list that can be saved in JSON.
                    scores = action_scores(values, args.alpha)[0]  # Set scores to the entries of the mean or lower-tail action scores computed from values, args.alpha selected by 0.
                action = int(scores.argmax().item())  # Set action to the index of the highest score in scores extracted as a plain scalar as a whole-number value.
                record = {  # Set record to the named settings or measurements assembled below.
                    "step": len(steps),  # Record 'step' as the number of entries in steps.
                    "action": action,  # Record 'action' as action.
                    "lane": int(env.unwrapped.vehicle.lane_index[2]),  # Record 'lane' as the entries of env.unwrapped.vehicle.lane index selected by 2 as a whole-number value.
                    **front_state(env),  # Include front state applied to env in the surrounding operation.
                    "mean_scores": means,  # Record 'mean_scores' as means.
                    "tail25_scores": tails,  # Record 'tail25_scores' as tails.
                    "selectors_disagree": int(np.argmax(means) != np.argmax(tails)),  # Record 'selectors_disagree' as the index of the highest score in np differs from the index of the highest score in np as a whole-number value.
                }  # Finish assembling the grouped values above.
                observation, reward, terminated, truncated, _ = env.step(action)  # Set the currently observed traffic features, reward, the true-terminal flags, the recording-cutoff flags, the unused result to env.step applied to action.
                record["reward"] = float(reward)  # Set the entries of record selected by 'reward' to reward as a floating-point value.
                steps.append(record)  # Add record to steps.
                if terminated or truncated:  # Use this branch when the true-terminal flags or the recording-cutoff flags.
                    rows.append(  # Add the named settings or measurements assembled below to rows.
                        {  # Include the named settings or measurements assembled below in the surrounding operation.
                            "density": density,  # Record 'density' as the traffic-density setting.
                            "seed": seed,  # Record 'seed' as seed.
                            "crashed": bool(env.unwrapped.vehicle.crashed),  # Record 'crashed' as env.unwrapped.vehicle.crashed as a true/false value.
                            "length": len(steps),  # Record 'length' as the number of entries in steps.
                            "steps": steps,  # Record 'steps' as steps.
                        }  # Finish assembling the grouped values above.
                    )  # Finish assembling rows.append inputs.
                    break  # Stop this loop now that its stopping condition has been reached.
        env.close()  # The result of env.close.
    summary = []  # Set summary to .
    for density in args.densities:  # Process each the traffic-density setting from args.densities.
        selected = [row for row in rows if row["density"] == density]  # Set the selected next-action return predictions to row for each row in rows, keeping only cases where the entries of row selected by 'density' equals the traffic-density setting.
        steps = [step for row in selected for step in row["steps"]]  # Set steps to step for each row in the selected next-action return predictions.
        first = [row["steps"][0] for row in selected]  # Set first to the entries of the entries of row selected by 'steps' selected by 0 for each row in the selected next-action return predictions.
        counts = np.bincount([step["action"] for step in steps], minlength=5)  # Set counts to np.bincount applied to the entries of step selected by 'action' for each step in steps.

        def fraction(items, condition):  # Calculate the fraction of recorded steps satisfying the requested condition.
            """Calculate the fraction of recorded steps satisfying the requested condition."""  # Describe this operation for readers and help().
            return sum(condition(item) for item in items) / len(items)  # Return (sum applied to condition applied to item for each item in items divided by the number of entries in items).

        summary.append(  # Add the named settings or measurements assembled below to summary.
            {  # Include the named settings or measurements assembled below in the surrounding operation.
                "density": density,  # Record 'density' as the traffic-density setting.
                "episodes": len(selected),  # Record 'episodes' as the number of entries in the selected next-action return predictions.
                "crash_rate": fraction(selected, lambda row: row["crashed"]),  # Record 'crash_rate' as fraction applied to the selected next-action return predictions, lambda row: row['crashed'].
                "mean_length": float(np.mean([row["length"] for row in selected])),  # Record 'mean_length' as the average of the entries of row selected by 'length' for each row in the selected next-action return predictions as a floating-point value.
                "crash_within_3_seconds_fraction": fraction(  # Record 'crash_within_3_seconds_fraction' as fraction applied to the selected next-action return predictions, lambda row: row['crashed'] and row['length'] <= 3.
                    selected, lambda row: row["crashed"] and row["length"] <= 3  # Include the selected next-action return predictions, lambda row: row['crashed'] and row['length'] <= 3 in the surrounding operation.
                ),  # Finish assembling fraction inputs.
                "action_fractions": (counts / counts.sum()).tolist(),  # Record 'action_fractions' as (counts divided by the result of counts.sum) as a list that can be saved in JSON.
                "selector_disagreement_fraction": fraction(  # Record 'selector_disagreement_fraction' as fraction applied to steps, lambda row: row['selectors disagree'].
                    steps, lambda row: row["selectors_disagree"]  # Include steps, lambda row: row['selectors disagree'] in the surrounding operation.
                ),  # Finish assembling fraction inputs.
                "initial_gap_under_10m_fraction": fraction(  # Record 'initial_gap_under_10m_fraction' as fraction applied to first, lambda row: row['gap m'] is not None and row['gap m'] < 10.
                    first, lambda row: row["gap_m"] is not None and row["gap_m"] < 10  # Include first, lambda row: row['gap m'] is not None and row['gap m'] < 10 in the surrounding operation.
                ),  # Finish assembling fraction inputs.
                "initial_ttc_under_3s_fraction": fraction(  # Record 'initial_ttc_under_3s_fraction' as fraction applied to first, lambda row: row['ttc seconds'] is not None and row['ttc seconds'] < 3.
                    first,  # Include first in the surrounding operation.
                    lambda row: row["ttc_seconds"] is not None  # Include lambda row: row['ttc seconds'] is not None and row['ttc seconds'] < 3 in the surrounding operation.
                    and row["ttc_seconds"] < 3,  # Include the entries of row selected by 'ttc_seconds' is below 3 in the surrounding operation.
                ),  # Finish assembling fraction inputs.
            }  # Finish assembling the grouped values above.
        )  # Finish assembling summary.append inputs.
    save_json(  # Save the following results/settings as json: args.out, the named settings or measurements assembled below.
        args.out,  # Include args.out in the surrounding operation.
        {  # Include the named settings or measurements assembled below in the surrounding operation.
            "checkpoint": str(args.checkpoint),  # Record 'checkpoint' as args.checkpoint as a text value.
            "training_seed": checkpoint["config"]["seed"],  # Record 'training_seed' as the entries of the entries of checkpoint selected by 'config' selected by 'seed'.
            "alpha": args.alpha,  # Record 'alpha' as args.alpha.
            "source_hashes": source_hashes(),  # Record 'source_hashes' as the result of source hashes.
            "action_names": ["LEFT", "IDLE", "RIGHT", "FASTER", "SLOWER"],  # Record 'action_names' as 'LEFT', 'IDLE', 'RIGHT', 'FASTER', 'SLOWER'.
            "reward_example": reward_example(),  # Record 'reward_example' as the result of reward example.
            "summary": summary,  # Record 'summary' as summary.
            "episodes": rows,  # Record 'episodes' as rows.
        },  # Finish assembling the grouped values above.
    )  # Finish assembling save json inputs.
    print(args.out, flush=True)  # Print applied to args.out.


def main():  # Read Terminal options, check settings, and run the requested workflow.
    """Read Terminal options, check settings, and run the requested workflow.
    Existing result files are checked before reuse so a rerun cannot silently mix
    different experiments. The main guard below prevents running jobs on import.
    """
    parser = argparse.ArgumentParser(description=__doc__)  # Set parser to the result of argparse.ArgumentParser.
    parser.add_argument("--checkpoint", required=True)  # Parser.add argument applied to '--checkpoint'.
    parser.add_argument("--alpha", type=float, default=1.0)  # Parser.add argument applied to '--alpha'.
    parser.add_argument("--episodes", type=int, default=30)  # Parser.add argument applied to '--episodes'.
    parser.add_argument("--densities", type=float, nargs="+", default=[1.0, 1.5, 2.0])  # Parser.add argument applied to '--densities'.
    parser.add_argument("--seed-start", type=int, default=11000)  # Parser.add argument applied to '--seed-start'.
    parser.add_argument("--out", required=True)  # Parser.add argument applied to '--out'.
    args = parser.parse_args()  # Set args to the result of parser.parse args.
    if args.episodes < 1 or not 0 < args.alpha <= 1:  # Use this branch when args.episodes is below 1 or not 0 is below args.alpha is at most 1.
        parser.error("Require episodes > 0 and alpha in (0,1].")  # Parser.error applied to 'require episodes > 0 and alpha in (0,1].'.
    diagnose(args)  # Diagnose applied to args.


if __name__ == "__main__":  # Run this workflow when launched from Terminal; importing its helpers must not launch experiments.
    main()  # The result of main.
