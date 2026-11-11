"""Look inside recorded decisions to understand behavior.

Extra simulator information, such as the true distance to another car, is used
for measurements only. It is not secretly given to the learned policy. A small
gap or a recent lane change can describe a crash without proving its cause.

Original implementation notes:
Inspect policy behavior using privileged state only for measurement.
"""

# import makes these modules available; "as" gives a module a shorter local name.
import argparse
# from pathlib import brings the named tools into this file; parentheses allow several lines.
from pathlib import Path

# import makes these modules available; "as" gives a module a shorter local name.
import numpy as np
# import makes these modules available; "as" gives a module a shorter local name.
import torch

# from agents import brings the named tools into this file; parentheses allow several lines.
from .agents import action_scores
# from envs import brings the named tools into this file; parentheses allow several lines.
from .envs import flatten, make_env, seed_everything
# from experiment import brings the named tools into this file; parentheses allow several lines.
from .experiment import load_agent, save_json, source_hashes


# def defines front_state; parentheses list its inputs, and the colon starts its indented body.
def front_state(env):
    """Measure the nearest car ahead and an approximate time to catch it.
    Divide the bumper gap by the closing speed only when the ego car is gaining
    on it. These measurements describe the situation; they do not prove crash causes.
    """
    # = stores a result in vehicle; use a dot to access vehicle on env.unwrapped.
    vehicle = env.unwrapped.vehicle
    # = unpacks the right-hand result into (front, _), matching values to names in order.
    front, _ = env.unwrapped.road.neighbour_vehicles(vehicle)
    # if checks front is None; only a true condition runs the indented block.
    if front is None:
        # return ends this function and sends its result back to the code that called it.
        return {"speed_m_s": float(vehicle.speed), "gap_m": None, "ttc_seconds": None}
    # lane_distance_to is center-to-center along the lane. Subtract half-lengths
    # for a bumper-gap approximation. This is not a full collision predictor.
    # = stores a result in gap; call float using parentheses to supply its inputs.
    gap = float(vehicle.lane_distance_to(front) - (vehicle.LENGTH + front.LENGTH) / 2)
    # = stores a result in closing_speed; call float using parentheses to supply its inputs.
    closing_speed = float(vehicle.speed - front.speed)
    # = stores a result in ttc; choose the value before "if" when its condition is true, otherwise the value after "else".
    ttc = max(0.0, gap) / closing_speed if closing_speed > 0.1 else None
    # return ends this function and sends its result back to the code that called it.
    return {"speed_m_s": float(vehicle.speed), "gap_m": gap, "ttc_seconds": ttc}


# def defines reward_example; parentheses list its inputs, and the colon starts its indented body.
def reward_example(gamma=0.9):
    """Compare two made-up reward sequences to explain the discount factor.
    This illustrates the reward design. It is not a simulator experiment or proof
    of why a particular learned policy chose an action.
    """
    # = stores a result in fast_rewards; combine the operands with the displayed operator; parentheses control grouping.
    fast_rewards = [0.4] * 9 + [-2.0]
    # = stores a result in slow_rewards; multiply the operands; NumPy and PyTorch arrays do this element by element.
    slow_rewards = [0.12] * 30
    # = stores a result in discounted; evaluate the expression on the right before using its result.
    discounted = lambda rewards: sum(
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        gamma**t * value for t, value in enumerate(rewards)
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # return ends this function and sends its result back to the code that called it.
    return {
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "gamma": gamma,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "description": "Illustration only: nine high-speed .4 rewards then a -2 crash, versus thirty .12 rewards (about 23 m/s) with no crash.",
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "fast_crash_discounted_return": discounted(fast_rewards),
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "slow_survival_discounted_return": discounted(slow_rewards),
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "fast_crash_undiscounted_return": sum(fast_rewards),
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "slow_survival_undiscounted_return": sum(slow_rewards),
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "caveat": "Stylized rewards, not counterfactual trajectories from the simulator; this does not prove why a learned policy crashed.",
    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
    }


# def defines diagnose; parentheses list its inputs, and the colon starts its indented body.
def diagnose(args):
    """Record decisions and nearby-traffic measurements to inspect behavior.
    Measurements can use simulator state, but policy actions still use their normal inputs.
    """
    # if checks Path(args.out).exists(); only a true condition runs the indented block.
    if Path(args.out).exists():
        # raise signals an error and stops this normal path, so invalid data are not silently accepted.
        raise ValueError(f"Refusing to overwrite diagnostic: {args.out}")
    # if checks args.seed_start < 10000 or args.seed_start + args.episodes > 20000; only a true condition runs the indented block.
    if args.seed_start < 10000 or args.seed_start + args.episodes > 20000:
        # raise signals an error and stops this normal path, so invalid data are not silently accepted.
        raise ValueError("Diagnostics must use the development block [10000, 20000).")
    # Call seed_everything using parentheses to supply its inputs; this line does not store a separate return value.
    seed_everything(0)
    # = unpacks the right-hand result into (agent, checkpoint), matching values to names in order.
    agent, checkpoint = load_agent(args.checkpoint)
    # = stores a result in rows; build a list, keeping its items in order.
    rows = []
    # for takes each item from args.densities in turn; density holds the current item.
    for density in args.densities:
        # = stores a result in env; call make_env using parentheses to supply its inputs.
        env = make_env(density=density)
        # for takes each item from range(args.episodes) in turn; index holds the current item.
        for index in range(args.episodes):
            # = stores a result in seed; combine the operands with the displayed operator; parentheses control grouping.
            seed = args.seed_start + index
            # = unpacks the right-hand result into (observation, _), matching values to names in order.
            observation, _ = env.reset(seed=seed)
            # = stores a result in steps; build a list, keeping its items in order.
            steps = []
            # while repeats the indented block as long as True is true.
            while True:
                # = stores a result in tensor; call torch.as_tensor(flatten(observation)).unsqueeze using parentheses to supply its inputs.
                tensor = torch.as_tensor(flatten(observation)).unsqueeze(0)
                # with temporarily disables gradient tracking for this block; these predictions are fixed answers, not a second training path.
                with torch.no_grad():
                    # = stores a result in values; call agent.online using parentheses to supply its inputs.
                    values = agent.online(tensor)
                    # = stores a result in means; call action_scores(values)[0].tolist using parentheses to supply its inputs.
                    means = action_scores(values)[0].tolist()
                    # = stores a result in tails; call action_scores(values, 0.25)[0].tolist using parentheses to supply its inputs.
                    tails = action_scores(values, 0.25)[0].tolist()
                    # = stores a result in scores; use square brackets to select part of action_scores(values, args.alpha).
                    scores = action_scores(values, args.alpha)[0]
                # = stores a result in action; call int using parentheses to supply its inputs.
                action = int(scores.argmax().item())
                # = stores a result in record; build a dictionary, which stores values under named keys.
                record = {
                    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                    "step": len(steps),
                    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                    "action": action,
                    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                    "lane": int(env.unwrapped.vehicle.lane_index[2]),
                    # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
                    **front_state(env),
                    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                    "mean_scores": means,
                    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                    "tail25_scores": tails,
                    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                    "selectors_disagree": int(np.argmax(means) != np.argmax(tails)),
                # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
                }
                # = unpacks the right-hand result into (observation, reward, terminated, truncated, _), matching values to names in order.
                observation, reward, terminated, truncated, _ = env.step(action)
                # = stores a result in record['reward']; call float using parentheses to supply its inputs.
                record["reward"] = float(reward)
                # Call steps.append using parentheses to supply its inputs; this line does not store a separate return value.
                steps.append(record)
                # if checks terminated or truncated; only a true condition runs the indented block.
                if terminated or truncated:
                    # Call rows.append using parentheses to supply its inputs; this line does not store a separate return value.
                    rows.append(
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        {
                            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                            "density": density,
                            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                            "seed": seed,
                            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                            "crashed": bool(env.unwrapped.vehicle.crashed),
                            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                            "length": len(steps),
                            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                            "steps": steps,
                        # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
                        }
                    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                    )
                    # break leaves the nearest loop immediately; execution continues after that loop.
                    break
        # Call env.close using parentheses to supply its inputs; this line does not store a separate return value.
        env.close()
    # = stores a result in summary; build a list, keeping its items in order.
    summary = []
    # for takes each item from args.densities in turn; density holds the current item.
    for density in args.densities:
        # = stores a result in selected; collect or generate one result per item; the "for" part names the item being processed.
        selected = [row for row in rows if row["density"] == density]
        # = stores a result in steps; collect or generate one result per item; the "for" part names the item being processed.
        steps = [step for row in selected for step in row["steps"]]
        # = stores a result in first; collect or generate one result per item; the "for" part names the item being processed.
        first = [row["steps"][0] for row in selected]
        # = stores a result in counts; call np.bincount using parentheses to supply its inputs.
        counts = np.bincount([step["action"] for step in steps], minlength=5)

        # def defines fraction; parentheses list its inputs, and the colon starts its indented body.
        def fraction(items, condition):
            # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
            """Calculate the fraction of recorded steps satisfying the requested condition."""
            # return ends this function and sends its result back to the code that called it.
            return sum(condition(item) for item in items) / len(items)

        # Call summary.append using parentheses to supply its inputs; this line does not store a separate return value.
        summary.append(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            {
                # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                "density": density,
                # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                "episodes": len(selected),
                # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                "crash_rate": fraction(selected, lambda row: row["crashed"]),
                # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                "mean_length": float(np.mean([row["length"] for row in selected])),
                # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                "crash_within_3_seconds_fraction": fraction(
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    selected, lambda row: row["crashed"] and row["length"] <= 3
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                ),
                # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                "action_fractions": (counts / counts.sum()).tolist(),
                # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                "selector_disagreement_fraction": fraction(
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    steps, lambda row: row["selectors_disagree"]
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                ),
                # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                "initial_gap_under_10m_fraction": fraction(
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    first, lambda row: row["gap_m"] is not None and row["gap_m"] < 10
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                ),
                # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                "initial_ttc_under_3s_fraction": fraction(
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    first,
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    lambda row: row["ttc_seconds"] is not None
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    and row["ttc_seconds"] < 3,
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                ),
            # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
            }
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
    # Call save_json using parentheses to supply its inputs; this line does not store a separate return value.
    save_json(
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        args.out,
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        {
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "checkpoint": str(args.checkpoint),
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "training_seed": checkpoint["config"]["seed"],
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "alpha": args.alpha,
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "source_hashes": source_hashes(),
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "action_names": ["LEFT", "IDLE", "RIGHT", "FASTER", "SLOWER"],
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "reward_example": reward_example(),
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "summary": summary,
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "episodes": rows,
        # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
        },
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # Call print using parentheses to supply its inputs; this line does not store a separate return value.
    print(args.out, flush=True)


# def defines main; parentheses list its inputs, and the colon starts its indented body.
def main():
    """Read Terminal options, check settings, and run the requested workflow.
    Existing result files are checked before reuse so a rerun cannot silently mix
    different experiments. The main guard below prevents running jobs on import.
    """
    # = stores a result in parser; call argparse.ArgumentParser using parentheses to supply its inputs.
    parser = argparse.ArgumentParser(description=__doc__)
    # Call parser.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    parser.add_argument("--checkpoint", required=True)
    # Call parser.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    parser.add_argument("--alpha", type=float, default=1.0)
    # Call parser.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    parser.add_argument("--episodes", type=int, default=30)
    # Call parser.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    parser.add_argument("--densities", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    # Call parser.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    parser.add_argument("--seed-start", type=int, default=11000)
    # Call parser.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    parser.add_argument("--out", required=True)
    # = stores a result in args; call parser.parse_args using parentheses to supply its inputs.
    args = parser.parse_args()
    # if checks args.episodes < 1 or not 0 < args.alpha <= 1; only a true condition runs the indented block.
    if args.episodes < 1 or not 0 < args.alpha <= 1:
        # Call parser.error using parentheses to supply its inputs; this line does not store a separate return value.
        parser.error("Require episodes > 0 and alpha in (0,1].")
    # Call diagnose using parentheses to supply its inputs; this line does not store a separate return value.
    diagnose(args)


# Run the entry point only when this file is executed, not when it is imported.
# __name__ equals "__main__" when this file runs directly; this guard avoids running the script merely because another file imports it.
if __name__ == "__main__":
    # Call main using parentheses to supply its inputs; this line does not store a separate return value.
    main()
