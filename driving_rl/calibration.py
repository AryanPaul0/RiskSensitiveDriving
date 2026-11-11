"""Check whether predicted future rewards agree with what actually happens.

A crash ends the driving task. A 30-second cutoff merely ends our recording.
For a surviving car we therefore know only a range for the complete future
return, not its exact value. We keep those surviving cars in the analysis.
The intervals here bound missing future rewards; they are not confidence
intervals about how much data we have collected.

Original implementation notes:
Initial-state quantile diagnostics with bounded, censored continuation.
"""

# import makes these modules available; "as" gives a module a shorter local name.
import argparse
# import makes these modules available; "as" gives a module a shorter local name.
import hashlib
# from pathlib import brings the named tools into this file; parentheses allow several lines.
from pathlib import Path

# import makes these modules available; "as" gives a module a shorter local name.
import numpy as np
# import makes these modules available; "as" gives a module a shorter local name.
import torch

# from agents import brings the named tools into this file; parentheses allow several lines.
from .agents import action_scores
# from diagnostics import brings the named tools into this file; parentheses allow several lines.
from .diagnostics import front_state
# from envs import brings the named tools into this file; parentheses allow several lines.
from .envs import flatten, govern_action, make_env, seed_everything
# from experiment import brings the named tools into this file; parentheses allow several lines.
from .experiment import load_agent, save_json, source_hashes, versions


# def defines discounted_return_bounds; parentheses list its inputs, and the colon starts its indented body.
def discounted_return_bounds(rewards, gamma, terminated):
    """Add the recorded rewards, weighting step t by gamma**t.
    A terminal crash gives an exact return. A recording cutoff leaves an unknown
    continuation. Bound that continuation using the task's known reward limits.

    Original implementation notes:
    Bound return beyond a sampling cutoff under this task's reward structure.

    Nonterminal rewards are nonnegative and at most .4. A collision terminates
    with reward >= -2, so any future return is in [-2, .4/(1-gamma)].
    Termination is exact; a time-limit truncation retains these bounds.
    """
    # if checks not 0 <= gamma < 1; only a true condition runs the indented block.
    if not 0 <= gamma < 1:
        # raise signals an error and stops this normal path, so invalid data are not silently accepted.
        raise ValueError("gamma must be in [0,1)")
    # Rewards already observed are known; later rewards beyond the cutoff are not.
    # = stores a result in prefix; call sum using parentheses to supply its inputs.
    prefix = sum(gamma**index * reward for index, reward in enumerate(rewards))
    # if checks terminated; only a true condition runs the indented block.
    if terminated:
        # return ends this function and sends its result back to the code that called it.
        return float(prefix), float(prefix)
    # Any missing future return starts after all recorded steps, so discount the whole remainder.
    # = stores a result in weight; evaluate the expression on the right before using its result.
    weight = gamma ** len(rewards)
    # return ends this function and sends its result back to the code that called it.
    return float(prefix - 2 * weight), float(prefix + 0.4 / (1 - gamma) * weight)


# def defines quantile_compatibility; parentheses list its inputs, and the colon starts its indented body.
def quantile_compatibility(quantiles, lower_returns, upper_returns):
    """Check whether each predicted quantile could agree with the observed returns.
    Some returns are ranges because recording stopped early. Equal values also
    matter: a strict less-than sign and a less-than-or-equal sign are different.
    These brackets handle both issues; they are not sampling confidence bands.

    Original implementation notes:
    Marginal quantile-compatibility bracket, including censoring and ties.

    A true tau-quantile satisfies P(G<q) <= tau <= P(G<=q). With G in [L,U],
    P(U<q) lower-bounds the first probability and P(L<=q) upper-bounds the
    second. Their empirical bracket handles point masses without falsely
    treating ties as prediction error. It is not a sampling confidence band.
    """
    # = stores a result in quantiles; call np.asarray using parentheses to supply its inputs.
    quantiles = np.asarray(quantiles, dtype=float)
    # = stores a result in low; call np.asarray using parentheses to supply its inputs.
    low = np.asarray(lower_returns, dtype=float)
    # = stores a result in high; call np.asarray using parentheses to supply its inputs.
    high = np.asarray(upper_returns, dtype=float)
    # if checks quantiles.ndim != 2 or low.ndim != 1 or high.ndim != 1 or (len(low) != len(quantiles)) or (len(hi...; only a true condition runs the indented block.
    if (
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        quantiles.ndim != 2
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        or low.ndim != 1
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        or high.ndim != 1
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        or len(low) != len(quantiles)
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        or len(high) != len(low)
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    ):
        # raise signals an error and stops this normal path, so invalid data are not silently accepted.
        raise ValueError("Incompatible quantile and return shapes")
    # if checks not len(low) or np.any(low > high) or (not np.isfinite(quantiles).all()) or (not np.isfinite(low)...; only a true condition runs the indented block.
    if (
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        not len(low)
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        or np.any(low > high)
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        or not np.isfinite(quantiles).all()
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        or not np.isfinite(low).all()
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        or not np.isfinite(high).all()
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    ):
        # raise signals an error and stops this normal path, so invalid data are not silently accepted.
        raise ValueError("Expected finite, nonempty ordered return bounds")
    # Strict and non-strict comparisons preserve ties in discrete or repeated returns.
    # = stores a result in lower; call (high[:, None] < quantiles).mean using parentheses to supply its inputs.
    lower = (high[:, None] < quantiles).mean(axis=0)
    # = stores a result in upper; call (low[:, None] <= quantiles).mean using parentheses to supply its inputs.
    upper = (low[:, None] <= quantiles).mean(axis=0)
    # = stores a result in taus; apply / to these operands; numbers are divided, while Path objects join folder names.
    taus = (np.arange(quantiles.shape[1]) + 0.5) / quantiles.shape[1]
    # = stores a result in gap; call np.maximum using parentheses to supply its inputs.
    gap = np.maximum(0.0, np.maximum(lower - taus, taus - upper))
    # return ends this function and sends its result back to the code that called it.
    return {
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "taus": taus.tolist(),
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "coverage_lower": lower.tolist(),
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "coverage_upper": upper.tolist(),
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "minimum_mean_absolute_compatibility_gap": float(gap.mean()),
    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
    }


# def defines summarize_rows; parentheses list its inputs, and the colon starts its indented body.
def summarize_rows(rows):
    """Summarize forecast error and driving outcomes from recorded rollouts.
    Predicted mean minus realized return measures optimism when positive. Keep the
    unknown future-return range instead of replacing surviving episodes with zero.
    """
    # = stores a result in quantiles; call np.array using parentheses to supply its inputs.
    quantiles = np.array([row["initial_quantiles"] for row in rows])
    # = stores a result in lower; call np.array using parentheses to supply its inputs.
    lower = np.array([row["return_lower"] for row in rows])
    # = stores a result in upper; call np.array using parentheses to supply its inputs.
    upper = np.array([row["return_upper"] for row in rows])
    # Average all predicted quantiles to obtain the network's predicted mean return.
    # = stores a result in means; call quantiles.mean using parentheses to supply its inputs.
    means = quantiles.mean(axis=1)
    # = stores a result in crashes; collect or generate one result per item; the "for" part names the item being processed.
    crashes = [row for row in rows if row["crashed"]]
    # return ends this function and sends its result back to the code that called it.
    return {
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "episodes": len(rows),
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        **quantile_compatibility(quantiles, lower, upper),
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "mean_prediction": float(means.mean()),
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "mean_return_lower": float(lower.mean()),
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "mean_return_upper": float(upper.mean()),
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "mean_prediction_bias_bounds": [
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            float((means - upper).mean()),
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            float((means - lower).mean()),
        # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
        ],
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "censored_fraction": float(np.mean([row["censored"] for row in rows])),
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "quantile_crossing_fraction": float(np.mean(np.diff(quantiles, axis=1) < 0)),
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "crash_rate": float(np.mean([row["crashed"] for row in rows])),
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "mean_speed_m_s": float(np.mean([row["mean_speed_m_s"] for row in rows])),
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "mean_duration_s": float(np.mean([row["length"] for row in rows])),
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "crash_within_3_seconds_fraction": float(
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            np.mean([row["crashed"] and row["length"] <= 3 for row in rows])
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        ),
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "recent_lateral_request_given_crash": (
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            float(np.mean([row["recent_lateral_request"] for row in crashes]))
            # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
            if crashes
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            else None
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        ),
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "recent_lane_transition_given_crash": (
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            float(np.mean([row["recent_lane_transition"] for row in crashes]))
            # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
            if crashes
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            else None
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        ),
    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
    }


# def defines terminal_neighbor; parentheses list its inputs, and the colon starts its indented body.
def terminal_neighbor(env):
    """Carry out the terminal neighbor part of this module's workflow.
    See the calling function for the settings and saved files used here.
    """
    # = stores a result in ego; use a dot to access vehicle on env.unwrapped.
    ego = env.unwrapped.vehicle
    # = stores a result in others; collect or generate one result per item; the "for" part names the item being processed.
    others = [vehicle for vehicle in env.unwrapped.road.vehicles if vehicle is not ego]
    # if checks not others; only a true condition runs the indented block.
    if not others:
        # return ends this function and sends its result back to the code that called it.
        return None
    # = stores a result in neighbor; call min using parentheses to supply its inputs.
    neighbor = min(
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        others, key=lambda vehicle: np.linalg.norm(vehicle.position - ego.position)
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # = stores a result in offset; combine the operands with the displayed operator; parentheses control grouping.
    offset = neighbor.position - ego.position
    # return ends this function and sends its result back to the code that called it.
    return {
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "longitudinal_offset_m": float(offset[0]),
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "lateral_offset_m": float(offset[1]),
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "distance_m": float(np.linalg.norm(offset)),
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "position_proxy": (
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "front" if offset[0] > 2.5 else "rear" if offset[0] < -2.5 else "side"
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        ),
    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
    }


# def defines collect; parentheses list its inputs, and the colon starts its indented body.
def collect(args):
    """Record initial return predictions and the resulting driving outcomes.
    Use fixed evaluation seeds, keep terminal events distinct from recording
    cutoffs, and save enough detail to check the result later.
    """
    # if checks Path(args.out).exists(); only a true condition runs the indented block.
    if Path(args.out).exists():
        # raise signals an error and stops this normal path, so invalid data are not silently accepted.
        raise ValueError(f"Refusing to overwrite {args.out}")
    # if checks args.seed_start < 10000 or args.seed_start + args.episodes > 20000; only a true condition runs the indented block.
    if args.seed_start < 10000 or args.seed_start + args.episodes > 20000:
        # raise signals an error and stops this normal path, so invalid data are not silently accepted.
        raise ValueError("Calibration must remain in development block [10000,20000)")
    # Call seed_everything using parentheses to supply its inputs; this line does not store a separate return value.
    seed_everything(0)
    # = unpacks the right-hand result into (agent, checkpoint), matching values to names in order.
    agent, checkpoint = load_agent(args.checkpoint)
    # if checks agent.algorithm != 'qrdqn'; only a true condition runs the indented block.
    if agent.algorithm != "qrdqn":
        # raise signals an error and stops this normal path, so invalid data are not silently accepted.
        raise ValueError("Quantile diagnostics require QR-DQN")
    # = stores a result in gamma; use square brackets to select part of checkpoint['config'].
    gamma = checkpoint["config"]["gamma"]
    # = stores a result in rows; build a list, keeping its items in order.
    rows = []
    # for takes each item from args.densities in turn; density holds the current item.
    for density in args.densities:
        # = stores a result in env; call make_env using parentheses to supply its inputs.
        env = make_env(density=density)
        # = stores a result in config; use a dot to access config on env.unwrapped.
        config = env.unwrapped.config
        # if checks config['collision_reward'] != -2 or config['high_speed_reward'] != 0.4 or config['normalize_rewar...; only a true condition runs the indented block.
        if (
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            config["collision_reward"] != -2
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            or config["high_speed_reward"] != 0.4
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            or config["normalize_reward"]
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            or config["right_lane_reward"] != 0
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        ):
            # raise signals an error and stops this normal path, so invalid data are not silently accepted.
            raise ValueError("Reward assumptions for continuation bounds do not hold")
        # for takes each item from range(args.episodes) in turn; index holds the current item.
        for index in range(args.episodes):
            # = unpacks the right-hand result into (observation, _), matching values to names in order.
            observation, _ = env.reset(seed=args.seed_start + index)
            # = unpacks the right-hand result into (rewards, states, speeds), matching values to names in order.
            rewards, states, speeds = [], [], []
            # = stores a result in initial_quantiles; use the literal value None.
            initial_quantiles = None
            # while repeats the indented block as long as True is true.
            while True:
                # with temporarily disables gradient tracking for this block; these predictions are fixed answers, not a second training path.
                with torch.no_grad():
                    # = stores a result in distribution; call agent.online using parentheses to supply its inputs.
                    distribution = agent.online(
                        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                        torch.as_tensor(flatten(observation)).unsqueeze(0)
                    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                    )
                    # = stores a result in action; call int using parentheses to supply its inputs.
                    action = int(
                        # Inside this continued call, name=value supplies an option by name rather than by its position.
                        action_scores(distribution, args.alpha).argmax(dim=1).item()
                    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                    )
                # if checks args.speed_cap is not None; only a true condition runs the indented block.
                if args.speed_cap is not None:
                    # = stores a result in action; call govern_action using parentheses to supply its inputs.
                    action = govern_action(env, observation, action, args.speed_cap)
                # if checks initial_quantiles is None; only a true condition runs the indented block.
                if initial_quantiles is None:
                    # Save the prediction before observing the outcome; using a later prediction would leak information.
                    # = stores a result in initial_quantiles; call distribution[0, action].tolist using parentheses to supply its inputs.
                    initial_quantiles = distribution[0, action].tolist()
                # = stores a result in vehicle; use a dot to access vehicle on env.unwrapped.
                vehicle = env.unwrapped.vehicle
                # Call states.append using parentheses to supply its inputs; this line does not store a separate return value.
                states.append(
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    {
                        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                        "step": len(rewards),
                        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                        "action": action,
                        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                        "lane": int(vehicle.lane_index[2]),
                        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                        "target_lane": int(vehicle.target_lane_index[2]),
                        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                        "observed_vehicle_slots": int(
                            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                            np.asarray(observation)[:, 0].sum()
                        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                        ),
                        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
                        **front_state(env),
                    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
                    }
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                )
                # = unpacks the right-hand result into (observation, reward, terminated, truncated, _), matching values to names in order.
                observation, reward, terminated, truncated, _ = env.step(action)
                # if checks not -2.000001 <= reward <= 0.400001 or (reward < 0 and (not terminated)); only a true condition runs the indented block.
                if not -2.000001 <= reward <= 0.400001 or (
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    reward < 0 and not terminated
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                ):
                    # raise signals an error and stops this normal path, so invalid data are not silently accepted.
                    raise ValueError(
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        "Observed reward violates continuation-bound assumptions"
                    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                    )
                # Call rewards.append using parentheses to supply its inputs; this line does not store a separate return value.
                rewards.append(float(reward))
                # Call speeds.append using parentheses to supply its inputs; this line does not store a separate return value.
                speeds.append(float(env.unwrapped.vehicle.speed))
                # if checks terminated or truncated; only a true condition runs the indented block.
                if terminated or truncated:
                    # A terminal return is exact. A surviving recording contributes a range, not a fake terminal reward.
                    # = unpacks the right-hand result into (lower, upper), matching values to names in order.
                    lower, upper = discounted_return_bounds(rewards, gamma, terminated)
                    # = stores a result in recent; use square brackets to select part of states.
                    recent = states[-3:]
                    # = stores a result in lanes; combine the operands with the displayed operator; parentheses control grouping.
                    lanes = [state["lane"] for state in recent] + [
                        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                        int(env.unwrapped.vehicle.lane_index[2])
                    # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
                    ]
                    # Call rows.append using parentheses to supply its inputs; this line does not store a separate return value.
                    rows.append(
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        {
                            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                            "density": density,
                            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                            "seed": args.seed_start + index,
                            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                            "initial_quantiles": initial_quantiles,
                            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                            "return_lower": lower,
                            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                            "return_upper": upper,
                            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                            "censored": bool(truncated and not terminated),
                            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                            "crashed": bool(env.unwrapped.vehicle.crashed),
                            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                            "length": len(rewards),
                            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                            "rewards": rewards,
                            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                            "mean_speed_m_s": float(np.mean(speeds)),
                            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                            "recent_states": recent,
                            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                            "recent_lateral_request": any(
                                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                                state["action"] in [0, 2] for state in recent
                            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                            ),
                            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                            "recent_lane_transition": len(set(lanes)) > 1,
                            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                            "terminal_nearest_neighbor": (
                                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                                terminal_neighbor(env) if terminated else None
                            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                            ),
                        # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
                        }
                    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                    )
                    # break leaves the nearest loop immediately; execution continues after that loop.
                    break
        # Call env.close using parentheses to supply its inputs; this line does not store a separate return value.
        env.close()
    # = stores a result in provenance; build a dictionary, which stores values under named keys.
    provenance = {
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "checkpoint": str(args.checkpoint),
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "checkpoint_sha256": hashlib.sha256(
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            Path(args.checkpoint).read_bytes()
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        ).hexdigest(),
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "training_seed": checkpoint["config"]["seed"],
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "training_steps": checkpoint["step"],
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "alpha": args.alpha,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "speed_cap": args.speed_cap,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "gamma": gamma,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "seed_start": args.seed_start,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "episodes_per_density": args.episodes,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "source_hashes": source_hashes(),
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "versions": versions(),
    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
    }
    # = stores a result in summary; collect or generate one result per item; the "for" part names the item being processed.
    summary = [
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        {
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "density": density,
            # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
            **summarize_rows([row for row in rows if row["density"] == density]),
        # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
        }
        # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
        for density in args.densities
    # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
    ]
    # Call save_json using parentheses to supply its inputs; this line does not store a separate return value.
    save_json(
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        args.out,
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        {
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "provenance": provenance,
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "summary": summary,
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "episodes": rows,
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "interpretation": "Marginal quantile compatibility brackets include censoring and ties, not sampling uncertainty. Changed deployment policies differ from training continuation. Terminal geometry and recent actions are descriptive proxies.",
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
    parser.add_argument("--speed-cap", type=float)
    # Call parser.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    parser.add_argument("--episodes", type=int, default=60)
    # Call parser.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    parser.add_argument("--seed-start", type=int, default=12000)
    # Call parser.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    parser.add_argument("--densities", nargs="+", type=float, default=[1.0, 1.5, 2.0])
    # Call parser.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    parser.add_argument("--out", required=True)
    # = stores a result in args; call parser.parse_args using parentheses to supply its inputs.
    args = parser.parse_args()
    # if checks args.episodes < 1 or not 0 < args.alpha <= 1; only a true condition runs the indented block.
    if args.episodes < 1 or not 0 < args.alpha <= 1:
        # Call parser.error using parentheses to supply its inputs; this line does not store a separate return value.
        parser.error("Require episodes > 0 and alpha in (0,1]")
    # if checks args.speed_cap is not None and (not np.isfinite(args.speed_cap) or args.speed_cap <= 0); only a true condition runs the indented block.
    if args.speed_cap is not None and (
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        not np.isfinite(args.speed_cap) or args.speed_cap <= 0
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    ):
        # Call parser.error using parentheses to supply its inputs; this line does not store a separate return value.
        parser.error("Require a finite positive speed cap")
    # Call collect using parentheses to supply its inputs; this line does not store a separate return value.
    collect(args)


# Run the entry point only when this file is executed, not when it is imported.
# __name__ equals "__main__" when this file runs directly; this guard avoids running the script merely because another file imports it.
if __name__ == "__main__":
    # Call main using parentheses to supply its inputs; this line does not store a separate return value.
    main()
