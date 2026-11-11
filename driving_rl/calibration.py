"""Check whether predicted future rewards agree with what actually happens.

A crash ends the driving task. A 30-second cutoff merely ends our recording.
For a surviving car we therefore know only a range for the complete future
return, not its exact value. We keep those surviving cars in the analysis.
The intervals here bound missing future rewards; they are not confidence
intervals about how much data we have collected.

Original implementation notes:
Initial-state quantile diagnostics with bounded, censored continuation.
"""

import argparse  # Make argparse available for this file's calculations and workflow.
import hashlib  # Make hashlib available for this file's calculations and workflow.
from pathlib import Path  # Make Path available for this file's calculations and workflow.

import numpy as np  # Make numpy available for this file's calculations and workflow.
import torch  # Make torch available for this file's calculations and workflow.

from .agents import action_scores  # Make action_scores available for this file's calculations and workflow.
from .diagnostics import front_state  # Make front_state available for this file's calculations and workflow.
from .envs import flatten, govern_action, make_env, seed_everything  # Make flatten, govern_action, make_env, seed_everything available for this file's calculations and workflow.
from .experiment import load_agent, save_json, source_hashes, versions  # Make load_agent, save_json, source_hashes, versions available for this file's calculations and workflow.


def discounted_return_bounds(rewards, gamma, terminated):  # Add the recorded rewards, weighting step t by gamma**t.
    """Add the recorded rewards, weighting step t by gamma**t.
    A terminal crash gives an exact return. A recording cutoff leaves an unknown
    continuation. Bound that continuation using the task's known reward limits.

    Original implementation notes:
    Bound return beyond a sampling cutoff under this task's reward structure.

    Nonterminal rewards are nonnegative and at most .4. A collision terminates
    with reward >= -2, so any future return is in [-2, .4/(1-gamma)].
    Termination is exact; a time-limit truncation retains these bounds.
    """
    if not 0 <= gamma < 1:  # Use this branch when not 0 is at most the future-reward discount is below 1.
        raise ValueError("gamma must be in [0,1)")  # Reject this invalid case instead of continuing with misleading experiment results.
    prefix = sum(gamma**index * reward for index, reward in enumerate(rewards))  # Rewards already observed are known; later rewards beyond the cutoff are not.
    if terminated:  # index starts at zero: the first reward has weight 1, the second gamma, the third gamma**2. This matches what the Bellman recursion predicts.
        return float(prefix), float(prefix)  # Return prefix as a floating-point value, prefix as a floating-point value.
    weight = gamma ** len(rewards)  # Any missing future return starts after all recorded steps, so discount the whole remainder.
    return float(prefix - 2 * weight), float(prefix + 0.4 / (1 - gamma) * weight)  # Lower bound: a crash might cost -2 immediately after the cutoff. Upper bound: earning .4 forever gives the geometric sum .4/(1-gamma). Discount either continuation by weight before adding the known prefix.


def quantile_compatibility(quantiles, lower_returns, upper_returns):  # Check whether each predicted quantile could agree with the observed returns.
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
    quantiles = np.asarray(quantiles, dtype=float)  # Set quantiles to quantiles as a numerical array/tensor for batch calculations.
    low = np.asarray(lower_returns, dtype=float)  # Set low to lower returns as a numerical array/tensor for batch calculations.
    high = np.asarray(upper_returns, dtype=float)  # Set high to upper returns as a numerical array/tensor for batch calculations.
    if (  # Use this branch when quantiles.ndim differs from 2 or low.ndim differs from 1 or high.ndim differs from 1 or the number of entries in low differs from the number of entries in quantiles or the number of entries in high differs from the number of entries in low.
        quantiles.ndim != 2  # Include quantiles.ndim differs from 2 or low.ndim differs from 1 or high.ndim differs from 1 or the number of entries in low differs from the number of entries in quantiles or the number of entries in high differs from the number of entries in low in the surrounding operation.
        or low.ndim != 1  # Include low.ndim differs from 1 in the surrounding operation.
        or high.ndim != 1  # Include high.ndim differs from 1 in the surrounding operation.
        or len(low) != len(quantiles)  # Include the number of entries in low differs from the number of entries in quantiles in the surrounding operation.
        or len(high) != len(low)  # Include the number of entries in high differs from the number of entries in low in the surrounding operation.
    ):  # Finish assembling the grouped values above.
        raise ValueError("Incompatible quantile and return shapes")  # Reject this invalid case instead of continuing with misleading experiment results.
    if (  # Use this branch when not the number of entries in low or np.any applied to low is above high or not the result of np.isfinite(quantiles).all or not the result of np.isfinite(low).all or not the result of np.isfinite(high).all.
        not len(low)  # Include not the number of entries in low or np.any applied to low is above high or not the result of np.isfinite(quantiles).all or not the result of np.isfinite(low).all or not the result of np.isfinite(high).all in the surrounding operation.
        or np.any(low > high)  # Include np.any applied to low is above high in the surrounding operation.
        or not np.isfinite(quantiles).all()  # Include not the result of np.isfinite(quantiles).all in the surrounding operation.
        or not np.isfinite(low).all()  # Include not the result of np.isfinite(low).all in the surrounding operation.
        or not np.isfinite(high).all()  # Include not the result of np.isfinite(high).all in the surrounding operation.
    ):  # Finish assembling the grouped values above.
        raise ValueError("Expected finite, nonempty ordered return bounds")  # Reject this invalid case instead of continuing with misleading experiment results.
    lower = (high[:, None] < quantiles).mean(axis=0)  # Strict and non-strict comparisons preserve ties in discrete or repeated returns. quantiles is [episodes, N]. [:, None] changes each episode's return bound from [episodes] to [episodes,1], so we compare it with all N heads. Averaging booleans over axis=0 gives one observed fraction per head. A high bound below a prediction is definitely below it; a low bound at/below it might be below it. Together these bracket observed coverage.
    upper = (low[:, None] <= quantiles).mean(axis=0)  # Set upper to the average of the entries of low selected by (:, None) is at most quantiles along axis 0.
    taus = (np.arange(quantiles.shape[1]) + 0.5) / quantiles.shape[1]  # Set the assigned quantile fractions to ((np.arange applied to the entries of quantiles.shape selected by 1 plus 0.5) divided by the entries of quantiles.shape selected by 1).
    gap = np.maximum(0.0, np.maximum(lower - taus, taus - upper))  # A tau-quantile is compatible when tau lies inside its coverage bracket. maximum clips negative distances to zero: overlapping brackets are not counted as errors. This is a diagnostic, not proof of calibrated risk.
    return {  # Return the named settings or measurements assembled below.
        "taus": taus.tolist(),  # Record 'taus' as the assigned quantile fractions as a list that can be saved in JSON.
        "coverage_lower": lower.tolist(),  # Record 'coverage_lower' as lower as a list that can be saved in JSON.
        "coverage_upper": upper.tolist(),  # Record 'coverage_upper' as upper as a list that can be saved in JSON.
        "minimum_mean_absolute_compatibility_gap": float(gap.mean()),  # Record 'minimum_mean_absolute_compatibility_gap' as the average of gap as a floating-point value.
    }  # Finish assembling the grouped values above.


def summarize_rows(rows):  # Summarize forecast error and driving outcomes from recorded rollouts.
    """Summarize forecast error and driving outcomes from recorded rollouts.
    Predicted mean minus realized return measures optimism when positive. Keep the
    unknown future-return range instead of replacing surviving episodes with zero.
    """
    quantiles = np.array([row["initial_quantiles"] for row in rows])  # Set quantiles to the entries of row selected by 'initial_quantiles' for each row in rows as a numerical array/tensor for batch calculations.
    lower = np.array([row["return_lower"] for row in rows])  # Set lower to the entries of row selected by 'return_lower' for each row in rows as a numerical array/tensor for batch calculations.
    upper = np.array([row["return_upper"] for row in rows])  # Set upper to the entries of row selected by 'return_upper' for each row in rows as a numerical array/tensor for batch calculations.
    means = quantiles.mean(axis=1)  # Average all predicted quantiles to obtain the network's predicted mean return.
    crashes = [row for row in rows if row["crashed"]]  # NumPy's axis=1 here plays the same role as PyTorch's dim=-1 for this [episodes,N] array: average quantile heads within each episode.
    return {  # Return the named settings or measurements assembled below.
        "episodes": len(rows),  # Record 'episodes' as the number of entries in rows.
        **quantile_compatibility(quantiles, lower, upper),  # Include quantile compatibility applied to quantiles, lower, upper in the surrounding operation.
        "mean_prediction": float(means.mean()),  # Record 'mean_prediction' as the average of means as a floating-point value.
        "mean_return_lower": float(lower.mean()),  # Record 'mean_return_lower' as the average of lower as a floating-point value.
        "mean_return_upper": float(upper.mean()),  # Record 'mean_return_upper' as the average of upper as a floating-point value.
        "mean_prediction_bias_bounds": [  # Record 'mean_prediction_bias_bounds' as the average of (means minus upper) as a floating-point value, the average of (means minus lower) as a floating-point value.
            float((means - upper).mean()),  # Include the average of (means minus upper) as a floating-point value in the surrounding operation.
            float((means - lower).mean()),  # Include the average of (means minus lower) as a floating-point value in the surrounding operation.
        ],  # Finish assembling the grouped values above.
        "censored_fraction": float(np.mean([row["censored"] for row in rows])),  # Record 'censored_fraction' as the average of the entries of row selected by 'censored' for each row in rows as a floating-point value.
        "quantile_crossing_fraction": float(np.mean(np.diff(quantiles, axis=1) < 0)),  # Record 'quantile_crossing_fraction' as the average of the differences between adjacent entries of quantiles is below 0 as a floating-point value.
        "crash_rate": float(np.mean([row["crashed"] for row in rows])),  # Record 'crash_rate' as the average of the entries of row selected by 'crashed' for each row in rows as a floating-point value.
        "mean_speed_m_s": float(np.mean([row["mean_speed_m_s"] for row in rows])),  # Record 'mean_speed_m_s' as the average of the entries of row selected by 'mean_speed_m_s' for each row in rows as a floating-point value.
        "mean_duration_s": float(np.mean([row["length"] for row in rows])),  # Record 'mean_duration_s' as the average of the entries of row selected by 'length' for each row in rows as a floating-point value.
        "crash_within_3_seconds_fraction": float(  # Record 'crash_within_3_seconds_fraction' as the average of row['crashed'] and row['length'] <= 3 for each row in rows as a floating-point value.
            np.mean([row["crashed"] and row["length"] <= 3 for row in rows])  # Include the average of the entries of row selected by 'crashed' and row['length'] is at most 3 for each row in rows in the surrounding operation.
        ),  # Finish assembling float inputs.
        "recent_lateral_request_given_crash": (  # Record 'recent_lateral_request_given_crash' as the average of row['recent lateral request'] for each row in crashes as a floating-point value when crashes, otherwise None.
            float(np.mean([row["recent_lateral_request"] for row in crashes]))  # Include the average of row['recent lateral request'] for each row in crashes as a floating-point value when crashes, otherwise None in the surrounding operation.
            if crashes  # Include crashes in the surrounding operation.
            else None  # Include None in the surrounding operation.
        ),  # Finish assembling the grouped values above.
        "recent_lane_transition_given_crash": (  # Record 'recent_lane_transition_given_crash' as the average of row['recent lane transition'] for each row in crashes as a floating-point value when crashes, otherwise None.
            float(np.mean([row["recent_lane_transition"] for row in crashes]))  # Include the average of row['recent lane transition'] for each row in crashes as a floating-point value when crashes, otherwise None in the surrounding operation.
            if crashes  # Include crashes in the surrounding operation.
            else None  # Include None in the surrounding operation.
        ),  # Finish assembling the grouped values above.
    }  # Finish assembling the grouped values above.


def terminal_neighbor(env):  # Carry out the terminal neighbor part of this module's workflow.
    """Carry out the terminal neighbor part of this module's workflow.
    See the calling function for the settings and saved files used here.
    """
    ego = env.unwrapped.vehicle  # Set ego to env.unwrapped.vehicle.
    others = [vehicle for vehicle in env.unwrapped.road.vehicles if vehicle is not ego]  # Set others to vehicle for each vehicle in env.unwrapped.road.vehicles, keeping only cases where vehicle is not ego.
    if not others:  # Use this branch when not others.
        return None  # Return None.
    neighbor = min(  # Set neighbor to min applied to others.
        others, key=lambda vehicle: np.linalg.norm(vehicle.position - ego.position)  # Include others, lambda vehicle: np.linalg.norm(vehicle.position - ego.position) in the surrounding operation.
    )  # Finish assembling min inputs.
    offset = neighbor.position - ego.position  # Set offset to (neighbor.position minus ego.position).
    return {  # Return the named settings or measurements assembled below.
        "longitudinal_offset_m": float(offset[0]),  # Record 'longitudinal_offset_m' as the entries of offset selected by 0 as a floating-point value.
        "lateral_offset_m": float(offset[1]),  # Record 'lateral_offset_m' as the entries of offset selected by 1 as a floating-point value.
        "distance_m": float(np.linalg.norm(offset)),  # Record 'distance_m' as np.linalg.norm applied to offset as a floating-point value.
        "position_proxy": (  # Record 'position_proxy' as 'front' when the entries of offset selected by 0 is above 2.5, otherwise 'rear' when the entries of offset selected by 0 is below negative 2.5, otherwise 'side'.
            "front" if offset[0] > 2.5 else "rear" if offset[0] < -2.5 else "side"  # Include 'front' when the entries of offset selected by 0 is above 2.5, otherwise 'rear' when the entries of offset selected by 0 is below negative 2.5, otherwise 'side' in the surrounding operation.
        ),  # Finish assembling the grouped values above.
    }  # Finish assembling the grouped values above.


def collect(args):  # Record initial return predictions and the resulting driving outcomes.
    """Record initial return predictions and the resulting driving outcomes.
    Use fixed evaluation seeds, keep terminal events distinct from recording
    cutoffs, and save enough detail to check the result later.
    """
    if Path(args.out).exists():  # Use this branch when the result of Path(args.out).exists.
        raise ValueError(f"Refusing to overwrite {args.out}")  # Reject this invalid case instead of continuing with misleading experiment results.
    if args.seed_start < 10000 or args.seed_start + args.episodes > 20000:  # Use this branch when args.seed start is below 10000 or (args.seed start plus args.episodes) is above 20000.
        raise ValueError("Calibration must remain in development block [10000,20000)")  # Reject this invalid case instead of continuing with misleading experiment results.
    seed_everything(0)  # Initialize repeatable random streams using 0.
    agent, checkpoint = load_agent(args.checkpoint)  # Set agent, checkpoint to the frozen learner and checkpoint loaded from args.checkpoint.
    if agent.algorithm != "qrdqn":  # Use this branch when agent.algorithm differs from 'qrdqn'.
        raise ValueError("Quantile diagnostics require QR-DQN")  # Reject this invalid case instead of continuing with misleading experiment results.
    gamma = checkpoint["config"]["gamma"]  # Set the future-reward discount to the entries of the entries of checkpoint selected by 'config' selected by 'gamma'.
    rows = []  # Set rows to .
    for density in args.densities:  # Process each the traffic-density setting from args.densities.
        env = make_env(density=density)  # Set env to a highway simulator configured with.
        config = env.unwrapped.config  # Set config to env.unwrapped.config.
        if (  # Use this branch when the entries of config selected by 'collision_reward' differs from negative 2 or the entries of config selected by 'high_speed_reward' differs from 0.4 or the entries of config selected by 'normalize_reward' or the entries of config selected by 'right_lane_reward' differs from 0.
            config["collision_reward"] != -2  # Include the entries of config selected by 'collision_reward' differs from negative 2 or the entries of config selected by 'high_speed_reward' differs from 0.4 or the entries of config selected by 'normalize_reward' or the entries of config selected by 'right_lane_reward' differs from 0 in the surrounding operation.
            or config["high_speed_reward"] != 0.4  # Include the entries of config selected by 'high_speed_reward' differs from 0.4 in the surrounding operation.
            or config["normalize_reward"]  # Include the entries of config selected by 'normalize_reward' in the surrounding operation.
            or config["right_lane_reward"] != 0  # Include the entries of config selected by 'right_lane_reward' differs from 0 in the surrounding operation.
        ):  # Finish assembling the grouped values above.
            raise ValueError("Reward assumptions for continuation bounds do not hold")  # Reject this invalid case instead of continuing with misleading experiment results.
        for index in range(args.episodes):  # Process each index from range applied to args.episodes.
            observation, _ = env.reset(seed=args.seed_start + index)  # Set the currently observed traffic features, the unused result to the result of env.reset.
            rewards, states, speeds = [], [], []  # Set the observed rewards, states, speeds to , , .
            initial_quantiles = None  # Set initial quantiles to None.
            while True:  # Keep processing while True.
                with torch.no_grad():  # Treat predictions in this block as fixed learning targets or inference results; do not backpropagate through them.
                    distribution = agent.online(  # Set distribution to agent.online applied to torch.as tensor(flatten(observation)).unsqueeze applied to 0.
                        torch.as_tensor(flatten(observation)).unsqueeze(0)  # Include torch.as tensor(flatten(observation)).unsqueeze applied to 0 in the surrounding operation.
                    )  # Finish assembling agent.online inputs.
                    action = int(  # Set action to the index of the highest score in the mean or lower-tail action scores computed from distribution, args.alpha extracted as a plain scalar as a whole-number value.
                        action_scores(distribution, args.alpha).argmax(dim=1).item()  # Include the index of the highest score in the mean or lower-tail action scores computed from distribution, args.alpha extracted as a plain scalar in the surrounding operation.
                    )  # Finish assembling int inputs.
                if args.speed_cap is not None:  # Use this branch when args.speed cap is not None.
                    action = govern_action(env, observation, action, args.speed_cap)  # Set action to govern action applied to env, the currently observed traffic features, action, args.speed cap.
                if initial_quantiles is None:  # Use this branch when initial quantiles is None.
                    initial_quantiles = distribution[0, action].tolist()  # Save the prediction before observing the outcome; using a later prediction would leak information.
                vehicle = env.unwrapped.vehicle  # Set vehicle to env.unwrapped.vehicle.
                states.append(  # Add the named settings or measurements assembled below to states.
                    {  # Include the named settings or measurements assembled below in the surrounding operation.
                        "step": len(rewards),  # Record 'step' as the number of entries in the observed rewards.
                        "action": action,  # Record 'action' as action.
                        "lane": int(vehicle.lane_index[2]),  # Record 'lane' as the entries of vehicle.lane index selected by 2 as a whole-number value.
                        "target_lane": int(vehicle.target_lane_index[2]),  # Record 'target_lane' as the entries of vehicle.target lane index selected by 2 as a whole-number value.
                        "observed_vehicle_slots": int(  # Record 'observed_vehicle_slots' as the result of np.asarray(observation)[:, 0].sum as a whole-number value.
                            np.asarray(observation)[:, 0].sum()  # Include the result of np.asarray(observation)[:, 0].sum in the surrounding operation.
                        ),  # Finish assembling int inputs.
                        **front_state(env),  # Include front state applied to env in the surrounding operation.
                    }  # Finish assembling the grouped values above.
                )  # Finish assembling states.append inputs.
                observation, reward, terminated, truncated, _ = env.step(action)  # Set the currently observed traffic features, reward, the true-terminal flags, the recording-cutoff flags, the unused result to env.step applied to action.
                if not -2.000001 <= reward <= 0.400001 or (  # Use this branch when not negative 2.000001 is at most reward is at most 0.400001 or reward is below 0 and not the true-terminal flags.
                    reward < 0 and not terminated  # Include reward is below 0 and not the true-terminal flags in the surrounding operation.
                ):  # Finish assembling the grouped values above.
                    raise ValueError(  # Reject this invalid case instead of continuing with misleading experiment results.
                        "Observed reward violates continuation-bound assumptions"  # Include 'Observed reward violates continuation-bound assumptions' in the surrounding operation.
                    )  # Finish assembling ValueError inputs.
                rewards.append(float(reward))  # Add reward as a floating-point value to the observed rewards.
                speeds.append(float(env.unwrapped.vehicle.speed))  # Add env.unwrapped.vehicle.speed as a floating-point value to speeds.
                if terminated or truncated:  # Use this branch when the true-terminal flags or the recording-cutoff flags.
                    lower, upper = discounted_return_bounds(rewards, gamma, terminated)  # A terminal return is exact. A surviving recording contributes a range, not a fake terminal reward.
                    recent = states[-3:]  # Set recent to the entries of states selected by -3:.
                    lanes = [state["lane"] for state in recent] + [  # Set lanes to (the entries of state selected by 'lane' for each state in recent plus the entries of env.unwrapped.vehicle.lane index selected by 2 as a whole-number value).
                        int(env.unwrapped.vehicle.lane_index[2])  # Include the entries of env.unwrapped.vehicle.lane index selected by 2 as a whole-number value in the surrounding operation.
                    ]  # Finish assembling the grouped values above.
                    rows.append(  # Add the named settings or measurements assembled below to rows.
                        {  # Include the named settings or measurements assembled below in the surrounding operation.
                            "density": density,  # Record 'density' as the traffic-density setting.
                            "seed": args.seed_start + index,  # Record 'seed' as (args.seed start plus index).
                            "initial_quantiles": initial_quantiles,  # Record 'initial_quantiles' as initial quantiles.
                            "return_lower": lower,  # Record 'return_lower' as lower.
                            "return_upper": upper,  # Record 'return_upper' as upper.
                            "censored": bool(truncated and not terminated),  # Record 'censored' as the recording-cutoff flags and not the true-terminal flags as a true/false value.
                            "crashed": bool(env.unwrapped.vehicle.crashed),  # Record 'crashed' as env.unwrapped.vehicle.crashed as a true/false value.
                            "length": len(rewards),  # Record 'length' as the number of entries in the observed rewards.
                            "rewards": rewards,  # Record 'rewards' as the observed rewards.
                            "mean_speed_m_s": float(np.mean(speeds)),  # Record 'mean_speed_m_s' as the average of speeds as a floating-point value.
                            "recent_states": recent,  # Record 'recent_states' as recent.
                            "recent_lateral_request": any(  # Record 'recent_lateral_request' as any applied to the entries of state selected by 'action' belongs to 0, 2 for each state in recent.
                                state["action"] in [0, 2] for state in recent  # Include the entries of state selected by 'action' belongs to 0, 2, state, recent in the surrounding operation.
                            ),  # Finish assembling any inputs.
                            "recent_lane_transition": len(set(lanes)) > 1,  # Record 'recent_lane_transition' as the number of entries in set applied to lanes is above 1.
                            "terminal_nearest_neighbor": (  # Record 'terminal_nearest_neighbor' as terminal neighbor applied to env when the true-terminal flags, otherwise None.
                                terminal_neighbor(env) if terminated else None  # Include terminal neighbor applied to env when the true-terminal flags, otherwise None in the surrounding operation.
                            ),  # Finish assembling the grouped values above.
                        }  # Finish assembling the grouped values above.
                    )  # Finish assembling rows.append inputs.
                    break  # Stop this loop now that its stopping condition has been reached.
        env.close()  # The result of env.close.
    provenance = {  # Set provenance to the named settings or measurements assembled below.
        "checkpoint": str(args.checkpoint),  # Record 'checkpoint' as args.checkpoint as a text value.
        "checkpoint_sha256": hashlib.sha256(  # Record 'checkpoint_sha256' as the result of hashlib.sha256(Path(args.checkpoint).read bytes()).hexdigest.
            Path(args.checkpoint).read_bytes()  # Include the result of Path(args.checkpoint).read bytes in the surrounding operation.
        ).hexdigest(),  # Finish assembling hashlib.sha256(Path(args.checkpoint).read bytes()).hexdigest inputs.
        "training_seed": checkpoint["config"]["seed"],  # Record 'training_seed' as the entries of the entries of checkpoint selected by 'config' selected by 'seed'.
        "training_steps": checkpoint["step"],  # Record 'training_steps' as the entries of checkpoint selected by 'step'.
        "alpha": args.alpha,  # Record 'alpha' as args.alpha.
        "speed_cap": args.speed_cap,  # Record 'speed_cap' as args.speed cap.
        "gamma": gamma,  # Record 'gamma' as the future-reward discount.
        "seed_start": args.seed_start,  # Record 'seed_start' as args.seed start.
        "episodes_per_density": args.episodes,  # Record 'episodes_per_density' as args.episodes.
        "source_hashes": source_hashes(),  # Record 'source_hashes' as the result of source hashes.
        "versions": versions(),  # Record 'versions' as the result of versions.
    }  # Finish assembling the grouped values above.
    summary = [  # Set summary to the named settings or measurements assembled below for each the traffic-density setting in args.densities.
        {  # Include the named settings or measurements assembled below in the surrounding operation.
            "density": density,  # Record 'density' as the traffic-density setting.
            **summarize_rows([row for row in rows if row["density"] == density]),  # Include summarize rows applied to row for each row in rows, keeping only cases where the entries of row selected by 'density' equals the traffic-density setting in the surrounding operation.
        }  # Finish assembling the grouped values above.
        for density in args.densities  # Include the traffic-density setting, args.densities in the surrounding operation.
    ]  # Finish assembling the grouped values above.
    save_json(  # Save the following results/settings as json: args.out, the named settings or measurements assembled below.
        args.out,  # Include args.out in the surrounding operation.
        {  # Include the named settings or measurements assembled below in the surrounding operation.
            "provenance": provenance,  # Record 'provenance' as provenance.
            "summary": summary,  # Record 'summary' as summary.
            "episodes": rows,  # Record 'episodes' as rows.
            "interpretation": "Marginal quantile compatibility brackets include censoring and ties, not sampling uncertainty. Changed deployment policies differ from training continuation. Terminal geometry and recent actions are descriptive proxies.",  # Record 'interpretation' as the report text.
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
    parser.add_argument("--speed-cap", type=float)  # Parser.add argument applied to '--speed-cap'.
    parser.add_argument("--episodes", type=int, default=60)  # Parser.add argument applied to '--episodes'.
    parser.add_argument("--seed-start", type=int, default=12000)  # Parser.add argument applied to '--seed-start'.
    parser.add_argument("--densities", nargs="+", type=float, default=[1.0, 1.5, 2.0])  # Parser.add argument applied to '--densities'.
    parser.add_argument("--out", required=True)  # Parser.add argument applied to '--out'.
    args = parser.parse_args()  # Set args to the result of parser.parse args.
    if args.episodes < 1 or not 0 < args.alpha <= 1:  # Use this branch when args.episodes is below 1 or not 0 is below args.alpha is at most 1.
        parser.error("Require episodes > 0 and alpha in (0,1]")  # Parser.error applied to 'require episodes > 0 and alpha in (0,1]'.
    if args.speed_cap is not None and (  # Use this branch when args.speed cap is not None and not a true/false check for finite numerical values in args.speed cap or args.speed cap is at most 0.
        not np.isfinite(args.speed_cap) or args.speed_cap <= 0  # Include not a true/false check for finite numerical values in args.speed cap or args.speed cap is at most 0 in the surrounding operation.
    ):  # Finish assembling the grouped values above.
        parser.error("Require a finite positive speed cap")  # Parser.error applied to 'require a finite positive speed cap'.
    collect(args)  # Collect applied to args.


if __name__ == "__main__":  # Run this workflow when launched from Terminal; importing its helpers must not launch experiments.
    main()  # The result of main.
