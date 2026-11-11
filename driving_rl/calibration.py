"""Check whether predicted future rewards agree with what actually happens.

A crash ends the driving task. A 30-second cutoff merely ends our recording.
For a surviving car we therefore know only a range for the complete future
return, not its exact value. We keep those surviving cars in the analysis.
The intervals here bound missing future rewards; they are not confidence
intervals about how much data we have collected.

Original implementation notes:
Initial-state quantile diagnostics with bounded, censored continuation.
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from .agents import action_scores
from .diagnostics import front_state
from .envs import flatten, govern_action, make_env, seed_everything
from .experiment import load_agent, save_json, source_hashes, versions


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
    if not 0 <= gamma < 1:
        raise ValueError("gamma must be in [0,1)")
    # Rewards already observed are known; later rewards beyond the cutoff are not.
    prefix = sum(gamma**index * reward for index, reward in enumerate(rewards))
    # index starts at zero: the first reward has weight 1, the second gamma,
    # the third gamma**2. This matches what the Bellman recursion predicts.
    if terminated:
        return float(prefix), float(prefix)
    # Any missing future return starts after all recorded steps, so discount the whole remainder.
    weight = gamma ** len(rewards)
    # Lower bound: a crash might cost -2 immediately after the cutoff. Upper
    # bound: earning .4 forever gives the geometric sum .4/(1-gamma).
    # Discount either continuation by weight before adding the known prefix.
    return float(prefix - 2 * weight), float(prefix + 0.4 / (1 - gamma) * weight)


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
    quantiles = np.asarray(quantiles, dtype=float)
    low = np.asarray(lower_returns, dtype=float)
    high = np.asarray(upper_returns, dtype=float)
    if (
        quantiles.ndim != 2
        or low.ndim != 1
        or high.ndim != 1
        or len(low) != len(quantiles)
        or len(high) != len(low)
    ):
        raise ValueError("Incompatible quantile and return shapes")
    if (
        not len(low)
        or np.any(low > high)
        or not np.isfinite(quantiles).all()
        or not np.isfinite(low).all()
        or not np.isfinite(high).all()
    ):
        raise ValueError("Expected finite, nonempty ordered return bounds")
    # Strict and non-strict comparisons preserve ties in discrete or repeated returns.
    # quantiles is [episodes, N]. [:, None] changes each episode's return
    # bound from [episodes] to [episodes,1], so we compare it with all N heads.
    # Averaging booleans over axis=0 gives one observed fraction per head.
    # A high bound below a prediction is definitely below it; a low bound
    # at/below it might be below it. Together these bracket observed coverage.
    lower = (high[:, None] < quantiles).mean(axis=0)
    upper = (low[:, None] <= quantiles).mean(axis=0)
    taus = (np.arange(quantiles.shape[1]) + 0.5) / quantiles.shape[1]
    # A tau-quantile is compatible when tau lies inside its coverage bracket.
    # maximum clips negative distances to zero: overlapping brackets are not
    # counted as errors. This is a diagnostic, not proof of calibrated risk.
    gap = np.maximum(0.0, np.maximum(lower - taus, taus - upper))
    return {
        "taus": taus.tolist(),
        "coverage_lower": lower.tolist(),
        "coverage_upper": upper.tolist(),
        "minimum_mean_absolute_compatibility_gap": float(gap.mean()),
    }


def summarize_rows(rows):
    """Summarize forecast error and driving outcomes from recorded rollouts.
    Predicted mean minus realized return measures optimism when positive. Keep the
    unknown future-return range instead of replacing surviving episodes with zero.
    """
    quantiles = np.array([row["initial_quantiles"] for row in rows])
    lower = np.array([row["return_lower"] for row in rows])
    upper = np.array([row["return_upper"] for row in rows])
    # Average all predicted quantiles to obtain the network's predicted mean return.
    means = quantiles.mean(axis=1)
    # NumPy's axis=1 here plays the same role as PyTorch's dim=-1 for this
    # [episodes,N] array: average quantile heads within each episode.
    crashes = [row for row in rows if row["crashed"]]
    return {
        "episodes": len(rows),
        **quantile_compatibility(quantiles, lower, upper),
        "mean_prediction": float(means.mean()),
        "mean_return_lower": float(lower.mean()),
        "mean_return_upper": float(upper.mean()),
        "mean_prediction_bias_bounds": [
            float((means - upper).mean()),
            float((means - lower).mean()),
        ],
        "censored_fraction": float(np.mean([row["censored"] for row in rows])),
        "quantile_crossing_fraction": float(np.mean(np.diff(quantiles, axis=1) < 0)),
        "crash_rate": float(np.mean([row["crashed"] for row in rows])),
        "mean_speed_m_s": float(np.mean([row["mean_speed_m_s"] for row in rows])),
        "mean_duration_s": float(np.mean([row["length"] for row in rows])),
        "crash_within_3_seconds_fraction": float(
            np.mean([row["crashed"] and row["length"] <= 3 for row in rows])
        ),
        "recent_lateral_request_given_crash": (
            float(np.mean([row["recent_lateral_request"] for row in crashes]))
            if crashes
            else None
        ),
        "recent_lane_transition_given_crash": (
            float(np.mean([row["recent_lane_transition"] for row in crashes]))
            if crashes
            else None
        ),
    }


def terminal_neighbor(env):
    """Carry out the terminal neighbor part of this module's workflow.
    See the calling function for the settings and saved files used here.
    """
    ego = env.unwrapped.vehicle
    others = [vehicle for vehicle in env.unwrapped.road.vehicles if vehicle is not ego]
    if not others:
        return None
    neighbor = min(
        others, key=lambda vehicle: np.linalg.norm(vehicle.position - ego.position)
    )
    offset = neighbor.position - ego.position
    return {
        "longitudinal_offset_m": float(offset[0]),
        "lateral_offset_m": float(offset[1]),
        "distance_m": float(np.linalg.norm(offset)),
        "position_proxy": (
            "front" if offset[0] > 2.5 else "rear" if offset[0] < -2.5 else "side"
        ),
    }


def collect(args):
    """Record initial return predictions and the resulting driving outcomes.
    Use fixed evaluation seeds, keep terminal events distinct from recording
    cutoffs, and save enough detail to check the result later.
    """
    if Path(args.out).exists():
        raise ValueError(f"Refusing to overwrite {args.out}")
    if args.seed_start < 10000 or args.seed_start + args.episodes > 20000:
        raise ValueError("Calibration must remain in development block [10000,20000)")
    seed_everything(0)
    agent, checkpoint = load_agent(args.checkpoint)
    if agent.algorithm != "qrdqn":
        raise ValueError("Quantile diagnostics require QR-DQN")
    gamma = checkpoint["config"]["gamma"]
    rows = []
    for density in args.densities:
        env = make_env(density=density)
        config = env.unwrapped.config
        if (
            config["collision_reward"] != -2
            or config["high_speed_reward"] != 0.4
            or config["normalize_reward"]
            or config["right_lane_reward"] != 0
        ):
            raise ValueError("Reward assumptions for continuation bounds do not hold")
        for index in range(args.episodes):
            observation, _ = env.reset(seed=args.seed_start + index)
            rewards, states, speeds = [], [], []
            initial_quantiles = None
            while True:
                with torch.no_grad():
                    distribution = agent.online(
                        torch.as_tensor(flatten(observation)).unsqueeze(0)
                    )
                    action = int(
                        action_scores(distribution, args.alpha).argmax(dim=1).item()
                    )
                if args.speed_cap is not None:
                    action = govern_action(env, observation, action, args.speed_cap)
                if initial_quantiles is None:
                    # Save the prediction before observing the outcome; using a later prediction would leak information.
                    initial_quantiles = distribution[0, action].tolist()
                vehicle = env.unwrapped.vehicle
                states.append(
                    {
                        "step": len(rewards),
                        "action": action,
                        "lane": int(vehicle.lane_index[2]),
                        "target_lane": int(vehicle.target_lane_index[2]),
                        "observed_vehicle_slots": int(
                            np.asarray(observation)[:, 0].sum()
                        ),
                        **front_state(env),
                    }
                )
                observation, reward, terminated, truncated, _ = env.step(action)
                if not -2.000001 <= reward <= 0.400001 or (
                    reward < 0 and not terminated
                ):
                    raise ValueError(
                        "Observed reward violates continuation-bound assumptions"
                    )
                rewards.append(float(reward))
                speeds.append(float(env.unwrapped.vehicle.speed))
                if terminated or truncated:
                    # A terminal return is exact. A surviving recording contributes a range, not a fake terminal reward.
                    lower, upper = discounted_return_bounds(rewards, gamma, terminated)
                    recent = states[-3:]
                    lanes = [state["lane"] for state in recent] + [
                        int(env.unwrapped.vehicle.lane_index[2])
                    ]
                    rows.append(
                        {
                            "density": density,
                            "seed": args.seed_start + index,
                            "initial_quantiles": initial_quantiles,
                            "return_lower": lower,
                            "return_upper": upper,
                            "censored": bool(truncated and not terminated),
                            "crashed": bool(env.unwrapped.vehicle.crashed),
                            "length": len(rewards),
                            "rewards": rewards,
                            "mean_speed_m_s": float(np.mean(speeds)),
                            "recent_states": recent,
                            "recent_lateral_request": any(
                                state["action"] in [0, 2] for state in recent
                            ),
                            "recent_lane_transition": len(set(lanes)) > 1,
                            "terminal_nearest_neighbor": (
                                terminal_neighbor(env) if terminated else None
                            ),
                        }
                    )
                    break
        env.close()
    provenance = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(
            Path(args.checkpoint).read_bytes()
        ).hexdigest(),
        "training_seed": checkpoint["config"]["seed"],
        "training_steps": checkpoint["step"],
        "alpha": args.alpha,
        "speed_cap": args.speed_cap,
        "gamma": gamma,
        "seed_start": args.seed_start,
        "episodes_per_density": args.episodes,
        "source_hashes": source_hashes(),
        "versions": versions(),
    }
    summary = [
        {
            "density": density,
            **summarize_rows([row for row in rows if row["density"] == density]),
        }
        for density in args.densities
    ]
    save_json(
        args.out,
        {
            "provenance": provenance,
            "summary": summary,
            "episodes": rows,
            "interpretation": "Marginal quantile compatibility brackets include censoring and ties, not sampling uncertainty. Changed deployment policies differ from training continuation. Terminal geometry and recent actions are descriptive proxies.",
        },
    )
    print(args.out, flush=True)


def main():
    """Read Terminal options, check settings, and run the requested workflow.
    Existing result files are checked before reuse so a rerun cannot silently mix
    different experiments. The main guard below prevents running jobs on import.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--speed-cap", type=float)
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--seed-start", type=int, default=12000)
    parser.add_argument("--densities", nargs="+", type=float, default=[1.0, 1.5, 2.0])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.episodes < 1 or not 0 < args.alpha <= 1:
        parser.error("Require episodes > 0 and alpha in (0,1]")
    if args.speed_cap is not None and (
        not np.isfinite(args.speed_cap) or args.speed_cap <= 0
    ):
        parser.error("Require a finite positive speed cap")
    collect(args)


# Run the entry point only when this file is executed, not when it is imported.
if __name__ == "__main__":
    main()
