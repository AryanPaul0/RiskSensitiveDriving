"""Set up the highway and translate between simulator data and the agent.

The neural network sees eight rows of five values: presence, x, y, vx and vy.
That is 40 inputs, not a complete description of all traffic. Observations are
partial: some vehicles and driver settings are missing. The five actions are
lane left, keep target speed, lane right, increase target speed and decrease it.
The simulator turns those requests into steering and acceleration.

Original implementation notes:
Environment and randomness shared by training and evaluation.
"""

import os
import random

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import gymnasium as gym
import highway_env  # Registers the environments with Gymnasium.
import numpy as np
import torch


def seed_everything(seed):
    """Give Python, NumPy and PyTorch repeatable random starting points.
    Using one CPU thread per process also avoids many threads competing for these
    small networks when several training jobs run at once.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # These small networks usually run faster with one CPU thread.
    torch.set_num_threads(1)


def make_env(density=1.0, render=False, collision_penalty=-2.0):
    """Create a highway with explicitly shared settings for fair comparisons.
    Density changes how closely vehicles start together. It does not change the
    five available actions or the 40 observation inputs.
    """
    return gym.make(
        # HighwayEnv supplies vehicle dynamics and traffic drivers. Our agent
        # chooses high-level maneuvers, not raw steering or throttle values.
        "highway-fast-v0",
        render_mode="rgb_array" if render else None,
        config={
            "lanes_count": 3,
            "vehicles_count": 15,
            "vehicles_density": float(density),
            # This is a rollout cutoff, not a claim that driving ends after 30 s.
            "duration": 30,
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 8,
                # Eight slots x five features = 40 network inputs. Presence
                # distinguishes a real nearby vehicle from an empty slot.
                "features": ["presence", "x", "y", "vx", "vy"],
                "normalize": True,
                "absolute": False,
                # Relative observations still omit distant cars and driver
                # internals: this is not a fully observable Markov state.
                "see_behind": True,
            },
            # Explicit, shared reward: speed bonus and substantial crash penalty.
            "collision_reward": float(collision_penalty),
            "high_speed_reward": 0.4,
            "right_lane_reward": 0.0,
            "lane_change_reward": 0.0,
            "normalize_reward": False,
            # Keep the configured reward scale: rescaling rewards would also
            # change what return predictions and loss magnitudes mean.
            "screen_width": 900,
            "screen_height": 240,
            "offscreen_rendering": True,
        },
    )


def flatten(observation):
    """Turn the 8-by-5 observation table into a row of 40 numbers.
    Keep float32 numbers because that is the number format used by the network.
    """
    return np.asarray(observation, dtype=np.float32).reshape(-1)


def observed_speed(env, observation):
    """Recover the ego car's speed from the observation it is allowed to see.
    Undo normalization using the simulator's feature ranges, then combine horizontal
    and vertical velocity with the Pythagorean theorem. No hidden traffic data is used.

    Original implementation notes:
    Decode ego velocity using observation metadata, without simulator state.
    """
    observer = env.unwrapped.observation_type
    ego = np.asarray(observation)[0]
    # Slot 0 is the controlled car. Decode only its two velocity features;
    # the governor must not gain access to hidden traffic through env.vehicle.
    velocity = []
    for feature in ["vx", "vy"]:
        value = float(ego[observer.features.index(feature)])
        if observer.normalize:
            low, high = observer.features_range[feature]
            value = low + (value + 1) * 0.5 * (high - low)
            # Invert normalization: -1 maps to low, +1 maps to high. The
            # resulting velocity component is in meters per second again.
        velocity.append(value)
    # sqrt(vx**2 + vy**2): combine two velocity components into speed.
    return float(np.linalg.norm(velocity))


def govern_action(env, observation, action, speed_cap):
    """Apply a simple speed rule after the network proposes an action.
    Above the threshold, request a lower target speed. Near it, block further
    acceleration. This is an action override, not a physical guarantee about speed.

    Original implementation notes:
    Simple ego-speed control; no traffic state or collision-avoidance claim.

    Above the cap, request a slower target speed. Near the cap, suppress an
    acceleration request. Other learned actions remain available below the cap.
    """
    actions = env.unwrapped.action_type.actions_indexes
    speed = observed_speed(env, observation)
    if speed > speed_cap:
        # Request a lower target speed; the simulator may take time to slow
        # down, so this governor does not enforce an instantaneous hard limit.
        return actions["SLOWER"]
    if action == actions["FASTER"] and speed > speed_cap - 2.5:
        # Near the cap, reject acceleration without otherwise changing the
        # network's maneuver. This is a baseline control, not learned caution.
        return actions["IDLE"]
    return action


def render_frame(env):
    """Draw one simulator image and copy its pixels for a replay.
    The simulator normally turns drawing off without a physical display, so we
    explicitly enable it. Copying prevents later frames from overwriting earlier ones.

    Original implementation notes:
    Draw on an offscreen surface even with SDL's dummy video driver.

    HighwayEnv disables its viewer under the dummy driver by default, returning
    black frames. Offscreen surfaces need no window, so enable their drawing.
    """
    if env.unwrapped.viewer is None:
        env.render()
    env.unwrapped.viewer.enabled = True
    return env.render().copy()


def heuristic_action(env):
    """Drive using a hand-written following-distance rule rather than a neural network.
    This comparator reads extra simulator state, so it must be identified as having
    more information than the learned policies.

    Original implementation notes:
    Privileged full-state comparator: stay in lane and keep a generous gap.

    This rule sees simulator state, unlike the learned policies. Report it as a
    sanity check, not as an observation-matched algorithm comparison.
    """
    vehicle = env.unwrapped.vehicle
    front, _ = env.unwrapped.road.neighbour_vehicles(vehicle)
    actions = env.unwrapped.action_type.actions_indexes
    if front is not None:
        gap = vehicle.lane_distance_to(front)
        if gap < max(15.0, 1.8 * vehicle.speed):
            return actions["SLOWER"]
    if vehicle.speed < 25:
        return actions["FASTER"]
    return actions["IDLE"]


def reset_training_episode(env, density_rng, densities=(1.0,), seed=None):
    """Choose a traffic density before starting the next driving episode.
    Use a separate random generator so choosing density does not consume the random
    numbers used for exploration or replay sampling. One density reproduces the old
    fixed-density behavior. Equal episode probabilities do not mean equal step counts.

    Original implementation notes:
    Sample density at reset, with randomness separate from exploration/replay.
    """
    if not densities or any(not np.isfinite(d) or d <= 0 for d in densities):
        raise ValueError("Training densities must be finite and positive")
    density = float(
        # A single density avoids making an unnecessary random draw; this
        # preserves the random sequence of the original fixed-density setup.
        densities[0] if len(densities) == 1 else density_rng.choice(densities)
    )
    if env.unwrapped.config["vehicles_density"] != density:
        env.unwrapped.configure({"vehicles_density": density})
    observation, info = env.reset(seed=seed)
    return observation, info, density
