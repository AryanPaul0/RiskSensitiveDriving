"""Environment and randomness shared by training and evaluation."""
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # These small networks usually run faster with one CPU thread.
    torch.set_num_threads(1)


def make_env(density=1.0, render=False, collision_penalty=-2.0):
    return gym.make(
        "highway-fast-v0",
        render_mode="rgb_array" if render else None,
        config={
            "lanes_count": 3,
            "vehicles_count": 15,
            "vehicles_density": float(density),
            "duration": 30,
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 8,
                "features": ["presence", "x", "y", "vx", "vy"],
                "normalize": True,
                "absolute": False,
                "see_behind": True,
            },
            # Explicit, shared reward: speed bonus and substantial crash penalty.
            "collision_reward": float(collision_penalty),
            "high_speed_reward": 0.4,
            "right_lane_reward": 0.0,
            "lane_change_reward": 0.0,
            "normalize_reward": False,
            "screen_width": 900,
            "screen_height": 240,
            "offscreen_rendering": True,
        },
    )


def flatten(observation):
    return np.asarray(observation, dtype=np.float32).reshape(-1)


def observed_speed(env, observation):
    """Decode ego velocity using observation metadata, without simulator state."""
    observer = env.unwrapped.observation_type
    ego = np.asarray(observation)[0]
    velocity = []
    for feature in ["vx", "vy"]:
        value = float(ego[observer.features.index(feature)])
        if observer.normalize:
            low, high = observer.features_range[feature]
            value = low + (value + 1) * .5 * (high - low)
        velocity.append(value)
    return float(np.linalg.norm(velocity))


def govern_action(env, observation, action, speed_cap):
    """Simple ego-speed control; no traffic state or collision-avoidance claim.

    Above the cap, request a slower target speed. Near the cap, suppress an
    acceleration request. Other learned actions remain available below the cap.
    """
    actions = env.unwrapped.action_type.actions_indexes
    speed = observed_speed(env, observation)
    if speed > speed_cap:
        return actions["SLOWER"]
    if action == actions["FASTER"] and speed > speed_cap - 2.5:
        return actions["IDLE"]
    return action


def render_frame(env):
    """Draw on an offscreen surface even with SDL's dummy video driver.

    HighwayEnv disables its viewer under the dummy driver by default, returning
    black frames. Offscreen surfaces need no window, so enable their drawing.
    """
    if env.unwrapped.viewer is None:
        env.render()
    env.unwrapped.viewer.enabled = True
    return env.render().copy()


def heuristic_action(env):
    """Privileged full-state comparator: stay in lane and keep a generous gap.

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
