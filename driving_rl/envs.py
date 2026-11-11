"""Set up the highway and translate between simulator data and the agent.

The neural network sees eight rows of five values: presence, x, y, vx and vy.
That is 40 inputs, not a complete description of all traffic. Observations are
partial: some vehicles and driver settings are missing. The five actions are
lane left, keep target speed, lane right, increase target speed and decrease it.
The simulator turns those requests into steering and acceleration.

Original implementation notes:
Environment and randomness shared by training and evaluation.
"""

# import makes these modules available; "as" gives a module a shorter local name.
import os
# import makes these modules available; "as" gives a module a shorter local name.
import random

# Call os.environ.setdefault using parentheses to supply its inputs; this line does not store a separate return value.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
# Call os.environ.setdefault using parentheses to supply its inputs; this line does not store a separate return value.
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
# Call os.environ.setdefault using parentheses to supply its inputs; this line does not store a separate return value.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

# import makes these modules available; "as" gives a module a shorter local name.
import gymnasium as gym
# import makes these modules available; "as" gives a module a shorter local name.
import highway_env  # Registers the environments with Gymnasium.
# import makes these modules available; "as" gives a module a shorter local name.
import numpy as np
# import makes these modules available; "as" gives a module a shorter local name.
import torch


# def defines seed_everything; parentheses list its inputs, and the colon starts its indented body.
def seed_everything(seed):
    """Give Python, NumPy and PyTorch repeatable random starting points.
    Using one CPU thread per process also avoids many threads competing for these
    small networks when several training jobs run at once.
    """
    # Call random.seed using parentheses to supply its inputs; this line does not store a separate return value.
    random.seed(seed)
    # Call np.random.seed using parentheses to supply its inputs; this line does not store a separate return value.
    np.random.seed(seed)
    # Call torch.manual_seed using parentheses to supply its inputs; this line does not store a separate return value.
    torch.manual_seed(seed)
    # These small networks usually run faster with one CPU thread.
    # Call torch.set_num_threads using parentheses to supply its inputs; this line does not store a separate return value.
    torch.set_num_threads(1)


# def defines make_env; parentheses list its inputs, and the colon starts its indented body.
def make_env(density=1.0, render=False, collision_penalty=-2.0):
    """Create a highway with explicitly shared settings for fair comparisons.
    Density changes how closely vehicles start together. It does not change the
    five available actions or the 40 observation inputs.
    """
    # return ends this function and sends its result back to the code that called it.
    return gym.make(
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "highway-fast-v0",
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        render_mode="rgb_array" if render else None,
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        config={
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "lanes_count": 3,
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "vehicles_count": 15,
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "vehicles_density": float(density),
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "duration": 30,
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "observation": {
                # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                "type": "Kinematics",
                # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                "vehicles_count": 8,
                # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                "features": ["presence", "x", "y", "vx", "vy"],
                # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                "normalize": True,
                # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                "absolute": False,
                # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                "see_behind": True,
            # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
            },
            # Explicit, shared reward: speed bonus and substantial crash penalty.
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "collision_reward": float(collision_penalty),
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "high_speed_reward": 0.4,
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "right_lane_reward": 0.0,
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "lane_change_reward": 0.0,
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "normalize_reward": False,
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "screen_width": 900,
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "screen_height": 240,
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "offscreen_rendering": True,
        # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
        },
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )


# def defines flatten; parentheses list its inputs, and the colon starts its indented body.
def flatten(observation):
    """Turn the 8-by-5 observation table into a row of 40 numbers.
    Keep float32 numbers because that is the number format used by the network.
    """
    # asarray makes a numerical array; dtype selects 32-bit numbers, and reshape(-1) infers a single flat dimension.
    return np.asarray(observation, dtype=np.float32).reshape(-1)


# def defines observed_speed; parentheses list its inputs, and the colon starts its indented body.
def observed_speed(env, observation):
    """Recover the ego car's speed from the observation it is allowed to see.
    Undo normalization using the simulator's feature ranges, then combine horizontal
    and vertical velocity with the Pythagorean theorem. No hidden traffic data is used.

    Original implementation notes:
    Decode ego velocity using observation metadata, without simulator state.
    """
    # = stores a result in observer; use a dot to access observation_type on env.unwrapped.
    observer = env.unwrapped.observation_type
    # = stores a result in ego; use square brackets to select part of np.asarray(observation).
    ego = np.asarray(observation)[0]
    # = stores a result in velocity; build a list, keeping its items in order.
    velocity = []
    # for takes each item from ['vx', 'vy'] in turn; feature holds the current item.
    for feature in ["vx", "vy"]:
        # = stores a result in value; call float using parentheses to supply its inputs.
        value = float(ego[observer.features.index(feature)])
        # if checks observer.normalize; only a true condition runs the indented block.
        if observer.normalize:
            # = unpacks the right-hand result into (low, high), matching values to names in order.
            low, high = observer.features_range[feature]
            # = stores a result in value; combine the operands with the displayed operator; parentheses control grouping.
            value = low + (value + 1) * 0.5 * (high - low)
        # Call velocity.append using parentheses to supply its inputs; this line does not store a separate return value.
        velocity.append(value)
    # return ends this function and sends its result back to the code that called it.
    return float(np.linalg.norm(velocity))


# def defines govern_action; parentheses list its inputs, and the colon starts its indented body.
def govern_action(env, observation, action, speed_cap):
    """Apply a simple speed rule after the network proposes an action.
    Above the threshold, request a lower target speed. Near it, block further
    acceleration. This is an action override, not a physical guarantee about speed.

    Original implementation notes:
    Simple ego-speed control; no traffic state or collision-avoidance claim.

    Above the cap, request a slower target speed. Near the cap, suppress an
    acceleration request. Other learned actions remain available below the cap.
    """
    # = stores a result in actions; use a dot to access actions_indexes on env.unwrapped.action_type.
    actions = env.unwrapped.action_type.actions_indexes
    # = stores a result in speed; call observed_speed using parentheses to supply its inputs.
    speed = observed_speed(env, observation)
    # if checks speed > speed_cap; only a true condition runs the indented block.
    if speed > speed_cap:
        # return ends this function and sends its result back to the code that called it.
        return actions["SLOWER"]
    # if checks action == actions['FASTER'] and speed > speed_cap - 2.5; only a true condition runs the indented block.
    if action == actions["FASTER"] and speed > speed_cap - 2.5:
        # return ends this function and sends its result back to the code that called it.
        return actions["IDLE"]
    # return ends this function and sends its result back to the code that called it.
    return action


# def defines render_frame; parentheses list its inputs, and the colon starts its indented body.
def render_frame(env):
    """Draw one simulator image and copy its pixels for a replay.
    The simulator normally turns drawing off without a physical display, so we
    explicitly enable it. Copying prevents later frames from overwriting earlier ones.

    Original implementation notes:
    Draw on an offscreen surface even with SDL's dummy video driver.

    HighwayEnv disables its viewer under the dummy driver by default, returning
    black frames. Offscreen surfaces need no window, so enable their drawing.
    """
    # if checks env.unwrapped.viewer is None; only a true condition runs the indented block.
    if env.unwrapped.viewer is None:
        # Call env.render using parentheses to supply its inputs; this line does not store a separate return value.
        env.render()
    # = stores a result in env.unwrapped.viewer.enabled; use the literal value True.
    env.unwrapped.viewer.enabled = True
    # return ends this function and sends its result back to the code that called it.
    return env.render().copy()


# def defines heuristic_action; parentheses list its inputs, and the colon starts its indented body.
def heuristic_action(env):
    """Drive using a hand-written following-distance rule rather than a neural network.
    This comparator reads extra simulator state, so it must be identified as having
    more information than the learned policies.

    Original implementation notes:
    Privileged full-state comparator: stay in lane and keep a generous gap.

    This rule sees simulator state, unlike the learned policies. Report it as a
    sanity check, not as an observation-matched algorithm comparison.
    """
    # = stores a result in vehicle; use a dot to access vehicle on env.unwrapped.
    vehicle = env.unwrapped.vehicle
    # = unpacks the right-hand result into (front, _), matching values to names in order.
    front, _ = env.unwrapped.road.neighbour_vehicles(vehicle)
    # = stores a result in actions; use a dot to access actions_indexes on env.unwrapped.action_type.
    actions = env.unwrapped.action_type.actions_indexes
    # if checks front is not None; only a true condition runs the indented block.
    if front is not None:
        # = stores a result in gap; call vehicle.lane_distance_to using parentheses to supply its inputs.
        gap = vehicle.lane_distance_to(front)
        # if checks gap < max(15.0, 1.8 * vehicle.speed); only a true condition runs the indented block.
        if gap < max(15.0, 1.8 * vehicle.speed):
            # return ends this function and sends its result back to the code that called it.
            return actions["SLOWER"]
    # if checks vehicle.speed < 25; only a true condition runs the indented block.
    if vehicle.speed < 25:
        # return ends this function and sends its result back to the code that called it.
        return actions["FASTER"]
    # return ends this function and sends its result back to the code that called it.
    return actions["IDLE"]


# def defines reset_training_episode; parentheses list its inputs, and the colon starts its indented body.
def reset_training_episode(env, density_rng, densities=(1.0,), seed=None):
    """Choose a traffic density before starting the next driving episode.
    Use a separate random generator so choosing density does not consume the random
    numbers used for exploration or replay sampling. One density reproduces the old
    fixed-density behavior. Equal episode probabilities do not mean equal step counts.

    Original implementation notes:
    Sample density at reset, with randomness separate from exploration/replay.
    """
    # if checks not densities or any((not np.isfinite(d) or d <= 0 for d in densities)); only a true condition runs the indented block.
    if not densities or any(not np.isfinite(d) or d <= 0 for d in densities):
        # raise signals an error and stops this normal path, so invalid data are not silently accepted.
        raise ValueError("Training densities must be finite and positive")
    # = stores a result in density; call float using parentheses to supply its inputs.
    density = float(
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        densities[0] if len(densities) == 1 else density_rng.choice(densities)
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # if checks env.unwrapped.config['vehicles_density'] != density; only a true condition runs the indented block.
    if env.unwrapped.config["vehicles_density"] != density:
        # Call env.unwrapped.configure using parentheses to supply its inputs; this line does not store a separate return value.
        env.unwrapped.configure({"vehicles_density": density})
    # = unpacks the right-hand result into (observation, info), matching values to names in order.
    observation, info = env.reset(seed=seed)
    # return ends this function and sends its result back to the code that called it.
    return observation, info, density
