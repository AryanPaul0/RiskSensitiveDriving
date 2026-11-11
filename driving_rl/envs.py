"""Set up the highway and translate between simulator data and the agent.

The neural network sees eight rows of five values: presence, x, y, vx and vy.
That is 40 inputs, not a complete description of all traffic. Observations are
partial: some vehicles and driver settings are missing. The five actions are
lane left, keep target speed, lane right, increase target speed and decrease it.
The simulator turns those requests into steering and acceleration.

Original implementation notes:
Environment and randomness shared by training and evaluation.
"""

import os  # Make os available for this file's calculations and workflow.
import random  # Make random available for this file's calculations and workflow.

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # Os.environ.setdefault applied to 'sdl_videodriver', 'dummy'.
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")  # Os.environ.setdefault applied to 'sdl_audiodriver', 'dummy'.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")  # Os.environ.setdefault applied to 'pygame_hide_support_prompt', '1'.

import gymnasium as gym  # Make gymnasium available for this file's calculations and workflow.
import highway_env  # Registers the environments with Gymnasium. Make highway_env available for this file's calculations and workflow.
import numpy as np  # Make numpy available for this file's calculations and workflow.
import torch  # Make torch available for this file's calculations and workflow.


def seed_everything(seed):  # Give Python, NumPy and PyTorch repeatable random starting points.
    """Give Python, NumPy and PyTorch repeatable random starting points.
    Using one CPU thread per process also avoids many threads competing for these
    small networks when several training jobs run at once.
    """
    random.seed(seed)  # Random.seed applied to seed.
    np.random.seed(seed)  # Np.random.seed applied to seed.
    torch.manual_seed(seed)  # Torch.manual seed applied to seed.
    torch.set_num_threads(1)  # These small networks usually run faster with one CPU thread.


def make_env(density=1.0, render=False, collision_penalty=-2.0):  # Create a highway with explicitly shared settings for fair comparisons.
    """Create a highway with explicitly shared settings for fair comparisons.
    Density changes how closely vehicles start together. It does not change the
    five available actions or the 40 observation inputs.
    """
    return gym.make(  # Return gym.make applied to 'highway-fast-v0'.
        "highway-fast-v0",  # HighwayEnv supplies vehicle dynamics and traffic drivers. Our agent chooses high-level maneuvers, not raw steering or throttle values.
        render_mode="rgb_array" if render else None,  # Include 'rgb_array' when render, otherwise None in the surrounding operation.
        config={  # Include the named settings or measurements assembled below in the surrounding operation.
            "lanes_count": 3,  # Record 'lanes_count' as 3.
            "vehicles_count": 15,  # Record 'vehicles_count' as 15.
            "vehicles_density": float(density),  # Record 'vehicles_density' as the traffic-density setting as a floating-point value.
            "duration": 30,  # This is a rollout cutoff, not a claim that driving ends after 30 s.
            "observation": {  # Record 'observation' as the named settings or measurements assembled below.
                "type": "Kinematics",  # Record 'type' as 'Kinematics'.
                "vehicles_count": 8,  # Record 'vehicles_count' as 8.
                "features": ["presence", "x", "y", "vx", "vy"],  # Eight slots x five features = 40 network inputs. Presence distinguishes a real nearby vehicle from an empty slot.
                "normalize": True,  # Record 'normalize' as True.
                "absolute": False,  # Record 'absolute' as False.
                "see_behind": True,  # Relative observations still omit distant cars and driver internals: this is not a fully observable Markov state.
            },  # Finish assembling the grouped values above.
            "collision_reward": float(collision_penalty),  # Explicit, shared reward: speed bonus and substantial crash penalty.
            "high_speed_reward": 0.4,  # Record 'high_speed_reward' as 0.4.
            "right_lane_reward": 0.0,  # Record 'right_lane_reward' as 0.0.
            "lane_change_reward": 0.0,  # Record 'lane_change_reward' as 0.0.
            "normalize_reward": False,  # Record 'normalize_reward' as False.
            "screen_width": 900,  # Keep the configured reward scale: rescaling rewards would also change what return predictions and loss magnitudes mean.
            "screen_height": 240,  # Record 'screen_height' as 240.
            "offscreen_rendering": True,  # Record 'offscreen_rendering' as True.
        },  # Finish assembling the grouped values above.
    )  # Finish assembling gym.make inputs.


def flatten(observation):  # Turn the 8-by-5 observation table into a row of 40 numbers.
    """Turn the 8-by-5 observation table into a row of 40 numbers.
    Keep float32 numbers because that is the number format used by the network.
    """
    return np.asarray(observation, dtype=np.float32).reshape(-1)  # Return np.asarray(observation, dtype=np.float32).reshape applied to negative 1.


def observed_speed(env, observation):  # Recover the ego car's speed from the observation it is allowed to see.
    """Recover the ego car's speed from the observation it is allowed to see.
    Undo normalization using the simulator's feature ranges, then combine horizontal
    and vertical velocity with the Pythagorean theorem. No hidden traffic data is used.

    Original implementation notes:
    Decode ego velocity using observation metadata, without simulator state.
    """
    observer = env.unwrapped.observation_type  # Set observer to env.unwrapped.observation type.
    ego = np.asarray(observation)[0]  # Set ego to the entries of the currently observed traffic features as a numerical array/tensor for batch calculations selected by 0.
    velocity = []  # Slot 0 is the controlled car. Decode only its two velocity features; the governor must not gain access to hidden traffic through env.vehicle.
    for feature in ["vx", "vy"]:  # Process each feature from 'vx', 'vy'.
        value = float(ego[observer.features.index(feature)])  # Set value to the entries of ego selected by observer.features.index(feature) as a floating-point value.
        if observer.normalize:  # Use this branch when observer.normalize.
            low, high = observer.features_range[feature]  # Set low, high to the entries of observer.features range selected by feature.
            value = low + (value + 1) * 0.5 * (high - low)  # Set value to (low plus (((value plus 1) times 0.5) times (high minus low))).
        velocity.append(value)  # Invert normalization: -1 maps to low, +1 maps to high. The resulting velocity component is in meters per second again.
    return float(np.linalg.norm(velocity))  # sqrt(vx**2 + vy**2): combine two velocity components into speed.


def govern_action(env, observation, action, speed_cap):  # Apply a simple speed rule after the network proposes an action.
    """Apply a simple speed rule after the network proposes an action.
    Above the threshold, request a lower target speed. Near it, block further
    acceleration. This is an action override, not a physical guarantee about speed.

    Original implementation notes:
    Simple ego-speed control; no traffic state or collision-avoidance claim.

    Above the cap, request a slower target speed. Near the cap, suppress an
    acceleration request. Other learned actions remain available below the cap.
    """
    actions = env.unwrapped.action_type.actions_indexes  # Set the recorded action indices to env.unwrapped.action type.actions indexes.
    speed = observed_speed(env, observation)  # Set speed to observed speed applied to env, the currently observed traffic features.
    if speed > speed_cap:  # Use this branch when speed is above speed cap.
        return actions["SLOWER"]  # Request a lower target speed; the simulator may take time to slow down, so this governor does not enforce an instantaneous hard limit.
    if action == actions["FASTER"] and speed > speed_cap - 2.5:  # Use this branch when action equals the entries of the recorded action indices selected by 'FASTER' and speed is above (speed cap minus 2.5).
        return actions["IDLE"]  # Near the cap, reject acceleration without otherwise changing the network's maneuver. This is a baseline control, not learned caution.
    return action  # Return action.


def render_frame(env):  # Draw one simulator image and copy its pixels for a replay.
    """Draw one simulator image and copy its pixels for a replay.
    The simulator normally turns drawing off without a physical display, so we
    explicitly enable it. Copying prevents later frames from overwriting earlier ones.

    Original implementation notes:
    Draw on an offscreen surface even with SDL's dummy video driver.

    HighwayEnv disables its viewer under the dummy driver by default, returning
    black frames. Offscreen surfaces need no window, so enable their drawing.
    """
    if env.unwrapped.viewer is None:  # Use this branch when env.unwrapped.viewer is None.
        env.render()  # The result of env.render.
    env.unwrapped.viewer.enabled = True  # Set env.unwrapped.viewer.enabled to True.
    return env.render().copy()  # Return an independent copy of the result of env.render.


def heuristic_action(env):  # Drive using a hand-written following-distance rule rather than a neural network.
    """Drive using a hand-written following-distance rule rather than a neural network.
    This comparator reads extra simulator state, so it must be identified as having
    more information than the learned policies.

    Original implementation notes:
    Privileged full-state comparator: stay in lane and keep a generous gap.

    This rule sees simulator state, unlike the learned policies. Report it as a
    sanity check, not as an observation-matched algorithm comparison.
    """
    vehicle = env.unwrapped.vehicle  # Set vehicle to env.unwrapped.vehicle.
    front, _ = env.unwrapped.road.neighbour_vehicles(vehicle)  # Set front, the unused result to env.unwrapped.road.neighbour vehicles applied to vehicle.
    actions = env.unwrapped.action_type.actions_indexes  # Set the recorded action indices to env.unwrapped.action type.actions indexes.
    if front is not None:  # Use this branch when front is not None.
        gap = vehicle.lane_distance_to(front)  # Set gap to vehicle.lane distance to applied to front.
        if gap < max(15.0, 1.8 * vehicle.speed):  # Use this branch when gap is below max applied to 15.0, (1.8 times vehicle.speed).
            return actions["SLOWER"]  # Return the entries of the recorded action indices selected by 'SLOWER'.
    if vehicle.speed < 25:  # Use this branch when vehicle.speed is below 25.
        return actions["FASTER"]  # Return the entries of the recorded action indices selected by 'FASTER'.
    return actions["IDLE"]  # Return the entries of the recorded action indices selected by 'IDLE'.


def reset_training_episode(env, density_rng, densities=(1.0,), seed=None):  # Choose a traffic density before starting the next driving episode.
    """Choose a traffic density before starting the next driving episode.
    Use a separate random generator so choosing density does not consume the random
    numbers used for exploration or replay sampling. One density reproduces the old
    fixed-density behavior. Equal episode probabilities do not mean equal step counts.

    Original implementation notes:
    Sample density at reset, with randomness separate from exploration/replay.
    """
    if not densities or any(not np.isfinite(d) or d <= 0 for d in densities):  # Use this branch when not densities or any applied to not np.isfinite(d) or d <= 0 for each d in densities.
        raise ValueError("Training densities must be finite and positive")  # Reject this invalid case instead of continuing with misleading experiment results.
    density = float(  # Set the traffic-density setting to the entries of densities selected by 0 when the number of entries in densities equals 1, otherwise density rng.choice applied to densities as a floating-point value.
        densities[0] if len(densities) == 1 else density_rng.choice(densities)  # A single density avoids making an unnecessary random draw; this preserves the random sequence of the original fixed-density setup.
    )  # Finish assembling float inputs.
    if env.unwrapped.config["vehicles_density"] != density:  # Use this branch when the entries of env.unwrapped.config selected by 'vehicles_density' differs from the traffic-density setting.
        env.unwrapped.configure({"vehicles_density": density})  # Env.unwrapped.configure applied to the named settings or measurements assembled below.
    observation, info = env.reset(seed=seed)  # Set the currently observed traffic features, info to the result of env.reset.
    return observation, info, density  # Return the currently observed traffic features, info, the traffic-density setting.
