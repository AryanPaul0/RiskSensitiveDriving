"""Connect the driving simulator to the learning code.

train: collect driving experiences, learn from them, and save the model.
evaluate: use a saved model without changing its learned weights.
benchmark: measure how quickly the computer can run these operations.
A seed is a number that makes random choices repeatable. Training and evaluation
use different seeds, so the final checks do not just repeat training situations.

Original implementation notes:
Train, checkpoint, benchmark, and evaluate on explicitly separated seeds.
"""

import csv  # Make csv available for this file's calculations and workflow.
import hashlib  # Make hashlib available for this file's calculations and workflow.
import json  # Make json available for this file's calculations and workflow.
from pathlib import Path  # Make Path available for this file's calculations and workflow.
import platform  # Make platform available for this file's calculations and workflow.
import time  # Make time available for this file's calculations and workflow.

import imageio.v2 as imageio  # Make imageio.v2 available for this file's calculations and workflow.
import numpy as np  # Make numpy available for this file's calculations and workflow.
import torch  # Make torch available for this file's calculations and workflow.

from .agents import Agent, ReplayBuffer  # Make Agent, ReplayBuffer available for this file's calculations and workflow.
from .envs import (  # Make flatten, govern_action, heuristic_action, make_env, render_frame, seed_everything, reset_training_episode available for this file's calculations and workflow.
    flatten,  # Continue the calculation or input list started above.
    govern_action,  # Continue the calculation or input list started above.
    heuristic_action,  # Continue the calculation or input list started above.
    make_env,  # Continue the calculation or input list started above.
    render_frame,  # Continue the calculation or input list started above.
    seed_everything,  # Continue the calculation or input list started above.
    reset_training_episode,  # Continue the calculation or input list started above.
)  # Finish assembling the grouped values above.


def save_json(path, value):  # Save dictionaries and lists as readable JSON, creating the destination folder if needed.
    """Save dictionaries and lists as readable JSON, creating the destination folder
    if needed. These files hold results and settings, not neural-network calculations.
    """
    path = Path(path)  # Set path to Path applied to path.
    path.parent.mkdir(parents=True, exist_ok=True)  # The result of path.parent.mkdir.
    path.write_text(json.dumps(value, indent=2) + "\n")  # Path.write text applied to (json.dumps applied to value plus '\n').


def versions():  # Record Python and library versions so an experiment can be reproduced later.
    """Record Python and library versions so an experiment can be reproduced later."""  # Describe this operation for readers and help().
    from importlib.metadata import version  # Make version available for this file's calculations and workflow.

    return {  # Return the named settings or measurements assembled below.
        "python": platform.python_version(),  # Record 'python' as the result of platform.python version.
        "platform": platform.platform(),  # Record 'platform' as the result of platform.platform.
        **{  # Include version applied to name for each name in 'torch', 'numpy', 'gymnasium', 'highway-env' in the surrounding operation.
            name: version(name)  # Include name, version applied to name in the surrounding operation.
            for name in ["torch", "numpy", "gymnasium", "highway-env"]  # Include name, 'torch', 'numpy', 'gymnasium', 'highway-env' in the surrounding operation.
        },  # Finish assembling the grouped values above.
    }  # Finish assembling the grouped values above.


def source_hashes():  # Record fingerprints of the Python source files used by this run.
    """Record fingerprints of the Python source files used by this run.
    A comment-only edit changes a file fingerprint even when calculations are unchanged.
    """
    return {  # Return the result of hashlib.sha256(path.read bytes()).hexdigest for each path in sorted applied to Path( file ).parent.glob applied to '*.py'.
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()  # Include path.name, the result of hashlib.sha256(path.read bytes()).hexdigest in the surrounding operation.
        for path in sorted(Path(__file__).parent.glob("*.py"))  # Include path, sorted applied to Path( file ).parent.glob applied to '*.py' in the surrounding operation.
    }  # Finish assembling the grouped values above.


def build_agent(config):  # Construct the learner described by a saved configuration dictionary.
    """Construct the learner described by a saved configuration dictionary."""  # Describe this operation for readers and help().
    return Agent(  # Return Agent applied to the entries of config selected by 'observation_dim', the entries of config selected by 'n_actions', the entries of config selected by 'algorithm', the entries of config selected by 'quantiles', the entries of config selected by 'learning_rate', the entries of config selected by 'gamma'.
        config["observation_dim"],  # Include the entries of config selected by 'observation_dim' in the surrounding operation.
        config["n_actions"],  # Include the entries of config selected by 'n_actions' in the surrounding operation.
        config["algorithm"],  # Include the entries of config selected by 'algorithm' in the surrounding operation.
        config["quantiles"],  # Include the entries of config selected by 'quantiles' in the surrounding operation.
        config["learning_rate"],  # Include the entries of config selected by 'learning_rate' in the surrounding operation.
        config["gamma"],  # Include the entries of config selected by 'gamma' in the surrounding operation.
    )  # Finish assembling Agent inputs.


def save_checkpoint(path, agent, config, step):  # Save learned weights, target weights, optimizer state, settings and step count.
    """Save learned weights, target weights, optimizer state, settings and step count.
    Write a temporary file first, then rename it, to avoid leaving a partly saved
    checkpoint. Replay memory and random-generator state are not saved for exact resume.
    """
    temporary = Path(str(path) + ".tmp")  # Set temporary to Path applied to (path as a text value plus '.tmp').
    torch.save(  # Torch.save applied to the named settings or measurements assembled below, temporary.
        {  # Include the named settings or measurements assembled below in the surrounding operation.
            "config": config,  # Record 'config' as config.
            "step": step,  # Record 'step' as step.
            "online": agent.online.state_dict(),  # Record 'online' as the result of agent.online.state dict.
            "target": agent.target.state_dict(),  # Record 'target' as the result of agent.target.state dict.
            "optimizer": agent.optimizer.state_dict(),  # Record 'optimizer' as the result of agent.optimizer.state dict.
        },  # Finish assembling the grouped values above.
        temporary,  # Include temporary in the surrounding operation.
    )  # Finish assembling torch.save inputs.
    temporary.replace(path)  # Temporary.replace applied to path.


def load_agent(path):  # Load a saved network for inference: making decisions without learning.
    """Load a saved network for inference: making decisions without learning.
    Evaluation reads the online network; it does not update its weights.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)  # Set checkpoint to torch.load applied to path.
    agent = build_agent(checkpoint["config"])  # Set agent to a learner built from the entries of checkpoint selected by 'config'.
    agent.online.load_state_dict(checkpoint["online"])  # Agent.online.load state dict applied to the entries of checkpoint selected by 'online'.
    agent.online.eval()  # The result of agent.online.eval.
    return agent, checkpoint  # eval() sets inference mode for layers such as dropout/batch normalization. It does not disable gradients by itself; Agent.act uses no_grad for that.


def exploration_epsilon(step, total_steps, fraction=0.6, final=0.05, decay_steps=None):  # Calculate the chance of taking a random action at this training step.
    """Calculate the chance of taking a random action at this training step.
    Start at 1 (always random), decrease linearly, and stop at the final chance.
    An absolute decay horizon keeps exploration comparable across training budgets.

    Original implementation notes:
    Optional absolute decay horizon keeps exploration fixed across budgets.
    """
    horizon = decay_steps if decay_steps is not None else total_steps * fraction  # Set horizon to decay steps when decay steps is not None, otherwise (total steps times fraction).
    progress = min(1.0, max(0.0, (step - 1) / max(1, horizon)))  # Clamp progress to [0,1]: before decay finishes epsilon falls linearly; afterward it stays at final rather than becoming negative.
    return 1.0 + progress * (final - 1.0)  # Return (1.0 plus (progress times (final minus 1.0))).


def train(args):  # Train one agent through repeated driving steps.
    """Train one agent through repeated driving steps.
    1. Observe traffic and choose either a random or a learned action.
    2. Store the resulting transition in replay memory.
    3. Sample earlier transitions and update the network.
    4. Occasionally copy weights to the target network and save checkpoints.
    5. Start another episode after a crash or recording cutoff.
    Episode counters describe data collection; learning happens from replay batches.
    """
    output = Path(args.out)  # Set output to Path applied to args.out.
    if output.exists() and any(output.iterdir()):  # Use this branch when the result of output.exists and any applied to the result of output.iterdir.
        raise ValueError(f"Refusing to overwrite existing experiment: {output}")  # Reject this invalid case instead of continuing with misleading experiment results.
    output.mkdir(parents=True, exist_ok=True)  # The result of output.mkdir.
    seed_everything(args.seed)  # Initialize repeatable random streams using args.seed.
    rng = np.random.default_rng(args.seed)  # Set rng to a separate reproducible random generator seeded with args.seed.
    env = make_env(collision_penalty=args.collision_penalty)  # Set env to a highway simulator configured with.
    densities = getattr(args, "training_densities", None) or [1.0]  # Set densities to getattr applied to args, 'training_densities', None or 1.0.
    density_rng = np.random.default_rng(np.random.SeedSequence([args.seed, 8675309]))  # Keep density sampling separate from exploration/replay randomness.
    observation, _, episode_density = reset_training_episode(  # Set the currently observed traffic features, the unused result, episode density to a fresh driving episode and sampled density from env, density rng, densities, args.seed.
        env, density_rng, densities, args.seed  # Include env, density rng, densities, args.seed in the surrounding operation.
    )  # Finish assembling reset training episode inputs.
    density_steps = {str(float(d)): 0 for d in densities}  # Set density steps to 0 for each d in densities.
    density_episodes = {str(float(d)): 0 for d in densities}  # Set density episodes to 0 for each d in densities.
    env.action_space.seed(args.seed)  # Env.action space.seed applied to args.seed.
    observation = flatten(observation)  # Set the currently observed traffic features to the 40 network inputs flattened from the currently observed traffic features.
    config = {  # Build the network from the actual simulator input and action sizes, rather than silently assuming a changed environment still has 40 and 5.
        **vars(args),  # Include vars applied to args in the surrounding operation.
        "observation_dim": len(observation),  # Record 'observation_dim' as the number of entries in the currently observed traffic features.
        "n_actions": int(env.action_space.n),  # Record 'n_actions' as env.action space.n as a whole-number value.
    }  # Finish assembling the grouped values above.
    config.pop("func", None)  # Config.pop applied to 'func', none.
    save_json(output / "config.json", config)  # Save the following results/settings as json: the path to 'config.json' inside output, config.
    save_json(output / "versions.json", versions())  # Save the following results/settings as json: the path to 'versions.json' inside output, the result of versions.
    save_json(output / "environment.json", env.unwrapped.config)  # Save the following results/settings as json: the path to 'environment.json' inside output, env.unwrapped.config.
    save_json(output / "source_hashes.json", source_hashes())  # Save the following results/settings as json: the path to 'source_hashes.json' inside output, the result of source hashes.
    agent = build_agent(config)  # Set agent to a learner built from config.
    replay = ReplayBuffer(args.buffer_size)  # Learning reuses this finite memory rather than relying only on the latest step.
    episode_return = 0.0  # Set episode return to 0.0.
    episode_length = 0  # Set episode length to 0.
    episode = 0  # Set episode to 0.
    started = time.perf_counter()  # Set started to the high-resolution elapsed-time clock reading.
    last_loss = last_grad = float("nan")  # Set last loss, last grad to 'nan' as a floating-point value.
    with (output / "training.csv").open("w", newline="") as file:  # Keep this temporary context active during the enclosed work, then clean it up.
        fields = [  # Set fields to the listed settings or measurements in their recorded order.
            "step",  # Include 'step' in the surrounding operation.
            "episode",  # Include 'episode' in the surrounding operation.
            "return",  # Include 'return' in the surrounding operation.
            "length",  # Include 'length' in the surrounding operation.
            "crashed",  # Include 'crashed' in the surrounding operation.
            "epsilon",  # Include 'epsilon' in the surrounding operation.
            "loss",  # Include 'loss' in the surrounding operation.
            "grad_norm",  # Include 'grad_norm' in the surrounding operation.
            "elapsed_seconds",  # Include 'elapsed_seconds' in the surrounding operation.
            "steps_per_second",  # Include 'steps_per_second' in the surrounding operation.
            "density",  # Include 'density' in the surrounding operation.
        ]  # Finish assembling the grouped values above.
        writer = csv.DictWriter(file, fieldnames=fields)  # Set writer to csv.DictWriter applied to file.
        writer.writeheader()  # The result of writer.writeheader.
        try:  # Run this work with the error handling or resource cleanup specified below.
            for step in range(1, args.steps + 1):  # The training budget counts simulator interactions, not episodes.
                epsilon = exploration_epsilon(  # Epsilon is the probability of exploring with a random action.
                    step,  # Include step in the surrounding operation.
                    args.steps,  # Include args.steps in the surrounding operation.
                    args.exploration_fraction,  # Include args.exploration fraction in the surrounding operation.
                    args.final_epsilon,  # Include args.final epsilon in the surrounding operation.
                    getattr(args, "exploration_steps", None),  # Include getattr applied to args, 'exploration_steps', None in the surrounding operation.
                )  # Finish assembling exploration epsilon inputs.
                action = (  # Exploration tries alternatives; exploitation uses the best current prediction.
                    int(rng.integers(env.action_space.n))  # A random integer is uniform over the five actions. The other branch uses mean-greedy values (alpha defaults to 1), including during QR-DQN training. Epsilon is independent of the later lower-tail deployment rule.
                    if rng.random() < epsilon  # Include the result of rng.random is below the probability of taking a random action in the surrounding operation.
                    else agent.act(observation)  # Include agent.act applied to the currently observed traffic features in the surrounding operation.
                )  # Finish assembling the grouped values above.
                next_observation, reward, terminated, truncated, info = env.step(action)  # terminated means the task truly ended; truncated means its recording limit was reached.
                next_observation = flatten(next_observation)  # Set the traffic features after the action to the 40 network inputs flattened from the traffic features after the action.
                density_steps[str(episode_density)] += 1  # Increase the entries of density steps selected by str(episode_density) using 1.
                replay.add(observation, action, reward, next_observation, terminated)  # Save the final next observation before any reset, so cutoff bootstrapping uses the right state.
                observation = next_observation  # Deliberately omit truncated from the stored terminal flag: cutoff transitions still learn from their final observation.
                episode_return += float(reward)  # Increase episode return using reward as a floating-point value.
                episode_length += 1  # This logged episode score is an UNDISCOUNTED reward sum. Bellman targets use gamma; the log need not equal Q(o,a).
                if step >= args.learning_starts and len(replay) >= args.batch_size:  # Wait for enough experience before sampling mini-batches for learning.
                    if step % args.train_frequency == 0:  # Use this branch when (step modulo args.train frequency) equals 0.
                        last_loss, last_grad = agent.update(  # % tests the update schedule: only multiples trigger learning. This counts environment steps, so replay updates and interaction budgets remain distinct.
                            replay.sample(args.batch_size, rng)  # Random minibatches can contain transitions from earlier policies/episodes: this is off-policy TD learning, not a complete-episode Monte Carlo update.
                        )  # Finish assembling agent.update inputs.
                if step % args.target_interval == 0:  # Refresh the target only on its schedule, rather than moving both networks at every update.
                    agent.sync_target()  # Hold bootstrap targets steady between scheduled copies; chasing a changing target at every update is less stable.
                if terminated or truncated:  # Both conditions start a fresh episode, but only true termination removes the bootstrap term.
                    episode += 1  # Increase episode using 1.
                    density_episodes[str(episode_density)] += 1  # Increase the entries of density episodes selected by str(episode_density) using 1.
                    elapsed = time.perf_counter() - started  # Set elapsed to (the high-resolution elapsed-time clock reading minus started).
                    writer.writerow(  # Write episode statistics immediately so another Terminal can show progress during training.
                        dict(  # Include dict applied to matching entries grouped across fields, the listed settings or measurements in their recorded order in the surrounding operation.
                            zip(  # Include matching entries grouped across fields, the listed settings or measurements in their recorded order in the surrounding operation.
                                fields,  # Include fields in the surrounding operation.
                                [  # Include the listed settings or measurements in their recorded order in the surrounding operation.
                                    step,  # Include step in the surrounding operation.
                                    episode,  # Include episode in the surrounding operation.
                                    episode_return,  # Include episode return in the surrounding operation.
                                    episode_length,  # Include episode length in the surrounding operation.
                                    int(  # Include info.get applied to 'crashed', env.unwrapped.vehicle.crashed as a whole-number value in the surrounding operation.
                                        info.get(  # Include info.get applied to 'crashed', env.unwrapped.vehicle.crashed in the surrounding operation.
                                            "crashed", env.unwrapped.vehicle.crashed  # Include 'crashed', env.unwrapped.vehicle.crashed in the surrounding operation.
                                        )  # Finish assembling info.get inputs.
                                    ),  # Finish assembling int inputs.
                                    epsilon,  # Include the probability of taking a random action in the surrounding operation.
                                    last_loss,  # Include last loss in the surrounding operation.
                                    last_grad,  # Include last grad in the surrounding operation.
                                    elapsed,  # Include elapsed in the surrounding operation.
                                    step / elapsed,  # Include (step divided by elapsed) in the surrounding operation.
                                    episode_density,  # Include episode density in the surrounding operation.
                                ],  # Finish assembling the grouped values above.
                            )  # Finish assembling zip inputs.
                        )  # Finish assembling dict inputs.
                    )  # Finish assembling writer.writerow inputs.
                    file.flush()  # The result of file.flush.
                    observation, _, episode_density = reset_training_episode(  # Set the currently observed traffic features, the unused result, episode density to a fresh driving episode and sampled density from env, density rng, densities.
                        env, density_rng, densities  # Reset only after storing the final transition. Using a new episode's first observation as its predecessor's next observation would create a false Bellman target.
                    )  # Finish assembling reset training episode inputs.
                    observation = flatten(observation)  # Set the currently observed traffic features to the 40 network inputs flattened from the currently observed traffic features.
                    episode_return = 0.0  # Set episode return to 0.0.
                    episode_length = 0  # Set episode length to 0.
                if step % args.checkpoint_interval == 0:  # Use this branch when (step modulo args.checkpoint interval) equals 0.
                    save_checkpoint(  # Save checkpoint applied to (output divided by f'checkpoint {step}.pt'), agent, config, step.
                        output / f"checkpoint_{step}.pt", agent, config, step  # Include (output divided by f'checkpoint {step}.pt'), agent, config, step in the surrounding operation.
                    )  # Finish assembling save checkpoint inputs.
                    elapsed = time.perf_counter() - started  # Set elapsed to (the high-resolution elapsed-time clock reading minus started).
                    print(  # Print applied to f'{args.algorithm} step={step}/{args.steps} episodes={episode} loss={last loss:.4f} steps/s={step / elapsed:.1f}'.
                        f"{args.algorithm} step={step}/{args.steps} episodes={episode} "  # Include f'{args.algorithm} step={step}/{args.steps} episodes={episode} loss={last loss:.4f} steps/s={step / elapsed:.1f}' in the surrounding operation.
                        f"loss={last_loss:.4f} steps/s={step / elapsed:.1f}",  # Include {last loss:.4f}, ' steps/s=', {step / elapsed:.1f} in the surrounding operation.
                        flush=True,  # Include True in the surrounding operation.
                    )  # Finish assembling print inputs.
            save_checkpoint(output / "final.pt", agent, config, args.steps)  # Save the requested final model, not a model chosen after inspecting test results.
            save_json(  # Save the following results/settings as json: the path to 'summary.json' inside output, the named settings or measurements assembled below.
                output / "summary.json",  # Include the path to 'summary.json' inside output in the surrounding operation.
                {  # Include the named settings or measurements assembled below in the surrounding operation.
                    "steps": args.steps,  # Record 'steps' as args.steps.
                    "episodes": episode,  # Record 'episodes' as episode.
                    "elapsed_seconds": time.perf_counter() - started,  # Record 'elapsed_seconds' as (the high-resolution elapsed-time clock reading minus started).
                    "training_complete": True,  # Record 'training_complete' as True.
                    "completed_episodes_by_density": density_episodes,  # Record 'completed_episodes_by_density' as density episodes.
                    "steps_by_density": density_steps,  # Record 'steps_by_density' as density steps.
                },  # Finish assembling the grouped values above.
            )  # Finish assembling save json inputs.
        except KeyboardInterrupt:  # Handle this error or interruption through the recovery block below.
            save_checkpoint(output / "interrupted.pt", agent, config, step)  # Save checkpoint applied to the path to 'interrupted.pt' inside output, agent, config, step.
            print(  # Print applied to the report text.
                "Interrupted: saved model and optimizer; replay/RNG resume is not supported.",  # Include the report text in the surrounding operation.
                flush=True,  # Include True in the surrounding operation.
            )  # Finish assembling print inputs.
        finally:  # Always release resources here, including after an error or interruption.
            env.close()  # The result of env.close.


def benchmark(args):  # Time simulator steps and network updates separately.
    """Time simulator steps and network updates separately.
    This is a rough speed estimate, not evidence that a policy drives well.
    """
    seed_everything(args.seed)  # Initialize repeatable random streams using args.seed.
    env = make_env()  # Set env to a highway simulator configured with.
    observation, _ = env.reset(seed=args.seed)  # Set the currently observed traffic features, the unused result to the result of env.reset.
    env.action_space.seed(args.seed)  # Env.action space.seed applied to args.seed.
    started = time.perf_counter()  # Set started to the high-resolution elapsed-time clock reading.
    for _ in range(args.steps):  # Process each the unused result from range applied to args.steps.
        observation, _, terminated, truncated, _ = env.step(env.action_space.sample())  # Set the currently observed traffic features, the unused result, the true-terminal flags, the recording-cutoff flags, the unused result to env.step applied to the result of env.action space.sample.
        if terminated or truncated:  # Both conditions start a fresh episode, but only true termination removes the bootstrap term.
            observation, _ = env.reset()  # Set the currently observed traffic features, the unused result to the result of env.reset.
    elapsed = time.perf_counter() - started  # Set elapsed to (the high-resolution elapsed-time clock reading minus started).
    agent = Agent(len(flatten(observation)), env.action_space.n, "qrdqn")  # Includes an actual optimizer update, but excludes rendering and disk I/O.
    rng = np.random.default_rng(args.seed)  # Set rng to a separate reproducible random generator seeded with args.seed.
    synthetic_batch = (  # Set synthetic batch to torch.from numpy applied to rng.normal(size=(64, len(flatten(observation)))).astype applied to np.float32, torch.from numpy applied to rng.integers applied to env.action space.n, torch.zeros applied to 64, torch.zeros applied to 64, the number of entries in the 40 network inputs flattened from the currently observed traffic features, torch.zeros applied to 64.
        torch.from_numpy(  # Include torch.from numpy applied to rng.normal(size=(64, len(flatten(observation)))).astype applied to np.float32 in the surrounding operation.
            rng.normal(size=(64, len(flatten(observation)))).astype(np.float32)  # Include rng.normal(size=(64, len(flatten(observation)))).astype applied to np.float32 in the surrounding operation.
        ),  # Finish assembling torch.from numpy inputs.
        torch.from_numpy(rng.integers(env.action_space.n, size=64)),  # Include torch.from numpy applied to rng.integers applied to env.action space.n in the surrounding operation.
        torch.zeros(64),  # Include torch.zeros applied to 64 in the surrounding operation.
        torch.zeros(64, len(flatten(observation))),  # Include torch.zeros applied to 64, the number of entries in the 40 network inputs flattened from the currently observed traffic features in the surrounding operation.
        torch.zeros(64),  # Include torch.zeros applied to 64 in the surrounding operation.
    )  # Finish assembling the grouped values above.
    agent.update(synthetic_batch)  # Agent.update applied to synthetic batch.
    update_start = time.perf_counter()  # Set update start to the high-resolution elapsed-time clock reading.
    for _ in range(200):  # Process each the unused result from range applied to 200.
        agent.update(synthetic_batch)  # Agent.update applied to synthetic batch.
    update_seconds = (time.perf_counter() - update_start) / 200  # Set update seconds to ((the high-resolution elapsed-time clock reading minus update start) divided by 200).
    result = {  # Set result to the named settings or measurements assembled below.
        "environment_steps": args.steps,  # Record 'environment_steps' as args.steps.
        "environment_seconds": elapsed,  # Record 'environment_seconds' as elapsed.
        "environment_steps_per_second": args.steps / elapsed,  # Record 'environment_steps_per_second' as (args.steps divided by elapsed).
        "qrdqn_update_seconds_batch64": update_seconds,  # Record 'qrdqn_update_seconds_batch64' as update seconds.
        "estimated_20000_step_run_seconds": 20000  # Record 'estimated_20000_step_run_seconds' as (20000 times ((elapsed divided by args.steps) plus update seconds)).
        * (elapsed / args.steps + update_seconds),  # Include ((elapsed divided by args.steps) plus update seconds) in the surrounding operation.
        "estimate_note": "Rough estimate; omits replay sampling, inference, logging and checkpointing.",  # Record 'estimate_note' as the report text.
        "versions": versions(),  # Record 'versions' as the result of versions.
    }  # Finish assembling the grouped values above.
    save_json(args.out, result)  # Save the following results/settings as json: args.out, result.
    env.close()  # The result of env.close.
    print(json.dumps(result, indent=2))  # Print applied to json.dumps applied to result.


def evaluate(args):  # Measure a frozen policy on specified traffic seeds and densities.
    """Measure a frozen policy on specified traffic seeds and densities.
    No optimizer updates happen here. Save outcomes for every episode so policies
    can be compared on the same scenarios later. A GIF records one predefined case.
    """
    seed_everything(0)  # Initialize repeatable random streams using 0.
    if args.seed_start < 10000:  # Use this branch when args.seed start is below 10000.
        raise ValueError(  # Reject this invalid case instead of continuing with misleading experiment results.
            "Use evaluation seeds >= 10000; reserve lower seeds for development."  # Include the report text in the surrounding operation.
        )  # Finish assembling ValueError inputs.
    if Path(args.out).exists():  # Use this branch when the result of Path(args.out).exists.
        raise ValueError(f"Refusing to overwrite evaluation: {args.out}")  # Reject this invalid case instead of continuing with misleading experiment results.
    agent = checkpoint = None  # Set agent, checkpoint to None.
    if args.checkpoint:  # Use this branch when args.checkpoint.
        agent, checkpoint = load_agent(args.checkpoint)  # Set agent, checkpoint to the frozen learner and checkpoint loaded from args.checkpoint.
    if args.policy == "learned" and agent is None:  # Use this branch when args.policy equals 'learned' and agent is None.
        raise ValueError("A learned policy requires --checkpoint")  # Reject this invalid case instead of continuing with misleading experiment results.
    if agent is not None and agent.algorithm == "dqn" and args.alpha != 1:  # Use this branch when agent is not None and agent.algorithm equals 'dqn' and args.alpha differs from 1.
        raise ValueError("Lower-tail selection requires a distributional checkpoint")  # Reject this invalid case instead of continuing with misleading experiment results.
    rows = []  # Set rows to .
    for density in args.densities:  # Evaluate the same policy in each specified traffic setting.
        env = make_env(density=density, render=bool(args.gif))  # Set env to a highway simulator configured with.
        for index in range(args.episodes):  # Process each index from range applied to args.episodes.
            episode_seed = args.seed_start + index  # Identical scenario seeds let different policies start from matching traffic situations.
            observation, _ = env.reset(seed=episode_seed)  # Set the currently observed traffic features, the unused result to the result of env.reset.
            action_rng = np.random.default_rng(episode_seed)  # Set action rng to a separate reproducible random generator seeded with episode seed.
            total_return = total_speed = distance = 0.0  # Give random-policy actions their own generator, so comparisons do not consume the simulator's traffic random-number stream.
            length = 0  # Set length to 0.
            action_counts = [0] * int(env.action_space.n)  # Set action counts to (0 times env.action space.n as a whole-number value).
            record = bool(args.gif) and index == 0 and density == args.densities[0]  # Render only the first episode at the first density. Only the predefined first case becomes a replay; do not select the best-looking outcome.
            frames = [render_frame(env)] if record else []  # Set frames to render frame applied to env when record, otherwise .
            while True:  # Keep processing while True.
                if args.policy == "random":  # Baselines and learned agents share the same simulator and reward measurements.
                    action = int(action_rng.integers(env.action_space.n))  # Set action to action rng.integers applied to env.action space.n as a whole-number value.
                elif args.policy == "heuristic":  # Use this branch when args.policy equals 'heuristic'.
                    action = heuristic_action(env)  # Set action to heuristic action applied to env.
                else:  # Handle the alternative when the earlier condition (args.policy equals 'heuristic') is false.
                    action = agent.act(flatten(observation), args.alpha)  # Set action to agent.act applied to the 40 network inputs flattened from the currently observed traffic features, args.alpha.
                    if args.speed_cap is not None:  # Use this branch when args.speed cap is not None.
                        action = govern_action(env, observation, action, args.speed_cap)  # Set action to govern action applied to env, the currently observed traffic features, action, args.speed cap.
                action_counts[action] += 1  # Increase the entries of action counts selected by action using 1.
                before = env.unwrapped.vehicle.position.copy()  # Set before to an independent copy of env.unwrapped.vehicle.position.
                observation, reward, terminated, truncated, info = env.step(action)  # Set the currently observed traffic features, reward, the true-terminal flags, the recording-cutoff flags, info to env.step applied to action.
                total_return += float(reward)  # Increase total return using reward as a floating-point value.
                total_speed += float(env.unwrapped.vehicle.speed)  # Increase total speed using env.unwrapped.vehicle.speed as a floating-point value.
                distance += float(  # Increase distance using np.linalg.norm applied to (env.unwrapped.vehicle.position minus before) as a floating-point value.
                    np.linalg.norm(env.unwrapped.vehicle.position - before)  # Include np.linalg.norm applied to (env.unwrapped.vehicle.position minus before) in the surrounding operation.
                )  # Finish assembling float inputs.
                length += 1  # Increase length using 1.
                if record:  # Use this branch when record.
                    frames.append(render_frame(env))  # Add render frame applied to env to frames.
                if terminated or truncated:  # Both conditions start a fresh episode, but only true termination removes the bootstrap term.
                    rows.append(  # Add the named settings or measurements assembled below to rows.
                        {  # Include the named settings or measurements assembled below in the surrounding operation.
                            "density": density,  # Record 'density' as the traffic-density setting.
                            "seed": episode_seed,  # Record 'seed' as episode seed.
                            "return": total_return,  # Record 'return' as total return.
                            "length": length,  # Record 'length' as length.
                            "crashed": int(env.unwrapped.vehicle.crashed),  # Record 'crashed' as env.unwrapped.vehicle.crashed as a whole-number value.
                            "mean_speed_m_s": total_speed / length,  # Record 'mean_speed_m_s' as (total speed divided by length).
                            "distance_m": distance,  # Record 'distance_m' as distance.
                            "action_counts": action_counts,  # Record 'action_counts' as action counts.
                            "terminated": bool(terminated),  # Record 'terminated' as the true-terminal flags as a true/false value.
                            "truncated": bool(truncated),  # Record 'truncated' as the recording-cutoff flags as a true/false value.
                        }  # Finish assembling the grouped values above.
                    )  # Finish assembling rows.append inputs.
                    break  # Stop this loop now that its stopping condition has been reached.
            if record:  # Use this branch when record.
                path = Path(args.gif)  # Set path to Path applied to args.gif.
                path.parent.mkdir(parents=True, exist_ok=True)  # The result of path.parent.mkdir.
                imageio.mimsave(path, frames, duration=1000, loop=0)  # Each policy decision represents one simulated second.
        env.close()  # The result of env.close.
    summary = []  # Set summary to .
    for density in args.densities:  # Evaluate the same policy in each specified traffic setting.
        selected = [row for row in rows if row["density"] == density]  # Set the selected next-action return predictions to row for each row in rows, keeping only cases where the entries of row selected by 'density' equals the traffic-density setting.
        returns = np.array([row["return"] for row in selected])  # Set returns to the entries of row selected by 'return' for each row in the selected next-action return predictions as a numerical array/tensor for batch calculations.
        count = max(1, int(np.ceil(0.1 * len(returns))))  # Set count to max applied to 1, np.ceil applied to (0.1 times len(returns)) as a whole-number value.
        summary.append(  # Add the named settings or measurements assembled below to summary.
            {  # Include the named settings or measurements assembled below in the surrounding operation.
                "density": density,  # Record 'density' as the traffic-density setting.
                "episodes": len(selected),  # Record 'episodes' as the number of entries in the selected next-action return predictions.
                "crash_rate": float(np.mean([row["crashed"] for row in selected])),  # Record 'crash_rate' as the average of the entries of row selected by 'crashed' for each row in the selected next-action return predictions as a floating-point value.
                "mean_return": float(returns.mean()),  # Record 'mean_return' as the average of returns as a floating-point value.
                "return_standard_error": (  # Record 'return_standard_error' as (the result of returns.std divided by np.sqrt applied to len(returns)) as a floating-point value when the number of entries in returns is above 1, otherwise None.
                    float(returns.std(ddof=1) / np.sqrt(len(returns)))  # Include (the result of returns.std divided by np.sqrt applied to len(returns)) as a floating-point value when the number of entries in returns is above 1, otherwise None in the surrounding operation.
                    if len(returns) > 1  # Include the number of entries in returns is above 1 in the surrounding operation.
                    else None  # Include None in the surrounding operation.
                ),  # Finish assembling the grouped values above.
                "worst_10_percent_mean_return": float(np.sort(returns)[:count].mean()),  # Record 'worst_10_percent_mean_return' as the average of the entries of the values sorted from lowest to highest in returns selected by :count as a floating-point value.
                "mean_speed_m_s": float(  # Record 'mean_speed_m_s' as the average of the entries of row selected by 'mean_speed_m_s' for each row in the selected next-action return predictions as a floating-point value.
                    np.mean([row["mean_speed_m_s"] for row in selected])  # Include the average of the entries of row selected by 'mean_speed_m_s' for each row in the selected next-action return predictions in the surrounding operation.
                ),  # Finish assembling float inputs.
                "mean_distance_m": float(  # Record 'mean_distance_m' as the average of the entries of row selected by 'distance_m' for each row in the selected next-action return predictions as a floating-point value.
                    np.mean([row["distance_m"] for row in selected])  # Include the average of the entries of row selected by 'distance_m' for each row in the selected next-action return predictions in the surrounding operation.
                ),  # Finish assembling float inputs.
            }  # Finish assembling the grouped values above.
        )  # Finish assembling summary.append inputs.
    provenance = {  # Set provenance to the named settings or measurements assembled below.
        "policy": args.policy,  # Record 'policy' as args.policy.
        "alpha": args.alpha,  # Record 'alpha' as args.alpha.
        "speed_governor_cap_m_s": args.speed_cap,  # Record 'speed_governor_cap_m_s' as args.speed cap.
        "seed_start": args.seed_start,  # Record 'seed_start' as args.seed start.
        "episodes_per_density": args.episodes,  # Record 'episodes_per_density' as args.episodes.
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,  # Record 'checkpoint' as args.checkpoint as a text value when args.checkpoint, otherwise None.
        "checkpoint_sha256": (  # Record 'checkpoint_sha256' as the result of hashlib.sha256(Path(args.checkpoint).read bytes()).hexdigest when args.checkpoint, otherwise None.
            hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()  # Include the result of hashlib.sha256(Path(args.checkpoint).read bytes()).hexdigest when args.checkpoint, otherwise None in the surrounding operation.
            if args.checkpoint  # Include args.checkpoint in the surrounding operation.
            else None  # Include None in the surrounding operation.
        ),  # Finish assembling the grouped values above.
        "training_seed": checkpoint["config"]["seed"] if checkpoint else None,  # Record 'training_seed' as the entries of the entries of checkpoint selected by 'config' selected by 'seed' when checkpoint, otherwise None.
        "training_steps": checkpoint["step"] if checkpoint else None,  # Record 'training_steps' as the entries of checkpoint selected by 'step' when checkpoint, otherwise None.
        "training_collision_penalty": (  # Record 'training_collision_penalty' as checkpoint['config'].get applied to 'collision_penalty', negative 2.0 when checkpoint, otherwise None.
            checkpoint["config"].get("collision_penalty", -2.0) if checkpoint else None  # Include checkpoint['config'].get applied to 'collision_penalty', negative 2.0 when checkpoint, otherwise None in the surrounding operation.
        ),  # Finish assembling the grouped values above.
        "evaluation_collision_penalty": -2.0,  # Record 'evaluation_collision_penalty' as negative 2.0.
        "algorithm": checkpoint["config"]["algorithm"] if checkpoint else args.policy,  # Record 'algorithm' as the entries of the entries of checkpoint selected by 'config' selected by 'algorithm' when checkpoint, otherwise args.policy.
        "source_hashes": source_hashes(),  # Record 'source_hashes' as the result of source hashes.
        "versions": versions(),  # Record 'versions' as the result of versions.
    }  # Finish assembling the grouped values above.
    result = {"provenance": provenance, "summary": summary, "episodes": rows}  # Set result to the named settings or measurements assembled below.
    save_json(args.out, result)  # Save the following results/settings as json: args.out, result.
    print(json.dumps({"provenance": provenance, "summary": summary}, indent=2))  # Print applied to json.dumps applied to the named settings or measurements assembled below.
