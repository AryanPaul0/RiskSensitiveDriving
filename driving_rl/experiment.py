"""Connect the driving simulator to the learning code.

train: collect driving experiences, learn from them, and save the model.
evaluate: use a saved model without changing its learned weights.
benchmark: measure how quickly the computer can run these operations.
A seed is a number that makes random choices repeatable. Training and evaluation
use different seeds, so the final checks do not just repeat training situations.

Original implementation notes:
Train, checkpoint, benchmark, and evaluate on explicitly separated seeds.
"""

# import makes these modules available; "as" gives a module a shorter local name.
import csv
# import makes these modules available; "as" gives a module a shorter local name.
import hashlib
# import makes these modules available; "as" gives a module a shorter local name.
import json
# from pathlib import brings the named tools into this file; parentheses allow several lines.
from pathlib import Path
# import makes these modules available; "as" gives a module a shorter local name.
import platform
# import makes these modules available; "as" gives a module a shorter local name.
import time

# import makes these modules available; "as" gives a module a shorter local name.
import imageio.v2 as imageio
# import makes these modules available; "as" gives a module a shorter local name.
import numpy as np
# import makes these modules available; "as" gives a module a shorter local name.
import torch

# from agents import brings the named tools into this file; parentheses allow several lines.
from .agents import Agent, ReplayBuffer
# from envs import brings the named tools into this file; parentheses allow several lines.
from .envs import (
    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
    flatten,
    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
    govern_action,
    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
    heuristic_action,
    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
    make_env,
    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
    render_frame,
    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
    seed_everything,
    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
    reset_training_episode,
# ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
)


# def defines save_json; parentheses list its inputs, and the colon starts its indented body.
def save_json(path, value):
    """Save dictionaries and lists as readable JSON, creating the destination folder
    if needed. These files hold results and settings, not neural-network calculations.
    """
    # = stores a result in path; call Path using parentheses to supply its inputs.
    path = Path(path)
    # Call path.parent.mkdir using parentheses to supply its inputs; this line does not store a separate return value.
    path.parent.mkdir(parents=True, exist_ok=True)
    # Call path.write_text using parentheses to supply its inputs; this line does not store a separate return value.
    path.write_text(json.dumps(value, indent=2) + "\n")


# def defines versions; parentheses list its inputs, and the colon starts its indented body.
def versions():
    # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
    """Record Python and library versions so an experiment can be reproduced later."""
    # from importlib.metadata import brings the named tools into this file; parentheses allow several lines.
    from importlib.metadata import version

    # return ends this function and sends its result back to the code that called it.
    return {
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "python": platform.python_version(),
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "platform": platform.platform(),
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        **{
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            name: version(name)
            # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
            for name in ["torch", "numpy", "gymnasium", "highway-env"]
        # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
        },
    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
    }


# def defines source_hashes; parentheses list its inputs, and the colon starts its indented body.
def source_hashes():
    """Record fingerprints of the Python source files used by this run.
    A comment-only edit changes a file fingerprint even when calculations are unchanged.
    """
    # return ends this function and sends its result back to the code that called it.
    return {
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
        for path in sorted(Path(__file__).parent.glob("*.py"))
    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
    }


# def defines build_agent; parentheses list its inputs, and the colon starts its indented body.
def build_agent(config):
    # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
    """Construct the learner described by a saved configuration dictionary."""
    # return ends this function and sends its result back to the code that called it.
    return Agent(
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        config["observation_dim"],
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        config["n_actions"],
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        config["algorithm"],
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        config["quantiles"],
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        config["learning_rate"],
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        config["gamma"],
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )


# def defines save_checkpoint; parentheses list its inputs, and the colon starts its indented body.
def save_checkpoint(path, agent, config, step):
    """Save learned weights, target weights, optimizer state, settings and step count.
    Write a temporary file first, then rename it, to avoid leaving a partly saved
    checkpoint. Replay memory and random-generator state are not saved for exact resume.
    """
    # = stores a result in temporary; call Path using parentheses to supply its inputs.
    temporary = Path(str(path) + ".tmp")
    # Call torch.save using parentheses to supply its inputs; this line does not store a separate return value.
    torch.save(
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        {
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "config": config,
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "step": step,
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "online": agent.online.state_dict(),
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "target": agent.target.state_dict(),
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "optimizer": agent.optimizer.state_dict(),
        # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
        },
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        temporary,
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # Call temporary.replace using parentheses to supply its inputs; this line does not store a separate return value.
    temporary.replace(path)


# def defines load_agent; parentheses list its inputs, and the colon starts its indented body.
def load_agent(path):
    """Load a saved network for inference: making decisions without learning.
    Evaluation reads the online network; it does not update its weights.
    """
    # = stores a result in checkpoint; call torch.load using parentheses to supply its inputs.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    # = stores a result in agent; call build_agent using parentheses to supply its inputs.
    agent = build_agent(checkpoint["config"])
    # Call agent.online.load_state_dict using parentheses to supply its inputs; this line does not store a separate return value.
    agent.online.load_state_dict(checkpoint["online"])
    # Call agent.online.eval using parentheses to supply its inputs; this line does not store a separate return value.
    agent.online.eval()
    # return ends this function and sends its result back to the code that called it.
    return agent, checkpoint


# def defines exploration_epsilon; parentheses list its inputs, and the colon starts its indented body.
def exploration_epsilon(step, total_steps, fraction=0.6, final=0.05, decay_steps=None):
    """Calculate the chance of taking a random action at this training step.
    Start at 1 (always random), decrease linearly, and stop at the final chance.
    An absolute decay horizon keeps exploration comparable across training budgets.

    Original implementation notes:
    Optional absolute decay horizon keeps exploration fixed across budgets.
    """
    # = stores a result in horizon; choose the value before "if" when its condition is true, otherwise the value after "else".
    horizon = decay_steps if decay_steps is not None else total_steps * fraction
    # = stores a result in progress; call min using parentheses to supply its inputs.
    progress = min(1.0, max(0.0, (step - 1) / max(1, horizon)))
    # return ends this function and sends its result back to the code that called it.
    return 1.0 + progress * (final - 1.0)


# def defines train; parentheses list its inputs, and the colon starts its indented body.
def train(args):
    """Train one agent through repeated driving steps.
    1. Observe traffic and choose either a random or a learned action.
    2. Store the resulting transition in replay memory.
    3. Sample earlier transitions and update the network.
    4. Occasionally copy weights to the target network and save checkpoints.
    5. Start another episode after a crash or recording cutoff.
    Episode counters describe data collection; learning happens from replay batches.
    """
    # = stores a result in output; call Path using parentheses to supply its inputs.
    output = Path(args.out)
    # if checks output.exists() and any(output.iterdir()); only a true condition runs the indented block.
    if output.exists() and any(output.iterdir()):
        # raise signals an error and stops this normal path, so invalid data are not silently accepted.
        raise ValueError(f"Refusing to overwrite existing experiment: {output}")
    # Call output.mkdir using parentheses to supply its inputs; this line does not store a separate return value.
    output.mkdir(parents=True, exist_ok=True)
    # Call seed_everything using parentheses to supply its inputs; this line does not store a separate return value.
    seed_everything(args.seed)
    # = stores a result in rng; call np.random.default_rng using parentheses to supply its inputs.
    rng = np.random.default_rng(args.seed)
    # = stores a result in env; call make_env using parentheses to supply its inputs.
    env = make_env(collision_penalty=args.collision_penalty)
    # = stores a result in densities; evaluate the expression on the right before using its result.
    densities = getattr(args, "training_densities", None) or [1.0]
    # Keep density sampling separate from exploration/replay randomness.
    # = stores a result in density_rng; call np.random.default_rng using parentheses to supply its inputs.
    density_rng = np.random.default_rng(np.random.SeedSequence([args.seed, 8675309]))
    # = unpacks the right-hand result into (observation, _, episode_density), matching values to names in order.
    observation, _, episode_density = reset_training_episode(
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        env, density_rng, densities, args.seed
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # = stores a result in density_steps; collect or generate one result per item; the "for" part names the item being processed.
    density_steps = {str(float(d)): 0 for d in densities}
    # = stores a result in density_episodes; collect or generate one result per item; the "for" part names the item being processed.
    density_episodes = {str(float(d)): 0 for d in densities}
    # Call env.action_space.seed using parentheses to supply its inputs; this line does not store a separate return value.
    env.action_space.seed(args.seed)
    # = stores a result in observation; call flatten using parentheses to supply its inputs.
    observation = flatten(observation)
    # = stores a result in config; build a dictionary, which stores values under named keys.
    config = {
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        **vars(args),
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "observation_dim": len(observation),
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "n_actions": int(env.action_space.n),
    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
    }
    # Call config.pop using parentheses to supply its inputs; this line does not store a separate return value.
    config.pop("func", None)
    # Call save_json using parentheses to supply its inputs; this line does not store a separate return value.
    save_json(output / "config.json", config)
    # Call save_json using parentheses to supply its inputs; this line does not store a separate return value.
    save_json(output / "versions.json", versions())
    # Call save_json using parentheses to supply its inputs; this line does not store a separate return value.
    save_json(output / "environment.json", env.unwrapped.config)
    # Call save_json using parentheses to supply its inputs; this line does not store a separate return value.
    save_json(output / "source_hashes.json", source_hashes())
    # = stores a result in agent; call build_agent using parentheses to supply its inputs.
    agent = build_agent(config)
    # Learning reuses this finite memory rather than relying only on the latest step.
    # = stores a result in replay; call ReplayBuffer using parentheses to supply its inputs.
    replay = ReplayBuffer(args.buffer_size)
    # = stores a result in episode_return; use the literal value 0.0.
    episode_return = 0.0
    # = stores a result in episode_length; use the literal value 0.
    episode_length = 0
    # = stores a result in episode; use the literal value 0.
    episode = 0
    # = stores a result in started; call time.perf_counter using parentheses to supply its inputs.
    started = time.perf_counter()
    # = stores a result in last_loss, last_grad; call float using parentheses to supply its inputs.
    last_loss = last_grad = float("nan")
    # with enters a temporary context and exits it afterward, even if an error interrupts the block.
    with (output / "training.csv").open("w", newline="") as file:
        # = stores a result in fields; build a list, keeping its items in order.
        fields = [
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "step",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "episode",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "return",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "length",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "crashed",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "epsilon",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "loss",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "grad_norm",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "elapsed_seconds",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "steps_per_second",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "density",
        # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
        ]
        # = stores a result in writer; call csv.DictWriter using parentheses to supply its inputs.
        writer = csv.DictWriter(file, fieldnames=fields)
        # Call writer.writeheader using parentheses to supply its inputs; this line does not store a separate return value.
        writer.writeheader()
        # try starts a block whose errors or cleanup are handled by the following except/finally blocks.
        try:
            # The training budget counts simulator interactions, not episodes.
            # for takes each item from range(1, args.steps + 1) in turn; step holds the current item.
            for step in range(1, args.steps + 1):
                # Epsilon is the probability of exploring with a random action.
                # = stores a result in epsilon; call exploration_epsilon using parentheses to supply its inputs.
                epsilon = exploration_epsilon(
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    step,
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    args.steps,
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    args.exploration_fraction,
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    args.final_epsilon,
                    # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                    getattr(args, "exploration_steps", None),
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                )
                # Exploration tries alternatives; exploitation uses the best current prediction.
                # = stores a result in action; choose the value before "if" when its condition is true, otherwise the value after "else".
                action = (
                    # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                    int(rng.integers(env.action_space.n))
                    # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                    if rng.random() < epsilon
                    # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                    else agent.act(observation)
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                )
                # terminated means the task truly ended; truncated means its recording limit was reached.
                # = unpacks the right-hand result into (next_observation, reward, terminated, truncated, info), matching values to names in order.
                next_observation, reward, terminated, truncated, info = env.step(action)
                # = stores a result in next_observation; call flatten using parentheses to supply its inputs.
                next_observation = flatten(next_observation)
                # The combined operator updates density_steps[str(episode_density)] using its previous value.
                density_steps[str(episode_density)] += 1
                # Save the final next observation before any reset, so cutoff bootstrapping uses the right state.
                # Call replay.add using parentheses to supply its inputs; this line does not store a separate return value.
                replay.add(observation, action, reward, next_observation, terminated)
                # = stores a result in observation; evaluate the expression on the right before using its result.
                observation = next_observation
                # The combined operator updates episode_return using its previous value.
                episode_return += float(reward)
                # The combined operator updates episode_length using its previous value.
                episode_length += 1
                # Wait for enough experience before sampling mini-batches for learning.
                # if checks step >= args.learning_starts and len(replay) >= args.batch_size; only a true condition runs the indented block.
                if step >= args.learning_starts and len(replay) >= args.batch_size:
                    # if checks step % args.train_frequency == 0; only a true condition runs the indented block.
                    if step % args.train_frequency == 0:
                        # = unpacks the right-hand result into (last_loss, last_grad), matching values to names in order.
                        last_loss, last_grad = agent.update(
                            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                            replay.sample(args.batch_size, rng)
                        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                        )
                # Refresh the target only on its schedule, rather than moving both networks at every update.
                # if checks step % args.target_interval == 0; only a true condition runs the indented block.
                if step % args.target_interval == 0:
                    # Call agent.sync_target using parentheses to supply its inputs; this line does not store a separate return value.
                    agent.sync_target()
                # Both conditions start a fresh episode, but only true termination removes the bootstrap term.
                # if checks terminated or truncated; only a true condition runs the indented block.
                if terminated or truncated:
                    # The combined operator updates episode using its previous value.
                    episode += 1
                    # The combined operator updates density_episodes[str(episode_density)] using its previous value.
                    density_episodes[str(episode_density)] += 1
                    # = stores a result in elapsed; combine the operands with the displayed operator; parentheses control grouping.
                    elapsed = time.perf_counter() - started
                    # Write episode statistics immediately so another Terminal can show progress during training.
                    # Call writer.writerow using parentheses to supply its inputs; this line does not store a separate return value.
                    writer.writerow(
                        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                        dict(
                            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                            zip(
                                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                                fields,
                                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                                [
                                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                                    step,
                                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                                    episode,
                                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                                    episode_return,
                                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                                    episode_length,
                                    # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                                    int(
                                        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                                        info.get(
                                            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                                            "crashed", env.unwrapped.vehicle.crashed
                                        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                                        )
                                    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                                    ),
                                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                                    epsilon,
                                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                                    last_loss,
                                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                                    last_grad,
                                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                                    elapsed,
                                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                                    step / elapsed,
                                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                                    episode_density,
                                # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
                                ],
                            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                            )
                        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                        )
                    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                    )
                    # Call file.flush using parentheses to supply its inputs; this line does not store a separate return value.
                    file.flush()
                    # = unpacks the right-hand result into (observation, _, episode_density), matching values to names in order.
                    observation, _, episode_density = reset_training_episode(
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        env, density_rng, densities
                    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                    )
                    # = stores a result in observation; call flatten using parentheses to supply its inputs.
                    observation = flatten(observation)
                    # = stores a result in episode_return; use the literal value 0.0.
                    episode_return = 0.0
                    # = stores a result in episode_length; use the literal value 0.
                    episode_length = 0
                # if checks step % args.checkpoint_interval == 0; only a true condition runs the indented block.
                if step % args.checkpoint_interval == 0:
                    # Call save_checkpoint using parentheses to supply its inputs; this line does not store a separate return value.
                    save_checkpoint(
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        output / f"checkpoint_{step}.pt", agent, config, step
                    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                    )
                    # = stores a result in elapsed; combine the operands with the displayed operator; parentheses control grouping.
                    elapsed = time.perf_counter() - started
                    # Call print using parentheses to supply its inputs; this line does not store a separate return value.
                    print(
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        f"{args.algorithm} step={step}/{args.steps} episodes={episode} "
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        f"loss={last_loss:.4f} steps/s={step / elapsed:.1f}",
                        # Inside this continued call, name=value supplies an option by name rather than by its position.
                        flush=True,
                    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                    )
            # Save the requested final model, not a model chosen after inspecting test results.
            # Call save_checkpoint using parentheses to supply its inputs; this line does not store a separate return value.
            save_checkpoint(output / "final.pt", agent, config, args.steps)
            # Call save_json using parentheses to supply its inputs; this line does not store a separate return value.
            save_json(
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                output / "summary.json",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                {
                    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                    "steps": args.steps,
                    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                    "episodes": episode,
                    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                    "elapsed_seconds": time.perf_counter() - started,
                    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                    "training_complete": True,
                    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                    "completed_episodes_by_density": density_episodes,
                    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                    "steps_by_density": density_steps,
                # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
                },
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            )
        # except catches the specified kind of error and runs this indented recovery block.
        except KeyboardInterrupt:
            # Call save_checkpoint using parentheses to supply its inputs; this line does not store a separate return value.
            save_checkpoint(output / "interrupted.pt", agent, config, step)
            # Call print using parentheses to supply its inputs; this line does not store a separate return value.
            print(
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Interrupted: saved model and optimizer; replay/RNG resume is not supported.",
                # Inside this continued call, name=value supplies an option by name rather than by its position.
                flush=True,
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            )
        # finally always runs this cleanup block when the try block ends, including after an error or return.
        finally:
            # Call env.close using parentheses to supply its inputs; this line does not store a separate return value.
            env.close()


# def defines benchmark; parentheses list its inputs, and the colon starts its indented body.
def benchmark(args):
    """Time simulator steps and network updates separately.
    This is a rough speed estimate, not evidence that a policy drives well.
    """
    # Call seed_everything using parentheses to supply its inputs; this line does not store a separate return value.
    seed_everything(args.seed)
    # = stores a result in env; call make_env using parentheses to supply its inputs.
    env = make_env()
    # = unpacks the right-hand result into (observation, _), matching values to names in order.
    observation, _ = env.reset(seed=args.seed)
    # Call env.action_space.seed using parentheses to supply its inputs; this line does not store a separate return value.
    env.action_space.seed(args.seed)
    # = stores a result in started; call time.perf_counter using parentheses to supply its inputs.
    started = time.perf_counter()
    # for takes each item from range(args.steps) in turn; _ holds the current item.
    for _ in range(args.steps):
        # = unpacks the right-hand result into (observation, _, terminated, truncated, _), matching values to names in order.
        observation, _, terminated, truncated, _ = env.step(env.action_space.sample())
        # Both conditions start a fresh episode, but only true termination removes the bootstrap term.
        # if checks terminated or truncated; only a true condition runs the indented block.
        if terminated or truncated:
            # = unpacks the right-hand result into (observation, _), matching values to names in order.
            observation, _ = env.reset()
    # = stores a result in elapsed; combine the operands with the displayed operator; parentheses control grouping.
    elapsed = time.perf_counter() - started
    # Includes an actual optimizer update, but excludes rendering and disk I/O.
    # = stores a result in agent; call Agent using parentheses to supply its inputs.
    agent = Agent(len(flatten(observation)), env.action_space.n, "qrdqn")
    # = stores a result in rng; call np.random.default_rng using parentheses to supply its inputs.
    rng = np.random.default_rng(args.seed)
    # = stores a result in synthetic_batch; group these values into a tuple, keeping their order.
    synthetic_batch = (
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        torch.from_numpy(
            # Inside this continued call, name=value supplies an option by name rather than by its position.
            rng.normal(size=(64, len(flatten(observation)))).astype(np.float32)
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        ),
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        torch.from_numpy(rng.integers(env.action_space.n, size=64)),
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        torch.zeros(64),
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        torch.zeros(64, len(flatten(observation))),
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        torch.zeros(64),
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # Call agent.update using parentheses to supply its inputs; this line does not store a separate return value.
    agent.update(synthetic_batch)
    # = stores a result in update_start; call time.perf_counter using parentheses to supply its inputs.
    update_start = time.perf_counter()
    # for takes each item from range(200) in turn; _ holds the current item.
    for _ in range(200):
        # Call agent.update using parentheses to supply its inputs; this line does not store a separate return value.
        agent.update(synthetic_batch)
    # = stores a result in update_seconds; apply / to these operands; numbers are divided, while Path objects join folder names.
    update_seconds = (time.perf_counter() - update_start) / 200
    # = stores a result in result; build a dictionary, which stores values under named keys.
    result = {
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "environment_steps": args.steps,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "environment_seconds": elapsed,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "environment_steps_per_second": args.steps / elapsed,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "qrdqn_update_seconds_batch64": update_seconds,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "estimated_20000_step_run_seconds": 20000
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        * (elapsed / args.steps + update_seconds),
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "estimate_note": "Rough estimate; omits replay sampling, inference, logging and checkpointing.",
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "versions": versions(),
    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
    }
    # Call save_json using parentheses to supply its inputs; this line does not store a separate return value.
    save_json(args.out, result)
    # Call env.close using parentheses to supply its inputs; this line does not store a separate return value.
    env.close()
    # Call print using parentheses to supply its inputs; this line does not store a separate return value.
    print(json.dumps(result, indent=2))


# def defines evaluate; parentheses list its inputs, and the colon starts its indented body.
def evaluate(args):
    """Measure a frozen policy on specified traffic seeds and densities.
    No optimizer updates happen here. Save outcomes for every episode so policies
    can be compared on the same scenarios later. A GIF records one predefined case.
    """
    # Call seed_everything using parentheses to supply its inputs; this line does not store a separate return value.
    seed_everything(0)
    # if checks args.seed_start < 10000; only a true condition runs the indented block.
    if args.seed_start < 10000:
        # raise signals an error and stops this normal path, so invalid data are not silently accepted.
        raise ValueError(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "Use evaluation seeds >= 10000; reserve lower seeds for development."
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
    # if checks Path(args.out).exists(); only a true condition runs the indented block.
    if Path(args.out).exists():
        # raise signals an error and stops this normal path, so invalid data are not silently accepted.
        raise ValueError(f"Refusing to overwrite evaluation: {args.out}")
    # = stores a result in agent, checkpoint; use the literal value None.
    agent = checkpoint = None
    # if checks args.checkpoint; only a true condition runs the indented block.
    if args.checkpoint:
        # = unpacks the right-hand result into (agent, checkpoint), matching values to names in order.
        agent, checkpoint = load_agent(args.checkpoint)
    # if checks args.policy == 'learned' and agent is None; only a true condition runs the indented block.
    if args.policy == "learned" and agent is None:
        # raise signals an error and stops this normal path, so invalid data are not silently accepted.
        raise ValueError("A learned policy requires --checkpoint")
    # if checks agent is not None and agent.algorithm == 'dqn' and (args.alpha != 1); only a true condition runs the indented block.
    if agent is not None and agent.algorithm == "dqn" and args.alpha != 1:
        # raise signals an error and stops this normal path, so invalid data are not silently accepted.
        raise ValueError("Lower-tail selection requires a distributional checkpoint")
    # = stores a result in rows; build a list, keeping its items in order.
    rows = []
    # Evaluate the same policy in each specified traffic setting.
    # for takes each item from args.densities in turn; density holds the current item.
    for density in args.densities:
        # = stores a result in env; call make_env using parentheses to supply its inputs.
        env = make_env(density=density, render=bool(args.gif))
        # for takes each item from range(args.episodes) in turn; index holds the current item.
        for index in range(args.episodes):
            # Identical scenario seeds let different policies start from matching traffic situations.
            # = stores a result in episode_seed; combine the operands with the displayed operator; parentheses control grouping.
            episode_seed = args.seed_start + index
            # = unpacks the right-hand result into (observation, _), matching values to names in order.
            observation, _ = env.reset(seed=episode_seed)
            # = stores a result in action_rng; call np.random.default_rng using parentheses to supply its inputs.
            action_rng = np.random.default_rng(episode_seed)
            # = stores a result in total_return, total_speed, distance; use the literal value 0.0.
            total_return = total_speed = distance = 0.0
            # = stores a result in length; use the literal value 0.
            length = 0
            # = stores a result in action_counts; multiply the operands; NumPy and PyTorch arrays do this element by element.
            action_counts = [0] * int(env.action_space.n)
            # Render only the first episode at the first density.
            # Only the predefined first case becomes a replay; do not select the best-looking outcome.
            # = stores a result in record; evaluate the expression on the right before using its result.
            record = bool(args.gif) and index == 0 and density == args.densities[0]
            # = stores a result in frames; choose the value before "if" when its condition is true, otherwise the value after "else".
            frames = [render_frame(env)] if record else []
            # while repeats the indented block as long as True is true.
            while True:
                # Baselines and learned agents share the same simulator and reward measurements.
                # if checks args.policy == 'random'; only a true condition runs the indented block.
                if args.policy == "random":
                    # = stores a result in action; call int using parentheses to supply its inputs.
                    action = int(action_rng.integers(env.action_space.n))
                # if checks args.policy == 'heuristic'; only a true condition runs the indented block.
                elif args.policy == "heuristic":
                    # = stores a result in action; call heuristic_action using parentheses to supply its inputs.
                    action = heuristic_action(env)
                # else provides the alternative branch; after a loop it instead means the loop finished without break.
                else:
                    # = stores a result in action; call agent.act using parentheses to supply its inputs.
                    action = agent.act(flatten(observation), args.alpha)
                    # if checks args.speed_cap is not None; only a true condition runs the indented block.
                    if args.speed_cap is not None:
                        # = stores a result in action; call govern_action using parentheses to supply its inputs.
                        action = govern_action(env, observation, action, args.speed_cap)
                # The combined operator updates action_counts[action] using its previous value.
                action_counts[action] += 1
                # = stores a result in before; call env.unwrapped.vehicle.position.copy using parentheses to supply its inputs.
                before = env.unwrapped.vehicle.position.copy()
                # = unpacks the right-hand result into (observation, reward, terminated, truncated, info), matching values to names in order.
                observation, reward, terminated, truncated, info = env.step(action)
                # The combined operator updates total_return using its previous value.
                total_return += float(reward)
                # The combined operator updates total_speed using its previous value.
                total_speed += float(env.unwrapped.vehicle.speed)
                # The combined operator updates distance using its previous value.
                distance += float(
                    # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                    np.linalg.norm(env.unwrapped.vehicle.position - before)
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                )
                # The combined operator updates length using its previous value.
                length += 1
                # if checks record; only a true condition runs the indented block.
                if record:
                    # Call frames.append using parentheses to supply its inputs; this line does not store a separate return value.
                    frames.append(render_frame(env))
                # Both conditions start a fresh episode, but only true termination removes the bootstrap term.
                # if checks terminated or truncated; only a true condition runs the indented block.
                if terminated or truncated:
                    # Call rows.append using parentheses to supply its inputs; this line does not store a separate return value.
                    rows.append(
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        {
                            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                            "density": density,
                            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                            "seed": episode_seed,
                            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                            "return": total_return,
                            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                            "length": length,
                            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                            "crashed": int(env.unwrapped.vehicle.crashed),
                            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                            "mean_speed_m_s": total_speed / length,
                            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                            "distance_m": distance,
                            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                            "action_counts": action_counts,
                            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                            "terminated": bool(terminated),
                            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                            "truncated": bool(truncated),
                        # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
                        }
                    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                    )
                    # break leaves the nearest loop immediately; execution continues after that loop.
                    break
            # if checks record; only a true condition runs the indented block.
            if record:
                # = stores a result in path; call Path using parentheses to supply its inputs.
                path = Path(args.gif)
                # Call path.parent.mkdir using parentheses to supply its inputs; this line does not store a separate return value.
                path.parent.mkdir(parents=True, exist_ok=True)
                # Each policy decision represents one simulated second.
                # Call imageio.mimsave using parentheses to supply its inputs; this line does not store a separate return value.
                imageio.mimsave(path, frames, duration=1000, loop=0)
        # Call env.close using parentheses to supply its inputs; this line does not store a separate return value.
        env.close()
    # = stores a result in summary; build a list, keeping its items in order.
    summary = []
    # Evaluate the same policy in each specified traffic setting.
    # for takes each item from args.densities in turn; density holds the current item.
    for density in args.densities:
        # = stores a result in selected; collect or generate one result per item; the "for" part names the item being processed.
        selected = [row for row in rows if row["density"] == density]
        # = stores a result in returns; call np.array using parentheses to supply its inputs.
        returns = np.array([row["return"] for row in selected])
        # = stores a result in count; call max using parentheses to supply its inputs.
        count = max(1, int(np.ceil(0.1 * len(returns))))
        # Call summary.append using parentheses to supply its inputs; this line does not store a separate return value.
        summary.append(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            {
                # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                "density": density,
                # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                "episodes": len(selected),
                # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                "crash_rate": float(np.mean([row["crashed"] for row in selected])),
                # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                "mean_return": float(returns.mean()),
                # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                "return_standard_error": (
                    # Inside this continued call, name=value supplies an option by name rather than by its position.
                    float(returns.std(ddof=1) / np.sqrt(len(returns)))
                    # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                    if len(returns) > 1
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    else None
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                ),
                # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                "worst_10_percent_mean_return": float(np.sort(returns)[:count].mean()),
                # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                "mean_speed_m_s": float(
                    # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                    np.mean([row["mean_speed_m_s"] for row in selected])
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                ),
                # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                "mean_distance_m": float(
                    # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                    np.mean([row["distance_m"] for row in selected])
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                ),
            # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
            }
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
    # = stores a result in provenance; build a dictionary, which stores values under named keys.
    provenance = {
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "policy": args.policy,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "alpha": args.alpha,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "speed_governor_cap_m_s": args.speed_cap,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "seed_start": args.seed_start,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "episodes_per_density": args.episodes,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "checkpoint_sha256": (
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
            # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
            if args.checkpoint
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            else None
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        ),
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "training_seed": checkpoint["config"]["seed"] if checkpoint else None,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "training_steps": checkpoint["step"] if checkpoint else None,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "training_collision_penalty": (
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            checkpoint["config"].get("collision_penalty", -2.0) if checkpoint else None
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        ),
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "evaluation_collision_penalty": -2.0,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "algorithm": checkpoint["config"]["algorithm"] if checkpoint else args.policy,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "source_hashes": source_hashes(),
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "versions": versions(),
    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
    }
    # = stores a result in result; build a dictionary, which stores values under named keys.
    result = {"provenance": provenance, "summary": summary, "episodes": rows}
    # Call save_json using parentheses to supply its inputs; this line does not store a separate return value.
    save_json(args.out, result)
    # Call print using parentheses to supply its inputs; this line does not store a separate return value.
    print(json.dumps({"provenance": provenance, "summary": summary}, indent=2))
