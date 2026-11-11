"""Train, checkpoint, benchmark, and evaluate on explicitly separated seeds."""
import csv
import hashlib
import json
from pathlib import Path
import platform
import time

import imageio.v2 as imageio
import numpy as np
import torch

from .agents import Agent, ReplayBuffer
from .envs import flatten, govern_action, heuristic_action, make_env, render_frame, seed_everything, reset_training_episode


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def versions():
    from importlib.metadata import version
    return {"python": platform.python_version(), "platform": platform.platform(),
            **{name: version(name) for name in
               ["torch", "numpy", "gymnasium", "highway-env"]}}


def source_hashes():
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.glob("*.py"))}


def build_agent(config):
    return Agent(config["observation_dim"], config["n_actions"],
                 config["algorithm"], config["quantiles"],
                 config["learning_rate"], config["gamma"])


def save_checkpoint(path, agent, config, step):
    temporary = Path(str(path) + ".tmp")
    torch.save({"config": config, "step": step,
                "online": agent.online.state_dict(),
                "target": agent.target.state_dict(),
                "optimizer": agent.optimizer.state_dict()}, temporary)
    temporary.replace(path)


def load_agent(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    agent = build_agent(checkpoint["config"])
    agent.online.load_state_dict(checkpoint["online"])
    agent.online.eval()
    return agent, checkpoint


def exploration_epsilon(step, total_steps, fraction=.6, final=.05, decay_steps=None):
    """Optional absolute decay horizon keeps exploration fixed across budgets."""
    horizon = decay_steps if decay_steps is not None else total_steps * fraction
    progress = min(1.0, max(0., (step - 1) / max(1, horizon)))
    return 1.0 + progress * (final - 1.0)


def train(args):
    output = Path(args.out)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Refusing to overwrite existing experiment: {output}")
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    env = make_env(collision_penalty=args.collision_penalty)
    densities = getattr(args, "training_densities", None) or [1.]
    density_rng = np.random.default_rng(np.random.SeedSequence([args.seed, 8675309]))
    observation, _, episode_density = reset_training_episode(env, density_rng, densities, args.seed)
    density_steps = {str(float(d)): 0 for d in densities}
    density_episodes = {str(float(d)): 0 for d in densities}
    env.action_space.seed(args.seed)
    observation = flatten(observation)
    config = {**vars(args), "observation_dim": len(observation),
              "n_actions": int(env.action_space.n)}
    config.pop("func", None)
    save_json(output / "config.json", config)
    save_json(output / "versions.json", versions())
    save_json(output / "environment.json", env.unwrapped.config)
    save_json(output / "source_hashes.json", source_hashes())
    agent = build_agent(config)
    replay = ReplayBuffer(args.buffer_size)
    episode_return = 0.0
    episode_length = 0
    episode = 0
    started = time.perf_counter()
    last_loss = last_grad = float("nan")
    with (output / "training.csv").open("w", newline="") as file:
        fields = ["step", "episode", "return", "length", "crashed", "epsilon",
                  "loss", "grad_norm", "elapsed_seconds", "steps_per_second", "density"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        try:
            for step in range(1, args.steps + 1):
                epsilon = exploration_epsilon(step, args.steps, args.exploration_fraction,
                                              args.final_epsilon, getattr(args, "exploration_steps", None))
                action = (int(rng.integers(env.action_space.n)) if rng.random() < epsilon
                          else agent.act(observation))
                next_observation, reward, terminated, truncated, info = env.step(action)
                next_observation = flatten(next_observation)
                density_steps[str(episode_density)] += 1
                replay.add(observation, action, reward, next_observation, terminated)
                observation = next_observation
                episode_return += float(reward)
                episode_length += 1
                if step >= args.learning_starts and len(replay) >= args.batch_size:
                    if step % args.train_frequency == 0:
                        last_loss, last_grad = agent.update(replay.sample(args.batch_size, rng))
                if step % args.target_interval == 0:
                    agent.sync_target()
                if terminated or truncated:
                    episode += 1
                    density_episodes[str(episode_density)] += 1
                    elapsed = time.perf_counter() - started
                    writer.writerow(dict(zip(fields, [
                        step, episode, episode_return, episode_length,
                        int(info.get("crashed", env.unwrapped.vehicle.crashed)),
                        epsilon, last_loss, last_grad, elapsed, step / elapsed, episode_density,
                    ])))
                    file.flush()
                    observation, _, episode_density = reset_training_episode(env, density_rng, densities)
                    observation = flatten(observation)
                    episode_return = 0.0
                    episode_length = 0
                if step % args.checkpoint_interval == 0:
                    save_checkpoint(output / f"checkpoint_{step}.pt", agent, config, step)
                    elapsed = time.perf_counter() - started
                    print(f"{args.algorithm} step={step}/{args.steps} episodes={episode} "
                          f"loss={last_loss:.4f} steps/s={step / elapsed:.1f}", flush=True)
            save_checkpoint(output / "final.pt", agent, config, args.steps)
            save_json(output / "summary.json", {
                "steps": args.steps, "episodes": episode,
                "elapsed_seconds": time.perf_counter() - started,
                "training_complete": True,
                "completed_episodes_by_density": density_episodes,
                "steps_by_density": density_steps,
            })
        except KeyboardInterrupt:
            save_checkpoint(output / "interrupted.pt", agent, config, step)
            print("Interrupted: saved model and optimizer; replay/RNG resume is not supported.", flush=True)
        finally:
            env.close()


def benchmark(args):
    seed_everything(args.seed)
    env = make_env()
    observation, _ = env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    started = time.perf_counter()
    for _ in range(args.steps):
        observation, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            observation, _ = env.reset()
    elapsed = time.perf_counter() - started
    # Includes an actual optimizer update, but excludes rendering and disk I/O.
    agent = Agent(len(flatten(observation)), env.action_space.n, "qrdqn")
    rng = np.random.default_rng(args.seed)
    synthetic_batch = (
        torch.from_numpy(rng.normal(size=(64, len(flatten(observation)))).astype(np.float32)),
        torch.from_numpy(rng.integers(env.action_space.n, size=64)),
        torch.zeros(64), torch.zeros(64, len(flatten(observation))), torch.zeros(64),
    )
    agent.update(synthetic_batch)
    update_start = time.perf_counter()
    for _ in range(200):
        agent.update(synthetic_batch)
    update_seconds = (time.perf_counter() - update_start) / 200
    result = {"environment_steps": args.steps, "environment_seconds": elapsed,
              "environment_steps_per_second": args.steps / elapsed,
              "qrdqn_update_seconds_batch64": update_seconds,
              "estimated_20000_step_run_seconds": 20000 * (elapsed / args.steps + update_seconds),
              "estimate_note": "Rough estimate; omits replay sampling, inference, logging and checkpointing.",
              "versions": versions()}
    save_json(args.out, result)
    env.close()
    print(json.dumps(result, indent=2))


def evaluate(args):
    seed_everything(0)
    if args.seed_start < 10000:
        raise ValueError("Use evaluation seeds >= 10000; reserve lower seeds for development.")
    if Path(args.out).exists():
        raise ValueError(f"Refusing to overwrite evaluation: {args.out}")
    agent = checkpoint = None
    if args.checkpoint:
        agent, checkpoint = load_agent(args.checkpoint)
    if args.policy == "learned" and agent is None:
        raise ValueError("A learned policy requires --checkpoint")
    if agent is not None and agent.algorithm == "dqn" and args.alpha != 1:
        raise ValueError("Lower-tail selection requires a distributional checkpoint")
    rows = []
    for density in args.densities:
        env = make_env(density=density, render=bool(args.gif))
        for index in range(args.episodes):
            episode_seed = args.seed_start + index
            observation, _ = env.reset(seed=episode_seed)
            action_rng = np.random.default_rng(episode_seed)
            total_return = total_speed = distance = 0.0
            length = 0
            action_counts = [0] * int(env.action_space.n)
            # Render only the first episode at the first density.
            record = bool(args.gif) and index == 0 and density == args.densities[0]
            frames = [render_frame(env)] if record else []
            while True:
                if args.policy == "random":
                    action = int(action_rng.integers(env.action_space.n))
                elif args.policy == "heuristic":
                    action = heuristic_action(env)
                else:
                    action = agent.act(flatten(observation), args.alpha)
                    if args.speed_cap is not None:
                        action = govern_action(env, observation, action, args.speed_cap)
                action_counts[action] += 1
                before = env.unwrapped.vehicle.position.copy()
                observation, reward, terminated, truncated, info = env.step(action)
                total_return += float(reward)
                total_speed += float(env.unwrapped.vehicle.speed)
                distance += float(np.linalg.norm(env.unwrapped.vehicle.position - before))
                length += 1
                if record:
                    frames.append(render_frame(env))
                if terminated or truncated:
                    rows.append({"density": density, "seed": episode_seed,
                                 "return": total_return, "length": length,
                                 "crashed": int(env.unwrapped.vehicle.crashed),
                                 "mean_speed_m_s": total_speed / length,
                                 "distance_m": distance,
                                 "action_counts": action_counts,
                                 "terminated": bool(terminated), "truncated": bool(truncated)})
                    break
            if record:
                path = Path(args.gif)
                path.parent.mkdir(parents=True, exist_ok=True)
                # Each policy decision represents one simulated second.
                imageio.mimsave(path, frames, duration=1000, loop=0)
        env.close()
    summary = []
    for density in args.densities:
        selected = [row for row in rows if row["density"] == density]
        returns = np.array([row["return"] for row in selected])
        count = max(1, int(np.ceil(0.1 * len(returns))))
        summary.append({
            "density": density, "episodes": len(selected),
            "crash_rate": float(np.mean([row["crashed"] for row in selected])),
            "mean_return": float(returns.mean()),
            "return_standard_error": float(returns.std(ddof=1) / np.sqrt(len(returns))) if len(returns) > 1 else None,
            "worst_10_percent_mean_return": float(np.sort(returns)[:count].mean()),
            "mean_speed_m_s": float(np.mean([row["mean_speed_m_s"] for row in selected])),
            "mean_distance_m": float(np.mean([row["distance_m"] for row in selected])),
        })
    provenance = {"policy": args.policy, "alpha": args.alpha,
                  "speed_governor_cap_m_s": args.speed_cap,
                  "seed_start": args.seed_start, "episodes_per_density": args.episodes,
                  "checkpoint": str(args.checkpoint) if args.checkpoint else None,
                  "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest() if args.checkpoint else None,
                  "training_seed": checkpoint["config"]["seed"] if checkpoint else None,
                  "training_steps": checkpoint["step"] if checkpoint else None,
                  "training_collision_penalty": checkpoint["config"].get("collision_penalty", -2.0) if checkpoint else None,
                  "evaluation_collision_penalty": -2.0,
                  "algorithm": checkpoint["config"]["algorithm"] if checkpoint else args.policy,
                  "source_hashes": source_hashes(),
                  "versions": versions()}
    result = {"provenance": provenance, "summary": summary, "episodes": rows}
    save_json(args.out, result)
    print(json.dumps({"provenance": provenance, "summary": summary}, indent=2))
