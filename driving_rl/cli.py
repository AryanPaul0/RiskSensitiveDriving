"""Command-line entry points; defaults are intentionally modest."""
import argparse
import math

from .experiment import benchmark, evaluate, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    bench = sub.add_parser("benchmark")
    bench.add_argument("--steps", type=int, default=2000)
    bench.add_argument("--seed", type=int, default=0)
    bench.add_argument("--out", default="results/benchmark.json")
    bench.set_defaults(func=benchmark)
    training = sub.add_parser("train")
    training.add_argument("--algorithm", choices=["dqn", "qrdqn"], default="dqn")
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--steps", type=int, default=20000)
    training.add_argument("--out", required=True)
    training.add_argument("--quantiles", type=int, default=32)
    training.add_argument("--learning-rate", type=float, default=5e-4)
    training.add_argument("--gamma", type=float, default=0.9)
    training.add_argument("--collision-penalty", type=float, default=-2.0,
                          help="Training reward only; evaluation always uses the common -2 reward.")
    training.add_argument("--buffer-size", type=int, default=30000)
    training.add_argument("--batch-size", type=int, default=64)
    training.add_argument("--learning-starts", type=int, default=500)
    training.add_argument("--train-frequency", type=int, default=1)
    training.add_argument("--target-interval", type=int, default=250)
    training.add_argument("--exploration-steps", type=int, help="Absolute epsilon decay horizon; overrides fraction.")
    training.add_argument("--exploration-fraction", type=float, default=0.6)
    training.add_argument("--final-epsilon", type=float, default=0.05)
    training.add_argument("--checkpoint-interval", type=int, default=5000)
    training.add_argument("--training-densities", nargs="+", type=float,
                          help="Sample uniformly from these densities at each episode reset.")
    training.set_defaults(func=train)
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--policy", choices=["random", "heuristic", "learned"], default="learned")
    evaluation.add_argument("--checkpoint")
    evaluation.add_argument("--alpha", type=float, default=1.0)
    evaluation.add_argument("--speed-cap", type=float,
                            help="Optional observation-based ego-speed governor for learned policies.")
    evaluation.add_argument("--episodes", type=int, default=50)
    evaluation.add_argument("--seed-start", type=int, default=10000)
    evaluation.add_argument("--densities", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    evaluation.add_argument("--out", required=True)
    evaluation.add_argument("--gif")
    evaluation.set_defaults(func=evaluate)
    plotting = sub.add_parser("report")
    plotting.add_argument("--results", default="results")
    plotting.add_argument("--out", default="results/report")
    from .plots import report
    plotting.set_defaults(func=report)
    args = parser.parse_args()
    for name in ["steps", "episodes", "batch_size", "quantiles", "buffer_size",
                 "train_frequency", "target_interval", "checkpoint_interval"]:
        if hasattr(args, name) and getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if getattr(args, "exploration_steps", None) is not None and args.exploration_steps <= 0:
        parser.error("--exploration-steps must be positive")
    if hasattr(args, "alpha") and not 0 < args.alpha <= 1:
        parser.error("--alpha must be in (0,1]")
    if hasattr(args, "collision_penalty") and (not math.isfinite(args.collision_penalty) or args.collision_penalty >= 0):
        parser.error("--collision-penalty must be finite and negative")
    if hasattr(args, "gamma") and not 0 <= args.gamma < 1:
        parser.error("--gamma must be in [0,1)")
    if hasattr(args, "speed_cap") and args.speed_cap is not None:
        if not math.isfinite(args.speed_cap) or args.speed_cap <= 0 or args.policy != "learned":
            parser.error("--speed-cap requires a learned policy and a finite positive cap")
    if getattr(args, "training_densities", None) is not None:
        if any(not math.isfinite(d) or d <= 0 for d in args.training_densities):
            parser.error("--training-densities must be finite and positive")
        if len(set(args.training_densities)) != len(args.training_densities):
            parser.error("--training-densities must be distinct")
    args.func(args)


if __name__ == "__main__":
    main()
