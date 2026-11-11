"""Read commands typed in Terminal and call the requested operation.

Arguments such as --steps and --seed become fields on args. The subcommands are
train, evaluate, benchmark and report. Input checks catch invalid settings before
a long job starts. This file chooses settings; agents.py contains the learning.

Original implementation notes:
Command-line entry points; defaults are intentionally modest.
"""

import argparse  # Make argparse available for this file's calculations and workflow.
import math  # Make math available for this file's calculations and workflow.

from .experiment import benchmark, evaluate, train  # Make benchmark, evaluate, train available for this file's calculations and workflow.


def main():  # Read Terminal options, check settings, and run the requested workflow.
    """Read Terminal options, check settings, and run the requested workflow.
    Existing result files are checked before reuse so a rerun cannot silently mix
    different experiments. The main guard below prevents running jobs on import.
    """
    parser = argparse.ArgumentParser(description=__doc__)  # Set parser to the result of argparse.ArgumentParser.
    sub = parser.add_subparsers(dest="command", required=True)  # Set sub to the result of parser.add subparsers.
    bench = sub.add_parser("benchmark")  # Set bench to sub.add parser applied to 'benchmark'.
    bench.add_argument("--steps", type=int, default=2000)  # Bench.add argument applied to '--steps'.
    bench.add_argument("--seed", type=int, default=0)  # Bench.add argument applied to '--seed'.
    bench.add_argument("--out", default="results/benchmark.json")  # Bench.add argument applied to '--out'.
    bench.set_defaults(func=benchmark)  # The result of bench.set defaults.
    training = sub.add_parser("train")  # Set training to sub.add parser applied to 'train'.
    training.add_argument("--algorithm", choices=["dqn", "qrdqn"], default="dqn")  # Training.add argument applied to '--algorithm'.
    training.add_argument("--seed", type=int, default=0)  # Training.add argument applied to '--seed'.
    training.add_argument("--steps", type=int, default=20000)  # Training.add argument applied to '--steps'.
    training.add_argument("--out", required=True)  # Training.add argument applied to '--out'.
    training.add_argument("--quantiles", type=int, default=32)  # Training.add argument applied to '--quantiles'.
    training.add_argument("--learning-rate", type=float, default=5e-4)  # Training.add argument applied to '--learning-rate'.
    training.add_argument("--gamma", type=float, default=0.9)  # Training.add argument applied to '--gamma'.
    training.add_argument(  # Training.add argument applied to '--collision-penalty'.
        "--collision-penalty",  # Include '--collision-penalty' in the surrounding operation.
        type=float,  # Include float in the surrounding operation.
        default=-2.0,  # Include negative 2.0 in the surrounding operation.
        help="Training reward only; evaluation always uses the common -2 reward.",  # Include the report text in the surrounding operation.
    )  # Finish assembling training.add argument inputs.
    training.add_argument("--buffer-size", type=int, default=30000)  # Training.add argument applied to '--buffer-size'.
    training.add_argument("--batch-size", type=int, default=64)  # Training.add argument applied to '--batch-size'.
    training.add_argument("--learning-starts", type=int, default=500)  # Training.add argument applied to '--learning-starts'.
    training.add_argument("--train-frequency", type=int, default=1)  # Training.add argument applied to '--train-frequency'.
    training.add_argument("--target-interval", type=int, default=250)  # Training.add argument applied to '--target-interval'.
    training.add_argument(  # Training.add argument applied to '--exploration-steps'.
        "--exploration-steps",  # Include '--exploration-steps' in the surrounding operation.
        type=int,  # Include int in the surrounding operation.
        help="Absolute epsilon decay horizon; overrides fraction.",  # Include 'Absolute epsilon decay horizon; overrides fraction.' in the surrounding operation.
    )  # Finish assembling training.add argument inputs.
    training.add_argument("--exploration-fraction", type=float, default=0.6)  # Training.add argument applied to '--exploration-fraction'.
    training.add_argument("--final-epsilon", type=float, default=0.05)  # Training.add argument applied to '--final-epsilon'.
    training.add_argument("--checkpoint-interval", type=int, default=5000)  # Training.add argument applied to '--checkpoint-interval'.
    training.add_argument(  # Training.add argument applied to '--training-densities'.
        "--training-densities",  # Include '--training-densities' in the surrounding operation.
        nargs="+",  # Include '+' in the surrounding operation.
        type=float,  # Include float in the surrounding operation.
        help="Sample uniformly from these densities at each episode reset.",  # Include 'Sample uniformly from these densities at each episode reset.' in the surrounding operation.
    )  # Finish assembling training.add argument inputs.
    training.set_defaults(func=train)  # The result of training.set defaults.
    evaluation = sub.add_parser("evaluate")  # Set evaluation to sub.add parser applied to 'evaluate'.
    evaluation.add_argument(  # Evaluation.add argument applied to '--policy'.
        "--policy", choices=["random", "heuristic", "learned"], default="learned"  # Include '--policy', 'random', 'heuristic', 'learned', 'learned' in the surrounding operation.
    )  # Finish assembling evaluation.add argument inputs.
    evaluation.add_argument("--checkpoint")  # Evaluation.add argument applied to '--checkpoint'.
    evaluation.add_argument("--alpha", type=float, default=1.0)  # Evaluation.add argument applied to '--alpha'.
    evaluation.add_argument(  # Evaluation.add argument applied to '--speed-cap'.
        "--speed-cap",  # Include '--speed-cap' in the surrounding operation.
        type=float,  # Include float in the surrounding operation.
        help="Optional observation-based ego-speed governor for learned policies.",  # Include the report text in the surrounding operation.
    )  # Finish assembling evaluation.add argument inputs.
    evaluation.add_argument("--episodes", type=int, default=50)  # Evaluation.add argument applied to '--episodes'.
    evaluation.add_argument("--seed-start", type=int, default=10000)  # Evaluation.add argument applied to '--seed-start'.
    evaluation.add_argument(  # Evaluation.add argument applied to '--densities'.
        "--densities", type=float, nargs="+", default=[1.0, 1.5, 2.0]  # Include '--densities', float, '+', 1.0, 1.5, 2.0 in the surrounding operation.
    )  # Finish assembling evaluation.add argument inputs.
    evaluation.add_argument("--out", required=True)  # Evaluation.add argument applied to '--out'.
    evaluation.add_argument("--gif")  # Evaluation.add argument applied to '--gif'.
    evaluation.set_defaults(func=evaluate)  # The result of evaluation.set defaults.
    plotting = sub.add_parser("report")  # Set plotting to sub.add parser applied to 'report'.
    plotting.add_argument("--results", default="results")  # Plotting.add argument applied to '--results'.
    plotting.add_argument("--out", default="results/report")  # Plotting.add argument applied to '--out'.
    from .plots import report  # Make report available for this file's calculations and workflow.

    plotting.set_defaults(func=report)  # The result of plotting.set defaults.
    args = parser.parse_args()  # Set args to the result of parser.parse args.
    for name in [  # Process each name from the listed settings or measurements in their recorded order.
        "steps",  # Include 'steps' in the surrounding operation.
        "episodes",  # Include 'episodes' in the surrounding operation.
        "batch_size",  # Include 'batch_size' in the surrounding operation.
        "quantiles",  # Include 'quantiles' in the surrounding operation.
        "buffer_size",  # Include 'buffer_size' in the surrounding operation.
        "train_frequency",  # Include 'train_frequency' in the surrounding operation.
        "target_interval",  # Include 'target_interval' in the surrounding operation.
        "checkpoint_interval",  # Include 'checkpoint_interval' in the surrounding operation.
    ]:  # Finish assembling the grouped values above.
        if hasattr(args, name) and getattr(args, name) <= 0:  # Use this branch when hasattr applied to args, name and getattr applied to args, name is at most 0.
            parser.error(f"--{name.replace('_', '-')} must be positive")  # Parser.error applied to f"--{name.replace(' ', '-')} must be positive".
    if (  # Use this branch when getattr applied to args, 'exploration_steps', None is not None and args.exploration steps is at most 0.
        getattr(args, "exploration_steps", None) is not None  # Include getattr applied to args, 'exploration_steps', None is not None and args.exploration steps is at most 0 in the surrounding operation.
        and args.exploration_steps <= 0  # Include args.exploration steps is at most 0 in the surrounding operation.
    ):  # Finish assembling the grouped values above.
        parser.error("--exploration-steps must be positive")  # Parser.error applied to '--exploration-steps must be positive'.
    if hasattr(args, "alpha") and not 0 < args.alpha <= 1:  # Use this branch when hasattr applied to args, 'alpha' and not 0 is below args.alpha is at most 1.
        parser.error("--alpha must be in (0,1]")  # Parser.error applied to '--alpha must be in (0,1]'.
    if hasattr(args, "collision_penalty") and (  # Use this branch when hasattr applied to args, 'collision_penalty' and not math.isfinite applied to args.collision penalty or args.collision penalty is at least 0.
        not math.isfinite(args.collision_penalty) or args.collision_penalty >= 0  # Include not math.isfinite applied to args.collision penalty or args.collision penalty is at least 0 in the surrounding operation.
    ):  # Finish assembling the grouped values above.
        parser.error("--collision-penalty must be finite and negative")  # Parser.error applied to '--collision-penalty must be finite and negative'.
    if hasattr(args, "gamma") and not 0 <= args.gamma < 1:  # Use this branch when hasattr applied to args, 'gamma' and not 0 is at most args.gamma is below 1.
        parser.error("--gamma must be in [0,1)")  # Parser.error applied to '--gamma must be in [0,1)'.
    if hasattr(args, "speed_cap") and args.speed_cap is not None:  # Use this branch when hasattr applied to args, 'speed_cap' and args.speed cap is not None.
        if (  # Use this branch when not math.isfinite applied to args.speed cap or args.speed cap is at most 0 or args.policy differs from 'learned'.
            not math.isfinite(args.speed_cap)  # Include not math.isfinite applied to args.speed cap or args.speed cap is at most 0 or args.policy differs from 'learned' in the surrounding operation.
            or args.speed_cap <= 0  # Include args.speed cap is at most 0 in the surrounding operation.
            or args.policy != "learned"  # Include args.policy differs from 'learned' in the surrounding operation.
        ):  # Finish assembling the grouped values above.
            parser.error(  # Parser.error applied to '--speed-cap requires a learned policy and a finite positive cap'.
                "--speed-cap requires a learned policy and a finite positive cap"  # Include '--speed-cap requires a learned policy and a finite positive cap' in the surrounding operation.
            )  # Finish assembling parser.error inputs.
    if getattr(args, "training_densities", None) is not None:  # Use this branch when getattr applied to args, 'training_densities', None is not None.
        if any(not math.isfinite(d) or d <= 0 for d in args.training_densities):  # Use this branch when any applied to not math.isfinite(d) or d is at most 0 for each d in args.training densities.
            parser.error("--training-densities must be finite and positive")  # Parser.error applied to '--training-densities must be finite and positive'.
        if len(set(args.training_densities)) != len(args.training_densities):  # Use this branch when the number of entries in set applied to args.training densities differs from the number of entries in args.training densities.
            parser.error("--training-densities must be distinct")  # Parser.error applied to '--training-densities must be distinct'.
    args.func(args)  # Args.func applied to args.


if __name__ == "__main__":  # Run this workflow when launched from Terminal; importing its helpers must not launch experiments.
    main()  # The result of main.
