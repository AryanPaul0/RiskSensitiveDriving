"""Read commands typed in Terminal and call the requested operation.

Arguments such as --steps and --seed become fields on args. The subcommands are
train, evaluate, benchmark and report. Input checks catch invalid settings before
a long job starts. This file chooses settings; agents.py contains the learning.

Original implementation notes:
Command-line entry points; defaults are intentionally modest.
"""

# import makes these modules available; "as" gives a module a shorter local name.
import argparse
# import makes these modules available; "as" gives a module a shorter local name.
import math

# from experiment import brings the named tools into this file; parentheses allow several lines.
from .experiment import benchmark, evaluate, train


# def defines main; parentheses list its inputs, and the colon starts its indented body.
def main():
    """Read Terminal options, check settings, and run the requested workflow.
    Existing result files are checked before reuse so a rerun cannot silently mix
    different experiments. The main guard below prevents running jobs on import.
    """
    # = stores a result in parser; call argparse.ArgumentParser using parentheses to supply its inputs.
    parser = argparse.ArgumentParser(description=__doc__)
    # = stores a result in sub; call parser.add_subparsers using parentheses to supply its inputs.
    sub = parser.add_subparsers(dest="command", required=True)
    # = stores a result in bench; call sub.add_parser using parentheses to supply its inputs.
    bench = sub.add_parser("benchmark")
    # Call bench.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    bench.add_argument("--steps", type=int, default=2000)
    # Call bench.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    bench.add_argument("--seed", type=int, default=0)
    # Call bench.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    bench.add_argument("--out", default="results/benchmark.json")
    # Call bench.set_defaults using parentheses to supply its inputs; this line does not store a separate return value.
    bench.set_defaults(func=benchmark)
    # = stores a result in training; call sub.add_parser using parentheses to supply its inputs.
    training = sub.add_parser("train")
    # Call training.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    training.add_argument("--algorithm", choices=["dqn", "qrdqn"], default="dqn")
    # Call training.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    training.add_argument("--seed", type=int, default=0)
    # Call training.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    training.add_argument("--steps", type=int, default=20000)
    # Call training.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    training.add_argument("--out", required=True)
    # Call training.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    training.add_argument("--quantiles", type=int, default=32)
    # Call training.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    training.add_argument("--learning-rate", type=float, default=5e-4)
    # Call training.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    training.add_argument("--gamma", type=float, default=0.9)
    # Call training.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    training.add_argument(
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "--collision-penalty",
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        type=float,
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        default=-2.0,
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        help="Training reward only; evaluation always uses the common -2 reward.",
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # Call training.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    training.add_argument("--buffer-size", type=int, default=30000)
    # Call training.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    training.add_argument("--batch-size", type=int, default=64)
    # Call training.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    training.add_argument("--learning-starts", type=int, default=500)
    # Call training.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    training.add_argument("--train-frequency", type=int, default=1)
    # Call training.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    training.add_argument("--target-interval", type=int, default=250)
    # Call training.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    training.add_argument(
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "--exploration-steps",
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        type=int,
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        help="Absolute epsilon decay horizon; overrides fraction.",
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # Call training.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    training.add_argument("--exploration-fraction", type=float, default=0.6)
    # Call training.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    training.add_argument("--final-epsilon", type=float, default=0.05)
    # Call training.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    training.add_argument("--checkpoint-interval", type=int, default=5000)
    # Call training.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    training.add_argument(
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "--training-densities",
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        nargs="+",
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        type=float,
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        help="Sample uniformly from these densities at each episode reset.",
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # Call training.set_defaults using parentheses to supply its inputs; this line does not store a separate return value.
    training.set_defaults(func=train)
    # = stores a result in evaluation; call sub.add_parser using parentheses to supply its inputs.
    evaluation = sub.add_parser("evaluate")
    # Call evaluation.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    evaluation.add_argument(
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        "--policy", choices=["random", "heuristic", "learned"], default="learned"
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # Call evaluation.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    evaluation.add_argument("--checkpoint")
    # Call evaluation.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    evaluation.add_argument("--alpha", type=float, default=1.0)
    # Call evaluation.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    evaluation.add_argument(
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "--speed-cap",
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        type=float,
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        help="Optional observation-based ego-speed governor for learned policies.",
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # Call evaluation.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    evaluation.add_argument("--episodes", type=int, default=50)
    # Call evaluation.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    evaluation.add_argument("--seed-start", type=int, default=10000)
    # Call evaluation.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    evaluation.add_argument(
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        "--densities", type=float, nargs="+", default=[1.0, 1.5, 2.0]
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # Call evaluation.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    evaluation.add_argument("--out", required=True)
    # Call evaluation.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    evaluation.add_argument("--gif")
    # Call evaluation.set_defaults using parentheses to supply its inputs; this line does not store a separate return value.
    evaluation.set_defaults(func=evaluate)
    # = stores a result in plotting; call sub.add_parser using parentheses to supply its inputs.
    plotting = sub.add_parser("report")
    # Call plotting.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    plotting.add_argument("--results", default="results")
    # Call plotting.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    plotting.add_argument("--out", default="results/report")
    # from plots import brings the named tools into this file; parentheses allow several lines.
    from .plots import report

    # Call plotting.set_defaults using parentheses to supply its inputs; this line does not store a separate return value.
    plotting.set_defaults(func=report)
    # = stores a result in args; call parser.parse_args using parentheses to supply its inputs.
    args = parser.parse_args()
    # for takes each item from ['steps', 'episodes', 'batch_size', 'quantiles', 'buffer_size', 'train_frequency', 'target_interv... in turn; name holds the current item.
    for name in [
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "steps",
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "episodes",
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "batch_size",
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "quantiles",
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "buffer_size",
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "train_frequency",
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "target_interval",
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "checkpoint_interval",
    # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
    ]:
        # if checks hasattr(args, name) and getattr(args, name) <= 0; only a true condition runs the indented block.
        if hasattr(args, name) and getattr(args, name) <= 0:
            # Call parser.error using parentheses to supply its inputs; this line does not store a separate return value.
            parser.error(f"--{name.replace('_', '-')} must be positive")
    # if checks getattr(args, 'exploration_steps', None) is not None and args.exploration_steps <= 0; only a true condition runs the indented block.
    if (
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        getattr(args, "exploration_steps", None) is not None
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        and args.exploration_steps <= 0
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    ):
        # Call parser.error using parentheses to supply its inputs; this line does not store a separate return value.
        parser.error("--exploration-steps must be positive")
    # if checks hasattr(args, 'alpha') and (not 0 < args.alpha <= 1); only a true condition runs the indented block.
    if hasattr(args, "alpha") and not 0 < args.alpha <= 1:
        # Call parser.error using parentheses to supply its inputs; this line does not store a separate return value.
        parser.error("--alpha must be in (0,1]")
    # if checks hasattr(args, 'collision_penalty') and (not math.isfinite(args.collision_penalty) or args.collisi...; only a true condition runs the indented block.
    if hasattr(args, "collision_penalty") and (
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        not math.isfinite(args.collision_penalty) or args.collision_penalty >= 0
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    ):
        # Call parser.error using parentheses to supply its inputs; this line does not store a separate return value.
        parser.error("--collision-penalty must be finite and negative")
    # if checks hasattr(args, 'gamma') and (not 0 <= args.gamma < 1); only a true condition runs the indented block.
    if hasattr(args, "gamma") and not 0 <= args.gamma < 1:
        # Call parser.error using parentheses to supply its inputs; this line does not store a separate return value.
        parser.error("--gamma must be in [0,1)")
    # if checks hasattr(args, 'speed_cap') and args.speed_cap is not None; only a true condition runs the indented block.
    if hasattr(args, "speed_cap") and args.speed_cap is not None:
        # if checks not math.isfinite(args.speed_cap) or args.speed_cap <= 0 or args.policy != 'learned'; only a true condition runs the indented block.
        if (
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            not math.isfinite(args.speed_cap)
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            or args.speed_cap <= 0
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            or args.policy != "learned"
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        ):
            # Call parser.error using parentheses to supply its inputs; this line does not store a separate return value.
            parser.error(
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "--speed-cap requires a learned policy and a finite positive cap"
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            )
    # if checks getattr(args, 'training_densities', None) is not None; only a true condition runs the indented block.
    if getattr(args, "training_densities", None) is not None:
        # if checks any((not math.isfinite(d) or d <= 0 for d in args.training_densities)); only a true condition runs the indented block.
        if any(not math.isfinite(d) or d <= 0 for d in args.training_densities):
            # Call parser.error using parentheses to supply its inputs; this line does not store a separate return value.
            parser.error("--training-densities must be finite and positive")
        # if checks len(set(args.training_densities)) != len(args.training_densities); only a true condition runs the indented block.
        if len(set(args.training_densities)) != len(args.training_densities):
            # Call parser.error using parentheses to supply its inputs; this line does not store a separate return value.
            parser.error("--training-densities must be distinct")
    # Call args.func using parentheses to supply its inputs; this line does not store a separate return value.
    args.func(args)


# Run the entry point only when this file is executed, not when it is imported.
# __name__ equals "__main__" when this file runs directly; this guard avoids running the script merely because another file imports it.
if __name__ == "__main__":
    # Call main using parentheses to supply its inputs; this line does not store a separate return value.
    main()
