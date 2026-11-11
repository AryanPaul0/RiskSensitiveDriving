"""Run the experiments experiment from Terminal.

This is an experiment coordinator, not the learning algorithm. It launches
separate Python commands, saves logs, and checks existing results before reuse.
Paths identify saved files; hashes are fingerprints that detect changed files.
Read main() for the order of operations. Learning itself lives in driving_rl/agents.py.

Original implementation notes:
Reproduce the fixed experiment, reusing only verified completed runs.

Run from the project root with the same Python environment used for training:
    python run_experiments.py --seeds 0 1 2 --steps 20000 --episodes 50

The initial seed block is development data. Use --seed-start 20000 only after
locking settings for final confirmation. This script never deletes run data.
"""

# import makes these modules available; "as" gives a module a shorter local name.
import argparse
# import makes these modules available; "as" gives a module a shorter local name.
import json
# from pathlib import brings the named tools into this file; parentheses allow several lines.
from pathlib import Path
# import makes these modules available; "as" gives a module a shorter local name.
import subprocess
# import makes these modules available; "as" gives a module a shorter local name.
import sys


# def defines run; parentheses list its inputs, and the colon starts its indented body.
def run(*arguments):
    """Run a command in a separate Python process and wait for it to finish.
    A nonzero exit status means the command failed; do not silently continue.
    """
    # Start another Python command and check whether it succeeded.
    # Call subprocess.run using parentheses to supply its inputs; this line does not store a separate return value.
    subprocess.run(
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        [sys.executable, "-m", "driving_rl.cli", *map(str, arguments)], check=True
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )


# def defines main; parentheses list its inputs, and the colon starts its indented body.
def main():
    """Read Terminal options, check settings, and run the requested workflow.
    Existing result files are checked before reuse so a rerun cannot silently mix
    different experiments. The main guard below prevents running jobs on import.
    """
    # = stores a result in parser; call argparse.ArgumentParser using parentheses to supply its inputs.
    parser = argparse.ArgumentParser(description=__doc__)
    # Call parser.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    # Call parser.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    parser.add_argument("--steps", type=int, default=20000)
    # Call parser.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    parser.add_argument("--episodes", type=int, default=30)
    # Call parser.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    parser.add_argument("--seed-start", type=int, default=10000)
    # = stores a result in args; call parser.parse_args using parentheses to supply its inputs.
    args = parser.parse_args()
    # = stores a result in root; use a dot to access parent on Path(__file__).resolve().
    root = Path(__file__).resolve().parent
    # if checks Path.cwd() != root; only a true condition runs the indented block.
    if Path.cwd() != root:
        # Call parser.error using parentheses to supply its inputs; this line does not store a separate return value.
        parser.error("Run this script from its project directory.")
    # = stores a result in suffix; choose the value before "if" when its condition is true, otherwise the value after "else".
    suffix = "" if args.seed_start == 10000 else f"_block{args.seed_start}"

    # def defines evaluate; parentheses list its inputs, and the colon starts its indented body.
    def evaluate(name, *options, gif=False):
        """Run or verify one frozen-policy evaluation.
        Check its model fingerprint, settings and scenario block before reusing saved
        data. This function must not quietly mix results from different experiments.
        """
        # = stores a result in path; apply / to these operands; numbers are divided, while Path objects join folder names.
        path = root / "results" / f"eval_{name}{suffix}.json"
        # Check existing data instead of silently replacing scientific results.
        # if checks path.exists(); only a true condition runs the indented block.
        if path.exists():
            # = stores a result in data; call json.loads using parentheses to supply its inputs.
            data = json.loads(path.read_text())
            # = stores a result in provenance; use square brackets to select part of data.
            provenance = data["provenance"]
            # if checks provenance['seed_start'] != args.seed_start or provenance['episodes_per_density'] != args.episodes; only a true condition runs the indented block.
            if (
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                provenance["seed_start"] != args.seed_start
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                or provenance["episodes_per_density"] != args.episodes
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            ):
                # raise signals an error and stops this normal path, so invalid data are not silently accepted.
                raise ValueError(f"Existing evaluation has different settings: {path}")
            # Only skip if the policy, selector, density matrix and checkpoint
            # content still match. This avoids silently reusing stale results.
            # import makes these modules available; "as" gives a module a shorter local name.
            import hashlib

            # = stores a result in expected; call dict using parentheses to supply its inputs.
            expected = dict(zip(options[::2], options[1::2]))
            # = stores a result in expected_policy; call expected.get using parentheses to supply its inputs.
            expected_policy = expected.get("--policy", "learned")
            # = stores a result in expected_alpha; call float using parentheses to supply its inputs.
            expected_alpha = float(expected.get("--alpha", 1))
            # = stores a result in checkpoint; call expected.get using parentheses to supply its inputs.
            checkpoint = expected.get("--checkpoint")
            # = stores a result in digest; choose the value before "if" when its condition is true, otherwise the value after "else".
            digest = (
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
                # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                if checkpoint
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                else None
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            )
            # if checks provenance['policy'] != expected_policy or provenance['alpha'] != expected_alpha or provenance.ge...; only a true condition runs the indented block.
            if (
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                provenance["policy"] != expected_policy
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                or provenance["alpha"] != expected_alpha
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                or provenance.get("speed_governor_cap_m_s") is not None
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                or provenance["checkpoint_sha256"] != digest
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                or [row["density"] for row in data["summary"]] != [1.0, 1.5, 2.0]
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            ):
                # raise signals an error and stops this normal path, so invalid data are not silently accepted.
                raise ValueError(
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    f"Existing evaluation has different policy/checkpoint: {path}"
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                )
            # Call print using parentheses to supply its inputs; this line does not store a separate return value.
            print(f"Reusing {path.name}", flush=True)
            # return ends this function and sends its result back to the code that called it.
            return
        # = stores a result in extra; choose the value before "if" when its condition is true, otherwise the value after "else".
        extra = ["--gif", f"results/{name}{suffix}.gif"] if gif else []
        # Call run using parentheses to supply its inputs; this line does not store a separate return value.
        run(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "evaluate",
            # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
            *options,
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "--episodes",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            args.episodes,
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "--seed-start",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            args.seed_start,
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "--out",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            path,
            # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
            *extra,
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )

    # for takes each item from ['random', 'heuristic'] in turn; policy holds the current item.
    for policy in ["random", "heuristic"]:
        # Call evaluate using parentheses to supply its inputs; this line does not store a separate return value.
        evaluate(policy, "--policy", policy, gif=True)
    # for takes each item from args.seeds in turn; seed holds the current item.
    for seed in args.seeds:
        # for takes each item from ['dqn', 'qrdqn'] in turn; algorithm holds the current item.
        for algorithm in ["dqn", "qrdqn"]:
            # = stores a result in directory; apply / to these operands; numbers are divided, while Path objects join folder names.
            directory = root / "results" / f"{algorithm}_seed{seed}"
            # = stores a result in checkpoint; apply / to these operands; numbers are divided, while Path objects join folder names.
            checkpoint = directory / "final.pt"
            # if checks checkpoint.exists(); only a true condition runs the indented block.
            if checkpoint.exists():
                # = stores a result in config; call json.loads using parentheses to supply its inputs.
                config = json.loads((directory / "config.json").read_text())
                # = stores a result in summary; call json.loads using parentheses to supply its inputs.
                summary = json.loads((directory / "summary.json").read_text())
                # if checks config['steps'] != args.steps or config['seed'] != seed or config['algorithm'] != algorithm or (n...; only a true condition runs the indented block.
                if (
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    config["steps"] != args.steps
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    or config["seed"] != seed
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    or config["algorithm"] != algorithm
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    or not summary["training_complete"]
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                ):
                    # raise signals an error and stops this normal path, so invalid data are not silently accepted.
                    raise ValueError(
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        f"Existing training has different settings: {directory}"
                    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                    )
                # Call print using parentheses to supply its inputs; this line does not store a separate return value.
                print(f"Reusing {directory.name}", flush=True)
            # else provides the alternative branch; after a loop it instead means the loop finished without break.
            else:
                # Call run using parentheses to supply its inputs; this line does not store a separate return value.
                run(
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    "train",
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    "--algorithm",
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    algorithm,
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    "--seed",
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    seed,
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    "--steps",
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    args.steps,
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    "--out",
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    directory,
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                )
            # Call evaluate using parentheses to supply its inputs; this line does not store a separate return value.
            evaluate(
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                f"{algorithm}_seed{seed}_mean",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "--checkpoint",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                checkpoint,
                # Inside this continued call, name=value supplies an option by name rather than by its position.
                gif=seed == args.seeds[0],
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            )
            # if checks algorithm == 'qrdqn'; only a true condition runs the indented block.
            if algorithm == "qrdqn":
                # Call evaluate using parentheses to supply its inputs; this line does not store a separate return value.
                evaluate(
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    f"{algorithm}_seed{seed}_tail25",
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    "--checkpoint",
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    checkpoint,
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    "--alpha",
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    0.25,
                    # Inside this continued call, name=value supplies an option by name rather than by its position.
                    gif=seed == args.seeds[0],
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                )
    # Call run using parentheses to supply its inputs; this line does not store a separate return value.
    run("report")


# Run the entry point only when this file is executed, not when it is imported.
# __name__ equals "__main__" when this file runs directly; this guard avoids running the script merely because another file imports it.
if __name__ == "__main__":
    # Call main using parentheses to supply its inputs; this line does not store a separate return value.
    main()
