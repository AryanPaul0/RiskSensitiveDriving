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

import argparse  # Make argparse available for this file's calculations and workflow.
import json  # Make json available for this file's calculations and workflow.
from pathlib import Path  # Make Path available for this file's calculations and workflow.
import subprocess  # Make subprocess available for this file's calculations and workflow.
import sys  # Make sys available for this file's calculations and workflow.


def run(*arguments):  # Run a command in a separate Python process and wait for it to finish.
    """Run a command in a separate Python process and wait for it to finish.
    A nonzero exit status means the command failed; do not silently continue.
    """
    subprocess.run(  # Start another Python command and check whether it succeeded.
        [sys.executable, "-m", "driving_rl.cli", *map(str, arguments)], check=True  # Include sys.executable, '-m', 'driving_rl.cli', *map(str, arguments), True in the surrounding operation.
    )  # Finish assembling subprocess.run inputs.


def main():  # Read Terminal options, check settings, and run the requested workflow.
    """Read Terminal options, check settings, and run the requested workflow.
    Existing result files are checked before reuse so a rerun cannot silently mix
    different experiments. The main guard below prevents running jobs on import.
    """
    parser = argparse.ArgumentParser(description=__doc__)  # Set parser to the result of argparse.ArgumentParser.
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])  # Parser.add argument applied to '--seeds'.
    parser.add_argument("--steps", type=int, default=20000)  # Parser.add argument applied to '--steps'.
    parser.add_argument("--episodes", type=int, default=30)  # Parser.add argument applied to '--episodes'.
    parser.add_argument("--seed-start", type=int, default=10000)  # Parser.add argument applied to '--seed-start'.
    args = parser.parse_args()  # Set args to the result of parser.parse args.
    root = Path(__file__).resolve().parent  # Set root to Path( file ).resolve().parent.
    if Path.cwd() != root:  # Use this branch when the result of Path.cwd differs from root.
        parser.error("Run this script from its project directory.")  # Parser.error applied to 'run this script from its project directory.'.
    suffix = "" if args.seed_start == 10000 else f"_block{args.seed_start}"  # Set suffix to '' when args.seed start equals 10000, otherwise f' block{args.seed start}'.

    def evaluate(name, *options, gif=False):  # Run or verify one frozen-policy evaluation.
        """Run or verify one frozen-policy evaluation.
        Check its model fingerprint, settings and scenario block before reusing saved
        data. This function must not quietly mix results from different experiments.
        """
        path = root / "results" / f"eval_{name}{suffix}.json"  # Set path to (the path to 'results' inside root divided by f'eval {name}{suffix}.json').
        if path.exists():  # Check existing data instead of silently replacing scientific results.
            data = json.loads(path.read_text())  # Set data to json.loads applied to the result of path.read text.
            provenance = data["provenance"]  # Set provenance to the entries of data selected by 'provenance'.
            if (  # Use this branch when the entries of provenance selected by 'seed_start' differs from args.seed start or the entries of provenance selected by 'episodes_per_density' differs from args.episodes.
                provenance["seed_start"] != args.seed_start  # Include the entries of provenance selected by 'seed_start' differs from args.seed start or the entries of provenance selected by 'episodes_per_density' differs from args.episodes in the surrounding operation.
                or provenance["episodes_per_density"] != args.episodes  # Include the entries of provenance selected by 'episodes_per_density' differs from args.episodes in the surrounding operation.
            ):  # Finish assembling the grouped values above.
                raise ValueError(f"Existing evaluation has different settings: {path}")  # Reject this invalid case instead of continuing with misleading experiment results.
            import hashlib  # Only skip if the policy, selector, density matrix and checkpoint content still match. This avoids silently reusing stale results.

            expected = dict(zip(options[::2], options[1::2]))  # Set expected to dict applied to matching entries grouped across the entries of options selected by ::2, the entries of options selected by 1::2.
            expected_policy = expected.get("--policy", "learned")  # Set expected policy to expected.get applied to '--policy', 'learned'.
            expected_alpha = float(expected.get("--alpha", 1))  # Set expected alpha to expected.get applied to '--alpha', 1 as a floating-point value.
            checkpoint = expected.get("--checkpoint")  # Set checkpoint to expected.get applied to '--checkpoint'.
            digest = (  # Set digest to the result of hashlib.sha256(Path(checkpoint).read bytes()).hexdigest when checkpoint, otherwise None.
                hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()  # Include the result of hashlib.sha256(Path(checkpoint).read bytes()).hexdigest when checkpoint, otherwise None in the surrounding operation.
                if checkpoint  # Include checkpoint in the surrounding operation.
                else None  # Include None in the surrounding operation.
            )  # Finish assembling the grouped values above.
            if (  # Use this branch when the entries of provenance selected by 'policy' differs from expected policy or the entries of provenance selected by 'alpha' differs from expected alpha or provenance.get applied to 'speed_governor_cap_m_s' is not None or the entries of provenance selected by 'checkpoint_sha256' differs from digest or the entries of row selected by 'density' for each row in the entries of data selected by 'summary' differs from 1.0, 1.5, 2.0.
                provenance["policy"] != expected_policy  # Include the entries of provenance selected by 'policy' differs from expected policy or the entries of provenance selected by 'alpha' differs from expected alpha or provenance.get applied to 'speed_governor_cap_m_s' is not None or the entries of provenance selected by 'checkpoint_sha256' differs from digest or the entries of row selected by 'density' for each row in the entries of data selected by 'summary' differs from 1.0, 1.5, 2.0 in the surrounding operation.
                or provenance["alpha"] != expected_alpha  # Include the entries of provenance selected by 'alpha' differs from expected alpha in the surrounding operation.
                or provenance.get("speed_governor_cap_m_s") is not None  # Include provenance.get applied to 'speed_governor_cap_m_s' is not None in the surrounding operation.
                or provenance["checkpoint_sha256"] != digest  # Include the entries of provenance selected by 'checkpoint_sha256' differs from digest in the surrounding operation.
                or [row["density"] for row in data["summary"]] != [1.0, 1.5, 2.0]  # Include the entries of row selected by 'density' for each row in the entries of data selected by 'summary' differs from 1.0, 1.5, 2.0 in the surrounding operation.
            ):  # Finish assembling the grouped values above.
                raise ValueError(  # Reject this invalid case instead of continuing with misleading experiment results.
                    f"Existing evaluation has different policy/checkpoint: {path}"  # Include f'Existing evaluation has different policy/checkpoint: {path}' in the surrounding operation.
                )  # Finish assembling ValueError inputs.
            print(f"Reusing {path.name}", flush=True)  # Print applied to f'reusing {path.name}'.
            return  # Finish this operation without producing another result.
        extra = ["--gif", f"results/{name}{suffix}.gif"] if gif else []  # Set extra to '--gif', f'results/{name}{suffix}.gif' when gif, otherwise .
        run(  # Run applied to 'evaluate', *options, '--episodes', args.episodes, '--seed-start', args.seed start, '--out', path, *extra.
            "evaluate",  # Include 'evaluate' in the surrounding operation.
            *options,  # Include *options in the surrounding operation.
            "--episodes",  # Include '--episodes' in the surrounding operation.
            args.episodes,  # Include args.episodes in the surrounding operation.
            "--seed-start",  # Include '--seed-start' in the surrounding operation.
            args.seed_start,  # Include args.seed start in the surrounding operation.
            "--out",  # Include '--out' in the surrounding operation.
            path,  # Include path in the surrounding operation.
            *extra,  # Include *extra in the surrounding operation.
        )  # Finish assembling run inputs.

    for policy in ["random", "heuristic"]:  # Process each policy from 'random', 'heuristic'.
        evaluate(policy, "--policy", policy, gif=True)  # Evaluate applied to policy, '--policy', policy.
    for seed in args.seeds:  # Process each seed from args.seeds.
        for algorithm in ["dqn", "qrdqn"]:  # Process each algorithm from 'dqn', 'qrdqn'.
            directory = root / "results" / f"{algorithm}_seed{seed}"  # Set directory to (the path to 'results' inside root divided by f'{algorithm} seed{seed}').
            checkpoint = directory / "final.pt"  # Set checkpoint to the path to 'final.pt' inside directory.
            if checkpoint.exists():  # Use this branch when the result of checkpoint.exists.
                config = json.loads((directory / "config.json").read_text())  # Set config to json.loads applied to the result of (directory / 'config.json').read text.
                summary = json.loads((directory / "summary.json").read_text())  # Set summary to json.loads applied to the result of (directory / 'summary.json').read text.
                if (  # Use this branch when the entries of config selected by 'steps' differs from args.steps or the entries of config selected by 'seed' differs from seed or the entries of config selected by 'algorithm' differs from algorithm or not the entries of summary selected by 'training_complete'.
                    config["steps"] != args.steps  # Include the entries of config selected by 'steps' differs from args.steps or the entries of config selected by 'seed' differs from seed or the entries of config selected by 'algorithm' differs from algorithm or not the entries of summary selected by 'training_complete' in the surrounding operation.
                    or config["seed"] != seed  # Include the entries of config selected by 'seed' differs from seed in the surrounding operation.
                    or config["algorithm"] != algorithm  # Include the entries of config selected by 'algorithm' differs from algorithm in the surrounding operation.
                    or not summary["training_complete"]  # Include not the entries of summary selected by 'training_complete' in the surrounding operation.
                ):  # Finish assembling the grouped values above.
                    raise ValueError(  # Reject this invalid case instead of continuing with misleading experiment results.
                        f"Existing training has different settings: {directory}"  # Include f'Existing training has different settings: {directory}' in the surrounding operation.
                    )  # Finish assembling ValueError inputs.
                print(f"Reusing {directory.name}", flush=True)  # Print applied to f'reusing {directory.name}'.
            else:  # Handle the alternative when the earlier condition (the result of checkpoint.exists) is false.
                run(  # Run applied to 'train', '--algorithm', algorithm, '--seed', seed, '--steps', args.steps, '--out', directory.
                    "train",  # Include 'train' in the surrounding operation.
                    "--algorithm",  # Include '--algorithm' in the surrounding operation.
                    algorithm,  # Include algorithm in the surrounding operation.
                    "--seed",  # Include '--seed' in the surrounding operation.
                    seed,  # Include seed in the surrounding operation.
                    "--steps",  # Include '--steps' in the surrounding operation.
                    args.steps,  # Include args.steps in the surrounding operation.
                    "--out",  # Include '--out' in the surrounding operation.
                    directory,  # Include directory in the surrounding operation.
                )  # Finish assembling run inputs.
            evaluate(  # Evaluate applied to f'{algorithm} seed{seed} mean', '--checkpoint', checkpoint.
                f"{algorithm}_seed{seed}_mean",  # Include f'{algorithm} seed{seed} mean' in the surrounding operation.
                "--checkpoint",  # Include '--checkpoint' in the surrounding operation.
                checkpoint,  # Include checkpoint in the surrounding operation.
                gif=seed == args.seeds[0],  # Include seed equals the entries of args.seeds selected by 0 in the surrounding operation.
            )  # Finish assembling evaluate inputs.
            if algorithm == "qrdqn":  # Use this branch when algorithm equals 'qrdqn'.
                evaluate(  # Evaluate applied to f'{algorithm} seed{seed} tail25', '--checkpoint', checkpoint, '--alpha', 0.25.
                    f"{algorithm}_seed{seed}_tail25",  # Include f'{algorithm} seed{seed} tail25' in the surrounding operation.
                    "--checkpoint",  # Include '--checkpoint' in the surrounding operation.
                    checkpoint,  # Include checkpoint in the surrounding operation.
                    "--alpha",  # Include '--alpha' in the surrounding operation.
                    0.25,  # Include 0.25 in the surrounding operation.
                    gif=seed == args.seeds[0],  # Include seed equals the entries of args.seeds selected by 0 in the surrounding operation.
                )  # Finish assembling evaluate inputs.
    run("report")  # Run applied to 'report'.


if __name__ == "__main__":  # Run this workflow when launched from Terminal; importing its helpers must not launch experiments.
    main()  # The result of main.
