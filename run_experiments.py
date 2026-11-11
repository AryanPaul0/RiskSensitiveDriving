"""Reproduce the fixed experiment, reusing only verified completed runs.

Run from the project root with the same Python environment used for training:
    python run_experiments.py --seeds 0 1 2 --steps 20000 --episodes 50

The initial seed block is development data. Use --seed-start 20000 only after
locking settings for final confirmation. This script never deletes run data.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def run(*arguments):
    subprocess.run([sys.executable, "-m", "driving_rl.cli", *map(str, arguments)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=10000)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if Path.cwd() != root:
        parser.error("Run this script from its project directory.")
    suffix = "" if args.seed_start == 10000 else f"_block{args.seed_start}"

    def evaluate(name, *options, gif=False):
        path = root / "results" / f"eval_{name}{suffix}.json"
        if path.exists():
            data = json.loads(path.read_text())
            provenance = data["provenance"]
            if (provenance["seed_start"] != args.seed_start or
                    provenance["episodes_per_density"] != args.episodes):
                raise ValueError(f"Existing evaluation has different settings: {path}")
            # Only skip if the policy, selector, density matrix and checkpoint
            # content still match. This avoids silently reusing stale results.
            import hashlib
            expected = dict(zip(options[::2], options[1::2]))
            expected_policy = expected.get("--policy", "learned")
            expected_alpha = float(expected.get("--alpha", 1))
            checkpoint = expected.get("--checkpoint")
            digest = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() if checkpoint else None
            if (provenance["policy"] != expected_policy or provenance["alpha"] != expected_alpha or
                    provenance.get("speed_governor_cap_m_s") is not None or
                    provenance["checkpoint_sha256"] != digest or
                    [row["density"] for row in data["summary"]] != [1., 1.5, 2.]):
                raise ValueError(f"Existing evaluation has different policy/checkpoint: {path}")
            print(f"Reusing {path.name}", flush=True)
            return
        extra = ["--gif", f"results/{name}{suffix}.gif"] if gif else []
        run("evaluate", *options, "--episodes", args.episodes,
            "--seed-start", args.seed_start, "--out", path, *extra)

    for policy in ["random", "heuristic"]:
        evaluate(policy, "--policy", policy, gif=True)
    for seed in args.seeds:
        for algorithm in ["dqn", "qrdqn"]:
            directory = root / "results" / f"{algorithm}_seed{seed}"
            checkpoint = directory / "final.pt"
            if checkpoint.exists():
                config = json.loads((directory / "config.json").read_text())
                summary = json.loads((directory / "summary.json").read_text())
                if (config["steps"] != args.steps or config["seed"] != seed or
                        config["algorithm"] != algorithm or not summary["training_complete"]):
                    raise ValueError(f"Existing training has different settings: {directory}")
                print(f"Reusing {directory.name}", flush=True)
            else:
                run("train", "--algorithm", algorithm, "--seed", seed,
                    "--steps", args.steps, "--out", directory)
            evaluate(f"{algorithm}_seed{seed}_mean", "--checkpoint", checkpoint, gif=seed == args.seeds[0])
            if algorithm == "qrdqn":
                evaluate(f"{algorithm}_seed{seed}_tail25", "--checkpoint", checkpoint,
                         "--alpha", .25, gif=seed == args.seeds[0])
    run("report")


if __name__ == "__main__":
    main()
