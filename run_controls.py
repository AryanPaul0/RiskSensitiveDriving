"""Run the controls experiment from Terminal.

This is an experiment coordinator, not the learning algorithm. It launches
separate Python commands, saves logs, and checks existing results before reuse.
Paths identify saved files; hashes are fingerprints that detect changed files.
Read main() for the order of operations. Learning itself lives in driving_rl/agents.py.

Original implementation notes:
Reproduce control experiments without overwriting completed experiments or using test seeds.

Run from this directory: python run_controls.py --jobs 2
At most --jobs scalar-baseline training jobs run concurrently. The primary
replication suite is run first and reuses verified completed results.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from driving_rl.analysis import choose_speed_targeted_baseline
from driving_rl.experiment import save_json

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
STAGE = RESULTS / "controls"


def execute(arguments, log):
    """Run the requested command with this project's Python interpreter.
    Save its console messages in a log so failures can be inspected later.
    """
    with Path(log).open("w") as file:
        # Start another Python command and check whether it succeeded.
        subprocess.run(
            [sys.executable, "-m", "driving_rl.cli", *map(str, arguments)],
            stdout=file,
            stderr=subprocess.STDOUT,
            check=True,
        )


def ensure_training(penalty, seed):
    """Train a missing model, or check that the saved run really matches the requested settings."""
    directory = RESULTS / f"dqn_penalty{abs(int(penalty))}_seed{seed}"
    if penalty == -2:
        directory = RESULTS / f"dqn_seed{seed}"
    checkpoint = directory / "final.pt"
    if checkpoint.exists():
        config = json.loads((directory / "config.json").read_text())
        summary = json.loads((directory / "summary.json").read_text())
        expected = {
            "steps": 20000,
            "seed": seed,
            "algorithm": "dqn",
            "gamma": 0.9,
            "learning_rate": 0.0005,
            "batch_size": 64,
            "target_interval": 250,
        }
        if (
            not summary["training_complete"]
            or config.get("collision_penalty", -2.0) != penalty
            or any(config[key] != value for key, value in expected.items())
        ):
            raise ValueError(f"Existing training settings do not match: {directory}")
        print(f"Reusing training {directory.name}", flush=True)
    else:
        execute(
            [
                "train",
                "--algorithm",
                "dqn",
                "--collision-penalty",
                penalty,
                "--seed",
                seed,
                "--steps",
                20000,
                "--out",
                directory,
            ],
            RESULTS / f"{directory.name}.log",
        )
    return checkpoint


def ensure_evaluation(penalty, seed, checkpoint):
    """Evaluate a missing result, or verify that saved results belong to the correct model and settings."""
    output = STAGE / f"eval_dqn_penalty{abs(int(penalty))}_seed{seed}.json"
    if output.exists():
        data = json.loads(output.read_text())
        metadata = data["provenance"]
        if (
            metadata["checkpoint_sha256"]
            != hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            or metadata["training_collision_penalty"] != penalty
            or metadata["evaluation_collision_penalty"] != -2.0
            or metadata["seed_start"] != 10000
            or metadata["episodes_per_density"] != 30
            or metadata["alpha"] != 1.0
            or metadata.get("speed_governor_cap_m_s") is not None
            or [row["density"] for row in data["summary"]] != [1.0, 1.5, 2.0]
        ):
            raise ValueError(f"Existing evaluation settings do not match: {output}")
    else:
        gif = (
            ["--gif", RESULTS / f"dqn_penalty{abs(int(penalty))}_seed{seed}.gif"]
            if seed == 0
            else []
        )
        execute(
            [
                "evaluate",
                "--checkpoint",
                checkpoint,
                "--episodes",
                30,
                "--out",
                output,
                *gif,
            ],
            STAGE / f"evaluation_penalty{abs(int(penalty))}_seed{seed}.log",
        )
    return output


def candidate(penalty):
    """Read one speed-control candidate and record its measured speed and file fingerprint."""
    checkpoint = ensure_training(penalty, 0)
    path = (
        RESULTS / "eval_dqn_seed0_mean.json"
        if penalty == -2
        else ensure_evaluation(penalty, 0, checkpoint)
    )
    data = json.loads(path.read_text())
    normal = next(row for row in data["summary"] if row["density"] == 1.0)
    return {
        "collision_penalty": penalty,
        "speed_m_s": normal["mean_speed_m_s"],
        "evaluation_path": str(path.relative_to(ROOT)),
        "evaluation_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main():
    """Read Terminal options, check settings, and run the requested workflow.
    Existing result files are checked before reuse so a rerun cannot silently mix
    different experiments. The main guard below prevents running jobs on import.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if Path.cwd() != ROOT or not 1 <= args.jobs <= 4:
        parser.error("Run from the project directory; --jobs must be between 1 and 4.")
    STAGE.mkdir(parents=True, exist_ok=True)
    with (STAGE / "core_suite.log").open("w") as file:
        # Start another Python command and check whether it succeeded.
        subprocess.run(
            [
                sys.executable,
                "run_experiments.py",
                "--seeds",
                "0",
                "1",
                "2",
                "--steps",
                "20000",
                "--episodes",
                "30",
            ],
            stdout=file,
            stderr=subprocess.STDOUT,
            check=True,
        )
    # Run independent jobs together. Each writes its own files; this is not a shared training model.
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        candidates = list(pool.map(candidate, [-2, -4, -8]))
    target_path = RESULTS / "eval_qrdqn_seed0_tail25.json"
    target = json.loads(target_path.read_text())["summary"][0]["mean_speed_m_s"]
    selected = choose_speed_targeted_baseline(candidates, target)
    selection = {
        "rule": "Closest normal-density seed-0 speed to QR-DQN lower25; ties prefer smaller penalty magnitude. Collision rate is not a selection input.",
        "target_speed_m_s": target,
        "candidates": candidates,
        "target_evaluation_sha256": hashlib.sha256(
            target_path.read_bytes()
        ).hexdigest(),
        "selected_collision_penalty": selected["collision_penalty"],
        "seed0_speed_difference_m_s": selected["speed_m_s"] - target,
        "confirmation_scenario_seeds_used": False,
    }
    selection_path = STAGE / "baseline_selection.json"
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise ValueError("Refusing to change the recorded baseline selection")
    save_json(selection_path, selection)
    penalty = selected["collision_penalty"]
    print(f"Selected collision penalty {penalty} by speed closeness only", flush=True)

    def replicate(seed):
        """Repeat the selected setup for another training seed, keeping its settings fixed."""
        checkpoint = ensure_training(penalty, seed)
        if penalty != -2:
            ensure_evaluation(penalty, seed, checkpoint)

    # Run independent jobs together. Each writes its own files; this is not a shared training model.
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(replicate, [1, 2]))
    for name, algorithm, alpha in [
        ("dqn", "dqn", 1.0),
        ("qrdqn_mean", "qrdqn", 1.0),
        ("qrdqn_tail25", "qrdqn", 0.25),
    ]:
        path = STAGE / f"diagnostic_{name}.json"
        if not path.exists():
            with (STAGE / f"diagnostic_{name}.log").open("w") as file:
                # Start another Python command and check whether it succeeded.
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "driving_rl.diagnostics",
                        "--checkpoint",
                        str(RESULTS / f"{algorithm}_seed0/final.pt"),
                        "--alpha",
                        str(alpha),
                        "--out",
                        str(path),
                    ],
                    stdout=file,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
    # Complete the exploratory control and regenerate the full current report.
    # Start another Python command and check whether it succeeded.
    subprocess.run([sys.executable, "run_speed_control.py"], check=True)
    print("control experiments complete: results/report/index.html", flush=True)


# Run the entry point only when this file is executed, not when it is imported.
if __name__ == "__main__":
    main()
