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

import argparse  # Make argparse available for this file's calculations and workflow.
from concurrent.futures import ThreadPoolExecutor  # Make ThreadPoolExecutor available for this file's calculations and workflow.
import hashlib  # Make hashlib available for this file's calculations and workflow.
import json  # Make json available for this file's calculations and workflow.
from pathlib import Path  # Make Path available for this file's calculations and workflow.
import subprocess  # Make subprocess available for this file's calculations and workflow.
import sys  # Make sys available for this file's calculations and workflow.

from driving_rl.analysis import choose_speed_targeted_baseline  # Make choose_speed_targeted_baseline available for this file's calculations and workflow.
from driving_rl.experiment import save_json  # Make save_json available for this file's calculations and workflow.

ROOT = Path(__file__).resolve().parent  # Set ROOT to Path( file ).resolve().parent.
RESULTS = ROOT / "results"  # Set RESULTS to the path to 'results' inside ROOT.
STAGE = RESULTS / "controls"  # Set STAGE to the path to 'controls' inside RESULTS.


def execute(arguments, log):  # Run the requested command with this project's Python interpreter.
    """Run the requested command with this project's Python interpreter.
    Save its console messages in a log so failures can be inspected later.
    """
    with Path(log).open("w") as file:  # Keep this temporary context active during the enclosed work, then clean it up.
        subprocess.run(  # Start another Python command and check whether it succeeded.
            [sys.executable, "-m", "driving_rl.cli", *map(str, arguments)],  # Include sys.executable, '-m', 'driving_rl.cli', *map(str, arguments) in the surrounding operation.
            stdout=file,  # Include file in the surrounding operation.
            stderr=subprocess.STDOUT,  # Include subprocess.STDOUT in the surrounding operation.
            check=True,  # Include True in the surrounding operation.
        )  # Finish assembling subprocess.run inputs.


def ensure_training(penalty, seed):  # Train a missing model, or check that the saved run really matches the requested settings.
    """Train a missing model, or check that the saved run really matches the requested settings."""  # Describe this operation for readers and help().
    directory = RESULTS / f"dqn_penalty{abs(int(penalty))}_seed{seed}"  # Set directory to (RESULTS divided by f'dqn penalty{abs(int(penalty))} seed{seed}').
    if penalty == -2:  # Use this branch when penalty equals negative 2.
        directory = RESULTS / f"dqn_seed{seed}"  # Set directory to (RESULTS divided by f'dqn seed{seed}').
    checkpoint = directory / "final.pt"  # Set checkpoint to the path to 'final.pt' inside directory.
    if checkpoint.exists():  # Use this branch when the result of checkpoint.exists.
        config = json.loads((directory / "config.json").read_text())  # Set config to json.loads applied to the result of (directory / 'config.json').read text.
        summary = json.loads((directory / "summary.json").read_text())  # Set summary to json.loads applied to the result of (directory / 'summary.json').read text.
        expected = {  # Set expected to the named settings or measurements assembled below.
            "steps": 20000,  # Record 'steps' as 20000.
            "seed": seed,  # Record 'seed' as seed.
            "algorithm": "dqn",  # Record 'algorithm' as 'dqn'.
            "gamma": 0.9,  # Record 'gamma' as 0.9.
            "learning_rate": 0.0005,  # Record 'learning_rate' as 0.0005.
            "batch_size": 64,  # Record 'batch_size' as 64.
            "target_interval": 250,  # Record 'target_interval' as 250.
        }  # Finish assembling the grouped values above.
        if (  # Use this branch when not the entries of summary selected by 'training_complete' or config.get applied to 'collision_penalty', negative 2.0 differs from penalty or any applied to config[key] differs from value for each key, value in the result of expected.items.
            not summary["training_complete"]  # Include not the entries of summary selected by 'training_complete' or config.get applied to 'collision_penalty', negative 2.0 differs from penalty or any applied to config[key] differs from value for each key, value in the result of expected.items in the surrounding operation.
            or config.get("collision_penalty", -2.0) != penalty  # Include config.get applied to 'collision_penalty', negative 2.0 differs from penalty in the surrounding operation.
            or any(config[key] != value for key, value in expected.items())  # Include any applied to the entries of config selected by key differs from value for each key, value in the result of expected.items in the surrounding operation.
        ):  # Finish assembling the grouped values above.
            raise ValueError(f"Existing training settings do not match: {directory}")  # Reject this invalid case instead of continuing with misleading experiment results.
        print(f"Reusing training {directory.name}", flush=True)  # Print applied to f'reusing training {directory.name}'.
    else:  # Handle the alternative when the earlier condition (the result of checkpoint.exists) is false.
        execute(  # Execute applied to the listed settings or measurements in their recorded order, (results divided by f'{directory.name}.log').
            [  # Include the listed settings or measurements in their recorded order in the surrounding operation.
                "train",  # Include 'train' in the surrounding operation.
                "--algorithm",  # Include '--algorithm' in the surrounding operation.
                "dqn",  # Include 'dqn' in the surrounding operation.
                "--collision-penalty",  # Include '--collision-penalty' in the surrounding operation.
                penalty,  # Include penalty in the surrounding operation.
                "--seed",  # Include '--seed' in the surrounding operation.
                seed,  # Include seed in the surrounding operation.
                "--steps",  # Include '--steps' in the surrounding operation.
                20000,  # Include 20000 in the surrounding operation.
                "--out",  # Include '--out' in the surrounding operation.
                directory,  # Include directory in the surrounding operation.
            ],  # Finish assembling the grouped values above.
            RESULTS / f"{directory.name}.log",  # Include (RESULTS divided by f'{directory.name}.log') in the surrounding operation.
        )  # Finish assembling execute inputs.
    return checkpoint  # Return checkpoint.


def ensure_evaluation(penalty, seed, checkpoint):  # Evaluate a missing result, or verify that saved results belong to the correct model and settings.
    """Evaluate a missing result, or verify that saved results belong to the correct model and settings."""  # Describe this operation for readers and help().
    output = STAGE / f"eval_dqn_penalty{abs(int(penalty))}_seed{seed}.json"  # Set output to (STAGE divided by f'eval dqn penalty{abs(int(penalty))} seed{seed}.json').
    if output.exists():  # Use this branch when the result of output.exists.
        data = json.loads(output.read_text())  # Set data to json.loads applied to the result of output.read text.
        metadata = data["provenance"]  # Set metadata to the entries of data selected by 'provenance'.
        if (  # Use this branch when the entries of metadata selected by 'checkpoint_sha256' differs from the result of hashlib.sha256(checkpoint.read bytes()).hexdigest or the entries of metadata selected by 'training_collision_penalty' differs from penalty or the entries of metadata selected by 'evaluation_collision_penalty' differs from negative 2.0 or the entries of metadata selected by 'seed_start' differs from 10000 or the entries of metadata selected by 'episodes_per_density' differs from 30 or the entries of metadata selected by 'alpha' differs from 1.0 or metadata.get applied to 'speed_governor_cap_m_s' is not None or the entries of row selected by 'density' for each row in the entries of data selected by 'summary' differs from 1.0, 1.5, 2.0.
            metadata["checkpoint_sha256"]  # Include the entries of metadata selected by 'checkpoint_sha256' differs from the result of hashlib.sha256(checkpoint.read bytes()).hexdigest or the entries of metadata selected by 'training_collision_penalty' differs from penalty or the entries of metadata selected by 'evaluation_collision_penalty' differs from negative 2.0 or the entries of metadata selected by 'seed_start' differs from 10000 or the entries of metadata selected by 'episodes_per_density' differs from 30 or the entries of metadata selected by 'alpha' differs from 1.0 or metadata.get applied to 'speed_governor_cap_m_s' is not None or the entries of row selected by 'density' for each row in the entries of data selected by 'summary' differs from 1.0, 1.5, 2.0 in the surrounding operation.
            != hashlib.sha256(checkpoint.read_bytes()).hexdigest()  # Include the result of hashlib.sha256(checkpoint.read bytes()).hexdigest in the surrounding operation.
            or metadata["training_collision_penalty"] != penalty  # Include the entries of metadata selected by 'training_collision_penalty' differs from penalty in the surrounding operation.
            or metadata["evaluation_collision_penalty"] != -2.0  # Include the entries of metadata selected by 'evaluation_collision_penalty' differs from negative 2.0 in the surrounding operation.
            or metadata["seed_start"] != 10000  # Include the entries of metadata selected by 'seed_start' differs from 10000 in the surrounding operation.
            or metadata["episodes_per_density"] != 30  # Include the entries of metadata selected by 'episodes_per_density' differs from 30 in the surrounding operation.
            or metadata["alpha"] != 1.0  # Include the entries of metadata selected by 'alpha' differs from 1.0 in the surrounding operation.
            or metadata.get("speed_governor_cap_m_s") is not None  # Include metadata.get applied to 'speed_governor_cap_m_s' is not None in the surrounding operation.
            or [row["density"] for row in data["summary"]] != [1.0, 1.5, 2.0]  # Include the entries of row selected by 'density' for each row in the entries of data selected by 'summary' differs from 1.0, 1.5, 2.0 in the surrounding operation.
        ):  # Finish assembling the grouped values above.
            raise ValueError(f"Existing evaluation settings do not match: {output}")  # Reject this invalid case instead of continuing with misleading experiment results.
    else:  # Handle the alternative when the earlier condition (the result of output.exists) is false.
        gif = (  # Set gif to '--gif', (RESULTS divided by f'dqn penalty{abs(int(penalty))} seed{seed}.gif') when seed equals 0, otherwise .
            ["--gif", RESULTS / f"dqn_penalty{abs(int(penalty))}_seed{seed}.gif"]  # Include '--gif', (RESULTS divided by f'dqn penalty{abs(int(penalty))} seed{seed}.gif') when seed equals 0, otherwise in the surrounding operation.
            if seed == 0  # Include seed equals 0 in the surrounding operation.
            else []  # Include in the surrounding operation.
        )  # Finish assembling the grouped values above.
        execute(  # Execute applied to the listed settings or measurements in their recorded order, (stage divided by f'evaluation penalty{abs(int(penalty))} seed{seed}.log').
            [  # Include the listed settings or measurements in their recorded order in the surrounding operation.
                "evaluate",  # Include 'evaluate' in the surrounding operation.
                "--checkpoint",  # Include '--checkpoint' in the surrounding operation.
                checkpoint,  # Include checkpoint in the surrounding operation.
                "--episodes",  # Include '--episodes' in the surrounding operation.
                30,  # Include 30 in the surrounding operation.
                "--out",  # Include '--out' in the surrounding operation.
                output,  # Include output in the surrounding operation.
                *gif,  # Include *gif in the surrounding operation.
            ],  # Finish assembling the grouped values above.
            STAGE / f"evaluation_penalty{abs(int(penalty))}_seed{seed}.log",  # Include (STAGE divided by f'evaluation penalty{abs(int(penalty))} seed{seed}.log') in the surrounding operation.
        )  # Finish assembling execute inputs.
    return output  # Return output.


def candidate(penalty):  # Read one speed-control candidate and record its measured speed and file fingerprint.
    """Read one speed-control candidate and record its measured speed and file fingerprint."""  # Describe this operation for readers and help().
    checkpoint = ensure_training(penalty, 0)  # Set checkpoint to ensure training applied to penalty, 0.
    path = (  # Set path to the path to 'eval_dqn_seed0_mean.json' inside RESULTS when penalty equals negative 2, otherwise ensure evaluation applied to penalty, 0, checkpoint.
        RESULTS / "eval_dqn_seed0_mean.json"  # Include the path to 'eval_dqn_seed0_mean.json' inside RESULTS when penalty equals negative 2, otherwise ensure evaluation applied to penalty, 0, checkpoint in the surrounding operation.
        if penalty == -2  # Include penalty equals negative 2 in the surrounding operation.
        else ensure_evaluation(penalty, 0, checkpoint)  # Include ensure evaluation applied to penalty, 0, checkpoint in the surrounding operation.
    )  # Finish assembling the grouped values above.
    data = json.loads(path.read_text())  # Set data to json.loads applied to the result of path.read text.
    normal = next(row for row in data["summary"] if row["density"] == 1.0)  # Set normal to next applied to row for each row in the entries of data selected by 'summary', keeping only cases where the entries of row selected by 'density' equals 1.0.
    return {  # Return the named settings or measurements assembled below.
        "collision_penalty": penalty,  # Record 'collision_penalty' as penalty.
        "speed_m_s": normal["mean_speed_m_s"],  # Record 'speed_m_s' as the entries of normal selected by 'mean_speed_m_s'.
        "evaluation_path": str(path.relative_to(ROOT)),  # Record 'evaluation_path' as path.relative to applied to ROOT as a text value.
        "evaluation_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),  # Record 'evaluation_sha256' as the result of hashlib.sha256(path.read bytes()).hexdigest.
    }  # Finish assembling the grouped values above.


def main():  # Read Terminal options, check settings, and run the requested workflow.
    """Read Terminal options, check settings, and run the requested workflow.
    Existing result files are checked before reuse so a rerun cannot silently mix
    different experiments. The main guard below prevents running jobs on import.
    """
    parser = argparse.ArgumentParser(description=__doc__)  # Set parser to the result of argparse.ArgumentParser.
    parser.add_argument("--jobs", type=int, default=2)  # Parser.add argument applied to '--jobs'.
    args = parser.parse_args()  # Set args to the result of parser.parse args.
    if Path.cwd() != ROOT or not 1 <= args.jobs <= 4:  # Use this branch when the result of Path.cwd differs from ROOT or not 1 is at most args.jobs is at most 4.
        parser.error("Run from the project directory; --jobs must be between 1 and 4.")  # Parser.error applied to 'run from the project directory; --jobs must be between 1 and 4.'.
    STAGE.mkdir(parents=True, exist_ok=True)  # The result of stage.mkdir.
    with (STAGE / "core_suite.log").open("w") as file:  # Keep this temporary context active during the enclosed work, then clean it up.
        subprocess.run(  # Start another Python command and check whether it succeeded.
            [  # Include the listed settings or measurements in their recorded order in the surrounding operation.
                sys.executable,  # Include sys.executable in the surrounding operation.
                "run_experiments.py",  # Include 'run_experiments.py' in the surrounding operation.
                "--seeds",  # Include '--seeds' in the surrounding operation.
                "0",  # Include '0' in the surrounding operation.
                "1",  # Include '1' in the surrounding operation.
                "2",  # Include '2' in the surrounding operation.
                "--steps",  # Include '--steps' in the surrounding operation.
                "20000",  # Include '20000' in the surrounding operation.
                "--episodes",  # Include '--episodes' in the surrounding operation.
                "30",  # Include '30' in the surrounding operation.
            ],  # Finish assembling the grouped values above.
            stdout=file,  # Include file in the surrounding operation.
            stderr=subprocess.STDOUT,  # Include subprocess.STDOUT in the surrounding operation.
            check=True,  # Include True in the surrounding operation.
        )  # Finish assembling subprocess.run inputs.
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:  # Run independent jobs together. Each writes its own files; this is not a shared training model.
        candidates = list(pool.map(candidate, [-2, -4, -8]))  # Set candidates to list applied to pool.map applied to candidate, negative 2, negative 4, negative 8.
    target_path = RESULTS / "eval_qrdqn_seed0_tail25.json"  # Set target path to the path to 'eval_qrdqn_seed0_tail25.json' inside RESULTS.
    target = json.loads(target_path.read_text())["summary"][0]["mean_speed_m_s"]  # Set target to the entries of the entries of the entries of json.loads applied to target path.read text() selected by 'summary' selected by 0 selected by 'mean_speed_m_s'.
    selected = choose_speed_targeted_baseline(candidates, target)  # Set the selected next-action return predictions to choose speed targeted baseline applied to candidates, target.
    selection = {  # Set selection to the named settings or measurements assembled below.
        "rule": "Closest normal-density seed-0 speed to QR-DQN lower25; ties prefer smaller penalty magnitude. Collision rate is not a selection input.",  # Record 'rule' as the report text.
        "target_speed_m_s": target,  # Record 'target_speed_m_s' as target.
        "candidates": candidates,  # Record 'candidates' as candidates.
        "target_evaluation_sha256": hashlib.sha256(  # Record 'target_evaluation_sha256' as the result of hashlib.sha256(target path.read bytes()).hexdigest.
            target_path.read_bytes()  # Include the result of target path.read bytes in the surrounding operation.
        ).hexdigest(),  # Finish assembling hashlib.sha256(target path.read bytes()).hexdigest inputs.
        "selected_collision_penalty": selected["collision_penalty"],  # Record 'selected_collision_penalty' as the entries of the selected next-action return predictions selected by 'collision_penalty'.
        "seed0_speed_difference_m_s": selected["speed_m_s"] - target,  # Record 'seed0_speed_difference_m_s' as (the entries of the selected next-action return predictions selected by 'speed_m_s' minus target).
        "confirmation_scenario_seeds_used": False,  # Record 'confirmation_scenario_seeds_used' as False.
    }  # Finish assembling the grouped values above.
    selection_path = STAGE / "baseline_selection.json"  # Set selection path to the path to 'baseline_selection.json' inside STAGE.
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:  # Use this branch when the result of selection path.exists and json.loads applied to the result of selection path.read text differs from selection.
        raise ValueError("Refusing to change the recorded baseline selection")  # Reject this invalid case instead of continuing with misleading experiment results.
    save_json(selection_path, selection)  # Save the following results/settings as json: selection path, selection.
    penalty = selected["collision_penalty"]  # Set penalty to the entries of the selected next-action return predictions selected by 'collision_penalty'.
    print(f"Selected collision penalty {penalty} by speed closeness only", flush=True)  # Print applied to f'selected collision penalty {penalty} by speed closeness only'.

    def replicate(seed):  # Repeat the selected setup for another training seed, keeping its settings fixed.
        """Repeat the selected setup for another training seed, keeping its settings fixed."""  # Describe this operation for readers and help().
        checkpoint = ensure_training(penalty, seed)  # Set checkpoint to ensure training applied to penalty, seed.
        if penalty != -2:  # Use this branch when penalty differs from negative 2.
            ensure_evaluation(penalty, seed, checkpoint)  # Ensure evaluation applied to penalty, seed, checkpoint.

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:  # Run independent jobs together. Each writes its own files; this is not a shared training model.
        list(pool.map(replicate, [1, 2]))  # List applied to pool.map applied to replicate, 1, 2.
    for name, algorithm, alpha in [  # Process each name, algorithm, the fraction of low-return predictions used to score actions from 'dqn', 'dqn', 1.0, 'qrdqn_mean', 'qrdqn', 1.0, 'qrdqn_tail25', 'qrdqn', 0.25.
        ("dqn", "dqn", 1.0),  # Include 'dqn', 'dqn', 1.0 in the surrounding operation.
        ("qrdqn_mean", "qrdqn", 1.0),  # Include 'qrdqn_mean', 'qrdqn', 1.0 in the surrounding operation.
        ("qrdqn_tail25", "qrdqn", 0.25),  # Include 'qrdqn_tail25', 'qrdqn', 0.25 in the surrounding operation.
    ]:  # Finish assembling the grouped values above.
        path = STAGE / f"diagnostic_{name}.json"  # Set path to (STAGE divided by f'diagnostic {name}.json').
        if not path.exists():  # Use this branch when not the result of path.exists.
            with (STAGE / f"diagnostic_{name}.log").open("w") as file:  # Keep this temporary context active during the enclosed work, then clean it up.
                subprocess.run(  # Start another Python command and check whether it succeeded.
                    [  # Include the listed settings or measurements in their recorded order in the surrounding operation.
                        sys.executable,  # Include sys.executable in the surrounding operation.
                        "-m",  # Include '-m' in the surrounding operation.
                        "driving_rl.diagnostics",  # Include 'driving_rl.diagnostics' in the surrounding operation.
                        "--checkpoint",  # Include '--checkpoint' in the surrounding operation.
                        str(RESULTS / f"{algorithm}_seed0/final.pt"),  # Include (RESULTS divided by f'{algorithm} seed0/final.pt') as a text value in the surrounding operation.
                        "--alpha",  # Include '--alpha' in the surrounding operation.
                        str(alpha),  # Include the fraction of low-return predictions used to score actions as a text value in the surrounding operation.
                        "--out",  # Include '--out' in the surrounding operation.
                        str(path),  # Include path as a text value in the surrounding operation.
                    ],  # Finish assembling the grouped values above.
                    stdout=file,  # Include file in the surrounding operation.
                    stderr=subprocess.STDOUT,  # Include subprocess.STDOUT in the surrounding operation.
                    check=True,  # Include True in the surrounding operation.
                )  # Finish assembling subprocess.run inputs.
    subprocess.run([sys.executable, "run_speed_control.py"], check=True)  # Complete the exploratory control and regenerate the full current report. Start another Python command and check whether it succeeded.
    print("control experiments complete: results/report/index.html", flush=True)  # Print applied to 'control experiments complete: results/report/index.html'.


if __name__ == "__main__":  # Run this workflow when launched from Terminal; importing its helpers must not launch experiments.
    main()  # The result of main.
