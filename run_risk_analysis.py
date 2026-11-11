"""Run the risk analysis experiment from Terminal.

This is an experiment coordinator, not the learning algorithm. It launches
separate Python commands, saves logs, and checks existing results before reuse.
Paths identify saved files; hashes are fingerprints that detect changed files.
Read main() for the order of operations. Learning itself lives in driving_rl/agents.py.

Original implementation notes:
Reproduce same-checkpoint controls, risk sweep and calibration diagnostics.
"""

import argparse  # Make argparse available for this file's calculations and workflow.
from concurrent.futures import ThreadPoolExecutor  # Make ThreadPoolExecutor available for this file's calculations and workflow.
import hashlib  # Make hashlib available for this file's calculations and workflow.
import json  # Make json available for this file's calculations and workflow.
from pathlib import Path  # Make Path available for this file's calculations and workflow.
import subprocess  # Make subprocess available for this file's calculations and workflow.
import sys  # Make sys available for this file's calculations and workflow.

from driving_rl.experiment import save_json  # Make save_json available for this file's calculations and workflow.

ROOT = Path(__file__).resolve().parent  # Set ROOT to Path( file ).resolve().parent.
RESULTS = ROOT / "results"  # Set RESULTS to the path to 'results' inside ROOT.
STAGE = RESULTS / "risk_analysis"  # Set STAGE to the path to 'risk_analysis' inside RESULTS.
CAPS = [24.0, 24.5, 25.0, 25.5, 26.0, 26.5]  # Set CAPS to the listed settings or measurements in their recorded order.
ALPHAS = [0.125, 0.25, 0.5, 0.75, 1.0]  # Set ALPHAS to 0.125, 0.25, 0.5, 0.75, 1.0.


def token(value):  # Make a short filename-friendly label for a numeric setting.
    """Make a short filename-friendly label for a numeric setting."""  # Describe this operation for readers and help().
    return f"{value:g}".replace(".", "p")  # Return f'{value:g}'.replace applied to '.', 'p'.


def eval_path(seed, alpha=1.0, cap=None):  # Find the evaluation file corresponding to this seed and action-selection rule.
    """Find the evaluation file corresponding to this seed and action-selection rule."""  # Describe this operation for readers and help().
    if cap is None and alpha in [0.25, 1.0]:  # Use this branch when cap is None and the fraction of low-return predictions used to score actions belongs to 0.25, 1.0.
        suffix = "tail25" if alpha == 0.25 else "mean"  # Set suffix to 'tail25' when the fraction of low-return predictions used to score actions equals 0.25, otherwise 'mean'.
        return RESULTS / f"eval_qrdqn_seed{seed}_{suffix}.json"  # Return (RESULTS divided by f'eval qrdqn seed{seed} {suffix}.json').
    suffix = f"cap{token(cap)}" if cap is not None else f"alpha{token(alpha)}"  # Set suffix to f'cap{token(cap)}' when cap is not None, otherwise f'alpha{token(alpha)}'.
    return STAGE / f"eval_qrdqn_{suffix}_seed{seed}.json"  # Return (STAGE divided by f'eval qrdqn {suffix} seed{seed}.json').


def evaluate(spec):  # Run or verify one frozen-policy evaluation.
    """Run or verify one frozen-policy evaluation.
    Check its model fingerprint, settings and scenario block before reusing saved
    data. This function must not quietly mix results from different experiments.
    """
    seed, alpha, cap = spec  # Set seed, the fraction of low-return predictions used to score actions, cap to spec.
    checkpoint = RESULTS / f"qrdqn_seed{seed}/final.pt"  # Set checkpoint to (RESULTS divided by f'qrdqn seed{seed}/final.pt').
    output = eval_path(seed, alpha, cap)  # Set output to eval path applied to seed, the fraction of low-return predictions used to score actions, cap.
    if output.exists():  # Use this branch when the result of output.exists.
        data = json.loads(output.read_text())  # Set data to json.loads applied to the result of output.read text.
        p = data["provenance"]  # Set p to the entries of data selected by 'provenance'.
        if (  # Use this branch when the entries of p selected by 'checkpoint_sha256' differs from the result of hashlib.sha256(checkpoint.read bytes()).hexdigest or the entries of p selected by 'training_seed' differs from seed or the entries of p selected by 'training_steps' differs from 20000 or the entries of p selected by 'alpha' differs from the fraction of low-return predictions used to score actions or p.get applied to 'speed_governor_cap_m_s' differs from cap or the entries of p selected by 'seed_start' differs from 10000 or the entries of p selected by 'episodes_per_density' differs from 30 or the entries of p selected by 'algorithm' differs from 'qrdqn' or p.get applied to 'evaluation_collision_penalty', negative 2.0 differs from negative 2.0 or the entries of row selected by 'density' for each row in the entries of data selected by 'summary' differs from 1.0, 1.5, 2.0.
            p["checkpoint_sha256"]  # Include the entries of p selected by 'checkpoint_sha256' differs from the result of hashlib.sha256(checkpoint.read bytes()).hexdigest or the entries of p selected by 'training_seed' differs from seed or the entries of p selected by 'training_steps' differs from 20000 or the entries of p selected by 'alpha' differs from the fraction of low-return predictions used to score actions or p.get applied to 'speed_governor_cap_m_s' differs from cap or the entries of p selected by 'seed_start' differs from 10000 or the entries of p selected by 'episodes_per_density' differs from 30 or the entries of p selected by 'algorithm' differs from 'qrdqn' or p.get applied to 'evaluation_collision_penalty', negative 2.0 differs from negative 2.0 or the entries of row selected by 'density' for each row in the entries of data selected by 'summary' differs from 1.0, 1.5, 2.0 in the surrounding operation.
            != hashlib.sha256(checkpoint.read_bytes()).hexdigest()  # Include the result of hashlib.sha256(checkpoint.read bytes()).hexdigest in the surrounding operation.
            or p["training_seed"] != seed  # Include the entries of p selected by 'training_seed' differs from seed in the surrounding operation.
            or p["training_steps"] != 20000  # Include the entries of p selected by 'training_steps' differs from 20000 in the surrounding operation.
            or p["alpha"] != alpha  # Include the entries of p selected by 'alpha' differs from the fraction of low-return predictions used to score actions in the surrounding operation.
            or p.get("speed_governor_cap_m_s") != cap  # Include p.get applied to 'speed_governor_cap_m_s' differs from cap in the surrounding operation.
            or p["seed_start"] != 10000  # Include the entries of p selected by 'seed_start' differs from 10000 in the surrounding operation.
            or p["episodes_per_density"] != 30  # Include the entries of p selected by 'episodes_per_density' differs from 30 in the surrounding operation.
            or p["algorithm"] != "qrdqn"  # Include the entries of p selected by 'algorithm' differs from 'qrdqn' in the surrounding operation.
            or p.get("evaluation_collision_penalty", -2.0) != -2.0  # Include p.get applied to 'evaluation_collision_penalty', negative 2.0 differs from negative 2.0 in the surrounding operation.
            or [row["density"] for row in data["summary"]] != [1.0, 1.5, 2.0]  # Include the entries of row selected by 'density' for each row in the entries of data selected by 'summary' differs from 1.0, 1.5, 2.0 in the surrounding operation.
        ):  # Finish assembling the grouped values above.
            raise ValueError(f"Refusing to reuse mismatched evaluation: {output}")  # Reject this invalid case instead of continuing with misleading experiment results.
        return output  # Return output.
    arguments = [  # Set arguments to the listed settings or measurements in their recorded order.
        sys.executable,  # Include sys.executable in the surrounding operation.
        "-m",  # Include '-m' in the surrounding operation.
        "driving_rl.cli",  # Include 'driving_rl.cli' in the surrounding operation.
        "evaluate",  # Include 'evaluate' in the surrounding operation.
        "--checkpoint",  # Include '--checkpoint' in the surrounding operation.
        str(checkpoint),  # Include checkpoint as a text value in the surrounding operation.
        "--alpha",  # Include '--alpha' in the surrounding operation.
        str(alpha),  # Include the fraction of low-return predictions used to score actions as a text value in the surrounding operation.
        "--episodes",  # Include '--episodes' in the surrounding operation.
        "30",  # Include '30' in the surrounding operation.
        "--out",  # Include '--out' in the surrounding operation.
        str(output),  # Include output as a text value in the surrounding operation.
    ]  # Finish assembling the grouped values above.
    if cap is not None:  # Use this branch when cap is not None.
        arguments += ["--speed-cap", str(cap)]  # Increase arguments using '--speed-cap', cap as a text value.
    with output.with_suffix(".log").open("w") as log:  # Keep this temporary context active during the enclosed work, then clean it up.
        subprocess.run(arguments, stdout=log, stderr=subprocess.STDOUT, check=True)  # Start another Python command and check whether it succeeded.
    print(f"Evaluated {output.name}", flush=True)  # Print applied to f'evaluated {output.name}'.
    return output  # Return output.


def controls(jobs):  # Choose the speed control on the designated development model, freeze it, then evaluate other models without selecting their thresholds again. The threshold is chosen before looking at seeds 1 and 2 under this control.
    """Choose the speed control on the designated development model, freeze it,
    then evaluate other models without selecting their thresholds again.
    """
    with ThreadPoolExecutor(max_workers=jobs) as pool:  # Run independent jobs together. Each writes its own files; this is not a shared training model.
        paths = list(pool.map(evaluate, [(0, 1.0, cap) for cap in CAPS]))  # Set paths to list applied to pool.map applied to evaluate, 0, 1.0, cap for each cap in CAPS.
    target_path = RESULTS / "eval_qrdqn_seed0_tail25.json"  # Set target path to the path to 'eval_qrdqn_seed0_tail25.json' inside RESULTS.
    target = json.loads(target_path.read_text())["summary"][0]["mean_speed_m_s"]  # Set target to the entries of the entries of the entries of json.loads applied to target path.read text() selected by 'summary' selected by 0 selected by 'mean_speed_m_s'.
    candidates = []  # Set candidates to .
    for cap, path in zip(CAPS, paths):  # Process each cap, path from matching entries grouped across CAPS, paths.
        speed = json.loads(path.read_text())["summary"][0]["mean_speed_m_s"]  # Set speed to the entries of the entries of the entries of json.loads applied to path.read text() selected by 'summary' selected by 0 selected by 'mean_speed_m_s'.
        candidates.append(  # Add the named settings or measurements assembled below to candidates.
            {  # Include the named settings or measurements assembled below in the surrounding operation.
                "cap_m_s": cap,  # Record 'cap_m_s' as cap.
                "speed_m_s": speed,  # Record 'speed_m_s' as speed.
                "evaluation_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),  # Record 'evaluation_sha256' as the result of hashlib.sha256(path.read bytes()).hexdigest.
            }  # Finish assembling the grouped values above.
        )  # Finish assembling candidates.append inputs.
    chosen = min(  # Set chosen to min applied to candidates.
        candidates, key=lambda row: (abs(row["speed_m_s"] - target), row["cap_m_s"])  # Include candidates, lambda row: (abs(row['speed m s'] - target), row['cap m s']) in the surrounding operation.
    )  # Finish assembling min inputs.
    selection = {  # Set selection to the named settings or measurements assembled below.
        "rule": "Same QR-DQN checkpoint, seed-0 normal-density speed closeness only; ties prefer lower cap.",  # Record 'rule' as the report text.
        "target_speed_m_s": target,  # Record 'target_speed_m_s' as target.
        "candidates": candidates,  # Record 'candidates' as candidates.
        "selected_cap_m_s": chosen["cap_m_s"],  # Record 'selected_cap_m_s' as the entries of chosen selected by 'cap_m_s'.
        "seed0_speed_difference_m_s": chosen["speed_m_s"] - target,  # Record 'seed0_speed_difference_m_s' as (the entries of chosen selected by 'speed_m_s' minus target).
        "target_evaluation_sha256": hashlib.sha256(  # Record 'target_evaluation_sha256' as the result of hashlib.sha256(target path.read bytes()).hexdigest.
            target_path.read_bytes()  # Include the result of target path.read bytes in the surrounding operation.
        ).hexdigest(),  # Finish assembling hashlib.sha256(target path.read bytes()).hexdigest inputs.
        "confirmation_seeds_used": False,  # Record 'confirmation_seeds_used' as False.
    }  # Finish assembling the grouped values above.
    path = STAGE / "governor_selection.json"  # Set path to the path to 'governor_selection.json' inside STAGE.
    if path.exists() and json.loads(path.read_text()) != selection:  # Use this branch when the result of path.exists and json.loads applied to the result of path.read text differs from selection.
        raise ValueError(  # Reject this invalid case instead of continuing with misleading experiment results.
            "Refusing to change the recorded same-checkpoint control selection"  # Include the report text in the surrounding operation.
        )  # Finish assembling ValueError inputs.
    save_json(path, selection)  # Save the following results/settings as json: path, selection.
    print(f"Frozen same-checkpoint threshold {chosen['cap_m_s']} m/s", flush=True)  # Print applied to f"frozen same-checkpoint threshold {chosen['cap m s']} m/s".
    tasks = [(seed, 1.0, chosen["cap_m_s"]) for seed in [1, 2]]  # Set tasks to seed, 1.0, the entries of chosen selected by 'cap_m_s' for each seed in 1, 2.
    tasks += [(seed, alpha, None) for seed in [0, 1, 2] for alpha in ALPHAS]  # Increase tasks using seed, the fraction of low-return predictions used to score actions, None for each seed in 0, 1, 2.
    with ThreadPoolExecutor(max_workers=jobs) as pool:  # Run independent jobs together. Each writes its own files; this is not a shared training model.
        list(pool.map(evaluate, tasks))  # List applied to pool.map applied to evaluate, tasks.


def calibration_task(spec):  # Collect or validate the return-prediction checks for one model and policy.
    """Collect or validate the return-prediction checks for one model and policy."""  # Describe this operation for readers and help().
    seed, policy, cap = spec  # Set seed, policy, cap to spec.
    alpha = 0.25 if policy == "tail25" else 1.0  # Set the fraction of low-return predictions used to score actions to 0.25 when policy equals 'tail25', otherwise 1.0.
    checkpoint = RESULTS / f"qrdqn_seed{seed}/final.pt"  # Set checkpoint to (RESULTS divided by f'qrdqn seed{seed}/final.pt').
    path = STAGE / f"calibration_{policy}_seed{seed}.json"  # Set path to (STAGE divided by f'calibration {policy} seed{seed}.json').
    if path.exists():  # Check existing data instead of silently replacing scientific results.
        data = json.loads(path.read_text())["provenance"]  # Set data to the entries of json.loads applied to the result of path.read text selected by 'provenance'.
        if (  # Use this branch when the entries of data selected by 'checkpoint_sha256' differs from the result of hashlib.sha256(checkpoint.read bytes()).hexdigest or the entries of data selected by 'alpha' differs from the fraction of low-return predictions used to score actions or the entries of data selected by 'speed_cap' differs from cap or the entries of data selected by 'seed_start' differs from 12000 or the entries of data selected by 'episodes_per_density' differs from 60.
            data["checkpoint_sha256"]  # Include the entries of data selected by 'checkpoint_sha256' differs from the result of hashlib.sha256(checkpoint.read bytes()).hexdigest or the entries of data selected by 'alpha' differs from the fraction of low-return predictions used to score actions or the entries of data selected by 'speed_cap' differs from cap or the entries of data selected by 'seed_start' differs from 12000 or the entries of data selected by 'episodes_per_density' differs from 60 in the surrounding operation.
            != hashlib.sha256(checkpoint.read_bytes()).hexdigest()  # Include the result of hashlib.sha256(checkpoint.read bytes()).hexdigest in the surrounding operation.
            or data["alpha"] != alpha  # Include the entries of data selected by 'alpha' differs from the fraction of low-return predictions used to score actions in the surrounding operation.
            or data["speed_cap"] != cap  # Include the entries of data selected by 'speed_cap' differs from cap in the surrounding operation.
            or data["seed_start"] != 12000  # Include the entries of data selected by 'seed_start' differs from 12000 in the surrounding operation.
            or data["episodes_per_density"] != 60  # Include the entries of data selected by 'episodes_per_density' differs from 60 in the surrounding operation.
        ):  # Finish assembling the grouped values above.
            raise ValueError(f"Refusing to reuse mismatched calibration: {path}")  # Reject this invalid case instead of continuing with misleading experiment results.
        return  # Finish this operation without producing another result.
    arguments = [  # Set arguments to the listed settings or measurements in their recorded order.
        sys.executable,  # Include sys.executable in the surrounding operation.
        "-m",  # Include '-m' in the surrounding operation.
        "driving_rl.calibration",  # Include 'driving_rl.calibration' in the surrounding operation.
        "--checkpoint",  # Include '--checkpoint' in the surrounding operation.
        str(checkpoint),  # Include checkpoint as a text value in the surrounding operation.
        "--alpha",  # Include '--alpha' in the surrounding operation.
        str(alpha),  # Include the fraction of low-return predictions used to score actions as a text value in the surrounding operation.
        "--out",  # Include '--out' in the surrounding operation.
        str(path),  # Include path as a text value in the surrounding operation.
    ]  # Finish assembling the grouped values above.
    if cap is not None:  # Use this branch when cap is not None.
        arguments += ["--speed-cap", str(cap)]  # Increase arguments using '--speed-cap', cap as a text value.
    with path.with_suffix(".log").open("w") as log:  # Keep this temporary context active during the enclosed work, then clean it up.
        subprocess.run(arguments, stdout=log, stderr=subprocess.STDOUT, check=True)  # Start another Python command and check whether it succeeded.
    print(f"Calibrated {policy}, training seed {seed}", flush=True)  # Print applied to f'calibrated {policy}, training seed {seed}'.


def ensure_replay(cap):  # Create the predefined recorded driving example if it is not already saved.
    """Create the predefined recorded driving example if it is not already saved."""  # Describe this operation for readers and help().
    checkpoint = RESULTS / "qrdqn_seed0/final.pt"  # Set checkpoint to the path to 'qrdqn_seed0/final.pt' inside RESULTS.
    path = STAGE / "replay_governed.json"  # Set path to the path to 'replay_governed.json' inside STAGE.
    if path.exists():  # Check existing data instead of silently replacing scientific results.
        p = json.loads(path.read_text())["provenance"]  # Set p to the entries of json.loads applied to the result of path.read text selected by 'provenance'.
        if (  # Use this branch when the entries of p selected by 'checkpoint_sha256' differs from the result of hashlib.sha256(checkpoint.read bytes()).hexdigest or the entries of p selected by 'alpha' differs from 1.0 or the entries of p selected by 'speed_governor_cap_m_s' differs from cap or the entries of p selected by 'seed_start' differs from 10000 or the entries of p selected by 'episodes_per_density' differs from 1.
            p["checkpoint_sha256"]  # Include the entries of p selected by 'checkpoint_sha256' differs from the result of hashlib.sha256(checkpoint.read bytes()).hexdigest or the entries of p selected by 'alpha' differs from 1.0 or the entries of p selected by 'speed_governor_cap_m_s' differs from cap or the entries of p selected by 'seed_start' differs from 10000 or the entries of p selected by 'episodes_per_density' differs from 1 in the surrounding operation.
            != hashlib.sha256(checkpoint.read_bytes()).hexdigest()  # Include the result of hashlib.sha256(checkpoint.read bytes()).hexdigest in the surrounding operation.
            or p["alpha"] != 1.0  # Include the entries of p selected by 'alpha' differs from 1.0 in the surrounding operation.
            or p["speed_governor_cap_m_s"] != cap  # Include the entries of p selected by 'speed_governor_cap_m_s' differs from cap in the surrounding operation.
            or p["seed_start"] != 10000  # Include the entries of p selected by 'seed_start' differs from 10000 in the surrounding operation.
            or p["episodes_per_density"] != 1  # Include the entries of p selected by 'episodes_per_density' differs from 1 in the surrounding operation.
        ):  # Finish assembling the grouped values above.
            raise ValueError("Existing replay does not match the selected control")  # Reject this invalid case instead of continuing with misleading experiment results.
        if not (STAGE / "replay_governed.gif").exists():  # Use this branch when not the result of (STAGE / 'replay governed.gif').exists.
            raise ValueError(  # Reject this invalid case instead of continuing with misleading experiment results.
                "Replay JSON exists without its GIF; preserve it and regenerate in a new path"  # Include the report text in the surrounding operation.
            )  # Finish assembling ValueError inputs.
        return  # Finish this operation without producing another result.
    with (STAGE / "replay_governed.log").open("w") as log:  # Keep this temporary context active during the enclosed work, then clean it up.
        subprocess.run(  # Start another Python command and check whether it succeeded.
            [  # Include the listed settings or measurements in their recorded order in the surrounding operation.
                sys.executable,  # Include sys.executable in the surrounding operation.
                "-m",  # Include '-m' in the surrounding operation.
                "driving_rl.cli",  # Include 'driving_rl.cli' in the surrounding operation.
                "evaluate",  # Include 'evaluate' in the surrounding operation.
                "--checkpoint",  # Include '--checkpoint' in the surrounding operation.
                str(checkpoint),  # Include checkpoint as a text value in the surrounding operation.
                "--speed-cap",  # Include '--speed-cap' in the surrounding operation.
                str(cap),  # Include cap as a text value in the surrounding operation.
                "--episodes",  # Include '--episodes' in the surrounding operation.
                "1",  # Include '1' in the surrounding operation.
                "--densities",  # Include '--densities' in the surrounding operation.
                "1",  # Include '1' in the surrounding operation.
                "--out",  # Include '--out' in the surrounding operation.
                str(path),  # Include path as a text value in the surrounding operation.
                "--gif",  # Include '--gif' in the surrounding operation.
                str(STAGE / "replay_governed.gif"),  # Include the path to 'replay_governed.gif' inside STAGE as a text value in the surrounding operation.
            ],  # Finish assembling the grouped values above.
            stdout=log,  # Include log in the surrounding operation.
            stderr=subprocess.STDOUT,  # Include subprocess.STDOUT in the surrounding operation.
            check=True,  # Include True in the surrounding operation.
        )  # Finish assembling subprocess.run inputs.


def main():  # Read Terminal options, check settings, and run the requested workflow.
    """Read Terminal options, check settings, and run the requested workflow.
    Existing result files are checked before reuse so a rerun cannot silently mix
    different experiments. The main guard below prevents running jobs on import.
    """
    parser = argparse.ArgumentParser(description=__doc__)  # Set parser to the result of argparse.ArgumentParser.
    parser.add_argument("--jobs", type=int, default=3)  # Parser.add argument applied to '--jobs'.
    parser.add_argument(  # Parser.add argument applied to '--stage'.
        "--stage", choices=["controls", "calibration", "all"], default="all"  # Include '--stage', 'controls', 'calibration', 'all', 'all' in the surrounding operation.
    )  # Finish assembling parser.add argument inputs.
    args = parser.parse_args()  # Set args to the result of parser.parse args.
    if Path.cwd() != ROOT or not 1 <= args.jobs <= 4:  # Use this branch when the result of Path.cwd differs from ROOT or not 1 is at most args.jobs is at most 4.
        parser.error("Run from the project directory with --jobs between 1 and 4")  # Parser.error applied to 'run from the project directory with --jobs between 1 and 4'.
    STAGE.mkdir(parents=True, exist_ok=True)  # The result of stage.mkdir.
    if args.stage in ["controls", "all"]:  # Use this branch when args.stage belongs to 'controls', 'all'.
        controls(args.jobs)  # Controls applied to args.jobs.
    if args.stage in ["calibration", "all"]:  # Use this branch when args.stage belongs to 'calibration', 'all'.
        cap = json.loads((STAGE / "governor_selection.json").read_text())[  # Set cap to the entries of json.loads applied to the result of (STAGE / 'governor selection.json').read text selected by 'selected_cap_m_s'.
            "selected_cap_m_s"  # Include 'selected_cap_m_s' in the surrounding operation.
        ]  # Finish assembling the grouped values above.
        tasks = [  # Set tasks to seed, policy, cap when policy equals 'governed', otherwise None for each seed in 0, 1, 2.
            (seed, policy, cap if policy == "governed" else None)  # Include seed, policy, cap when policy equals 'governed', otherwise None in the surrounding operation.
            for seed in [0, 1, 2]  # Include seed, 0, 1, 2 in the surrounding operation.
            for policy in ["mean", "tail25", "governed"]  # Include policy, 'mean', 'tail25', 'governed' in the surrounding operation.
        ]  # Finish assembling the grouped values above.
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:  # Run independent jobs together. Each writes its own files; this is not a shared training model.
            list(pool.map(calibration_task, tasks))  # List applied to pool.map applied to calibration task, tasks.
        ensure_replay(cap)  # Ensure replay applied to cap.
        from driving_rl.risk_analysis_analysis import analyze  # Make analyze available for this file's calculations and workflow.
        from driving_rl.risk_analysis_report import create_report  # Make create_report available for this file's calculations and workflow.

        analyze(RESULTS)  # Analyze applied to results.
        create_report(RESULTS)  # Create report applied to results.


if __name__ == "__main__":  # Run this workflow when launched from Terminal; importing its helpers must not launch experiments.
    main()  # The result of main.
