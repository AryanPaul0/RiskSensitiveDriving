"""Run the speed control experiment from Terminal.

This is an experiment coordinator, not the learning algorithm. It launches
separate Python commands, saves logs, and checks existing results before reuse.
Paths identify saved files; hashes are fingerprints that detect changed files.
Read main() for the order of operations. Learning itself lives in driving_rl/agents.py.

Original implementation notes:
Explore a predeclared ego-speed control grid on existing scalar checkpoints.
"""

import hashlib  # Make hashlib available for this file's calculations and workflow.
import json  # Make json available for this file's calculations and workflow.
from pathlib import Path  # Make Path available for this file's calculations and workflow.
import subprocess  # Make subprocess available for this file's calculations and workflow.
import sys  # Make sys available for this file's calculations and workflow.

from driving_rl.analysis import analyze  # Make analyze available for this file's calculations and workflow.
from driving_rl.experiment import save_json  # Make save_json available for this file's calculations and workflow.
from driving_rl.controls_report import create_report  # Make create_report available for this file's calculations and workflow.

ROOT = Path(__file__).resolve().parent  # Set ROOT to Path( file ).resolve().parent.
RESULTS = ROOT / "results"  # Set RESULTS to the path to 'results' inside ROOT.
STAGE = RESULTS / "controls"  # Set STAGE to the path to 'controls' inside RESULTS.


def evaluate(cap, seed):  # Run or verify one frozen-policy evaluation.
    """Run or verify one frozen-policy evaluation.
    Check its model fingerprint, settings and scenario block before reusing saved
    data. This function must not quietly mix results from different experiments.
    """
    checkpoint = RESULTS / f"dqn_seed{seed}/final.pt"  # Set checkpoint to (RESULTS divided by f'dqn seed{seed}/final.pt').
    path = STAGE / f"eval_dqn_cap{cap}_seed{seed}.json"  # Set path to (STAGE divided by f'eval dqn cap{cap} seed{seed}.json').
    if path.exists():  # Check existing data instead of silently replacing scientific results.
        data = json.loads(path.read_text())  # Set data to json.loads applied to the result of path.read text.
        metadata = data["provenance"]  # Set metadata to the entries of data selected by 'provenance'.
        if (  # Use this branch when the entries of metadata selected by 'speed_governor_cap_m_s' differs from cap or the entries of metadata selected by 'checkpoint_sha256' differs from the result of hashlib.sha256(checkpoint.read bytes()).hexdigest or the entries of metadata selected by 'episodes_per_density' differs from 30 or the entries of metadata selected by 'seed_start' differs from 10000 or the entries of metadata selected by 'alpha' differs from 1 or the entries of metadata selected by 'evaluation_collision_penalty' differs from negative 2 or the entries of row selected by 'density' for each row in the entries of data selected by 'summary' differs from 1.0, 1.5, 2.0.
            metadata["speed_governor_cap_m_s"] != cap  # Include the entries of metadata selected by 'speed_governor_cap_m_s' differs from cap or the entries of metadata selected by 'checkpoint_sha256' differs from the result of hashlib.sha256(checkpoint.read bytes()).hexdigest or the entries of metadata selected by 'episodes_per_density' differs from 30 or the entries of metadata selected by 'seed_start' differs from 10000 or the entries of metadata selected by 'alpha' differs from 1 or the entries of metadata selected by 'evaluation_collision_penalty' differs from negative 2 or the entries of row selected by 'density' for each row in the entries of data selected by 'summary' differs from 1.0, 1.5, 2.0 in the surrounding operation.
            or metadata["checkpoint_sha256"]  # Include the entries of metadata selected by 'checkpoint_sha256' differs from the result of hashlib.sha256(checkpoint.read bytes()).hexdigest in the surrounding operation.
            != hashlib.sha256(checkpoint.read_bytes()).hexdigest()  # Include the result of hashlib.sha256(checkpoint.read bytes()).hexdigest in the surrounding operation.
            or metadata["episodes_per_density"] != 30  # Include the entries of metadata selected by 'episodes_per_density' differs from 30 in the surrounding operation.
            or metadata["seed_start"] != 10000  # Include the entries of metadata selected by 'seed_start' differs from 10000 in the surrounding operation.
            or metadata["alpha"] != 1  # Include the entries of metadata selected by 'alpha' differs from 1 in the surrounding operation.
            or metadata["evaluation_collision_penalty"] != -2  # Include the entries of metadata selected by 'evaluation_collision_penalty' differs from negative 2 in the surrounding operation.
            or [row["density"] for row in data["summary"]] != [1.0, 1.5, 2.0]  # Include the entries of row selected by 'density' for each row in the entries of data selected by 'summary' differs from 1.0, 1.5, 2.0 in the surrounding operation.
        ):  # Finish assembling the grouped values above.
            raise ValueError(f"Refusing to reuse mismatched evaluation {path}")  # Reject this invalid case instead of continuing with misleading experiment results.
    else:  # Handle the alternative when the earlier condition (the result of path.exists) is false.
        gif = (  # Set gif to '--gif', (RESULTS divided by f'dqn cap{cap} seed{seed}.gif') as a text value when seed equals 0, otherwise .
            ["--gif", str(RESULTS / f"dqn_cap{cap}_seed{seed}.gif")]  # Include '--gif', (RESULTS divided by f'dqn cap{cap} seed{seed}.gif') as a text value when seed equals 0, otherwise in the surrounding operation.
            if seed == 0  # Include seed equals 0 in the surrounding operation.
            else []  # Include in the surrounding operation.
        )  # Finish assembling the grouped values above.
        with (STAGE / f"cap{cap}_seed{seed}.log").open("w") as file:  # Keep this temporary context active during the enclosed work, then clean it up.
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
                    "30",  # Include '30' in the surrounding operation.
                    "--out",  # Include '--out' in the surrounding operation.
                    str(path),  # Include path as a text value in the surrounding operation.
                    *gif,  # Include *gif in the surrounding operation.
                ],  # Finish assembling the grouped values above.
                stdout=file,  # Include file in the surrounding operation.
                stderr=subprocess.STDOUT,  # Include subprocess.STDOUT in the surrounding operation.
                check=True,  # Include True in the surrounding operation.
            )  # Finish assembling subprocess.run inputs.
        data = json.loads(path.read_text())  # Set data to json.loads applied to the result of path.read text.
    return path, data  # Return path, data.


def main():  # Read Terminal options, check settings, and run the requested workflow.
    """Read Terminal options, check settings, and run the requested workflow.
    Existing result files are checked before reuse so a rerun cannot silently mix
    different experiments. The main guard below prevents running jobs on import.
    """
    if Path.cwd() != ROOT:  # Use this branch when the result of Path.cwd differs from ROOT.
        raise ValueError("Run from the project directory")  # Reject this invalid case instead of continuing with misleading experiment results.
    candidates = []  # Set candidates to .
    for cap in [22, 24, 26]:  # Process each cap from 22, 24, 26.
        path, data = evaluate(cap, 0)  # Set path, data to evaluate applied to cap, 0.
        candidates.append(  # Add the named settings or measurements assembled below to candidates.
            {  # Include the named settings or measurements assembled below in the surrounding operation.
                "cap_m_s": cap,  # Record 'cap_m_s' as cap.
                "speed_m_s": data["summary"][0]["mean_speed_m_s"],  # Record 'speed_m_s' as the entries of the entries of the entries of data selected by 'summary' selected by 0 selected by 'mean_speed_m_s'.
                "evaluation_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),  # Record 'evaluation_sha256' as the result of hashlib.sha256(path.read bytes()).hexdigest.
            }  # Finish assembling the grouped values above.
        )  # Finish assembling candidates.append inputs.
    target = json.loads((RESULTS / "eval_qrdqn_seed0_tail25.json").read_text())[  # Set target to the entries of the entries of the entries of json.loads applied to (RESULTS / 'eval qrdqn seed0 tail25.json').read text() selected by 'summary' selected by 0 selected by 'mean_speed_m_s'.
        "summary"  # Include 'summary' in the surrounding operation.
    ][0]["mean_speed_m_s"]  # Finish assembling the grouped values above.
    selected = min(  # Set the selected next-action return predictions to min applied to candidates.
        candidates, key=lambda row: (abs(row["speed_m_s"] - target), row["cap_m_s"])  # Include candidates, lambda row: (abs(row['speed m s'] - target), row['cap m s']) in the surrounding operation.
    )  # Finish assembling min inputs.
    choice = {  # Set choice to the named settings or measurements assembled below.
        "rule": "Closest normal-density seed-0 speed; ties prefer lower cap. Exploratory development control.",  # Record 'rule' as the report text.
        "target_speed_m_s": target,  # Record 'target_speed_m_s' as target.
        "candidates": candidates,  # Record 'candidates' as candidates.
        "selected_cap_m_s": selected["cap_m_s"],  # Record 'selected_cap_m_s' as the entries of the selected next-action return predictions selected by 'cap_m_s'.
        "seed0_speed_difference_m_s": selected["speed_m_s"] - target,  # Record 'seed0_speed_difference_m_s' as (the entries of the selected next-action return predictions selected by 'speed_m_s' minus target).
        "confirmation_scenario_seeds_used": False,  # Record 'confirmation_scenario_seeds_used' as False.
    }  # Finish assembling the grouped values above.
    path = STAGE / "governor_selection.json"  # Set path to the path to 'governor_selection.json' inside STAGE.
    if path.exists() and json.loads(path.read_text()) != choice:  # Use this branch when the result of path.exists and json.loads applied to the result of path.read text differs from choice.
        raise ValueError("Refusing to change recorded governor selection")  # Reject this invalid case instead of continuing with misleading experiment results.
    save_json(path, choice)  # Save the following results/settings as json: path, choice.
    print(json.dumps(choice, indent=2), flush=True)  # Print applied to json.dumps applied to choice.
    for seed in [1, 2]:  # Process each seed from 1, 2.
        evaluate(selected["cap_m_s"], seed)  # Evaluate applied to the entries of the selected next-action return predictions selected by 'cap_m_s', seed.
    analyze(RESULTS)  # Analyze applied to results.
    create_report(RESULTS)  # Create report applied to results.


if __name__ == "__main__":  # Run this workflow when launched from Terminal; importing its helpers must not launch experiments.
    main()  # The result of main.
