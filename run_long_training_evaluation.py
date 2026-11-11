"""Compare completed 40K and 100K models without training or tuning anything.

Every model gets the same fresh scenarios. Earlier confirmation cases are kept
separate. Run this file again to check and reuse saved evaluations.
"""

import argparse  # Make argparse available for this file's calculations and workflow.
from concurrent.futures import ThreadPoolExecutor  # Make ThreadPoolExecutor available for this file's calculations and workflow.
import hashlib  # Make hashlib available for this file's calculations and workflow.
import json  # Make json available for this file's calculations and workflow.
from pathlib import Path  # Make Path available for this file's calculations and workflow.
import subprocess  # Make subprocess available for this file's calculations and workflow.
import sys  # Make sys available for this file's calculations and workflow.

from driving_rl.experiment import save_json  # Make save_json available for this file's calculations and workflow.
from run_confirmation import POLICIES  # Make POLICIES available for this file's calculations and workflow.

ROOT = Path(__file__).resolve().parent  # Set ROOT to Path( file ).resolve().parent.
OUTPUT = ROOT / "results/long_training_evaluation"  # Set OUTPUT to the path to 'results/long_training_evaluation' inside ROOT.
DENSITIES = [1.0, 1.5, 2.0, 2.5]  # Set DENSITIES to 1.0, 1.5, 2.0, 2.5.


def read(path):  # Read settings or results that have already been saved.
    """Read settings or results that have already been saved."""  # Describe this operation for readers and help().
    return json.loads(path.read_text())  # Return json.loads applied to the result of path.read text.


def fingerprint(path):  # Identify the exact file, not just a model with a similar name.
    """Identify the exact file, not just a model with a similar name."""  # Describe this operation for readers and help().
    return hashlib.sha256(path.read_bytes()).hexdigest()  # Return the result of hashlib.sha256(path.read bytes()).hexdigest.


def model_path(budget, policy, seed):  # The policy chooses a decision rule; several rules share one network.
    """The policy chooses a decision rule; several rules share one network."""  # Describe this operation for readers and help().
    algorithm = POLICIES[policy][0]  # Set algorithm to the entries of the entries of POLICIES selected by policy selected by 0.
    if budget == 100000:  # Use this branch when the environment-interaction budget equals 100000.
        folder = "long_training_100k"  # Set folder to 'long_training_100k'.
    elif algorithm == "dqn":  # Use this branch when algorithm equals 'dqn'.
        folder = "baseline_controls"  # Set folder to 'baseline_controls'.
    else:  # Handle the alternative when the earlier condition (algorithm equals 'dqn') is false.
        folder = "density_coverage"  # Set folder to 'density_coverage'.
    return ROOT / f"results/{folder}/{algorithm}_seed{seed}/final.pt"  # Return (ROOT divided by f'results/{folder}/{algorithm} seed{seed}/final.pt').


def lock_design():  # Fix settings and model fingerprints before reading new outcomes.
    """Fix settings and model fingerprints before reading new outcomes."""  # Describe this operation for readers and help().
    OUTPUT.mkdir(parents=True, exist_ok=True)  # The result of output.mkdir.
    models = {}  # Set models to the named settings or measurements assembled below.
    for budget in [40000, 100000]:  # Each budget gets its own results. Keep 40K and 100K separate until the paired comparison.
        models[str(budget)] = {}  # Set the entries of models selected by str(budget) to the named settings or measurements assembled below.
        for policy in POLICIES:  # A policy is a decision rule. Several rules can share the very same QR-DQN network.
            models[str(budget)][policy] = {}  # Set the entries of the entries of models selected by str(budget) selected by policy to the named settings or measurements assembled below.
            for seed in range(3):  # A training seed identifies one independently trained model, not a new traffic scenario.
                path = model_path(budget, policy, seed)  # Set path to model path applied to the environment-interaction budget, policy, seed.
                summary = read(path.parent / "summary.json")  # Set summary to read applied to the path to 'summary.json' inside path.parent.
                if summary["steps"] != budget or not summary["training_complete"]:  # Use this branch when the entries of summary selected by 'steps' differs from the environment-interaction budget or not the entries of summary selected by 'training_complete'.
                    raise ValueError("A requested training run is incomplete")  # Reject this invalid case instead of continuing with misleading experiment results.
                models[str(budget)][policy][str(seed)] = fingerprint(path)  # Set the entries of the entries of the entries of models selected by str(budget) selected by policy selected by str(seed) to fingerprint applied to path.
    design = {  # Set design to the named settings or measurements assembled below.
        "budgets": [40000, 100000],  # Record 'budgets' as 40000, 100000.
        "model_hashes": models,  # Record 'model_hashes' as models.
        "scenario_seed_start": 20100,  # Record 'scenario_seed_start' as 20100.
        "episodes_per_density": 100,  # Record 'episodes_per_density' as 100.
        "densities": DENSITIES,  # Record 'densities' as DENSITIES.
        "policies": POLICIES,  # Record 'policies' as POLICIES.
        "primary": "100K QR lower quarter minus 100K DQN with the frozen 25 m/s governor, density 1, collision fraction.",  # Record 'primary' as the report text.
        "primary_rule": "Conditional scenario-bootstrap 95% interval must have upper endpoint below zero; report speed regardless. No tuning.",  # Record 'primary_rule' as the report text.
        "secondary": "Budget effects pair the SAME new scenarios. Other policies/densities and three-seed crossed intervals are exploratory.",  # Record 'secondary' as the report text.
        "scope": "Primary inference conditions on selected networks. Equal aggregate speed is not guaranteed. No new-training or real-road safety claim.",  # Record 'scope' as the report text.
        "consumed_prior_confirmation": [20000, 20099],  # Record 'consumed_prior_confirmation' as 20000, 20099.
        "future_untouched_seeds_begin": 20200,  # Record 'future_untouched_seeds_begin' as 20200.
    }  # Finish assembling the grouped values above.
    design = json.loads(json.dumps(design))  # JSON turns tuples into lists. Normalize that detail before checking a recorded design.
    path = OUTPUT / "design.json"  # Set path to the path to 'design.json' inside OUTPUT.
    if path.exists():  # A saved filename is not enough: the checks below verify the contents belong to this experiment.
        if read(path) != design:  # Use this branch when read applied to path differs from design.
            raise ValueError("The recorded study cannot be silently changed")  # Reject this invalid case instead of continuing with misleading experiment results.
    else:  # Handle the alternative when the earlier condition (the result of path.exists) is false.
        for old in (ROOT / "results").rglob("*.json"):  # Process each old from (ROOT / 'results').rglob applied to '*.json'.
            if OUTPUT in old.parents:  # Use this branch when OUTPUT belongs to old.parents.
                continue  # Skip this case and move on to the next loop item.
            data = read(old)  # Set data to read applied to old.
            if isinstance(data, dict) and isinstance(data.get("episodes"), list):  # Use this branch when isinstance applied to data, dict and isinstance applied to data.get applied to 'episodes', list.
                for row in data["episodes"]:  # Process each row from the entries of data selected by 'episodes'.
                    if 20100 <= row.get("seed", -1) < 20200:  # Use this branch when 20100 is at most row.get applied to 'seed', negative 1 is below 20200.
                        raise ValueError("The requested scenario block was used before")  # Reject this invalid case instead of continuing with misleading experiment results.
        save_json(path, design)  # Save the following results/settings as json: path, design.
        print("Design locked; fresh scenario block verified unused.", flush=True)  # Print applied to 'design locked; fresh scenario block verified unused.'.


def evaluate(job):  # Use the existing evaluator; this code never calls training.
    """Use the existing evaluator; this code never calls training."""  # Describe this operation for readers and help().
    budget, policy, seed = job  # Set the environment-interaction budget, policy, seed to job.
    path = model_path(budget, policy, seed)  # Set path to model path applied to the environment-interaction budget, policy, seed.
    algorithm, alpha, cap = POLICIES[policy]  # Set algorithm, the fraction of low-return predictions used to score actions, cap to the entries of POLICIES selected by policy.
    output = OUTPUT / f"eval_{budget}_{policy}_seed{seed}.json"  # Set output to (OUTPUT divided by f'eval {budget} {policy} seed{seed}.json').
    if output.exists():  # Do not overwrite an experiment. First check whether its saved settings match this request.
        data = read(output)  # Set data to read applied to output.
        saved = data["provenance"]  # Set saved to the entries of data selected by 'provenance'.
        actual = (  # Set actual to the listed settings or measurements in their recorded order.
            saved["checkpoint_sha256"],  # Include the entries of saved selected by 'checkpoint_sha256' in the surrounding operation.
            saved["alpha"],  # Include the entries of saved selected by 'alpha' in the surrounding operation.
            saved["speed_governor_cap_m_s"],  # Include the entries of saved selected by 'speed_governor_cap_m_s' in the surrounding operation.
            saved["training_steps"],  # Include the entries of saved selected by 'training_steps' in the surrounding operation.
            saved["seed_start"],  # Include the entries of saved selected by 'seed_start' in the surrounding operation.
            saved["episodes_per_density"],  # Include the entries of saved selected by 'episodes_per_density' in the surrounding operation.
        )  # Finish assembling the grouped values above.
        expected = (fingerprint(path), alpha, cap, budget, 20100, 100)  # Set expected to the listed settings or measurements in their recorded order.
        if actual != expected or len(data["episodes"]) != 400:  # Use this branch when actual differs from expected or the number of entries in the entries of data selected by 'episodes' differs from 400.
            raise ValueError(f"Mismatched evaluation: {output}")  # Reject this invalid case instead of continuing with misleading experiment results.
        cases = set()  # A set keeps each density/scenario pair once, even if a row was duplicated.
        for episode in data["episodes"]:  # Process each episode from the entries of data selected by 'episodes'.
            cases.add((episode["density"], episode["seed"]))  # Cases.add applied to the entries of episode selected by 'density', the entries of episode selected by 'seed'.
        expected_cases = set()  # Set expected cases to the result of set.
        for density in DENSITIES:  # Process each the traffic-density setting from DENSITIES.
            for scenario in range(20100, 20200):  # Process each the starting-traffic seed from range applied to 20100, 20200.
                expected_cases.add((density, scenario))  # Expected cases.add applied to the traffic-density setting, the starting-traffic seed.
        if cases != expected_cases:  # We need every requested density/scenario pair, not just the right total number of rows.
            raise ValueError("Missing or duplicated scenarios")  # Reject this invalid case instead of continuing with misleading experiment results.
    else:  # Handle the alternative when the earlier condition (the result of output.exists) is false.
        command = [  # Set command to the listed settings or measurements in their recorded order.
            sys.executable,  # Include sys.executable in the surrounding operation.
            "-m",  # Include '-m' in the surrounding operation.
            "driving_rl.cli",  # Include 'driving_rl.cli' in the surrounding operation.
            "evaluate",  # Include 'evaluate' in the surrounding operation.
            "--checkpoint",  # Include '--checkpoint' in the surrounding operation.
            str(path),  # Include path as a text value in the surrounding operation.
            "--alpha",  # Include '--alpha' in the surrounding operation.
            str(alpha),  # Include the fraction of low-return predictions used to score actions as a text value in the surrounding operation.
            "--seed-start",  # Include '--seed-start' in the surrounding operation.
            "20100",  # Include '20100' in the surrounding operation.
            "--episodes",  # Include '--episodes' in the surrounding operation.
            "100",  # Include '100' in the surrounding operation.
            "--densities",  # Include '--densities' in the surrounding operation.
            "1",  # Include '1' in the surrounding operation.
            "1.5",  # Include '1.5' in the surrounding operation.
            "2",  # Include '2' in the surrounding operation.
            "2.5",  # Include '2.5' in the surrounding operation.
            "--out",  # Include '--out' in the surrounding operation.
            str(output),  # Include output as a text value in the surrounding operation.
        ]  # Finish assembling the grouped values above.
        if cap is not None:  # The governor is an optional action override. Its threshold is not a guaranteed physical speed limit.
            command.extend(["--speed-cap", str(cap)])  # Add all entries from '--speed-cap', cap as a text value to command.
        with output.with_suffix(".log").open("w") as log:  # Keep this temporary context active during the enclosed work, then clean it up.
            subprocess.run(  # Run the requested Python command and stop on failure. This is evaluation, not a training call.
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True  # Include command, ROOT, log, subprocess.STDOUT, True in the surrounding operation.
            )  # Finish assembling subprocess.run inputs.
    print(f"Evaluated {budget} {policy} seed {seed}", flush=True)  # Print applied to f'evaluated {budget} {policy} seed {seed}'.


def main():  # Read the Terminal options and coordinate the workflow.
    """Read the Terminal options and coordinate the workflow. The learning algorithm is not implemented here."""  # Describe this operation for readers and help().
    parser = argparse.ArgumentParser(description=__doc__)  # Set parser to the result of argparse.ArgumentParser.
    parser.add_argument("--jobs", type=int, default=4)  # Parser.add argument applied to '--jobs'.
    parser.add_argument("--report-only", action="store_true")  # Parser.add argument applied to '--report-only'.
    args = parser.parse_args()  # Set args to the result of parser.parse args.
    if args.jobs < 1:  # Use this branch when args.jobs is below 1.
        parser.error("--jobs must be positive")  # Parser.error applied to '--jobs must be positive'.
    lock_design()  # The result of lock design.
    if not args.report_only:  # Use this branch when not args.report only.
        jobs = []  # Set jobs to .
        for budget in [100000, 40000]:  # Each budget gets its own results. Keep 40K and 100K separate until the paired comparison.
            for policy in POLICIES:  # A policy is a decision rule. Several rules can share the very same QR-DQN network.
                for seed in range(3):  # A training seed identifies one independently trained model, not a new traffic scenario.
                    jobs.append((budget, policy, seed))  # Add the environment-interaction budget, policy, seed to jobs.
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:  # This only schedules independent jobs. Each actual experiment runs in a separate Python process.
            list(pool.map(evaluate, jobs))  # Wait for all requested jobs, and let errors surface instead of quietly ignoring failed experiments.
    from driving_rl.long_training_report import report  # Make report available for this file's calculations and workflow.

    report()  # The result of report.


if __name__ == "__main__":  # Run this workflow when launched from Terminal; importing its helpers must not launch experiments.
    main()  # The result of main.
