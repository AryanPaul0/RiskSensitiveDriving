"""Run the confirmation experiment from Terminal.

This is an experiment coordinator, not the learning algorithm. It launches
separate Python commands, saves logs, and checks existing results before reuse.
Paths identify saved files; hashes are fingerprints that detect changed files.
Read main() for the order of operations. Learning itself lives in driving_rl/agents.py.

Original implementation notes:
Evaluate locked policies once on the reserved scenario block.
"""

import argparse  # Make argparse available for this file's calculations and workflow.
from concurrent.futures import ThreadPoolExecutor  # Make ThreadPoolExecutor available for this file's calculations and workflow.
import hashlib  # Make hashlib available for this file's calculations and workflow.
import json  # Make json available for this file's calculations and workflow.
from pathlib import Path  # Make Path available for this file's calculations and workflow.
import subprocess  # Make subprocess available for this file's calculations and workflow.
import sys  # Make sys available for this file's calculations and workflow.
from driving_rl.experiment import save_json  # Make save_json available for this file's calculations and workflow.
from driving_rl.source_integrity import source_matches_frozen_version  # Make source_matches_frozen_version available for this file's calculations and workflow.

ROOT = Path(__file__).resolve().parent  # Set ROOT to Path( file ).resolve().parent.
STAGE = ROOT / "results/confirmation"  # Set STAGE to the path to 'results/confirmation' inside ROOT.
POLICIES = {  # Set POLICIES to the named settings or measurements assembled below.
    "qr_tail": ("qrdqn", 0.25, None),  # Record 'qr_tail' as 'qrdqn', 0.25, None.
    "dqn_governed": ("dqn", 1.0, 25.0),  # Record 'dqn_governed' as 'dqn', 1.0, 25.0.
    "qr_governed": ("qrdqn", 1.0, 25.0),  # Record 'qr_governed' as 'qrdqn', 1.0, 25.0.
    "dqn_mean": ("dqn", 1.0, None),  # Record 'dqn_mean' as 'dqn', 1.0, None.
    "qr_mean": ("qrdqn", 1.0, None),  # Record 'qr_mean' as 'qrdqn', 1.0, None.
}  # Finish assembling the grouped values above.


def read(p):  # Read saved JSON data into ordinary Python dictionaries and lists.
    """Read saved JSON data into ordinary Python dictionaries and lists."""  # Describe this operation for readers and help().
    return json.loads(p.read_text())  # Return json.loads applied to the result of p.read text.


def sha(p):  # Calculate a file fingerprint.
    """Calculate a file fingerprint. If even one byte changes, its SHA-256 fingerprint
    will almost certainly change too. This helps detect changed frozen artifacts.
    """
    return hashlib.sha256(p.read_bytes()).hexdigest()  # Return the result of hashlib.sha256(p.read bytes()).hexdigest.


def checkpoint(policy, seed):  # Find the saved model file for the requested algorithm and training seed.
    """Find the saved model file for the requested algorithm and training seed."""  # Describe this operation for readers and help().
    algo = POLICIES[policy][0]  # Set algo to the entries of the entries of POLICIES selected by policy selected by 0.
    return (  # Return (ROOT divided by f"results/{('density coverage' if algo == 'qrdqn' else 'baseline controls')}/{algo} seed{seed}/final.pt").
        ROOT  # Include (ROOT divided by f"results/{('density coverage' if algo == 'qrdqn' else 'baseline controls')}/{algo} seed{seed}/final.pt") in the surrounding operation.
        / f'results/{"density_coverage" if algo=="qrdqn" else "baseline_controls"}/{algo}_seed{seed}/final.pt'  # Include f"results/{('density coverage' if algo == 'qrdqn' else 'baseline controls')}/{algo} seed{seed}/final.pt" in the surrounding operation.
    )  # Finish assembling the grouped values above.


def design():  # Describe the fixed confirmation plan, including model fingerprints, scenario seeds, the primary comparison and the limits of the planned uncertainty interval.
    """Describe the fixed confirmation plan, including model fingerprints, scenario
    seeds, the primary comparison and the limits of the planned uncertainty interval.
    """
    return {  # Return the named settings or measurements assembled below.
        "status": "Locked before confirmation evaluations",  # Record 'status' as 'Locked before confirmation evaluations'.
        "training_steps": 40000,  # Record 'training_steps' as 40000.
        "training_seeds": [0, 1, 2],  # Record 'training_seeds' as 0, 1, 2.
        "training_densities": [1.0, 1.5, 2.0],  # Record 'training_densities' as 1.0, 1.5, 2.0.
        "scenario_seed_start": 20000,  # Record 'scenario_seed_start' as 20000.
        "episodes_per_density": 100,  # Record 'episodes_per_density' as 100.
        "densities": [1.0, 1.5, 2.0, 2.5],  # Record 'densities' as 1.0, 1.5, 2.0, 2.5.
        "policies": {  # Record 'policies' as the named settings or measurements assembled below for each p, a, the fraction of low-return predictions used to score actions, cap in the result of POLICIES.items.
            p: {"algorithm": a, "alpha": alpha, "speed_cap_m_s": cap}  # Record 'algorithm' as a.
            for p, (a, alpha, cap) in POLICIES.items()  # Include p, a, the fraction of low-return predictions used to score actions, cap, the result of POLICIES.items in the surrounding operation.
        },  # Finish assembling the grouped values above.
        "primary": {  # Record 'primary' as the named settings or measurements assembled below.
            "left": "qr_tail",  # Record 'left' as 'qr_tail'.
            "right": "dqn_governed",  # Record 'right' as 'dqn_governed'.
            "density": 1.0,  # Record 'density' as 1.0.
            "metric": "crashed",  # Record 'metric' as 'crashed'.
            "direction": "Lower collision fraction is better",  # Record 'direction' as 'Lower collision fraction is better'.
            "estimand": "Expected paired collision difference over fresh scenarios, averaged over the three fixed trained-model pairs.",  # Record 'estimand' as the report text.
            "interval": "Two-sided percentile 95% scenario-cluster bootstrap; 10000 replicates; RNG seed 20261111. Resample scenario IDs while retaining all fixed models and policy pairing.",  # Record 'interval' as the report text.
            "decision_rule": "Evidence of fewer collisions for the locked QR-tail policy if the upper conditional confidence endpoint is strictly below zero. Report speed/distance regardless of collision result. Do not claim speed-independent or real-road safety benefit.",  # Record 'decision_rule' as the report text.
        },  # Finish assembling the grouped values above.
        "secondary": "Other policies, traffic densities, seed variation and crossed bootstrap intervals are descriptive/exploratory; no family-wise inference claimed.",  # Record 'secondary' as the report text.
        "generalization_limit": "Primary interval conditions on this selected set of trained models, architecture, rewards and governor caps; it does not cover new training runs or remove speed/action-rule differences.",  # Record 'generalization_limit' as the report text.
        "no_tuning": "No retraining, checkpoint selection, alpha adjustment or governor adjustment using these results. Further tuned models need a new untouched scenario block.",  # Record 'no_tuning' as the report text.
        "replays": "Training seed 0, first reserved scenario 20000 at density 1, for QR tail and both governed controls; fixed before outcomes.",  # Record 'replays' as the report text.
        "checkpoint_hashes": {  # Record 'checkpoint_hashes' as the named settings or measurements assembled below for each s in range applied to 3 for each p in POLICIES.
            p: {  # Include p, the named settings or measurements assembled below for each s in range applied to 3 in the surrounding operation.
                str(s): {  # Include s as a text value, the named settings or measurements assembled below in the surrounding operation.
                    "path": str(checkpoint(p, s).relative_to(ROOT)),  # Record 'path' as checkpoint(p, s).relative to applied to ROOT as a text value.
                    "sha256": sha(checkpoint(p, s)),  # Record 'sha256' as sha applied to checkpoint applied to p, s.
                }  # Finish assembling the grouped values above.
                for s in range(3)  # Include s, range applied to 3 in the surrounding operation.
            }  # Finish assembling the grouped values above.
            for p in POLICIES  # Include p, POLICIES in the surrounding operation.
        },  # Finish assembling the grouped values above.
        "source_hashes": {  # Record 'source_hashes' as sha applied to p for each p in sorted applied to (ROOT / 'driving rl').glob applied to '*.py', keeping only cases where p.name differs from 'confirmation_report.py'.
            str(p.relative_to(ROOT)): sha(p)  # Include p.relative to applied to ROOT as a text value, sha applied to p in the surrounding operation.
            for p in sorted((ROOT / "driving_rl").glob("*.py"))  # Include p, sorted applied to (ROOT / 'driving rl').glob applied to '*.py' in the surrounding operation.
            if p.name != "confirmation_report.py"  # Include p.name differs from 'confirmation_report.py' in the surrounding operation.
        },  # Finish assembling the grouped values above.
    }  # Finish assembling the grouped values above.


def lock():  # Record the confirmation settings before looking at confirmation outcomes.
    """Record the confirmation settings before looking at confirmation outcomes.
    On later runs, check the model files and settings rather than choosing again.
    """
    STAGE.mkdir(parents=True, exist_ok=True)  # The result of stage.mkdir.
    path = STAGE / "design.json"  # Set path to the path to 'design.json' inside STAGE.
    value = design()  # Set value to the result of design.
    if path.exists():  # Check existing data instead of silently replacing scientific results.
        old = read(path)  # Set old to read applied to path.
        if {k: v for k, v in old.items() if k != "source_hashes"} != {  # Source additions do not alter the already locked design. Check all fixed parameters and hashes.
            k: v for k, v in value.items() if k != "source_hashes"  # Include k, v, k, v, the result of value.items, k differs from 'source_hashes' in the surrounding operation.
        }:  # Finish assembling the grouped values above.
            raise ValueError("Refusing to change the locked confirmation design")  # Reject this invalid case instead of continuing with misleading experiment results.
        for name in ["agents.py", "envs.py", "experiment.py", "cli.py"]:  # Process each name from 'agents.py', 'envs.py', 'experiment.py', 'cli.py'.
            key = "driving_rl/" + name  # Set key to ('driving_rl/' plus name).
            if not source_matches_frozen_version(ROOT, key, old["source_hashes"][key]):  # Use this branch when not source matches frozen version applied to ROOT, key, the entries of the entries of old selected by 'source_hashes' selected by key.
                raise ValueError(  # Reject this invalid case instead of continuing with misleading experiment results.
                    "Evaluation code changed after confirmation lock: " + name  # Include ('Evaluation code changed after confirmation lock: ' plus name) in the surrounding operation.
                )  # Finish assembling ValueError inputs.
        return old  # Return old.
    audited = 0  # Verify that this scenario block has not appeared in any previous saved episodes.
    for p in (ROOT / "results").rglob("*.json"):  # Process each p from (ROOT / 'results').rglob applied to '*.json'.
        if STAGE in p.parents:  # Use this branch when STAGE belongs to p.parents.
            continue  # Skip this case and move on to the next loop item.
        data = read(p)  # Set data to read applied to p.
        if isinstance(data, dict) and isinstance(data.get("episodes"), list):  # Use this branch when isinstance applied to data, dict and isinstance applied to data.get applied to 'episodes', list.
            if any(20000 <= r.get("seed", -1) < 20100 for r in data["episodes"]):  # Use this branch when any applied to 20000 is at most r.get applied to 'seed', -1 is below 20100 for each r in the entries of data selected by 'episodes'.
                raise ValueError(f"Confirmation block already used: {p}")  # Reject this invalid case instead of continuing with misleading experiment results.
            audited += 1  # Increase audited using 1.
    save_json(  # Save the following results/settings as json: the path to 'preflight.json' inside stage, the named settings or measurements assembled below.
        STAGE / "preflight.json",  # Include the path to 'preflight.json' inside STAGE in the surrounding operation.
        {  # Include the named settings or measurements assembled below in the surrounding operation.
            "existing_episode_files_checked": audited,  # Record 'existing_episode_files_checked' as audited.
            "reserved_block_previously_unused": True,  # Record 'reserved_block_previously_unused' as True.
        },  # Finish assembling the grouped values above.
    )  # Finish assembling save json inputs.
    save_json(path, value)  # Save the following results/settings as json: path, value.
    print(  # Print applied to the report text.
        "Locked design before evaluation; reserved scenario block was unused.",  # Include the report text in the surrounding operation.
        flush=True,  # Include True in the surrounding operation.
    )  # Finish assembling print inputs.
    return value  # Return value.


def evaluate(spec):  # Run or verify one frozen-policy evaluation.
    """Run or verify one frozen-policy evaluation.
    Check its model fingerprint, settings and scenario block before reusing saved
    data. This function must not quietly mix results from different experiments.
    """
    policy, seed = spec  # Set policy, seed to spec.
    algo, alpha, cap = POLICIES[policy]  # Set algo, the fraction of low-return predictions used to score actions, cap to the entries of POLICIES selected by policy.
    ck = checkpoint(policy, seed)  # Set ck to checkpoint applied to policy, seed.
    out = STAGE / f"eval_{policy}_seed{seed}.json"  # Set out to (STAGE divided by f'eval {policy} seed{seed}.json').
    if out.exists():  # A saved file is useful only if its model and settings match this request.
        d = read(out)  # Set d to read applied to out.
        p = d["provenance"]  # Set p to the entries of d selected by 'provenance'.
        if (  # Use this branch when the listed settings or measurements in their recorded order differs from the listed settings or measurements in their recorded order.
            p["checkpoint_sha256"],  # Include the entries of p selected by 'checkpoint_sha256' in the surrounding operation.
            p["alpha"],  # Include the entries of p selected by 'alpha' in the surrounding operation.
            p["speed_governor_cap_m_s"],  # Include the entries of p selected by 'speed_governor_cap_m_s' in the surrounding operation.
            p["seed_start"],  # Include the entries of p selected by 'seed_start' in the surrounding operation.
            p["episodes_per_density"],  # Include the entries of p selected by 'episodes_per_density' in the surrounding operation.
            p["training_steps"],  # Include the entries of p selected by 'training_steps' in the surrounding operation.
            p["training_seed"],  # Include the entries of p selected by 'training_seed' in the surrounding operation.
            [r["density"] for r in d["summary"]],  # Include the entries of r selected by 'density' for each r in the entries of d selected by 'summary' in the surrounding operation.
        ) != (sha(ck), alpha, cap, 20000, 100, 40000, seed, [1.0, 1.5, 2.0, 2.5]):  # Finish assembling the grouped values above.
            raise ValueError(f"Mismatched confirmation {out}")  # Reject this invalid case instead of continuing with misleading experiment results.
        expected = {  # Set expected to the traffic-density setting, the starting-traffic seed for each the traffic-density setting in 1.0, 1.5, 2.0, 2.5.
            (density, scenario)  # Include the traffic-density setting, the starting-traffic seed in the surrounding operation.
            for density in [1.0, 1.5, 2.0, 2.5]  # Include the traffic-density setting, 1.0, 1.5, 2.0, 2.5 in the surrounding operation.
            for scenario in range(20000, 20100)  # Include the starting-traffic seed, range applied to 20000, 20100 in the surrounding operation.
        }  # Finish assembling the grouped values above.
        if {(r["density"], r["seed"]) for r in d["episodes"]} != expected or len(  # Use this branch when r['density'], r['seed'] for each r in the entries of d selected by 'episodes' differs from expected or the number of entries in the entries of d selected by 'episodes' differs from 400.
            d["episodes"]  # Include the entries of d selected by 'episodes' in the surrounding operation.
        ) != 400:  # Finish assembling len inputs.
            raise ValueError("Incomplete/duplicate confirmation cases")  # Reject this invalid case instead of continuing with misleading experiment results.
    else:  # Handle the alternative when the earlier condition (the result of out.exists) is false.
        cmd = [  # Set cmd to the listed settings or measurements in their recorded order.
            sys.executable,  # Include sys.executable in the surrounding operation.
            "-m",  # Include '-m' in the surrounding operation.
            "driving_rl.cli",  # Include 'driving_rl.cli' in the surrounding operation.
            "evaluate",  # Include 'evaluate' in the surrounding operation.
            "--checkpoint",  # Include '--checkpoint' in the surrounding operation.
            str(ck),  # Include ck as a text value in the surrounding operation.
            "--alpha",  # Include '--alpha' in the surrounding operation.
            str(alpha),  # Include the fraction of low-return predictions used to score actions as a text value in the surrounding operation.
            "--episodes",  # Include '--episodes' in the surrounding operation.
            "100",  # Include '100' in the surrounding operation.
            "--seed-start",  # Include '--seed-start' in the surrounding operation.
            "20000",  # Include '20000' in the surrounding operation.
            "--densities",  # Include '--densities' in the surrounding operation.
            "1",  # Include '1' in the surrounding operation.
            "1.5",  # Include '1.5' in the surrounding operation.
            "2",  # Include '2' in the surrounding operation.
            "2.5",  # Include '2.5' in the surrounding operation.
            "--out",  # Include '--out' in the surrounding operation.
            str(out),  # Include out as a text value in the surrounding operation.
        ]  # Finish assembling the grouped values above.
        if cap is not None:  # Use this branch when cap is not None.
            cmd += ["--speed-cap", str(cap)]  # Increase cmd using '--speed-cap', cap as a text value.
        if seed == 0 and policy in ["qr_tail", "dqn_governed", "qr_governed"]:  # Use this branch when seed equals 0 and policy belongs to 'qr_tail', 'dqn_governed', 'qr_governed'.
            cmd += ["--gif", str(STAGE / f"replay_{policy}.gif")]  # Increase cmd using '--gif', (STAGE divided by f'replay {policy}.gif') as a text value.
        with out.with_suffix(".log").open("w") as f:  # Keep this temporary context active during the enclosed work, then clean it up.
            subprocess.run(  # Start another Python command and check whether it succeeded.
                cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True  # Include cmd, ROOT, f, subprocess.STDOUT, True in the surrounding operation.
            )  # Finish assembling subprocess.run inputs.
    print(f"Confirmed {policy} seed {seed}", flush=True)  # Print applied to f'confirmed {policy} seed {seed}'.


def main():  # Read Terminal options, check settings, and run the requested workflow.
    """Read Terminal options, check settings, and run the requested workflow.
    Existing result files are checked before reuse so a rerun cannot silently mix
    different experiments. The main guard below prevents running jobs on import.
    """
    p = argparse.ArgumentParser(description=__doc__)  # Set p to the result of argparse.ArgumentParser.
    p.add_argument("--jobs", type=int, default=3)  # P.add argument applied to '--jobs'.
    p.add_argument("--stage", choices=["all", "report"], default="all")  # P.add argument applied to '--stage'.
    a = p.parse_args()  # Set a to the result of p.parse args.
    if a.jobs < 1:  # Use this branch when a.jobs is below 1.
        p.error("--jobs must be positive")  # P.error applied to '--jobs must be positive'.
    lock()  # The result of lock.
    if a.stage == "all":  # Use this branch when a.stage equals 'all'.
        with ThreadPoolExecutor(max_workers=a.jobs) as pool:  # Run independent jobs together. Each writes its own files; this is not a shared training model.
            list(  # List applied to pool.map applied to evaluate, policy, s for each policy in policies.
                pool.map(  # Include pool.map applied to evaluate, policy, s for each policy in POLICIES in the surrounding operation.
                    evaluate, [(policy, s) for policy in POLICIES for s in range(3)]  # Include evaluate, policy, s for each policy in POLICIES in the surrounding operation.
                )  # Finish assembling pool.map inputs.
            )  # Finish assembling list inputs.
    from driving_rl.confirmation_report import report  # Make report available for this file's calculations and workflow.

    report()  # The result of report.


if __name__ == "__main__":  # Run this workflow when launched from Terminal; importing its helpers must not launch experiments.
    main()  # The result of main.
