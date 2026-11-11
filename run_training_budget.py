"""Run the training budget experiment from Terminal.

This is an experiment coordinator, not the learning algorithm. It launches
separate Python commands, saves logs, and checks existing results before reuse.
Paths identify saved files; hashes are fingerprints that detect changed files.
Read main() for the order of operations. Learning itself lives in driving_rl/agents.py.

Original implementation notes:
Controlled budget study: train, validate, evaluate and report.
"""

import argparse  # Make argparse available for this file's calculations and workflow.
from concurrent.futures import ThreadPoolExecutor  # Make ThreadPoolExecutor available for this file's calculations and workflow.
import hashlib  # Make hashlib available for this file's calculations and workflow.
import json  # Make json available for this file's calculations and workflow.
from pathlib import Path  # Make Path available for this file's calculations and workflow.
import subprocess  # Make subprocess available for this file's calculations and workflow.
import sys  # Make sys available for this file's calculations and workflow.
import torch  # Make torch available for this file's calculations and workflow.
from driving_rl.experiment import save_json  # Make save_json available for this file's calculations and workflow.

ROOT = Path(__file__).resolve().parent  # Set ROOT to Path( file ).resolve().parent.
STAGE = ROOT / "results/training_budget"  # Set STAGE to the path to 'results/training_budget' inside ROOT.


def run(command, log):  # Run a command in a separate Python process and wait for it to finish.
    """Run a command in a separate Python process and wait for it to finish.
    A nonzero exit status means the command failed; do not silently continue.
    """
    with log.open("w") as f:  # Keep this temporary context active during the enclosed work, then clean it up.
        subprocess.run(  # Start another Python command and check whether it succeeded.
            [sys.executable, "-m", *command],  # Include sys.executable, '-m', *command in the surrounding operation.
            cwd=ROOT,  # Include ROOT in the surrounding operation.
            stdout=f,  # Include f in the surrounding operation.
            stderr=subprocess.STDOUT,  # Include subprocess.STDOUT in the surrounding operation.
            check=True,  # Include True in the surrounding operation.
        )  # Finish assembling subprocess.run inputs.


def train(spec):  # Train one missing experiment, or verify its completed saved model.
    """Train one missing experiment, or verify its completed saved model.
    Keep model settings fixed across the requested seeds. An incomplete directory
    is not silently resumed because replay memory and RNG state are not saved.
    """
    algo, seed = spec  # Set algo, seed to spec.
    folder = STAGE / f"{algo}_seed{seed}"  # Set folder to (STAGE divided by f'{algo} seed{seed}').
    if not (folder / "summary.json").exists():  # Use this branch when not the result of (folder / 'summary.json').exists.
        run(  # Run applied to the listed settings or measurements in their recorded order, (stage divided by f'train {algo} {seed}.log').
            [  # Include the listed settings or measurements in their recorded order in the surrounding operation.
                "driving_rl.cli",  # Include 'driving_rl.cli' in the surrounding operation.
                "train",  # Include 'train' in the surrounding operation.
                "--algorithm",  # Include '--algorithm' in the surrounding operation.
                algo,  # Include algo in the surrounding operation.
                "--seed",  # Include '--seed' in the surrounding operation.
                str(seed),  # Include seed as a text value in the surrounding operation.
                "--steps",  # Include '--steps' in the surrounding operation.
                "40000",  # Include '40000' in the surrounding operation.
                "--exploration-steps",  # Include '--exploration-steps' in the surrounding operation.
                "12000",  # Include '12000' in the surrounding operation.
                "--out",  # Include '--out' in the surrounding operation.
                str(folder),  # Include folder as a text value in the surrounding operation.
            ],  # Finish assembling the grouped values above.
            STAGE / f"train_{algo}_{seed}.log",  # Include (STAGE divided by f'train {algo} {seed}.log') in the surrounding operation.
        )  # Finish assembling run inputs.
    c = json.loads((folder / "config.json").read_text())  # Set c to json.loads applied to the result of (folder / 'config.json').read text.
    if (c["steps"], c["exploration_steps"], c["seed"], c["algorithm"]) != (  # Use this branch when the entries of c selected by 'steps', the entries of c selected by 'exploration_steps', the entries of c selected by 'seed', the entries of c selected by 'algorithm' differs from 40000, 12000, seed, algo.
        40000,  # Include 40000 in the surrounding operation.
        12000,  # Include 12000 in the surrounding operation.
        seed,  # Include seed in the surrounding operation.
        algo,  # Include algo in the surrounding operation.
    ):  # Finish assembling the grouped values above.
        raise ValueError(f"Mismatched training configuration: {folder}")  # Reject this invalid case instead of continuing with misleading experiment results.
    for key, value in {  # Process each key, value from the result of {'gamma': 0.9, 'learning rate': 0.0005, 'collision penalty': -2.0, 'buffer size': 30000, 'batch size': 64, 'learning starts': 500, 'train frequency': 1, 'target interval': 250, 'final epsilon': 0.05, 'quantiles': 32}.items.
        "gamma": 0.9,  # Record 'gamma' as 0.9.
        "learning_rate": 0.0005,  # Record 'learning_rate' as 0.0005.
        "collision_penalty": -2.0,  # Record 'collision_penalty' as negative 2.0.
        "buffer_size": 30000,  # Record 'buffer_size' as 30000.
        "batch_size": 64,  # Record 'batch_size' as 64.
        "learning_starts": 500,  # Record 'learning_starts' as 500.
        "train_frequency": 1,  # Record 'train_frequency' as 1.
        "target_interval": 250,  # Record 'target_interval' as 250.
        "final_epsilon": 0.05,  # Record 'final_epsilon' as 0.05.
        "quantiles": 32,  # Record 'quantiles' as 32.
    }.items():  # Finish assembling {'gamma': 0.9, 'learning rate': 0.0005, 'collision penalty': -2.0, 'buffer size': 30000, 'batch size': 64, 'learning starts': 500, 'train frequency': 1, 'target interval': 250, 'final epsilon': 0.05, 'quantiles': 32}.items inputs.
        if c[key] != value:  # Use this branch when the entries of c selected by key differs from value.
            raise ValueError(f"Mismatched {key}: {folder}")  # Reject this invalid case instead of continuing with misleading experiment results.
    old = torch.load(ROOT / f"results/{algo}_seed{seed}/final.pt", weights_only=True)  # Set old to torch.load applied to (ROOT divided by f'results/{algo} seed{seed}/final.pt').
    new = torch.load(folder / "checkpoint_20000.pt", weights_only=True)  # Set new to torch.load applied to the path to 'checkpoint_20000.pt' inside folder.
    match = all(torch.equal(old["online"][k], new["online"][k]) for k in old["online"])  # Set match to all applied to torch.equal applied to the entries of old['online'] selected by k, the entries of new['online'] selected by k for each k in the entries of old selected by 'online'.
    print(f"Trained {algo} seed {seed}; exact 20k weights: {match}", flush=True)  # Print applied to f'trained {algo} seed {seed}; exact 20k weights: {match}'.
    return dict(algorithm=algo, seed=seed, prefix_weights_equal=match)  # Return the result of dict.


def evaluate(spec):  # Run or verify one frozen-policy evaluation.
    """Run or verify one frozen-policy evaluation.
    Check its model fingerprint, settings and scenario block before reusing saved
    data. This function must not quietly mix results from different experiments.
    """
    algo, seed, steps, policy = spec  # Set algo, seed, steps, policy to spec.
    ckpt = STAGE / f"{algo}_seed{seed}/checkpoint_{steps}.pt"  # Set ckpt to (STAGE divided by f'{algo} seed{seed}/checkpoint {steps}.pt').
    out = STAGE / f"eval_{algo}_{seed}_{steps}_{policy}.json"  # Set out to (STAGE divided by f'eval {algo} {seed} {steps} {policy}.json').
    alpha = 0.25 if policy == "tail25" else 1.0  # Set the fraction of low-return predictions used to score actions to 0.25 when policy equals 'tail25', otherwise 1.0.
    cap = 25.0 if policy == "governed" else None  # Set cap to 25.0 when policy equals 'governed', otherwise None.
    digest = hashlib.sha256(ckpt.read_bytes()).hexdigest()  # Set digest to the result of hashlib.sha256(ckpt.read bytes()).hexdigest.
    if out.exists():  # A saved file is useful only if its model and settings match this request.
        p = json.loads(out.read_text())["provenance"]  # Set p to the entries of json.loads applied to the result of out.read text selected by 'provenance'.
        if (  # Use this branch when the entries of p selected by 'checkpoint_sha256', the entries of p selected by 'alpha', the entries of p selected by 'speed_governor_cap_m_s', the entries of p selected by 'seed_start', the entries of p selected by 'episodes_per_density' differs from digest, the fraction of low-return predictions used to score actions, cap, 10000, 30.
            p["checkpoint_sha256"],  # Include the entries of p selected by 'checkpoint_sha256' in the surrounding operation.
            p["alpha"],  # Include the entries of p selected by 'alpha' in the surrounding operation.
            p["speed_governor_cap_m_s"],  # Include the entries of p selected by 'speed_governor_cap_m_s' in the surrounding operation.
            p["seed_start"],  # Include the entries of p selected by 'seed_start' in the surrounding operation.
            p["episodes_per_density"],  # Include the entries of p selected by 'episodes_per_density' in the surrounding operation.
        ) != (digest, alpha, cap, 10000, 30):  # Finish assembling the grouped values above.
            raise ValueError(f"Mismatched evaluation: {out}")  # Reject this invalid case instead of continuing with misleading experiment results.
        data = json.loads(out.read_text())  # Set data to json.loads applied to the result of out.read text.
        if [r["density"] for r in data["summary"]] != [1.0, 1.5, 2.0] or len(  # Use this branch when the entries of r selected by 'density' for each r in the entries of data selected by 'summary' differs from 1.0, 1.5, 2.0 or the number of entries in the entries of data selected by 'episodes' differs from 90.
            data["episodes"]  # Include the entries of data selected by 'episodes' in the surrounding operation.
        ) != 90:  # Finish assembling len inputs.
            raise ValueError(f"Incomplete evaluation: {out}")  # Reject this invalid case instead of continuing with misleading experiment results.
    else:  # Handle the alternative when the earlier condition (the result of out.exists) is false.
        cmd = [  # Set cmd to the listed settings or measurements in their recorded order.
            "driving_rl.cli",  # Include 'driving_rl.cli' in the surrounding operation.
            "evaluate",  # Include 'evaluate' in the surrounding operation.
            "--checkpoint",  # Include '--checkpoint' in the surrounding operation.
            str(ckpt),  # Include ckpt as a text value in the surrounding operation.
            "--alpha",  # Include '--alpha' in the surrounding operation.
            str(alpha),  # Include the fraction of low-return predictions used to score actions as a text value in the surrounding operation.
            "--episodes",  # Include '--episodes' in the surrounding operation.
            "30",  # Include '30' in the surrounding operation.
            "--out",  # Include '--out' in the surrounding operation.
            str(out),  # Include out as a text value in the surrounding operation.
        ]  # Finish assembling the grouped values above.
        if cap:  # Use this branch when cap.
            cmd += ["--speed-cap", str(cap)]  # Increase cmd using '--speed-cap', cap as a text value.
        run(cmd, out.with_suffix(".log"))  # Run applied to cmd, out.with suffix applied to '.log'.
    print(f"Evaluated {out.name}", flush=True)  # Print applied to f'evaluated {out.name}'.


def calibrate(spec):  # Run prediction checks for the specified saved model, or validate saved checks.
    """Run prediction checks for the specified saved model, or validate saved checks."""  # Describe this operation for readers and help().
    seed, policy = spec  # Set seed, policy to spec.
    ckpt = STAGE / f"qrdqn_seed{seed}/checkpoint_40000.pt"  # Set ckpt to (STAGE divided by f'qrdqn seed{seed}/checkpoint 40000.pt').
    out = STAGE / f"calibration_{policy}_seed{seed}.json"  # Set out to (STAGE divided by f'calibration {policy} seed{seed}.json').
    alpha = 0.25 if policy == "tail25" else 1.0  # Set the fraction of low-return predictions used to score actions to 0.25 when policy equals 'tail25', otherwise 1.0.
    if out.exists():  # A saved file is useful only if its model and settings match this request.
        p = json.loads(out.read_text())["provenance"]  # Set p to the entries of json.loads applied to the result of out.read text selected by 'provenance'.
        if (  # Use this branch when the entries of p selected by 'checkpoint_sha256', the entries of p selected by 'alpha', the entries of p selected by 'seed_start', the entries of p selected by 'episodes_per_density' differs from the result of hashlib.sha256(ckpt.read bytes()).hexdigest, the fraction of low-return predictions used to score actions, 12000, 60.
            p["checkpoint_sha256"],  # Include the entries of p selected by 'checkpoint_sha256' in the surrounding operation.
            p["alpha"],  # Include the entries of p selected by 'alpha' in the surrounding operation.
            p["seed_start"],  # Include the entries of p selected by 'seed_start' in the surrounding operation.
            p["episodes_per_density"],  # Include the entries of p selected by 'episodes_per_density' in the surrounding operation.
        ) != (hashlib.sha256(ckpt.read_bytes()).hexdigest(), alpha, 12000, 60):  # Finish assembling the grouped values above.
            raise ValueError(f"Mismatched calibration: {out}")  # Reject this invalid case instead of continuing with misleading experiment results.
        data = json.loads(out.read_text())  # Set data to json.loads applied to the result of out.read text.
        if [r["density"] for r in data["summary"]] != [1.0, 1.5, 2.0] or len(  # Use this branch when the entries of r selected by 'density' for each r in the entries of data selected by 'summary' differs from 1.0, 1.5, 2.0 or the number of entries in the entries of data selected by 'episodes' differs from 180.
            data["episodes"]  # Include the entries of data selected by 'episodes' in the surrounding operation.
        ) != 180:  # Finish assembling len inputs.
            raise ValueError(f"Incomplete calibration: {out}")  # Reject this invalid case instead of continuing with misleading experiment results.
    else:  # Handle the alternative when the earlier condition (the result of out.exists) is false.
        run(  # Run applied to the listed settings or measurements in their recorded order, out.with suffix applied to '.log'.
            [  # Include the listed settings or measurements in their recorded order in the surrounding operation.
                "driving_rl.calibration",  # Include 'driving_rl.calibration' in the surrounding operation.
                "--checkpoint",  # Include '--checkpoint' in the surrounding operation.
                str(ckpt),  # Include ckpt as a text value in the surrounding operation.
                "--alpha",  # Include '--alpha' in the surrounding operation.
                str(alpha),  # Include the fraction of low-return predictions used to score actions as a text value in the surrounding operation.
                "--out",  # Include '--out' in the surrounding operation.
                str(out),  # Include out as a text value in the surrounding operation.
            ],  # Finish assembling the grouped values above.
            out.with_suffix(".log"),  # Include out.with suffix applied to '.log' in the surrounding operation.
        )  # Finish assembling run inputs.
    print(f"Calibrated {out.name}", flush=True)  # Print applied to f'calibrated {out.name}'.


def main():  # Read Terminal options, check settings, and run the requested workflow.
    """Read Terminal options, check settings, and run the requested workflow.
    Existing result files are checked before reuse so a rerun cannot silently mix
    different experiments. The main guard below prevents running jobs on import.
    """
    parser = argparse.ArgumentParser(description=__doc__)  # Set parser to the result of argparse.ArgumentParser.
    parser.add_argument("--jobs", type=int, default=3)  # Parser.add argument applied to '--jobs'.
    args = parser.parse_args()  # Set args to the result of parser.parse args.
    if args.jobs < 1:  # Use this branch when args.jobs is below 1.
        parser.error("--jobs must be positive")  # Parser.error applied to '--jobs must be positive'.
    STAGE.mkdir(parents=True, exist_ok=True)  # The result of stage.mkdir.
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:  # Run independent jobs together. Each writes its own files; this is not a shared training model.
        prefixes = list(  # Set prefixes to list applied to pool.map applied to train, a, s for each a in 'dqn', 'qrdqn'.
            pool.map(train, [(a, s) for a in ["dqn", "qrdqn"] for s in range(3)])  # Include pool.map applied to train, a, s for each a in 'dqn', 'qrdqn' in the surrounding operation.
        )  # Finish assembling list inputs.
        save_json(STAGE / "prefix_verification.json", prefixes)  # Save the following results/settings as json: the path to 'prefix_verification.json' inside stage, prefixes.
        list(  # List applied to pool.map applied to evaluate, a, s, n, p for each a in 'dqn', 'qrdqn'.
            pool.map(  # Include pool.map applied to evaluate, a, s, n, p for each a in 'dqn', 'qrdqn' in the surrounding operation.
                evaluate,  # Include evaluate in the surrounding operation.
                [  # Include a, s, n, p for each a in 'dqn', 'qrdqn' in the surrounding operation.
                    (a, s, n, p)  # Include a, s, n, p in the surrounding operation.
                    for a in ["dqn", "qrdqn"]  # Include a, 'dqn', 'qrdqn' in the surrounding operation.
                    for s in range(3)  # Include s, range applied to 3 in the surrounding operation.
                    for n in [20000, 40000]  # Include n, 20000, 40000 in the surrounding operation.
                    for p in (  # Include p in the surrounding operation.
                        ["mean"] if a == "dqn" else ["mean", "tail25", "governed"]  # Include 'mean' when a equals 'dqn', otherwise 'mean', 'tail25', 'governed' in the surrounding operation.
                    )  # Finish assembling pool.map inputs.
                ],  # Finish assembling pool.map inputs.
            )  # Finish assembling pool.map inputs.
        )  # Finish assembling list inputs.
        list(  # List applied to pool.map applied to calibrate, s, p for each s in range applied to 3.
            pool.map(calibrate, [(s, p) for s in range(3) for p in ["mean", "tail25"]])  # Include pool.map applied to calibrate, s, p for each s in range applied to 3 in the surrounding operation.
        )  # Finish assembling list inputs.
    from driving_rl.training_budget_report import report  # Make report available for this file's calculations and workflow.

    report()  # The result of report.


if __name__ == "__main__":  # Run this workflow when launched from Terminal; importing its helpers must not launch experiments.
    main()  # The result of main.
