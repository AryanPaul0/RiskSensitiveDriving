"""Run the density coverage experiment from Terminal.

This is an experiment coordinator, not the learning algorithm. It launches
separate Python commands, saves logs, and checks existing results before reuse.
Paths identify saved files; hashes are fingerprints that detect changed files.
Read main() for the order of operations. Learning itself lives in driving_rl/agents.py.

Original implementation notes:
Compare fixed-density and mixed-density QR-DQN training at equal budgets.
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
STAGE = ROOT / "results/density_coverage"  # Set STAGE to the path to 'results/density_coverage' inside ROOT.
POLICIES = {"mean": (1.0, None), "tail25": (0.25, None), "governed": (1.0, 25.0)}  # Set POLICIES to the named settings or measurements assembled below.
DESIGN = {  # Set DESIGN to the named settings or measurements assembled below.
    "training_steps": 40000,  # Record 'training_steps' as 40000.
    "training_seeds": [0, 1, 2],  # Record 'training_seeds' as 0, 1, 2.
    "training_densities": [1.0, 1.5, 2.0],  # Record 'training_densities' as 1.0, 1.5, 2.0.
    "density_sampling": "Uniform per episode, independent RNG; equal episodes do not imply equal transitions.",  # Record 'density_sampling' as the report text.
    "comparison": "Existing fixed-density 40k QR-DQN, same architecture/reward/optimizer/exploration.",  # Record 'comparison' as the report text.
    "epsilon_decay_steps": 12000,  # Record 'epsilon_decay_steps' as 12000.
    "primary_evaluation": {  # Record 'primary_evaluation' as the named settings or measurements assembled below.
        "densities": [1.0, 1.5, 2.0],  # Record 'densities' as 1.0, 1.5, 2.0.
        "seed_start": 10000,  # Record 'seed_start' as 10000.
        "episodes": 30,  # Record 'episodes' as 30.
    },  # Finish assembling the grouped values above.
    "transfer_evaluation": {  # Record 'transfer_evaluation' as the named settings or measurements assembled below.
        "densities": [1.25, 2.5],  # Record 'densities' as 1.25, 2.5.
        "seed_start": 16000,  # Record 'seed_start' as 16000.
        "episodes": 30,  # Record 'episodes' as 30.
    },  # Finish assembling the grouped values above.
    "calibration": {"densities": [1.0, 1.5, 2.0], "seed_start": 12000, "episodes": 60},  # Record 'calibration' as the named settings or measurements assembled below.
    "decision_rules": {"mean": 1.0, "tail25": 0.25, "governor_cap_m_s": 25.0},  # Record 'decision_rules' as the named settings or measurements assembled below.
    "primary_contrast": "Mixed minus fixed cautious collision fraction at density 2; also report speed/distance and all seeds.",  # Record 'primary_contrast' as the report text.
    "limitations": "Development comparisons, three matched seed IDs; no tuned selector, safety claim or global CVaR guarantee. Density 1.5/2 are within mixture support; 2.5 is beyond both training supports. Test mixture changes hidden episode dynamics.",  # Record 'limitations' as the report text.
    "confirmation_scenarios_used": False,  # Record 'confirmation_scenarios_used' as False.
}  # Finish assembling the grouped values above.


def read(p):  # Read saved JSON data into ordinary Python dictionaries and lists.
    """Read saved JSON data into ordinary Python dictionaries and lists."""  # Describe this operation for readers and help().
    return json.loads(p.read_text())  # Return json.loads applied to the result of p.read text.


def run(cmd, log):  # Run a command in a separate Python process and wait for it to finish.
    """Run a command in a separate Python process and wait for it to finish.
    A nonzero exit status means the command failed; do not silently continue.
    """
    with log.open("w") as f:  # Keep this temporary context active during the enclosed work, then clean it up.
        subprocess.run(  # Start another Python command and check whether it succeeded.
            [sys.executable, "-m", *cmd],  # Include sys.executable, '-m', *cmd in the surrounding operation.
            cwd=ROOT,  # Include ROOT in the surrounding operation.
            stdout=f,  # Include f in the surrounding operation.
            stderr=subprocess.STDOUT,  # Include subprocess.STDOUT in the surrounding operation.
            check=True,  # Include True in the surrounding operation.
        )  # Finish assembling subprocess.run inputs.


def checkpoint(regime, seed):  # Find the saved model file for the requested algorithm and training seed.
    """Find the saved model file for the requested algorithm and training seed."""  # Describe this operation for readers and help().
    return (  # Return (STAGE divided by f'qrdqn seed{seed}/final.pt') when regime equals 'mixed', otherwise (ROOT divided by f'results/training budget/qrdqn seed{seed}/checkpoint 40000.pt').
        STAGE / f"qrdqn_seed{seed}/final.pt"  # Include (STAGE divided by f'qrdqn seed{seed}/final.pt') when regime equals 'mixed', otherwise (ROOT divided by f'results/training budget/qrdqn seed{seed}/checkpoint 40000.pt') in the surrounding operation.
        if regime == "mixed"  # Include regime equals 'mixed' in the surrounding operation.
        else ROOT / f"results/training_budget/qrdqn_seed{seed}/checkpoint_40000.pt"  # Include (ROOT divided by f'results/training budget/qrdqn seed{seed}/checkpoint 40000.pt') in the surrounding operation.
    )  # Finish assembling the grouped values above.


def train(seed):  # Train one missing experiment, or verify its completed saved model.
    """Train one missing experiment, or verify its completed saved model.
    Keep model settings fixed across the requested seeds. An incomplete directory
    is not silently resumed because replay memory and RNG state are not saved.
    """
    folder = STAGE / f"qrdqn_seed{seed}"  # Set folder to (STAGE divided by f'qrdqn seed{seed}').
    if not (folder / "summary.json").exists():  # Use this branch when not the result of (folder / 'summary.json').exists.
        run(  # Run applied to the listed settings or measurements in their recorded order, (stage divided by f'train seed{seed}.log').
            [  # Include the listed settings or measurements in their recorded order in the surrounding operation.
                "driving_rl.cli",  # Include 'driving_rl.cli' in the surrounding operation.
                "train",  # Include 'train' in the surrounding operation.
                "--algorithm",  # Include '--algorithm' in the surrounding operation.
                "qrdqn",  # Include 'qrdqn' in the surrounding operation.
                "--seed",  # Include '--seed' in the surrounding operation.
                str(seed),  # Include seed as a text value in the surrounding operation.
                "--steps",  # Include '--steps' in the surrounding operation.
                "40000",  # Include '40000' in the surrounding operation.
                "--exploration-steps",  # Include '--exploration-steps' in the surrounding operation.
                "12000",  # Include '12000' in the surrounding operation.
                "--training-densities",  # Include '--training-densities' in the surrounding operation.
                "1",  # Include '1' in the surrounding operation.
                "1.5",  # Include '1.5' in the surrounding operation.
                "2",  # Include '2' in the surrounding operation.
                "--out",  # Include '--out' in the surrounding operation.
                str(folder),  # Include folder as a text value in the surrounding operation.
            ],  # Finish assembling the grouped values above.
            STAGE / f"train_seed{seed}.log",  # Include (STAGE divided by f'train seed{seed}.log') in the surrounding operation.
        )  # Finish assembling run inputs.
    c = read(folder / "config.json")  # Set c to read applied to the path to 'config.json' inside folder.
    base = read(ROOT / f"results/training_budget/qrdqn_seed{seed}/config.json")  # Set base to read applied to (ROOT divided by f'results/training budget/qrdqn seed{seed}/config.json').
    for k in [  # Process each k from the listed settings or measurements in their recorded order.
        "seed",  # Include 'seed' in the surrounding operation.
        "steps",  # Include 'steps' in the surrounding operation.
        "algorithm",  # Include 'algorithm' in the surrounding operation.
        "gamma",  # Include 'gamma' in the surrounding operation.
        "learning_rate",  # Include 'learning_rate' in the surrounding operation.
        "collision_penalty",  # Include 'collision_penalty' in the surrounding operation.
        "buffer_size",  # Include 'buffer_size' in the surrounding operation.
        "batch_size",  # Include 'batch_size' in the surrounding operation.
        "learning_starts",  # Include 'learning_starts' in the surrounding operation.
        "train_frequency",  # Include 'train_frequency' in the surrounding operation.
        "target_interval",  # Include 'target_interval' in the surrounding operation.
        "final_epsilon",  # Include 'final_epsilon' in the surrounding operation.
        "quantiles",  # Include 'quantiles' in the surrounding operation.
        "exploration_steps",  # Include 'exploration_steps' in the surrounding operation.
        "observation_dim",  # Include 'observation_dim' in the surrounding operation.
        "n_actions",  # Include 'n_actions' in the surrounding operation.
    ]:  # Finish assembling the grouped values above.
        if c[k] != base[k]:  # Use this branch when the entries of c selected by k differs from the entries of base selected by k.
            raise ValueError(f"Mismatched {k}: {folder}")  # Reject this invalid case instead of continuing with misleading experiment results.
    if (  # Use this branch when the entries of c selected by 'training_densities' differs from 1.0, 1.5, 2.0 or not the entries of read applied to folder / 'summary.json' selected by 'training_complete'.
        c["training_densities"] != [1.0, 1.5, 2.0]  # Include the entries of c selected by 'training_densities' differs from 1.0, 1.5, 2.0 or not the entries of read applied to folder / 'summary.json' selected by 'training_complete' in the surrounding operation.
        or not read(folder / "summary.json")["training_complete"]  # Include not the entries of read applied to the path to 'summary.json' inside folder selected by 'training_complete' in the surrounding operation.
    ):  # Finish assembling the grouped values above.
        raise ValueError("Incomplete/mismatched mixture run")  # Reject this invalid case instead of continuing with misleading experiment results.
    print(f"Training complete: seed {seed}", flush=True)  # Print applied to f'training complete: seed {seed}'.


def evaluate(spec):  # Run or verify one frozen-policy evaluation.
    """Run or verify one frozen-policy evaluation.
    Check its model fingerprint, settings and scenario block before reusing saved
    data. This function must not quietly mix results from different experiments.
    """
    regime, seed, policy, block = spec  # Set regime, seed, policy, block to spec.
    alpha, cap = POLICIES[policy]  # Set the fraction of low-return predictions used to score actions, cap to the entries of POLICIES selected by policy.
    d = DESIGN["primary_evaluation" if block == "main" else "transfer_evaluation"]  # Set d to the entries of DESIGN selected by 'primary_evaluation' if block == 'main' else 'transfer_evaluation'.
    ck = checkpoint(regime, seed)  # Set ck to checkpoint applied to regime, seed.
    out = (  # Set out to (ROOT divided by f'results/training budget/eval qrdqn {seed} 40000 {policy}.json') when regime equals 'fixed' and block equals 'main', otherwise (STAGE divided by f'eval {regime} {policy} seed{seed} {block}.json').
        ROOT / f"results/training_budget/eval_qrdqn_{seed}_40000_{policy}.json"  # Include (ROOT divided by f'results/training budget/eval qrdqn {seed} 40000 {policy}.json') when regime equals 'fixed' and block equals 'main', otherwise (STAGE divided by f'eval {regime} {policy} seed{seed} {block}.json') in the surrounding operation.
        if regime == "fixed" and block == "main"  # Include regime equals 'fixed' and block equals 'main' in the surrounding operation.
        else STAGE / f"eval_{regime}_{policy}_seed{seed}_{block}.json"  # Include (STAGE divided by f'eval {regime} {policy} seed{seed} {block}.json') in the surrounding operation.
    )  # Finish assembling the grouped values above.
    if out.exists():  # A saved file is useful only if its model and settings match this request.
        data = read(out)  # Set data to read applied to out.
        p = data["provenance"]  # Set p to the entries of data selected by 'provenance'.
        if (  # Use this branch when the listed settings or measurements in their recorded order differs from the listed settings or measurements in their recorded order.
            p["checkpoint_sha256"],  # Include the entries of p selected by 'checkpoint_sha256' in the surrounding operation.
            p["alpha"],  # Include the entries of p selected by 'alpha' in the surrounding operation.
            p["speed_governor_cap_m_s"],  # Include the entries of p selected by 'speed_governor_cap_m_s' in the surrounding operation.
            p["seed_start"],  # Include the entries of p selected by 'seed_start' in the surrounding operation.
            p["episodes_per_density"],  # Include the entries of p selected by 'episodes_per_density' in the surrounding operation.
            [r["density"] for r in data["summary"]],  # Include the entries of r selected by 'density' for each r in the entries of data selected by 'summary' in the surrounding operation.
            len(data["episodes"]),  # Include the number of entries in the entries of data selected by 'episodes' in the surrounding operation.
        ) != (  # Finish assembling the grouped values above.
            hashlib.sha256(ck.read_bytes()).hexdigest(),  # Include the result of hashlib.sha256(ck.read bytes()).hexdigest in the surrounding operation.
            alpha,  # Include the fraction of low-return predictions used to score actions in the surrounding operation.
            cap,  # Include cap in the surrounding operation.
            d["seed_start"],  # Include the entries of d selected by 'seed_start' in the surrounding operation.
            d["episodes"],  # Include the entries of d selected by 'episodes' in the surrounding operation.
            d["densities"],  # Include the entries of d selected by 'densities' in the surrounding operation.
            len(d["densities"]) * d["episodes"],  # Include (the number of entries in the entries of d selected by 'densities' times the entries of d selected by 'episodes') in the surrounding operation.
        ):  # Finish assembling the grouped values above.
            raise ValueError(f"Mismatched evaluation: {out}")  # Reject this invalid case instead of continuing with misleading experiment results.
    else:  # Handle the alternative when the earlier condition (the result of out.exists) is false.
        cmd = [  # Set cmd to the listed settings or measurements in their recorded order.
            "driving_rl.cli",  # Include 'driving_rl.cli' in the surrounding operation.
            "evaluate",  # Include 'evaluate' in the surrounding operation.
            "--checkpoint",  # Include '--checkpoint' in the surrounding operation.
            str(ck),  # Include ck as a text value in the surrounding operation.
            "--alpha",  # Include '--alpha' in the surrounding operation.
            str(alpha),  # Include the fraction of low-return predictions used to score actions as a text value in the surrounding operation.
            "--seed-start",  # Include '--seed-start' in the surrounding operation.
            str(d["seed_start"]),  # Include the entries of d selected by 'seed_start' as a text value in the surrounding operation.
            "--episodes",  # Include '--episodes' in the surrounding operation.
            str(d["episodes"]),  # Include the entries of d selected by 'episodes' as a text value in the surrounding operation.
            "--densities",  # Include '--densities' in the surrounding operation.
            *[str(x) for x in d["densities"]],  # Include *[str(x) for x in d['densities']] in the surrounding operation.
            "--out",  # Include '--out' in the surrounding operation.
            str(out),  # Include out as a text value in the surrounding operation.
        ]  # Finish assembling the grouped values above.
        if cap:  # Use this branch when cap.
            cmd += ["--speed-cap", str(cap)]  # Increase cmd using '--speed-cap', cap as a text value.
        run(cmd, out.with_suffix(".log"))  # Run applied to cmd, out.with suffix applied to '.log'.
    print(f"Evaluated {regime} {seed} {policy} {block}", flush=True)  # Print applied to f'evaluated {regime} {seed} {policy} {block}'.
    return out  # Return out.


def calibrate(spec):  # Run prediction checks for the specified saved model, or validate saved checks.
    """Run prediction checks for the specified saved model, or validate saved checks."""  # Describe this operation for readers and help().
    seed, policy = spec  # Set seed, policy to spec.
    alpha, _ = POLICIES[policy]  # Set the fraction of low-return predictions used to score actions, the unused result to the entries of POLICIES selected by policy.
    ck = checkpoint("mixed", seed)  # Set ck to checkpoint applied to 'mixed', seed.
    out = STAGE / f"calibration_{policy}_seed{seed}.json"  # Set out to (STAGE divided by f'calibration {policy} seed{seed}.json').
    if out.exists():  # A saved file is useful only if its model and settings match this request.
        data = read(out)  # Set data to read applied to out.
        p = data["provenance"]  # Set p to the entries of data selected by 'provenance'.
        if (  # Use this branch when the listed settings or measurements in their recorded order differs from the listed settings or measurements in their recorded order.
            p["checkpoint_sha256"],  # Include the entries of p selected by 'checkpoint_sha256' in the surrounding operation.
            p["alpha"],  # Include the entries of p selected by 'alpha' in the surrounding operation.
            p["seed_start"],  # Include the entries of p selected by 'seed_start' in the surrounding operation.
            p["episodes_per_density"],  # Include the entries of p selected by 'episodes_per_density' in the surrounding operation.
            p["speed_cap"],  # Include the entries of p selected by 'speed_cap' in the surrounding operation.
            len(data["episodes"]),  # Include the number of entries in the entries of data selected by 'episodes' in the surrounding operation.
        ) != (hashlib.sha256(ck.read_bytes()).hexdigest(), alpha, 12000, 60, None, 180):  # Finish assembling the grouped values above.
            raise ValueError(f"Mismatched calibration: {out}")  # Reject this invalid case instead of continuing with misleading experiment results.
    else:  # Handle the alternative when the earlier condition (the result of out.exists) is false.
        run(  # Run applied to the listed settings or measurements in their recorded order, out.with suffix applied to '.log'.
            [  # Include the listed settings or measurements in their recorded order in the surrounding operation.
                "driving_rl.calibration",  # Include 'driving_rl.calibration' in the surrounding operation.
                "--checkpoint",  # Include '--checkpoint' in the surrounding operation.
                str(ck),  # Include ck as a text value in the surrounding operation.
                "--alpha",  # Include '--alpha' in the surrounding operation.
                str(alpha),  # Include the fraction of low-return predictions used to score actions as a text value in the surrounding operation.
                "--out",  # Include '--out' in the surrounding operation.
                str(out),  # Include out as a text value in the surrounding operation.
            ],  # Finish assembling the grouped values above.
            out.with_suffix(".log"),  # Include out.with suffix applied to '.log' in the surrounding operation.
        )  # Finish assembling run inputs.
    print(f"Calibrated mixed {seed} {policy}", flush=True)  # Print applied to f'calibrated mixed {seed} {policy}'.


def replay(regime):  # Record the scenario chosen in advance, not a scenario selected because it looks good.
    """Record the scenario chosen in advance, not a scenario selected because it looks good."""  # Describe this operation for readers and help().
    ck = checkpoint(regime, 0)  # Set ck to checkpoint applied to regime, 0.
    out = STAGE / f"replay_{regime}.json"  # Set out to (STAGE divided by f'replay {regime}.json').
    if out.exists():  # A saved file is useful only if its model and settings match this request.
        data = read(out)  # Set data to read applied to out.
        p = data["provenance"]  # Set p to the entries of data selected by 'provenance'.
        if (  # Use this branch when the entries of p selected by 'checkpoint_sha256', the entries of p selected by 'alpha', the entries of p selected by 'seed_start', the entries of p selected by 'episodes_per_density' differs from the result of hashlib.sha256(ck.read bytes()).hexdigest, 0.25, 10000, 1.
            p["checkpoint_sha256"],  # Include the entries of p selected by 'checkpoint_sha256' in the surrounding operation.
            p["alpha"],  # Include the entries of p selected by 'alpha' in the surrounding operation.
            p["seed_start"],  # Include the entries of p selected by 'seed_start' in the surrounding operation.
            p["episodes_per_density"],  # Include the entries of p selected by 'episodes_per_density' in the surrounding operation.
        ) != (hashlib.sha256(ck.read_bytes()).hexdigest(), 0.25, 10000, 1):  # Finish assembling the grouped values above.
            raise ValueError("Replay metadata mismatch")  # Reject this invalid case instead of continuing with misleading experiment results.
    else:  # Handle the alternative when the earlier condition (the result of out.exists) is false.
        run(  # Run applied to the listed settings or measurements in their recorded order, out.with suffix applied to '.log'.
            [  # Include the listed settings or measurements in their recorded order in the surrounding operation.
                "driving_rl.cli",  # Include 'driving_rl.cli' in the surrounding operation.
                "evaluate",  # Include 'evaluate' in the surrounding operation.
                "--checkpoint",  # Include '--checkpoint' in the surrounding operation.
                str(ck),  # Include ck as a text value in the surrounding operation.
                "--alpha",  # Include '--alpha' in the surrounding operation.
                ".25",  # Include '.25' in the surrounding operation.
                "--episodes",  # Include '--episodes' in the surrounding operation.
                "1",  # Include '1' in the surrounding operation.
                "--seed-start",  # Include '--seed-start' in the surrounding operation.
                "10000",  # Include '10000' in the surrounding operation.
                "--densities",  # Include '--densities' in the surrounding operation.
                "2",  # Include '2' in the surrounding operation.
                "--out",  # Include '--out' in the surrounding operation.
                str(out),  # Include out as a text value in the surrounding operation.
                "--gif",  # Include '--gif' in the surrounding operation.
                str(STAGE / f"replay_{regime}.gif"),  # Include (STAGE divided by f'replay {regime}.gif') as a text value in the surrounding operation.
            ],  # Finish assembling the grouped values above.
            out.with_suffix(".log"),  # Include out.with suffix applied to '.log' in the surrounding operation.
        )  # Finish assembling run inputs.
    print(f"Recorded {regime} cautious replay", flush=True)  # Print applied to f'recorded {regime} cautious replay'.


def main():  # Read Terminal options, check settings, and run the requested workflow.
    """Read Terminal options, check settings, and run the requested workflow.
    Existing result files are checked before reuse so a rerun cannot silently mix
    different experiments. The main guard below prevents running jobs on import.
    """
    p = argparse.ArgumentParser(description=__doc__)  # Set p to the result of argparse.ArgumentParser.
    p.add_argument("--jobs", type=int, default=3)  # P.add argument applied to '--jobs'.
    p.add_argument("--stage", choices=["all", "evaluate", "report"], default="all")  # P.add argument applied to '--stage'.
    a = p.parse_args()  # Set a to the result of p.parse args.
    if a.jobs < 1:  # Use this branch when a.jobs is below 1.
        p.error("--jobs must be positive")  # P.error applied to '--jobs must be positive'.
    STAGE.mkdir(parents=True, exist_ok=True)  # The result of stage.mkdir.
    design = STAGE / "design.json"  # Set design to the path to 'design.json' inside STAGE.
    if design.exists() and read(design) != DESIGN:  # Use this branch when the result of design.exists and read applied to design differs from DESIGN.
        raise ValueError("Refusing to change recorded design")  # Reject this invalid case instead of continuing with misleading experiment results.
    if not design.exists():  # Use this branch when not the result of design.exists.
        save_json(design, DESIGN)  # Save the following results/settings as json: design, design.
    if a.stage != "report":  # Use this branch when a.stage differs from 'report'.
        with ThreadPoolExecutor(max_workers=a.jobs) as pool:  # Run independent jobs together. Each writes its own files; this is not a shared training model.
            if a.stage == "all":  # Use this branch when a.stage equals 'all'.
                list(pool.map(train, range(3)))  # List applied to pool.map applied to train, range applied to 3.
            list(  # List applied to pool.map applied to evaluate, regime, s, policy, block for each regime in 'fixed', 'mixed'.
                pool.map(  # Include pool.map applied to evaluate, regime, s, policy, block for each regime in 'fixed', 'mixed' in the surrounding operation.
                    evaluate,  # Include evaluate in the surrounding operation.
                    [  # Include regime, s, policy, block for each regime in 'fixed', 'mixed' in the surrounding operation.
                        (regime, s, policy, block)  # Include regime, s, policy, block in the surrounding operation.
                        for regime in ["fixed", "mixed"]  # Include regime, 'fixed', 'mixed' in the surrounding operation.
                        for s in range(3)  # Include s, range applied to 3 in the surrounding operation.
                        for policy in POLICIES  # Include policy, POLICIES in the surrounding operation.
                        for block in ["main", "transfer"]  # Include block, 'main', 'transfer' in the surrounding operation.
                    ],  # Finish assembling pool.map inputs.
                )  # Finish assembling pool.map inputs.
            )  # Finish assembling list inputs.
            list(  # List applied to pool.map applied to calibrate, s, policy for each s in range applied to 3.
                pool.map(  # Include pool.map applied to calibrate, s, policy for each s in range applied to 3 in the surrounding operation.
                    calibrate,  # Include calibrate in the surrounding operation.
                    [(s, policy) for s in range(3) for policy in ["mean", "tail25"]],  # Include s, policy for each s in range applied to 3 in the surrounding operation.
                )  # Finish assembling pool.map inputs.
            )  # Finish assembling list inputs.
            list(pool.map(replay, ["fixed", "mixed"]))  # List applied to pool.map applied to replay, 'fixed', 'mixed'.
    from driving_rl.density_coverage_report import report  # Make report available for this file's calculations and workflow.

    report()  # The result of report.


if __name__ == "__main__":  # Run this workflow when launched from Terminal; importing its helpers must not launch experiments.
    main()  # The result of main.
