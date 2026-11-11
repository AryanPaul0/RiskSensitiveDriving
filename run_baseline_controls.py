"""Run the baseline controls experiment from Terminal.

This is an experiment coordinator, not the learning algorithm. It launches
separate Python commands, saves logs, and checks existing results before reuse.
Paths identify saved files; hashes are fingerprints that detect changed files.
Read main() for the order of operations. Learning itself lives in driving_rl/agents.py.

Original implementation notes:
Equal-coverage scalar baseline and independently speed-matched controls.
"""

import argparse  # Make argparse available for this file's calculations and workflow.
from concurrent.futures import ThreadPoolExecutor  # Make ThreadPoolExecutor available for this file's calculations and workflow.
import hashlib  # Make hashlib available for this file's calculations and workflow.
import json  # Make json available for this file's calculations and workflow.
from pathlib import Path  # Make Path available for this file's calculations and workflow.
import subprocess  # Make subprocess available for this file's calculations and workflow.
import sys  # Make sys available for this file's calculations and workflow.
from driving_rl.experiment import save_json  # Make save_json available for this file's calculations and workflow.
from driving_rl.analysis import choose_speed_matched_control  # Make choose_speed_matched_control available for this file's calculations and workflow.

ROOT = Path(__file__).resolve().parent  # Set ROOT to Path( file ).resolve().parent.
STAGE = ROOT / "results/baseline_controls"  # Set STAGE to the path to 'results/baseline_controls' inside ROOT.
CAPS = [21.0, 22.0, 23.0, 24.0, 25.0]  # Set CAPS to 21.0, 22.0, 23.0, 24.0, 25.0.
DESIGN = {  # Set DESIGN to the named settings or measurements assembled below.
    "training_steps": 40000,  # Record 'training_steps' as 40000.
    "training_seeds": [0, 1, 2],  # Record 'training_seeds' as 0, 1, 2.
    "training_densities": [1.0, 1.5, 2.0],  # Record 'training_densities' as 1.0, 1.5, 2.0.
    "algorithm": "dqn",  # Record 'algorithm' as 'dqn'.
    "epsilon_decay_steps": 12000,  # Record 'epsilon_decay_steps' as 12000.
    "comparison": "DQN and QR-DQN with equal interaction budgets, identical configured episode-density mixtures, rewards, network hidden sizes and optimizer settings; actual policy-dependent experience differs.",  # Record 'comparison' as the report text.
    "primary_scenarios": {  # Record 'primary_scenarios' as the named settings or measurements assembled below.
        "seed_start": 10000,  # Record 'seed_start' as 10000.
        "episodes": 30,  # Record 'episodes' as 30.
        "densities": [1.0, 1.5, 2.0],  # Record 'densities' as 1.0, 1.5, 2.0.
    },  # Finish assembling the grouped values above.
    "transfer_scenarios": {  # Record 'transfer_scenarios' as the named settings or measurements assembled below.
        "seed_start": 16000,  # Record 'seed_start' as 16000.
        "episodes": 30,  # Record 'episodes' as 30.
        "densities": [1.25, 2.5],  # Record 'densities' as 1.25, 2.5.
    },  # Finish assembling the grouped values above.
    "speed_control": {  # Record 'speed_control' as the named settings or measurements assembled below.
        "caps_m_s": CAPS,  # Record 'caps_m_s' as CAPS.
        "selection": "Independently for mixed DQN and QR mean, choose smallest absolute speed difference from mixed QR lower-quarter at seed 0, density 1; ties prefer lower cap. Collision rates are not selection inputs. Freeze before seeds 1/2.",  # Record 'selection' as the report text.
        "primary_comparison": "Mixed QR lower-quarter versus selected DQN governor and same-network QR governor. Report residual speeds and exploratory intervals conditional on selection.",  # Record 'primary_comparison' as the report text.
    },  # Finish assembling the grouped values above.
    "no_additional_alpha_or_reward_tuning": True,  # Record 'no_additional_alpha_or_reward_tuning' as True.
    "confirmation_scenarios_used": False,  # Record 'confirmation_scenarios_used' as False.
}  # Finish assembling the grouped values above.


def read(p):  # Read saved JSON data into ordinary Python dictionaries and lists.
    """Read saved JSON data into ordinary Python dictionaries and lists."""  # Describe this operation for readers and help().
    return json.loads(p.read_text())  # Return json.loads applied to the result of p.read text.


def digest(p):  # Calculate a SHA-256 fingerprint so saved evaluations can be tied to one exact model.
    """Calculate a SHA-256 fingerprint so saved evaluations can be tied to one exact model."""  # Describe this operation for readers and help().
    return hashlib.sha256(p.read_bytes()).hexdigest()  # Return the result of hashlib.sha256(p.read bytes()).hexdigest.


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


def checkpoint(algorithm, seed, regime="mixed"):  # Find the saved model file for the requested algorithm and training seed.
    """Find the saved model file for the requested algorithm and training seed."""  # Describe this operation for readers and help().
    if regime == "fixed":  # Use this branch when regime equals 'fixed'.
        return ROOT / f"results/training_budget/dqn_seed{seed}/checkpoint_40000.pt"  # Return (ROOT divided by f'results/training budget/dqn seed{seed}/checkpoint 40000.pt').
    return (  # Return (STAGE divided by f'dqn seed{seed}/final.pt') when algorithm equals 'dqn', otherwise (ROOT divided by f'results/density coverage/qrdqn seed{seed}/final.pt').
        STAGE / f"dqn_seed{seed}/final.pt"  # Include (STAGE divided by f'dqn seed{seed}/final.pt') when algorithm equals 'dqn', otherwise (ROOT divided by f'results/density coverage/qrdqn seed{seed}/final.pt') in the surrounding operation.
        if algorithm == "dqn"  # Include algorithm equals 'dqn' in the surrounding operation.
        else ROOT / f"results/density_coverage/qrdqn_seed{seed}/final.pt"  # Include (ROOT divided by f'results/density coverage/qrdqn seed{seed}/final.pt') in the surrounding operation.
    )  # Finish assembling the grouped values above.


def train(seed):  # Train one missing experiment, or verify its completed saved model.
    """Train one missing experiment, or verify its completed saved model.
    Keep model settings fixed across the requested seeds. An incomplete directory
    is not silently resumed because replay memory and RNG state are not saved.
    """
    out = STAGE / f"dqn_seed{seed}"  # Set out to (STAGE divided by f'dqn seed{seed}').
    if not (out / "summary.json").exists():  # Use this branch when not the result of (out / 'summary.json').exists.
        run(  # Run applied to the listed settings or measurements in their recorded order, (stage divided by f'train seed{seed}.log').
            [  # Include the listed settings or measurements in their recorded order in the surrounding operation.
                "driving_rl.cli",  # Include 'driving_rl.cli' in the surrounding operation.
                "train",  # Include 'train' in the surrounding operation.
                "--algorithm",  # Include '--algorithm' in the surrounding operation.
                "dqn",  # Include 'dqn' in the surrounding operation.
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
                str(out),  # Include out as a text value in the surrounding operation.
            ],  # Finish assembling the grouped values above.
            STAGE / f"train_seed{seed}.log",  # Include (STAGE divided by f'train seed{seed}.log') in the surrounding operation.
        )  # Finish assembling run inputs.
    c = read(out / "config.json")  # Set c to read applied to the path to 'config.json' inside out.
    reference = read(ROOT / f"results/density_coverage/qrdqn_seed{seed}/config.json")  # Set reference to read applied to (ROOT divided by f'results/density coverage/qrdqn seed{seed}/config.json').
    for k in [  # Process each k from the listed settings or measurements in their recorded order.
        "seed",  # Include 'seed' in the surrounding operation.
        "steps",  # Include 'steps' in the surrounding operation.
        "training_densities",  # Include 'training_densities' in the surrounding operation.
        "exploration_steps",  # Include 'exploration_steps' in the surrounding operation.
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
        "observation_dim",  # Include 'observation_dim' in the surrounding operation.
        "n_actions",  # Include 'n_actions' in the surrounding operation.
    ]:  # Finish assembling the grouped values above.
        if c[k] != reference[k]:  # Use this branch when the entries of c selected by k differs from the entries of reference selected by k.
            raise ValueError(f"Configuration mismatch: {k}")  # Reject this invalid case instead of continuing with misleading experiment results.
    if c["algorithm"] != "dqn" or not read(out / "summary.json")["training_complete"]:  # Use this branch when the entries of c selected by 'algorithm' differs from 'dqn' or not the entries of read applied to out / 'summary.json' selected by 'training_complete'.
        raise ValueError("Incomplete/mismatched baseline")  # Reject this invalid case instead of continuing with misleading experiment results.
    print(f"Trained mixed DQN seed {seed}", flush=True)  # Print applied to f'trained mixed dqn seed {seed}'.


def evaluate(spec):  # Run or verify one frozen-policy evaluation.
    """Run or verify one frozen-policy evaluation.
    Check its model fingerprint, settings and scenario block before reusing saved
    data. This function must not quietly mix results from different experiments.
    """
    algorithm, regime, seed, cap, block = spec  # Set algorithm, regime, seed, cap, block to spec.
    config = DESIGN[  # Set config to the entries of DESIGN selected by 'primary_scenarios' if block in ['main', 'selection'] else 'transfer_scenarios'.
        "primary_scenarios" if block in ["main", "selection"] else "transfer_scenarios"  # Include 'primary_scenarios' when block belongs to 'main', 'selection', otherwise 'transfer_scenarios' in the surrounding operation.
    ]  # Finish assembling the grouped values above.
    densities = [1.0] if block == "selection" else config["densities"]  # Set densities to 1.0 when block equals 'selection', otherwise the entries of config selected by 'densities'.
    ck = checkpoint(algorithm, seed, regime)  # Set ck to checkpoint applied to algorithm, seed, regime.
    token = "mean" if cap is None else f"cap{cap:g}".replace(".", "p")  # Set token to 'mean' when cap is None, otherwise f'cap{cap:g}'.replace applied to '.', 'p'.
    out = STAGE / f"eval_{algorithm}_{regime}_{token}_seed{seed}_{block}.json"  # Set out to (STAGE divided by f'eval {algorithm} {regime} {token} seed{seed} {block}.json').
    if algorithm == "dqn" and regime == "fixed" and block == "main" and cap is None:  # Use this branch when algorithm equals 'dqn' and regime equals 'fixed' and block equals 'main' and cap is None.
        out = ROOT / f"results/training_budget/eval_dqn_{seed}_40000_mean.json"  # Set out to (ROOT divided by f'results/training budget/eval dqn {seed} 40000 mean.json').
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
            digest(ck),  # Include digest applied to ck in the surrounding operation.
            1.0,  # Include 1.0 in the surrounding operation.
            cap,  # Include cap in the surrounding operation.
            config["seed_start"],  # Include the entries of config selected by 'seed_start' in the surrounding operation.
            30,  # Include 30 in the surrounding operation.
            densities,  # Include densities in the surrounding operation.
            30 * len(densities),  # Include (30 times the number of entries in densities) in the surrounding operation.
        ):  # Finish assembling the grouped values above.
            raise ValueError(f"Mismatched evaluation {out}")  # Reject this invalid case instead of continuing with misleading experiment results.
    else:  # Handle the alternative when the earlier condition (the result of out.exists) is false.
        cmd = [  # Set cmd to the listed settings or measurements in their recorded order.
            "driving_rl.cli",  # Include 'driving_rl.cli' in the surrounding operation.
            "evaluate",  # Include 'evaluate' in the surrounding operation.
            "--checkpoint",  # Include '--checkpoint' in the surrounding operation.
            str(ck),  # Include ck as a text value in the surrounding operation.
            "--episodes",  # Include '--episodes' in the surrounding operation.
            "30",  # Include '30' in the surrounding operation.
            "--seed-start",  # Include '--seed-start' in the surrounding operation.
            str(config["seed_start"]),  # Include the entries of config selected by 'seed_start' as a text value in the surrounding operation.
            "--densities",  # Include '--densities' in the surrounding operation.
            *[str(d) for d in densities],  # Include *[str(d) for d in densities] in the surrounding operation.
            "--out",  # Include '--out' in the surrounding operation.
            str(out),  # Include out as a text value in the surrounding operation.
        ]  # Finish assembling the grouped values above.
        if cap is not None:  # Use this branch when cap is not None.
            cmd += ["--speed-cap", str(cap)]  # Increase cmd using '--speed-cap', cap as a text value.
        run(cmd, out.with_suffix(".log"))  # Run applied to cmd, out.with suffix applied to '.log'.
    print(f"Evaluated {out.name}", flush=True)  # Print applied to f'evaluated {out.name}'.
    return out  # Return out.


def choose(algorithm):  # Select a threshold using the recorded speed-only rule, then save the choice.
    """Select a threshold using the recorded speed-only rule, then save the choice.
    Reusing a different recorded choice is an error: later results must not silently
    change the rule selected during development.
    """
    target_path = ROOT / "results/density_coverage/eval_mixed_tail25_seed0_main.json"  # Set target path to the path to 'results/density_coverage/eval_mixed_tail25_seed0_main.json' inside ROOT.
    target = read(target_path)["summary"][0]["mean_speed_m_s"]  # Set target to the entries of the entries of the entries of read applied to target path selected by 'summary' selected by 0 selected by 'mean_speed_m_s'.
    rows = []  # Set rows to .
    for cap in CAPS:  # Process each cap from CAPS.
        p = STAGE / f"eval_{algorithm}_mixed_cap{cap:g}_seed0_selection.json"  # Set p to (STAGE divided by f'eval {algorithm} mixed cap{cap:g} seed0 selection.json').
        speed = read(p)["summary"][0]["mean_speed_m_s"]  # Set speed to the entries of the entries of the entries of read applied to p selected by 'summary' selected by 0 selected by 'mean_speed_m_s'.
        rows.append(  # Add the named settings or measurements assembled below to rows.
            {"cap_m_s": cap, "speed_m_s": speed, "evaluation_sha256": digest(p)}  # Record 'cap_m_s' as cap.
        )  # Finish assembling rows.append inputs.
    selected = choose_speed_matched_control(rows, target)  # Set the selected next-action return predictions to choose speed matched control applied to rows, target.
    result = {  # Set result to the named settings or measurements assembled below.
        "algorithm": algorithm,  # Record 'algorithm' as algorithm.
        "rule": DESIGN["speed_control"]["selection"],  # Record 'rule' as the entries of the entries of DESIGN selected by 'speed_control' selected by 'selection'.
        "target_speed_m_s": target,  # Record 'target_speed_m_s' as target.
        "target_evaluation_sha256": digest(target_path),  # Record 'target_evaluation_sha256' as digest applied to target path.
        "candidates": rows,  # Record 'candidates' as rows.
        "selected_cap_m_s": selected["cap_m_s"],  # Record 'selected_cap_m_s' as the entries of the selected next-action return predictions selected by 'cap_m_s'.
        "seed0_speed_difference_m_s": selected["speed_m_s"] - target,  # Record 'seed0_speed_difference_m_s' as (the entries of the selected next-action return predictions selected by 'speed_m_s' minus target).
        "confirmation_scenarios_used": False,  # Record 'confirmation_scenarios_used' as False.
    }  # Finish assembling the grouped values above.
    path = STAGE / f"{algorithm}_governor_selection.json"  # Set path to (STAGE divided by f'{algorithm} governor selection.json').
    if path.exists() and read(path) != result:  # Use this branch when the result of path.exists and read applied to path differs from result.
        raise ValueError("Refusing to alter recorded governor selection")  # Reject this invalid case instead of continuing with misleading experiment results.
    if not path.exists():  # Use this branch when not the result of path.exists.
        save_json(path, result)  # Save the following results/settings as json: path, result.
    print(f'Frozen {algorithm} governor: {selected["cap_m_s"]} m/s', flush=True)  # Print applied to f"frozen {algorithm} governor: {selected['cap m s']} m/s".
    return selected["cap_m_s"]  # Return the entries of the selected next-action return predictions selected by 'cap_m_s'.


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
    d = STAGE / "design.json"  # Set d to the path to 'design.json' inside STAGE.
    if d.exists() and read(d) != DESIGN:  # Use this branch when the result of d.exists and read applied to d differs from DESIGN.
        raise ValueError("Recorded design mismatch")  # Reject this invalid case instead of continuing with misleading experiment results.
    if not d.exists():  # Use this branch when not the result of d.exists.
        save_json(d, DESIGN)  # Save the following results/settings as json: d, design.
    if a.stage != "report":  # Use this branch when a.stage differs from 'report'.
        with ThreadPoolExecutor(max_workers=a.jobs) as pool:  # Run independent jobs together. Each writes its own files; this is not a shared training model.
            if a.stage == "all":  # Use this branch when a.stage equals 'all'.
                list(pool.map(train, range(3)))  # List applied to pool.map applied to train, range applied to 3.
            list(  # List applied to pool.map applied to evaluate, algo, 'mixed', 0, cap, 'selection' for each algo in 'dqn', 'qrdqn'.
                pool.map(  # Include pool.map applied to evaluate, algo, 'mixed', 0, cap, 'selection' for each algo in 'dqn', 'qrdqn' in the surrounding operation.
                    evaluate,  # Include evaluate in the surrounding operation.
                    [  # Include algo, 'mixed', 0, cap, 'selection' for each algo in 'dqn', 'qrdqn' in the surrounding operation.
                        (algo, "mixed", 0, cap, "selection")  # Include algo, 'mixed', 0, cap, 'selection' in the surrounding operation.
                        for algo in ["dqn", "qrdqn"]  # Include algo, 'dqn', 'qrdqn' in the surrounding operation.
                        for cap in CAPS  # Include cap, CAPS in the surrounding operation.
                    ],  # Finish assembling pool.map inputs.
                )  # Finish assembling pool.map inputs.
            )  # Finish assembling list inputs.
            selections = {algo: choose(algo) for algo in ["dqn", "qrdqn"]}  # Set selections to choose applied to algo for each algo in 'dqn', 'qrdqn'.
            tasks = [  # Set tasks to 'dqn', regime, s, None, block for each regime in 'fixed', 'mixed'.
                ("dqn", regime, s, None, block)  # Include 'dqn', regime, s, None, block in the surrounding operation.
                for regime in ["fixed", "mixed"]  # Include regime, 'fixed', 'mixed' in the surrounding operation.
                for s in range(3)  # Include s, range applied to 3 in the surrounding operation.
                for block in ["main", "transfer"]  # Include block, 'main', 'transfer' in the surrounding operation.
            ]  # Finish assembling the grouped values above.
            tasks += [  # Increase tasks using algo, 'mixed', s, cap, block for each algo, cap in the result of selections.items.
                (algo, "mixed", s, cap, block)  # Include algo, 'mixed', s, cap, block in the surrounding operation.
                for algo, cap in selections.items()  # Include algo, cap, the result of selections.items in the surrounding operation.
                for s in range(3)  # Include s, range applied to 3 in the surrounding operation.
                for block in ["main", "transfer"]  # Include block, 'main', 'transfer' in the surrounding operation.
            ]  # Finish assembling the grouped values above.
            list(pool.map(evaluate, tasks))  # List applied to pool.map applied to evaluate, tasks.
    from driving_rl.baseline_controls_report import report  # Make report available for this file's calculations and workflow.

    report()  # The result of report.


if __name__ == "__main__":  # Run this workflow when launched from Terminal; importing its helpers must not launch experiments.
    main()  # The result of main.
