"""Run the training budget experiment from Terminal.

This is an experiment coordinator, not the learning algorithm. It launches
separate Python commands, saves logs, and checks existing results before reuse.
Paths identify saved files; hashes are fingerprints that detect changed files.
Read main() for the order of operations. Learning itself lives in driving_rl/agents.py.

Original implementation notes:
Controlled budget study: train, validate, evaluate and report.
"""

# import makes these modules available; "as" gives a module a shorter local name.
import argparse
# from concurrent.futures import brings the named tools into this file; parentheses allow several lines.
from concurrent.futures import ThreadPoolExecutor
# import makes these modules available; "as" gives a module a shorter local name.
import hashlib
# import makes these modules available; "as" gives a module a shorter local name.
import json
# from pathlib import brings the named tools into this file; parentheses allow several lines.
from pathlib import Path
# import makes these modules available; "as" gives a module a shorter local name.
import subprocess
# import makes these modules available; "as" gives a module a shorter local name.
import sys
# import makes these modules available; "as" gives a module a shorter local name.
import torch
# from driving_rl.experiment import brings the named tools into this file; parentheses allow several lines.
from driving_rl.experiment import save_json

# = stores a result in ROOT; use a dot to access parent on Path(__file__).resolve().
ROOT = Path(__file__).resolve().parent
# = stores a result in STAGE; apply / to these operands; numbers are divided, while Path objects join folder names.
STAGE = ROOT / "results/training_budget"


# def defines run; parentheses list its inputs, and the colon starts its indented body.
def run(command, log):
    """Run a command in a separate Python process and wait for it to finish.
    A nonzero exit status means the command failed; do not silently continue.
    """
    # with enters a temporary context and exits it afterward, even if an error interrupts the block.
    with log.open("w") as f:
        # Start another Python command and check whether it succeeded.
        # Call subprocess.run using parentheses to supply its inputs; this line does not store a separate return value.
        subprocess.run(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            [sys.executable, "-m", *command],
            # Inside this continued call, name=value supplies an option by name rather than by its position.
            cwd=ROOT,
            # Inside this continued call, name=value supplies an option by name rather than by its position.
            stdout=f,
            # Inside this continued call, name=value supplies an option by name rather than by its position.
            stderr=subprocess.STDOUT,
            # Inside this continued call, name=value supplies an option by name rather than by its position.
            check=True,
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )


# def defines train; parentheses list its inputs, and the colon starts its indented body.
def train(spec):
    """Train one missing experiment, or verify its completed saved model.
    Keep model settings fixed across the requested seeds. An incomplete directory
    is not silently resumed because replay memory and RNG state are not saved.
    """
    # = unpacks the right-hand result into (algo, seed), matching values to names in order.
    algo, seed = spec
    # = stores a result in folder; apply / to these operands; numbers are divided, while Path objects join folder names.
    folder = STAGE / f"{algo}_seed{seed}"
    # if checks not (folder / 'summary.json').exists(); only a true condition runs the indented block.
    if not (folder / "summary.json").exists():
        # Call run using parentheses to supply its inputs; this line does not store a separate return value.
        run(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            [
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "driving_rl.cli",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "train",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "--algorithm",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                algo,
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "--seed",
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                str(seed),
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "--steps",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "40000",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "--exploration-steps",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "12000",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "--out",
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                str(folder),
            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
            ],
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            STAGE / f"train_{algo}_{seed}.log",
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
    # = stores a result in c; call json.loads using parentheses to supply its inputs.
    c = json.loads((folder / "config.json").read_text())
    # if checks (c['steps'], c['exploration_steps'], c['seed'], c['algorithm']) != (40000, 12000, seed, algo); only a true condition runs the indented block.
    if (c["steps"], c["exploration_steps"], c["seed"], c["algorithm"]) != (
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        40000,
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        12000,
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        seed,
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        algo,
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    ):
        # raise signals an error and stops this normal path, so invalid data are not silently accepted.
        raise ValueError(f"Mismatched training configuration: {folder}")
    # for takes each item from {'gamma': 0.9, 'learning_rate': 0.0005, 'collision_penalty': -2.0, 'buffer_size': 30000, 'batch_s... in turn; (key, value) holds the current item.
    for key, value in {
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "gamma": 0.9,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "learning_rate": 0.0005,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "collision_penalty": -2.0,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "buffer_size": 30000,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "batch_size": 64,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "learning_starts": 500,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "train_frequency": 1,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "target_interval": 250,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "final_epsilon": 0.05,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "quantiles": 32,
    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
    }.items():
        # if checks c[key] != value; only a true condition runs the indented block.
        if c[key] != value:
            # raise signals an error and stops this normal path, so invalid data are not silently accepted.
            raise ValueError(f"Mismatched {key}: {folder}")
    # = stores a result in old; call torch.load using parentheses to supply its inputs.
    old = torch.load(ROOT / f"results/{algo}_seed{seed}/final.pt", weights_only=True)
    # = stores a result in new; call torch.load using parentheses to supply its inputs.
    new = torch.load(folder / "checkpoint_20000.pt", weights_only=True)
    # = stores a result in match; call all using parentheses to supply its inputs.
    match = all(torch.equal(old["online"][k], new["online"][k]) for k in old["online"])
    # Call print using parentheses to supply its inputs; this line does not store a separate return value.
    print(f"Trained {algo} seed {seed}; exact 20k weights: {match}", flush=True)
    # return ends this function and sends its result back to the code that called it.
    return dict(algorithm=algo, seed=seed, prefix_weights_equal=match)


# def defines evaluate; parentheses list its inputs, and the colon starts its indented body.
def evaluate(spec):
    """Run or verify one frozen-policy evaluation.
    Check its model fingerprint, settings and scenario block before reusing saved
    data. This function must not quietly mix results from different experiments.
    """
    # = unpacks the right-hand result into (algo, seed, steps, policy), matching values to names in order.
    algo, seed, steps, policy = spec
    # = stores a result in ckpt; apply / to these operands; numbers are divided, while Path objects join folder names.
    ckpt = STAGE / f"{algo}_seed{seed}/checkpoint_{steps}.pt"
    # = stores a result in out; apply / to these operands; numbers are divided, while Path objects join folder names.
    out = STAGE / f"eval_{algo}_{seed}_{steps}_{policy}.json"
    # = stores a result in alpha; choose the value before "if" when its condition is true, otherwise the value after "else".
    alpha = 0.25 if policy == "tail25" else 1.0
    # = stores a result in cap; choose the value before "if" when its condition is true, otherwise the value after "else".
    cap = 25.0 if policy == "governed" else None
    # = stores a result in digest; call hashlib.sha256(ckpt.read_bytes()).hexdigest using parentheses to supply its inputs.
    digest = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    # A saved file is useful only if its model and settings match this request.
    # if checks out.exists(); only a true condition runs the indented block.
    if out.exists():
        # = stores a result in p; use square brackets to select part of json.loads(out.read_text()).
        p = json.loads(out.read_text())["provenance"]
        # if checks (p['checkpoint_sha256'], p['alpha'], p['speed_governor_cap_m_s'], p['seed_start'], p['episodes_pe...; only a true condition runs the indented block.
        if (
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            p["checkpoint_sha256"],
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            p["alpha"],
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            p["speed_governor_cap_m_s"],
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            p["seed_start"],
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            p["episodes_per_density"],
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        ) != (digest, alpha, cap, 10000, 30):
            # raise signals an error and stops this normal path, so invalid data are not silently accepted.
            raise ValueError(f"Mismatched evaluation: {out}")
        # = stores a result in data; call json.loads using parentheses to supply its inputs.
        data = json.loads(out.read_text())
        # if checks [r['density'] for r in data['summary']] != [1.0, 1.5, 2.0] or len(data['episodes']) != 90; only a true condition runs the indented block.
        if [r["density"] for r in data["summary"]] != [1.0, 1.5, 2.0] or len(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            data["episodes"]
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        ) != 90:
            # raise signals an error and stops this normal path, so invalid data are not silently accepted.
            raise ValueError(f"Incomplete evaluation: {out}")
    # else provides the alternative branch; after a loop it instead means the loop finished without break.
    else:
        # = stores a result in cmd; build a list, keeping its items in order.
        cmd = [
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "driving_rl.cli",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "evaluate",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "--checkpoint",
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            str(ckpt),
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "--alpha",
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            str(alpha),
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "--episodes",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "30",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "--out",
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            str(out),
        # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
        ]
        # if checks cap; only a true condition runs the indented block.
        if cap:
            # The combined operator updates cmd using its previous value.
            cmd += ["--speed-cap", str(cap)]
        # Call run using parentheses to supply its inputs; this line does not store a separate return value.
        run(cmd, out.with_suffix(".log"))
    # Call print using parentheses to supply its inputs; this line does not store a separate return value.
    print(f"Evaluated {out.name}", flush=True)


# def defines calibrate; parentheses list its inputs, and the colon starts its indented body.
def calibrate(spec):
    # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
    """Run prediction checks for the specified saved model, or validate saved checks."""
    # = unpacks the right-hand result into (seed, policy), matching values to names in order.
    seed, policy = spec
    # = stores a result in ckpt; apply / to these operands; numbers are divided, while Path objects join folder names.
    ckpt = STAGE / f"qrdqn_seed{seed}/checkpoint_40000.pt"
    # = stores a result in out; apply / to these operands; numbers are divided, while Path objects join folder names.
    out = STAGE / f"calibration_{policy}_seed{seed}.json"
    # = stores a result in alpha; choose the value before "if" when its condition is true, otherwise the value after "else".
    alpha = 0.25 if policy == "tail25" else 1.0
    # A saved file is useful only if its model and settings match this request.
    # if checks out.exists(); only a true condition runs the indented block.
    if out.exists():
        # = stores a result in p; use square brackets to select part of json.loads(out.read_text()).
        p = json.loads(out.read_text())["provenance"]
        # if checks (p['checkpoint_sha256'], p['alpha'], p['seed_start'], p['episodes_per_density']) != (hashlib.sha2...; only a true condition runs the indented block.
        if (
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            p["checkpoint_sha256"],
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            p["alpha"],
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            p["seed_start"],
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            p["episodes_per_density"],
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        ) != (hashlib.sha256(ckpt.read_bytes()).hexdigest(), alpha, 12000, 60):
            # raise signals an error and stops this normal path, so invalid data are not silently accepted.
            raise ValueError(f"Mismatched calibration: {out}")
        # = stores a result in data; call json.loads using parentheses to supply its inputs.
        data = json.loads(out.read_text())
        # if checks [r['density'] for r in data['summary']] != [1.0, 1.5, 2.0] or len(data['episodes']) != 180; only a true condition runs the indented block.
        if [r["density"] for r in data["summary"]] != [1.0, 1.5, 2.0] or len(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            data["episodes"]
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        ) != 180:
            # raise signals an error and stops this normal path, so invalid data are not silently accepted.
            raise ValueError(f"Incomplete calibration: {out}")
    # else provides the alternative branch; after a loop it instead means the loop finished without break.
    else:
        # Call run using parentheses to supply its inputs; this line does not store a separate return value.
        run(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            [
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "driving_rl.calibration",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "--checkpoint",
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                str(ckpt),
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "--alpha",
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                str(alpha),
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "--out",
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                str(out),
            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
            ],
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            out.with_suffix(".log"),
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
    # Call print using parentheses to supply its inputs; this line does not store a separate return value.
    print(f"Calibrated {out.name}", flush=True)


# def defines main; parentheses list its inputs, and the colon starts its indented body.
def main():
    """Read Terminal options, check settings, and run the requested workflow.
    Existing result files are checked before reuse so a rerun cannot silently mix
    different experiments. The main guard below prevents running jobs on import.
    """
    # = stores a result in parser; call argparse.ArgumentParser using parentheses to supply its inputs.
    parser = argparse.ArgumentParser(description=__doc__)
    # Call parser.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    parser.add_argument("--jobs", type=int, default=3)
    # = stores a result in args; call parser.parse_args using parentheses to supply its inputs.
    args = parser.parse_args()
    # if checks args.jobs < 1; only a true condition runs the indented block.
    if args.jobs < 1:
        # Call parser.error using parentheses to supply its inputs; this line does not store a separate return value.
        parser.error("--jobs must be positive")
    # Call stage.mkdir using parentheses to supply its inputs; this line does not store a separate return value.
    STAGE.mkdir(parents=True, exist_ok=True)
    # Run independent jobs together. Each writes its own files; this is not a shared training model.
    # with enters a temporary context and exits it afterward, even if an error interrupts the block.
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        # = stores a result in prefixes; call list using parentheses to supply its inputs.
        prefixes = list(
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            pool.map(train, [(a, s) for a in ["dqn", "qrdqn"] for s in range(3)])
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
        # Call save_json using parentheses to supply its inputs; this line does not store a separate return value.
        save_json(STAGE / "prefix_verification.json", prefixes)
        # Call list using parentheses to supply its inputs; this line does not store a separate return value.
        list(
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            pool.map(
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                evaluate,
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                [
                    # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                    (a, s, n, p)
                    # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                    for a in ["dqn", "qrdqn"]
                    # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                    for s in range(3)
                    # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                    for n in [20000, 40000]
                    # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                    for p in (
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        ["mean"] if a == "dqn" else ["mean", "tail25", "governed"]
                    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                    )
                # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
                ],
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            )
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
        # Call list using parentheses to supply its inputs; this line does not store a separate return value.
        list(
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            pool.map(calibrate, [(s, p) for s in range(3) for p in ["mean", "tail25"]])
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
    # from driving_rl.training_budget_report import brings the named tools into this file; parentheses allow several lines.
    from driving_rl.training_budget_report import report

    # Call report using parentheses to supply its inputs; this line does not store a separate return value.
    report()


# Run the entry point only when this file is executed, not when it is imported.
# __name__ equals "__main__" when this file runs directly; this guard avoids running the script merely because another file imports it.
if __name__ == "__main__":
    # Call main using parentheses to supply its inputs; this line does not store a separate return value.
    main()
