"""Run the baseline controls experiment from Terminal.

This is an experiment coordinator, not the learning algorithm. It launches
separate Python commands, saves logs, and checks existing results before reuse.
Paths identify saved files; hashes are fingerprints that detect changed files.
Read main() for the order of operations. Learning itself lives in driving_rl/agents.py.

Original implementation notes:
Equal-coverage scalar baseline and independently speed-matched controls.
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
# from driving_rl.experiment import brings the named tools into this file; parentheses allow several lines.
from driving_rl.experiment import save_json
# from driving_rl.analysis import brings the named tools into this file; parentheses allow several lines.
from driving_rl.analysis import choose_speed_matched_control

# = stores a result in ROOT; use a dot to access parent on Path(__file__).resolve().
ROOT = Path(__file__).resolve().parent
# = stores a result in STAGE; apply / to these operands; numbers are divided, while Path objects join folder names.
STAGE = ROOT / "results/baseline_controls"
# = stores a result in CAPS; build a list, keeping its items in order.
CAPS = [21.0, 22.0, 23.0, 24.0, 25.0]
# = stores a result in DESIGN; build a dictionary, which stores values under named keys.
DESIGN = {
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "training_steps": 40000,
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "training_seeds": [0, 1, 2],
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "training_densities": [1.0, 1.5, 2.0],
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "algorithm": "dqn",
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "epsilon_decay_steps": 12000,
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "comparison": "DQN and QR-DQN with equal interaction budgets, identical configured episode-density mixtures, rewards, network hidden sizes and optimizer settings; actual policy-dependent experience differs.",
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "primary_scenarios": {
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "seed_start": 10000,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "episodes": 30,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "densities": [1.0, 1.5, 2.0],
    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
    },
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "transfer_scenarios": {
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "seed_start": 16000,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "episodes": 30,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "densities": [1.25, 2.5],
    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
    },
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "speed_control": {
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "caps_m_s": CAPS,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "selection": "Independently for mixed DQN and QR mean, choose smallest absolute speed difference from mixed QR lower-quarter at seed 0, density 1; ties prefer lower cap. Collision rates are not selection inputs. Freeze before seeds 1/2.",
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "primary_comparison": "Mixed QR lower-quarter versus selected DQN governor and same-network QR governor. Report residual speeds and exploratory intervals conditional on selection.",
    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
    },
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "no_additional_alpha_or_reward_tuning": True,
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "confirmation_scenarios_used": False,
# } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
}


# def defines read; parentheses list its inputs, and the colon starts its indented body.
def read(p):
    # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
    """Read saved JSON data into ordinary Python dictionaries and lists."""
    # return ends this function and sends its result back to the code that called it.
    return json.loads(p.read_text())


# def defines digest; parentheses list its inputs, and the colon starts its indented body.
def digest(p):
    # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
    """Calculate a SHA-256 fingerprint so saved evaluations can be tied to one exact model."""
    # return ends this function and sends its result back to the code that called it.
    return hashlib.sha256(p.read_bytes()).hexdigest()


# def defines run; parentheses list its inputs, and the colon starts its indented body.
def run(cmd, log):
    """Run a command in a separate Python process and wait for it to finish.
    A nonzero exit status means the command failed; do not silently continue.
    """
    # with enters a temporary context and exits it afterward, even if an error interrupts the block.
    with log.open("w") as f:
        # Start another Python command and check whether it succeeded.
        # Call subprocess.run using parentheses to supply its inputs; this line does not store a separate return value.
        subprocess.run(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            [sys.executable, "-m", *cmd],
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


# def defines checkpoint; parentheses list its inputs, and the colon starts its indented body.
def checkpoint(algorithm, seed, regime="mixed"):
    # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
    """Find the saved model file for the requested algorithm and training seed."""
    # if checks regime == 'fixed'; only a true condition runs the indented block.
    if regime == "fixed":
        # return ends this function and sends its result back to the code that called it.
        return ROOT / f"results/training_budget/dqn_seed{seed}/checkpoint_40000.pt"
    # return ends this function and sends its result back to the code that called it.
    return (
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        STAGE / f"dqn_seed{seed}/final.pt"
        # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
        if algorithm == "dqn"
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        else ROOT / f"results/density_coverage/qrdqn_seed{seed}/final.pt"
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )


# def defines train; parentheses list its inputs, and the colon starts its indented body.
def train(seed):
    """Train one missing experiment, or verify its completed saved model.
    Keep model settings fixed across the requested seeds. An incomplete directory
    is not silently resumed because replay memory and RNG state are not saved.
    """
    # = stores a result in out; apply / to these operands; numbers are divided, while Path objects join folder names.
    out = STAGE / f"dqn_seed{seed}"
    # if checks not (out / 'summary.json').exists(); only a true condition runs the indented block.
    if not (out / "summary.json").exists():
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
                "dqn",
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
                "--training-densities",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "1",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "1.5",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "2",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "--out",
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                str(out),
            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
            ],
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            STAGE / f"train_seed{seed}.log",
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
    # = stores a result in c; call read using parentheses to supply its inputs.
    c = read(out / "config.json")
    # = stores a result in reference; call read using parentheses to supply its inputs.
    reference = read(ROOT / f"results/density_coverage/qrdqn_seed{seed}/config.json")
    # for takes each item from ['seed', 'steps', 'training_densities', 'exploration_steps', 'gamma', 'learning_rate', 'collision... in turn; k holds the current item.
    for k in [
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "seed",
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "steps",
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "training_densities",
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "exploration_steps",
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "gamma",
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "learning_rate",
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "collision_penalty",
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "buffer_size",
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "batch_size",
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "learning_starts",
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "train_frequency",
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "target_interval",
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "final_epsilon",
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "quantiles",
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "observation_dim",
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "n_actions",
    # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
    ]:
        # if checks c[k] != reference[k]; only a true condition runs the indented block.
        if c[k] != reference[k]:
            # raise signals an error and stops this normal path, so invalid data are not silently accepted.
            raise ValueError(f"Configuration mismatch: {k}")
    # if checks c['algorithm'] != 'dqn' or not read(out / 'summary.json')['training_complete']; only a true condition runs the indented block.
    if c["algorithm"] != "dqn" or not read(out / "summary.json")["training_complete"]:
        # raise signals an error and stops this normal path, so invalid data are not silently accepted.
        raise ValueError("Incomplete/mismatched baseline")
    # Call print using parentheses to supply its inputs; this line does not store a separate return value.
    print(f"Trained mixed DQN seed {seed}", flush=True)


# def defines evaluate; parentheses list its inputs, and the colon starts its indented body.
def evaluate(spec):
    """Run or verify one frozen-policy evaluation.
    Check its model fingerprint, settings and scenario block before reusing saved
    data. This function must not quietly mix results from different experiments.
    """
    # = unpacks the right-hand result into (algorithm, regime, seed, cap, block), matching values to names in order.
    algorithm, regime, seed, cap, block = spec
    # = stores a result in config; use square brackets to select part of DESIGN.
    config = DESIGN[
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "primary_scenarios" if block in ["main", "selection"] else "transfer_scenarios"
    # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
    ]
    # = stores a result in densities; choose the value before "if" when its condition is true, otherwise the value after "else".
    densities = [1.0] if block == "selection" else config["densities"]
    # = stores a result in ck; call checkpoint using parentheses to supply its inputs.
    ck = checkpoint(algorithm, seed, regime)
    # = stores a result in token; choose the value before "if" when its condition is true, otherwise the value after "else".
    token = "mean" if cap is None else f"cap{cap:g}".replace(".", "p")
    # = stores a result in out; apply / to these operands; numbers are divided, while Path objects join folder names.
    out = STAGE / f"eval_{algorithm}_{regime}_{token}_seed{seed}_{block}.json"
    # if checks algorithm == 'dqn' and regime == 'fixed' and (block == 'main') and (cap is None); only a true condition runs the indented block.
    if algorithm == "dqn" and regime == "fixed" and block == "main" and cap is None:
        # = stores a result in out; apply / to these operands; numbers are divided, while Path objects join folder names.
        out = ROOT / f"results/training_budget/eval_dqn_{seed}_40000_mean.json"
    # A saved file is useful only if its model and settings match this request.
    # if checks out.exists(); only a true condition runs the indented block.
    if out.exists():
        # = stores a result in data; call read using parentheses to supply its inputs.
        data = read(out)
        # = stores a result in p; use square brackets to select part of data.
        p = data["provenance"]
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
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            [r["density"] for r in data["summary"]],
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            len(data["episodes"]),
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        ) != (
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            digest(ck),
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            1.0,
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            cap,
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            config["seed_start"],
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            30,
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            densities,
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            30 * len(densities),
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        ):
            # raise signals an error and stops this normal path, so invalid data are not silently accepted.
            raise ValueError(f"Mismatched evaluation {out}")
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
            str(ck),
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "--episodes",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "30",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "--seed-start",
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            str(config["seed_start"]),
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "--densities",
            # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
            *[str(d) for d in densities],
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "--out",
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            str(out),
        # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
        ]
        # if checks cap is not None; only a true condition runs the indented block.
        if cap is not None:
            # The combined operator updates cmd using its previous value.
            cmd += ["--speed-cap", str(cap)]
        # Call run using parentheses to supply its inputs; this line does not store a separate return value.
        run(cmd, out.with_suffix(".log"))
    # Call print using parentheses to supply its inputs; this line does not store a separate return value.
    print(f"Evaluated {out.name}", flush=True)
    # return ends this function and sends its result back to the code that called it.
    return out


# def defines choose; parentheses list its inputs, and the colon starts its indented body.
def choose(algorithm):
    """Select a threshold using the recorded speed-only rule, then save the choice.
    Reusing a different recorded choice is an error: later results must not silently
    change the rule selected during development.
    """
    # = stores a result in target_path; apply / to these operands; numbers are divided, while Path objects join folder names.
    target_path = ROOT / "results/density_coverage/eval_mixed_tail25_seed0_main.json"
    # = stores a result in target; use square brackets to select part of read(target_path)['summary'][0].
    target = read(target_path)["summary"][0]["mean_speed_m_s"]
    # = stores a result in rows; build a list, keeping its items in order.
    rows = []
    # for takes each item from CAPS in turn; cap holds the current item.
    for cap in CAPS:
        # = stores a result in p; apply / to these operands; numbers are divided, while Path objects join folder names.
        p = STAGE / f"eval_{algorithm}_mixed_cap{cap:g}_seed0_selection.json"
        # = stores a result in speed; use square brackets to select part of read(p)['summary'][0].
        speed = read(p)["summary"][0]["mean_speed_m_s"]
        # Call rows.append using parentheses to supply its inputs; this line does not store a separate return value.
        rows.append(
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            {"cap_m_s": cap, "speed_m_s": speed, "evaluation_sha256": digest(p)}
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
    # = stores a result in selected; call choose_speed_matched_control using parentheses to supply its inputs.
    selected = choose_speed_matched_control(rows, target)
    # = stores a result in result; build a dictionary, which stores values under named keys.
    result = {
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "algorithm": algorithm,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "rule": DESIGN["speed_control"]["selection"],
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "target_speed_m_s": target,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "target_evaluation_sha256": digest(target_path),
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "candidates": rows,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "selected_cap_m_s": selected["cap_m_s"],
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "seed0_speed_difference_m_s": selected["speed_m_s"] - target,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "confirmation_scenarios_used": False,
    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
    }
    # = stores a result in path; apply / to these operands; numbers are divided, while Path objects join folder names.
    path = STAGE / f"{algorithm}_governor_selection.json"
    # if checks path.exists() and read(path) != result; only a true condition runs the indented block.
    if path.exists() and read(path) != result:
        # raise signals an error and stops this normal path, so invalid data are not silently accepted.
        raise ValueError("Refusing to alter recorded governor selection")
    # if checks not path.exists(); only a true condition runs the indented block.
    if not path.exists():
        # Call save_json using parentheses to supply its inputs; this line does not store a separate return value.
        save_json(path, result)
    # Call print using parentheses to supply its inputs; this line does not store a separate return value.
    print(f'Frozen {algorithm} governor: {selected["cap_m_s"]} m/s', flush=True)
    # return ends this function and sends its result back to the code that called it.
    return selected["cap_m_s"]


# def defines main; parentheses list its inputs, and the colon starts its indented body.
def main():
    """Read Terminal options, check settings, and run the requested workflow.
    Existing result files are checked before reuse so a rerun cannot silently mix
    different experiments. The main guard below prevents running jobs on import.
    """
    # = stores a result in p; call argparse.ArgumentParser using parentheses to supply its inputs.
    p = argparse.ArgumentParser(description=__doc__)
    # Call p.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    p.add_argument("--jobs", type=int, default=3)
    # Call p.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    p.add_argument("--stage", choices=["all", "evaluate", "report"], default="all")
    # = stores a result in a; call p.parse_args using parentheses to supply its inputs.
    a = p.parse_args()
    # if checks a.jobs < 1; only a true condition runs the indented block.
    if a.jobs < 1:
        # Call p.error using parentheses to supply its inputs; this line does not store a separate return value.
        p.error("--jobs must be positive")
    # Call stage.mkdir using parentheses to supply its inputs; this line does not store a separate return value.
    STAGE.mkdir(parents=True, exist_ok=True)
    # = stores a result in d; apply / to these operands; numbers are divided, while Path objects join folder names.
    d = STAGE / "design.json"
    # if checks d.exists() and read(d) != DESIGN; only a true condition runs the indented block.
    if d.exists() and read(d) != DESIGN:
        # raise signals an error and stops this normal path, so invalid data are not silently accepted.
        raise ValueError("Recorded design mismatch")
    # if checks not d.exists(); only a true condition runs the indented block.
    if not d.exists():
        # Call save_json using parentheses to supply its inputs; this line does not store a separate return value.
        save_json(d, DESIGN)
    # if checks a.stage != 'report'; only a true condition runs the indented block.
    if a.stage != "report":
        # Run independent jobs together. Each writes its own files; this is not a shared training model.
        # with enters a temporary context and exits it afterward, even if an error interrupts the block.
        with ThreadPoolExecutor(max_workers=a.jobs) as pool:
            # if checks a.stage == 'all'; only a true condition runs the indented block.
            if a.stage == "all":
                # Call list using parentheses to supply its inputs; this line does not store a separate return value.
                list(pool.map(train, range(3)))
            # Call list using parentheses to supply its inputs; this line does not store a separate return value.
            list(
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                pool.map(
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    evaluate,
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    [
                        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                        (algo, "mixed", 0, cap, "selection")
                        # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                        for algo in ["dqn", "qrdqn"]
                        # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                        for cap in CAPS
                    # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
                    ],
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                )
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            )
            # = stores a result in selections; collect or generate one result per item; the "for" part names the item being processed.
            selections = {algo: choose(algo) for algo in ["dqn", "qrdqn"]}
            # = stores a result in tasks; collect or generate one result per item; the "for" part names the item being processed.
            tasks = [
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                ("dqn", regime, s, None, block)
                # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                for regime in ["fixed", "mixed"]
                # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                for s in range(3)
                # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                for block in ["main", "transfer"]
            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
            ]
            # The combined operator updates tasks using its previous value.
            tasks += [
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                (algo, "mixed", s, cap, block)
                # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                for algo, cap in selections.items()
                # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                for s in range(3)
                # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                for block in ["main", "transfer"]
            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
            ]
            # Call list using parentheses to supply its inputs; this line does not store a separate return value.
            list(pool.map(evaluate, tasks))
    # from driving_rl.baseline_controls_report import brings the named tools into this file; parentheses allow several lines.
    from driving_rl.baseline_controls_report import report

    # Call report using parentheses to supply its inputs; this line does not store a separate return value.
    report()


# Run the entry point only when this file is executed, not when it is imported.
# __name__ equals "__main__" when this file runs directly; this guard avoids running the script merely because another file imports it.
if __name__ == "__main__":
    # Call main using parentheses to supply its inputs; this line does not store a separate return value.
    main()
