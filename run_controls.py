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

# from driving_rl.analysis import brings the named tools into this file; parentheses allow several lines.
from driving_rl.analysis import choose_speed_targeted_baseline
# from driving_rl.experiment import brings the named tools into this file; parentheses allow several lines.
from driving_rl.experiment import save_json

# = stores a result in ROOT; use a dot to access parent on Path(__file__).resolve().
ROOT = Path(__file__).resolve().parent
# = stores a result in RESULTS; apply / to these operands; numbers are divided, while Path objects join folder names.
RESULTS = ROOT / "results"
# = stores a result in STAGE; apply / to these operands; numbers are divided, while Path objects join folder names.
STAGE = RESULTS / "controls"


# def defines execute; parentheses list its inputs, and the colon starts its indented body.
def execute(arguments, log):
    """Run the requested command with this project's Python interpreter.
    Save its console messages in a log so failures can be inspected later.
    """
    # with enters a temporary context and exits it afterward, even if an error interrupts the block.
    with Path(log).open("w") as file:
        # Start another Python command and check whether it succeeded.
        # Call subprocess.run using parentheses to supply its inputs; this line does not store a separate return value.
        subprocess.run(
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            [sys.executable, "-m", "driving_rl.cli", *map(str, arguments)],
            # Inside this continued call, name=value supplies an option by name rather than by its position.
            stdout=file,
            # Inside this continued call, name=value supplies an option by name rather than by its position.
            stderr=subprocess.STDOUT,
            # Inside this continued call, name=value supplies an option by name rather than by its position.
            check=True,
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )


# def defines ensure_training; parentheses list its inputs, and the colon starts its indented body.
def ensure_training(penalty, seed):
    # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
    """Train a missing model, or check that the saved run really matches the requested settings."""
    # = stores a result in directory; apply / to these operands; numbers are divided, while Path objects join folder names.
    directory = RESULTS / f"dqn_penalty{abs(int(penalty))}_seed{seed}"
    # if checks penalty == -2; only a true condition runs the indented block.
    if penalty == -2:
        # = stores a result in directory; apply / to these operands; numbers are divided, while Path objects join folder names.
        directory = RESULTS / f"dqn_seed{seed}"
    # = stores a result in checkpoint; apply / to these operands; numbers are divided, while Path objects join folder names.
    checkpoint = directory / "final.pt"
    # if checks checkpoint.exists(); only a true condition runs the indented block.
    if checkpoint.exists():
        # = stores a result in config; call json.loads using parentheses to supply its inputs.
        config = json.loads((directory / "config.json").read_text())
        # = stores a result in summary; call json.loads using parentheses to supply its inputs.
        summary = json.loads((directory / "summary.json").read_text())
        # = stores a result in expected; build a dictionary, which stores values under named keys.
        expected = {
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "steps": 20000,
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "seed": seed,
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "algorithm": "dqn",
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "gamma": 0.9,
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "learning_rate": 0.0005,
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "batch_size": 64,
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "target_interval": 250,
        # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
        }
        # if checks not summary['training_complete'] or config.get('collision_penalty', -2.0) != penalty or any((conf...; only a true condition runs the indented block.
        if (
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            not summary["training_complete"]
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            or config.get("collision_penalty", -2.0) != penalty
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            or any(config[key] != value for key, value in expected.items())
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        ):
            # raise signals an error and stops this normal path, so invalid data are not silently accepted.
            raise ValueError(f"Existing training settings do not match: {directory}")
        # Call print using parentheses to supply its inputs; this line does not store a separate return value.
        print(f"Reusing training {directory.name}", flush=True)
    # else provides the alternative branch; after a loop it instead means the loop finished without break.
    else:
        # Call execute using parentheses to supply its inputs; this line does not store a separate return value.
        execute(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            [
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "train",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "--algorithm",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "dqn",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "--collision-penalty",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                penalty,
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "--seed",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                seed,
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "--steps",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                20000,
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "--out",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                directory,
            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
            ],
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            RESULTS / f"{directory.name}.log",
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
    # return ends this function and sends its result back to the code that called it.
    return checkpoint


# def defines ensure_evaluation; parentheses list its inputs, and the colon starts its indented body.
def ensure_evaluation(penalty, seed, checkpoint):
    # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
    """Evaluate a missing result, or verify that saved results belong to the correct model and settings."""
    # = stores a result in output; apply / to these operands; numbers are divided, while Path objects join folder names.
    output = STAGE / f"eval_dqn_penalty{abs(int(penalty))}_seed{seed}.json"
    # if checks output.exists(); only a true condition runs the indented block.
    if output.exists():
        # = stores a result in data; call json.loads using parentheses to supply its inputs.
        data = json.loads(output.read_text())
        # = stores a result in metadata; use square brackets to select part of data.
        metadata = data["provenance"]
        # if checks metadata['checkpoint_sha256'] != hashlib.sha256(checkpoint.read_bytes()).hexdigest() or metadata[...; only a true condition runs the indented block.
        if (
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            metadata["checkpoint_sha256"]
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            != hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            or metadata["training_collision_penalty"] != penalty
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            or metadata["evaluation_collision_penalty"] != -2.0
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            or metadata["seed_start"] != 10000
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            or metadata["episodes_per_density"] != 30
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            or metadata["alpha"] != 1.0
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            or metadata.get("speed_governor_cap_m_s") is not None
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            or [row["density"] for row in data["summary"]] != [1.0, 1.5, 2.0]
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        ):
            # raise signals an error and stops this normal path, so invalid data are not silently accepted.
            raise ValueError(f"Existing evaluation settings do not match: {output}")
    # else provides the alternative branch; after a loop it instead means the loop finished without break.
    else:
        # = stores a result in gif; choose the value before "if" when its condition is true, otherwise the value after "else".
        gif = (
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            ["--gif", RESULTS / f"dqn_penalty{abs(int(penalty))}_seed{seed}.gif"]
            # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
            if seed == 0
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            else []
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
        # Call execute using parentheses to supply its inputs; this line does not store a separate return value.
        execute(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            [
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "evaluate",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "--checkpoint",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                checkpoint,
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "--episodes",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                30,
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "--out",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                output,
                # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
                *gif,
            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
            ],
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            STAGE / f"evaluation_penalty{abs(int(penalty))}_seed{seed}.log",
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
    # return ends this function and sends its result back to the code that called it.
    return output


# def defines candidate; parentheses list its inputs, and the colon starts its indented body.
def candidate(penalty):
    # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
    """Read one speed-control candidate and record its measured speed and file fingerprint."""
    # = stores a result in checkpoint; call ensure_training using parentheses to supply its inputs.
    checkpoint = ensure_training(penalty, 0)
    # = stores a result in path; choose the value before "if" when its condition is true, otherwise the value after "else".
    path = (
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        RESULTS / "eval_dqn_seed0_mean.json"
        # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
        if penalty == -2
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        else ensure_evaluation(penalty, 0, checkpoint)
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # = stores a result in data; call json.loads using parentheses to supply its inputs.
    data = json.loads(path.read_text())
    # = stores a result in normal; call next using parentheses to supply its inputs.
    normal = next(row for row in data["summary"] if row["density"] == 1.0)
    # return ends this function and sends its result back to the code that called it.
    return {
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "collision_penalty": penalty,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "speed_m_s": normal["mean_speed_m_s"],
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "evaluation_path": str(path.relative_to(ROOT)),
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "evaluation_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
    }


# def defines main; parentheses list its inputs, and the colon starts its indented body.
def main():
    """Read Terminal options, check settings, and run the requested workflow.
    Existing result files are checked before reuse so a rerun cannot silently mix
    different experiments. The main guard below prevents running jobs on import.
    """
    # = stores a result in parser; call argparse.ArgumentParser using parentheses to supply its inputs.
    parser = argparse.ArgumentParser(description=__doc__)
    # Call parser.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    parser.add_argument("--jobs", type=int, default=2)
    # = stores a result in args; call parser.parse_args using parentheses to supply its inputs.
    args = parser.parse_args()
    # if checks Path.cwd() != ROOT or not 1 <= args.jobs <= 4; only a true condition runs the indented block.
    if Path.cwd() != ROOT or not 1 <= args.jobs <= 4:
        # Call parser.error using parentheses to supply its inputs; this line does not store a separate return value.
        parser.error("Run from the project directory; --jobs must be between 1 and 4.")
    # Call stage.mkdir using parentheses to supply its inputs; this line does not store a separate return value.
    STAGE.mkdir(parents=True, exist_ok=True)
    # with enters a temporary context and exits it afterward, even if an error interrupts the block.
    with (STAGE / "core_suite.log").open("w") as file:
        # Start another Python command and check whether it succeeded.
        # Call subprocess.run using parentheses to supply its inputs; this line does not store a separate return value.
        subprocess.run(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            [
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                sys.executable,
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "run_experiments.py",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "--seeds",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "0",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "1",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "2",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "--steps",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "20000",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "--episodes",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "30",
            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
            ],
            # Inside this continued call, name=value supplies an option by name rather than by its position.
            stdout=file,
            # Inside this continued call, name=value supplies an option by name rather than by its position.
            stderr=subprocess.STDOUT,
            # Inside this continued call, name=value supplies an option by name rather than by its position.
            check=True,
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
    # Run independent jobs together. Each writes its own files; this is not a shared training model.
    # with enters a temporary context and exits it afterward, even if an error interrupts the block.
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        # = stores a result in candidates; call list using parentheses to supply its inputs.
        candidates = list(pool.map(candidate, [-2, -4, -8]))
    # = stores a result in target_path; apply / to these operands; numbers are divided, while Path objects join folder names.
    target_path = RESULTS / "eval_qrdqn_seed0_tail25.json"
    # = stores a result in target; use square brackets to select part of json.loads(target_path.read_text())['summary'][0].
    target = json.loads(target_path.read_text())["summary"][0]["mean_speed_m_s"]
    # = stores a result in selected; call choose_speed_targeted_baseline using parentheses to supply its inputs.
    selected = choose_speed_targeted_baseline(candidates, target)
    # = stores a result in selection; build a dictionary, which stores values under named keys.
    selection = {
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "rule": "Closest normal-density seed-0 speed to QR-DQN lower25; ties prefer smaller penalty magnitude. Collision rate is not a selection input.",
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "target_speed_m_s": target,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "candidates": candidates,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "target_evaluation_sha256": hashlib.sha256(
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            target_path.read_bytes()
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        ).hexdigest(),
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "selected_collision_penalty": selected["collision_penalty"],
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "seed0_speed_difference_m_s": selected["speed_m_s"] - target,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "confirmation_scenario_seeds_used": False,
    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
    }
    # = stores a result in selection_path; apply / to these operands; numbers are divided, while Path objects join folder names.
    selection_path = STAGE / "baseline_selection.json"
    # if checks selection_path.exists() and json.loads(selection_path.read_text()) != selection; only a true condition runs the indented block.
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        # raise signals an error and stops this normal path, so invalid data are not silently accepted.
        raise ValueError("Refusing to change the recorded baseline selection")
    # Call save_json using parentheses to supply its inputs; this line does not store a separate return value.
    save_json(selection_path, selection)
    # = stores a result in penalty; use square brackets to select part of selected.
    penalty = selected["collision_penalty"]
    # Call print using parentheses to supply its inputs; this line does not store a separate return value.
    print(f"Selected collision penalty {penalty} by speed closeness only", flush=True)

    # def defines replicate; parentheses list its inputs, and the colon starts its indented body.
    def replicate(seed):
        # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
        """Repeat the selected setup for another training seed, keeping its settings fixed."""
        # = stores a result in checkpoint; call ensure_training using parentheses to supply its inputs.
        checkpoint = ensure_training(penalty, seed)
        # if checks penalty != -2; only a true condition runs the indented block.
        if penalty != -2:
            # Call ensure_evaluation using parentheses to supply its inputs; this line does not store a separate return value.
            ensure_evaluation(penalty, seed, checkpoint)

    # Run independent jobs together. Each writes its own files; this is not a shared training model.
    # with enters a temporary context and exits it afterward, even if an error interrupts the block.
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        # Call list using parentheses to supply its inputs; this line does not store a separate return value.
        list(pool.map(replicate, [1, 2]))
    # for takes each item from [('dqn', 'dqn', 1.0), ('qrdqn_mean', 'qrdqn', 1.0), ('qrdqn_tail25', 'qrdqn', 0.25)] in turn; (name, algorithm, alpha) holds the current item.
    for name, algorithm, alpha in [
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        ("dqn", "dqn", 1.0),
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        ("qrdqn_mean", "qrdqn", 1.0),
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        ("qrdqn_tail25", "qrdqn", 0.25),
    # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
    ]:
        # = stores a result in path; apply / to these operands; numbers are divided, while Path objects join folder names.
        path = STAGE / f"diagnostic_{name}.json"
        # if checks not path.exists(); only a true condition runs the indented block.
        if not path.exists():
            # with enters a temporary context and exits it afterward, even if an error interrupts the block.
            with (STAGE / f"diagnostic_{name}.log").open("w") as file:
                # Start another Python command and check whether it succeeded.
                # Call subprocess.run using parentheses to supply its inputs; this line does not store a separate return value.
                subprocess.run(
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    [
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        sys.executable,
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        "-m",
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        "driving_rl.diagnostics",
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        "--checkpoint",
                        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                        str(RESULTS / f"{algorithm}_seed0/final.pt"),
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        "--alpha",
                        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                        str(alpha),
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        "--out",
                        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                        str(path),
                    # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
                    ],
                    # Inside this continued call, name=value supplies an option by name rather than by its position.
                    stdout=file,
                    # Inside this continued call, name=value supplies an option by name rather than by its position.
                    stderr=subprocess.STDOUT,
                    # Inside this continued call, name=value supplies an option by name rather than by its position.
                    check=True,
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                )
    # Complete the exploratory control and regenerate the full current report.
    # Start another Python command and check whether it succeeded.
    # Call subprocess.run using parentheses to supply its inputs; this line does not store a separate return value.
    subprocess.run([sys.executable, "run_speed_control.py"], check=True)
    # Call print using parentheses to supply its inputs; this line does not store a separate return value.
    print("control experiments complete: results/report/index.html", flush=True)


# Run the entry point only when this file is executed, not when it is imported.
# __name__ equals "__main__" when this file runs directly; this guard avoids running the script merely because another file imports it.
if __name__ == "__main__":
    # Call main using parentheses to supply its inputs; this line does not store a separate return value.
    main()
