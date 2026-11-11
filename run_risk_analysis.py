"""Run the risk analysis experiment from Terminal.

This is an experiment coordinator, not the learning algorithm. It launches
separate Python commands, saves logs, and checks existing results before reuse.
Paths identify saved files; hashes are fingerprints that detect changed files.
Read main() for the order of operations. Learning itself lives in driving_rl/agents.py.

Original implementation notes:
Reproduce same-checkpoint controls, risk sweep and calibration diagnostics.
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

# = stores a result in ROOT; use a dot to access parent on Path(__file__).resolve().
ROOT = Path(__file__).resolve().parent
# = stores a result in RESULTS; apply / to these operands; numbers are divided, while Path objects join folder names.
RESULTS = ROOT / "results"
# = stores a result in STAGE; apply / to these operands; numbers are divided, while Path objects join folder names.
STAGE = RESULTS / "risk_analysis"
# = stores a result in CAPS; build a list, keeping its items in order.
CAPS = [24.0, 24.5, 25.0, 25.5, 26.0, 26.5]
# = stores a result in ALPHAS; build a list, keeping its items in order.
ALPHAS = [0.125, 0.25, 0.5, 0.75, 1.0]


# def defines token; parentheses list its inputs, and the colon starts its indented body.
def token(value):
    # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
    """Make a short filename-friendly label for a numeric setting."""
    # return ends this function and sends its result back to the code that called it.
    return f"{value:g}".replace(".", "p")


# def defines eval_path; parentheses list its inputs, and the colon starts its indented body.
def eval_path(seed, alpha=1.0, cap=None):
    # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
    """Find the evaluation file corresponding to this seed and action-selection rule."""
    # if checks cap is None and alpha in [0.25, 1.0]; only a true condition runs the indented block.
    if cap is None and alpha in [0.25, 1.0]:
        # = stores a result in suffix; choose the value before "if" when its condition is true, otherwise the value after "else".
        suffix = "tail25" if alpha == 0.25 else "mean"
        # return ends this function and sends its result back to the code that called it.
        return RESULTS / f"eval_qrdqn_seed{seed}_{suffix}.json"
    # = stores a result in suffix; choose the value before "if" when its condition is true, otherwise the value after "else".
    suffix = f"cap{token(cap)}" if cap is not None else f"alpha{token(alpha)}"
    # return ends this function and sends its result back to the code that called it.
    return STAGE / f"eval_qrdqn_{suffix}_seed{seed}.json"


# def defines evaluate; parentheses list its inputs, and the colon starts its indented body.
def evaluate(spec):
    """Run or verify one frozen-policy evaluation.
    Check its model fingerprint, settings and scenario block before reusing saved
    data. This function must not quietly mix results from different experiments.
    """
    # = unpacks the right-hand result into (seed, alpha, cap), matching values to names in order.
    seed, alpha, cap = spec
    # = stores a result in checkpoint; apply / to these operands; numbers are divided, while Path objects join folder names.
    checkpoint = RESULTS / f"qrdqn_seed{seed}/final.pt"
    # = stores a result in output; call eval_path using parentheses to supply its inputs.
    output = eval_path(seed, alpha, cap)
    # if checks output.exists(); only a true condition runs the indented block.
    if output.exists():
        # = stores a result in data; call json.loads using parentheses to supply its inputs.
        data = json.loads(output.read_text())
        # = stores a result in p; use square brackets to select part of data.
        p = data["provenance"]
        # if checks p['checkpoint_sha256'] != hashlib.sha256(checkpoint.read_bytes()).hexdigest() or p['training_seed...; only a true condition runs the indented block.
        if (
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            p["checkpoint_sha256"]
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            != hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            or p["training_seed"] != seed
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            or p["training_steps"] != 20000
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            or p["alpha"] != alpha
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            or p.get("speed_governor_cap_m_s") != cap
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            or p["seed_start"] != 10000
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            or p["episodes_per_density"] != 30
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            or p["algorithm"] != "qrdqn"
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            or p.get("evaluation_collision_penalty", -2.0) != -2.0
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            or [row["density"] for row in data["summary"]] != [1.0, 1.5, 2.0]
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        ):
            # raise signals an error and stops this normal path, so invalid data are not silently accepted.
            raise ValueError(f"Refusing to reuse mismatched evaluation: {output}")
        # return ends this function and sends its result back to the code that called it.
        return output
    # = stores a result in arguments; build a list, keeping its items in order.
    arguments = [
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        sys.executable,
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "-m",
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "driving_rl.cli",
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "evaluate",
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "--checkpoint",
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        str(checkpoint),
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
        str(output),
    # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
    ]
    # if checks cap is not None; only a true condition runs the indented block.
    if cap is not None:
        # The combined operator updates arguments using its previous value.
        arguments += ["--speed-cap", str(cap)]
    # with enters a temporary context and exits it afterward, even if an error interrupts the block.
    with output.with_suffix(".log").open("w") as log:
        # Start another Python command and check whether it succeeded.
        # Call subprocess.run using parentheses to supply its inputs; this line does not store a separate return value.
        subprocess.run(arguments, stdout=log, stderr=subprocess.STDOUT, check=True)
    # Call print using parentheses to supply its inputs; this line does not store a separate return value.
    print(f"Evaluated {output.name}", flush=True)
    # return ends this function and sends its result back to the code that called it.
    return output


# def defines controls; parentheses list its inputs, and the colon starts its indented body.
def controls(jobs):
    # The threshold is chosen before looking at seeds 1 and 2 under this control.
    """Choose the speed control on the designated development model, freeze it,
    then evaluate other models without selecting their thresholds again.
    """
    # Run independent jobs together. Each writes its own files; this is not a shared training model.
    # with enters a temporary context and exits it afterward, even if an error interrupts the block.
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        # = stores a result in paths; call list using parentheses to supply its inputs.
        paths = list(pool.map(evaluate, [(0, 1.0, cap) for cap in CAPS]))
    # = stores a result in target_path; apply / to these operands; numbers are divided, while Path objects join folder names.
    target_path = RESULTS / "eval_qrdqn_seed0_tail25.json"
    # = stores a result in target; use square brackets to select part of json.loads(target_path.read_text())['summary'][0].
    target = json.loads(target_path.read_text())["summary"][0]["mean_speed_m_s"]
    # = stores a result in candidates; build a list, keeping its items in order.
    candidates = []
    # for takes each item from zip(CAPS, paths) in turn; (cap, path) holds the current item.
    for cap, path in zip(CAPS, paths):
        # = stores a result in speed; use square brackets to select part of json.loads(path.read_text())['summary'][0].
        speed = json.loads(path.read_text())["summary"][0]["mean_speed_m_s"]
        # Call candidates.append using parentheses to supply its inputs; this line does not store a separate return value.
        candidates.append(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            {
                # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                "cap_m_s": cap,
                # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                "speed_m_s": speed,
                # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                "evaluation_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
            }
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
    # = stores a result in chosen; call min using parentheses to supply its inputs.
    chosen = min(
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        candidates, key=lambda row: (abs(row["speed_m_s"] - target), row["cap_m_s"])
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # = stores a result in selection; build a dictionary, which stores values under named keys.
    selection = {
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "rule": "Same QR-DQN checkpoint, seed-0 normal-density speed closeness only; ties prefer lower cap.",
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "target_speed_m_s": target,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "candidates": candidates,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "selected_cap_m_s": chosen["cap_m_s"],
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "seed0_speed_difference_m_s": chosen["speed_m_s"] - target,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "target_evaluation_sha256": hashlib.sha256(
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            target_path.read_bytes()
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        ).hexdigest(),
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "confirmation_seeds_used": False,
    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
    }
    # = stores a result in path; apply / to these operands; numbers are divided, while Path objects join folder names.
    path = STAGE / "governor_selection.json"
    # if checks path.exists() and json.loads(path.read_text()) != selection; only a true condition runs the indented block.
    if path.exists() and json.loads(path.read_text()) != selection:
        # raise signals an error and stops this normal path, so invalid data are not silently accepted.
        raise ValueError(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "Refusing to change the recorded same-checkpoint control selection"
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
    # Call save_json using parentheses to supply its inputs; this line does not store a separate return value.
    save_json(path, selection)
    # Call print using parentheses to supply its inputs; this line does not store a separate return value.
    print(f"Frozen same-checkpoint threshold {chosen['cap_m_s']} m/s", flush=True)
    # = stores a result in tasks; collect or generate one result per item; the "for" part names the item being processed.
    tasks = [(seed, 1.0, chosen["cap_m_s"]) for seed in [1, 2]]
    # The combined operator updates tasks using its previous value.
    tasks += [(seed, alpha, None) for seed in [0, 1, 2] for alpha in ALPHAS]
    # Run independent jobs together. Each writes its own files; this is not a shared training model.
    # with enters a temporary context and exits it afterward, even if an error interrupts the block.
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        # Call list using parentheses to supply its inputs; this line does not store a separate return value.
        list(pool.map(evaluate, tasks))


# def defines calibration_task; parentheses list its inputs, and the colon starts its indented body.
def calibration_task(spec):
    # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
    """Collect or validate the return-prediction checks for one model and policy."""
    # = unpacks the right-hand result into (seed, policy, cap), matching values to names in order.
    seed, policy, cap = spec
    # = stores a result in alpha; choose the value before "if" when its condition is true, otherwise the value after "else".
    alpha = 0.25 if policy == "tail25" else 1.0
    # = stores a result in checkpoint; apply / to these operands; numbers are divided, while Path objects join folder names.
    checkpoint = RESULTS / f"qrdqn_seed{seed}/final.pt"
    # = stores a result in path; apply / to these operands; numbers are divided, while Path objects join folder names.
    path = STAGE / f"calibration_{policy}_seed{seed}.json"
    # Check existing data instead of silently replacing scientific results.
    # if checks path.exists(); only a true condition runs the indented block.
    if path.exists():
        # = stores a result in data; use square brackets to select part of json.loads(path.read_text()).
        data = json.loads(path.read_text())["provenance"]
        # if checks data['checkpoint_sha256'] != hashlib.sha256(checkpoint.read_bytes()).hexdigest() or data['alpha']...; only a true condition runs the indented block.
        if (
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            data["checkpoint_sha256"]
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            != hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            or data["alpha"] != alpha
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            or data["speed_cap"] != cap
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            or data["seed_start"] != 12000
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            or data["episodes_per_density"] != 60
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        ):
            # raise signals an error and stops this normal path, so invalid data are not silently accepted.
            raise ValueError(f"Refusing to reuse mismatched calibration: {path}")
        # return ends this function and sends its result back to the code that called it.
        return
    # = stores a result in arguments; build a list, keeping its items in order.
    arguments = [
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        sys.executable,
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "-m",
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "driving_rl.calibration",
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "--checkpoint",
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        str(checkpoint),
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "--alpha",
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        str(alpha),
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "--out",
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        str(path),
    # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
    ]
    # if checks cap is not None; only a true condition runs the indented block.
    if cap is not None:
        # The combined operator updates arguments using its previous value.
        arguments += ["--speed-cap", str(cap)]
    # with enters a temporary context and exits it afterward, even if an error interrupts the block.
    with path.with_suffix(".log").open("w") as log:
        # Start another Python command and check whether it succeeded.
        # Call subprocess.run using parentheses to supply its inputs; this line does not store a separate return value.
        subprocess.run(arguments, stdout=log, stderr=subprocess.STDOUT, check=True)
    # Call print using parentheses to supply its inputs; this line does not store a separate return value.
    print(f"Calibrated {policy}, training seed {seed}", flush=True)


# def defines ensure_replay; parentheses list its inputs, and the colon starts its indented body.
def ensure_replay(cap):
    # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
    """Create the predefined recorded driving example if it is not already saved."""
    # = stores a result in checkpoint; apply / to these operands; numbers are divided, while Path objects join folder names.
    checkpoint = RESULTS / "qrdqn_seed0/final.pt"
    # = stores a result in path; apply / to these operands; numbers are divided, while Path objects join folder names.
    path = STAGE / "replay_governed.json"
    # Check existing data instead of silently replacing scientific results.
    # if checks path.exists(); only a true condition runs the indented block.
    if path.exists():
        # = stores a result in p; use square brackets to select part of json.loads(path.read_text()).
        p = json.loads(path.read_text())["provenance"]
        # if checks p['checkpoint_sha256'] != hashlib.sha256(checkpoint.read_bytes()).hexdigest() or p['alpha'] != 1....; only a true condition runs the indented block.
        if (
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            p["checkpoint_sha256"]
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            != hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            or p["alpha"] != 1.0
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            or p["speed_governor_cap_m_s"] != cap
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            or p["seed_start"] != 10000
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            or p["episodes_per_density"] != 1
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        ):
            # raise signals an error and stops this normal path, so invalid data are not silently accepted.
            raise ValueError("Existing replay does not match the selected control")
        # if checks not (STAGE / 'replay_governed.gif').exists(); only a true condition runs the indented block.
        if not (STAGE / "replay_governed.gif").exists():
            # raise signals an error and stops this normal path, so invalid data are not silently accepted.
            raise ValueError(
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "Replay JSON exists without its GIF; preserve it and regenerate in a new path"
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            )
        # return ends this function and sends its result back to the code that called it.
        return
    # with enters a temporary context and exits it afterward, even if an error interrupts the block.
    with (STAGE / "replay_governed.log").open("w") as log:
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
                "driving_rl.cli",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "evaluate",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "--checkpoint",
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                str(checkpoint),
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "--speed-cap",
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                str(cap),
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "--episodes",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "1",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "--densities",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "1",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "--out",
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                str(path),
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "--gif",
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                str(STAGE / "replay_governed.gif"),
            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
            ],
            # Inside this continued call, name=value supplies an option by name rather than by its position.
            stdout=log,
            # Inside this continued call, name=value supplies an option by name rather than by its position.
            stderr=subprocess.STDOUT,
            # Inside this continued call, name=value supplies an option by name rather than by its position.
            check=True,
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )


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
    # Call parser.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    parser.add_argument(
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        "--stage", choices=["controls", "calibration", "all"], default="all"
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # = stores a result in args; call parser.parse_args using parentheses to supply its inputs.
    args = parser.parse_args()
    # if checks Path.cwd() != ROOT or not 1 <= args.jobs <= 4; only a true condition runs the indented block.
    if Path.cwd() != ROOT or not 1 <= args.jobs <= 4:
        # Call parser.error using parentheses to supply its inputs; this line does not store a separate return value.
        parser.error("Run from the project directory with --jobs between 1 and 4")
    # Call stage.mkdir using parentheses to supply its inputs; this line does not store a separate return value.
    STAGE.mkdir(parents=True, exist_ok=True)
    # if checks args.stage in ['controls', 'all']; only a true condition runs the indented block.
    if args.stage in ["controls", "all"]:
        # Call controls using parentheses to supply its inputs; this line does not store a separate return value.
        controls(args.jobs)
    # if checks args.stage in ['calibration', 'all']; only a true condition runs the indented block.
    if args.stage in ["calibration", "all"]:
        # = stores a result in cap; use square brackets to select part of json.loads((STAGE / 'governor_selection.json').read_text()).
        cap = json.loads((STAGE / "governor_selection.json").read_text())[
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "selected_cap_m_s"
        # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
        ]
        # = stores a result in tasks; collect or generate one result per item; the "for" part names the item being processed.
        tasks = [
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            (seed, policy, cap if policy == "governed" else None)
            # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
            for seed in [0, 1, 2]
            # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
            for policy in ["mean", "tail25", "governed"]
        # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
        ]
        # Run independent jobs together. Each writes its own files; this is not a shared training model.
        # with enters a temporary context and exits it afterward, even if an error interrupts the block.
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            # Call list using parentheses to supply its inputs; this line does not store a separate return value.
            list(pool.map(calibration_task, tasks))
        # Call ensure_replay using parentheses to supply its inputs; this line does not store a separate return value.
        ensure_replay(cap)
        # from driving_rl.risk_analysis_analysis import brings the named tools into this file; parentheses allow several lines.
        from driving_rl.risk_analysis_analysis import analyze
        # from driving_rl.risk_analysis_report import brings the named tools into this file; parentheses allow several lines.
        from driving_rl.risk_analysis_report import create_report

        # Call analyze using parentheses to supply its inputs; this line does not store a separate return value.
        analyze(RESULTS)
        # Call create_report using parentheses to supply its inputs; this line does not store a separate return value.
        create_report(RESULTS)


# Run the entry point only when this file is executed, not when it is imported.
# __name__ equals "__main__" when this file runs directly; this guard avoids running the script merely because another file imports it.
if __name__ == "__main__":
    # Call main using parentheses to supply its inputs; this line does not store a separate return value.
    main()
