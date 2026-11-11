"""Run the speed control experiment from Terminal.

This is an experiment coordinator, not the learning algorithm. It launches
separate Python commands, saves logs, and checks existing results before reuse.
Paths identify saved files; hashes are fingerprints that detect changed files.
Read main() for the order of operations. Learning itself lives in driving_rl/agents.py.

Original implementation notes:
Explore a predeclared ego-speed control grid on existing scalar checkpoints.
"""

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
from driving_rl.analysis import analyze
# from driving_rl.experiment import brings the named tools into this file; parentheses allow several lines.
from driving_rl.experiment import save_json
# from driving_rl.controls_report import brings the named tools into this file; parentheses allow several lines.
from driving_rl.controls_report import create_report

# = stores a result in ROOT; use a dot to access parent on Path(__file__).resolve().
ROOT = Path(__file__).resolve().parent
# = stores a result in RESULTS; apply / to these operands; numbers are divided, while Path objects join folder names.
RESULTS = ROOT / "results"
# = stores a result in STAGE; apply / to these operands; numbers are divided, while Path objects join folder names.
STAGE = RESULTS / "controls"


# def defines evaluate; parentheses list its inputs, and the colon starts its indented body.
def evaluate(cap, seed):
    """Run or verify one frozen-policy evaluation.
    Check its model fingerprint, settings and scenario block before reusing saved
    data. This function must not quietly mix results from different experiments.
    """
    # = stores a result in checkpoint; apply / to these operands; numbers are divided, while Path objects join folder names.
    checkpoint = RESULTS / f"dqn_seed{seed}/final.pt"
    # = stores a result in path; apply / to these operands; numbers are divided, while Path objects join folder names.
    path = STAGE / f"eval_dqn_cap{cap}_seed{seed}.json"
    # Check existing data instead of silently replacing scientific results.
    # if checks path.exists(); only a true condition runs the indented block.
    if path.exists():
        # = stores a result in data; call json.loads using parentheses to supply its inputs.
        data = json.loads(path.read_text())
        # = stores a result in metadata; use square brackets to select part of data.
        metadata = data["provenance"]
        # if checks metadata['speed_governor_cap_m_s'] != cap or metadata['checkpoint_sha256'] != hashlib.sha256(chec...; only a true condition runs the indented block.
        if (
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            metadata["speed_governor_cap_m_s"] != cap
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            or metadata["checkpoint_sha256"]
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            != hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            or metadata["episodes_per_density"] != 30
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            or metadata["seed_start"] != 10000
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            or metadata["alpha"] != 1
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            or metadata["evaluation_collision_penalty"] != -2
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            or [row["density"] for row in data["summary"]] != [1.0, 1.5, 2.0]
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        ):
            # raise signals an error and stops this normal path, so invalid data are not silently accepted.
            raise ValueError(f"Refusing to reuse mismatched evaluation {path}")
    # else provides the alternative branch; after a loop it instead means the loop finished without break.
    else:
        # = stores a result in gif; choose the value before "if" when its condition is true, otherwise the value after "else".
        gif = (
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            ["--gif", str(RESULTS / f"dqn_cap{cap}_seed{seed}.gif")]
            # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
            if seed == 0
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            else []
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
        # with enters a temporary context and exits it afterward, even if an error interrupts the block.
        with (STAGE / f"cap{cap}_seed{seed}.log").open("w") as file:
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
                    "30",
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    "--out",
                    # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                    str(path),
                    # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
                    *gif,
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
        # = stores a result in data; call json.loads using parentheses to supply its inputs.
        data = json.loads(path.read_text())
    # return ends this function and sends its result back to the code that called it.
    return path, data


# def defines main; parentheses list its inputs, and the colon starts its indented body.
def main():
    """Read Terminal options, check settings, and run the requested workflow.
    Existing result files are checked before reuse so a rerun cannot silently mix
    different experiments. The main guard below prevents running jobs on import.
    """
    # if checks Path.cwd() != ROOT; only a true condition runs the indented block.
    if Path.cwd() != ROOT:
        # raise signals an error and stops this normal path, so invalid data are not silently accepted.
        raise ValueError("Run from the project directory")
    # = stores a result in candidates; build a list, keeping its items in order.
    candidates = []
    # for takes each item from [22, 24, 26] in turn; cap holds the current item.
    for cap in [22, 24, 26]:
        # = unpacks the right-hand result into (path, data), matching values to names in order.
        path, data = evaluate(cap, 0)
        # Call candidates.append using parentheses to supply its inputs; this line does not store a separate return value.
        candidates.append(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            {
                # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                "cap_m_s": cap,
                # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                "speed_m_s": data["summary"][0]["mean_speed_m_s"],
                # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                "evaluation_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
            }
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
    # = stores a result in target; use square brackets to select part of json.loads((RESULTS / 'eval_qrdqn_seed0_tail25.json').read_text())['summary'][0].
    target = json.loads((RESULTS / "eval_qrdqn_seed0_tail25.json").read_text())[
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "summary"
    # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
    ][0]["mean_speed_m_s"]
    # = stores a result in selected; call min using parentheses to supply its inputs.
    selected = min(
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        candidates, key=lambda row: (abs(row["speed_m_s"] - target), row["cap_m_s"])
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # = stores a result in choice; build a dictionary, which stores values under named keys.
    choice = {
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "rule": "Closest normal-density seed-0 speed; ties prefer lower cap. Exploratory development control.",
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "target_speed_m_s": target,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "candidates": candidates,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "selected_cap_m_s": selected["cap_m_s"],
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "seed0_speed_difference_m_s": selected["speed_m_s"] - target,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "confirmation_scenario_seeds_used": False,
    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
    }
    # = stores a result in path; apply / to these operands; numbers are divided, while Path objects join folder names.
    path = STAGE / "governor_selection.json"
    # if checks path.exists() and json.loads(path.read_text()) != choice; only a true condition runs the indented block.
    if path.exists() and json.loads(path.read_text()) != choice:
        # raise signals an error and stops this normal path, so invalid data are not silently accepted.
        raise ValueError("Refusing to change recorded governor selection")
    # Call save_json using parentheses to supply its inputs; this line does not store a separate return value.
    save_json(path, choice)
    # Call print using parentheses to supply its inputs; this line does not store a separate return value.
    print(json.dumps(choice, indent=2), flush=True)
    # for takes each item from [1, 2] in turn; seed holds the current item.
    for seed in [1, 2]:
        # Call evaluate using parentheses to supply its inputs; this line does not store a separate return value.
        evaluate(selected["cap_m_s"], seed)
    # Call analyze using parentheses to supply its inputs; this line does not store a separate return value.
    analyze(RESULTS)
    # Call create_report using parentheses to supply its inputs; this line does not store a separate return value.
    create_report(RESULTS)


# Run the entry point only when this file is executed, not when it is imported.
# __name__ equals "__main__" when this file runs directly; this guard avoids running the script merely because another file imports it.
if __name__ == "__main__":
    # Call main using parentheses to supply its inputs; this line does not store a separate return value.
    main()
