"""Compare completed 40K and 100K models without training or tuning anything.

Every model gets the same fresh scenarios. Earlier confirmation cases are kept
separate. Run this file again to check and reuse saved evaluations.
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
# from run_confirmation import brings the named tools into this file; parentheses allow several lines.
from run_confirmation import POLICIES

# = stores a result in ROOT; use a dot to access parent on Path(__file__).resolve().
ROOT = Path(__file__).resolve().parent
# = stores a result in OUTPUT; apply / to these operands; numbers are divided, while Path objects join folder names.
OUTPUT = ROOT / "results/long_training_evaluation"
# = stores a result in DENSITIES; build a list, keeping its items in order.
DENSITIES = [1.0, 1.5, 2.0, 2.5]


# def defines read; parentheses list its inputs, and the colon starts its indented body.
def read(path):
    # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
    """Read settings or results that have already been saved."""
    # return ends this function and sends its result back to the code that called it.
    return json.loads(path.read_text())


# def defines fingerprint; parentheses list its inputs, and the colon starts its indented body.
def fingerprint(path):
    # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
    """Identify the exact file, not just a model with a similar name."""
    # return ends this function and sends its result back to the code that called it.
    return hashlib.sha256(path.read_bytes()).hexdigest()


# def defines model_path; parentheses list its inputs, and the colon starts its indented body.
def model_path(budget, policy, seed):
    # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
    """The policy chooses a decision rule; several rules share one network."""
    # = stores a result in algorithm; use square brackets to select part of POLICIES[policy].
    algorithm = POLICIES[policy][0]
    # if checks budget == 100000; only a true condition runs the indented block.
    if budget == 100000:
        # = stores a result in folder; use quoted text as data; Python does not execute the words inside it.
        folder = "long_training_100k"
    # if checks algorithm == 'dqn'; only a true condition runs the indented block.
    elif algorithm == "dqn":
        # = stores a result in folder; use quoted text as data; Python does not execute the words inside it.
        folder = "baseline_controls"
    # else provides the alternative branch; after a loop it instead means the loop finished without break.
    else:
        # = stores a result in folder; use quoted text as data; Python does not execute the words inside it.
        folder = "density_coverage"
    # return ends this function and sends its result back to the code that called it.
    return ROOT / f"results/{folder}/{algorithm}_seed{seed}/final.pt"


# def defines lock_design; parentheses list its inputs, and the colon starts its indented body.
def lock_design():
    # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
    """Fix settings and model fingerprints before reading new outcomes."""
    # Call output.mkdir using parentheses to supply its inputs; this line does not store a separate return value.
    OUTPUT.mkdir(parents=True, exist_ok=True)
    # = stores a result in models; build a dictionary, which stores values under named keys.
    models = {}
    # Each budget gets its own results. Keep 40K and 100K separate until the paired comparison.
    # for takes each item from [40000, 100000] in turn; budget holds the current item.
    for budget in [40000, 100000]:
        # = stores a result in models[str(budget)]; build a dictionary, which stores values under named keys.
        models[str(budget)] = {}
        # A policy is a decision rule. Several rules can share the very same QR-DQN network.
        # for takes each item from POLICIES in turn; policy holds the current item.
        for policy in POLICIES:
            # = stores a result in models[str(budget)][policy]; build a dictionary, which stores values under named keys.
            models[str(budget)][policy] = {}
            # A training seed identifies one independently trained model, not a new traffic scenario.
            # for takes each item from range(3) in turn; seed holds the current item.
            for seed in range(3):
                # = stores a result in path; call model_path using parentheses to supply its inputs.
                path = model_path(budget, policy, seed)
                # = stores a result in summary; call read using parentheses to supply its inputs.
                summary = read(path.parent / "summary.json")
                # if checks summary['steps'] != budget or not summary['training_complete']; only a true condition runs the indented block.
                if summary["steps"] != budget or not summary["training_complete"]:
                    # raise signals an error and stops this normal path, so invalid data are not silently accepted.
                    raise ValueError("A requested training run is incomplete")
                # = stores a result in models[str(budget)][policy][str(seed)]; call fingerprint using parentheses to supply its inputs.
                models[str(budget)][policy][str(seed)] = fingerprint(path)
    # = stores a result in design; build a dictionary, which stores values under named keys.
    design = {
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "budgets": [40000, 100000],
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "model_hashes": models,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "scenario_seed_start": 20100,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "episodes_per_density": 100,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "densities": DENSITIES,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "policies": POLICIES,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "primary": "100K QR lower quarter minus 100K DQN with the frozen 25 m/s governor, density 1, collision fraction.",
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "primary_rule": "Conditional scenario-bootstrap 95% interval must have upper endpoint below zero; report speed regardless. No tuning.",
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "secondary": "Budget effects pair the SAME new scenarios. Other policies/densities and three-seed crossed intervals are exploratory.",
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "scope": "Primary inference conditions on selected networks. Equal aggregate speed is not guaranteed. No new-training or real-road safety claim.",
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "consumed_prior_confirmation": [20000, 20099],
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "future_untouched_seeds_begin": 20200,
    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
    }
    # JSON turns tuples into lists. Normalize that detail before checking a recorded design.
    # = stores a result in design; call json.loads using parentheses to supply its inputs.
    design = json.loads(json.dumps(design))
    # = stores a result in path; apply / to these operands; numbers are divided, while Path objects join folder names.
    path = OUTPUT / "design.json"
    # A saved filename is not enough: the checks below verify the contents belong to this experiment.
    # if checks path.exists(); only a true condition runs the indented block.
    if path.exists():
        # if checks read(path) != design; only a true condition runs the indented block.
        if read(path) != design:
            # raise signals an error and stops this normal path, so invalid data are not silently accepted.
            raise ValueError("The recorded study cannot be silently changed")
    # else provides the alternative branch; after a loop it instead means the loop finished without break.
    else:
        # for takes each item from (ROOT / 'results').rglob('*.json') in turn; old holds the current item.
        for old in (ROOT / "results").rglob("*.json"):
            # if checks OUTPUT in old.parents; only a true condition runs the indented block.
            if OUTPUT in old.parents:
                # continue skips the rest of this loop iteration and moves to its next item.
                continue
            # = stores a result in data; call read using parentheses to supply its inputs.
            data = read(old)
            # if checks isinstance(data, dict) and isinstance(data.get('episodes'), list); only a true condition runs the indented block.
            if isinstance(data, dict) and isinstance(data.get("episodes"), list):
                # for takes each item from data['episodes'] in turn; row holds the current item.
                for row in data["episodes"]:
                    # if checks 20100 <= row.get('seed', -1) < 20200; only a true condition runs the indented block.
                    if 20100 <= row.get("seed", -1) < 20200:
                        # raise signals an error and stops this normal path, so invalid data are not silently accepted.
                        raise ValueError("The requested scenario block was used before")
        # Call save_json using parentheses to supply its inputs; this line does not store a separate return value.
        save_json(path, design)
        # Call print using parentheses to supply its inputs; this line does not store a separate return value.
        print("Design locked; fresh scenario block verified unused.", flush=True)


# def defines evaluate; parentheses list its inputs, and the colon starts its indented body.
def evaluate(job):
    # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
    """Use the existing evaluator; this code never calls training."""
    # = unpacks the right-hand result into (budget, policy, seed), matching values to names in order.
    budget, policy, seed = job
    # = stores a result in path; call model_path using parentheses to supply its inputs.
    path = model_path(budget, policy, seed)
    # = unpacks the right-hand result into (algorithm, alpha, cap), matching values to names in order.
    algorithm, alpha, cap = POLICIES[policy]
    # = stores a result in output; apply / to these operands; numbers are divided, while Path objects join folder names.
    output = OUTPUT / f"eval_{budget}_{policy}_seed{seed}.json"
    # Do not overwrite an experiment. First check whether its saved settings match this request.
    # if checks output.exists(); only a true condition runs the indented block.
    if output.exists():
        # = stores a result in data; call read using parentheses to supply its inputs.
        data = read(output)
        # = stores a result in saved; use square brackets to select part of data.
        saved = data["provenance"]
        # = stores a result in actual; group these values into a tuple, keeping their order.
        actual = (
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            saved["checkpoint_sha256"],
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            saved["alpha"],
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            saved["speed_governor_cap_m_s"],
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            saved["training_steps"],
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            saved["seed_start"],
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            saved["episodes_per_density"],
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
        # = stores a result in expected; group these values into a tuple, keeping their order.
        expected = (fingerprint(path), alpha, cap, budget, 20100, 100)
        # if checks actual != expected or len(data['episodes']) != 400; only a true condition runs the indented block.
        if actual != expected or len(data["episodes"]) != 400:
            # raise signals an error and stops this normal path, so invalid data are not silently accepted.
            raise ValueError(f"Mismatched evaluation: {output}")
        # A set keeps each density/scenario pair once, even if a row was duplicated.
        # = stores a result in cases; call set using parentheses to supply its inputs.
        cases = set()
        # for takes each item from data['episodes'] in turn; episode holds the current item.
        for episode in data["episodes"]:
            # Call cases.add using parentheses to supply its inputs; this line does not store a separate return value.
            cases.add((episode["density"], episode["seed"]))
        # = stores a result in expected_cases; call set using parentheses to supply its inputs.
        expected_cases = set()
        # for takes each item from DENSITIES in turn; density holds the current item.
        for density in DENSITIES:
            # for takes each item from range(20100, 20200) in turn; scenario holds the current item.
            for scenario in range(20100, 20200):
                # Call expected_cases.add using parentheses to supply its inputs; this line does not store a separate return value.
                expected_cases.add((density, scenario))
        # We need every requested density/scenario pair, not just the right total number of rows.
        # if checks cases != expected_cases; only a true condition runs the indented block.
        if cases != expected_cases:
            # raise signals an error and stops this normal path, so invalid data are not silently accepted.
            raise ValueError("Missing or duplicated scenarios")
    # else provides the alternative branch; after a loop it instead means the loop finished without break.
    else:
        # = stores a result in command; build a list, keeping its items in order.
        command = [
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
            str(path),
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "--alpha",
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            str(alpha),
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "--seed-start",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "20100",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "--episodes",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "100",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "--densities",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "1",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "1.5",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "2",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "2.5",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "--out",
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            str(output),
        # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
        ]
        # The governor is an optional action override. Its threshold is not a guaranteed physical speed limit.
        # if checks cap is not None; only a true condition runs the indented block.
        if cap is not None:
            # Call command.extend using parentheses to supply its inputs; this line does not store a separate return value.
            command.extend(["--speed-cap", str(cap)])
        # with enters a temporary context and exits it afterward, even if an error interrupts the block.
        with output.with_suffix(".log").open("w") as log:
            # Run the requested Python command and stop on failure. This is evaluation, not a training call.
            # Call subprocess.run using parentheses to supply its inputs; this line does not store a separate return value.
            subprocess.run(
                # Inside this continued call, name=value supplies an option by name rather than by its position.
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            )
    # Call print using parentheses to supply its inputs; this line does not store a separate return value.
    print(f"Evaluated {budget} {policy} seed {seed}", flush=True)


# def defines main; parentheses list its inputs, and the colon starts its indented body.
def main():
    # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
    """Read the Terminal options and coordinate the workflow. The learning algorithm is not implemented here."""
    # = stores a result in parser; call argparse.ArgumentParser using parentheses to supply its inputs.
    parser = argparse.ArgumentParser(description=__doc__)
    # Call parser.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    parser.add_argument("--jobs", type=int, default=4)
    # Call parser.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    parser.add_argument("--report-only", action="store_true")
    # = stores a result in args; call parser.parse_args using parentheses to supply its inputs.
    args = parser.parse_args()
    # if checks args.jobs < 1; only a true condition runs the indented block.
    if args.jobs < 1:
        # Call parser.error using parentheses to supply its inputs; this line does not store a separate return value.
        parser.error("--jobs must be positive")
    # Call lock_design using parentheses to supply its inputs; this line does not store a separate return value.
    lock_design()
    # if checks not args.report_only; only a true condition runs the indented block.
    if not args.report_only:
        # = stores a result in jobs; build a list, keeping its items in order.
        jobs = []
        # Each budget gets its own results. Keep 40K and 100K separate until the paired comparison.
        # for takes each item from [100000, 40000] in turn; budget holds the current item.
        for budget in [100000, 40000]:
            # A policy is a decision rule. Several rules can share the very same QR-DQN network.
            # for takes each item from POLICIES in turn; policy holds the current item.
            for policy in POLICIES:
                # A training seed identifies one independently trained model, not a new traffic scenario.
                # for takes each item from range(3) in turn; seed holds the current item.
                for seed in range(3):
                    # Call jobs.append using parentheses to supply its inputs; this line does not store a separate return value.
                    jobs.append((budget, policy, seed))
        # This only schedules independent jobs. Each actual experiment runs in a separate Python process.
        # with enters a temporary context and exits it afterward, even if an error interrupts the block.
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            # Wait for all requested jobs, and let errors surface instead of quietly ignoring failed experiments.
            # Call list using parentheses to supply its inputs; this line does not store a separate return value.
            list(pool.map(evaluate, jobs))
    # from driving_rl.long_training_report import brings the named tools into this file; parentheses allow several lines.
    from driving_rl.long_training_report import report

    # Call report using parentheses to supply its inputs; this line does not store a separate return value.
    report()


# __name__ equals "__main__" when this file runs directly; this guard avoids running the script merely because another file imports it.
if __name__ == "__main__":
    # Call main using parentheses to supply its inputs; this line does not store a separate return value.
    main()
