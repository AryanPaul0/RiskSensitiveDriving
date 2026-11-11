"""Run the confirmation experiment from Terminal.

This is an experiment coordinator, not the learning algorithm. It launches
separate Python commands, saves logs, and checks existing results before reuse.
Paths identify saved files; hashes are fingerprints that detect changed files.
Read main() for the order of operations. Learning itself lives in driving_rl/agents.py.

Original implementation notes:
Evaluate locked policies once on the reserved scenario block.
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
# from driving_rl.source_integrity import brings the named tools into this file; parentheses allow several lines.
from driving_rl.source_integrity import source_matches_frozen_version

# = stores a result in ROOT; use a dot to access parent on Path(__file__).resolve().
ROOT = Path(__file__).resolve().parent
# = stores a result in STAGE; apply / to these operands; numbers are divided, while Path objects join folder names.
STAGE = ROOT / "results/confirmation"
# = stores a result in POLICIES; build a dictionary, which stores values under named keys.
POLICIES = {
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "qr_tail": ("qrdqn", 0.25, None),
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "dqn_governed": ("dqn", 1.0, 25.0),
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "qr_governed": ("qrdqn", 1.0, 25.0),
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "dqn_mean": ("dqn", 1.0, None),
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "qr_mean": ("qrdqn", 1.0, None),
# } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
}


# def defines read; parentheses list its inputs, and the colon starts its indented body.
def read(p):
    # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
    """Read saved JSON data into ordinary Python dictionaries and lists."""
    # return ends this function and sends its result back to the code that called it.
    return json.loads(p.read_text())


# def defines sha; parentheses list its inputs, and the colon starts its indented body.
def sha(p):
    """Calculate a file fingerprint. If even one byte changes, its SHA-256 fingerprint
    will almost certainly change too. This helps detect changed frozen artifacts.
    """
    # return ends this function and sends its result back to the code that called it.
    return hashlib.sha256(p.read_bytes()).hexdigest()


# def defines checkpoint; parentheses list its inputs, and the colon starts its indented body.
def checkpoint(policy, seed):
    # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
    """Find the saved model file for the requested algorithm and training seed."""
    # = stores a result in algo; use square brackets to select part of POLICIES[policy].
    algo = POLICIES[policy][0]
    # return ends this function and sends its result back to the code that called it.
    return (
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        ROOT
        # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
        / f'results/{"density_coverage" if algo=="qrdqn" else "baseline_controls"}/{algo}_seed{seed}/final.pt'
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )


# def defines design; parentheses list its inputs, and the colon starts its indented body.
def design():
    """Describe the fixed confirmation plan, including model fingerprints, scenario
    seeds, the primary comparison and the limits of the planned uncertainty interval.
    """
    # return ends this function and sends its result back to the code that called it.
    return {
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "status": "Locked before confirmation evaluations",
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "training_steps": 40000,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "training_seeds": [0, 1, 2],
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "training_densities": [1.0, 1.5, 2.0],
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "scenario_seed_start": 20000,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "episodes_per_density": 100,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "densities": [1.0, 1.5, 2.0, 2.5],
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "policies": {
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            p: {"algorithm": a, "alpha": alpha, "speed_cap_m_s": cap}
            # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
            for p, (a, alpha, cap) in POLICIES.items()
        # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
        },
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "primary": {
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "left": "qr_tail",
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "right": "dqn_governed",
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "density": 1.0,
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "metric": "crashed",
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "direction": "Lower collision fraction is better",
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "estimand": "Expected paired collision difference over fresh scenarios, averaged over the three fixed trained-model pairs.",
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "interval": "Two-sided percentile 95% scenario-cluster bootstrap; 10000 replicates; RNG seed 20261111. Resample scenario IDs while retaining all fixed models and policy pairing.",
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "decision_rule": "Evidence of fewer collisions for the locked QR-tail policy if the upper conditional confidence endpoint is strictly below zero. Report speed/distance regardless of collision result. Do not claim speed-independent or real-road safety benefit.",
        # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
        },
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "secondary": "Other policies, traffic densities, seed variation and crossed bootstrap intervals are descriptive/exploratory; no family-wise inference claimed.",
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "generalization_limit": "Primary interval conditions on this selected set of trained models, architecture, rewards and governor caps; it does not cover new training runs or remove speed/action-rule differences.",
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "no_tuning": "No retraining, checkpoint selection, alpha adjustment or governor adjustment using these results. Further tuned models need a new untouched scenario block.",
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "replays": "Training seed 0, first reserved scenario 20000 at density 1, for QR tail and both governed controls; fixed before outcomes.",
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "checkpoint_hashes": {
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            p: {
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                str(s): {
                    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                    "path": str(checkpoint(p, s).relative_to(ROOT)),
                    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                    "sha256": sha(checkpoint(p, s)),
                # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
                }
                # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                for s in range(3)
            # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
            }
            # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
            for p in POLICIES
        # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
        },
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "source_hashes": {
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            str(p.relative_to(ROOT)): sha(p)
            # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
            for p in sorted((ROOT / "driving_rl").glob("*.py"))
            # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
            if p.name != "confirmation_report.py"
        # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
        },
    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
    }


# def defines lock; parentheses list its inputs, and the colon starts its indented body.
def lock():
    """Record the confirmation settings before looking at confirmation outcomes.
    On later runs, check the model files and settings rather than choosing again.
    """
    # Call stage.mkdir using parentheses to supply its inputs; this line does not store a separate return value.
    STAGE.mkdir(parents=True, exist_ok=True)
    # = stores a result in path; apply / to these operands; numbers are divided, while Path objects join folder names.
    path = STAGE / "design.json"
    # = stores a result in value; call design using parentheses to supply its inputs.
    value = design()
    # Check existing data instead of silently replacing scientific results.
    # if checks path.exists(); only a true condition runs the indented block.
    if path.exists():
        # = stores a result in old; call read using parentheses to supply its inputs.
        old = read(path)
        # Source additions do not alter the already locked design. Check all fixed parameters and hashes.
        # if checks {k: v for k, v in old.items() if k != 'source_hashes'} != {k: v for k, v in value.items() if k !=...; only a true condition runs the indented block.
        if {k: v for k, v in old.items() if k != "source_hashes"} != {
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            k: v for k, v in value.items() if k != "source_hashes"
        # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
        }:
            # raise signals an error and stops this normal path, so invalid data are not silently accepted.
            raise ValueError("Refusing to change the locked confirmation design")
        # for takes each item from ['agents.py', 'envs.py', 'experiment.py', 'cli.py'] in turn; name holds the current item.
        for name in ["agents.py", "envs.py", "experiment.py", "cli.py"]:
            # = stores a result in key; combine the operands with the displayed operator; parentheses control grouping.
            key = "driving_rl/" + name
            # if checks not source_matches_frozen_version(ROOT, key, old['source_hashes'][key]); only a true condition runs the indented block.
            if not source_matches_frozen_version(ROOT, key, old["source_hashes"][key]):
                # raise signals an error and stops this normal path, so invalid data are not silently accepted.
                raise ValueError(
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    "Evaluation code changed after confirmation lock: " + name
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                )
        # return ends this function and sends its result back to the code that called it.
        return old
    # Verify that this scenario block has not appeared in any previous saved episodes.
    # = stores a result in audited; use the literal value 0.
    audited = 0
    # for takes each item from (ROOT / 'results').rglob('*.json') in turn; p holds the current item.
    for p in (ROOT / "results").rglob("*.json"):
        # if checks STAGE in p.parents; only a true condition runs the indented block.
        if STAGE in p.parents:
            # continue skips the rest of this loop iteration and moves to its next item.
            continue
        # = stores a result in data; call read using parentheses to supply its inputs.
        data = read(p)
        # if checks isinstance(data, dict) and isinstance(data.get('episodes'), list); only a true condition runs the indented block.
        if isinstance(data, dict) and isinstance(data.get("episodes"), list):
            # if checks any((20000 <= r.get('seed', -1) < 20100 for r in data['episodes'])); only a true condition runs the indented block.
            if any(20000 <= r.get("seed", -1) < 20100 for r in data["episodes"]):
                # raise signals an error and stops this normal path, so invalid data are not silently accepted.
                raise ValueError(f"Confirmation block already used: {p}")
            # The combined operator updates audited using its previous value.
            audited += 1
    # Call save_json using parentheses to supply its inputs; this line does not store a separate return value.
    save_json(
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        STAGE / "preflight.json",
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        {
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "existing_episode_files_checked": audited,
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "reserved_block_previously_unused": True,
        # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
        },
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # Call save_json using parentheses to supply its inputs; this line does not store a separate return value.
    save_json(path, value)
    # Call print using parentheses to supply its inputs; this line does not store a separate return value.
    print(
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "Locked design before evaluation; reserved scenario block was unused.",
        # Inside this continued call, name=value supplies an option by name rather than by its position.
        flush=True,
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # return ends this function and sends its result back to the code that called it.
    return value


# def defines evaluate; parentheses list its inputs, and the colon starts its indented body.
def evaluate(spec):
    """Run or verify one frozen-policy evaluation.
    Check its model fingerprint, settings and scenario block before reusing saved
    data. This function must not quietly mix results from different experiments.
    """
    # = unpacks the right-hand result into (policy, seed), matching values to names in order.
    policy, seed = spec
    # = unpacks the right-hand result into (algo, alpha, cap), matching values to names in order.
    algo, alpha, cap = POLICIES[policy]
    # = stores a result in ck; call checkpoint using parentheses to supply its inputs.
    ck = checkpoint(policy, seed)
    # = stores a result in out; apply / to these operands; numbers are divided, while Path objects join folder names.
    out = STAGE / f"eval_{policy}_seed{seed}.json"
    # A saved file is useful only if its model and settings match this request.
    # if checks out.exists(); only a true condition runs the indented block.
    if out.exists():
        # = stores a result in d; call read using parentheses to supply its inputs.
        d = read(out)
        # = stores a result in p; use square brackets to select part of d.
        p = d["provenance"]
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
            p["training_steps"],
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            p["training_seed"],
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            [r["density"] for r in d["summary"]],
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        ) != (sha(ck), alpha, cap, 20000, 100, 40000, seed, [1.0, 1.5, 2.0, 2.5]):
            # raise signals an error and stops this normal path, so invalid data are not silently accepted.
            raise ValueError(f"Mismatched confirmation {out}")
        # = stores a result in expected; collect or generate one result per item; the "for" part names the item being processed.
        expected = {
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            (density, scenario)
            # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
            for density in [1.0, 1.5, 2.0, 2.5]
            # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
            for scenario in range(20000, 20100)
        # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
        }
        # if checks {(r['density'], r['seed']) for r in d['episodes']} != expected or len(d['episodes']) != 400; only a true condition runs the indented block.
        if {(r["density"], r["seed"]) for r in d["episodes"]} != expected or len(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            d["episodes"]
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        ) != 400:
            # raise signals an error and stops this normal path, so invalid data are not silently accepted.
            raise ValueError("Incomplete/duplicate confirmation cases")
    # else provides the alternative branch; after a loop it instead means the loop finished without break.
    else:
        # = stores a result in cmd; build a list, keeping its items in order.
        cmd = [
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
            str(ck),
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "--alpha",
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            str(alpha),
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "--episodes",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "100",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "--seed-start",
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "20000",
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
            str(out),
        # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
        ]
        # if checks cap is not None; only a true condition runs the indented block.
        if cap is not None:
            # The combined operator updates cmd using its previous value.
            cmd += ["--speed-cap", str(cap)]
        # if checks seed == 0 and policy in ['qr_tail', 'dqn_governed', 'qr_governed']; only a true condition runs the indented block.
        if seed == 0 and policy in ["qr_tail", "dqn_governed", "qr_governed"]:
            # The combined operator updates cmd using its previous value.
            cmd += ["--gif", str(STAGE / f"replay_{policy}.gif")]
        # with enters a temporary context and exits it afterward, even if an error interrupts the block.
        with out.with_suffix(".log").open("w") as f:
            # Start another Python command and check whether it succeeded.
            # Call subprocess.run using parentheses to supply its inputs; this line does not store a separate return value.
            subprocess.run(
                # Inside this continued call, name=value supplies an option by name rather than by its position.
                cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            )
    # Call print using parentheses to supply its inputs; this line does not store a separate return value.
    print(f"Confirmed {policy} seed {seed}", flush=True)


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
    p.add_argument("--stage", choices=["all", "report"], default="all")
    # = stores a result in a; call p.parse_args using parentheses to supply its inputs.
    a = p.parse_args()
    # if checks a.jobs < 1; only a true condition runs the indented block.
    if a.jobs < 1:
        # Call p.error using parentheses to supply its inputs; this line does not store a separate return value.
        p.error("--jobs must be positive")
    # Call lock using parentheses to supply its inputs; this line does not store a separate return value.
    lock()
    # if checks a.stage == 'all'; only a true condition runs the indented block.
    if a.stage == "all":
        # Run independent jobs together. Each writes its own files; this is not a shared training model.
        # with enters a temporary context and exits it afterward, even if an error interrupts the block.
        with ThreadPoolExecutor(max_workers=a.jobs) as pool:
            # Call list using parentheses to supply its inputs; this line does not store a separate return value.
            list(
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                pool.map(
                    # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                    evaluate, [(policy, s) for policy in POLICIES for s in range(3)]
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                )
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            )
    # from driving_rl.confirmation_report import brings the named tools into this file; parentheses allow several lines.
    from driving_rl.confirmation_report import report

    # Call report using parentheses to supply its inputs; this line does not store a separate return value.
    report()


# Run the entry point only when this file is executed, not when it is imported.
# __name__ equals "__main__" when this file runs directly; this guard avoids running the script merely because another file imports it.
if __name__ == "__main__":
    # Call main using parentheses to supply its inputs; this line does not store a separate return value.
    main()
