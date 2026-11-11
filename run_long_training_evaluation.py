"""Compare completed 40K and 100K models without training or tuning anything.

Every model gets the same fresh scenarios. Earlier confirmation cases are kept
separate. Run this file again to check and reuse saved evaluations.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from driving_rl.experiment import save_json
from run_confirmation import POLICIES

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results/long_training_evaluation"
DENSITIES = [1.0, 1.5, 2.0, 2.5]


def read(path):
    """Read settings or results that have already been saved."""
    return json.loads(path.read_text())


def fingerprint(path):
    """Identify the exact file, not just a model with a similar name."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_path(budget, policy, seed):
    """The policy chooses a decision rule; several rules share one network."""
    algorithm = POLICIES[policy][0]
    if budget == 100000:
        folder = "long_training_100k"
    elif algorithm == "dqn":
        folder = "baseline_controls"
    else:
        folder = "density_coverage"
    return ROOT / f"results/{folder}/{algorithm}_seed{seed}/final.pt"


def lock_design():
    """Fix settings and model fingerprints before reading new outcomes."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    models = {}
    # Each budget gets its own results. Keep 40K and 100K separate until the paired comparison.
    for budget in [40000, 100000]:
        models[str(budget)] = {}
        # A policy is a decision rule. Several rules can share the very same QR-DQN network.
        for policy in POLICIES:
            models[str(budget)][policy] = {}
            # A training seed identifies one independently trained model, not a new traffic scenario.
            for seed in range(3):
                path = model_path(budget, policy, seed)
                summary = read(path.parent / "summary.json")
                if summary["steps"] != budget or not summary["training_complete"]:
                    raise ValueError("A requested training run is incomplete")
                models[str(budget)][policy][str(seed)] = fingerprint(path)
    design = {
        "budgets": [40000, 100000],
        "model_hashes": models,
        "scenario_seed_start": 20100,
        "episodes_per_density": 100,
        "densities": DENSITIES,
        "policies": POLICIES,
        "primary": "100K QR lower quarter minus 100K DQN with the frozen 25 m/s governor, density 1, collision fraction.",
        "primary_rule": "Conditional scenario-bootstrap 95% interval must have upper endpoint below zero; report speed regardless. No tuning.",
        "secondary": "Budget effects pair the SAME new scenarios. Other policies/densities and three-seed crossed intervals are exploratory.",
        "scope": "Primary inference conditions on selected networks. Equal aggregate speed is not guaranteed. No new-training or real-road safety claim.",
        "consumed_prior_confirmation": [20000, 20099],
        "future_untouched_seeds_begin": 20200,
    }
    # JSON turns tuples into lists. Normalize that detail before checking a recorded design.
    design = json.loads(json.dumps(design))
    path = OUTPUT / "design.json"
    # A saved filename is not enough: the checks below verify the contents belong to this experiment.
    if path.exists():
        if read(path) != design:
            raise ValueError("The recorded study cannot be silently changed")
    else:
        for old in (ROOT / "results").rglob("*.json"):
            if OUTPUT in old.parents:
                continue
            data = read(old)
            if isinstance(data, dict) and isinstance(data.get("episodes"), list):
                for row in data["episodes"]:
                    if 20100 <= row.get("seed", -1) < 20200:
                        raise ValueError("The requested scenario block was used before")
        save_json(path, design)
        print("Design locked; fresh scenario block verified unused.", flush=True)


def evaluate(job):
    """Use the existing evaluator; this code never calls training."""
    budget, policy, seed = job
    path = model_path(budget, policy, seed)
    algorithm, alpha, cap = POLICIES[policy]
    output = OUTPUT / f"eval_{budget}_{policy}_seed{seed}.json"
    # Do not overwrite an experiment. First check whether its saved settings match this request.
    if output.exists():
        data = read(output)
        saved = data["provenance"]
        actual = (
            saved["checkpoint_sha256"],
            saved["alpha"],
            saved["speed_governor_cap_m_s"],
            saved["training_steps"],
            saved["seed_start"],
            saved["episodes_per_density"],
        )
        expected = (fingerprint(path), alpha, cap, budget, 20100, 100)
        if actual != expected or len(data["episodes"]) != 400:
            raise ValueError(f"Mismatched evaluation: {output}")
        # A set keeps each density/scenario pair once, even if a row was duplicated.
        cases = set()
        for episode in data["episodes"]:
            cases.add((episode["density"], episode["seed"]))
        expected_cases = set()
        for density in DENSITIES:
            for scenario in range(20100, 20200):
                expected_cases.add((density, scenario))
        # We need every requested density/scenario pair, not just the right total number of rows.
        if cases != expected_cases:
            raise ValueError("Missing or duplicated scenarios")
    else:
        command = [
            sys.executable,
            "-m",
            "driving_rl.cli",
            "evaluate",
            "--checkpoint",
            str(path),
            "--alpha",
            str(alpha),
            "--seed-start",
            "20100",
            "--episodes",
            "100",
            "--densities",
            "1",
            "1.5",
            "2",
            "2.5",
            "--out",
            str(output),
        ]
        # The governor is an optional action override. Its threshold is not a guaranteed physical speed limit.
        if cap is not None:
            command.extend(["--speed-cap", str(cap)])
        with output.with_suffix(".log").open("w") as log:
            # Run the requested Python command and stop on failure. This is evaluation, not a training call.
            subprocess.run(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True
            )
    print(f"Evaluated {budget} {policy} seed {seed}", flush=True)


def main():
    """Read the Terminal options and coordinate the workflow. The learning algorithm is not implemented here."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    lock_design()
    if not args.report_only:
        jobs = []
        # Each budget gets its own results. Keep 40K and 100K separate until the paired comparison.
        for budget in [100000, 40000]:
            # A policy is a decision rule. Several rules can share the very same QR-DQN network.
            for policy in POLICIES:
                # A training seed identifies one independently trained model, not a new traffic scenario.
                for seed in range(3):
                    jobs.append((budget, policy, seed))
        # This only schedules independent jobs. Each actual experiment runs in a separate Python process.
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            # Wait for all requested jobs, and let errors surface instead of quietly ignoring failed experiments.
            list(pool.map(evaluate, jobs))
    from driving_rl.long_training_report import report

    report()


if __name__ == "__main__":
    main()
