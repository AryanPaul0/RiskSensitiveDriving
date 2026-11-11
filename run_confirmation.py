"""Run the confirmation experiment from Terminal.

This is an experiment coordinator, not the learning algorithm. It launches
separate Python commands, saves logs, and checks existing results before reuse.
Paths identify saved files; hashes are fingerprints that detect changed files.
Read main() for the order of operations. Learning itself lives in driving_rl/agents.py.

Original implementation notes:
Evaluate locked policies once on the reserved scenario block.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from driving_rl.experiment import save_json
from driving_rl.source_integrity import source_matches_frozen_version

ROOT = Path(__file__).resolve().parent
STAGE = ROOT / "results/confirmation"
POLICIES = {
    "qr_tail": ("qrdqn", 0.25, None),
    "dqn_governed": ("dqn", 1.0, 25.0),
    "qr_governed": ("qrdqn", 1.0, 25.0),
    "dqn_mean": ("dqn", 1.0, None),
    "qr_mean": ("qrdqn", 1.0, None),
}


def read(p):
    """Read saved JSON data into ordinary Python dictionaries and lists."""
    return json.loads(p.read_text())


def sha(p):
    """Calculate a file fingerprint. If even one byte changes, its SHA-256 fingerprint
    will almost certainly change too. This helps detect changed frozen artifacts.
    """
    return hashlib.sha256(p.read_bytes()).hexdigest()


def checkpoint(policy, seed):
    """Find the saved model file for the requested algorithm and training seed."""
    algo = POLICIES[policy][0]
    return (
        ROOT
        / f'results/{"density_coverage" if algo=="qrdqn" else "baseline_controls"}/{algo}_seed{seed}/final.pt'
    )


def design():
    """Describe the fixed confirmation plan, including model fingerprints, scenario
    seeds, the primary comparison and the limits of the planned uncertainty interval.
    """
    return {
        "status": "Locked before confirmation evaluations",
        "training_steps": 40000,
        "training_seeds": [0, 1, 2],
        "training_densities": [1.0, 1.5, 2.0],
        "scenario_seed_start": 20000,
        "episodes_per_density": 100,
        "densities": [1.0, 1.5, 2.0, 2.5],
        "policies": {
            p: {"algorithm": a, "alpha": alpha, "speed_cap_m_s": cap}
            for p, (a, alpha, cap) in POLICIES.items()
        },
        "primary": {
            "left": "qr_tail",
            "right": "dqn_governed",
            "density": 1.0,
            "metric": "crashed",
            "direction": "Lower collision fraction is better",
            "estimand": "Expected paired collision difference over fresh scenarios, averaged over the three fixed trained-model pairs.",
            "interval": "Two-sided percentile 95% scenario-cluster bootstrap; 10000 replicates; RNG seed 20261111. Resample scenario IDs while retaining all fixed models and policy pairing.",
            "decision_rule": "Evidence of fewer collisions for the locked QR-tail policy if the upper conditional confidence endpoint is strictly below zero. Report speed/distance regardless of collision result. Do not claim speed-independent or real-road safety benefit.",
        },
        "secondary": "Other policies, traffic densities, seed variation and crossed bootstrap intervals are descriptive/exploratory; no family-wise inference claimed.",
        "generalization_limit": "Primary interval conditions on this selected set of trained models, architecture, rewards and governor caps; it does not cover new training runs or remove speed/action-rule differences.",
        "no_tuning": "No retraining, checkpoint selection, alpha adjustment or governor adjustment using these results. Further tuned models need a new untouched scenario block.",
        "replays": "Training seed 0, first reserved scenario 20000 at density 1, for QR tail and both governed controls; fixed before outcomes.",
        "checkpoint_hashes": {
            p: {
                str(s): {
                    "path": str(checkpoint(p, s).relative_to(ROOT)),
                    "sha256": sha(checkpoint(p, s)),
                }
                for s in range(3)
            }
            for p in POLICIES
        },
        "source_hashes": {
            str(p.relative_to(ROOT)): sha(p)
            for p in sorted((ROOT / "driving_rl").glob("*.py"))
            if p.name != "confirmation_report.py"
        },
    }


def lock():
    """Record the confirmation settings before looking at confirmation outcomes.
    On later runs, check the model files and settings rather than choosing again.
    """
    STAGE.mkdir(parents=True, exist_ok=True)
    path = STAGE / "design.json"
    value = design()
    # Check existing data instead of silently replacing scientific results.
    if path.exists():
        old = read(path)
        # Source additions do not alter the already locked design. Check all fixed parameters and hashes.
        if {k: v for k, v in old.items() if k != "source_hashes"} != {
            k: v for k, v in value.items() if k != "source_hashes"
        }:
            raise ValueError("Refusing to change the locked confirmation design")
        for name in ["agents.py", "envs.py", "experiment.py", "cli.py"]:
            key = "driving_rl/" + name
            if not source_matches_frozen_version(ROOT, key, old["source_hashes"][key]):
                raise ValueError(
                    "Evaluation code changed after confirmation lock: " + name
                )
        return old
    # Verify that this scenario block has not appeared in any previous saved episodes.
    audited = 0
    for p in (ROOT / "results").rglob("*.json"):
        if STAGE in p.parents:
            continue
        data = read(p)
        if isinstance(data, dict) and isinstance(data.get("episodes"), list):
            if any(20000 <= r.get("seed", -1) < 20100 for r in data["episodes"]):
                raise ValueError(f"Confirmation block already used: {p}")
            audited += 1
    save_json(
        STAGE / "preflight.json",
        {
            "existing_episode_files_checked": audited,
            "reserved_block_previously_unused": True,
        },
    )
    save_json(path, value)
    print(
        "Locked design before evaluation; reserved scenario block was unused.",
        flush=True,
    )
    return value


def evaluate(spec):
    """Run or verify one frozen-policy evaluation.
    Check its model fingerprint, settings and scenario block before reusing saved
    data. This function must not quietly mix results from different experiments.
    """
    policy, seed = spec
    algo, alpha, cap = POLICIES[policy]
    ck = checkpoint(policy, seed)
    out = STAGE / f"eval_{policy}_seed{seed}.json"
    # A saved file is useful only if its model and settings match this request.
    if out.exists():
        d = read(out)
        p = d["provenance"]
        if (
            p["checkpoint_sha256"],
            p["alpha"],
            p["speed_governor_cap_m_s"],
            p["seed_start"],
            p["episodes_per_density"],
            p["training_steps"],
            p["training_seed"],
            [r["density"] for r in d["summary"]],
        ) != (sha(ck), alpha, cap, 20000, 100, 40000, seed, [1.0, 1.5, 2.0, 2.5]):
            raise ValueError(f"Mismatched confirmation {out}")
        expected = {
            (density, scenario)
            for density in [1.0, 1.5, 2.0, 2.5]
            for scenario in range(20000, 20100)
        }
        if {(r["density"], r["seed"]) for r in d["episodes"]} != expected or len(
            d["episodes"]
        ) != 400:
            raise ValueError("Incomplete/duplicate confirmation cases")
    else:
        cmd = [
            sys.executable,
            "-m",
            "driving_rl.cli",
            "evaluate",
            "--checkpoint",
            str(ck),
            "--alpha",
            str(alpha),
            "--episodes",
            "100",
            "--seed-start",
            "20000",
            "--densities",
            "1",
            "1.5",
            "2",
            "2.5",
            "--out",
            str(out),
        ]
        if cap is not None:
            cmd += ["--speed-cap", str(cap)]
        if seed == 0 and policy in ["qr_tail", "dqn_governed", "qr_governed"]:
            cmd += ["--gif", str(STAGE / f"replay_{policy}.gif")]
        with out.with_suffix(".log").open("w") as f:
            # Start another Python command and check whether it succeeded.
            subprocess.run(
                cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True
            )
    print(f"Confirmed {policy} seed {seed}", flush=True)


def main():
    """Read Terminal options, check settings, and run the requested workflow.
    Existing result files are checked before reuse so a rerun cannot silently mix
    different experiments. The main guard below prevents running jobs on import.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jobs", type=int, default=3)
    p.add_argument("--stage", choices=["all", "report"], default="all")
    a = p.parse_args()
    if a.jobs < 1:
        p.error("--jobs must be positive")
    lock()
    if a.stage == "all":
        # Run independent jobs together. Each writes its own files; this is not a shared training model.
        with ThreadPoolExecutor(max_workers=a.jobs) as pool:
            list(
                pool.map(
                    evaluate, [(policy, s) for policy in POLICIES for s in range(3)]
                )
            )
    from driving_rl.confirmation_report import report

    report()


# Run the entry point only when this file is executed, not when it is imported.
if __name__ == "__main__":
    main()
