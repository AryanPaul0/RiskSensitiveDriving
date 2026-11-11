"""Check future-reward predictions without training or tuning any models.

Use development seeds only. DQN has a mean estimate, not a distribution.
For QR-DQN we can also check its 32 assigned quantile predictions.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from driving_rl.agents import action_scores
from driving_rl.calibration import discounted_return_bounds
from driving_rl.envs import make_env, flatten, seed_everything
from driving_rl.experiment import load_agent, save_json

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results/prediction_audit"
DENSITIES = [1.0, 1.5, 2.0, 2.5]
POLICIES = {
    "dqn_mean": ("dqn", 1.0),
    "qr_mean": ("qrdqn", 1.0),
    "qr_tail": ("qrdqn", 0.25),
}


def read(path):
    """Load an ordinary JSON result or settings file."""
    return json.loads(path.read_text())


def model_path(budget, policy, seed):
    """Find exactly the same frozen models used in the driving study."""
    algorithm = POLICIES[policy][0]
    if budget == 100000:
        folder = "long_training_100k"
    elif algorithm == "dqn":
        folder = "baseline_controls"
    else:
        folder = "density_coverage"
    return ROOT / f"results/{folder}/{algorithm}_seed{seed}/final.pt"


def fingerprint(path):
    """Make a file fingerprint, like an exact-content ID. We use it to catch accidental model changes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def settings(budget, policy, seed):
    """Save enough information to reject accidentally mixed experiments."""
    return {
        "budget": budget,
        "policy": policy,
        "training_seed": seed,
        "checkpoint_sha256": fingerprint(model_path(budget, policy, seed)),
        "seed_start": 19000,
        "episodes_per_density": 60,
        "densities": DENSITIES,
    }


def record_path(budget, policy, seed):
    """Choose a separate output filename for this budget, policy and training seed."""
    return OUTPUT / f"audit_{budget}_{policy}_seed{seed}.json"


def collect(budget, policy, seed):
    """Predict first, then observe the outcome. Never learn from this rollout."""
    seed_everything(0)
    agent, saved = load_agent(model_path(budget, policy, seed))
    gamma = saved["config"]["gamma"]
    alpha = POLICIES[policy][1]
    rows = []
    # Traffic density changes vehicle spacing. It does not add more network inputs.
    for density in DENSITIES:
        env = make_env(density=density)
        try:
            # A scenario seed fixes the starting traffic. Reuse the same IDs for fair policy comparisons.
            for scenario in range(19000, 19060):
                observation, _ = env.reset(seed=scenario)
                inputs = torch.as_tensor(flatten(observation)).unsqueeze(0)
                # We are asking the saved network for predictions. No learning or weight update happens here.
                with torch.no_grad():
                    outputs = agent.online(inputs)
                    # Choose using the stated rule before observing what happens next.
                    first_action = int(action_scores(outputs, alpha).argmax().item())
                values = outputs[0, first_action].tolist()
                # 0 selects the only observation in this inference batch;
                # first_action selects its chosen maneuver. What remains is
                # that action's 32 quantiles (or one DQN scalar), not all actions.
                # This is the mean future-return estimate for the selected action, not the next reward or a crash probability.
                prediction = float(outputs[0, first_action].mean().item())
                # Average all heads even for the cautious policy: this audit
                # separately checks its mean prediction. Its action was chosen
                # with a tail score, which is a different quantity.
                # A scalar DQN output is NOT a median or a set of quantiles.
                quantiles = values if agent.algorithm == "qrdqn" else None
                rewards = []
                speeds = []
                action = first_action
                while True:
                    observation, reward, terminal, cutoff, _ = env.step(action)
                    if not -2.000001 <= reward <= 0.400001 or (
                        reward < 0 and not terminal
                    ):
                        raise ValueError(
                            "Reward assumptions for the return bounds changed"
                        )
                    rewards.append(float(reward))
                    speeds.append(float(env.unwrapped.vehicle.speed))
                    if terminal or cutoff:
                        break
                    action = agent.act(flatten(observation), alpha)
                    # Continue under the deployed selector being audited.
                    # Tail-policy outcomes need not follow the mean-greedy
                    # continuation used to construct QR-DQN's training targets.
                # A crash makes the return exact. A recording cutoff leaves a bounded unknown future.
                low, high = discounted_return_bounds(rewards, gamma, terminal)
                rows.append(
                    {
                        "density": density,
                        "seed": scenario,
                        "mean_prediction": prediction,
                        "quantiles": quantiles,
                        "return_lower": low,
                        "return_upper": high,
                        "first_action": first_action,
                        "length": len(rewards),
                        "rewards": rewards,
                        "censored": bool(cutoff and not terminal),
                        "crashed": bool(env.unwrapped.vehicle.crashed),
                        "mean_speed_m_s": float(np.mean(speeds)),
                    }
                )
        finally:
            env.close()
    save_json(
        record_path(budget, policy, seed),
        {"provenance": settings(budget, policy, seed), "episodes": rows},
    )


def run_collection(job):
    """Separate processes keep random generators independent between jobs."""
    budget, policy, seed = job
    path = record_path(budget, policy, seed)
    # A saved filename is not enough: the checks below verify the contents belong to this experiment.
    if path.exists():
        data = read(path)
        if data["provenance"] != settings(budget, policy, seed):
            raise ValueError("Cached prediction checks belong to another model")
        # A set keeps each density/scenario pair once, even if a row was duplicated.
        cases = set()
        for episode in data["episodes"]:
            cases.add((episode["density"], episode["seed"]))
        expected = set()
        # Traffic density changes vehicle spacing. It does not add more network inputs.
        for density in DENSITIES:
            # A scenario seed fixes the starting traffic. Reuse the same IDs for fair policy comparisons.
            for scenario in range(19000, 19060):
                expected.add((density, scenario))
        if len(data["episodes"]) != 240 or cases != expected:
            raise ValueError("Missing or duplicate prediction cases")
    else:
        command = [
            sys.executable,
            __file__,
            "--collect",
            str(budget),
            policy,
            str(seed),
        ]
        with path.with_suffix(".log").open("w") as log:
            # Run the requested Python command and stop on failure. This is evaluation, not a training call.
            subprocess.run(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True
            )
    print(f"Checked {budget} {policy} seed {seed}", flush=True)


def main():
    """Read the Terminal options and coordinate the workflow. The learning algorithm is not implemented here."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--collect", nargs=3, metavar=("BUDGET", "POLICY", "SEED"))
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(exist_ok=True)
    if args.collect:
        budget, policy, seed = args.collect
        if (
            int(budget) not in [40000, 100000]
            or policy not in POLICIES
            or int(seed) not in range(3)
        ):
            parser.error("Unknown model or policy")
        collect(int(budget), policy, int(seed))
        return
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    design = {
        "budgets": [40000, 100000],
        "policies": POLICIES,
        "development_seeds": [19000, 19059],
        "densities": DENSITIES,
        "training_seeds": [0, 1, 2],
        "tuning": False,
        "purpose": "Descriptive prediction diagnostics, not confirmation or a replacement primary test.",
        "return_bounds": "A terminal return is exact. Successful 30-step recordings retain bounded unknown continuation.",
        "interpretation": "Predictions concern the initial chosen action. Policies can choose different actions and induce different outcomes. QR-tail continuation differs from mean-greedy training.",
    }
    # JSON turns tuples into lists. Normalize that detail before checking a recorded design.
    design = json.loads(json.dumps(design))
    design_path = OUTPUT / "design.json"
    if design_path.exists() and read(design_path) != design:
        raise ValueError("Refusing to silently change the diagnostic design")
    if not design_path.exists():
        save_json(design_path, design)
    if not args.report_only:
        jobs = []
        # Each budget gets its own results. Keep 40K and 100K separate until the paired comparison.
        for budget in [40000, 100000]:
            # A policy is a decision rule. Several rules can share the very same QR-DQN network.
            for policy in POLICIES:
                # A training seed identifies one independently trained model, not a new traffic scenario.
                for seed in range(3):
                    jobs.append((budget, policy, seed))
        # This only schedules independent jobs. Each actual experiment runs in a separate Python process.
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            # Wait for all requested jobs, and let errors surface instead of quietly ignoring failed experiments.
            list(pool.map(run_collection, jobs))
    from driving_rl.prediction_audit_report import report

    report()


if __name__ == "__main__":
    main()
