"""Check future-reward predictions without training or tuning any models.

Use development seeds only. DQN has a mean estimate, not a distribution.
For QR-DQN we can also check its 32 assigned quantile predictions.
"""

import argparse  # Make argparse available for this file's calculations and workflow.
from concurrent.futures import ThreadPoolExecutor  # Make ThreadPoolExecutor available for this file's calculations and workflow.
import hashlib  # Make hashlib available for this file's calculations and workflow.
import json  # Make json available for this file's calculations and workflow.
from pathlib import Path  # Make Path available for this file's calculations and workflow.
import subprocess  # Make subprocess available for this file's calculations and workflow.
import sys  # Make sys available for this file's calculations and workflow.

import numpy as np  # Make numpy available for this file's calculations and workflow.
import torch  # Make torch available for this file's calculations and workflow.

from driving_rl.agents import action_scores  # Make action_scores available for this file's calculations and workflow.
from driving_rl.calibration import discounted_return_bounds  # Make discounted_return_bounds available for this file's calculations and workflow.
from driving_rl.envs import make_env, flatten, seed_everything  # Make make_env, flatten, seed_everything available for this file's calculations and workflow.
from driving_rl.experiment import load_agent, save_json  # Make load_agent, save_json available for this file's calculations and workflow.

ROOT = Path(__file__).resolve().parent  # Set ROOT to Path( file ).resolve().parent.
OUTPUT = ROOT / "results/prediction_audit"  # Set OUTPUT to the path to 'results/prediction_audit' inside ROOT.
DENSITIES = [1.0, 1.5, 2.0, 2.5]  # Set DENSITIES to 1.0, 1.5, 2.0, 2.5.
POLICIES = {  # Set POLICIES to the named settings or measurements assembled below.
    "dqn_mean": ("dqn", 1.0),  # Record 'dqn_mean' as 'dqn', 1.0.
    "qr_mean": ("qrdqn", 1.0),  # Record 'qr_mean' as 'qrdqn', 1.0.
    "qr_tail": ("qrdqn", 0.25),  # Record 'qr_tail' as 'qrdqn', 0.25.
}  # Finish assembling the grouped values above.


def read(path):  # Load an ordinary JSON result or settings file.
    """Load an ordinary JSON result or settings file."""  # Describe this operation for readers and help().
    return json.loads(path.read_text())  # Return json.loads applied to the result of path.read text.


def model_path(budget, policy, seed):  # Find exactly the same frozen models used in the driving study.
    """Find exactly the same frozen models used in the driving study."""  # Describe this operation for readers and help().
    algorithm = POLICIES[policy][0]  # Set algorithm to the entries of the entries of POLICIES selected by policy selected by 0.
    if budget == 100000:  # Use this branch when the environment-interaction budget equals 100000.
        folder = "long_training_100k"  # Set folder to 'long_training_100k'.
    elif algorithm == "dqn":  # Use this branch when algorithm equals 'dqn'.
        folder = "baseline_controls"  # Set folder to 'baseline_controls'.
    else:  # Handle the alternative when the earlier condition (algorithm equals 'dqn') is false.
        folder = "density_coverage"  # Set folder to 'density_coverage'.
    return ROOT / f"results/{folder}/{algorithm}_seed{seed}/final.pt"  # Return (ROOT divided by f'results/{folder}/{algorithm} seed{seed}/final.pt').


def fingerprint(path):  # Make a file fingerprint, like an exact-content ID.
    """Make a file fingerprint, like an exact-content ID. We use it to catch accidental model changes."""  # Describe this operation for readers and help().
    return hashlib.sha256(path.read_bytes()).hexdigest()  # Return the result of hashlib.sha256(path.read bytes()).hexdigest.


def settings(budget, policy, seed):  # Save enough information to reject accidentally mixed experiments.
    """Save enough information to reject accidentally mixed experiments."""  # Describe this operation for readers and help().
    return {  # Return the named settings or measurements assembled below.
        "budget": budget,  # Record 'budget' as the environment-interaction budget.
        "policy": policy,  # Record 'policy' as policy.
        "training_seed": seed,  # Record 'training_seed' as seed.
        "checkpoint_sha256": fingerprint(model_path(budget, policy, seed)),  # Record 'checkpoint_sha256' as fingerprint applied to model path applied to the environment-interaction budget, policy, seed.
        "seed_start": 19000,  # Record 'seed_start' as 19000.
        "episodes_per_density": 60,  # Record 'episodes_per_density' as 60.
        "densities": DENSITIES,  # Record 'densities' as DENSITIES.
    }  # Finish assembling the grouped values above.


def record_path(budget, policy, seed):  # Choose a separate output filename for this budget, policy and training seed.
    """Choose a separate output filename for this budget, policy and training seed."""  # Describe this operation for readers and help().
    return OUTPUT / f"audit_{budget}_{policy}_seed{seed}.json"  # Return (OUTPUT divided by f'audit {budget} {policy} seed{seed}.json').


def collect(budget, policy, seed):  # Predict first, then observe the outcome.
    """Predict first, then observe the outcome. Never learn from this rollout."""  # Describe this operation for readers and help().
    seed_everything(0)  # Initialize repeatable random streams using 0.
    agent, saved = load_agent(model_path(budget, policy, seed))  # Set agent, saved to the frozen learner and checkpoint loaded from model path applied to the environment-interaction budget, policy, seed.
    gamma = saved["config"]["gamma"]  # Set the future-reward discount to the entries of the entries of saved selected by 'config' selected by 'gamma'.
    alpha = POLICIES[policy][1]  # Set the fraction of low-return predictions used to score actions to the entries of the entries of POLICIES selected by policy selected by 1.
    rows = []  # Set rows to .
    for density in DENSITIES:  # Traffic density changes vehicle spacing. It does not add more network inputs.
        env = make_env(density=density)  # Set env to a highway simulator configured with.
        try:  # Run this work with the error handling or resource cleanup specified below.
            for scenario in range(19000, 19060):  # A scenario seed fixes the starting traffic. Reuse the same IDs for fair policy comparisons.
                observation, _ = env.reset(seed=scenario)  # Set the currently observed traffic features, the unused result to the result of env.reset.
                inputs = torch.as_tensor(flatten(observation)).unsqueeze(0)  # Set inputs to torch.as tensor(flatten(observation)).unsqueeze applied to 0.
                with torch.no_grad():  # Treat predictions in this block as fixed learning targets or inference results; do not backpropagate through them.
                    outputs = agent.online(inputs)  # Set outputs to agent.online applied to inputs.
                    first_action = int(action_scores(outputs, alpha).argmax().item())  # Choose using the stated rule before observing what happens next.
                values = outputs[0, first_action].tolist()  # Set values to the entries of outputs selected by (0, first_action) as a list that can be saved in JSON.
                prediction = float(outputs[0, first_action].mean().item())  # 0 selects the only observation in this inference batch; first_action selects its chosen maneuver. What remains is that action's 32 quantiles (or one DQN scalar), not all actions. This is the mean future-return estimate for the selected action, not the next reward or a crash probability.
                quantiles = values if agent.algorithm == "qrdqn" else None  # Average all heads even for the cautious policy: this audit separately checks its mean prediction. Its action was chosen with a tail score, which is a different quantity. A scalar DQN output is NOT a median or a set of quantiles.
                rewards = []  # Set the observed rewards to .
                speeds = []  # Set speeds to .
                action = first_action  # Set action to first action.
                while True:  # Keep processing while True.
                    observation, reward, terminal, cutoff, _ = env.step(action)  # Set the currently observed traffic features, reward, terminal, cutoff, the unused result to env.step applied to action.
                    if not -2.000001 <= reward <= 0.400001 or (  # Use this branch when not negative 2.000001 is at most reward is at most 0.400001 or reward is below 0 and not terminal.
                        reward < 0 and not terminal  # Include reward is below 0 and not terminal in the surrounding operation.
                    ):  # Finish assembling the grouped values above.
                        raise ValueError(  # Reject this invalid case instead of continuing with misleading experiment results.
                            "Reward assumptions for the return bounds changed"  # Include 'Reward assumptions for the return bounds changed' in the surrounding operation.
                        )  # Finish assembling ValueError inputs.
                    rewards.append(float(reward))  # Add reward as a floating-point value to the observed rewards.
                    speeds.append(float(env.unwrapped.vehicle.speed))  # Add env.unwrapped.vehicle.speed as a floating-point value to speeds.
                    if terminal or cutoff:  # Use this branch when terminal or cutoff.
                        break  # Stop this loop now that its stopping condition has been reached.
                    action = agent.act(flatten(observation), alpha)  # Set action to agent.act applied to the 40 network inputs flattened from the currently observed traffic features, the fraction of low-return predictions used to score actions.
                low, high = discounted_return_bounds(rewards, gamma, terminal)  # Continue under the deployed selector being audited. Tail-policy outcomes need not follow the mean-greedy continuation used to construct QR-DQN's training targets. A crash makes the return exact. A recording cutoff leaves a bounded unknown future.
                rows.append(  # Add the named settings or measurements assembled below to rows.
                    {  # Include the named settings or measurements assembled below in the surrounding operation.
                        "density": density,  # Record 'density' as the traffic-density setting.
                        "seed": scenario,  # Record 'seed' as the starting-traffic seed.
                        "mean_prediction": prediction,  # Record 'mean_prediction' as prediction.
                        "quantiles": quantiles,  # Record 'quantiles' as quantiles.
                        "return_lower": low,  # Record 'return_lower' as low.
                        "return_upper": high,  # Record 'return_upper' as high.
                        "first_action": first_action,  # Record 'first_action' as first action.
                        "length": len(rewards),  # Record 'length' as the number of entries in the observed rewards.
                        "rewards": rewards,  # Record 'rewards' as the observed rewards.
                        "censored": bool(cutoff and not terminal),  # Record 'censored' as cutoff and not terminal as a true/false value.
                        "crashed": bool(env.unwrapped.vehicle.crashed),  # Record 'crashed' as env.unwrapped.vehicle.crashed as a true/false value.
                        "mean_speed_m_s": float(np.mean(speeds)),  # Record 'mean_speed_m_s' as the average of speeds as a floating-point value.
                    }  # Finish assembling the grouped values above.
                )  # Finish assembling rows.append inputs.
        finally:  # Always release resources here, including after an error or interruption.
            env.close()  # The result of env.close.
    save_json(  # Save the following results/settings as json: record path applied to the environment-interaction budget, policy, seed, the named settings or measurements assembled below.
        record_path(budget, policy, seed),  # Include record path applied to the environment-interaction budget, policy, seed in the surrounding operation.
        {"provenance": settings(budget, policy, seed), "episodes": rows},  # Record 'provenance' as settings applied to the environment-interaction budget, policy, seed.
    )  # Finish assembling save json inputs.


def run_collection(job):  # Separate processes keep random generators independent between jobs.
    """Separate processes keep random generators independent between jobs."""  # Describe this operation for readers and help().
    budget, policy, seed = job  # Set the environment-interaction budget, policy, seed to job.
    path = record_path(budget, policy, seed)  # Set path to record path applied to the environment-interaction budget, policy, seed.
    if path.exists():  # A saved filename is not enough: the checks below verify the contents belong to this experiment.
        data = read(path)  # Set data to read applied to path.
        if data["provenance"] != settings(budget, policy, seed):  # Use this branch when the entries of data selected by 'provenance' differs from settings applied to the environment-interaction budget, policy, seed.
            raise ValueError("Cached prediction checks belong to another model")  # Reject this invalid case instead of continuing with misleading experiment results.
        cases = set()  # A set keeps each density/scenario pair once, even if a row was duplicated.
        for episode in data["episodes"]:  # Process each episode from the entries of data selected by 'episodes'.
            cases.add((episode["density"], episode["seed"]))  # Cases.add applied to the entries of episode selected by 'density', the entries of episode selected by 'seed'.
        expected = set()  # Set expected to the result of set.
        for density in DENSITIES:  # Traffic density changes vehicle spacing. It does not add more network inputs.
            for scenario in range(19000, 19060):  # A scenario seed fixes the starting traffic. Reuse the same IDs for fair policy comparisons.
                expected.add((density, scenario))  # Expected.add applied to the traffic-density setting, the starting-traffic seed.
        if len(data["episodes"]) != 240 or cases != expected:  # Use this branch when the number of entries in the entries of data selected by 'episodes' differs from 240 or cases differs from expected.
            raise ValueError("Missing or duplicate prediction cases")  # Reject this invalid case instead of continuing with misleading experiment results.
    else:  # Handle the alternative when the earlier condition (the result of path.exists) is false.
        command = [  # Set command to the listed settings or measurements in their recorded order.
            sys.executable,  # Include sys.executable in the surrounding operation.
            __file__,  # Include file in the surrounding operation.
            "--collect",  # Include '--collect' in the surrounding operation.
            str(budget),  # Include the environment-interaction budget as a text value in the surrounding operation.
            policy,  # Include policy in the surrounding operation.
            str(seed),  # Include seed as a text value in the surrounding operation.
        ]  # Finish assembling the grouped values above.
        with path.with_suffix(".log").open("w") as log:  # Keep this temporary context active during the enclosed work, then clean it up.
            subprocess.run(  # Run the requested Python command and stop on failure. This is evaluation, not a training call.
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True  # Include command, ROOT, log, subprocess.STDOUT, True in the surrounding operation.
            )  # Finish assembling subprocess.run inputs.
    print(f"Checked {budget} {policy} seed {seed}", flush=True)  # Print applied to f'checked {budget} {policy} seed {seed}'.


def main():  # Read the Terminal options and coordinate the workflow.
    """Read the Terminal options and coordinate the workflow. The learning algorithm is not implemented here."""  # Describe this operation for readers and help().
    parser = argparse.ArgumentParser(description=__doc__)  # Set parser to the result of argparse.ArgumentParser.
    parser.add_argument("--jobs", type=int, default=4)  # Parser.add argument applied to '--jobs'.
    parser.add_argument("--collect", nargs=3, metavar=("BUDGET", "POLICY", "SEED"))  # Parser.add argument applied to '--collect'.
    parser.add_argument("--report-only", action="store_true")  # Parser.add argument applied to '--report-only'.
    args = parser.parse_args()  # Set args to the result of parser.parse args.
    OUTPUT.mkdir(exist_ok=True)  # The result of output.mkdir.
    if args.collect:  # Use this branch when args.collect.
        budget, policy, seed = args.collect  # Set the environment-interaction budget, policy, seed to args.collect.
        if (  # Use this branch when the environment-interaction budget as a whole-number value does not belong to 40000, 100000 or policy does not belong to POLICIES or seed as a whole-number value does not belong to range applied to 3.
            int(budget) not in [40000, 100000]  # Include the environment-interaction budget as a whole-number value does not belong to 40000, 100000 or policy does not belong to POLICIES or seed as a whole-number value does not belong to range applied to 3 in the surrounding operation.
            or policy not in POLICIES  # Include policy does not belong to POLICIES in the surrounding operation.
            or int(seed) not in range(3)  # Include seed as a whole-number value does not belong to range applied to 3 in the surrounding operation.
        ):  # Finish assembling the grouped values above.
            parser.error("Unknown model or policy")  # Parser.error applied to 'unknown model or policy'.
        collect(int(budget), policy, int(seed))  # Collect applied to the environment-interaction budget as a whole-number value, policy, seed as a whole-number value.
        return  # Finish this operation without producing another result.
    if args.jobs < 1:  # Use this branch when args.jobs is below 1.
        parser.error("--jobs must be positive")  # Parser.error applied to '--jobs must be positive'.
    design = {  # Set design to the named settings or measurements assembled below.
        "budgets": [40000, 100000],  # Record 'budgets' as 40000, 100000.
        "policies": POLICIES,  # Record 'policies' as POLICIES.
        "development_seeds": [19000, 19059],  # Record 'development_seeds' as 19000, 19059.
        "densities": DENSITIES,  # Record 'densities' as DENSITIES.
        "training_seeds": [0, 1, 2],  # Record 'training_seeds' as 0, 1, 2.
        "tuning": False,  # Record 'tuning' as False.
        "purpose": "Descriptive prediction diagnostics, not confirmation or a replacement primary test.",  # Record 'purpose' as the report text.
        "return_bounds": "A terminal return is exact. Successful 30-step recordings retain bounded unknown continuation.",  # Record 'return_bounds' as the report text.
        "interpretation": "Predictions concern the initial chosen action. Policies can choose different actions and induce different outcomes. QR-tail continuation differs from mean-greedy training.",  # Record 'interpretation' as the report text.
    }  # Finish assembling the grouped values above.
    design = json.loads(json.dumps(design))  # JSON turns tuples into lists. Normalize that detail before checking a recorded design.
    design_path = OUTPUT / "design.json"  # Set design path to the path to 'design.json' inside OUTPUT.
    if design_path.exists() and read(design_path) != design:  # Use this branch when the result of design path.exists and read applied to design path differs from design.
        raise ValueError("Refusing to silently change the diagnostic design")  # Reject this invalid case instead of continuing with misleading experiment results.
    if not design_path.exists():  # Use this branch when not the result of design path.exists.
        save_json(design_path, design)  # Save the following results/settings as json: design path, design.
    if not args.report_only:  # Use this branch when not args.report only.
        jobs = []  # Set jobs to .
        for budget in [40000, 100000]:  # Each budget gets its own results. Keep 40K and 100K separate until the paired comparison.
            for policy in POLICIES:  # A policy is a decision rule. Several rules can share the very same QR-DQN network.
                for seed in range(3):  # A training seed identifies one independently trained model, not a new traffic scenario.
                    jobs.append((budget, policy, seed))  # Add the environment-interaction budget, policy, seed to jobs.
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:  # This only schedules independent jobs. Each actual experiment runs in a separate Python process.
            list(pool.map(run_collection, jobs))  # Wait for all requested jobs, and let errors surface instead of quietly ignoring failed experiments.
    from driving_rl.prediction_audit_report import report  # Make report available for this file's calculations and workflow.

    report()  # The result of report.


if __name__ == "__main__":  # Run this workflow when launched from Terminal; importing its helpers must not launch experiments.
    main()  # The result of main.
