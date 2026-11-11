"""Check future-reward predictions without training or tuning any models.

Use development seeds only. DQN has a mean estimate, not a distribution.
For QR-DQN we can also check its 32 assigned quantile predictions.
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

# import makes these modules available; "as" gives a module a shorter local name.
import numpy as np
# import makes these modules available; "as" gives a module a shorter local name.
import torch

# from driving_rl.agents import brings the named tools into this file; parentheses allow several lines.
from driving_rl.agents import action_scores
# from driving_rl.calibration import brings the named tools into this file; parentheses allow several lines.
from driving_rl.calibration import discounted_return_bounds
# from driving_rl.envs import brings the named tools into this file; parentheses allow several lines.
from driving_rl.envs import make_env, flatten, seed_everything
# from driving_rl.experiment import brings the named tools into this file; parentheses allow several lines.
from driving_rl.experiment import load_agent, save_json

# = stores a result in ROOT; use a dot to access parent on Path(__file__).resolve().
ROOT = Path(__file__).resolve().parent
# = stores a result in OUTPUT; apply / to these operands; numbers are divided, while Path objects join folder names.
OUTPUT = ROOT / "results/prediction_audit"
# = stores a result in DENSITIES; build a list, keeping its items in order.
DENSITIES = [1.0, 1.5, 2.0, 2.5]
# = stores a result in POLICIES; build a dictionary, which stores values under named keys.
POLICIES = {
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "dqn_mean": ("dqn", 1.0),
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "qr_mean": ("qrdqn", 1.0),
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "qr_tail": ("qrdqn", 0.25),
# } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
}


# def defines read; parentheses list its inputs, and the colon starts its indented body.
def read(path):
    # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
    """Load an ordinary JSON result or settings file."""
    # return ends this function and sends its result back to the code that called it.
    return json.loads(path.read_text())


# def defines model_path; parentheses list its inputs, and the colon starts its indented body.
def model_path(budget, policy, seed):
    # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
    """Find exactly the same frozen models used in the driving study."""
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


# def defines fingerprint; parentheses list its inputs, and the colon starts its indented body.
def fingerprint(path):
    # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
    """Make a file fingerprint, like an exact-content ID. We use it to catch accidental model changes."""
    # return ends this function and sends its result back to the code that called it.
    return hashlib.sha256(path.read_bytes()).hexdigest()


# def defines settings; parentheses list its inputs, and the colon starts its indented body.
def settings(budget, policy, seed):
    # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
    """Save enough information to reject accidentally mixed experiments."""
    # return ends this function and sends its result back to the code that called it.
    return {
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "budget": budget,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "policy": policy,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "training_seed": seed,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "checkpoint_sha256": fingerprint(model_path(budget, policy, seed)),
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "seed_start": 19000,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "episodes_per_density": 60,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "densities": DENSITIES,
    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
    }


# def defines record_path; parentheses list its inputs, and the colon starts its indented body.
def record_path(budget, policy, seed):
    # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
    """Choose a separate output filename for this budget, policy and training seed."""
    # return ends this function and sends its result back to the code that called it.
    return OUTPUT / f"audit_{budget}_{policy}_seed{seed}.json"


# def defines collect; parentheses list its inputs, and the colon starts its indented body.
def collect(budget, policy, seed):
    # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
    """Predict first, then observe the outcome. Never learn from this rollout."""
    # Call seed_everything using parentheses to supply its inputs; this line does not store a separate return value.
    seed_everything(0)
    # = unpacks the right-hand result into (agent, saved), matching values to names in order.
    agent, saved = load_agent(model_path(budget, policy, seed))
    # = stores a result in gamma; use square brackets to select part of saved['config'].
    gamma = saved["config"]["gamma"]
    # = stores a result in alpha; use square brackets to select part of POLICIES[policy].
    alpha = POLICIES[policy][1]
    # = stores a result in rows; build a list, keeping its items in order.
    rows = []
    # Traffic density changes vehicle spacing. It does not add more network inputs.
    # for takes each item from DENSITIES in turn; density holds the current item.
    for density in DENSITIES:
        # = stores a result in env; call make_env using parentheses to supply its inputs.
        env = make_env(density=density)
        # try starts a block whose errors or cleanup are handled by the following except/finally blocks.
        try:
            # A scenario seed fixes the starting traffic. Reuse the same IDs for fair policy comparisons.
            # for takes each item from range(19000, 19060) in turn; scenario holds the current item.
            for scenario in range(19000, 19060):
                # = unpacks the right-hand result into (observation, _), matching values to names in order.
                observation, _ = env.reset(seed=scenario)
                # = stores a result in inputs; call torch.as_tensor(flatten(observation)).unsqueeze using parentheses to supply its inputs.
                inputs = torch.as_tensor(flatten(observation)).unsqueeze(0)
                # We are asking the saved network for predictions. No learning or weight update happens here.
                # with temporarily disables gradient tracking for this block; these predictions are fixed answers, not a second training path.
                with torch.no_grad():
                    # = stores a result in outputs; call agent.online using parentheses to supply its inputs.
                    outputs = agent.online(inputs)
                    # Choose using the stated rule before observing what happens next.
                    # = stores a result in first_action; call int using parentheses to supply its inputs.
                    first_action = int(action_scores(outputs, alpha).argmax().item())
                # = stores a result in values; call outputs[0, first_action].tolist using parentheses to supply its inputs.
                values = outputs[0, first_action].tolist()
                # This is the mean future-return estimate for the selected action, not the next reward or a crash probability.
                # = stores a result in prediction; call float using parentheses to supply its inputs.
                prediction = float(outputs[0, first_action].mean().item())
                # A scalar DQN output is NOT a median or a set of quantiles.
                # = stores a result in quantiles; choose the value before "if" when its condition is true, otherwise the value after "else".
                quantiles = values if agent.algorithm == "qrdqn" else None
                # = stores a result in rewards; build a list, keeping its items in order.
                rewards = []
                # = stores a result in speeds; build a list, keeping its items in order.
                speeds = []
                # = stores a result in action; evaluate the expression on the right before using its result.
                action = first_action
                # while repeats the indented block as long as True is true.
                while True:
                    # = unpacks the right-hand result into (observation, reward, terminal, cutoff, _), matching values to names in order.
                    observation, reward, terminal, cutoff, _ = env.step(action)
                    # if checks not -2.000001 <= reward <= 0.400001 or (reward < 0 and (not terminal)); only a true condition runs the indented block.
                    if not -2.000001 <= reward <= 0.400001 or (
                        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                        reward < 0 and not terminal
                    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                    ):
                        # raise signals an error and stops this normal path, so invalid data are not silently accepted.
                        raise ValueError(
                            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                            "Reward assumptions for the return bounds changed"
                        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                        )
                    # Call rewards.append using parentheses to supply its inputs; this line does not store a separate return value.
                    rewards.append(float(reward))
                    # Call speeds.append using parentheses to supply its inputs; this line does not store a separate return value.
                    speeds.append(float(env.unwrapped.vehicle.speed))
                    # if checks terminal or cutoff; only a true condition runs the indented block.
                    if terminal or cutoff:
                        # break leaves the nearest loop immediately; execution continues after that loop.
                        break
                    # = stores a result in action; call agent.act using parentheses to supply its inputs.
                    action = agent.act(flatten(observation), alpha)
                # A crash makes the return exact. A recording cutoff leaves a bounded unknown future.
                # = unpacks the right-hand result into (low, high), matching values to names in order.
                low, high = discounted_return_bounds(rewards, gamma, terminal)
                # Call rows.append using parentheses to supply its inputs; this line does not store a separate return value.
                rows.append(
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    {
                        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                        "density": density,
                        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                        "seed": scenario,
                        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                        "mean_prediction": prediction,
                        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                        "quantiles": quantiles,
                        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                        "return_lower": low,
                        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                        "return_upper": high,
                        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                        "first_action": first_action,
                        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                        "length": len(rewards),
                        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                        "rewards": rewards,
                        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                        "censored": bool(cutoff and not terminal),
                        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                        "crashed": bool(env.unwrapped.vehicle.crashed),
                        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                        "mean_speed_m_s": float(np.mean(speeds)),
                    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
                    }
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                )
        # finally always runs this cleanup block when the try block ends, including after an error or return.
        finally:
            # Call env.close using parentheses to supply its inputs; this line does not store a separate return value.
            env.close()
    # Call save_json using parentheses to supply its inputs; this line does not store a separate return value.
    save_json(
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        record_path(budget, policy, seed),
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        {"provenance": settings(budget, policy, seed), "episodes": rows},
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )


# def defines run_collection; parentheses list its inputs, and the colon starts its indented body.
def run_collection(job):
    # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
    """Separate processes keep random generators independent between jobs."""
    # = unpacks the right-hand result into (budget, policy, seed), matching values to names in order.
    budget, policy, seed = job
    # = stores a result in path; call record_path using parentheses to supply its inputs.
    path = record_path(budget, policy, seed)
    # A saved filename is not enough: the checks below verify the contents belong to this experiment.
    # if checks path.exists(); only a true condition runs the indented block.
    if path.exists():
        # = stores a result in data; call read using parentheses to supply its inputs.
        data = read(path)
        # if checks data['provenance'] != settings(budget, policy, seed); only a true condition runs the indented block.
        if data["provenance"] != settings(budget, policy, seed):
            # raise signals an error and stops this normal path, so invalid data are not silently accepted.
            raise ValueError("Cached prediction checks belong to another model")
        # A set keeps each density/scenario pair once, even if a row was duplicated.
        # = stores a result in cases; call set using parentheses to supply its inputs.
        cases = set()
        # for takes each item from data['episodes'] in turn; episode holds the current item.
        for episode in data["episodes"]:
            # Call cases.add using parentheses to supply its inputs; this line does not store a separate return value.
            cases.add((episode["density"], episode["seed"]))
        # = stores a result in expected; call set using parentheses to supply its inputs.
        expected = set()
        # Traffic density changes vehicle spacing. It does not add more network inputs.
        # for takes each item from DENSITIES in turn; density holds the current item.
        for density in DENSITIES:
            # A scenario seed fixes the starting traffic. Reuse the same IDs for fair policy comparisons.
            # for takes each item from range(19000, 19060) in turn; scenario holds the current item.
            for scenario in range(19000, 19060):
                # Call expected.add using parentheses to supply its inputs; this line does not store a separate return value.
                expected.add((density, scenario))
        # if checks len(data['episodes']) != 240 or cases != expected; only a true condition runs the indented block.
        if len(data["episodes"]) != 240 or cases != expected:
            # raise signals an error and stops this normal path, so invalid data are not silently accepted.
            raise ValueError("Missing or duplicate prediction cases")
    # else provides the alternative branch; after a loop it instead means the loop finished without break.
    else:
        # = stores a result in command; build a list, keeping its items in order.
        command = [
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            sys.executable,
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            __file__,
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "--collect",
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            str(budget),
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            policy,
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            str(seed),
        # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
        ]
        # with enters a temporary context and exits it afterward, even if an error interrupts the block.
        with path.with_suffix(".log").open("w") as log:
            # Run the requested Python command and stop on failure. This is evaluation, not a training call.
            # Call subprocess.run using parentheses to supply its inputs; this line does not store a separate return value.
            subprocess.run(
                # Inside this continued call, name=value supplies an option by name rather than by its position.
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            )
    # Call print using parentheses to supply its inputs; this line does not store a separate return value.
    print(f"Checked {budget} {policy} seed {seed}", flush=True)


# def defines main; parentheses list its inputs, and the colon starts its indented body.
def main():
    # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
    """Read the Terminal options and coordinate the workflow. The learning algorithm is not implemented here."""
    # = stores a result in parser; call argparse.ArgumentParser using parentheses to supply its inputs.
    parser = argparse.ArgumentParser(description=__doc__)
    # Call parser.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    parser.add_argument("--jobs", type=int, default=4)
    # Call parser.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    parser.add_argument("--collect", nargs=3, metavar=("BUDGET", "POLICY", "SEED"))
    # Call parser.add_argument using parentheses to supply its inputs; this line does not store a separate return value.
    parser.add_argument("--report-only", action="store_true")
    # = stores a result in args; call parser.parse_args using parentheses to supply its inputs.
    args = parser.parse_args()
    # Call output.mkdir using parentheses to supply its inputs; this line does not store a separate return value.
    OUTPUT.mkdir(exist_ok=True)
    # if checks args.collect; only a true condition runs the indented block.
    if args.collect:
        # = unpacks the right-hand result into (budget, policy, seed), matching values to names in order.
        budget, policy, seed = args.collect
        # if checks int(budget) not in [40000, 100000] or policy not in POLICIES or int(seed) not in range(3); only a true condition runs the indented block.
        if (
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            int(budget) not in [40000, 100000]
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            or policy not in POLICIES
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            or int(seed) not in range(3)
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        ):
            # Call parser.error using parentheses to supply its inputs; this line does not store a separate return value.
            parser.error("Unknown model or policy")
        # Call collect using parentheses to supply its inputs; this line does not store a separate return value.
        collect(int(budget), policy, int(seed))
        # return ends this function and sends its result back to the code that called it.
        return
    # if checks args.jobs < 1; only a true condition runs the indented block.
    if args.jobs < 1:
        # Call parser.error using parentheses to supply its inputs; this line does not store a separate return value.
        parser.error("--jobs must be positive")
    # = stores a result in design; build a dictionary, which stores values under named keys.
    design = {
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "budgets": [40000, 100000],
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "policies": POLICIES,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "development_seeds": [19000, 19059],
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "densities": DENSITIES,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "training_seeds": [0, 1, 2],
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "tuning": False,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "purpose": "Descriptive prediction diagnostics, not confirmation or a replacement primary test.",
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "return_bounds": "A terminal return is exact. Successful 30-step recordings retain bounded unknown continuation.",
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "interpretation": "Predictions concern the initial chosen action. Policies can choose different actions and induce different outcomes. QR-tail continuation differs from mean-greedy training.",
    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
    }
    # JSON turns tuples into lists. Normalize that detail before checking a recorded design.
    # = stores a result in design; call json.loads using parentheses to supply its inputs.
    design = json.loads(json.dumps(design))
    # = stores a result in design_path; apply / to these operands; numbers are divided, while Path objects join folder names.
    design_path = OUTPUT / "design.json"
    # if checks design_path.exists() and read(design_path) != design; only a true condition runs the indented block.
    if design_path.exists() and read(design_path) != design:
        # raise signals an error and stops this normal path, so invalid data are not silently accepted.
        raise ValueError("Refusing to silently change the diagnostic design")
    # if checks not design_path.exists(); only a true condition runs the indented block.
    if not design_path.exists():
        # Call save_json using parentheses to supply its inputs; this line does not store a separate return value.
        save_json(design_path, design)
    # if checks not args.report_only; only a true condition runs the indented block.
    if not args.report_only:
        # = stores a result in jobs; build a list, keeping its items in order.
        jobs = []
        # Each budget gets its own results. Keep 40K and 100K separate until the paired comparison.
        # for takes each item from [40000, 100000] in turn; budget holds the current item.
        for budget in [40000, 100000]:
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
            list(pool.map(run_collection, jobs))
    # from driving_rl.prediction_audit_report import brings the named tools into this file; parentheses allow several lines.
    from driving_rl.prediction_audit_report import report

    # Call report using parentheses to supply its inputs; this line does not store a separate return value.
    report()


# __name__ equals "__main__" when this file runs directly; this guard avoids running the script merely because another file imports it.
if __name__ == "__main__":
    # Call main using parentheses to supply its inputs; this line does not store a separate return value.
    main()
