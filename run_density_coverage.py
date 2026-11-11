"""Run the density coverage experiment from Terminal.

This is an experiment coordinator, not the learning algorithm. It launches
separate Python commands, saves logs, and checks existing results before reuse.
Paths identify saved files; hashes are fingerprints that detect changed files.
Read main() for the order of operations. Learning itself lives in driving_rl/agents.py.

Original implementation notes:
Compare fixed-density and mixed-density QR-DQN training at equal budgets.
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
# = stores a result in STAGE; apply / to these operands; numbers are divided, while Path objects join folder names.
STAGE = ROOT / "results/density_coverage"
# = stores a result in POLICIES; build a dictionary, which stores values under named keys.
POLICIES = {"mean": (1.0, None), "tail25": (0.25, None), "governed": (1.0, 25.0)}
# = stores a result in DESIGN; build a dictionary, which stores values under named keys.
DESIGN = {
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "training_steps": 40000,
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "training_seeds": [0, 1, 2],
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "training_densities": [1.0, 1.5, 2.0],
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "density_sampling": "Uniform per episode, independent RNG; equal episodes do not imply equal transitions.",
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "comparison": "Existing fixed-density 40k QR-DQN, same architecture/reward/optimizer/exploration.",
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "epsilon_decay_steps": 12000,
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "primary_evaluation": {
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "densities": [1.0, 1.5, 2.0],
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "seed_start": 10000,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "episodes": 30,
    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
    },
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "transfer_evaluation": {
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "densities": [1.25, 2.5],
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "seed_start": 16000,
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "episodes": 30,
    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
    },
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "calibration": {"densities": [1.0, 1.5, 2.0], "seed_start": 12000, "episodes": 60},
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "decision_rules": {"mean": 1.0, "tail25": 0.25, "governor_cap_m_s": 25.0},
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "primary_contrast": "Mixed minus fixed cautious collision fraction at density 2; also report speed/distance and all seeds.",
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "limitations": "Development comparisons, three matched seed IDs; no tuned selector, safety claim or global CVaR guarantee. Density 1.5/2 are within mixture support; 2.5 is beyond both training supports. Test mixture changes hidden episode dynamics.",
    # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
    "confirmation_scenarios_used": False,
# } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
}


# def defines read; parentheses list its inputs, and the colon starts its indented body.
def read(p):
    # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
    """Read saved JSON data into ordinary Python dictionaries and lists."""
    # return ends this function and sends its result back to the code that called it.
    return json.loads(p.read_text())


# def defines run; parentheses list its inputs, and the colon starts its indented body.
def run(cmd, log):
    """Run a command in a separate Python process and wait for it to finish.
    A nonzero exit status means the command failed; do not silently continue.
    """
    # with enters a temporary context and exits it afterward, even if an error interrupts the block.
    with log.open("w") as f:
        # Start another Python command and check whether it succeeded.
        # Call subprocess.run using parentheses to supply its inputs; this line does not store a separate return value.
        subprocess.run(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            [sys.executable, "-m", *cmd],
            # Inside this continued call, name=value supplies an option by name rather than by its position.
            cwd=ROOT,
            # Inside this continued call, name=value supplies an option by name rather than by its position.
            stdout=f,
            # Inside this continued call, name=value supplies an option by name rather than by its position.
            stderr=subprocess.STDOUT,
            # Inside this continued call, name=value supplies an option by name rather than by its position.
            check=True,
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )


# def defines checkpoint; parentheses list its inputs, and the colon starts its indented body.
def checkpoint(regime, seed):
    # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
    """Find the saved model file for the requested algorithm and training seed."""
    # return ends this function and sends its result back to the code that called it.
    return (
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        STAGE / f"qrdqn_seed{seed}/final.pt"
        # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
        if regime == "mixed"
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        else ROOT / f"results/training_budget/qrdqn_seed{seed}/checkpoint_40000.pt"
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )


# def defines train; parentheses list its inputs, and the colon starts its indented body.
def train(seed):
    """Train one missing experiment, or verify its completed saved model.
    Keep model settings fixed across the requested seeds. An incomplete directory
    is not silently resumed because replay memory and RNG state are not saved.
    """
    # = stores a result in folder; apply / to these operands; numbers are divided, while Path objects join folder names.
    folder = STAGE / f"qrdqn_seed{seed}"
    # if checks not (folder / 'summary.json').exists(); only a true condition runs the indented block.
    if not (folder / "summary.json").exists():
        # Call run using parentheses to supply its inputs; this line does not store a separate return value.
        run(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            [
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "driving_rl.cli",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "train",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "--algorithm",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "qrdqn",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "--seed",
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                str(seed),
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "--steps",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "40000",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "--exploration-steps",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "12000",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "--training-densities",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "1",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "1.5",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "2",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "--out",
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                str(folder),
            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
            ],
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            STAGE / f"train_seed{seed}.log",
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
    # = stores a result in c; call read using parentheses to supply its inputs.
    c = read(folder / "config.json")
    # = stores a result in base; call read using parentheses to supply its inputs.
    base = read(ROOT / f"results/training_budget/qrdqn_seed{seed}/config.json")
    # for takes each item from ['seed', 'steps', 'algorithm', 'gamma', 'learning_rate', 'collision_penalty', 'buffer_size', 'bat... in turn; k holds the current item.
    for k in [
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "seed",
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "steps",
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "algorithm",
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "gamma",
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "learning_rate",
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "collision_penalty",
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "buffer_size",
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "batch_size",
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "learning_starts",
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "train_frequency",
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "target_interval",
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "final_epsilon",
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "quantiles",
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "exploration_steps",
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "observation_dim",
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        "n_actions",
    # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
    ]:
        # if checks c[k] != base[k]; only a true condition runs the indented block.
        if c[k] != base[k]:
            # raise signals an error and stops this normal path, so invalid data are not silently accepted.
            raise ValueError(f"Mismatched {k}: {folder}")
    # if checks c['training_densities'] != [1.0, 1.5, 2.0] or not read(folder / 'summary.json')['training_complete']; only a true condition runs the indented block.
    if (
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        c["training_densities"] != [1.0, 1.5, 2.0]
        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
        or not read(folder / "summary.json")["training_complete"]
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    ):
        # raise signals an error and stops this normal path, so invalid data are not silently accepted.
        raise ValueError("Incomplete/mismatched mixture run")
    # Call print using parentheses to supply its inputs; this line does not store a separate return value.
    print(f"Training complete: seed {seed}", flush=True)


# def defines evaluate; parentheses list its inputs, and the colon starts its indented body.
def evaluate(spec):
    """Run or verify one frozen-policy evaluation.
    Check its model fingerprint, settings and scenario block before reusing saved
    data. This function must not quietly mix results from different experiments.
    """
    # = unpacks the right-hand result into (regime, seed, policy, block), matching values to names in order.
    regime, seed, policy, block = spec
    # = unpacks the right-hand result into (alpha, cap), matching values to names in order.
    alpha, cap = POLICIES[policy]
    # = stores a result in d; use square brackets to select part of DESIGN.
    d = DESIGN["primary_evaluation" if block == "main" else "transfer_evaluation"]
    # = stores a result in ck; call checkpoint using parentheses to supply its inputs.
    ck = checkpoint(regime, seed)
    # = stores a result in out; choose the value before "if" when its condition is true, otherwise the value after "else".
    out = (
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        ROOT / f"results/training_budget/eval_qrdqn_{seed}_40000_{policy}.json"
        # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
        if regime == "fixed" and block == "main"
        # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
        else STAGE / f"eval_{regime}_{policy}_seed{seed}_{block}.json"
    # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
    )
    # A saved file is useful only if its model and settings match this request.
    # if checks out.exists(); only a true condition runs the indented block.
    if out.exists():
        # = stores a result in data; call read using parentheses to supply its inputs.
        data = read(out)
        # = stores a result in p; use square brackets to select part of data.
        p = data["provenance"]
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
            [r["density"] for r in data["summary"]],
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            len(data["episodes"]),
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        ) != (
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            hashlib.sha256(ck.read_bytes()).hexdigest(),
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            alpha,
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            cap,
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            d["seed_start"],
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            d["episodes"],
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            d["densities"],
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            len(d["densities"]) * d["episodes"],
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        ):
            # raise signals an error and stops this normal path, so invalid data are not silently accepted.
            raise ValueError(f"Mismatched evaluation: {out}")
    # else provides the alternative branch; after a loop it instead means the loop finished without break.
    else:
        # = stores a result in cmd; build a list, keeping its items in order.
        cmd = [
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
            "--seed-start",
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            str(d["seed_start"]),
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "--episodes",
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            str(d["episodes"]),
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "--densities",
            # Continue the expression above with this operator; the surrounding parentheses allow the calculation to span lines.
            *[str(x) for x in d["densities"]],
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            "--out",
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            str(out),
        # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
        ]
        # if checks cap; only a true condition runs the indented block.
        if cap:
            # The combined operator updates cmd using its previous value.
            cmd += ["--speed-cap", str(cap)]
        # Call run using parentheses to supply its inputs; this line does not store a separate return value.
        run(cmd, out.with_suffix(".log"))
    # Call print using parentheses to supply its inputs; this line does not store a separate return value.
    print(f"Evaluated {regime} {seed} {policy} {block}", flush=True)
    # return ends this function and sends its result back to the code that called it.
    return out


# def defines calibrate; parentheses list its inputs, and the colon starts its indented body.
def calibrate(spec):
    # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
    """Run prediction checks for the specified saved model, or validate saved checks."""
    # = unpacks the right-hand result into (seed, policy), matching values to names in order.
    seed, policy = spec
    # = unpacks the right-hand result into (alpha, _), matching values to names in order.
    alpha, _ = POLICIES[policy]
    # = stores a result in ck; call checkpoint using parentheses to supply its inputs.
    ck = checkpoint("mixed", seed)
    # = stores a result in out; apply / to these operands; numbers are divided, while Path objects join folder names.
    out = STAGE / f"calibration_{policy}_seed{seed}.json"
    # A saved file is useful only if its model and settings match this request.
    # if checks out.exists(); only a true condition runs the indented block.
    if out.exists():
        # = stores a result in data; call read using parentheses to supply its inputs.
        data = read(out)
        # = stores a result in p; use square brackets to select part of data.
        p = data["provenance"]
        # if checks (p['checkpoint_sha256'], p['alpha'], p['seed_start'], p['episodes_per_density'], p['speed_cap'], ...; only a true condition runs the indented block.
        if (
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            p["checkpoint_sha256"],
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            p["alpha"],
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            p["seed_start"],
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            p["episodes_per_density"],
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            p["speed_cap"],
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            len(data["episodes"]),
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        ) != (hashlib.sha256(ck.read_bytes()).hexdigest(), alpha, 12000, 60, None, 180):
            # raise signals an error and stops this normal path, so invalid data are not silently accepted.
            raise ValueError(f"Mismatched calibration: {out}")
    # else provides the alternative branch; after a loop it instead means the loop finished without break.
    else:
        # Call run using parentheses to supply its inputs; this line does not store a separate return value.
        run(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            [
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "driving_rl.calibration",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "--checkpoint",
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                str(ck),
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "--alpha",
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                str(alpha),
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "--out",
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                str(out),
            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
            ],
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            out.with_suffix(".log"),
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
    # Call print using parentheses to supply its inputs; this line does not store a separate return value.
    print(f"Calibrated mixed {seed} {policy}", flush=True)


# def defines replay; parentheses list its inputs, and the colon starts its indented body.
def replay(regime):
    # This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
    """Record the scenario chosen in advance, not a scenario selected because it looks good."""
    # = stores a result in ck; call checkpoint using parentheses to supply its inputs.
    ck = checkpoint(regime, 0)
    # = stores a result in out; apply / to these operands; numbers are divided, while Path objects join folder names.
    out = STAGE / f"replay_{regime}.json"
    # A saved file is useful only if its model and settings match this request.
    # if checks out.exists(); only a true condition runs the indented block.
    if out.exists():
        # = stores a result in data; call read using parentheses to supply its inputs.
        data = read(out)
        # = stores a result in p; use square brackets to select part of data.
        p = data["provenance"]
        # if checks (p['checkpoint_sha256'], p['alpha'], p['seed_start'], p['episodes_per_density']) != (hashlib.sha2...; only a true condition runs the indented block.
        if (
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            p["checkpoint_sha256"],
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            p["alpha"],
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            p["seed_start"],
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            p["episodes_per_density"],
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        ) != (hashlib.sha256(ck.read_bytes()).hexdigest(), 0.25, 10000, 1):
            # raise signals an error and stops this normal path, so invalid data are not silently accepted.
            raise ValueError("Replay metadata mismatch")
    # else provides the alternative branch; after a loop it instead means the loop finished without break.
    else:
        # Call run using parentheses to supply its inputs; this line does not store a separate return value.
        run(
            # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
            [
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
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                ".25",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "--episodes",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "1",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "--seed-start",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "10000",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "--densities",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "2",
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "--out",
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                str(out),
                # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                "--gif",
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                str(STAGE / f"replay_{regime}.gif"),
            # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
            ],
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            out.with_suffix(".log"),
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        )
    # Call print using parentheses to supply its inputs; this line does not store a separate return value.
    print(f"Recorded {regime} cautious replay", flush=True)


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
    p.add_argument("--stage", choices=["all", "evaluate", "report"], default="all")
    # = stores a result in a; call p.parse_args using parentheses to supply its inputs.
    a = p.parse_args()
    # if checks a.jobs < 1; only a true condition runs the indented block.
    if a.jobs < 1:
        # Call p.error using parentheses to supply its inputs; this line does not store a separate return value.
        p.error("--jobs must be positive")
    # Call stage.mkdir using parentheses to supply its inputs; this line does not store a separate return value.
    STAGE.mkdir(parents=True, exist_ok=True)
    # = stores a result in design; apply / to these operands; numbers are divided, while Path objects join folder names.
    design = STAGE / "design.json"
    # if checks design.exists() and read(design) != DESIGN; only a true condition runs the indented block.
    if design.exists() and read(design) != DESIGN:
        # raise signals an error and stops this normal path, so invalid data are not silently accepted.
        raise ValueError("Refusing to change recorded design")
    # if checks not design.exists(); only a true condition runs the indented block.
    if not design.exists():
        # Call save_json using parentheses to supply its inputs; this line does not store a separate return value.
        save_json(design, DESIGN)
    # if checks a.stage != 'report'; only a true condition runs the indented block.
    if a.stage != "report":
        # Run independent jobs together. Each writes its own files; this is not a shared training model.
        # with enters a temporary context and exits it afterward, even if an error interrupts the block.
        with ThreadPoolExecutor(max_workers=a.jobs) as pool:
            # if checks a.stage == 'all'; only a true condition runs the indented block.
            if a.stage == "all":
                # Call list using parentheses to supply its inputs; this line does not store a separate return value.
                list(pool.map(train, range(3)))
            # Call list using parentheses to supply its inputs; this line does not store a separate return value.
            list(
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                pool.map(
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    evaluate,
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    [
                        # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                        (regime, s, policy, block)
                        # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                        for regime in ["fixed", "mixed"]
                        # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                        for s in range(3)
                        # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                        for policy in POLICIES
                        # This continued comprehension states which items to process or keep; it belongs to the expression opened above.
                        for block in ["main", "transfer"]
                    # ] closes the list or indexing expression opened above; any following comma separates this item from the next.
                    ],
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                )
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            )
            # Call list using parentheses to supply its inputs; this line does not store a separate return value.
            list(
                # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                pool.map(
                    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
                    calibrate,
                    # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                    [(s, policy) for s in range(3) for policy in ["mean", "tail25"]],
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                )
            # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
            )
            # Call list using parentheses to supply its inputs; this line does not store a separate return value.
            list(pool.map(replay, ["fixed", "mixed"]))
    # from driving_rl.density_coverage_report import brings the named tools into this file; parentheses allow several lines.
    from driving_rl.density_coverage_report import report

    # Call report using parentheses to supply its inputs; this line does not store a separate return value.
    report()


# Run the entry point only when this file is executed, not when it is imported.
# __name__ equals "__main__" when this file runs directly; this guard avoids running the script merely because another file imports it.
if __name__ == "__main__":
    # Call main using parentheses to supply its inputs; this line does not store a separate return value.
    main()
