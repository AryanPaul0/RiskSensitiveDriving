"""Reproduce same-checkpoint controls, risk sweep and calibration diagnostics."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from driving_rl.experiment import save_json


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
STAGE = RESULTS / "risk_analysis"
CAPS = [24., 24.5, 25., 25.5, 26., 26.5]
ALPHAS = [.125, .25, .5, .75, 1.]


def token(value):
    return f"{value:g}".replace(".", "p")


def eval_path(seed, alpha=1., cap=None):
    if cap is None and alpha in [.25, 1.]:
        suffix = "tail25" if alpha == .25 else "mean"
        return RESULTS / f"eval_qrdqn_seed{seed}_{suffix}.json"
    suffix = f"cap{token(cap)}" if cap is not None else f"alpha{token(alpha)}"
    return STAGE / f"eval_qrdqn_{suffix}_seed{seed}.json"


def evaluate(spec):
    seed, alpha, cap = spec
    checkpoint = RESULTS / f"qrdqn_seed{seed}/final.pt"
    output = eval_path(seed, alpha, cap)
    if output.exists():
        data = json.loads(output.read_text())
        p = data["provenance"]
        if (p["checkpoint_sha256"] != hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                or p["training_seed"] != seed or p["training_steps"] != 20000
                or p["alpha"] != alpha or p.get("speed_governor_cap_m_s") != cap
                or p["seed_start"] != 10000 or p["episodes_per_density"] != 30
                or p["algorithm"] != "qrdqn" or p.get("evaluation_collision_penalty", -2.) != -2.
                or [row["density"] for row in data["summary"]] != [1., 1.5, 2.]):
            raise ValueError(f"Refusing to reuse mismatched evaluation: {output}")
        return output
    arguments = [sys.executable, "-m", "driving_rl.cli", "evaluate", "--checkpoint",
                 str(checkpoint), "--alpha", str(alpha), "--episodes", "30", "--out", str(output)]
    if cap is not None:
        arguments += ["--speed-cap", str(cap)]
    with output.with_suffix(".log").open("w") as log:
        subprocess.run(arguments, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(f"Evaluated {output.name}", flush=True)
    return output


def controls(jobs):
    # The threshold is chosen before looking at seeds 1 and 2 under this control.
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        paths = list(pool.map(evaluate, [(0, 1., cap) for cap in CAPS]))
    target_path = RESULTS / "eval_qrdqn_seed0_tail25.json"
    target = json.loads(target_path.read_text())["summary"][0]["mean_speed_m_s"]
    candidates = []
    for cap, path in zip(CAPS, paths):
        speed = json.loads(path.read_text())["summary"][0]["mean_speed_m_s"]
        candidates.append({"cap_m_s": cap, "speed_m_s": speed,
                           "evaluation_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    chosen = min(candidates, key=lambda row: (abs(row["speed_m_s"]-target), row["cap_m_s"]))
    selection = {"rule": "Same QR-DQN checkpoint, seed-0 normal-density speed closeness only; ties prefer lower cap.",
                 "target_speed_m_s": target, "candidates": candidates,
                 "selected_cap_m_s": chosen["cap_m_s"],
                 "seed0_speed_difference_m_s": chosen["speed_m_s"]-target,
                 "target_evaluation_sha256": hashlib.sha256(target_path.read_bytes()).hexdigest(),
                 "confirmation_seeds_used": False}
    path = STAGE / "governor_selection.json"
    if path.exists() and json.loads(path.read_text()) != selection:
        raise ValueError("Refusing to change the recorded same-checkpoint control selection")
    save_json(path, selection)
    print(f"Frozen same-checkpoint threshold {chosen['cap_m_s']} m/s", flush=True)
    tasks = [(seed, 1., chosen["cap_m_s"]) for seed in [1, 2]]
    tasks += [(seed, alpha, None) for seed in [0, 1, 2] for alpha in ALPHAS]
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        list(pool.map(evaluate, tasks))


def calibration_task(spec):
    seed, policy, cap = spec
    alpha = .25 if policy == "tail25" else 1.
    checkpoint = RESULTS / f"qrdqn_seed{seed}/final.pt"
    path = STAGE / f"calibration_{policy}_seed{seed}.json"
    if path.exists():
        data = json.loads(path.read_text())["provenance"]
        if (data["checkpoint_sha256"] != hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                or data["alpha"] != alpha or data["speed_cap"] != cap
                or data["seed_start"] != 12000 or data["episodes_per_density"] != 60):
            raise ValueError(f"Refusing to reuse mismatched calibration: {path}")
        return
    arguments = [sys.executable, "-m", "driving_rl.calibration", "--checkpoint", str(checkpoint),
                 "--alpha", str(alpha), "--out", str(path)]
    if cap is not None:
        arguments += ["--speed-cap", str(cap)]
    with path.with_suffix(".log").open("w") as log:
        subprocess.run(arguments, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(f"Calibrated {policy}, training seed {seed}", flush=True)


def ensure_replay(cap):
    checkpoint = RESULTS / "qrdqn_seed0/final.pt"
    path = STAGE / "replay_governed.json"
    if path.exists():
        p = json.loads(path.read_text())["provenance"]
        if (p["checkpoint_sha256"] != hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                or p["alpha"] != 1. or p["speed_governor_cap_m_s"] != cap
                or p["seed_start"] != 10000 or p["episodes_per_density"] != 1):
            raise ValueError("Existing replay does not match the selected control")
        if not (STAGE / "replay_governed.gif").exists():
            raise ValueError("Replay JSON exists without its GIF; preserve it and regenerate in a new path")
        return
    with (STAGE / "replay_governed.log").open("w") as log:
        subprocess.run([sys.executable, "-m", "driving_rl.cli", "evaluate", "--checkpoint", str(checkpoint),
                        "--speed-cap", str(cap), "--episodes", "1", "--densities", "1",
                        "--out", str(path), "--gif", str(STAGE / "replay_governed.gif")],
                       stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--stage", choices=["controls", "calibration", "all"], default="all")
    args = parser.parse_args()
    if Path.cwd() != ROOT or not 1 <= args.jobs <= 4:
        parser.error("Run from the project directory with --jobs between 1 and 4")
    STAGE.mkdir(parents=True, exist_ok=True)
    if args.stage in ["controls", "all"]:
        controls(args.jobs)
    if args.stage in ["calibration", "all"]:
        cap = json.loads((STAGE / "governor_selection.json").read_text())["selected_cap_m_s"]
        tasks = [(seed, policy, cap if policy == "governed" else None)
                 for seed in [0, 1, 2] for policy in ["mean", "tail25", "governed"]]
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            list(pool.map(calibration_task, tasks))
        ensure_replay(cap)
        from driving_rl.risk_analysis_analysis import analyze
        from driving_rl.risk_analysis_report import create_report
        analyze(RESULTS)
        create_report(RESULTS)


if __name__ == "__main__":
    main()
