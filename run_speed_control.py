"""Explore a predeclared ego-speed control grid on existing scalar checkpoints."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from driving_rl.analysis import analyze
from driving_rl.experiment import save_json
from driving_rl.controls_report import create_report


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
STAGE = RESULTS / "controls"


def evaluate(cap, seed):
    checkpoint = RESULTS / f"dqn_seed{seed}/final.pt"
    path = STAGE / f"eval_dqn_cap{cap}_seed{seed}.json"
    if path.exists():
        data = json.loads(path.read_text())
        metadata = data["provenance"]
        if (metadata["speed_governor_cap_m_s"] != cap or
                metadata["checkpoint_sha256"] != hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                or metadata["episodes_per_density"] != 30 or metadata["seed_start"] != 10000
                or metadata["alpha"] != 1 or metadata["evaluation_collision_penalty"] != -2
                or [row["density"] for row in data["summary"]] != [1., 1.5, 2.]):
            raise ValueError(f"Refusing to reuse mismatched evaluation {path}")
    else:
        gif = ["--gif", str(RESULTS / f"dqn_cap{cap}_seed{seed}.gif")] if seed == 0 else []
        with (STAGE / f"cap{cap}_seed{seed}.log").open("w") as file:
            subprocess.run([sys.executable, "-m", "driving_rl.cli", "evaluate", "--checkpoint",
                            str(checkpoint), "--speed-cap", str(cap), "--episodes", "30", "--out",
                            str(path), *gif], stdout=file, stderr=subprocess.STDOUT, check=True)
        data = json.loads(path.read_text())
    return path, data


def main():
    if Path.cwd() != ROOT:
        raise ValueError("Run from the project directory")
    candidates = []
    for cap in [22, 24, 26]:
        path, data = evaluate(cap, 0)
        candidates.append({"cap_m_s": cap, "speed_m_s": data["summary"][0]["mean_speed_m_s"],
                           "evaluation_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    target = json.loads((RESULTS / "eval_qrdqn_seed0_tail25.json").read_text())["summary"][0]["mean_speed_m_s"]
    selected = min(candidates, key=lambda row: (abs(row["speed_m_s"]-target), row["cap_m_s"]))
    choice = {"rule": "Closest normal-density seed-0 speed; ties prefer lower cap. Exploratory development control.",
              "target_speed_m_s": target, "candidates": candidates,
              "selected_cap_m_s": selected["cap_m_s"],
              "seed0_speed_difference_m_s": selected["speed_m_s"]-target,
              "confirmation_scenario_seeds_used": False}
    path = STAGE / "governor_selection.json"
    if path.exists() and json.loads(path.read_text()) != choice:
        raise ValueError("Refusing to change recorded governor selection")
    save_json(path, choice)
    print(json.dumps(choice, indent=2), flush=True)
    for seed in [1, 2]:
        evaluate(selected["cap_m_s"], seed)
    analyze(RESULTS)
    create_report(RESULTS)


if __name__ == "__main__":
    main()
