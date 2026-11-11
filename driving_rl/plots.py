"""Static figures and a portable HTML experiment viewer."""
import csv
import html
import json
from pathlib import Path

import numpy as np

from .metrics import wilson_interval


def report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": "#f6f8fb", "axes.facecolor": "white",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 11, "axes.titlesize": 13})
    root = Path(args.results)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    evaluations = []
    for path in sorted(root.glob("eval_*.json")):
        data = json.loads(path.read_text())
        if "provenance" in data:
            evaluations.append((path.stem.removeprefix("eval_"), data))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    for path in sorted(root.glob("*/training.csv")):
        with path.open() as file:
            rows = list(csv.DictReader(file))
        if not rows:
            continue
        steps = np.array([float(row["step"]) for row in rows])
        returns = np.array([float(row["return"]) for row in rows])
        crashes = np.array([float(row["crashed"]) for row in rows])
        window = min(50, len(rows))
        kernel = np.ones(window) / window
        axes[0].plot(steps[window-1:], np.convolve(returns, kernel, "valid"), label=path.parent.name)
        axes[1].plot(steps[window-1:], np.convolve(crashes, kernel, "valid"), label=path.parent.name)
    for ax in axes:
        ax.set_xlabel("Environment interactions")
        ax.grid(alpha=0.18)
        if ax.lines:
            ax.legend(fontsize=8)
    axes[0].set(title="Training return · rolling 50 episodes", ylabel="Undiscounted return")
    axes[1].set(title="Exploratory training collisions", ylabel="Collision fraction", ylim=(0, 1))
    fig.savefig(output / "learning_curves.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    for label, data in evaluations:
        summary = data["summary"]
        density = [row["density"] for row in summary]
        rates = np.array([row["crash_rate"] for row in summary])
        intervals = np.array([wilson_interval(round(row["crash_rate"] * row["episodes"]), row["episodes"]) for row in summary])
        axes[0].errorbar(density, rates,
                         yerr=np.maximum(0, np.stack([rates - intervals[:, 0], intervals[:, 1] - rates])),
                         fmt="o-", capsize=3, label=label)
        axes[1].plot(density, [row["mean_speed_m_s"] for row in summary], "o-", label=label)
        axes[2].plot([row["mean_speed_m_s"] for row in summary],
                     [row["crash_rate"] for row in summary], "o-", label=label)
    axes[0].set(title="Held-out collision rate", xlabel="Traffic density multiplier", ylabel="Collision fraction", ylim=(-0.03, 1.03))
    axes[1].set(title="Held-out driving speed", xlabel="Traffic density multiplier", ylabel="Mean speed (m/s)")
    axes[2].set(title="Speed–collision tradeoff", xlabel="Mean speed (m/s)", ylabel="Collision fraction", ylim=(-0.03, 1.03))
    for ax in axes:
        ax.grid(alpha=0.18)
        if ax.lines:
            ax.legend(fontsize=7)
    fig.savefig(output / "evaluation.png", dpi=160)
    plt.close(fig)
    table_rows = []
    for label, data in evaluations:
        for row in data["summary"]:
            values = [html.escape(label), row["density"], row["episodes"],
                      f'{100 * row["crash_rate"]:.1f}%', f'{row["mean_speed_m_s"]:.1f}',
                      f'{row["mean_return"]:.2f}', f'{row["worst_10_percent_mean_return"]:.2f}',
                      f'{row["mean_distance_m"]:.0f}']
            table_rows.append("<tr>" + "".join(f"<td>{v}</td>" for v in values) + "</tr>")
    gifs = sorted(root.glob("*.gif"))
    video_cards = "".join(f'<article><h3>{html.escape(path.stem)}</h3><img src="../{html.escape(path.name)}" alt="Driving rollout"></article>' for path in gifs)
    document = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Risk-aware driving · experiment lab</title><style>
body{margin:0;background:#0d1423;color:#e9eef7;font:16px/1.6 system-ui,sans-serif}
main{max-width:1180px;margin:auto;padding:52px 28px}h1{font-size:44px;line-height:1.15;max-width:850px}
.eyebrow{color:#62dbbf;letter-spacing:.12em;font-size:13px;font-weight:700}.intro{color:#acbad0;max-width:850px}
section{margin-top:42px}.figure{width:100%;background:white;border-radius:12px}table{border-collapse:collapse;width:100%;font-size:14px}
td,th{text-align:left;padding:12px;border-bottom:1px solid #283348}th{color:#8ddfcd}.scroll{overflow:auto}
.notice{padding:18px 22px;background:#172439;border-left:3px solid #62dbbf;border-radius:6px}
article{padding:12px 0}article img{width:100%;border-radius:10px}h3{font-size:16px}footer{color:#acbad0;margin-top:40px}
</style><main><div class="eyebrow">DEEP REINFORCEMENT LEARNING / initial experiment LAB</div>
<h1>How much speed should an agent trade for fewer crashes?</h1>
<p class="intro">A controlled comparison of DQN, distributional QR-DQN, and lower-tail action selection in simulated highway traffic.</p>
<p class="notice"><strong>Preliminary experiments.</strong> Individual training seeds and finite evaluation samples do not establish a reliable improvement. Lower-tail selection is a deployment heuristic; its return predictions were trained for mean-greedy control. The heuristic comparator uses privileged simulator state. All policies operate in a simplified simulator.</p>
<section><h2>Held-out evaluation</h2><img class="figure" src="evaluation.png" alt="Collision rate and speed by traffic density">
<div class="scroll"><table><thead><tr><th>Policy</th><th>Density</th><th>Episodes</th><th>Crashes</th><th>Speed m/s</th><th>Return</th><th>Worst 10%</th><th>Distance m</th></tr></thead><tbody>'''
    document += "".join(table_rows) + "</tbody></table></div></section>"
    document += '<section><h2>Learning dynamics</h2><p>These training curves include exploratory actions; assess deployed policies using the held-out evaluation above.</p><img class="figure" src="learning_curves.png" alt="Training curves"></section>'
    document += '<section><h2>Rollout replays</h2><p>Each replay uses the first evaluation seed, selected before seeing outcomes.</p>' + video_cards + '</section>'
    document += '<footer>Collision error bars are 95% Wilson intervals across evaluation episodes, conditional on the evaluated policy. They do not measure training-seed uncertainty. Units: speed in meters/second; distance is traveled path length. Worst 10% is the empirical lower-tail mean of undiscounted episode returns. Episode averages give each episode equal weight. Raw evaluation JSON includes per-episode results, seeds, checkpoint hashes, and package versions.</footer></main></html>'
    (output / "index.html").write_text(document)
    print(f"Wrote {output / 'index.html'}")
