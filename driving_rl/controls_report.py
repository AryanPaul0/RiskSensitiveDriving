"""Create the controls results page.

Read saved numbers, calculate clearly labeled summaries, draw figures, and write
an HTML page. This file presents evidence; it does not train or select models.
Long HTML strings are page templates. The reinforcement-learning calculations
live in agents.py and experiment.py, not inside the page layout.

Original implementation notes:
Publication-style figures plus a portable viewer for the control experiments.
"""

import html
import json
from pathlib import Path

import numpy as np

from .diagnostics import reward_example

COLORS = {
    "DQN": "#4889c9",
    "QR-DQN mean": "#9563c2",
    "QR-DQN lower 25%": "#159c87",
    "Reward-tuned DQN": "#df9144",
    "DQN + speed governor": "#d7677e",
}


def create_report(root):
    """Build graphs and an HTML page from saved result files.
    The arrays below hold measured results; plotting does not change the models.
    HTML strings describe page layout, while Python inserts the saved numbers.
    Keep primary comparisons separate from exploratory comparisons in the report.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(root)
    data = json.loads((root / "controls/summary.json").read_text())
    groups = data["groups"]
    output = root / "report"
    output.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "figure.facecolor": "#f5f7fb",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
            "axes.titlesize": 13,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for label, rows in groups.items():
        densities = [row["density"] for row in rows]
        for axis, metric, multiplier in [
            (axes[0], "crash_rate", 100),
            (axes[1], "mean_speed_m_s", 1),
        ]:
            means = [row[metric] * multiplier for row in rows]
            lows = [row[metric + "_seed_range"][0] * multiplier for row in rows]
            highs = [row[metric + "_seed_range"][1] * multiplier for row in rows]
            axis.plot(densities, means, "o-", color=COLORS[label], label=label)
            axis.fill_between(densities, lows, highs, color=COLORS[label], alpha=0.10)
            for row in rows:
                axis.scatter(
                    [row["density"]] * len(row["per_seed"]),
                    [item[metric] * multiplier for item in row["per_seed"]],
                    s=20,
                    alpha=0.5,
                    color=COLORS[label],
                )
    axes[0].set(
        title="Collision rate across traffic densities",
        ylabel="Episodes ending in collision (%)",
        ylim=(-3, 103),
    )
    axes[1].set(
        title="Speed across traffic densities", ylabel="Episode-averaged speed (m/s)"
    )
    for axis in axes:
        axis.set_xlabel("Traffic density multiplier")
        axis.grid(alpha=0.15)
        axis.legend(fontsize=8)
    # Save a figure for the local results page; this does not affect agent behavior.
    fig.savefig(output / "controls_replication.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
    for axis, density in zip(axes, [1.0, 1.5, 2.0]):
        for label, rows in groups.items():
            row = next(item for item in rows if item["density"] == density)
            axis.scatter(
                [item["mean_speed_m_s"] for item in row["per_seed"]],
                [item["crash_rate"] * 100 for item in row["per_seed"]],
                color=COLORS[label],
                s=32,
                alpha=0.5,
            )
            axis.scatter(
                row["mean_speed_m_s"],
                row["crash_rate"] * 100,
                color=COLORS[label],
                s=100,
                marker="D",
                label=label,
                edgecolors="white",
                linewidths=1,
            )
        axis.set(
            title=f"Density {density:.1f}", xlabel="Mean speed (m/s)", ylim=(-3, 103)
        )
        axis.grid(alpha=0.15)
    axes[0].set_ylabel("Collision rate (%)")
    axes[-1].legend(fontsize=7, loc="lower right")
    # Save a figure for the local results page; this does not affect agent behavior.
    fig.savefig(output / "controls_tradeoff.png", dpi=180)
    plt.close(fig)
    contrasts = sorted(
        {(row["left"], row["right"]) for row in data["paired_comparisons"]}
    )
    fig, axis = plt.subplots(figsize=(10, 4.2), constrained_layout=True)
    for index, (left, right) in enumerate(contrasts):
        selected = [
            row
            for row in data["paired_comparisons"]
            if row["left"] == left and row["right"] == right
        ]
        x = (
            np.array([row["density"] for row in selected])
            + (index - (len(contrasts) - 1) / 2) * 0.025
        )
        y = np.array([row["crashed"]["mean_difference"] * 100 for row in selected])
        intervals = (
            np.array(
                [row["crashed"]["exploratory_95_percent_interval"] for row in selected]
            )
            * 100
        )
        axis.errorbar(
            x,
            y,
            yerr=np.maximum(0, np.stack([y - intervals[:, 0], intervals[:, 1] - y])),
            fmt="o-",
            capsize=5,
            label=f"Lower 25% minus {right}",
        )
    axis.axhline(0, color="#707d8f", linestyle="--", linewidth=1)
    axis.set(
        title="Paired collision-rate differences · exploratory two-way bootstrap",
        xlabel="Traffic density multiplier",
        ylabel="Collision-rate difference (percentage points)",
        xticks=[1.0, 1.5, 2.0],
    )
    axis.grid(alpha=0.15)
    axis.legend(fontsize=9)
    # Save a figure for the local results page; this does not affect agent behavior.
    fig.savefig(output / "controls_paired.png", dpi=180)
    plt.close(fig)
    diagnostics = []
    for name, label in [
        ("dqn", "DQN"),
        ("qrdqn_mean", "QR-DQN mean"),
        ("qrdqn_tail25", "QR-DQN lower 25%"),
    ]:
        path = root / "controls" / f"diagnostic_{name}.json"
        # Check existing data instead of silently replacing scientific results.
        if path.exists():
            diagnostics.append((label, json.loads(path.read_text())))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    action_colors = ["#4889c9", "#98a2b3", "#9563c2", "#df9144", "#159c87"]
    starts = np.zeros(len(diagnostics))
    for action, color in enumerate(action_colors):
        values = [
            item["summary"][0]["action_fractions"][action] * 100
            for _, item in diagnostics
        ]
        axes[0].barh(
            [label for label, _ in diagnostics],
            values,
            left=starts,
            color=color,
            label=["Left", "Idle", "Right", "Faster", "Slower"][action],
        )
        starts += values
    axes[0].set(
        title="Decision mix · separate normal-traffic diagnostic block",
        xlabel="Actions (%)",
        xlim=(0, 100),
    )
    axes[0].legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    example = reward_example()
    axes[1].bar(
        ["Fast, then crash", "Slower, survive"],
        [
            example["fast_crash_discounted_return"],
            example["slow_survival_discounted_return"],
        ],
        color=["#df9144", "#159c87"],
    )
    axes[1].set(
        title="The reward can favor a crash · illustrative paths",
        ylabel="Discounted return (gamma = 0.9)",
    )
    # Save a figure for the local results page; this does not affect agent behavior.
    fig.savefig(output / "controls_diagnostics.png", dpi=180)
    plt.close(fig)
    selection = json.loads((root / "controls/baseline_selection.json").read_text())
    nominal = {label: rows[0] for label, rows in groups.items()}
    mean = nominal["QR-DQN mean"]
    cautious = nominal["QR-DQN lower 25%"]
    reduction = (mean["crash_rate"] - cautious["crash_rate"]) * 100
    slow = (1 - cautious["mean_speed_m_s"] / mean["mean_speed_m_s"]) * 100
    per_seed_rows = []
    for label, rows in groups.items():
        for row in rows:
            for item in row["per_seed"]:
                per_seed_rows.append(
                    f'<tr><td>{html.escape(label)}</td><td>{row["density"]:.1f}</td><td>{item["training_seed"]}</td><td>{item["crash_rate"]*100:.1f}%</td><td>{item["mean_speed_m_s"]:.2f}</td><td>{item["mean_distance_m"]:.0f}</td><td>{item["mean_duration_s"]:.1f}</td></tr>'
                )
    replay_names = [
        ("Mean-return QR-DQN", "qrdqn_seed0_mean.gif"),
        ("Cautious QR-DQN", "qrdqn_seed0_tail25.gif"),
        (
            "Reward-tuned DQN",
            f'dqn_penalty{abs(int(selection["selected_collision_penalty"]))}_seed0.gif',
        ),
    ]
    if selection["selected_collision_penalty"] == -2:
        replay_names[-1] = ("Reward-tuned DQN (same as standard)", "dqn_seed0_mean.gif")
    governor_text = ""
    governor_path = root / "controls/governor_selection.json"
    if governor_path.exists():
        governor = json.loads(governor_path.read_text())
        cap = governor["selected_cap_m_s"]
        replay_names.append(("DQN + speed governor", f"dqn_cap{cap}_seed0.gif"))
        governor_text = f'<p>The additional observation-based governor uses a <strong>{cap} m/s</strong> threshold, selected from 22, 24 and 26 m/s for seed-0 speed closeness. Its seed-0 mismatch is <strong>{governor["seed0_speed_difference_m_s"]:+.2f} m/s</strong>. This exploratory control was added after reward tuning failed to match speed. It uses only ego velocity from the same observation and adds no training.</p>'
    replays = "".join(
        f'<article><h3>{html.escape(label)}</h3><img src="../{name}" alt="{html.escape(label)} driving replay"></article>'
        for label, name in replay_names
    )
    payload = json.dumps(groups).replace("<", "\\u003c")
    document = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Risk-aware driving · replicated experiment</title><style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#0d1423;color:#e9eef7;font:16px/1.65 system-ui,sans-serif}
main{max-width:1200px;margin:auto;padding:50px 28px}h1{font-size:clamp(30px,4vw,48px);line-height:1.13;max-width:1000px;letter-spacing:-.025em}
.eyebrow{color:#62dbbf;letter-spacing:.12em;font-size:12px;font-weight:750}.intro,small,footer{color:#acbad0}.intro{max-width:950px}
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:15px;margin:30px 0}.card{background:#172439;padding:20px;border-radius:12px}.number{font-size:32px;font-weight:750;color:#8be3cf}.caption{color:#acbad0;font-size:13px}
section{margin-top:44px}.figure{width:100%;border-radius:12px;background:white}h2{font-size:24px;letter-spacing:-.015em}p{max-width:1000px}
.notice{padding:18px 22px;background:#172439;border-left:3px solid #62dbbf;border-radius:6px}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;padding:12px;border-bottom:1px solid #283348;white-space:nowrap}th{color:#8ddfcd}tr.highlight{background:#13332f}
select{font:inherit;padding:8px 12px;border:1px solid #4a5b74;background:#172439;border-radius:7px;color:#e9eef7}label{margin-right:12px}a{color:#8ddfcd}summary{cursor:pointer;font-weight:650}details{margin-top:25px}article img{width:100%;border-radius:8px}article{margin:22px 0}h3{font-size:16px}footer{font-size:13px;margin-top:40px}@media(max-width:650px){.cards{grid-template-columns:1fr}main{padding:28px 16px}}
</style><main><div class="eyebrow">DEEP REINFORCEMENT LEARNING / DEVELOPMENT EXPERIMENT</div>
<h1>Can a cautious return selector make deep RL driving safer?</h1>
<p class="intro">Three independent training seeds, a stronger scalar baseline, and paired evaluations separate a promising first run from a repeatable result.</p>
<div class="cards"><div class="card"><div class="number">REDUCTION pp</div><div class="caption">fewer normal-traffic collisions vs QR-DQN mean</div></div><div class="card"><div class="number">SLOWER% slower</div><div class="caption">normal-traffic speed cost of the cautious selector</div></div><div class="card"><div class="number">3 training seeds</div><div class="caption">20,000 interactions per model · 30 scenarios per density</div></div></div>
<p class="notice"><strong>Development evidence.</strong> No final confirmation scenarios have been used. All learned methods are evaluated with the same reward. The scalar baseline was selected for speed closeness using development data; actual residual speed differences are reported. Three training seeds give limited evidence of robustness.</p>
<section><h2>Compare the policies</h2><label for="density">Traffic density</label><select id="density"><option value="1">1.0 · training density</option><option value="1.5">1.5 · denser traffic</option><option value="2">2.0 · stress test</option></select>
<div class="scroll"><table><thead><tr><th>Method</th><th>Crashes</th><th>Seed range</th><th>Speed m/s</th><th>Distance m</th><th>Duration s</th><th>Common return</th></tr></thead><tbody id="metrics"></tbody></table></div><p><small>Equal-weight averages across trained models. Ranges show the three training-seed estimates; they are not confidence intervals. Each model sees the same 30 scenario seeds at each density.</small></p>
<img class="figure" src="controls_replication.png" alt="Mean collision rate and speed with training seed ranges"><p><small>Lines: seed means. Dots: individual trained models. Shading: minimum to maximum across the three training seeds.</small></p></section>
<section><h2>Does the advantage survive stronger controls?</h2><p>Reward-tuned DQN uses a training collision penalty of <strong>PENALTY</strong>, selected only for proximity to the cautious seed-0 policy's normal-traffic speed. Its seed-0 speed mismatch is <strong>MISMATCH m/s</strong>. Selection used neither collision rate nor the other training seeds. This is speed targeting, not an exact speed match.</p>GOVERNOR
<img class="figure" src="controls_tradeoff.png" alt="Speed versus collisions for all seeds and densities"><p><small>Small dots: individual models. Diamonds: seed averages. Lower collision rate and higher speed are preferable; a faster policy and a safer policy can trade places without either dominating.</small></p>
<img class="figure" src="controls_paired.png" alt="Paired policy differences with exploratory intervals"><p><small>Negative differences favor the cautious selector on collisions. Intervals use 5,000 two-way bootstrap replicates over training seeds and shared scenario seeds while retaining policy pairing. With only three trained models, treat these intervals as exploratory.</small></p></section>
<section><h2>What changed in the behavior?</h2><img class="figure" src="controls_diagnostics.png" alt="Action mix and illustrative discounted reward paths">
<p>The action trace uses separate development scenarios (11000–11029). Gap and time-to-collision measurements use privileged simulator state for diagnosis only. They never enter the learned policy.</p>
<p>The reward illustration compares nine rewards of 0.4 followed by a -2 crash against thirty rewards of 0.12 without a crash. At gamma 0.9 the first path has higher discounted return. This demonstrates an objective-design issue; it does not establish the cause of a particular failure.</p></section>
<section><h2>Replay the same initial scenario</h2><p><small>Training seed 0, scenario seed 10000, normal traffic. These rollouts were selected by seed before inspecting outcomes.</small></p>REPLAYS</section>
<details><summary>Inspect every training-seed result</summary><div class="scroll"><table><thead><tr><th>Method</th><th>Density</th><th>Training seed</th><th>Crashes</th><th>Speed m/s</th><th>Distance m</th><th>Duration s</th></tr></thead><tbody>SEEDROWS</tbody></table></div></details>
<section><h2>Reproduce and inspect</h2><p><a href="../controls/summary.json">Raw aggregated results and paired comparisons</a> · <a href="../controls/baseline_selection.json">Recorded baseline selection</a> · <a href="../baseline-report/index.html">Preserved initial experiment report</a></p><p><code>python run_controls.py --jobs 2</code> reuses verified results and reproduces this experiment. Source code, configuration, model checkpoints, scenario results and diagnostic traces are included in the project.</p></section>
<footer>Lower-tail action selection remains a deployment heuristic, without a global CVaR optimality or safety guarantee. The network was trained for mean-greedy continuation; changing its selector changes the continuation policy. Inputs partially observe the simulator. Density changes vehicle spacing, not the entire driving domain. The full-state cautious rule from initial experiment is preserved separately because it has privileged information. Final confirmation seeds 20000 onward remain reserved.</footer>
<script>const groups=PAYLOAD;function render(){const d=Number(document.getElementById('density').value);const body=document.getElementById('metrics');body.replaceChildren();for(const [label,rows] of Object.entries(groups)){const r=rows.find(x=>x.density===d);const tr=document.createElement('tr');if(label==='QR-DQN lower 25%')tr.className='highlight';const values=[label,(100*r.crash_rate).toFixed(1)+'%',r.crash_rate_seed_range.map(x=>(100*x).toFixed(1)+'%').join(' – '),r.mean_speed_m_s.toFixed(2),r.mean_distance_m.toFixed(0),r.mean_duration_s.toFixed(1),r.mean_return.toFixed(2)];for(const value of values){const td=document.createElement('td');td.textContent=value;tr.appendChild(td)}body.appendChild(tr)}}document.getElementById('density').addEventListener('change',render);render();</script></main></html>"""
    replacements = {
        "REDUCTION": f"{reduction:.1f}",
        "SLOWER": f"{slow:.1f}",
        "PENALTY": str(selection["selected_collision_penalty"]),
        "MISMATCH": f'{selection["seed0_speed_difference_m_s"]:+.2f}',
        "REPLAYS": replays,
        "SEEDROWS": "".join(per_seed_rows),
        "PAYLOAD": payload,
        "GOVERNOR": governor_text,
    }
    for key, value in replacements.items():
        document = document.replace(key, value)
    (output / "index.html").write_text(document)
    print(f"Wrote {output / 'index.html'}")
