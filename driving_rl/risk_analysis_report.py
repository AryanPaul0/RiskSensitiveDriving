"""Scientific figures and an interactive viewer for risk analysis development results."""
import json
from pathlib import Path

import numpy as np


COLORS = {"QR-DQN mean": "#9563c2", "QR-DQN lower 25%": "#159c87",
          "Same QR-DQN + governor": "#df9144"}
POLICIES = {"mean": "QR-DQN mean", "tail25": "QR-DQN lower 25%",
            "governed": "Same QR-DQN + governor"}


def create_report(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(root)
    data = json.loads((root / "risk_analysis/summary.json").read_text())
    output = root / "report"
    output.mkdir(exist_ok=True)
    plt.rcParams.update({"figure.facecolor": "#f5f7fb", "axes.facecolor": "white",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 10, "axes.titlesize": 12})
    fig, axes = plt.subplots(1, 3, figsize=(15,4.2), constrained_layout=True)
    for name, rows in data["groups"].items():
        x = [row["density"] for row in rows]
        for axis, metric, scale in [(axes[0],"crash_rate",100), (axes[1],"mean_speed_m_s",1)]:
            axis.plot(x, [row[metric]*scale for row in rows], "o-", color=COLORS[name], label=name)
            for row in rows:
                axis.scatter([row["density"]]*3, [seed[metric]*scale for seed in row["per_seed"]],
                             color=COLORS[name], alpha=.35, s=16)
    comparisons = data["paired_comparisons"]
    x = [row["density"] for row in comparisons]
    y = np.array([row["crashed"]["mean_difference"]*100 for row in comparisons])
    bounds = np.array([row["crashed"]["exploratory_95_percent_interval"] for row in comparisons])*100
    axes[2].errorbar(x, y, yerr=np.maximum(0,np.stack([y-bounds[:,0],bounds[:,1]-y])),
                     fmt="o-", capsize=5, color="#159c87")
    axes[2].axhline(0,color="#7a8697",linestyle="--")
    axes[0].set(title="Same models, different deployment", ylabel="Collision rate (%)", ylim=(-3,103))
    axes[1].set(title="Check the remaining speed mismatch", ylabel="Average speed (m/s)")
    axes[2].set(title="Cautious minus governed · paired", ylabel="Collision difference (percentage points)")
    for axis in axes:
        axis.set_xlabel("Traffic density")
        axis.grid(alpha=.15)
    axes[0].legend(fontsize=7, loc="lower right")
    fig.savefig(output / "risk_analysis_control.png",dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1,3,figsize=(15,4.3),constrained_layout=True)
    for axis, density in zip(axes,[1.,1.5,2.]):
        rows = [next(row for row in item["summary"] if row["density"]==density)
                for item in data["risk_frontier"]]
        speeds = [row["mean_speed_m_s"] for row in rows]
        crashes = [row["crash_rate"]*100 for row in rows]
        axis.plot(speeds,crashes,"o-",color="#159c87",label="Quantile tail selector")
        for item,speed,crash in zip(data["risk_frontier"],speeds,crashes):
            axis.annotate(f'α={item["alpha"]:g}',(speed,crash),xytext=(4,-12),textcoords="offset points",fontsize=8)
        control = next(row for row in data["groups"]["Same QR-DQN + governor"] if row["density"]==density)
        axis.scatter(control["mean_speed_m_s"],control["crash_rate"]*100,color="#df9144",marker="*",s=160,label="Mean + fixed governor")
        axis.set(title=f"Density {density:g}",xlabel="Mean speed (m/s)",ylim=(-3,108))
        axis.grid(alpha=.15)
    axes[0].set_ylabel("Collision rate (%)")
    axes[0].legend(fontsize=8)
    fig.savefig(output / "risk_analysis_frontier.png",dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1,3,figsize=(15,4.4),constrained_layout=True)
    for axis,density in zip(axes,[1.,1.5,2.]):
        axis.plot([0,1],[0,1],"--",color="#7a8697",label="Nominal quantile fraction")
        for policy,name in POLICIES.items():
            row = next(row for row in data["calibration"][policy] if row["density"]==density)
            taus = row["taus"]
            low,high = np.array(row["coverage_lower"]),np.array(row["coverage_upper"])
            axis.plot(taus,(low+high)/2,color=COLORS[name],label=name)
            axis.fill_between(taus,low,high,color=COLORS[name],alpha=.15)
        axis.set(title=f"Density {density:g}",xlabel="Predicted quantile fraction τ",xlim=(0,1),ylim=(0,1))
        axis.grid(alpha=.15)
    axes[0].set_ylabel("Empirical quantile-compatibility bracket")
    axes[0].legend(fontsize=7,loc="lower right")
    fig.savefig(output / "risk_analysis_calibration.png",dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1,3,figsize=(15,3.8),constrained_layout=True)
    for axis,density in zip(axes,[1.,1.5,2.]):
        for index,(policy,name) in enumerate(POLICIES.items()):
            row = next(row for row in data["calibration"][policy] if row["density"]==density)
            low,high = row["mean_prediction_bias_bounds"]
            middle = (low+high)/2
            axis.errorbar(middle,index,xerr=[[middle-low],[high-middle]],fmt="o",capsize=5,color=COLORS[name])
        axis.axvline(0,color="#7a8697",linestyle="--")
        axis.set(title=f"Density {density:g}",xlabel="Mean prediction minus realized return",yticks=[0,1,2],yticklabels=["Mean", "Lower 25%", "Governed"])
        axis.grid(alpha=.15)
    fig.savefig(output / "risk_analysis_bias.png",dpi=180)
    plt.close(fig)
    calibration = {policy:[{key:row[key] for key in ["density","mean_prediction_bias_bounds",
                                "censored_fraction","quantile_crossing_fraction",
                                "minimum_mean_absolute_compatibility_gap"]} for row in rows]
                   for policy,rows in data["calibration"].items()}
    payload = {"groups":data["groups"],"frontier":data["risk_frontier"],
               "calibration":calibration,"failures":data["failure_diagnostics"]}
    nominal = data["paired_comparisons"][0]
    document = (Path(__file__).parent / "report_risk_analysis.html").read_text()
    replacements = {"__PAYLOAD__":json.dumps(payload).replace("<","\\u003c"),
                    "__DELTA__":f'{-100*nominal["crashed"]["mean_difference"]:.1f}',
                    "__SPEED__":f'{abs(nominal["mean_speed_m_s"]["mean_difference"]):.2f}',
                    "__INTERVAL__":" to ".join(f'{100*value:.1f}' for value in nominal["crashed"]["exploratory_95_percent_interval"]),
                    "__CAP__":f'{data["selection"]["selected_cap_m_s"]:g}'}
    for key,value in replacements.items():
        document = document.replace(key,value)
    (output / "index.html").write_text(document)
    print(output / "index.html")
