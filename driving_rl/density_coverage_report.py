"""Reports for equal-budget traffic-density coverage experiments."""
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .analysis import paired_matrix, crossed_bootstrap, summarize_group
from .calibration import summarize_rows
from .experiment import save_json
ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'results/density_coverage'
POLICIES={'mean':'Mean selector','tail25':'Lower-quarter selector','governed':'Mean + 25 m/s governor'}
def read(p):return json.loads(p.read_text())
def load(regime,policy,block):
 return [read(ROOT/f'results/training_budget/eval_qrdqn_{s}_40000_{policy}.json' if regime=='fixed' and block=='main' else STAGE/f'eval_{regime}_{policy}_seed{s}_{block}.json') for s in range(3)]
def table(headers,rows):
 return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+str(v)+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def report():
 viewer=ROOT/'results/report';backup=ROOT/'results/training_budget-report'
 if not backup.exists():shutil.copytree(viewer,backup)
 groups={};contrasts=[];calibration=[];exposure=[];controls=[]
 for block,densities in [('main',[1.,1.5,2.]),('transfer',[1.25,2.5])]:
  groups[block]={}
  for regime in ['fixed','mixed']:
   groups[block][regime]={p:summarize_group(load(regime,p,block)) for p in POLICIES}
  for policy in POLICIES:
   for density in densities:
    item={'block':block,'policy':policy,'density':density}
    for metric in ['crashed','mean_speed_m_s','distance_m','return']:
     _,matrix=paired_matrix(load('mixed',policy,block),load('fixed',policy,block),density,metric)
     item[metric]=crossed_bootstrap(matrix)
    contrasts.append(item)
  for regime in ['fixed','mixed']:
   for density in densities:
    item={'block':block,'regime':regime,'density':density}
    for metric in ['crashed','mean_speed_m_s']:
     _,matrix=paired_matrix(load(regime,'tail25',block),load(regime,'governed',block),density,metric)
     item[metric]=crossed_bootstrap(matrix)
    controls.append(item)
 for regime in ['fixed','mixed']:
  folder=ROOT/'results/training_budget' if regime=='fixed' else STAGE
  for policy in ['mean','tail25']:
   data=[read(folder/f'calibration_{policy}_seed{s}.json') for s in range(3)]
   for density in [1.,1.5,2.]:
    rows=[r for d in data for r in d['episodes'] if r['density']==density]
    calibration.append({'regime':regime,'policy':policy,'density':density,**summarize_rows(rows)})
 for seed in range(3):
  d=read(STAGE/f'qrdqn_seed{seed}/summary.json')
  if sum(d['steps_by_density'].values())!=40000:raise ValueError('Training exposure does not sum to budget')
  exposure.append({'seed':seed,**d})
 result={'design':read(STAGE/'design.json'),'groups':groups,'mixed_minus_fixed':contrasts,'tail_minus_governed':controls,'calibration':calibration,'training_exposure':exposure}
 save_json(STAGE/'summary.json',result)
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(2,3,figsize=(14,8))
 for col,policy in enumerate(POLICIES):
  for regime,color in [('fixed','C0'),('mixed','C1')]:
   rows=groups['main'][regime][policy];x=[r['density'] for r in rows]
   for row,metric,scale in [(0,'crash_rate',100),(1,'mean_speed_m_s',1)]:
    ax=axes[row,col];ax.plot(x,[r[metric]*scale for r in rows],'-o',color=color,label=regime)
    for s in range(3):ax.plot(x,[r['per_seed'][s][metric]*scale for r in rows],color=color,lw=.8,alpha=.25)
    ax.grid(alpha=.2);ax.set_xticks(x);ax.set_xlabel('Evaluation density')
  axes[0,col].set_title(POLICIES[policy]);axes[0,col].set_ylim(0,105);axes[0,col].legend()
 axes[0,0].set_ylabel('Episodes with collision (%)');axes[1,0].set_ylabel('Mean episode speed (m/s)')
 fig.suptitle('Equal 40k-step budgets; faint lines show individual training seeds')
 fig.tight_layout();fig.savefig(viewer/'density_coverage_performance.png',dpi=150);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(11,4))
 for ax,policy in zip(axes,['mean','tail25']):
  for regime,color,offset in [('fixed','C0',-.035),('mixed','C1',.035)]:
   rows=[r for r in calibration if r['regime']==regime and r['policy']==policy]
   lo=np.array([r['mean_prediction_bias_bounds'][0] for r in rows]);hi=np.array([r['mean_prediction_bias_bounds'][1] for r in rows]);mid=(lo+hi)/2
   ax.errorbar(np.array([1.,1.5,2.])+offset,mid,yerr=[mid-lo,hi-mid],fmt='o-',color=color,label=regime,capsize=4)
  ax.axhline(0,color='black',lw=.8);ax.set_xticks([1,1.5,2]);ax.set_xlabel('Evaluation density');ax.set_title(POLICIES[policy]);ax.grid(alpha=.2);ax.legend()
 axes[0].set_ylabel('Prediction minus discounted return')
 fig.suptitle('Bias bounds account for censoring, not sampling uncertainty')
 fig.tight_layout();fig.savefig(viewer/'density_coverage_bias.png',dpi=150);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(10,4))
 for ax,key,title in zip(axes,['completed_episodes_by_density','steps_by_density'],['Completed episodes','Environment transitions']):
  bottom=np.zeros(3)
  for d in [1.,1.5,2.]:
   total=np.array([sum(r[key].values()) for r in exposure]);v=np.array([r[key][str(d)] for r in exposure])/total*100
   ax.bar(range(3),v,bottom=bottom,label=f'Density {d:g}');bottom+=v
  ax.set_xticks(range(3),['Seed 0','Seed 1','Seed 2']);ax.set_ylim(0,100);ax.set_title(title);ax.set_ylabel('Share (%)')
 axes[1].legend(loc='upper left',bbox_to_anchor=(1,1))
 fig.suptitle('Uniform episode sampling does not produce equal transition shares')
 fig.tight_layout();fig.savefig(viewer/'density_coverage_exposure.png',dpi=150);plt.close(fig)
 primary=next(r for r in contrasts if r['block']=='main' and r['policy']=='tail25' and r['density']==2.)
 value=primary['crashed'];interval=[100*x for x in value['exploratory_95_percent_interval']]
 primary_text=f"At density 2, the lower-quarter selector's mixed-minus-fixed collision difference is {100*value['mean_difference']:+.1f} percentage points; exploratory 95% interval [{interval[0]:+.1f}, {interval[1]:+.1f}]."
 document='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Traffic-density coverage · Risk-sensitive driving</title><style>body{font:17px/1.6 system-ui;color:#18283d;background:#f4f7fb;margin:auto;max-width:1160px;padding:36px}h1{font-size:38px;line-height:1.2}section{background:white;padding:24px;margin:24px 0;border-radius:14px}img{width:100%;height:auto}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;text-align:left;border-bottom:1px solid #dde3eb}th{background:#eef3fa}a{color:#175bbc}summary{cursor:pointer}</style></head><body><p>DISTRIBUTIONAL RL / TRAFFIC COVERAGE</p><h1>Does broader training traffic improve driving?</h1><p>QR-DQN trained at density 1 versus a per-episode mixture of densities 1, 1.5 and 2. Each setting has three training seeds and 40,000 interactions per model. Rewards, architecture, optimizer and exploration schedule remain fixed.</p><p><a href="../training_budget-report/index.html">Training-budget comparison</a> · <a href="../risk_analysis-report/index.html">Interactive risk frontier and recorded driving</a> · <a href="../density_coverage/summary.json">Complete numerical results</a> · <a href="../density_coverage/design.json">Recorded experiment design</a></p>'''
 document+='<section><h2>Predeclared primary comparison</h2><p>'+primary_text+'</p><p>Negative differences favor mixture training. The selectors and 25 m/s governor are unchanged. This is a development comparison with only three training seeds; all confirmation scenarios remain reserved.</p></section>'
 def metric(regime,policy,density,key):
  return next(r for r in groups['main'][regime][policy] if r['density']==density)[key]
 d=1.5
 coverage_text=f"At density 1.5, cautious-selector collisions change from {100*metric('fixed','tail25',d,'crash_rate'):.1f}% to {100*metric('mixed','tail25',d,'crash_rate'):.1f}%, while speed changes from {metric('fixed','tail25',d,'mean_speed_m_s'):.2f} to {metric('mixed','tail25',d,'mean_speed_m_s'):.2f} m/s."
 governor_text=f"The normal-traffic governor result worsens: {100*metric('fixed','governed',1.,'crash_rate'):.1f}% to {100*metric('mixed','governed',1.,'crash_rate'):.1f}% collisions. Broader training is not an improvement for every deployment rule."
 document+='<section><h2>Main tradeoffs</h2><p>'+coverage_text+'</p><p>'+governor_text+'</p><p>Fewer observed collisions come with slower driving. The same-network governor comparisons retain residual speed differences, and uncertainty intervals remain exploratory. This study isolates coverage within QR-DQN; it does not compare DQN and QR-DQN under identical randomized training.</p></section>'
 document+='<section><h2>Performance across covered densities</h2><p>Same evaluation scenario block for both training settings (10000–10029). Densities 1.5 and 2 were absent from fixed-density training and are included in mixed training. Faint curves show seed variation.</p><img src="density_coverage_performance.png"></section>'
 for block,title in [('main','Covered traffic settings'),('transfer','Additional traffic settings')]:
  document+='<section><h2>'+title+'</h2>'
  if block=='transfer':document+='<p>Densities 1.25 and 2.5 use a separate development scenario block, 16000–16029. Density 1.25 is an intermediate value absent from the sampled mixture; density 2.5 is beyond both training ranges. These rows use different scenarios from the covered-settings table.</p>'
  document+=table(['Training','Selector','Density','Collisions','Speed m/s','Distance m','Worst 10% return'],[[regime,POLICIES[p],r['density'],f"{100*r['crash_rate']:.1f}%",f"{r['mean_speed_m_s']:.2f}",f"{r['mean_distance_m']:.0f}",f"{r['worst_10_percent_mean_return']:.2f}"] for regime in ['fixed','mixed'] for p in POLICIES for r in groups[block][regime][p]])+'</section>'
 document+='<section><h2>Paired collision effects: mixed minus fixed</h2><p>Exploratory intervals use 5,000 crossed bootstrap replicates over training seed IDs and shared scenarios, preserving policy pairing. Matching seed IDs shares initialization and exploration streams, but not the simulated trajectories. Three seeds and repeated development evaluation limit inference.</p>'+table(['Block','Selector','Density','Change pp','Exploratory interval pp','Per-seed changes pp','Speed change m/s'],[[r['block'],POLICIES[r['policy']],r['density'],f"{100*r['crashed']['mean_difference']:+.1f}",str([round(100*x,1) for x in r['crashed']['exploratory_95_percent_interval']]),str([round(100*x,1) for x in r['crashed']['per_training_seed_differences']]),f"{r['mean_speed_m_s']['mean_difference']:+.2f}"] for r in contrasts])+'</section>'
 document+='<section><h2>Cautious decisions versus speed control</h2><p>Lower quarter minus the mean selector with the fixed 25 m/s governor, within the same networks. The governor threshold is not reselected for mixed training; check residual speed differences.</p>'+table(['Block','Training','Density','Collision change pp','Exploratory interval pp','Speed change m/s'],[[r['block'],r['regime'],r['density'],f"{100*r['crashed']['mean_difference']:+.1f}",str([round(100*x,1) for x in r['crashed']['exploratory_95_percent_interval']]),f"{r['mean_speed_m_s']['mean_difference']:+.2f}"] for r in controls])+'</section>'
 document+='<section><h2>What the networks predict</h2><p>Same separate calibration scenarios (12000–12059) for both settings. The 30-step time cutoff right-censors continuing returns. Bounds retain surviving episodes; they are not statistical confidence intervals. Lower-quarter deployment changes the continuation policy relative to training.</p><img src="density_coverage_bias.png">'+table(['Training','Selector','Density','Bias bounds','Compatibility gap','Adjacent crossings'],[[r['regime'],r['policy'],r['density'],str([round(x,3) for x in r['mean_prediction_bias_bounds']]),f"{r['minimum_mean_absolute_compatibility_gap']:.3f}",f"{100*r['quantile_crossing_fraction']:.1f}%"] for r in calibration])+'</section>'
 document+='<section><h2>Same-scenario recorded driving</h2><p>Both lower-quarter policies use training seed 0, scenario seed 10000, density 2. The scenario was fixed before inspecting the new outcomes and is illustrative rather than a performance estimate.</p><div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px"><article><h3>Fixed-density training</h3><img src="../density_coverage/replay_fixed.gif"></article><article><h3>Mixed-density training</h3><img src="../density_coverage/replay_mixed.gif"></article></div></section>'
 document+='<section><h2>Actual training exposure</h2><p>Density is sampled uniformly at episode reset with a separate seeded generator. Short dense-traffic episodes contribute fewer transitions. The density label is not added to the observation; the network still receives the same 40 inputs. Equal step budgets therefore do not imply equal episode counts or equal data by density.</p><img src="density_coverage_exposure.png"></section>'
 document+='<section><h2>Reproduce and interpret</h2><p><code>python run_density_coverage.py --jobs 3</code> validates and reuses saved artifacts. <code>--stage evaluate</code> skips training; <code>--stage report</code> rebuilds the viewer. Run <code>python -m pytest -q</code> for checks.</p><p>Only traffic density changes during training. The simulator retains simplified vehicle interactions; the observation is partial. Quantile forecasts are not collision probabilities, and cautious deployment has no guaranteed safety or global CVaR optimality. Results concern sampled development scenarios and should not be presented as real-road validation.</p></section></body></html>'
 (viewer/'index.html').write_text(document)
 print(primary_text,flush=True)
 print('Density coverage report saved',flush=True)
if __name__=='__main__':report()
