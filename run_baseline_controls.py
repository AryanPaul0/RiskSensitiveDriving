"""Equal-coverage scalar baseline and independently speed-matched controls."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from driving_rl.experiment import save_json
from driving_rl.analysis import choose_speed_matched_control
ROOT=Path(__file__).resolve().parent
STAGE=ROOT/'results/baseline_controls'
CAPS=[21.,22.,23.,24.,25.]
DESIGN={'training_steps':40000,'training_seeds':[0,1,2],'training_densities':[1.,1.5,2.],
 'algorithm':'dqn','epsilon_decay_steps':12000,
 'comparison':'DQN and QR-DQN with equal interaction budgets, identical configured episode-density mixtures, rewards, network hidden sizes and optimizer settings; actual policy-dependent experience differs.',
 'primary_scenarios':{'seed_start':10000,'episodes':30,'densities':[1.,1.5,2.]},
 'transfer_scenarios':{'seed_start':16000,'episodes':30,'densities':[1.25,2.5]},
 'speed_control':{'caps_m_s':CAPS,'selection':'Independently for mixed DQN and QR mean, choose smallest absolute speed difference from mixed QR lower-quarter at seed 0, density 1; ties prefer lower cap. Collision rates are not selection inputs. Freeze before seeds 1/2.',
 'primary_comparison':'Mixed QR lower-quarter versus selected DQN governor and same-network QR governor. Report residual speeds and exploratory intervals conditional on selection.'},
 'no_additional_alpha_or_reward_tuning':True,'confirmation_scenarios_used':False}
def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(cmd,log):
 with log.open('w') as f:subprocess.run([sys.executable,'-m',*cmd],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
def checkpoint(algorithm,seed,regime='mixed'):
 if regime=='fixed':return ROOT/f'results/training_budget/dqn_seed{seed}/checkpoint_40000.pt'
 return STAGE/f'dqn_seed{seed}/final.pt' if algorithm=='dqn' else ROOT/f'results/density_coverage/qrdqn_seed{seed}/final.pt'
def train(seed):
 out=STAGE/f'dqn_seed{seed}'
 if not (out/'summary.json').exists():run(['driving_rl.cli','train','--algorithm','dqn','--seed',str(seed),'--steps','40000','--exploration-steps','12000','--training-densities','1','1.5','2','--out',str(out)],STAGE/f'train_seed{seed}.log')
 c=read(out/'config.json');reference=read(ROOT/f'results/density_coverage/qrdqn_seed{seed}/config.json')
 for k in ['seed','steps','training_densities','exploration_steps','gamma','learning_rate','collision_penalty','buffer_size','batch_size','learning_starts','train_frequency','target_interval','final_epsilon','quantiles','observation_dim','n_actions']:
  if c[k]!=reference[k]:raise ValueError(f'Configuration mismatch: {k}')
 if c['algorithm']!='dqn' or not read(out/'summary.json')['training_complete']:raise ValueError('Incomplete/mismatched baseline')
 print(f'Trained mixed DQN seed {seed}',flush=True)
def evaluate(spec):
 algorithm,regime,seed,cap,block=spec
 config=DESIGN['primary_scenarios' if block in ['main','selection'] else 'transfer_scenarios']
 densities=[1.] if block=='selection' else config['densities']
 ck=checkpoint(algorithm,seed,regime)
 token='mean' if cap is None else f'cap{cap:g}'.replace('.','p')
 out=STAGE/f'eval_{algorithm}_{regime}_{token}_seed{seed}_{block}.json'
 if algorithm=='dqn' and regime=='fixed' and block=='main' and cap is None:out=ROOT/f'results/training_budget/eval_dqn_{seed}_40000_mean.json'
 if out.exists():
  data=read(out);p=data['provenance']
  if (p['checkpoint_sha256'],p['alpha'],p['speed_governor_cap_m_s'],p['seed_start'],p['episodes_per_density'],[r['density'] for r in data['summary']],len(data['episodes']))!=(digest(ck),1.,cap,config['seed_start'],30,densities,30*len(densities)):raise ValueError(f'Mismatched evaluation {out}')
 else:
  cmd=['driving_rl.cli','evaluate','--checkpoint',str(ck),'--episodes','30','--seed-start',str(config['seed_start']),'--densities',*[str(d) for d in densities],'--out',str(out)]
  if cap is not None:cmd+=['--speed-cap',str(cap)]
  run(cmd,out.with_suffix('.log'))
 print(f'Evaluated {out.name}',flush=True)
 return out
def choose(algorithm):
 target_path=ROOT/'results/density_coverage/eval_mixed_tail25_seed0_main.json'
 target=read(target_path)['summary'][0]['mean_speed_m_s']
 rows=[]
 for cap in CAPS:
  p=STAGE/f'eval_{algorithm}_mixed_cap{cap:g}_seed0_selection.json'
  speed=read(p)['summary'][0]['mean_speed_m_s']
  rows.append({'cap_m_s':cap,'speed_m_s':speed,'evaluation_sha256':digest(p)})
 selected=choose_speed_matched_control(rows,target)
 result={'algorithm':algorithm,'rule':DESIGN['speed_control']['selection'],'target_speed_m_s':target,'target_evaluation_sha256':digest(target_path),'candidates':rows,'selected_cap_m_s':selected['cap_m_s'],'seed0_speed_difference_m_s':selected['speed_m_s']-target,'confirmation_scenarios_used':False}
 path=STAGE/f'{algorithm}_governor_selection.json'
 if path.exists() and read(path)!=result:raise ValueError('Refusing to alter recorded governor selection')
 if not path.exists():save_json(path,result)
 print(f'Frozen {algorithm} governor: {selected["cap_m_s"]} m/s',flush=True)
 return selected['cap_m_s']
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--jobs',type=int,default=3);p.add_argument('--stage',choices=['all','evaluate','report'],default='all');a=p.parse_args()
 if a.jobs<1:p.error('--jobs must be positive')
 STAGE.mkdir(parents=True,exist_ok=True);d=STAGE/'design.json'
 if d.exists() and read(d)!=DESIGN:raise ValueError('Recorded design mismatch')
 if not d.exists():save_json(d,DESIGN)
 if a.stage!='report':
  with ThreadPoolExecutor(max_workers=a.jobs) as pool:
   if a.stage=='all':list(pool.map(train,range(3)))
   list(pool.map(evaluate,[(algo,'mixed',0,cap,'selection') for algo in ['dqn','qrdqn'] for cap in CAPS]))
   selections={algo:choose(algo) for algo in ['dqn','qrdqn']}
   tasks=[('dqn',regime,s,None,block) for regime in ['fixed','mixed'] for s in range(3) for block in ['main','transfer']]
   tasks += [(algo,'mixed',s,cap,block) for algo,cap in selections.items() for s in range(3) for block in ['main','transfer']]
   list(pool.map(evaluate,tasks))
 from driving_rl.baseline_controls_report import report
 report()
if __name__=='__main__':main()
