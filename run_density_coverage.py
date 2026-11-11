"""Compare fixed-density and mixed-density QR-DQN training at equal budgets."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from driving_rl.experiment import save_json
ROOT=Path(__file__).resolve().parent
STAGE=ROOT/'results/density_coverage'
POLICIES={'mean':(1.,None),'tail25':(.25,None),'governed':(1.,25.)}
DESIGN={'training_steps':40000,'training_seeds':[0,1,2],'training_densities':[1.,1.5,2.],
 'density_sampling':'Uniform per episode, independent RNG; equal episodes do not imply equal transitions.',
 'comparison':'Existing fixed-density 40k QR-DQN, same architecture/reward/optimizer/exploration.',
 'epsilon_decay_steps':12000,'primary_evaluation':{'densities':[1.,1.5,2.],'seed_start':10000,'episodes':30},
 'transfer_evaluation':{'densities':[1.25,2.5],'seed_start':16000,'episodes':30},
 'calibration':{'densities':[1.,1.5,2.],'seed_start':12000,'episodes':60},
 'decision_rules':{'mean':1.,'tail25':.25,'governor_cap_m_s':25.},
 'primary_contrast':'Mixed minus fixed cautious collision fraction at density 2; also report speed/distance and all seeds.',
 'limitations':'Development comparisons, three matched seed IDs; no tuned selector, safety claim or global CVaR guarantee. Density 1.5/2 are within mixture support; 2.5 is beyond both training supports. Test mixture changes hidden episode dynamics.',
 'confirmation_scenarios_used':False}
def read(p):return json.loads(p.read_text())
def run(cmd,log):
 with log.open('w') as f:subprocess.run([sys.executable,'-m',*cmd],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
def checkpoint(regime,seed):
 return (STAGE/f'qrdqn_seed{seed}/final.pt' if regime=='mixed' else ROOT/f'results/training_budget/qrdqn_seed{seed}/checkpoint_40000.pt')
def train(seed):
 folder=STAGE/f'qrdqn_seed{seed}'
 if not (folder/'summary.json').exists():
  run(['driving_rl.cli','train','--algorithm','qrdqn','--seed',str(seed),'--steps','40000','--exploration-steps','12000','--training-densities','1','1.5','2','--out',str(folder)],STAGE/f'train_seed{seed}.log')
 c=read(folder/'config.json'); base=read(ROOT/f'results/training_budget/qrdqn_seed{seed}/config.json')
 for k in ['seed','steps','algorithm','gamma','learning_rate','collision_penalty','buffer_size','batch_size','learning_starts','train_frequency','target_interval','final_epsilon','quantiles','exploration_steps','observation_dim','n_actions']:
  if c[k]!=base[k]:raise ValueError(f'Mismatched {k}: {folder}')
 if c['training_densities']!=[1.,1.5,2.] or not read(folder/'summary.json')['training_complete']:raise ValueError('Incomplete/mismatched mixture run')
 print(f'Training complete: seed {seed}',flush=True)
def evaluate(spec):
 regime,seed,policy,block=spec
 alpha,cap=POLICIES[policy]; d=DESIGN['primary_evaluation' if block=='main' else 'transfer_evaluation']
 ck=checkpoint(regime,seed)
 out=ROOT/f'results/training_budget/eval_qrdqn_{seed}_40000_{policy}.json' if regime=='fixed' and block=='main' else STAGE/f'eval_{regime}_{policy}_seed{seed}_{block}.json'
 if out.exists():
  data=read(out);p=data['provenance']
  if (p['checkpoint_sha256'],p['alpha'],p['speed_governor_cap_m_s'],p['seed_start'],p['episodes_per_density'],[r['density'] for r in data['summary']],len(data['episodes']))!=(hashlib.sha256(ck.read_bytes()).hexdigest(),alpha,cap,d['seed_start'],d['episodes'],d['densities'],len(d['densities'])*d['episodes']):raise ValueError(f'Mismatched evaluation: {out}')
 else:
  cmd=['driving_rl.cli','evaluate','--checkpoint',str(ck),'--alpha',str(alpha),'--seed-start',str(d['seed_start']),'--episodes',str(d['episodes']),'--densities',*[str(x) for x in d['densities']],'--out',str(out)]
  if cap:cmd+=['--speed-cap',str(cap)]
  run(cmd,out.with_suffix('.log'))
 print(f'Evaluated {regime} {seed} {policy} {block}',flush=True)
 return out
def calibrate(spec):
 seed,policy=spec;alpha,_=POLICIES[policy];ck=checkpoint('mixed',seed)
 out=STAGE/f'calibration_{policy}_seed{seed}.json'
 if out.exists():
  data=read(out);p=data['provenance']
  if (p['checkpoint_sha256'],p['alpha'],p['seed_start'],p['episodes_per_density'],p['speed_cap'],len(data['episodes']))!=(hashlib.sha256(ck.read_bytes()).hexdigest(),alpha,12000,60,None,180):raise ValueError(f'Mismatched calibration: {out}')
 else:run(['driving_rl.calibration','--checkpoint',str(ck),'--alpha',str(alpha),'--out',str(out)],out.with_suffix('.log'))
 print(f'Calibrated mixed {seed} {policy}',flush=True)
def replay(regime):
 ck=checkpoint(regime,0);out=STAGE/f'replay_{regime}.json'
 if out.exists():
  data=read(out);p=data['provenance']
  if (p['checkpoint_sha256'],p['alpha'],p['seed_start'],p['episodes_per_density'])!=(hashlib.sha256(ck.read_bytes()).hexdigest(),.25,10000,1):raise ValueError('Replay metadata mismatch')
 else:
  run(['driving_rl.cli','evaluate','--checkpoint',str(ck),'--alpha','.25','--episodes','1','--seed-start','10000','--densities','2','--out',str(out),'--gif',str(STAGE/f'replay_{regime}.gif')],out.with_suffix('.log'))
 print(f'Recorded {regime} cautious replay',flush=True)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--jobs',type=int,default=3);p.add_argument('--stage',choices=['all','evaluate','report'],default='all');a=p.parse_args()
 if a.jobs<1:p.error('--jobs must be positive')
 STAGE.mkdir(parents=True,exist_ok=True)
 design=STAGE/'design.json'
 if design.exists() and read(design)!=DESIGN:raise ValueError('Refusing to change recorded design')
 if not design.exists():save_json(design,DESIGN)
 if a.stage!='report':
  with ThreadPoolExecutor(max_workers=a.jobs) as pool:
   if a.stage=='all':list(pool.map(train,range(3)))
   list(pool.map(evaluate,[(regime,s,policy,block) for regime in ['fixed','mixed'] for s in range(3) for policy in POLICIES for block in ['main','transfer']]))
   list(pool.map(calibrate,[(s,policy) for s in range(3) for policy in ['mean','tail25']]))
   list(pool.map(replay,['fixed','mixed']))
 from driving_rl.density_coverage_report import report
 report()
if __name__=='__main__':main()
