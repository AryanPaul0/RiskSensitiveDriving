"""Controlled budget study: train, validate, evaluate and report."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import torch
from driving_rl.experiment import save_json
ROOT=Path(__file__).resolve().parent
STAGE=ROOT/'results/training_budget'
def run(command,log):
    with log.open('w') as f:
        subprocess.run([sys.executable,'-m',*command],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
def train(spec):
    algo,seed=spec
    folder=STAGE/f'{algo}_seed{seed}'
    if not (folder/'summary.json').exists():
        run(['driving_rl.cli','train','--algorithm',algo,'--seed',str(seed),'--steps','40000','--exploration-steps','12000','--out',str(folder)],STAGE/f'train_{algo}_{seed}.log')
    c=json.loads((folder/'config.json').read_text())
    if (c['steps'],c['exploration_steps'],c['seed'],c['algorithm'])!=(40000,12000,seed,algo):
        raise ValueError(f'Mismatched training configuration: {folder}')
    for key,value in {'gamma':.9,'learning_rate':.0005,'collision_penalty':-2.,'buffer_size':30000,'batch_size':64,'learning_starts':500,'train_frequency':1,'target_interval':250,'final_epsilon':.05,'quantiles':32}.items():
        if c[key] != value: raise ValueError(f'Mismatched {key}: {folder}')
    old=torch.load(ROOT/f'results/{algo}_seed{seed}/final.pt',weights_only=True)
    new=torch.load(folder/'checkpoint_20000.pt',weights_only=True)
    match=all(torch.equal(old['online'][k],new['online'][k]) for k in old['online'])
    print(f'Trained {algo} seed {seed}; exact 20k weights: {match}',flush=True)
    return dict(algorithm=algo,seed=seed,prefix_weights_equal=match)
def evaluate(spec):
    algo,seed,steps,policy=spec
    ckpt=STAGE/f'{algo}_seed{seed}/checkpoint_{steps}.pt'
    out=STAGE/f'eval_{algo}_{seed}_{steps}_{policy}.json'
    alpha=.25 if policy=='tail25' else 1.
    cap=25. if policy=='governed' else None
    digest=hashlib.sha256(ckpt.read_bytes()).hexdigest()
    if out.exists():
        p=json.loads(out.read_text())['provenance']
        if (p['checkpoint_sha256'],p['alpha'],p['speed_governor_cap_m_s'],p['seed_start'],p['episodes_per_density'])!=(digest,alpha,cap,10000,30):
            raise ValueError(f'Mismatched evaluation: {out}')
        data=json.loads(out.read_text())
        if [r['density'] for r in data['summary']] != [1.,1.5,2.] or len(data['episodes']) != 90:
            raise ValueError(f'Incomplete evaluation: {out}')
    else:
        cmd=['driving_rl.cli','evaluate','--checkpoint',str(ckpt),'--alpha',str(alpha),'--episodes','30','--out',str(out)]
        if cap: cmd+=['--speed-cap',str(cap)]
        run(cmd,out.with_suffix('.log'))
    print(f'Evaluated {out.name}',flush=True)
def calibrate(spec):
    seed,policy=spec
    ckpt=STAGE/f'qrdqn_seed{seed}/checkpoint_40000.pt'
    out=STAGE/f'calibration_{policy}_seed{seed}.json'
    alpha=.25 if policy=='tail25' else 1.
    if out.exists():
        p=json.loads(out.read_text())['provenance']
        if (p['checkpoint_sha256'],p['alpha'],p['seed_start'],p['episodes_per_density'])!=(hashlib.sha256(ckpt.read_bytes()).hexdigest(),alpha,12000,60):
            raise ValueError(f'Mismatched calibration: {out}')
        data=json.loads(out.read_text())
        if [r['density'] for r in data['summary']] != [1.,1.5,2.] or len(data['episodes']) != 180:
            raise ValueError(f'Incomplete calibration: {out}')
    else:
        run(['driving_rl.calibration','--checkpoint',str(ckpt),'--alpha',str(alpha),'--out',str(out)],out.with_suffix('.log'))
    print(f'Calibrated {out.name}',flush=True)
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs',type=int,default=3)
    args=parser.parse_args()
    if args.jobs<1: parser.error('--jobs must be positive')
    STAGE.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        prefixes=list(pool.map(train,[(a,s) for a in ['dqn','qrdqn'] for s in range(3)]))
        save_json(STAGE/'prefix_verification.json',prefixes)
        list(pool.map(evaluate,[(a,s,n,p) for a in ['dqn','qrdqn'] for s in range(3) for n in [20000,40000] for p in (['mean'] if a=='dqn' else ['mean','tail25','governed'])]))
        list(pool.map(calibrate,[(s,p) for s in range(3) for p in ['mean','tail25']]))
    from driving_rl.training_budget_report import report
    report()
if __name__=='__main__': main()
