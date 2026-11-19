"""Hardware-independent PSO search engine. Requires an explicit measured evaluator.
No software fallback: a missing/failed hardware evaluator stops the experiment.
"""
import json,time
from pathlib import Path
import numpy as np

def optimize(evaluate, baseline_unit, centers, seed, budget, output):
 if budget<64:raise ValueError('Budget must include all 64 initial particles')
 path=Path(output)
 if path.exists():raise FileExistsError('Refusing to overwrite evaluation ledger')
 rng=np.random.default_rng(seed);n=64;x=rng.random((n,8))
 for j in range(8):x[:,j]=(rng.permutation(n)+rng.random(n))/n
 x[0]=baseline_unit;v=rng.uniform(-.1,.1,(n,8));noise=np.random.default_rng(seed+910000)
 centers=np.asarray(centers,dtype=float).reshape(-1,8)
 if len(centers)>31:raise ValueError('Too many initializer centers')
 if len(centers):
  x[1:1+len(centers)]=np.clip(centers,0,1)
  for i in range(1+len(centers),33):x[i]=np.clip(centers[(i-1)%len(centers)]+noise.normal(0,.025,8),0,1)
 personal=x.copy();pf=np.full(n,np.inf);cache={};best=float('inf');bestx=None;iteration=0;start=time.perf_counter()
 with path.open('x',encoding='utf-8') as f:
  while len(cache)<budget:
   values=[]
   for p in x:
    key=tuple(np.round(p,13))
    if key not in cache:
     if len(cache)>=budget:break
     result=evaluate(p.copy())
     if result.get('execution')!='hardware':raise RuntimeError('Hardware evidence required; refusing software substitution')
     value=float(result['fitness'])
     if not np.isfinite(value):raise RuntimeError('Evaluator returned nonfinite fitness')
     cache[key]=value
     if value<best:best=value;bestx=p.copy()
     f.write(json.dumps({'evaluation':len(cache),'iteration':iteration,'unit_gains':p.tolist(),'fitness':value,'best_fitness':best,'elapsed_s':time.perf_counter()-start,'hardware_evidence':result})+'\n');f.flush()
    values.append(cache[key])
   if len(cache)>=budget:break
   values=np.array(values);better=values<pf;personal[better]=x[better];pf[better]=values[better]
   progress=np.clip((len(cache)-64)/(budget-64),0,1)
   v=(.9-.5*progress)*v+1.7*rng.random((n,8))*(personal-x)+1.7*rng.random((n,8))*(bestx-x)
   v=np.clip(v,-.2,.2);trial=x+v;hit=(trial<0)|(trial>1);x=np.clip(trial,0,1);v[hit]*=-.5;iteration+=1
   if iteration>10000:raise RuntimeError('PSO stalled')
 return {'best_unit_gains':bestx.tolist(),'fitness':best,'unique_evaluations':len(cache),'wall_seconds':time.perf_counter()-start}
