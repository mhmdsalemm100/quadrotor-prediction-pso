"""Additive, simulation-only initialization study. Originals are read-only inputs."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse, csv, hashlib, json, sys, time, platform
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.stats import qmc, wilcoxon, rankdata

ROOT=Path(__file__).resolve().parent
STUDY=Path.home()/'Desktop/Quadrotor_PSO_Fresh_Study'
PROJECT=STUDY/'neural network by python'
OLD=PROJECT/'final_unseen_1800'
sys.path.insert(0,str(PROJECT))
import experiment as ex
b=ex.backend
METHODS=['random','lhs','obl','chaotic','predictions','library','hybrid']
SEEDS=[48001,48002,48003]
BUDGET=1800
CHECKPOINTS=[300,600,900,1200,1800]
N=64

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,o): b.save(p,o)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csvout(name,rows):
    with (ROOT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def initial(method,seed,inputs):
    rng=np.random.default_rng(seed)
    x=rng.random((N,8))
    if method=='lhs': x=qmc.LatinHypercube(d=8,seed=seed).random(N)
    elif method=='obl': x=np.vstack((x,1-x))
    elif method=='chaotic':
        z=np.clip(rng.random(8),np.finfo(float).eps,1-np.finfo(float).eps)
        out=[]
        for _ in range(N):
            z=4*z*(1-z); out.append(z.copy())
        x=np.array(out)
    elif method in ('predictions','library','hybrid'):
        cfg=b.original.config()
        x[0]=b.encode(b.baseline(),cfg)
        centers=np.array((inputs['predictions'] if method in ('predictions','hybrid') else [])+
                         (inputs['library'] if method in ('library','hybrid') else []))
        x[1:1+len(centers)]=centers
        noise=np.random.default_rng(seed+910000)
        for i in range(1+len(centers),33):
            x[i]=np.clip(centers[(i-1)%len(centers)]+noise.normal(0,.025,8),0,1)
    assert x.shape[1]==8 and np.all((x>=0)&(x<=1))
    return x

def optimize(method,seed,inputs,objective,budget=BUDGET):
    # Separate streams ensure initialization never changes update random draws.
    rng=np.random.default_rng(seed+2000000)
    v=np.random.default_rng(seed+1000000).uniform(-.1,.1,(N,8))
    initial_v=v.copy(); pool=initial(method,seed,inputs); x=pool.copy()
    ledger=[]; cache={}; best=float('inf'); bestx=None; iteration=0
    def evaluate(p):
        nonlocal best,bestx
        key=tuple(p.round(13))
        if key not in cache:
            value=float(objective(p)); assert np.isfinite(value)
            cache[key]=value
            if value<best: best=value; bestx=p.copy()
            ledger.append([len(cache),iteration,value,best,*p])
        return cache[key]
    if method=='obl':
        f=np.array([evaluate(p) for p in pool])
        assert len(ledger)==2*N, 'OBL must charge all 128 initial evaluations'
        x=pool[np.argsort(f,kind='stable')[:N]].copy()
    initial_swarm=x.copy(); personal=x.copy(); pf=np.full(N,np.inf)
    while len(cache)<budget:
        before=len(cache); values=[]
        for p in x:
            if len(cache)>=budget: break
            values.append(evaluate(p))
        if len(cache)>=budget: break
        f=np.array(values); better=f<pf
        personal[better]=x[better]; pf[better]=f[better]
        progress=np.clip((len(cache)-N)/(budget-N),0,1)
        v=(.9-.5*progress)*v+1.7*rng.random((N,8))*(personal-x)+1.7*rng.random((N,8))*(bestx-x)
        v=np.clip(v,-.2,.2); trial=x+v; hit=(trial<0)|(trial>1)
        x=np.clip(trial,0,1); v[hit]*=-.5; iteration+=1
        if iteration>10000: raise RuntimeError('Search stalled')
    z=np.array(ledger)
    assert z.shape==(budget,12) and np.array_equal(z[:,0],np.arange(1,budget+1))
    assert np.array_equal(z[:,3],np.minimum.accumulate(z[:,2]))
    assert len(np.unique(z[:,4:].round(13),axis=0))==budget
    return z,initial_swarm,initial_v,pool

def passed(r,ref):
    return bool(r['training_feasible'] and ref['training_feasible'] and
       r['mean_settling_s']<=1.05*ref['mean_settling_s'] and
       r['worst_settling_s']<=1.05*ref['worst_settling_s'] and
       r['local_passes']>=ref['local_passes'] and r['combined_passes']>=ref['combined_passes'])

def prepare():
    ROOT.mkdir(exist_ok=True)
    paths=[STUDY/n for n in ('model.py','study.py','pipeline.py','config.json','contract.json')]
    paths += [OLD/n for n in ('plants.json','inputs.json','protocol.json','offline_costs.json')]
    paths += [PROJECT/'models'/f'{n}.joblib' for n in read(PROJECT/'selection.json')['selected']]
    paths += list(OLD.glob('Drone_*/reference_*.json'))
    paths += [Path(__file__)]
    protocol=dict(methods=METHODS,seeds=SEEDS,plants=[d['id'] for d in read(OLD/'plants.json')],
      total_runs=210,budget=BUDGET,particles=N,dimensions=8,checkpoints=CHECKPOINTS,
      coordinate_system='Original normalized log-gain search coordinates [0,1]^8. LHS and OBL operate here; physical gains use original decode. OBL is 1-u, not arithmetic opposition in physical log-scaled gains.',
      original_preservation='All previous simulation and hardware results remain unchanged. No new hardware replay. Hardware evidence applies only to the original five methods/controllers.',
      initialization='Random iid uniform; LHS scipy; OBL 64 random plus 64 opposites, charge all 128 then select 64; chaotic 64 successive logistic-map values with 8 random starting coordinates, no burn-in. Guided methods preserve original centers/perturbations/baseline, with iid random global remainder.',
      pso='Identical 0.9 to 0.4 evaluation-dependent inertia, c1=c2=1.7, vmax=.2, boundary reflection=-.5. Independent common velocity/update streams. Unique objective calls counted; cache hits free. No postprocessing or local search.',
      comparability='New additive cohort, not numerically identical to earlier stratified ordinary/guided runs. Same previously examined Drone_31..40; this extension is not a fresh unseen validation set.',
      quality='Original paired 3264-reference criterion: both feasible, mean/worst settling <=1.05 reference and no fewer local/combined passes. Checkpoint validation is separate, excluded from optimizer budget and separately timed.',
      statistics='Six two-sided paired Wilcoxon comparisons, Holm correction; conventional Cliff delta across distributions plus paired rank-biserial effect. Additional whole-plant sign-flip check recognizes 10 plants with 3 seeds each. No selection on extension outcomes.',
      costs='One objective evaluation aggregates six scenario simulations. Warm single-process timing. Retrain selected fixed NN/GP only for timing, never replace frozen models. Report reused-library and data-generation-included scenarios; historical worker-duration sums are not elapsed wall time.',
      hashes={str(p):sha(p) for p in paths},python=sys.version,platform=platform.platform())
    p=ROOT/'protocol.json'
    if p.exists(): assert read(p)==protocol,'Protocol or frozen input changed'
    else: save(p,protocol)
    return read(OLD/'plants.json'),read(OLD/'inputs.json')['rows']

def warm():
    d=read(OLD/'plants.json')[0]
    b.evaluate(b.baseline(),b.original.cases_for(d),b.original.config())

def run_one(job):
    d,seed,method,inputs=job; cfg=b.original.config(); cases=b.original.cases_for(d)
    stem=ROOT/'runs'/d['id']/f'{method}_{seed}'; stem.parent.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    z,x,v,pool=optimize(method,seed,inputs,lambda u:b.evaluate(b.decode(u,cfg),cases,cfg))
    elapsed=time.perf_counter()-start
    np.savez_compressed(stem.with_suffix('.npz'),ledger=z,initial=x,velocity=v,initial_pool=pool)
    ref=read(OLD/d['id']/f'reference_{seed}.json'); checkpoints=[]
    start=time.perf_counter()
    for budget in CHECKPOINTS:
        row=z[:budget][np.argmin(z[:budget,2])]; gains=b.decode(row[4:],cfg)
        metrics=b.metrics(d,gains,cfg)
        assert np.isclose(metrics['fitness'],z[budget-1,3],rtol=1e-10,atol=1e-10)
        checkpoints.append(dict(evaluations=budget,gains=gains.tolist(),passed=passed(metrics,ref),**metrics))
    result=dict(method=method,plant=d['id'],seed=seed,evaluations=BUDGET,optimizer_seconds=elapsed,
                validation_seconds=time.perf_counter()-start,checkpoints=checkpoints)
    save(stem.with_suffix('.json'),result)
    return f"{d['id']} {seed} {method}"

def selftest():
    inputs=read(OLD/'inputs.json')['rows']['Drone_31']; velocities=[]
    for method in METHODS:
        z,x,v,pool=optimize(method,48001,inputs,lambda u:np.sum((u-.37)**2),budget=300)
        velocities.append(v)
        assert x.shape==(64,8)
        if method=='obl':
            assert np.allclose(pool[:64]+pool[64:],1)
            costs=np.sum((pool-.37)**2,axis=1)
            assert np.array_equal(x,pool[np.argsort(costs,kind='stable')[:64]])
        if method=='lhs':
            assert all(len(np.unique((pool[:,j]*64).astype(int)))==64 for j in range(8))
    assert all(np.array_equal(velocities[0],v) for v in velocities)
    print('PASS: 7 initializers, OBL accounting/selection, bounds, LHS strata, shared velocities, complete unique ledgers',flush=True)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['test','run']); parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.action=='test': selftest(); return
    plants,inputs=prepare()
    jobs=[(d,s,m,inputs[d['id']]) for d in plants for s in SEEDS for m in METHODS]
    np.random.default_rng(20261118).shuffle(jobs)
    jobs=[j for j in jobs if not (ROOT/'runs'/j[0]['id']/f'{j[2]}_{j[1]}.json').exists()]
    done=210-len(jobs); save(ROOT/'status.json',dict(status='RUNNING',completed=done,total=210))
    with ProcessPoolExecutor(max_workers=args.workers,initializer=warm) as pool:
        for f in as_completed([pool.submit(run_one,j) for j in jobs]):
            last=f.result(); done+=1
            save(ROOT/'status.json',dict(status='RUNNING',completed=done,total=210,last=last))
            print(done,'/210',last,flush=True)
    save(ROOT/'status.json',dict(status='SIMULATIONS_COMPLETE',completed=210,total=210))

if __name__=='__main__': main()
