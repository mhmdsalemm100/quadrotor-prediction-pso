"""Frozen independent holdout. Run with the existing scientific Python runtime."""
import hashlib
import json
import msvcrt
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import qmc

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parent
STUDY = PROJECT.parent
sys.path[:0] = [str(PROJECT),str(PROJECT/'hybrid_1800')]
import experiment as ex
from optimizer_1800 import optimize as seeded
from model import NOMINAL, baseline

b = ex.backend
KEYS = ['inertia_scale','motor_tau_s','rotor_scale']
SEEDS = [48001,48002,48003]
METHODS = ['ordinary','predictions','library','hybrid','reference']
SKPY = Path.home()/'Desktop/MASTER/Independent/fdr/python.exe'

def read(p):
    return json.loads(p.read_text())

def save(p,x):
    b.save(p,x)

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def prepare():
    b.original.verify_contract()
    training = [d for d in read(STUDY/'manifest.json')['drones'] if d['role']=='development_train']
    assert len(training)==20
    sources=[OUT/'run.py',OUT/'infer.py',PROJECT/'selection.json',PROJECT/'portfolio.py',
             PROJECT/'hybrid_1800/optimizer_1800.py',STUDY/'Orange_900_Pilot/runner.py',
             STUDY/'contract.json',PROJECT/'experiment.py']
    sources += [PROJECT/'models'/(n+'.joblib') for n in read(PROJECT/'selection.json')['selected']]
    sources += [p for d in training for p in sorted((STUDY/'drones'/d['id']/'runs').glob('*.json'))]
    contract=dict(hashes={str(p):sha(p) for p in sources},design_seed=20261103,
        design='First 10 Latin hypercube draws, same original factor bounds; Drone_31..40; no rejection or reselection.',
        seeds=SEEDS,methods=METHODS,short_budget=1800,reference_budget=3264,
        arms='Ordinary stratified PSO; predictions = frozen NN ensemble + GP; library = 5 lowest-fitness controllers from each of 4 nearest training drones; hybrid = both in original order. Seeded arms fill remaining guided slots with frozen sigma=.025 perturbations, keep same 31 global positions and baseline.',
        quality='Both training feasible, mean and worst settling <=1.05 times paired reference, no fewer local or combined passes. Retain all failures. Reference excluded from primary pass-rate comparison.',
        timing='Four workers, jobs shuffled with seed 9601. Warm objective in every worker. Measure elapsed optimizer call. Add median warm prediction/library lookup time per query as applicable. Model load, disk library construction, validation and offline development separate.',
        primary='Hybrid minus ordinary pass rate on paired cases at 1800; also report component arms, per-drone three-seed outcomes, settling, both robustness counts and runtime. No test-set retraining or method selection.')
    path=OUT/'protocol.json'
    if path.exists():
        assert read(path)['contract']==contract, 'Frozen dependencies changed'
    else:
        save(path,dict(frozen_utc=datetime.now(timezone.utc).isoformat(),contract=contract))
    units=qmc.LatinHypercube(3,seed=contract['design_seed']).random(10)
    plants=[]
    previous=np.array([[d[k] for k in KEYS] for d in read(STUDY/'manifest.json')['drones']])
    for index,(j,t,r) in enumerate(units,31):
        factors=[.85+.3*j,.014+.012*t,.9+.2*r]
        assert not np.any(np.all(np.isclose(previous,factors,rtol=0,atol=1e-12),axis=1))
        plant=NOMINAL.copy(); plant[:3]*=factors[0]; plant[10]=factors[1]; plant[4:6]*=factors[2]
        plants.append(dict(id=f'Drone_{index:02d}',**dict(zip(KEYS,factors)),plant=plant.tolist(),
                           source_mass_kg=1.4,throttle=.609,role='final_unseen'))
    if (OUT/'plants.json').exists(): assert read(OUT/'plants.json')==plants
    else: save(OUT/'plants.json',plants)
    if not (OUT/'predictions.json').exists():
        subprocess.run([str(SKPY),str(OUT/'infer.py')],check=True)
    pred=read(OUT/'predictions.json')['rows']
    cfg=b.original.config()
    start=time.perf_counter()
    bank={}
    for d in training:
        runs=sorted([read(p) for p in sorted((STUDY/'drones'/d['id']/'runs').glob('*.json'))],key=lambda x:x['fitness'])
        assert len(runs)==30
        bank[d['id']]=[(r['seed'],b.encode(np.array(r['gains']),cfg).tolist()) for r in runs[:5]]
    load=time.perf_counter()-start
    coordinates=np.array([[d[k] for k in KEYS] for d in training])
    data={}
    for d in plants:
        name=d['id']; timings=[]
        for _ in range(11):
            start=time.perf_counter()
            distance=np.sum(((coordinates-[d[k] for k in KEYS])/[.3,.012,.2])**2,axis=1)
            nearest=[training[i]['id'] for i in np.argsort(distance,kind='stable')[:4]]
            library=[v for n in nearest for _,v in bank[n]]
            timings.append(time.perf_counter()-start)
        data[name]=dict(predictions=pred[name]['vectors'],library=library,nearest=nearest,
                        library_seeds={n:[s for s,_ in bank[n]] for n in nearest},
                        prediction_seconds=pred[name]['seconds'],library_seconds=float(np.median(timings)))
    if not (OUT/'inputs.json').exists(): save(OUT/'inputs.json',dict(library_load_seconds=load,rows=data))
    frozen=read(OUT/'inputs.json')['rows']
    for name in data:
        for k in ('predictions','library','nearest','library_seeds'): assert data[name][k]==frozen[name][k]
    return plants,frozen

def warm():
    plant=read(OUT/'plants.json')[0]
    b.evaluate(baseline(),b.original.cases_for(plant),b.original.config())

def run_one(job):
    drone,seed,method,inputs=job
    cfg=b.original.config(); budget=3264 if method=='reference' else 1800
    start=time.perf_counter()
    if method in ('ordinary','reference'):
        result,z,x,v=b.original.optimize(cfg,b.original.cases_for(drone),seed,budget)
    else:
        centers=(inputs['predictions'] if method in ('predictions','hybrid') else []) + (inputs['library'] if method in ('library','hybrid') else [])
        result,z,x,v=seeded(drone,seed,centers,budget)
        assert np.array_equal(x[1:1+len(centers)],np.array(centers))
    elapsed=time.perf_counter()-start
    assert z.shape==(budget,12) and np.isfinite(z).all() and len(np.unique(z[:,4:].round(13),axis=0))==budget
    assert np.array_equal(z[:,0],np.arange(1,budget+1)) and np.array_equal(z[:,3],np.minimum.accumulate(z[:,2]))
    assert np.allclose(b.decode(z[z[:,2].argmin(),4:],cfg),result['gains'])
    metrics=b.metrics(drone,result['gains'],cfg)
    assert np.isclose(metrics['fitness'],z[-1,3])
    lookup=(inputs['prediction_seconds'] if method in ('predictions','hybrid') else 0)+(inputs['library_seconds'] if method in ('library','hybrid') else 0)
    stem=OUT/drone['id']/f'{method}_{seed}'; stem.parent.mkdir(exist_ok=True)
    np.savez_compressed(stem.with_suffix('.npz'),ledger=z,initial=x,velocity=v)
    save(stem.with_suffix('.json'),dict(drone=drone['id'],seed=seed,method=method,evaluations=budget,
         gains=result['gains'],optimizer_seconds=elapsed,initializer_lookup_seconds=lookup,
         tuning_seconds=elapsed+lookup,**metrics))
    return drone['id'],seed,method

def main():
    lock=open(OUT/'study.lock','a+b'); lock.write(b'0'); lock.flush(); lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    plants,inputs=prepare()
    jobs=[(d,s,m,inputs[d['id']]) for d in plants for s in SEEDS for m in METHODS]
    np.random.default_rng(9601).shuffle(jobs)
    jobs=[j for j in jobs if not (OUT/j[0]['id']/f'{j[2]}_{j[1]}.json').exists()]
    done=150-len(jobs)
    save(OUT/'status.json',dict(status='RUNNING',completed=done,total=150))
    with ProcessPoolExecutor(max_workers=4,initializer=warm) as pool:
        for f in as_completed([pool.submit(run_one,j) for j in jobs]):
            key=f.result(); done+=1
            save(OUT/'status.json',dict(status='RUNNING',completed=done,total=150,last=key))
            print(done,'/150',key,flush=True)
    save(OUT/'status.json',dict(status='SIMULATIONS_COMPLETE',completed=150,total=150))
if __name__=='__main__':
    main()
