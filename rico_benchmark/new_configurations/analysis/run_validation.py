from pathlib import Path
import sys,json,hashlib,importlib.util,subprocess,time
import numpy as np
from scipy.stats import qmc
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'RICO_extension'
sys.path.insert(0,str(BASE))
import study_extension as s
ST=s.STUDY; PROJECT=s.PROJECT; OLD=s.OLD
sys.path.insert(0,str(PROJECT/'hybrid_1800'))
spec=importlib.util.spec_from_file_location('original_reference',OLD/'run.py')
ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
SEEDS=[590101,590102,590103]
KEYS=['inertia_scale','motor_tau_s','rotor_scale']
def save(p,x):p.write_text(json.dumps(x,indent=2))
def prepare():
    previous=s.read(ST/'manifest.json')['drones']+s.read(OLD/'plants.json')
    previous=list({d['id']:d for d in previous if 1<=int(d['id'].split('_')[1])<=40}.values())
    assert len(previous)==40
    # Check all archived JSON scalar seeds without evaluating any controller.
    seen=set()
    for root in [ST,BASE]:
        for p in root.rglob('*.json'):
            try:
                obj=json.loads(p.read_text(encoding='utf-8'))
                def visit(x):
                    if isinstance(x,dict):
                        for k,v in x.items():
                            if 'seed' in k.lower() and isinstance(v,int):seen.add(v)
                            elif 'seed' in k.lower() and isinstance(v,list):seen.update(i for i in v if isinstance(i,int))
                            visit(v)
                    elif isinstance(x,list):
                        for v in x:visit(v)
                visit(obj)
            except (ValueError,UnicodeError):pass
    assert not set(SEEDS)&seen
    old=np.array([[d[k] for k in KEYS] for d in previous]); lo=np.array([.85,.014,.9]); span=np.array([.3,.012,.2])
    # Prespecified proximity rule: Euclidean distance >= .05 in range-normalized coordinates.
    # Reject whole LHS designs before seeing outcomes, preserving 10-point LHS strata.
    rejected=[]
    for design_seed in range(2026111901,2026112001):
        u=qmc.LatinHypercube(3,seed=design_seed).random(10)
        dist=np.linalg.norm(u[:,None,:]-(old-lo)[None,:,:]/span,axis=2)
        within=np.linalg.norm(u[:,None,:]-u[None,:,:],axis=2)+np.eye(10)*100
        if min(dist.min(),within.min())>=.05:break
        rejected.append(design_seed)
    else:raise RuntimeError('No separated design')
    plants=[]
    for i,f in enumerate(lo+span*u,41):
        plant=ref.NOMINAL.copy();plant[:3]*=f[0];plant[10]=f[1];plant[4:6]*=f[2]
        plants.append(dict(id=f'Drone_{i}',**dict(zip(KEYS,f)),plant=plant.tolist(),source_mass_kg=1.4,throttle=.609,role='new_configuration_validation'))
    hashes=s.read(BASE/'protocol.json')['hashes']
    assert all(s.sha(Path(p))==h for p,h in hashes.items())
    protocol=dict(seeds=SEEDS,design_seed=design_seed,rejected_design_seeds=rejected,normalized_min_distance=.05,nearest_old_distances=dist.min(axis=1).tolist(),minimum_between_new=float(within.min()),bounds=[lo.tolist(),(lo+span).tolist()],frozen_hashes=hashes,reference_budget=3264,short_budget=1800,reference_runs=30,benchmark_runs=210,proximity_rule='Reject complete LHS designs solely on geometry before any controller outcomes; minimum Euclidean distance .05 after scaling each factor by its original range.',seed_check='No overlap with seed fields in archived study and benchmark JSON files')
    for name,value in [('plants.json',plants),('protocol.json',protocol)]:
        p=ROOT/name
        if p.exists():assert s.read(p)==value
        else:save(p,value)
    infer=(OLD/'infer.py').read_text().replace('project = root.parent','project = Path('+repr(str(PROJECT))+')')
    (ROOT/'infer_frozen.py').write_text(infer)
    if not (ROOT/'predictions.json').exists():subprocess.run([str(ref.SKPY),str(ROOT/'infer_frozen.py')],check=True)
    pred=s.read(ROOT/'predictions.json')['rows'];training=[d for d in s.read(ST/'manifest.json')['drones'] if d['role']=='development_train'];cfg=s.b.original.config();bank={}
    for d in training:
        runs=sorted([s.read(p) for p in sorted((ST/'drones'/d['id']/'runs').glob('*.json'))],key=lambda x:x['fitness'])
        bank[d['id']]=[s.b.encode(np.array(r['gains']),cfg).tolist() for r in runs[:5]]
    coords=np.array([[d[k] for k in KEYS] for d in training]);inputs={}
    for d in plants:
        near=[training[i]['id'] for i in np.argsort(np.sum(((coords-[d[k] for k in KEYS])/span)**2,axis=1),kind='stable')[:4]]
        inputs[d['id']]=dict(predictions=pred[d['id']]['vectors'],library=[v for n in near for v in bank[n]],nearest=near,prediction_seconds=pred[d['id']]['seconds'],library_seconds=0)
    save(ROOT/'inputs.json',dict(rows=inputs))
    return plants,inputs
def worker(job):
    s.ROOT=ROOT;s.OLD=ROOT;ref.OUT=ROOT
    if job[2]=='reference':return ref.run_one(job)
    return s.run_one(job)
def main():
    plants,inputs=prepare()
    for methods,total,label in [(['reference'],30,'REFERENCES'),(s.METHODS,210,'BENCHMARK')]:
        jobs=[(d,seed,m,inputs[d['id']]) for d in plants for seed in SEEDS for m in methods]
        def finished(j):return (ROOT/(j[0]['id'] if j[2]=='reference' else 'runs/'+j[0]['id'])/f'{j[2]}_{j[1]}.json').exists()
        jobs=[j for j in jobs if not finished(j)];done=total-len(jobs)
        np.random.default_rng(20261119).shuffle(jobs)
        with ProcessPoolExecutor(max_workers=4) as pool:
            for f in as_completed([pool.submit(worker,j) for j in jobs]):
                last=f.result();done+=1;save(ROOT/'status.json',dict(stage=label,completed=done,total=total,last=last));print(label,done,'/',total,flush=True)
    save(ROOT/'status.json',dict(stage='SIMULATIONS_COMPLETE',references=30,benchmark=210))
if __name__=='__main__':main()
