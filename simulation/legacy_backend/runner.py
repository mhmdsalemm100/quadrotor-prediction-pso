"""Simulation backend launched by Orange. Never trains a predictor."""
from pathlib import Path
import sys, json, time, os, uuid, csv, hashlib, argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
STUDY=ROOT.parent
sys.path.insert(0,str(STUDY))
import pipeline as original
from study import encode,decode,evaluate,write_csv
from model import baseline,run
SEEDS=[28001,28002,28003]
METHODS=['PSO_900','Orange_NN_PSO_900']

def save(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
    for i in range(120):
        try: os.replace(tmp,path); return
        except PermissionError:
            if i==119: raise
            time.sleep(.25)

def optimize(cfg,cases,seed,prediction=None,budget=900,objective=None):
    rng=np.random.default_rng(seed); n=64; x=rng.random((n,8))
    for j in range(8): x[:,j]=(rng.permutation(n)+rng.random(n))/n
    x[0]=encode(baseline(),cfg)
    v=rng.uniform(-.1,.1,(n,8))
    if prediction is not None:
        center=np.clip(prediction,0,1); noise=np.random.default_rng(seed+910000)
        x[1]=center; x[2:33]=np.clip(center+noise.normal(0,.05,(31,8)),0,1)
    initial=x.copy(); initial_velocity=v.copy(); cache={}; ledger=[]; best=np.inf; bestx=None; iteration=0; stalls=0
    personal=x.copy(); pf=np.full(n,np.inf); started=time.perf_counter()
    while len(cache)<budget:
        values=[]; before=len(cache)
        for p in x:
            key=tuple(np.round(p,13))
            if key not in cache:
                if len(cache)>=budget: break
                value=float(objective(p) if objective else evaluate(decode(p,cfg),cases,cfg))
                if not np.isfinite(value): raise ValueError('Nonfinite fitness')
                cache[key]=value
                if value<best: best=value; bestx=p.copy()
                ledger.append([len(cache),iteration,value,best,*p.tolist()])
            values.append(cache[key])
        if len(cache)>=budget: break
        stalls=stalls+1 if len(cache)==before else 0
        if stalls>100: raise RuntimeError('No unique candidates; budget not padded')
        f=np.asarray(values); better=f<pf; personal[better]=x[better]; pf[better]=f[better]
        progress=max(0,min(1,(len(cache)-64)/max(1,budget-64)))
        v=(.9-.5*progress)*v+1.7*rng.random((n,8))*(personal-x)+1.7*rng.random((n,8))*(bestx-x)
        v=np.clip(v,-.2,.2); trial=x+v; hit=(trial<0)|(trial>1)
        x=np.clip(trial,0,1); v[hit]*=-.5; iteration+=1
    result={'seed':seed,'fitness':best,'gains':decode(bestx,cfg).tolist(),'unique_evaluations':len(cache),
            'iterations':iteration,'seconds':time.perf_counter()-started,'initial_sha256':hashlib.sha256(initial.tobytes()).hexdigest()}
    return result,np.asarray(ledger),initial,initial_velocity

def metrics(d,g,cfg):
    value,details=evaluate(np.asarray(g),original.cases_for(d),cfg,True)
    active=[(c,int(np.flatnonzero(c['scenario']['target'])[0])) for c in details]
    _,val=evaluate(np.asarray(g),original.validation_cases(d),cfg,True)
    return {'fitness':value,'training_feasible':all(c['feasible'] for c in details),
        'mean_settling_s':float(np.mean([c['metrics'][j][1] for c,j in active])),
        'worst_settling_s':max(c['metrics'][j][1] for c,j in active),
        'max_overshoot_pct':100*max(c['metrics'][j][2] for c,j in active),
        'local_passes':sum(c['feasible'] for c in val if c['scenario']['group']=='local_uncertainty'),
        'combined_passes':sum(c['feasible'] for c in val if c['scenario']['group']=='combined_stress')}

def worker(d,seed,method,prediction):
    cfg=original.config(); p=prediction if method=='Orange_NN_PSO_900' else None
    result,z,x,v=optimize(cfg,original.cases_for(d),seed,p)
    result.update(drone=d['id'],method=method,**metrics(d,result['gains'],cfg))
    folder=ROOT/'results'/d['id']; folder.mkdir(parents=True,exist_ok=True)
    name=f'{method}_{seed}'
    np.savez_compressed(folder/f'{name}.npz',ledger=z,initial=x,initial_velocity=v)
    save(folder/f'{name}.json',result)
    return result

def report(drones,predictions):
    cfg=original.config(); rows=[]; prediction_only=[]
    for d in drones:
        pred=predictions[d['id']]; prediction_only.append(dict(drone=d['id'],**metrics(d,decode(np.array(pred),cfg),cfg)))
        for seed in SEEDS:
            base=json.loads((STUDY/'drones'/d['id']/'runs'/f'{seed}.json').read_text())
            bm=metrics(d,base['gains'],cfg)
            rows.append(dict(drone=d['id'],seed=seed,method='PSO_3264_reference',evaluations=3264,**bm,settling_ratio_to_reference=1.,fitness_ratio_to_reference=1.,meets_pilot_target=bm['training_feasible']))
            for method in METHODS:
                r=json.loads((ROOT/'results'/d['id']/f'{method}_{seed}.json').read_text())
                sr=r['mean_settling_s']/bm['mean_settling_s']; fr=r['fitness']/bm['fitness']
                ok=r['training_feasible'] and bm['training_feasible'] and sr<=1.05 and r['worst_settling_s']<=1.05*bm['worst_settling_s'] and r['local_passes']>=bm['local_passes'] and r['combined_passes']>=bm['combined_passes']
                rows.append(dict(drone=d['id'],seed=seed,method=method,evaluations=900,**{k:r[k] for k in bm},settling_ratio_to_reference=sr,fitness_ratio_to_reference=fr,meets_pilot_target=ok))
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for method,color in [('PSO_900','#64748b'),('Orange_NN_PSO_900','#2563eb')]:
            for seed in SEEDS:
                z=np.load(ROOT/'results'/d['id']/f'{method}_{seed}.npz')['ledger']
                axes[0].plot(z[:,0],z[:,3],color=color,alpha=.65,label=method if seed==SEEDS[0] else None)
        axes[0].set(yscale='log',xlabel='Unique evaluations',ylabel='Best constrained fitness'); axes[0].legend(fontsize=8)
        subset=[r for r in rows if r['drone']==d['id']]; methods=['PSO_3264_reference']+METHODS
        axes[1].bar(np.arange(3),[np.median([r['mean_settling_s'] for r in subset if r['method']==m]) for m in methods],color=['#059669','#64748b','#2563eb'])
        axes[1].set(xticks=np.arange(3),xticklabels=['PSO 3264','PSO 900','Orange NN + PSO 900'],ylabel='Median mean settling time (s)')
        fig.suptitle(d['id']+' — development pilot, 3 paired seeds'); fig.tight_layout(); fig.savefig(ROOT/'results'/d['id']/'comparison.png',dpi=160); plt.close(fig)
    write_csv(ROOT/'comparison.csv',rows); write_csv(ROOT/'prediction_only.csv',prediction_only)
    summary=[]
    for method in ['PSO_3264_reference']+METHODS:
        rr=[r for r in rows if r['method']==method]
        summary.append({'method':method,'runs':len(rr),'training_feasible':sum(r['training_feasible'] for r in rr),
            'meets_pilot_target':sum(r['meets_pilot_target'] for r in rr),'median_settling_ratio':float(np.median([r['settling_ratio_to_reference'] for r in rr])),
            'mean_local_passes_out_of_60':float(np.mean([r['local_passes'] for r in rr])),'candidate_reduction_pct':0 if method=='PSO_3264_reference' else 100*(1-900/3264)})
    write_csv(ROOT/'summary.csv',summary)
    fig,ax=plt.subplots(figsize=(9,5)); ax.bar(['PSO 900','Orange NN + PSO 900'],[summary[1]['meets_pilot_target'],summary[2]['meets_pilot_target']],color=['#64748b','#2563eb'])
    ax.set(ylim=(0,32),ylabel='Paired runs meeting the pilot target / 30',title='900-evaluation test: quality must match the full-budget reference'); fig.tight_layout(); fig.savefig(ROOT/'pilot_outcome.png',dpi=180); plt.close(fig)
    save(ROOT/'STATUS.json',{'status':'COMPLETE','completed_runs':60,'total_runs':60,'summary':summary,
         'scope':'Development pilot only; 10 previously inspected validation configurations, 3 seeds. No final unseen test claim.'})

def main():
    import msvcrt
    lock=open(ROOT/'runner.lock','a+b'); lock.write(b'0'); lock.flush(); lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    original.verify_contract()
    cfg=original.config(); bundle=json.loads((ROOT/'orange_predictions.json').read_text())
    phash=hashlib.sha256((ROOT/'orange_predictions.json').read_bytes()).hexdigest()
    contract=ROOT/'pilot_contract.json'
    if contract.exists(): assert json.loads(contract.read_text())['prediction_sha256']==phash,'Predictions changed; use new experiment folder'
    else: save(contract,{'prediction_sha256':phash,'seeds':SEEDS,'budget':900,'sigma':.05,'initializer':'1 baseline, 1 prediction, 31 perturbed predictions, 31 global particles',
        'schedule':'0.9 to 0.4 over each method evaluation budget','target':'training feasible; mean and worst settling <=1.05x paired reference; no fewer local/combined validation passes',
        'scope':'Pilot on development validation only','code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    pred={r['drone']:r['clipped_unit_gains'] for r in bundle['predictions']}
    drones=[d for d in json.loads((STUDY/'manifest.json').read_text())['drones'] if d['role']=='development_validation']
    assert set(pred)=={d['id'] for d in drones}
    jobs=[]; done=0
    for d in drones:
        for seed in SEEDS:
            for method in METHODS:
                f=ROOT/'results'/d['id']/f'{method}_{seed}.json'
                if f.exists():
                    r=json.loads(f.read_text()); assert r['unique_evaluations']==900; done+=1
                else: jobs.append((d,seed,method,pred[d['id']]))
    save(ROOT/'STATUS.json',{'status':'RUNNING','completed_runs':done,'total_runs':60})
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(worker,*j) for j in jobs]
        for future in as_completed(futures):
            r=future.result(); done+=1
            print(f"{done}/60 {r['drone']} {r['method']} seed={r['seed']} evaluations={r['unique_evaluations']} fitness={r['fitness']:.8g}",flush=True)
            save(ROOT/'STATUS.json',{'status':'RUNNING','completed_runs':done,'total_runs':60,'last_result':r})
    report(drones,pred)

def tests():
    cfg=original.config(); objective=lambda x:float(np.sum((x-.35)**2))
    for budget in [130,900]:
        a,z,x,v=optimize(cfg,[],28001,budget=budget,objective=objective)
        b,zz,xx,vv=original.optimize(cfg,[],28001,budget=budget,objective=objective)
        assert np.array_equal(z,zz) and np.array_equal(x,xx) and np.array_equal(v,vv)
        c,w,wx,wv=optimize(cfg,[],28001,np.full(8,.35),budget,objective)
        assert c['unique_evaluations']==budget and len(np.unique(np.round(w[:,4:],13),axis=0))==budget
        assert np.array_equal(v,wv) and np.allclose(wx[1],.35)
        assert np.all(np.diff(w[:,3])<=0)
    print('PASS: unchanged standard PSO trajectory; exact 900 and partial budgets; paired velocities; seeded predictions; unique evaluations.')

if __name__=='__main__':
    try:
        if '--test' in sys.argv: tests()
        else: main()
    except Exception:
        import traceback
        if '--test' not in sys.argv: save(ROOT/'STATUS.json',{'status':'FAILED','traceback':traceback.format_exc()})
        raise
