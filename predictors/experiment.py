"""Training-only model selection and frozen 900-evaluation development pilot."""
from pathlib import Path
import sys,json,time,hashlib,csv,html
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
ROOT=Path(__file__).resolve().parent; STUDY=ROOT.parent
sys.path.insert(0,str(STUDY/'Orange_900_Pilot'))
import runner as backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
SEEDS=[28001,28002,28003]
def save(name,value): backend.save(ROOT/name,value)
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def score_candidate(c,ids,plants):
    cfg=backend.original.config(); rows=[]
    for name,u in zip(ids,c['oof']):
        g=backend.decode(np.array(u),cfg)
        f,detail=backend.evaluate(g,backend.original.cases_for(plants[name]),cfg,True)
        reference=min(json.loads(p.read_text())['fitness'] for p in (STUDY/'drones'/name/'runs').glob('*.json'))
        rows.append({'drone':name,'fitness':f,'feasible':all(x['feasible'] for x in detail),'ratio':f/reference})
    # Prespecified feasible-rate first, then mean log fitness ratio. No validation labels used.
    return {'name':c['name'],'neural':c['neural'],'feasible_count':sum(r['feasible'] for r in rows),'mean_log_ratio':float(np.mean(np.log([r['ratio'] for r in rows]))),'rows':rows}
def run_job(d,seed,name,u):
    cfg=backend.original.config()
    result,z,x,v=backend.optimize(cfg,backend.original.cases_for(d),seed,np.array(u),budget=900)
    result.update(drone=d['id'],method=name,**backend.metrics(d,result['gains'],cfg))
    folder=ROOT/'results'/d['id']; folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/f'{name}_{seed}.npz',ledger=z,initial=x,initial_velocity=v)
    backend.save(folder/f'{name}_{seed}.json',result)
    return result
def report(selected,bundle,plants):
    cfg=backend.original.config(); names=[p['name'] for p in cfg['parameters']]; rows=[]; gains=[]; predrows=[]; audit=[]
    for name in selected:
        candidate=next(c for c in bundle['candidates'] if c['name']==name)
        for drone,u in zip(bundle['validation_ids'],candidate['predictions']):
            g=backend.decode(np.array(u),cfg); pm=backend.metrics(plants[drone],g,cfg)
            predrows.append(dict(drone=drone,model=name,**dict(zip(names,g)),**pm))
            for seed in SEEDS:
                path=ROOT/'results'/drone/f'{name}_{seed}.json'; r=json.loads(path.read_text()); z=np.load(path.with_suffix('.npz'))
                ledger=z['ledger']; assert ledger.shape==(900,12) and np.isfinite(ledger).all()
                assert len(np.unique(np.round(ledger[:,4:],13),axis=0))==900
                assert np.array_equal(ledger[:,0],np.arange(1,901))
                assert np.array_equal(ledger[:,3],np.minimum.accumulate(ledger[:,2]))
                best=int(np.argmin(ledger[:,2])); assert np.allclose(r['gains'],backend.decode(ledger[best,4:],cfg))
                assert np.array_equal(z['initial'][1],np.array(u))
                fresh=backend.metrics(plants[drone],r['gains'],cfg)
                assert all(np.isclose(fresh[k],r[k],rtol=1e-10,atol=1e-10) for k in fresh)
                old=json.loads((STUDY/'drones'/drone/'runs'/f'{seed}.json').read_text()); base=backend.metrics(plants[drone],old['gains'],cfg)
                ratio=r['mean_settling_s']/base['mean_settling_s']
                ok=bool(r['training_feasible'] and base['training_feasible'] and ratio<=1.05 and r['worst_settling_s']<=1.05*base['worst_settling_s'] and r['local_passes']>=base['local_passes'] and r['combined_passes']>=base['combined_passes'])
                rows.append(dict(drone=drone,seed=seed,model=name,evaluations=900,**fresh,settling_ratio_to_reference=ratio,meets_target=ok))
                gains.append(dict(drone=drone,seed=seed,model=name,**dict(zip(names,r['gains'])),meets_target=ok))
                audit.append({'drone':drone,'seed':seed,'model':name,'unique_candidates':900,'verified':True})
    with (STUDY/'neural_network_preparation'/'development_validation.csv').open(newline='') as f:
        truth={r['drone']:r for r in csv.DictReader(f)}
    errors=[]
    fig,axes=plt.subplots(2,4,figsize=(15,7))
    for n in selected:
        candidate=next(c for c in bundle['candidates'] if c['name']==n)
        U=np.array(candidate['predictions']); T=np.array([[float(truth[d]['unit_'+k]) for k in names] for d in bundle['validation_ids']])
        G=backend.decode(U,cfg); target=backend.decode(T,cfg)
        for j,k in enumerate(names):
            errors.append({'model':n,'gain':k,'unit_MAE':float(np.mean(abs(U[:,j]-T[:,j]))),'unit_RMSE':float(np.sqrt(np.mean((U[:,j]-T[:,j])**2))),'physical_MAE':float(np.mean(abs(G[:,j]-target[:,j]))),'physical_MAPE_pct':float(100*np.mean(abs(G[:,j]-target[:,j])/abs(target[:,j])))})
            axes.flat[j].scatter(T[:,j],U[:,j],label=n,alpha=.7)
    for j,a in enumerate(axes.flat):
        a.plot([0,1],[0,1],'k--',lw=1); a.set(title=names[j],xlabel='Reference unit gain',ylabel='Predicted unit gain',xlim=(0,1),ylim=(0,1)); a.title.set_fontsize(9)
    axes.flat[0].legend(fontsize=6); fig.tight_layout(); fig.savefig(ROOT/'prediction_accuracy.png',dpi=160); plt.close(fig)
    for fn,data in [('comparison.csv',rows),('final_gains.csv',gains),('predicted_gains.csv',predrows),('prediction_errors.csv',errors),('audit.csv',audit)]: backend.write_csv(ROOT/fn,data)
    summary=[{'model':n,'runs':30,'passed':sum(r['meets_target'] for r in rows if r['model']==n),'feasible':sum(r['training_feasible'] for r in rows if r['model']==n),'median_settling_ratio':float(np.median([r['settling_ratio_to_reference'] for r in rows if r['model']==n]))} for n in selected]
    backend.write_csv(ROOT/'summary.csv',summary)
    fig,ax=plt.subplots(1,2,figsize=(12,5))
    labels=['Ordinary PSO','Orange NN']+selected
    ax[0].bar(range(len(labels)),[0,7]+[s['passed'] for s in summary]); ax[0].set(ylim=(0,32),ylabel='Runs meeting unchanged target / 30',xticks=range(len(labels)),xticklabels=labels)
    ax[0].tick_params(axis='x',labelrotation=25)
    for n in selected: ax[1].plot(range(1,31),[r['settling_ratio_to_reference'] for r in rows if r['model']==n],'.',label=n)
    ax[1].axhline(1.05,color='red',ls='--',label='5% settling threshold'); ax[1].set(xlabel='Paired run',ylabel='Mean settling / full PSO reference'); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(ROOT/'performance.png',dpi=180); plt.close(fig)
    for drone in bundle['validation_ids']:
        fig,ax=plt.subplots(figsize=(9,4))
        for n in selected:
            for seed in SEEDS:
                z=np.load(ROOT/'results'/drone/f'{n}_{seed}.npz')['ledger']; ax.plot(z[:,0],z[:,3],label=n if seed==SEEDS[0] else None,alpha=.7)
        ax.set(yscale='log',xlabel='Unique candidate evaluations',ylabel='Best constrained fitness',title=drone); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(ROOT/'results'/drone/'convergence.png',dpi=150); plt.close(fig)
    body=''.join(f"<tr><td>{s['model']}</td><td>{s['passed']}/30</td><td>{s['feasible']}/30</td><td>{100*(s['median_settling_ratio']-1):.2f}%</td></tr>" for s in summary)
    (ROOT/'REPORT.html').write_text('<!doctype html><meta charset="utf-8"><title>Python gain prediction pilot</title><style>body{font:17px Arial;max-width:1100px;margin:40px auto;line-height:1.5}td,th{padding:12px;border:1px solid #ccc}table{border-collapse:collapse}img{max-width:100%}</style><h1>Python gain prediction: 900-evaluation pilot</h1><p>Models selected by five-fold held-out simulation performance using only 20 training drones. Two finalists are frozen before this development-validation test. Each neural predictor is a three-seed ensemble. No validation gains entered training or model selection.</p><table><tr><th>Model</th><th>Target passed</th><th>Training feasible</th><th>Median settling increase</th></tr>'+body+'</table><p>Target: feasible; mean and worst settling within 5% of same-seed full PSO; no fewer local or combined validation passes. Combined-axis failures remain a limitation, including in the full-budget reference. This is not flight validation or a final unseen-data test. 900 versus 3264 is a 72.4% evaluation reduction, not total computational cost savings.</p><img src="performance.png"><p><a href="final_gains.csv">Final controller gains</a> · <a href="predicted_gains.csv">Initial predicted gains and performance</a> · <a href="comparison.csv">Every paired result</a> · <a href="selection.json">Training-only model selection</a></p>',encoding='utf-8')
    save('STATUS.json',{'status':'COMPLETE','completed_runs':len(rows),'total_runs':len(rows),'summary':summary,'audit':'PASS','scope':'Development pilot; 10 configurations x 3 seeds, not 30 independent drones'})
    print(json.dumps(summary,indent=2),flush=True)
def main():
    import msvcrt
    lock=open(ROOT/'experiment.lock','a+b'); lock.write(b'0'); lock.flush(); lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    backend.original.verify_contract(); backend.tests()
    bundle=json.loads((ROOT/'candidate_predictions.json').read_text()); plants={d['id']:d for d in json.loads((STUDY/'manifest.json').read_text())['drones']}
    for source,h in bundle['training_source_sha256'].items(): assert sha(STUDY/source)==h
    for c in bundle['candidates']:
        assert np.asarray(c['oof']).shape==(20,8) and np.asarray(c['predictions']).shape==(10,8)
        assert np.isfinite(c['oof']).all() and np.isfinite(c['predictions']).all()
    if not (ROOT/'selection.json').exists():
        scored=[]
        with ProcessPoolExecutor(max_workers=4) as pool:
            for s in pool.map(score_candidate,bundle['candidates'],[bundle['training_ids']]*len(bundle['candidates']),[plants]*len(bundle['candidates'])):
                scored.append(s); print('CV '+s['name']+' feasible '+str(s['feasible_count'])+'/20',flush=True)
        scored.sort(key=lambda s:(-s['feasible_count'],s['mean_log_ratio']))
        nn=next(s['name'] for s in scored if s['neural']); other=next(s['name'] for s in scored if s['name']!=nn)
        save('selection.json',{'selected':[nn,other],'ranking':scored,'rule':'Highest held-out training feasibility, then lowest mean log fitness ratio; best NN plus strongest remaining model','prediction_sha256':sha(ROOT/'candidate_predictions.json'),'train_sha256':sha(ROOT/'train.py'),'experiment_sha256':sha(Path(__file__))})
    sel=json.loads((ROOT/'selection.json').read_text()); assert sel['prediction_sha256']==sha(ROOT/'candidate_predictions.json')
    assert sel['experiment_sha256']==sha(Path(__file__)) and sel['train_sha256']==sha(ROOT/'train.py')
    selected=sel['selected']; jobs=[]; done=0
    for name in selected:
        c=next(c for c in bundle['candidates'] if c['name']==name)
        for d,u in zip(bundle['validation_ids'],c['predictions']):
            for seed in SEEDS:
                if (ROOT/'results'/d/f'{name}_{seed}.json').exists(): done+=1
                else: jobs.append((plants[d],seed,name,u))
    save('STATUS.json',{'status':'RUNNING','completed_runs':done,'total_runs':60,'selected':selected})
    with ProcessPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(run_job,*j) for j in jobs]):
            r=f.result(); done+=1; print(f"{done}/60 {r['drone']} {r['method']} seed {r['seed']} fitness {r['fitness']:.7g}",flush=True)
            save('STATUS.json',{'status':'RUNNING','completed_runs':done,'total_runs':60,'selected':selected})
    report(selected,bundle,plants)
if __name__=='__main__':
    try: main()
    except Exception:
        import traceback
        save('STATUS.json',{'status':'FAILED','traceback':traceback.format_exc()}); raise
