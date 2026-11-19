"""Fresh PSO-only experiment. No previous result files are read."""
from pathlib import Path
import argparse, csv, hashlib, json, time, sys, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.stats import qmc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import NOMINAL, baseline, coefficients, run
from study import encode, decode, evaluate, training_scenarios, metric_rows, write_csv

ROOT=Path(__file__).resolve().parent
SEEDS=list(range(28001,28031))
NAMES=['Jxx','Jyy','Jzz','radius','Ct','Cm','damping_x','damping_y','damping_z','rotor_inertia','motor_tau','motor_slope','motor_offset']
def save(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8'); tmp.replace(path)
def config(): return json.loads((ROOT/'config.json').read_text())
def cases_for(drone):
    return [dict(s,plant=drone['plant'],throttle=.609) for s in training_scenarios()]
def optimize(cfg,cases,seed,budget=3264,objective=None):
    rng=np.random.default_rng(seed); n=64; x=rng.random((n,8))
    for j in range(8): x[:,j]=(rng.permutation(n)+rng.random(n))/n
    x[0]=encode(baseline(),cfg); initial=x.copy()
    v=rng.uniform(-.1,.1,(n,8)); initial_v=v.copy()
    cache={}; ledger=[]; best=np.inf; bestx=None; iteration=0; stalls=0
    personal=x.copy(); pf=np.full(n,np.inf)
    started=time.perf_counter()
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
        if stalls>100: raise RuntimeError('Search stalled; budget not padded')
        f=np.asarray(values); improve=f<pf
        personal[improve]=x[improve]; pf[improve]=f[improve]
        progress=max(0,min(1,(len(cache)-64)/max(1,budget-64)))
        inertia=.9-.5*progress
        v=inertia*v+1.7*rng.random((n,8))*(personal-x)+1.7*rng.random((n,8))*(bestx-x)
        v=np.clip(v,-.2,.2); trial=x+v; hit=(trial<0)|(trial>1)
        x=np.clip(trial,0,1); v[hit]*=-.5; iteration+=1
    return {'seed':seed,'fitness':best,'gains':decode(bestx,cfg).tolist(),
            'unique_evaluations':len(cache),'iterations':iteration,'seconds':time.perf_counter()-started,
            'initial_sha256':hashlib.sha256(initial.tobytes()).hexdigest()},np.asarray(ledger),initial,initial_v

def prepare():
    if (ROOT/'manifest.json').exists(): raise RuntimeError('Already prepared; refusing to regenerate configurations')
    cfg=config(); cfg.update(study_name='Fresh healthy-configuration PSO study',seeds=SEEDS,workers=4)
    save(ROOT/'config.json',cfg)
    units=qmc.LatinHypercube(3,seed=20261028).random(30)
    drones=[]
    for i,u in enumerate([np.array([.5,.5,.5]),*units]):
        sj=.85+.3*u[0]; tau=.014+.012*u[1]; sr=.9+.2*u[2]
        p=NOMINAL.copy(); p[:3]*=sj; p[10]=tau; p[4:6]*=sr
        d={'id':f'Drone_{i:02d}','role':'reference' if i==0 else ('development_train' if i<=20 else 'development_validation'),
           'inertia_scale':float(sj),'motor_tau_s':float(tau),'rotor_scale':float(sr),'plant':p.tolist(),
           'source_mass_kg':1.4,'throttle':.609,'parameters':dict(zip(NAMES,p.tolist()))}
        drones.append(d); folder=ROOT/'drones'/d['id']; folder.mkdir(parents=True)
        save(folder/'specification.json',d)
        write_csv(folder/'parameters.csv',[{'parameter':k,'value':v} for k,v in d['parameters'].items()])
        (folder/'README.md').write_text(f"# {d['id']}\n\nRole: {d['role']}.\n\nInertia scale: {sj:.8f}; motor time constant: {tau:.8f} s; rotor coefficient scale: {sr:.8f}.\n\nSee specification.json for exact plant values. All 30 runs start fresh; no previous tuned gains.\n\nAfter completion: summary.csv contains every run; selected_gains.json is selected by training fitness only; figures/ contains convergence, gain variation, response/motor plots and validation. Validation is for the selected controller only.\n",encoding='utf-8')
    save(ROOT/'manifest.json',{'design_seed':20261028,'seeds':SEEDS,'drones':drones,'total_runs':930,'budget_per_run':3264,
         'training_cases_per_evaluation':6,'total_candidate_evaluations':3035520,'total_training_simulations':18213120,
         'neural_network':'NOT STARTED','future_test_configurations':'Not generated; must remain unseen'})
    write_csv(ROOT/'DRONE_INDEX.csv',[{k:v for k,v in d.items() if k not in ['plant','parameters']} for d in drones])
    fig,ax=plt.subplots(1,3,figsize=(13,4))
    for a,(x,y) in zip(ax,[('inertia_scale','motor_tau_s'),('inertia_scale','rotor_scale'),('motor_tau_s','rotor_scale')]):
        for d in drones:
            a.scatter(d[x],d[y],c='black' if d['role']=='reference' else ('#2563eb' if d['role']=='development_train' else '#d97706'))
            a.annotate(d['id'][-2:],(d[x],d[y]),fontsize=6)
        a.set(xlabel=x,ylabel=y); a.grid(alpha=.2)
    fig.suptitle('Fresh configuration design: black reference, blue training, orange development validation')
    fig.tight_layout(); fig.savefig(ROOT/'configuration_design.png',dpi=180); plt.close(fig)
    save(ROOT/'contract.json',{'manifest_sha256':hashfile(ROOT/'manifest.json'),'code_hashes':{p.name:hashfile(p) for p in ROOT.glob('*.py')},'config_sha256':hashfile(ROOT/'config.json')})
def hashfile(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify_contract():
    c=json.loads((ROOT/'contract.json').read_text())
    assert hashfile(ROOT/'manifest.json')==c['manifest_sha256']
    assert hashfile(ROOT/'config.json')==c['config_sha256']
    for n,h in c['code_hashes'].items(): assert hashfile(ROOT/n)==h, f'Changed code: {n}'

def worker(d,seed):
    folder=ROOT/'drones'/d['id']/'runs'; folder.mkdir(exist_ok=True)
    result,ledger,initial,velocity=optimize(config(),cases_for(d),seed)
    result['drone']=d['id']; result['coefficients']=coefficients(result['gains'])
    np.savez_compressed(folder/f'{seed}_ledger.npz',ledger=ledger,initial=initial,initial_velocity=velocity)
    save(folder/f'{seed}.json',result)
    return result

def validation_cases(d):
    rng=np.random.default_rng(281000+int(d['id'][-2:])); p=np.array(d['plant']); cases=[]
    for i in range(60):
        j=i%3; target=[0.,0.,0.]; target[j]=float(rng.choice([-1,1])*rng.uniform(5 if j<2 else 20,20 if j<2 else 100))
        q=p.copy(); q[:3]*=rng.uniform(.97,1.03); q[10]*=rng.uniform(.95,1.05); q[4:6]*=rng.uniform(.97,1.03)
        cases.append({'name':f'local_uncertainty_{i:02d}','target':target,'plant':q.tolist(),'group':'local_uncertainty'})
    for i in range(12):
        target=(rng.choice([-1,1],3)*rng.uniform([5,5,20],[18,15,70])).tolist()
        cases.append({'name':f'combined_{i:02d}','target':target,'plant':p.tolist(),'group':'combined_stress'})
    return cases

def report_drone(d):
    cfg=config(); folder=ROOT/'drones'/d['id']; figures=folder/'figures'; figures.mkdir(exist_ok=True)
    results=[json.loads((folder/'runs'/f'{s}.json').read_text()) for s in SEEDS]
    write_csv(folder/'summary.csv',[{k:v for k,v in r.items() if k not in ['gains','coefficients']} for r in results])
    best=min(results,key=lambda r:r['fitness']); save(folder/'selected_gains.json',best)
    write_csv(folder/'all_gains.csv',[dict(seed=r['seed'],fitness=r['fitness'],**{p['name']:v for p,v in zip(cfg['parameters'],r['gains'])}) for r in results])
    fig,ax=plt.subplots(figsize=(9,5))
    for r in results:
        z=np.load(folder/'runs'/f"{r['seed']}_ledger.npz")['ledger']; ax.plot(z[:,0],z[:,3],alpha=.4,lw=.8)
    ax.set(xlabel='Unique candidate evaluations',ylabel='Best constrained fitness',yscale='log',title=f"{d['id']} — all 30 independent runs")
    ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(figures/'convergence_all_runs.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5)); ax.boxplot(np.array([r['gains'] for r in results]),tick_labels=[p['name'].replace('_','\n') for p in cfg['parameters']])
    ax.set(yscale='log',ylabel='Gain value (log scale)',title=f"{d['id']} — gain variability over 30 runs"); fig.tight_layout(); fig.savefig(figures/'gain_distribution.png',dpi=180); plt.close(fig)
    metric_data=[]
    for label,g in [('source_baseline',baseline()),('selected_PSO',best['gains'])]:
        rows,_=metric_rows(np.asarray(g),cases_for(d),cfg,label); metric_data.extend(rows)
    write_csv(folder/'training_metrics.csv',metric_data)
    for scenario in cases_for(d):
        fig,axes=plt.subplots(3,1,figsize=(9,9),sharex=True); j=int(np.flatnonzero(scenario['target'])[0]); output_col=[1,2,6][j]
        for label,g,color in [('Original gains',baseline(),'#64748b'),('Selected PSO',best['gains'],'#2563eb')]:
            _,_,trace=run(np.asarray(g),scenario,cfg,store=True); t=trace[:,0]-.2
            axes[0].plot(t,np.rad2deg(trace[:,output_col]),label=label,color=color)
            if label=='Selected PSO':
                np.savetxt(folder/f"trace_{scenario['name']}.csv",trace,delimiter=',',header='t,roll,pitch,yaw,p,q,r,ref_roll,ref_pitch,ref_yaw_rate,desired_p,desired_q,desired_r,raw_roll,raw_pitch,raw_yaw,control_roll,control_pitch,control_yaw,raw_motor1,raw_motor2,raw_motor3,raw_motor4,motor1,motor2,motor3,motor4,min_motor_speed,omega1,omega2,omega3,omega4,qw,qx,qy,qz',comments='')
                for k in range(4):
                    axes[1].plot(t,trace[:,23+k],label=f'Motor {k+1}'); axes[2].plot(t,trace[:,28+k],label=f'Motor {k+1}')
        target=scenario['target'][j]; axes[0].plot(t,np.where(t>=0,target,0),'k--',label='Command')
        axes[0].set_ylabel('Yaw rate (deg/s)' if j==2 else 'Angle (deg)'); axes[0].legend()
        axes[1].set_ylabel('Motor command'); axes[1].legend(ncol=4)
        axes[2].set(ylabel='Motor speed (rad/s)',xlabel='Time after command (s)')
        for ax in axes: ax.grid(alpha=.2)
        fig.suptitle(f"{d['id']} — {scenario['name']} — seed {best['seed']}"); fig.tight_layout(); fig.savefig(figures/f"response_{scenario['name']}.png",dpi=180); plt.close(fig)
    val=validation_cases(d); save(folder/'validation_scenarios.json',val); outcomes=[]
    for label,g in [('source_baseline',baseline()),('selected_PSO',best['gains'])]:
        _,details=evaluate(np.asarray(g),val,cfg,True)
        for c in details: outcomes.append({'controller':label,'case':c['scenario']['name'],'group':c['scenario']['group'],'feasible':c['feasible'],'violation':c['violation']})
    write_csv(folder/'validation.csv',outcomes)
    fig,ax=plt.subplots(figsize=(8,4)); groups=['local_uncertainty','combined_stress']; xx=np.arange(2)
    for off,label in [(-.18,'source_baseline'),(.18,'selected_PSO')]:
        vals=[sum(r['feasible'] for r in outcomes if r['controller']==label and r['group']==g)/sum(1 for r in outcomes if r['controller']==label and r['group']==g)*100 for g in groups]
        ax.bar(xx+off,vals,width=.36,label=label)
    ax.set(xticks=xx,xticklabels=groups,ylabel='Cases passing all requirements (%)',ylim=(0,105),title=f"{d['id']} — selected-controller validation"); ax.legend(); fig.tight_layout(); fig.savefig(figures/'validation.png',dpi=180); plt.close(fig)
    save(folder/'COMPLETE.json',{'runs':30,'selected_seed':best['seed'],'validation_scope':'selected gains and source baseline only','figures':len(list(figures.glob('*.png')))})

def execute(workers):
    verify_contract(); manifest=json.loads((ROOT/'manifest.json').read_text()); drones=manifest['drones']; started=time.time()
    jobs=[]; completed=0
    for d in drones:
        for s in SEEDS:
            file=ROOT/'drones'/d['id']/'runs'/f'{s}.json'
            if file.exists():
                r=json.loads(file.read_text()); assert r['unique_evaluations']==3264
                assert file.with_name(f'{s}_ledger.npz').exists(); completed+=1
            else: jobs.append((d,s))
    save(ROOT/'STATUS.json',{'status':'RUNNING','completed_runs':completed,'total_runs':930,'workers':workers})
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures={pool.submit(worker,d,s):(d,s) for d,s in jobs}
            for f in as_completed(futures):
                r=f.result(); completed+=1
                print(f"{completed}/930 {r['drone']} seed={r['seed']} fitness={r['fitness']:.8g} seconds={r['seconds']:.1f}",flush=True)
                save(ROOT/'STATUS.json',{'status':'RUNNING','completed_runs':completed,'total_runs':930,'last_completed':r['drone'],'elapsed_this_session_s':time.time()-started})
                d,_=futures[f]; folder=ROOT/'drones'/d['id']
                if all((folder/'runs'/f'{s}.json').exists() for s in SEEDS) and not (folder/'COMPLETE.json').exists(): report_drone(d)
        for d in drones:
            if not (ROOT/'drones'/d['id']/'COMPLETE.json').exists(): report_drone(d)
        overview=[]
        for d in drones:
            folder=ROOT/'drones'/d['id']; rows=list(csv.DictReader((folder/'summary.csv').open()))
            chosen=json.loads((folder/'selected_gains.json').read_text())
            overview.append({'drone':d['id'],'role':d['role'],'best_fitness':chosen['fitness'],
                'median_fitness':float(np.median([float(r['fitness']) for r in rows])),
                'feasible_runs':sum(float(r['fitness'])<1000 for r in rows),'selected_seed':chosen['seed']})
        write_csv(ROOT/'STUDY_SUMMARY.csv',overview)
        fig,ax=plt.subplots(figsize=(13,5)); xx=np.arange(len(overview))
        ax.plot(xx,[r['median_fitness'] for r in overview],'o-',label='Median of 30 runs')
        ax.plot(xx,[r['best_fitness'] for r in overview],'o-',label='Best of 30 runs')
        ax.set(xticks=xx,xticklabels=[r['drone'][-2:] for r in overview],xlabel='Drone ID',ylabel='Constrained fitness',yscale='log',title='Fresh PSO study: training performance by configuration')
        ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(ROOT/'study_fitness.png',dpi=180); plt.close(fig)
        html='<html><head><meta charset="utf-8"><title>Fresh PSO study</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto;color:#16324f}img{max-width:100%}table{border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #ddd}</style></head><body><h1>Fresh PSO-only study</h1><p>31 configurations; 930 fresh runs. Neural-network work not started. See SPECIFICATION.md for assumptions and constraints.</p><img src="configuration_design.png"><img src="study_fitness.png"><table><tr><th>Drone</th><th>Best fitness</th><th>Median fitness</th><th>Feasible runs</th><th>Figures</th></tr>'
        for r in overview:
            name=r['drone']; html+=f'<tr><td>{name}</td><td>{r["best_fitness"]:.6g}</td><td>{r["median_fitness"]:.6g}</td><td>{r["feasible_runs"]}/30</td><td><a href="drones/{name}/figures/convergence_all_runs.png">Convergence</a> | <a href="drones/{name}/figures/validation.png">Validation</a></td></tr>'
        (ROOT/'REPORT.html').write_text(html+'</table><p>Validation plots describe training-selected controllers only. Simulation results are not hardware validation.</p></body></html>',encoding='utf-8')
        save(ROOT/'STATUS.json',{'status':'COMPLETE','completed_runs':930,'neural_network':'NOT STARTED'})
    except Exception:
        save(ROOT/'STATUS.json',{'status':'FAILED','completed_runs':completed,'traceback':traceback.format_exc()}); raise

def test():
    cfg=config(); objective=lambda x:float(np.sum((x-.35)**2))
    a,z,x,v=optimize(cfg,[],73,130,objective); b,zz,_,_=optimize(cfg,[],73,130,objective)
    assert a['unique_evaluations']==130 and np.array_equal(z,zz)
    assert np.all(np.diff(z[:,3])<=0) and len({tuple(np.round(q,13)) for q in z[:,4:]})==130
    assert np.allclose(x[0],encode(baseline(),cfg)) and np.max(np.abs(v))<=.1
    assert np.allclose(decode(encode(baseline(),cfg),cfg),baseline())
    cases=training_scenarios(); first=evaluate(baseline(),cases,cfg); second=evaluate(baseline(),cases,cfg)
    assert np.isfinite(first) and first==second
    print(json.dumps({'tests':'PASS','exact_partial_budget':130,'deterministic':True,'source_baseline_fitness':first}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('action',choices=['prepare','test','run']); ap.add_argument('--workers',type=int,default=4); args=ap.parse_args()
    if args.action=='prepare': prepare()
    elif args.action=='test': test()
    else: execute(args.workers)
