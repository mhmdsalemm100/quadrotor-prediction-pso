"""Tables A-D and figures B-C. Run only after all 210 outcomes exist."""
import itertools, math
import numpy as np
from scipy.stats import wilcoxon, rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import study_extension as s

def holm(p):
    order=np.argsort(p); adjusted=np.empty(len(p)); prev=0.
    for i,j in enumerate(order):
        prev=max(prev,min(1.,(len(p)-i)*p[j])); adjusted[j]=prev
    return adjusted

def main():
    plants,inputs=s.prepare(); results=[]; trajectories={m:[] for m in s.METHODS}; finals={m:[] for m in s.METHODS}
    allrows=[]
    for d in plants:
        for seed in s.SEEDS:
            for m in s.METHODS:
                stem=s.ROOT/'runs'/d['id']/f'{m}_{seed}'
                r=s.read(stem.with_suffix('.json')); z=np.load(stem.with_suffix('.npz'))['ledger']
                assert z.shape==(1800,12) and np.array_equal(z[:,3],np.minimum.accumulate(z[:,2]))
                trajectories[m].append(z[:,3]); finals[m].append(z[-1,3]); results.append(r)
                for cp in r['checkpoints']:
                    allrows.append(dict(method=m,plant=d['id'],seed=seed,**{k:v for k,v in cp.items() if k!='gains'}))
    assert len(results)==210
    s.csvout('all_checkpoint_metrics.csv',allrows)
    s.csvout('per_run.csv',[dict(method=r['method'],plant=r['plant'],seed=r['seed'],
        best_cost=r['checkpoints'][-1]['fitness'],passed=r['checkpoints'][-1]['passed'],
        evaluations=1800,optimizer_seconds=r['optimizer_seconds'],validation_seconds=r['validation_seconds']) for r in results])
    with (s.ROOT/'best_cost_histories.csv').open('w',newline='',encoding='utf-8') as f:
        import csv
        w=csv.writer(f); w.writerow(['method','plant','seed','evaluations','best_cost'])
        for m in s.METHODS:
            for i,trajectory in enumerate(trajectories[m]):
                w.writerows((m,plants[i//3]['id'],s.SEEDS[i%3],n,float(cost)) for n,cost in enumerate(trajectory,1))
    s.csvout('final_gains.csv',[dict(method=r['method'],plant=r['plant'],seed=r['seed'],
        **dict(zip([p['name'] for p in s.b.original.config()['parameters']],r['checkpoints'][-1]['gains']))) for r in results])
    tablea=[]; tableb=[]
    for m in s.METHODS:
        q=np.quantile(finals[m],[.25,.5,.75])
        tablea.append(dict(method=m,passes=sum(r['passed'] for r in allrows if r['method']==m and r['evaluations']==1800),runs=30,median_best_cost=q[1],q25=q[0],q75=q[2],IQR=q[2]-q[0]))
        for budget in s.CHECKPOINTS:
            rows=[r for r in allrows if r['method']==m and r['evaluations']==budget]
            q=np.quantile([r['fitness'] for r in rows],[.25,.5,.75])
            tableb.append(dict(method=m,evaluations=budget,passes=sum(r['passed'] for r in rows),runs=30,median=q[1],q25=q[0],q75=q[2]))
    s.csvout('Table_A.csv',tablea); s.csvout('Table_B.csv',tableb)
    colors=plt.get_cmap('tab10').colors
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    for m,c in zip(s.METHODS,colors):
        q=np.quantile(np.array(trajectories[m])[:,np.array(s.CHECKPOINTS)-1],[.25,.5,.75],axis=0)
        ax.plot(s.CHECKPOINTS,q[1],label=m,color=c,marker='o'); ax.fill_between(s.CHECKPOINTS,q[0],q[2],alpha=.12,color=c)
    ax.set(yscale='log',xlabel='Unique objective evaluations',ylabel='Best cost (median and interquartile band)'); ax.legend(ncol=3); ax.grid(alpha=.2)
    for ext in ('png','pdf'): fig.savefig(s.ROOT/f'Figure_B.{ext}',dpi=220)
    plt.close(fig)
    h=np.array(finals['hybrid']); comparisons=[]
    signs=np.array(list(itertools.product([-1,1],repeat=10)))
    for m in s.METHODS:
        if m=='hybrid': continue
        other=np.array(finals[m]); diff=h-other
        p=1. if np.all(diff==0) else float(wilcoxon(diff,alternative='two-sided',method='auto').pvalue)
        delta=float(np.mean(np.sign(h[:,None]-other[None,:])))
        nz=diff[diff!=0]; ranks=rankdata(abs(nz)); rbc=float(np.sum(np.sign(nz)*ranks)/ranks.sum()) if len(nz) else 0.
        # Plant-level paired log-cost means; exact 2^10 sign flips.
        clusters=np.log(h/other).reshape(10,3).mean(axis=1)
        cp=float(np.mean(abs(signs@clusters/10)>=abs(clusters.mean())-1e-14))
        label=next(label for cutoff,label in [(0.147,'negligible'),(.33,'small'),(.474,'medium'),(float('inf'),'large')] if abs(delta)<cutoff)
        comparisons.append(dict(comparator=m,wilcoxon_p=p,cliffs_delta=delta,effect_label=label,paired_rank_biserial=rbc,plant_signflip_p=cp))
    adjusted=holm([r['wilcoxon_p'] for r in comparisons]); cluster_adjusted=holm([r['plant_signflip_p'] for r in comparisons])
    for r,p,cp in zip(comparisons,adjusted,cluster_adjusted): r.update(holm_p=float(p),plant_holm_p=float(cp))
    s.csvout('Table_C.csv',comparisons)
    values=np.array([finals[m] for m in s.METHODS]); assert np.all(values>0)
    ratios=values/values.min(axis=0); tau=np.unique(np.r_[1.,ratios.ravel()])
    fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
    profile=[]
    for m,r,c in zip(s.METHODS,ratios,colors):
        y=np.mean(r[:,None]<=tau[None,:],axis=0)
        ax.step(tau,y,where='post',label=m,color=c)
        profile.extend(dict(method=m,tau=float(t),fraction=float(v)) for t,v in zip(tau,y))
    ax.set(xlabel='Cost ratio to the best method on the same plant and seed',ylabel='Fraction of paired problems',ylim=(0,1.02),xlim=(1,float(tau.max())*1.01)); ax.legend(); ax.grid(alpha=.2)
    for ext in ('png','pdf'): fig.savefig(s.ROOT/f'Figure_C.{ext}',dpi=220)
    plt.close(fig); s.csvout('performance_profile.csv',profile)
    target=float(np.median(finals['random'])); crossing=[]
    for m in s.METHODS:
        for i,t in enumerate(trajectories[m]):
            hits=np.flatnonzero(t<=target)
            crossing.append(dict(method=m,plant=plants[i//3]['id'],seed=s.SEEDS[i%3],target_cost=target,first_evaluation=int(hits[0]+1) if len(hits) else '',reached=bool(len(hits))))
    s.csvout('target_crossings.csv',crossing)
    med=np.median(trajectories['hybrid'],axis=0); hits=np.flatnonzero(med<=target)
    reach=int(hits[0]+1) if len(hits) else None
    timing=s.read(s.ROOT/'timing.json'); te=s.read(s.ROOT/'evaluation_timing.json')
    random_reach=int(np.flatnonzero(np.median(trajectories['random'],axis=0)<=target)[0]+1)
    deltaE=random_reach-reach if reach is not None else None
    saving=deltaE*te['mean_objective_seconds']-timing['prediction_seconds_mean'] if deltaE is not None else None
    historical=s.read(s.OLD/'offline_costs.json')
    offline=historical['label_optimizer_sum_seconds']
    rows=[]
    for scenario,cost in [('reused_training_data',timing['training_seconds']),('data_generation_included_serial_equivalent',timing['training_seconds']+offline)]:
        rows.append(dict(scenario=scenario,T_eval=te['mean_objective_seconds'],T_train=timing['training_seconds'],T_pred=timing['prediction_seconds_mean'],
          target_cost=target,random_median_curve_reaches_at=random_reach,hybrid_median_curve_reaches_at=reach,delta_evaluations=deltaE,saving_seconds_per_plant=saving,
          offline_seconds=cost,break_even_plants=math.ceil(cost/saving) if saving is not None and saving>0 else 'not reached'))
    s.csvout('Table_D.csv',rows)
    summary=dict(table_A=tablea,table_C=comparisons,table_D=rows,threshold=target,hybrid_reach=reach,random_reach=random_reach,
      saving_evaluations_percent=100*(1800-reach)/1800 if reach is not None else None,
      observed_crossing_saving_percent=100*deltaE/random_reach if deltaE is not None else None,
      quality_note='Table B uses full original validation criterion, not just a fitness threshold. Anytime curves are prefixes of 1800-budget trajectories; they are not independently retuned smaller-budget runs.',
      effect_note='Negative delta/rank-biserial favors hybrid (lower costs). Cliff delta is the conventional all-cross-pairs distribution effect, not a paired statistic. Paired rank-biserial and plant-level sensitivity are included.',
      break_even_note='Target is the random PSO median final cost; break-even uses the difference between random and hybrid median-curve FIRST crossings, allowing random PSO to reach its final median before 1800. The separate full-budget saving compares hybrid crossing against 1800. Per-run nonattainment is retained in target_crossings.csv. Data-included numerator uses historical summed optimization durations, not historical parallel elapsed time. Predictor selection/development and validation remain excluded.',
      hardware_scope='No new HIL/PIL tests. Original five-method frozen-controller hardware results remain unchanged.')
    s.save(s.ROOT/'summary.json',summary)
    s.save(s.ROOT/'status.json',dict(status='COMPLETE',completed=210,total=210))
    print(json_text(summary))

def json_text(o):
    import json
    return json.dumps(o,indent=2)

if __name__=='__main__': main()
