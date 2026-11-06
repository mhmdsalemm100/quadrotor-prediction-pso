"""Read-only analysis of frozen results. Writes manuscript artifacts only."""
from pathlib import Path
import csv, json, itertools, hashlib, sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT=Path(__file__).resolve().parent
STUDY=Path(os.environ.get('QUADROTOR_STUDY_ROOT',str(ROOT.parent/'study')))
TEST=STUDY/'neural network by python/final_unseen_1800'
DATA=ROOT/'analysis'; DATA.mkdir(exist_ok=True)
def write(name,obj): (DATA/name).write_text(json.dumps(obj,indent=2),encoding='utf-8')
rows=list(csv.DictReader((TEST/'per_run.csv').open()))
pair={(r['drone'],int(r['seed']),r['method']):r for r in rows}
drones=sorted({r['drone'] for r in rows}); seeds=[48001,48002,48003]
methods=['ordinary','predictions','library','hybrid','reference']
names=['Ordinary','NN + GP','Library','Hybrid','Reference (3264)']
colors=['#6c7b8e','#2171b5','#29947b','#7952a4','#d0782e']
stats=[]
for a,b in [('hybrid','ordinary'),('predictions','ordinary'),('library','ordinary'),('hybrid','predictions'),('hybrid','library')]:
    diffs=np.array([np.mean([int(pair[d,s,a]['passed']=='True')-int(pair[d,s,b]['passed']=='True') for s in seeds]) for d in drones])
    signs=np.array(list(itertools.product([-1,1],repeat=10)))
    distribution=(signs*diffs).mean(axis=1)
    p=float(np.mean(np.abs(distribution)>=abs(diffs.mean())-1e-12))
    boot=np.random.default_rng(6801).choice(diffs,(20000,10),replace=True).mean(axis=1)
    stats.append(dict(method=a,comparator=b,plant_mean_differences=diffs.tolist(),difference_pp=100*float(diffs.mean()),cluster_sign_flip_p=p,bootstrap95_pp=(100*np.quantile(boot,[.025,.975])).tolist(),inference='Post-hoc exact whole-plant label-swap sensitivity analysis; requires exchangeability under the null. Only hybrid-ordinary contrast was prespecified.'))
write('statistics.json',stats)
offline=json.loads((TEST/'offline_costs.json').read_text())
cost=[]
for m in ['predictions','hybrid']:
    savings=np.array([float(pair[d,s,'reference']['tuning_seconds'])-float(pair[d,s,m]['tuning_seconds']) for d in drones for s in seeds])
    cost.append(dict(method=m,mean_paired_saving_s=float(savings.mean()),median_paired_saving_s=float(np.median(savings)),label_only_seconds=offline['label_optimizer_sum_seconds'],illustrative_label_only_break_even=float(offline['label_optimizer_sum_seconds']/savings.mean()),label_evaluation_break_even=offline['label_generation_evals']/1464,scope='Measured durations under concurrent workload; excludes offline model selection, validation and development, and failure fallback. Conditional accounting illustration, not deployment forecast.'))
write('break_even.json',cost)

hist={}; export=[]
for m in methods:
    h=[]
    for d in drones:
        for s in seeds:
            p=TEST/d/f'{m}_{s}.npz'
            z=np.load(p)['ledger']; n=3264 if m=='reference' else 1800
            assert z.shape==(n,12) and np.array_equal(z[:,3],np.minimum.accumulate(z[:,2]))
            h.append(z[:,3])
    hist[m]=np.array(h)
    for e in range(hist[m].shape[1]):
        q=np.quantile(hist[m][:,e],[.25,.5,.75])
        export.append(dict(method=m,evaluation=e+1,q25=q[0],median=q[1],q75=q[2],feasible_fraction=float(np.mean(hist[m][:,e]<1000))))
with (DATA/'convergence_summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(export[0]));w.writeheader();w.writerows(export)
fig,axs=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
for m,n,c in zip(methods,names,colors):
    h=hist[m];x=np.arange(1,h.shape[1]+1);q=np.quantile(h,[.25,.5,.75],axis=0)
    axs[0].plot(x,q[1],label=n,color=c,lw=1.5);axs[0].fill_between(x,q[0],q[2],color=c,alpha=.10)
    axs[1].plot(x,q[1],color=c,lw=1.5)
    axs[1].fill_between(x,q[0],q[2],color=c,alpha=.10)
axs[0].set(yscale='log',xlabel='Unique fitness evaluations',ylabel='Best constrained fitness',title='Median and interquartile range (30 runs)')
axs[0].legend(fontsize=10)
zoom=np.concatenate([np.quantile(hist[m][:,255:1800],[.25,.75],axis=0).ravel() for m in methods])
axs[1].set(xlabel='Unique fitness evaluations',ylabel='Best constrained fitness',title='Refinement detail after evaluation 256',xlim=(256,1800),ylim=(float(zoom.min())*.97,float(zoom.max())*1.03))
for ax in axs:ax.axvline(1800,color='black',ls=':',lw=.8);ax.grid(alpha=.15)
fig.savefig(ROOT/'convergence.png',dpi=240);plt.close(fig)
fig,ax=plt.subplots(figsize=(11,4.1),layout='constrained');ax.set(xlim=(0,11),ylim=(0,4.1));ax.axis('off')
def box(x,y,w,h,text):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',facecolor='#edf3fa',edgecolor='#54748a'))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=11)
def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#54748a',lw=1.4))
box(.1,2.65,2.35,1,'20 training plants\n30 PSO runs per plant\n600 archived controllers')
box(3,2.65,2.15,1,'Top-five medoid labels\nPlant-level CV\nFit NN ensemble + GP')
box(5.75,2.65,2.05,1,'Development plants\nChoose 1800 budget\nFreeze protocol and models')
box(8.4,2.65,2.35,1,'10 unseen plants\n3 paired seeds\nFour arms + reference')
box(.1,.45,2.35,1.15,'New plant parameters\nInertia, motor lag,\nrotor coefficient scale')
box(3,.45,2.15,1.15,'Predict two gain vectors\nOptional library lookup\nKeep global exploration')
box(5.75,.45,2.05,1.15,'PSO refinement\n1800 unique evaluations\nSix single-axis simulations\nper evaluation')
box(8.4,.45,2.35,1.15,'Evaluate final controller\nSettling and constraints\nLocal and combined checks\nOnline computation time')
for y in [3.15,1.02]:
    for a,b in [(2.45,3),(5.15,5.75),(7.8,8.4)]:arrow((a,y),(b,y))
arrow((4.08,2.65),(4.08,1.6));ax.text(.1,3.95,'OFFLINE DEVELOPMENT AND FROZEN EVALUATION',fontsize=10,weight='bold');ax.text(.1,1.95,'ONLINE USE AND ASSESSMENT',fontsize=10,weight='bold')
fig.savefig(ROOT/'pipeline.png',dpi=240);plt.close(fig)

# Diagnostic replay on development data only. Preserve frozen simulator unchanged.
sys.path.insert(0,str(STUDY));import model
cfg=json.loads((STUDY/'config.json').read_text())
scenario=next(s for s in json.loads((STUDY/'drones/Drone_21/validation_scenarios.json').read_text()) if s['name']=='combined_00')
diagnosis=[]
for name,p in [('hybrid',STUDY/'neural network by python/hybrid_1800/Drone_21/28001.json'),('reference',STUDY/'drones/Drone_21/runs/28001.json')]:
    g=np.array(json.loads(p.read_text())['gains']);m,d,t=model.run(g,scenario,cfg,True)
    tail=t[t[:,0]>=2.95];phi=tail[:,1];theta=tail[:,2];pp=tail[:,4];qq=tail[:,5];rr=tail[:,6]
    pdot=pp+np.sin(phi)*np.tan(theta)*qq+np.cos(phi)*np.tan(theta)*rr
    tdot=np.cos(phi)*qq-np.sin(phi)*rr
    required_p=-rr*np.tan(theta)/np.cos(phi);required_q=rr*np.tan(phi)
    diagnosis.append(dict(method=name,command_deg_or_deg_s=scenario['target'],final_normalized_errors=m[:,3].tolist(),roll_pitch_rates_deg_s=np.rad2deg([pp.mean(),qq.mean()]).tolist(),required_body_rates_deg_s=np.rad2deg([required_p.mean(),required_q.mean()]).tolist(),euler_derivatives_deg_s=np.rad2deg([pdot.mean(),tdot.mean()]).tolist(),outer_rate_refs_deg_s=np.rad2deg(tail[:,10:12].mean(axis=0)).tolist(),motor_clip_fraction=float(d[0]),gain_source=str(p),simulator_sha256=hashlib.sha256((STUDY/'model.py').read_bytes()).hexdigest()))
    np.savez_compressed(DATA/f'development_replay_{name}.npz',trace=t,metrics=m,diagnostics=d,gains=g)
write('development_diagnosis.json',diagnosis)
fig,axs=plt.subplots(2,2,figsize=(10,7),layout='constrained')
for name,color,label in [('hybrid','#2171b5','Hybrid'),('reference','#d0782e','Full-budget PSO')]:
    t=np.load(DATA/f'development_replay_{name}.npz')['trace'];time=t[:,0]-.2
    for ax,col,refcol,ylabel in zip(axs.flat,[1,2,6],[7,8,9],['Roll (deg)','Pitch (deg)','Yaw body rate (deg/s)']):
        ax.plot(time,np.rad2deg(t[:,col]),color=color,label=label)
        if name=='hybrid':ax.plot(time,np.rad2deg(t[:,refcol]),'k:',label='Command')
        ax.set(xlabel='Time from command (s)',ylabel=ylabel);ax.grid(alpha=.2)
    axs[1,1].plot(time,t[:,23:27].max(axis=1),color=color,label=label+' max')
    axs[1,1].plot(time,t[:,23:27].min(axis=1),color=color,ls='--',label=label+' min')
axs[0,0].legend(fontsize=8);axs[1,1].legend(fontsize=7)
axs[1,1].set(xlabel='Time from command (s)',ylabel='Motor command envelope');axs[1,1].grid(alpha=.2)
fig.savefig(ROOT/'combined_response.png',dpi=240);plt.close(fig)
write('source_hashes.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [TEST/'protocol.json',TEST/'per_run.csv',STUDY/'model.py',STUDY/'study.py',STUDY/'config.json']})
print(json.dumps(dict(statistics=stats,break_even=cost,diagnosis=diagnosis),indent=2))
