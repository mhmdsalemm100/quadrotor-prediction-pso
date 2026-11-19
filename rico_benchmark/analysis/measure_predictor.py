"""Run in the original sklearn runtime; refit for timing only, save no models."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import sys,time,json,platform
from pathlib import Path
import numpy as np
import joblib,sklearn
ROOT=Path(__file__).resolve().parent
STUDY=Path.home()/'Desktop/Quadrotor_PSO_Fresh_Study'
PROJECT=STUDY/'neural network by python'
sys.path.insert(0,str(PROJECT))
from train import fit_predict, INPUTS
def read(p): return json.loads(p.read_text())
train=[d for d in read(STUDY/'manifest.json')['drones'] if d['role']=='development_train']
cfg=read(STUDY/'config.json'); pars=cfg['parameters']
lo=np.log([p['lower'] for p in pars]); span=np.log([p['upper']/p['lower'] for p in pars])
X=np.array([[d[k] for k in INPUTS] for d in train]); labels={'best':[],'elite_medoid':[]}
for d in train:
    runs=sorted([read(p) for p in (STUDY/'drones'/d['id']/'runs').glob('*.json')],key=lambda r:r['fitness'])
    assert len(runs)==30
    U=(np.log([r['gains'] for r in runs])-lo)/span
    elite=U[:5]; medoid=int(np.argmin(np.linalg.norm(elite[:,None]-elite[None,:],axis=2).sum(axis=1)))
    labels['best'].append(U[0]); labels['elite_medoid'].append(U[medoid])
names=read(PROJECT/'selection.json')['selected']; fresh=[]; per_model=[]
for name in names:
    label,kind=name.split('__'); start=time.perf_counter()
    models,_=fit_predict(kind,X,np.array(labels[label]),X[:1]); elapsed=time.perf_counter()-start
    fresh.append(models); per_model.append(dict(model=name,seconds=elapsed))
frozen=[joblib.load(PROJECT/'models'/f'{n}.joblib') for n in names]
queries=np.array([[d[k] for k in INPUTS] for d in read(PROJECT/'final_unseen_1800/plants.json')])
def predict(q): return [np.clip(np.mean([m.predict(q) for m in g['models']],axis=0),0,1) for g in frozen]
predict(queries[:1]); times=[]
for q in queries:
    for _ in range(30):
        start=time.perf_counter(); predict(q[None,:]); times.append(time.perf_counter()-start)
out=dict(training_seconds=sum(r['seconds'] for r in per_model),per_model=per_model,
 prediction_seconds_mean=float(np.mean(times)),prediction_seconds_median=float(np.median(times)),prediction_samples=len(times),
 prediction_samples_seconds=times,python=sys.version,sklearn=sklearn.__version__,platform=platform.platform(),
 scope='Fixed selected NN ensemble and GP refit once on 20 original training plants. Includes fit_predict first-query overhead. Excludes search/model selection and label generation. Original frozen models used for inference and optimization, never overwritten.')
(ROOT/'timing.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
