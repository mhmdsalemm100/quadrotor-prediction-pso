"""Fit candidate predictors using only Drone_01..20; no simulator evaluations."""
from pathlib import Path
import json, time, warnings, hashlib
import numpy as np
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold
from sklearn.compose import TransformedTargetRegressor

ROOT=Path(__file__).resolve().parent
STUDY=ROOT.parent
INPUTS=['inertia_scale','motor_tau_s','rotor_scale']

def save(name,data):
    (ROOT/name).write_text(json.dumps(data,indent=2,allow_nan=False))

def fit_predict(kind,X,Y,Q):
    if kind.startswith('nn'):
        width,alpha={'nn_small':(8,.1),'nn_regularized':(16,1.),'nn_deep':((12,8),.1)}[kind]
        width=(width,) if isinstance(width,int) else width
        models=[]; predicted=[]
        for seed in [11,29,47]:
            m=TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=width,activation='tanh',solver='adam',alpha=alpha,max_iter=6000,tol=1e-6,n_iter_no_change=100,random_state=seed)),transformer=StandardScaler())
            m.fit(X,Y); models.append(m); predicted.append(m.predict(Q))
        return models,np.mean(predicted,axis=0)
    if kind=='ridge': m=make_pipeline(StandardScaler(),Ridge(alpha=.1))
    elif kind=='quadratic_ridge': m=make_pipeline(StandardScaler(),PolynomialFeatures(2,include_bias=False),StandardScaler(),Ridge(alpha=1.))
    elif kind=='gp': m=make_pipeline(StandardScaler(),GaussianProcessRegressor(kernel=ConstantKernel(1.,(.01,100))*RBF([1.,1.,1.],(.1,100)),alpha=.01,normalize_y=True,random_state=11))
    elif kind=='extra_trees': m=ExtraTreesRegressor(n_estimators=250,min_samples_leaf=2,random_state=11,n_jobs=1)
    m.fit(X,Y)
    return [m],m.predict(Q)

def main():
    started=time.perf_counter()
    manifest=json.loads((STUDY/'manifest.json').read_text())
    train=[d for d in manifest['drones'] if d['role']=='development_train']
    validation=[d for d in manifest['drones'] if d['role']=='development_validation']
    assert [d['id'] for d in train]==[f'Drone_{i:02d}' for i in range(1,21)]
    cfg=json.loads((STUDY/'config.json').read_text()); pars=cfg['parameters']
    lo=np.log([p['lower'] for p in pars]); span=np.log([p['upper']/p['lower'] for p in pars])
    X=np.array([[d[k] for k in INPUTS] for d in train]); Q=np.array([[d[k] for k in INPUTS] for d in validation])
    labels={'best':[],'elite_medoid':[]}; diagnosis=[]; hashes={}
    for d in train:
        files=sorted((STUDY/'drones'/d['id']/'runs').glob('*.json'))
        assert len(files)==30
        runs=[]
        for p in files:
            hashes[str(p.relative_to(STUDY))]=hashlib.sha256(p.read_bytes()).hexdigest(); runs.append(json.loads(p.read_text()))
        runs.sort(key=lambda r:r['fitness']); U=(np.log([r['gains'] for r in runs])-lo)/span
        # Select an actual coherent solution, never average incompatible gains.
        elite=U[:5]; medoid=int(np.argmin(np.sum(np.linalg.norm(elite[:,None]-elite[None,:],axis=2),axis=1)))
        labels['best'].append(U[0]); labels['elite_medoid'].append(U[medoid])
        diagnosis.append({'drone':d['id'],'best_seed':runs[0]['seed'],'medoid_seed':runs[medoid]['seed'],'top5_unit_std':np.std(elite,axis=0).tolist(),'top5_fitness':[r['fitness'] for r in runs[:5]]})
    folds=list(KFold(5,shuffle=True,random_state=20261101).split(X)); candidates=[]
    (ROOT/'models').mkdir(exist_ok=True)
    kinds=['nn_small','nn_regularized','nn_deep','ridge','quadratic_ridge','gp','extra_trees']
    for label,Y0 in labels.items():
        Y=np.array(Y0)
        for kind in kinds:
            name=label+'__'+kind; out=np.empty_like(Y)
            for tr,va in folds:
                _,pred=fit_predict(kind,X[tr],Y[tr],X[va]); out[va]=np.clip(pred,0,1)
            models,pred=fit_predict(kind,X,Y,Q)
            joblib.dump({'models':models,'inputs':INPUTS,'gain_names':[p['name'] for p in pars],'label':label,'kind':kind,'lower_log':lo,'span_log':span},ROOT/'models'/f'{name}.joblib')
            candidates.append({'name':name,'neural':kind.startswith('nn'),'oof':out.tolist(),'predictions':np.clip(pred,0,1).tolist(),'oof_unit_rmse':np.sqrt(np.mean((out-Y)**2,axis=0)).tolist()})
            print('Trained '+name,flush=True)
    save('candidate_predictions.json',{'training_ids':[d['id'] for d in train],'validation_ids':[d['id'] for d in validation],'candidates':candidates,'folds':[{'train':[train[i]['id'] for i in tr],'held_out':[train[i]['id'] for i in va]} for tr,va in folds],'training_source_sha256':hashes,'seconds':time.perf_counter()-started})
    save('label_diagnosis.json',diagnosis)

if __name__=='__main__': main()
