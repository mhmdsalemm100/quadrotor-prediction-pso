from pathlib import Path
import shutil,json,hashlib,joblib,numpy as np,sklearn
R=Path(__file__).resolve().parents[1];S=Path(r'SOURCE_STUDY');P=S/'neural network by python'
for name in ('elite_medoid__nn_regularized','elite_medoid__gp'):shutil.copy2(P/'models'/(name+'.joblib'),R/'models'/(name+'.joblib'))
shutil.copy2(S/'config.json',R/'configs/paper_config.json');shutil.copy2(P/'final_unseen_1800/plants.json',R/'configs/hil_plants.json')
models={name:joblib.load(R/'models'/(name+'.joblib')) for name in ('elite_medoid__nn_regularized','elite_medoid__gp')}
rows=[]
for d in json.loads((R/'configs/hil_plants.json').read_text()):
 out={'plant':d,'predictions':{}}
 for name,m in models.items():
  x=np.array([[d[k] for k in m['inputs']]]);u=np.clip(np.mean([p.predict(x) for p in m['models']],axis=0)[0],0,1);g=np.exp(m['lower_log']+u*m['span_log'])
  out['predictions'][name]={'unit_coordinates':u.tolist(),'gains':dict(zip(m['gain_names'],g.tolist()))}
 rows.append(out)
report={'scope':'Frozen predictors reused without retraining; predictions are initial candidates, not HIL results','sklearn_version':sklearn.__version__,'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (R/'models').glob('*.joblib')},'rows':rows}
(R/'experiment/predicted_initial_gains.json').write_text(json.dumps(report,indent=2));print('Prepared predictions for',len(rows),'plants; sklearn',sklearn.__version__)
