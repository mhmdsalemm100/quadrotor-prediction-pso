from pathlib import Path
import csv,json,hashlib,shutil,collections,math
R=Path(__file__).resolve().parents[1];S=Path(r'SOURCE_STUDY/neural network by python/final_unseen_1800')
rows=list(csv.DictReader((S/'final_gains.csv').open()));assert len(rows)==150
counts=collections.Counter(x['method'] for x in rows);assert set(counts.values())=={30}
plants={d['id']:d for d in json.loads((R/'configs/hil_plants.json').read_text())};assert len(plants)==10
cfg=json.loads((R/'configs/paper_config.json').read_text());names=[p['name'] for p in cfg['parameters']]
items=[]
for row in rows:
 gains=[float(row[n]) for n in names]
 assert all(math.isfinite(g) and p['lower']*(1-1e-12)<=g<=p['upper']*(1+1e-12) for g,p in zip(gains,cfg['parameters']))
 assert row['drone'] in plants
 for axis,amplitude in ((0,5),(0,20),(1,5),(1,20),(2,30),(2,100)):
  items.append({'id':f"{row['drone']}_{row['seed']}_{row['method']}_axis{axis}_{amplitude}",'drone':row['drone'],'seed':int(row['seed']),'method':row['method'],'gains':gains,'axis':axis,'amplitude_rad':math.radians(amplitude),'plant':plants[row['drone']]['plant'],'status':'not_started'})
shutil.copy2(S/'final_gains.csv',R/'experiment/frozen_gains.csv')
manifest={'purpose':'Hardware validation of frozen software-optimized controllers; no new optimization or NN retraining','controllers':150,'scenarios':900,'methods':dict(counts),'scenario_duration_s':3.2,'minimum_scenario_wall_seconds':2880,'source_sha256':hashlib.sha256((S/'final_gains.csv').read_bytes()).hexdigest(),'feedback':'ideal simulated attitude and body rate; onboard study controller, not PX4 estimator','hardware_runs_started':False,'items':items}
(R/'experiment/hil_validation_plan.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({k:v for k,v in manifest.items() if k!='items'},indent=2))
