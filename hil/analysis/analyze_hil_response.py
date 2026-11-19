from pathlib import Path
import json,math
from rotational_plant import euler
R=Path(__file__).resolve().parents[1]
rows=[json.loads(s) for s in (R/'logs/fixture_20261114T223012Z.jsonl').read_text().splitlines()]
a=[x for x in rows if x['kind']=='plant' and x['target_roll']>0];t0=a[0]['t']
v=[(x['t']-t0,math.degrees(euler(x['state'][:4])[0])) for x in a]
def crossing(level):return next((t for t,y in v if y>=level),None)
def settling(band):
 if abs(v[-1][1]-5)>band:return None
 last=next((i for i in range(len(v)-1,-1,-1) if abs(v[i][1]-5)>band),None)
 return 0 if last is None else v[last+1][0]
tail=[y for t,y in v if t>=v[-1][0]-1];mean=sum(tail)/len(tail)
r={'source_log':'fixture_20261114T223012Z.jsonl','scope':'single stock-PX4 rotational-fixture commissioning run','target_deg':5,'observed_after_step_s':v[-1][0],'rise_10_90_s':crossing(4.5)-crossing(.5),'settling_5percent_s':settling(.25),'settling_2percent_s':settling(.1),'mean_final_second_deg':mean,'target_minus_final_mean_deg':5-mean,'relative_final_mean_error_percent':100*(5-mean)/5,'peak_roll_deg':max(y for t,y in v),'overshoot_percent':max(0,(max(y for t,y in v)-5)/5*100),'final_plant_roll_deg':v[-1][1],'final_estimated_roll_deg':math.degrees(a[-1]['estimated_attitude'][0]),'interpolation':'none; sample crossing times','optimization_validation':False}
(R/'reports/hil_response_metrics.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
