"""Descriptive response metrics; not the paper's paired optimization quality test."""
from pathlib import Path
import json, math, argparse
import numpy as np
R=Path(__file__).resolve().parents[1]
a=argparse.ArgumentParser();a.add_argument('run');args=a.parse_args();folder=Path(args.run)
results=[]
for p in sorted(folder.glob('*.json')):
 d=json.loads(p.read_text())
 if d.get('status')!='recorded_requires_analysis':continue
 rows=[json.loads(s) for s in p.with_suffix('.jsonl').read_text().splitlines()]
 states=[r for r in rows if 'state' in r];packets=[r for r in rows if 'hardware_time_us' in r]
 t=np.array([r['host_elapsed_s'] for r in states]);x=np.array([r['state'] for r in states]);q=x[:,:4];w,b,c,z=q.T
 roll=np.arctan2(2*(w*b+c*z),1-2*(b*b+c*c));pitch=np.arcsin(np.clip(2*(w*c-b*z),-1,1))
 y=[roll,pitch,x[:,6]][d['axis']];target=d['amplitude_rad']
 offsets=np.array([r['host_elapsed_s']-r['tick']*.001 for r in packets]);step=.2+float(np.median(offsets));post=t>=step;tp=t[post]-step;yp=y[post];err=np.abs(yp-target)/abs(target)
 outside=np.flatnonzero(err>.02);settle_index=int(outside[-1]+1) if len(outside) else 0
 settling=float(tp[settle_index]) if settle_index<len(tp) and tp[-1]-tp[settle_index]>=.25 else None
 def crossing(f):
  ix=np.flatnonzero(yp/target>=f);return float(tp[ix[0]]) if len(ix) else None
 t10,t90=crossing(.1),crossing(.9)
 result=dict(id=d['id'],axis=d['axis'],target_degrees=math.degrees(target),final_degrees=math.degrees(float(yp[-1])),final_error_percent=float(err[-1]*100),overshoot_percent=max(0,float(np.max(yp/target)-1)*100),rise_10_90_s=t90-t10 if t10 is not None and t90 is not None else None,settling_2pct_s=settling,max_host_interval_ms=d['max_host_interval_s']*1000,max_input_age_ms=d['controller_completion'][3]/1000,max_controller_lateness_ms=d['controller_completion'][2]/1000,hardware_packets=d['hardware_packets'],alignment_spread_ms=float(np.ptp(offsets)*1000),scope='degrees for roll/pitch; degrees/s for yaw; descriptive only')
 results.append(result)
out={'run':str(folder),'completed_scenarios':len(results),'note':'Asynchronous physical-controller/Python-plant loop. Not evidence of 1 kHz transport, all-cohort superiority, or hardware optimization savings. Step time estimated from board ticks and host receive times.','results':results}
(folder/'response_analysis.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
