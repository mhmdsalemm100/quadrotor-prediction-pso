"""Analyze every first transport-valid trial; retain/report missing cases and all attempts."""
import json,math,csv,sys,collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
D=Path(sys.argv[1]) if len(sys.argv)>1 else Path((R/'results/active_cohort.txt').read_text())
ledger=[json.loads(s) for s in (D/'attempts.jsonl').read_text().splitlines() if s.strip()]
results=[]
for entry in ledger:
 if entry['status']!='transport_valid':continue
 p=D/entry['directory']/entry['result_file'];meta=json.loads(p.read_text());raw=[json.loads(s) for s in p.with_suffix('.jsonl').read_text().splitlines()]
 states=[s for s in raw if 'state'in s];packets=[s for s in raw if 'hardware_time_us'in s]
 t=np.array([s['host_elapsed_s'] for s in states]);x=np.array([s['state'] for s in states]);w,b,c,d=x[:,:4].T
 responses=np.array([np.arctan2(2*(w*b+c*d),1-2*(b*b+c*c)),np.arcsin(np.clip(2*(w*c-b*d),-1,1)),x[:,6]]).T
 axis=meta['axis'];target=meta['amplitude_rad'];offsets=np.array([s['host_elapsed_s']-.001*s['tick'] for s in packets]);step=.2+np.median(offsets)
 mask=t>=step;tp=t[mask]-step;y=responses[mask,axis];err=np.abs(y-target)/abs(target)
 bad=np.flatnonzero(err>.02);si=int(bad[-1]+1) if len(bad) else 0
 settling=float(tp[si]) if si<len(tp) and tp[-1]-tp[si]>=.25 else None
 def crossing(frac):
  ids=np.flatnonzero(y/target>=frac);return float(tp[ids[0]]) if len(ids) else None
 lo,hi=crossing(.1),crossing(.9);cross=[float(np.max(np.abs(responses[mask,j])))*180/np.pi if j!=axis else 0 for j in range(3)]
 overshoot=max(0,float(np.max(y/target)-1));final=float(err[-1]);active_pass=settling is not None and final<=.02 and overshoot<=.1
 cross_pass=cross[0]<=.5 and cross[1]<=.5 and cross[2]<=3
 row={k:meta[k] for k in ('id','drone','seed','method','axis')};row.update(index=entry['index'],attempt=entry['attempt'],target=target*180/np.pi,final_response=float(y[-1])*180/np.pi,unit='deg/s' if axis==2 else 'deg',final_error_pct=100*final,overshoot_pct=100*overshoot,rise_s=hi-lo if lo is not None and hi is not None else None,settling_s=settling,active_tracking_pass=active_pass,cross_axis_pass=cross_pass,roll_cross_deg=cross[0],pitch_cross_deg=cross[1],yaw_cross_deg_s=cross[2],max_host_ms=1000*meta['max_host_interval_s'],max_input_age_ms=meta['controller_completion'][3]/1000,max_late_ms=meta['controller_completion'][2]/1000,alignment_spread_ms=float(np.ptp(offsets))*1000,hardware_packets=meta['hardware_packets'],trace=str(p.with_suffix('.jsonl')))
 results.append(row)
if not results:raise SystemExit('No valid traces yet')
with (D/'scenario_metrics.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
plan=json.loads((R/'experiment/hil_validation_plan.json').read_text())['items'];gain_names=[p['name'] for p in json.loads((R/'configs/paper_config.json').read_text())['parameters']];controller_rows=[];controller_items={}
for item in plan:controller_items.setdefault((item['drone'],item['seed'],item['method']),item)
for key,item in controller_items.items():
 rows=[r for r in results if (r['drone'],r['seed'],r['method'])==key];complete=len(rows)==6
 cr={'drone':key[0],'seed':key[1],'method':key[2],'valid_commands':len(rows),'status':'complete' if complete else 'incomplete','tracking_subset_pass':all(r['active_tracking_pass'] and r['cross_axis_pass'] for r in rows) if complete else None,'worst_final_error_pct':max([r['final_error_pct'] for r in rows],default=None),'worst_overshoot_pct':max([r['overshoot_pct'] for r in rows],default=None),'worst_defined_settling_s':max([r['settling_s'] for r in rows if r['settling_s'] is not None],default=None)}
 cr.update(dict(zip(gain_names,item['gains'])));controller_rows.append(cr)
with (D/'controller_gains_and_results.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(controller_rows[0]));writer.writeheader();writer.writerows(controller_rows)
methods=['ordinary','predictions','library','hybrid','reference'];summary=[]
for method in methods:
 rows=[r for r in results if r['method']==method];groups=collections.defaultdict(list)
 for r in rows:groups[(r['drone'],r['seed'])].append(r)
 complete=[v for v in groups.values() if len(v)==6]
 summary.append(dict(method=method,valid_scenarios=len(rows),planned_scenarios=180,complete_controllers=len(complete),planned_controllers=30,controllers_meeting_tracking_subset=sum(all(r['active_tracking_pass'] and r['cross_axis_pass'] for r in v) for v in complete),median_final_error_pct=float(np.median([r['final_error_pct'] for r in rows])) if rows else None,median_settling_s=float(np.median([r['settling_s'] for r in rows if r['settling_s'] is not None])) if any(r['settling_s'] is not None for r in rows) else None))
failures=collections.Counter(e.get('error','').split(':')[0].split(' 0.')[0] for e in ledger if e['status']!='transport_valid')
progress=json.loads((D/'progress.json').read_text());out={'progress':progress,'attempt_count':len(ledger),'first_attempt_valid':sum(e['attempt']==1 and e['status']=='transport_valid' for e in ledger),'failure_reasons':dict(failures),'methods':summary,'scope':'Frozen saved-gain replay, ideal-state rotational HIL. Tracking subset excludes motor clipping and full original fitness/robustness constraints. No hardware optimization speedup measured. Incomplete controllers are not counted as performance failures. Repeated seeds share plants.'}
out['timing_summary']={key:{'median':float(np.median([r[key] for r in results])),'maximum':float(np.max([r[key] for r in results]))} for key in ('max_host_ms','max_input_age_ms','max_late_ms','alignment_spread_ms','hardware_packets')}
(D/'cohort_summary.json').write_text(json.dumps(out,indent=2))
fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained');colors=['#687d8f','#167aa5','#b98732','#178b71','#7c5b9b']
axes[0].bar(methods,[s['valid_scenarios'] for s in summary],color=colors);axes[0].axhline(180,color='black',ls='--',lw=1);axes[0].set(title='Transport-valid scenarios',ylabel='Count out of 180')
for j,key,title in [(1,'final_error_pct','Final tracking error (%)'),(2,'settling_s','2% settling time (s)')]:
 for i,m in enumerate(methods):
  values=[r[key] for r in results if r['method']==m and r[key] is not None]
  if values:axes[j].scatter(np.full(len(values),i),values,s=9,alpha=.4,color=colors[i])
 axes[j].set_xticks(range(5),methods);axes[j].set_title(title)
for a in axes:a.tick_params(axis='x',rotation=25);a.grid(axis='y',alpha=.2)
fig.suptitle('Saved-controller hardware replay — descriptive results; timing exclusions retained');fig.savefig(D/'cohort_overview.png',dpi=160);plt.close(fig)
trs=''.join('<tr>'+''.join(f'<td>{v}</td>' for v in [s['method'],s['valid_scenarios'],s['complete_controllers'],s['controllers_meeting_tracking_subset'],round(s['median_final_error_pct'],4) if s['median_final_error_pct'] is not None else 'pending'])+'</tr>' for s in summary)
html=f'''<!doctype html><meta charset="utf-8"><title>Frozen-controller HIL cohort</title><style>body{{font:17px system-ui;max-width:1150px;margin:35px auto;line-height:1.5}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;border:1px solid #ccc}}img{{width:100%}}</style><h1>Frozen-controller HIL cohort</h1><p>Status: <b>{progress['status']}</b>. Processed {progress['next_index']}/900; timing-valid {progress['valid']}; unresolved timing cases {progress['invalid_scenarios']}; total attempts {progress['attempts']}.</p><p>{out['scope']}</p><img src="cohort_overview.png"><table><tr><th>Method</th><th>Valid scenarios /180</th><th>Complete controllers /30</th><th>Complete controllers meeting tracking subset</th><th>Median final error %</th></tr>{trs}</table><p>Tracking subset: final error ≤2%, overshoot ≤10%, 2% settling with ≥0.25 s final hold; inactive roll/pitch ≤0.5° and yaw rate ≤3°/s. It is not the original paired-reference quality criterion. All failures and attempt selection are in attempts.jsonl. Values are reported conditionally on transport validity.</p><p><a href="controller_gains_and_results.csv">All 150 gain sets and controller results</a> · <a href="controller_traces/index.html">Individual controller plots</a> · <a href="scenario_metrics.csv">Scenario metrics CSV</a> · <a href="cohort_summary.json">Summary and timing failures</a></p>'''
(D/'HIL_Cohort_Report.html').write_text(html,encoding='utf-8');print(json.dumps(out,indent=2))
