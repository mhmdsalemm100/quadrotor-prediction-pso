from pathlib import Path
import json,collections,math,html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];D=Path((R/'results/active_cohort.txt').read_text());progress=json.loads((D/'progress.json').read_text())
if progress['status']=='running':raise RuntimeError('Wait until hardware run ends before plotting all controllers')
plan=json.loads((R/'experiment/hil_validation_plan.json').read_text())['items'];ledger=[json.loads(s) for s in (D/'attempts.jsonl').read_text().splitlines()];selected={e['index']:e for e in ledger if e['status']=='transport_valid'};groups=collections.defaultdict(list)
for i,item in enumerate(plan):groups[(item['drone'],item['seed'],item['method'])].append((i,item))
O=D/'controller_traces';O.mkdir(exist_ok=True);links=[]
for key,cases in groups.items():
 fig,axes=plt.subplots(3,2,figsize=(11,8),layout='constrained');valid=0
 for ax,(index,item) in zip(axes.flat,cases):
  axis=item['axis'];target=math.degrees(item['amplitude_rad']);unit='deg/s' if axis==2 else 'deg';ax.set(title=f"{('Roll','Pitch','Yaw rate')[axis]}: {target:g} {unit}",xlabel='Time after step (s)',ylabel=unit,xlim=(-.2,3));ax.grid(alpha=.25)
  if index not in selected:
   ax.text(.5,.5,'No timing-valid cohort trace',ha='center',va='center',transform=ax.transAxes,color='#a21');continue
  e=selected[index];p=D/e['directory']/e['result_file'];rows=[json.loads(s) for s in p.with_suffix('.jsonl').read_text().splitlines()];states=[v for v in rows if 'state'in v];packets=[v for v in rows if 'hardware_time_us'in v];t=np.array([v['host_elapsed_s'] for v in states]);x=np.array([v['state'] for v in states]);w,b,c,d=x[:,:4].T
  values=[np.arctan2(2*(w*b+c*d),1-2*(b*b+c*c)),np.arcsin(np.clip(2*(w*c-b*d),-1,1)),x[:,6]]
  step=.2+np.median([v['host_elapsed_s']-.001*v['tick'] for v in packets]);t-=step;y=np.rad2deg(values[axis]);ax.plot(t,y,lw=1,label='Hardware-controller response');ax.plot(t,np.where(t>=0,target,0),'--',lw=1,label='Command');ax.legend(fontsize=6);valid+=1
 title=f'{key[0]} | seed {key[1]} | {key[2]}';fig.suptitle(title+f' | timing-valid {valid}/6\nIdeal-state rotational HIL; asynchronous timing',fontsize=11)
 filename=f'{key[0]}_{key[1]}_{key[2]}.png';fig.savefig(O/filename,dpi=130);plt.close(fig);links.append(f'<li><a href="{filename}">{html.escape(title)}</a> — {valid}/6 timing-valid</li>')
(O/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>All hardware controller traces</title><style>body{font:17px system-ui;max-width:1000px;margin:30px auto}li{margin:8px}</style><h1>All 150 saved-controller responses</h1><p>Each figure shows the six fixed commands. Missing timing-valid traces are marked explicitly; pilot or failed attempts are not substituted.</p><ul>'+''.join(links)+'</ul>',encoding='utf-8');print('Plotted',len(groups),'saved controllers')
