"""Scenarios, dimensionless constrained fitness, and PSO-compatible evaluation helpers."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import csv
import json
import time
import numpy as np
from model import NOMINAL, baseline, run

def training_scenarios():
    return [{'name':name, 'target':target, 'group':'training'} for name,target in [
        ('roll_5deg',[5,0,0]),('roll_20deg',[20,0,0]),
        ('pitch_5deg',[0,5,0]),('pitch_20deg',[0,20,0]),
        ('yaw_30dps',[0,0,30]),('yaw_100dps',[0,0,100])]]

def validation_scenarios():
    cases=[]
    for name,target in [('roll_minus12',[-12,0,0]),('pitch_minus12',[0,-12,0]),
                        ('yaw_minus60',[0,0,-60]),('roll_max35',[35,0,0]),
                        ('pitch_max35',[0,35,0]),('yaw_max200',[0,0,200]),
                        ('coupled_10_8_40',[10,8,40])]:
        cases.append({'name':name,'target':target,'group':'held_out'})
    for label, inertia,tau,thrust in [('heavier_slower',1.15,1.3,.9),('lighter_faster',.85,.7,1.1)]:
        plant=NOMINAL.copy(); plant[:3]*=inertia; plant[10]*=tau
        plant[4:6]*=thrust
        cases.append({'name':label,'target':[12,8,40], 'plant':plant.tolist(),'group':'uncertainty'})
    for thr in [.50,.72]:
        cases.append({'name':f'throttle_{thr:.2f}','target':[12,8,40],
                      'throttle':thr,'group':'operating_point'})
    cases += [
        {'name':'torque_pulse','target':[10,0,0],'pulse_axis':0,'pulse_amplitude':.03,'group':'disturbance'},
        {'name':'measurement_perturbation','target':[10,0,0], 'noise_amplitude':.001,'group':'noise'},
        {'name':'source_startup35','target':[35,0,0],'startup':True,'pre_step':1.,'horizon':9.,'group':'source_startup'},
        {'name':'controller_4ms','target':[10,8,40],'control_dt':.004,'group':'timing'},
        {'name':'mixer_1ms','target':[10,8,40],'mixer_dt':.001,'group':'timing'},
        {'name':'inner_roll30','target':[30,0,0],'inner':True,'horizon':1.5,'group':'inner_loop'},
        {'name':'inner_pitch30','target':[0,30,0],'inner':True,'horizon':1.5,'group':'inner_loop'}]
    return cases

def robust_training_scenarios():
    cases=training_scenarios()
    for j in range(3):
        for label,scale,tau,force in [('slow',1.15,1.3,.9),('fast',.85,.7,1.1)]:
            p=NOMINAL.copy(); p[:3]*=scale; p[10]*=tau; p[4:6]*=force
            target=[0.,0.,0.]; target[j]=20. if j<2 else 100.
            cases.append({'name':f'robust_{j}_{label}','target':target,'plant':p.tolist(),'group':'robust_training'})
    return cases

def assess(metrics,diag,scenario,cfg):
    active=np.abs(scenario['target'])>1e-12
    horizon=scenario.get('horizon',cfg['response_horizon_s'])
    w=cfg['fitness_weights']
    # Dimensionless weighted mean over commanded channels.
    channel=(w['settling']*metrics[:,1]/horizon + w['rise']*metrics[:,0]/horizon
             +w['itae']*2*metrics[:,5]/horizon**2+w['iae']*metrics[:,4]/horizon)
    speed=float(np.mean(channel[active]))
    # Feasibility is independent of weights; infeasible members cannot beat feasible ones.
    violation=0.
    violation+=float(np.sum(np.maximum(0.,metrics[active,2]/cfg['max_overshoot_fraction']-1)))
    violation+=float(np.sum(np.maximum(0.,metrics[active,3]/cfg['max_final_error_fraction']-1)))
    violation+=float(np.sum(1-metrics[active,8]))
    violation+=max(0.,diag[0]/cfg['max_motor_clipping_fraction']-1)
    # Uncommanded roll/pitch must stay below 0.5 deg; uncommanded yaw below 3 deg/s.
    crosslimit=np.deg2rad([.5,.5,3.])
    violation+=float(np.sum(np.maximum(0.,metrics[~active,6]/crosslimit[~active]-1)))
    violation+=max(0.,diag[4]/np.deg2rad(75)-1)
    # Keep the source equations; constrain optimization to nonnegative motor speeds.
    violation+=max(0.,(cfg.get('minimum_motor_speed_rad_s',0.)-diag[6])/100.)
    if diag[5]: violation+=1000.
    if not np.isfinite(metrics).all() or not np.isfinite(diag).all():
        return {'speed':1e3,'violation':1e6,'effort':1.,'feasible':False}
    return {'speed':speed,'violation':float(violation),'effort':float(diag[3]),'feasible':bool(violation<=1e-10)}

def evaluate(gains,scenarios,cfg,details=False):
    cases=[]; assessments=[]
    for scenario in scenarios:
        m,d,_=run(gains,scenario,cfg)
        a=assess(m,d,scenario,cfg)
        assessments.append(a)
        if details: cases.append({'scenario':scenario,'metrics':m.tolist(),'diagnostics':d.tolist(),**a})
    speeds=np.array([a['speed'] for a in assessments]); worst=cfg['worst_case_weight']
    quality=(1-worst)*speeds.mean()+worst*speeds.max()+cfg['effort_weight']*np.mean([a['effort'] for a in assessments])
    violation=sum(a['violation'] for a in assessments)
    # atan bounds the tie-breaking quality term below 1, guaranteeing feasibility priority.
    value=float(quality if violation<=1e-10 else 1000.+100.*violation+2/np.pi*np.arctan(quality))
    return (value,cases) if details else value

def decode(unit,cfg):
    values=np.empty_like(unit)
    for j,p in enumerate(cfg['parameters']):
        if p['scale']=='log': values[...,j]=np.exp(np.log(p['lower'])+unit[...,j]*np.log(p['upper']/p['lower']))
        else: values[...,j]=p['lower']+unit[...,j]*(p['upper']-p['lower'])
    return values

def encode(gains,cfg):
    vals=np.empty_like(gains)
    for j,p in enumerate(cfg['parameters']):
        vals[j]=(np.log(gains[j]/p['lower'])/np.log(p['upper']/p['lower']) if p['scale']=='log'
                 else (gains[j]-p['lower'])/(p['upper']-p['lower']))
    return np.clip(vals,0,1)

def write_csv(path,rows):
    rows=list(rows)
    if not rows: return
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def metric_rows(gains,scenarios,cfg,controller):
    _,cases=evaluate(gains,scenarios,cfg,True); rows=[]
    for case in cases:
        s=case['scenario']; m=np.asarray(case['metrics']); d=case['diagnostics']
        for j in np.flatnonzero(np.abs(s['target'])>0):
            rows.append({'controller':controller,'scenario':s['name'],'group':s['group'],
                         'axis':['roll','pitch','yaw_rate'][j], 'command_deg_or_deg_s':s['target'][j],
                         'rise_s':m[j,0] if m[j,7] else None,
                         'settling_s':m[j,1] if m[j,8] else None,
                         'overshoot_pct':100*m[j,2],'final_error_pct':100*m[j,3],
                         'normalized_IAE_s':m[j,4],'normalized_ITAE_s2':m[j,5],
                         'motor_clip_pct':100*d[0],'controller_clip_pct':100*d[1],
                         'minimum_motor_speed_rad_s':d[6],
                         'feasible_under_design_constraints':case['feasible'],
                         'violation':case['violation']})
    return rows,cases
