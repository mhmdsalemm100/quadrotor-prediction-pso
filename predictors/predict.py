"""Predict physical gains for a configuration without tuning or reading its PSO gains."""
from pathlib import Path
import argparse,json
import numpy as np
import joblib
ROOT=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser(); p.add_argument('--inertia',type=float,required=True); p.add_argument('--motor-tau',type=float,required=True); p.add_argument('--rotor',type=float,required=True); p.add_argument('--model'); a=p.parse_args()
    name=a.model or json.loads((ROOT/'selection.json').read_text())['selected'][0]
    if name not in [f.stem for f in (ROOT/'models').glob('*.joblib')]: raise ValueError('Unknown model')
    m=joblib.load(ROOT/'models'/f'{name}.joblib'); X=np.array([[a.inertia,a.motor_tau,a.rotor]])
    u=np.clip(np.mean([s.predict(X) for s in m['models']],axis=0)[0],0,1)
    g=np.exp(m['lower_log']+u*m['span_log'])
    inside=.85<=a.inertia<=1.15 and .014<=a.motor_tau<=.026 and .9<=a.rotor<=1.1
    print(json.dumps({'model':name,'within_training_design_bounds':inside,'gains':dict(zip(m['gain_names'],g)),'note':'Initial gains only; simulation validation and PSO refinement required.'},indent=2))
if __name__=='__main__': main()
