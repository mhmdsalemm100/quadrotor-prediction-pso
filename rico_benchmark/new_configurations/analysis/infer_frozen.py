"""Warm inference timings in the scikit-learn environment used for training."""
import json
import time
from pathlib import Path

import joblib
import numpy as np

root = Path(__file__).resolve().parent
project = Path('USER_HOME\\Desktop\\Quadrotor_PSO_Fresh_Study\\neural network by python')
names = json.loads((project / 'selection.json').read_text())['selected']
start = time.perf_counter()
models = [joblib.load(project / 'models' / (n + '.joblib')) for n in names]
load_seconds = time.perf_counter() - start

def predict(X):
    return [np.clip(np.mean([m.predict(X) for m in group['models']], axis=0)[0], 0, 1).tolist() for group in models]

assert models[0]['inputs'] == models[1]['inputs']
predict(np.array([[1., .02, 1.]]))
rows = {}
for drone in json.loads((root / 'plants.json').read_text()):
    X = np.array([[drone[k] for k in models[0]['inputs']]])
    timings = []
    for _ in range(11):
        start = time.perf_counter()
        result = predict(X)
        timings.append(time.perf_counter() - start)
    rows[drone['id']] = dict(vectors=result,seconds=float(np.median(timings)))
(root / 'predictions.json').write_text(json.dumps(dict(models=names,load_seconds=load_seconds,rows=rows),indent=2))
