"""Serial warm objective timings, run after optimizers finish to avoid contention."""
import time
import numpy as np
import study_extension as s
s.warm(); cfg=s.b.original.config(); plants=s.read(s.OLD/'plants.json')
timings=[]; scenario_times=[]
for d in plants:
    candidates=[s.b.baseline(),np.array(s.read(s.ROOT/'runs'/d['id']/'hybrid_48001.json')['checkpoints'][-1]['gains'])]
    cases=s.b.original.cases_for(d)
    for gains in candidates:
        for _ in range(5):
            start=time.perf_counter(); s.b.evaluate(gains,cases,cfg); timings.append(time.perf_counter()-start)
    start=time.perf_counter(); s.b.evaluate(candidates[1],cases[:1],cfg); scenario_times.append(time.perf_counter()-start)
s.save(s.ROOT/'evaluation_timing.json',dict(mean_objective_seconds=float(np.mean(timings)),median_objective_seconds=float(np.median(timings)),
    objective_samples=len(timings),objective_samples_seconds=timings,single_scenario_samples_seconds=scenario_times,mean_single_scenario_seconds=float(np.mean(scenario_times)),scenarios_per_objective=6,
    scope='Warm serial timings after comparison workers stop; 10 plants x baseline/final hybrid controller x 5 repeats. Excludes compilation, optimizer overhead and validation.'))
print('Objective timing complete')
