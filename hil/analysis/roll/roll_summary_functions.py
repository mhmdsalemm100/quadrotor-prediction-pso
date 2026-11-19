"""Original roll-summary functions, extracted without executing analysis or manuscript generation."""
import collections
import numpy as np

def truth(v):return v is True or v=='True'

def summarize(rows,method):
 rr=[r for r in rows if r['method']==method];groups=collections.defaultdict(list)
 for r in rr:groups[(r['drone'],str(r['seed']))].append(r)
 complete=[v for v in groups.values() if len(v)==2]
 out=dict(method=method,valid=len(rr),complete=len(complete),pass_sets=sum(all(truth(r['active_tracking_pass']) for r in v) for v in complete),pass_scenarios=sum(truth(r['active_tracking_pass']) for r in rr))
 for key in ['rise_s','settling_s','overshoot_pct','final_error_pct']:
  vals=[float(r[key]) for r in rr if r[key] not in ('',None)];out['median_'+key]=float(np.median(vals)) if vals else None
 return out
