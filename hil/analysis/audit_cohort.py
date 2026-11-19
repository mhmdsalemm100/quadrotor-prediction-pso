from pathlib import Path
import json,hashlib,collections
R=Path(__file__).resolve().parents[1];D=Path((R/'results/active_cohort.txt').read_text());planpath=R/'experiment/hil_validation_plan.json';items=json.loads(planpath.read_text())['items'];protocol=json.loads((D/'protocol.json').read_text());ledger=[json.loads(s) for s in (D/'attempts.jsonl').read_text().splitlines()];seen=set();selected=set();errors=[]
if hashlib.sha256(planpath.read_bytes()).hexdigest()!=protocol['plan_sha256']:errors.append('Frozen plan hash mismatch')
for e in ledger:
 key=(e['index'],e['attempt'])
 if key in seen:errors.append('Repeated attempt '+str(key))
 seen.add(key)
 if e['attempt'] not in (1,2,3):errors.append('Attempt limit exceeded')
 item=items[e['index']]
 if e['id']!=item['id']:errors.append('Scenario identity mismatch')
 if e['status']!='transport_valid':continue
 if e['index'] in selected:errors.append('Multiple selected results')
 selected.add(e['index']);p=D/e['directory']/e['result_file'];m=json.loads(p.read_text())
 for k in ('id','gains','plant','drone','method','seed','axis','amplitude_rad'):
  if m[k]!=item[k]:errors.append(f'{key} mismatched {k}')
 code,ticks,late,age=m['controller_completion']
 if code!=0 or ticks!=3201 or late>5000 or age>20000 or m['max_host_interval_s']>.02:errors.append('Invalid timing accepted '+str(key))
 if not p.with_suffix('.jsonl').exists():errors.append('Missing trace '+str(key))
out={'audit_pass':not errors,'ledger_entries_checked':len(ledger),'selected_results_checked':len(selected),'errors':errors,'note':'Structural/provenance audit only; does not establish controller quality or complete cohort coverage.'}
(D/'integrity_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if errors:raise SystemExit(1)
