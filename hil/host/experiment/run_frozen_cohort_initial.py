"""Frozen cohort runner: at most 3 infrastructure-only attempts per scenario.
All attempts retained; first transport-valid trial selected regardless of response quality.
No gains, reference commands, timing limits, or model equations changed.
"""
import os,sys,time,json,datetime,ctypes,gc,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from replay_hardware import scenario,ROOT,mavutil
RETRYABLE=('Host integration gap','Excessive actuator stream gap','Missed start','Board rejected trial: [2,','Incomplete hardware output coverage')
def drain(c,seconds):
 end=time.perf_counter()+seconds;heartbeat=None
 while time.perf_counter()<end:
  m=c.recv_match(blocking=True,timeout=.05)
  if m and m.get_type()=='HEARTBEAT' and m.get_srcSystem()==c.target_system:
   heartbeat=m
   if m.base_mode&128: raise RuntimeError('Unexpected armed board')
 return heartbeat

def main():
 planfile=ROOT/'experiment/hil_validation_plan.json';plan=json.loads(planfile.read_text());items=plan['items']
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
 dest=ROOT/'results'/('cohort_'+stamp);dest.mkdir();print('COHORT '+str(dest),flush=True)
 protocol={'planned':len(items),'max_attempts_per_scenario':3,'selection':'first transport-valid attempt, irrespective of controller quality','retryable_prefixes':RETRYABLE,'timing_gates_unchanged':True,'plan_sha256':hashlib.sha256(planfile.read_bytes()).hexdigest(),'status':'running','valid':0,'invalid_scenarios':0,'attempts':0,'next_index':0}
 (dest/'protocol.json').write_text(json.dumps(protocol,indent=2));(ROOT/'results/active_cohort.txt').write_text(str(dest))
 c=None;gc.disable();winmm=ctypes.windll.winmm;winmm.timeBeginPeriod(1)
 ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(),0x80)
 try:
  c=mavutil.mavlink_connection('COM4',baud=2000000,source_system=254,source_component=190);c.port.write_timeout=.5
  h=c.wait_heartbeat(timeout=10)
  if not h or h.autopilot!=12 or h.base_mode&128 or not h.base_mode&32:raise RuntimeError('Disarmed PX4 HIL heartbeat required')
  c.target_system=h.get_srcSystem();c.target_component=h.get_srcComponent()
  c.mav.command_long_send(c.target_system,c.target_component,511,0,93,1000,0,0,0,0,0)
  with (dest/'attempts.jsonl').open('x') as ledger:
   for index,item in enumerate(items):
    valid=False
    for attempt in range(1,4):
     trial=dest/f'{index:04d}_attempt{attempt}';trial.mkdir();entry={'index':index,'id':item['id'],'attempt':attempt,'directory':trial.name};protocol['attempts']+=1
     try:
      result=scenario(c,item,trial);entry.update(status='transport_valid',result_file=item['id']+'.json');valid=True
     except Exception as e:
      entry.update(status='infrastructure_failure',error=str(e));(trial/'FAILED.json').write_text(json.dumps(entry,indent=2))
      if not str(e).startswith(RETRYABLE):
       ledger.write(json.dumps(entry)+'\n');ledger.flush();raise
     ledger.write(json.dumps(entry)+'\n');ledger.flush()
     if valid:break
     if drain(c,4) is None:raise RuntimeError('No heartbeat after failed attempt')
    protocol['valid']+=int(valid);protocol['invalid_scenarios']+=int(not valid);protocol['next_index']=index+1
    (dest/'progress.json').write_text(json.dumps(protocol,indent=2))
    print(json.dumps({'processed':index+1,'total':len(items),'valid':protocol['valid'],'invalid':protocol['invalid_scenarios'],'attempts':protocol['attempts']}),flush=True)
   protocol['status']='completed_with_invalid' if protocol['invalid_scenarios'] else 'completed'
 except Exception as e:
  protocol.update(status='stopped',reason=str(e));raise
 finally:
  (dest/'progress.json').write_text(json.dumps(protocol,indent=2))
  if c:
   try:
    c.mav.command_long_send(c.target_system,c.target_component,511,0,93,0,0,0,0,0,0)
    c.mav.serial_control_send(10,0,0,0,0,[0]*70)
   finally:c.close()
  winmm.timeEndPeriod(1)
if __name__=='__main__':main()
