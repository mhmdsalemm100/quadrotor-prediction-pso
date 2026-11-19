import os,time,json
os.environ['MAVLINK20']='1'
from pathlib import Path
from pymavlink import mavutil
root=Path(__file__).resolve().parents[1]
c=mavutil.mavlink_connection('COM4',baud=2000000,source_system=254,source_component=190)
c.port.write_timeout=.5
report=[]
try:
 h=c.wait_heartbeat(timeout=10)
 if not h or h.base_mode&128: raise RuntimeError('Disarmed heartbeat required')
 for rate in [0,1000,1001]:
  if rate==1001: c.mav.command_long_send(h.get_srcSystem(),h.get_srcComponent(),511,0,93,1000,0,0,0,0,0)
  print('STAGE',rate,flush=True); start=time.monotonic(); next_send=start; count=0; received=0
  while time.monotonic()-start<3:
   now=time.monotonic()
   if rate and now>=next_send:
    c.mav.hil_state_quaternion_send(time.time_ns()//1000,[1,0,0,0],0,0,0,0,0,0,0,0,0,0,0,0,0,-1000)
    count+=1;next_send=now+1/rate
   m=c.recv_match(blocking=False)
   if m: received+=1
   time.sleep(.001)
  report.append(dict(rate=rate,sent=count,received=received));print(report[-1],flush=True)
except Exception as e:
 report.append(dict(error=str(e)));raise
finally:
 c.close();(root/'reports/study_transport_probe.json').write_text(json.dumps(report,indent=2))
