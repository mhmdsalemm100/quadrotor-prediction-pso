import os
os.environ['MAVLINK20']='1'
import time,json
from pathlib import Path
from pymavlink import mavutil
r=Path(__file__).resolve().parents[1]
c=mavutil.mavlink_connection('COM4',baud=115200,source_system=254,source_component=190)
report={'purpose':'USB-only simulated-flight cleanup','physical_motors_connected':False}
try:
 h=c.wait_heartbeat(timeout=10)
 if h is None:raise RuntimeError('No heartbeat')
 report['initially_armed']=bool(h.base_mode&128)
 if h.base_mode&128:
  # No physical motors or battery: terminate simulated armed state after normal disarm was unconfirmed.
  c.mav.command_long_send(h.get_srcSystem(),h.get_srcComponent(),400,0,0,21196,0,0,0,0,0)
  report['force_disarm_sent']=True
 end=time.monotonic()+10;confirmed=False
 while time.monotonic()<end:
  m=c.recv_match(type=['HEARTBEAT','COMMAND_ACK'],blocking=True,timeout=.5)
  if not m:continue
  if m.get_type()=='COMMAND_ACK':report['ack']=m.to_dict()
  if m.get_type()=='HEARTBEAT' and m.get_srcSystem()==h.get_srcSystem() and not m.base_mode&128:
   confirmed=True;report['final_heartbeat']=m.to_dict();break
 report['disarm_confirmed']=confirmed
 if not confirmed:raise RuntimeError('Disarm NOT confirmed')
finally:
 c.close();(r/'reports/final_board_state.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
