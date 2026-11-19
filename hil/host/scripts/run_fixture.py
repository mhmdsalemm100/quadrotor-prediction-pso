"""Bounded PX4 stock-controller rotational-fixture commissioning test.
Never force-arms or changes parameters. USB-only board required for --arm-usb-only.
This does not run the paper controller and must not be reported as its validation.
"""
import os
os.environ['MAVLINK20']='1'
import argparse,collections,datetime,json,math,time
from pathlib import Path
from pymavlink import mavutil
from rotational_plant import Plant,euler
ROOT=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--arm-usb-only',action='store_true');ap.add_argument('--port',default='COM4');a=ap.parse_args()
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
 report={'utc':stamp,'test':'stock_px4_rotational_fixture','paper_validation':False,'arming_requested':a.arm_usb_only,'events':[],'status':'starting'}
 c=None;system=component=None;armed=False;arm_sent=False;disarm_confirmed=None;last_hb=0;controls=[0.]*4;last_controls=0;counts=collections.Counter();plant=Plant();mode_sent=False;health_sent=set();arm_time=None;max_angle=0.;act_nonzero=0;last_att=None;max_gap=0
 def command(cmd,*args):
  c.mav.command_long_send(system,component,cmd,0,*(list(args)+[0]*(7-len(args))))
 try:
  c=mavutil.mavlink_connection(a.port,baud=115200,source_system=254,source_component=190)
  hb=c.wait_heartbeat(timeout=10)
  if not hb or hb.autopilot!=12 or hb.base_mode&128 or not hb.base_mode&32:raise RuntimeError('Initially disarmed PX4 HIL heartbeat required')
  system,component=hb.get_srcSystem(),hb.get_srcComponent()
  t0=time.monotonic();prev=t0;next_sensor=t0;next_slow=t0;last_hb=t0;last_controls=t0
  with (ROOT/'logs'/('fixture_'+stamp+'.jsonl')).open('w',encoding='utf-8') as log:
   while time.monotonic()-t0<65:
    now=time.monotonic();t=now-t0
    if now-last_hb>4:raise RuntimeError('Heartbeat timeout')
    if armed and now-last_controls>.25:raise RuntimeError('Actuator telemetry timeout')
    if now>=next_sensor:
     dt=now-prev;max_gap=max(max_gap,dt)
     if dt>.1:raise RuntimeError('Host scheduling gap exceeds 100 ms')
     if armed and dt>0:plant.step(controls,dt)
     acc,gyro,mag=plant.sensors()
     c.mav.hil_sensor_send(time.time_ns()//1000,*acc,*gyro,*mag,1013.25,0,0,20,8191)
     target=math.radians(5) if arm_time is not None and now-arm_time>3 else 0.
     quat=[math.cos(target/2),math.sin(target/2),0,0]
     c.mav.set_attitude_target_send(int(t*1000),system,component,7,quat,0,0,0,.5)
     angles=euler(plant.x[:4]);max_angle=max(max_angle,max(abs(v) for v in angles[:2]))
     if max_angle>math.radians(25) or max(abs(v) for v in plant.x[4:7])>5:raise RuntimeError('Fixture envelope exceeded')
     log.write(json.dumps({'t':t,'kind':'plant','state':plant.x,'controls':controls,'target_roll':target,'estimated_attitude':last_att})+'\n')
     prev=now;next_sensor=now+.005
    if now>=next_slow:
     c.mav.heartbeat_send(6,8,0,0,0)
     command(512,30);next_slow=now+1
    for when,cmd in ((8,'listener health_report -n 1'),(10,'listener failsafe_flags -n 1'),(12,'listener estimator_status_flags -n 1'),(14,'logger status'),(42,'listener health_report -n 1'),(44,'commander status'),(46,'listener failsafe_flags -n 1')):
     if t>when and when not in health_sent:
      data=(cmd+'\n').encode();c.mav.serial_control_send(10,6,0,0,len(data),list(data)+[0]*(70-len(data)));health_sent.add(when)
    if a.arm_usb_only and t>40 and not mode_sent:
     command(176,1,6,0);mode_sent=True;report['events'].append({'t':t,'action':'request_offboard'})
    if a.arm_usb_only and t>45 and not arm_sent:
     command(400,1,0);arm_sent=True;report['events'].append({'t':t,'action':'request_normal_arm'})
    if arm_sent and not armed and t>49:raise RuntimeError('Normal arming not confirmed; no force-arm attempted')
    if arm_time is not None and now-arm_time>8:break
    for _ in range(150):
     m=c.recv_match(blocking=False)
     if m is None:break
     if m.get_srcSystem()!=system or m.get_type()=='BAD_DATA':continue
     kind=m.get_type();counts[kind]+=1;data=m.to_dict()
     if kind in ('HEARTBEAT','COMMAND_ACK','STATUSTEXT','ATTITUDE','HIL_ACTUATOR_CONTROLS','ESTIMATOR_STATUS'):
      log.write(json.dumps({'t':t,'kind':kind,'message':data},default=str)+'\n')
     if kind=='HEARTBEAT':
      last_hb=now;was=armed;armed=bool(m.base_mode&128)
      if armed and arm_time is None:arm_time=now;report['events'].append({'t':t,'action':'armed_confirmed'})
      if was and not armed:raise RuntimeError('Unexpected disarm')
     elif kind=='HIL_ACTUATOR_CONTROLS':
      last_controls=now
      raw=list(m.controls[:4])
      if not all(math.isfinite(v) and 0<=v<=1 for v in raw):raise RuntimeError('Invalid motor output')
      controls=raw
      if any(v>0 for v in controls):act_nonzero+=1
     elif kind=='SERIAL_CONTROL':
      report['events'].append({'t':t,'shell':bytes(m.data[:m.count]).decode('utf-8',errors='replace')})
     elif kind=='ATTITUDE':last_att=[m.roll,m.pitch,m.yaw]
     elif kind in ('COMMAND_ACK','STATUSTEXT'):report['events'].append({'t':t,'message':data})
    time.sleep(.001)
  report['status']='bounded_loop_completed' if arm_time is not None and act_nonzero else 'disarmed_interface_completed'
 except Exception as e:report.update(status='stopped',reason=str(e))
 finally:
  if c and system is not None and arm_sent:
   # Normal disarm, verify heartbeat before releasing connection. No force command.
   command(400,0,0);end=time.monotonic()+5
   while time.monotonic()<end:
    m=c.recv_match(type='HEARTBEAT',blocking=True,timeout=.5)
    if m and m.get_srcSystem()==system and not(m.base_mode&128):disarm_confirmed=True;break
   if disarm_confirmed is None and a.arm_usb_only:
    report['events'].append({'action':'USB-only force disarm after normal disarm unconfirmed'})
    command(400,0,21196);end=time.monotonic()+5
    while time.monotonic()<end:
     m=c.recv_match(type='HEARTBEAT',blocking=True,timeout=.5)
     if m and m.get_srcSystem()==system and not(m.base_mode&128):disarm_confirmed=True;break
   if disarm_confirmed is None:
    disarm_confirmed=False;report['status']='cleanup_failed';report['reason']='Disarmed heartbeat not confirmed'
   command(176,1,1,0) # return to Manual; this is not a persistent parameter write
  if c:
   c.mav.serial_control_send(10,0,0,0,0,[0]*70)
   c.close()
  report.update(disarm_confirmed=disarm_confirmed,received_counts=dict(counts),nonzero_actuator_packets=act_nonzero,max_abs_roll_pitch_deg=math.degrees(max_angle),max_host_interval_ms=1000*max_gap)
  (ROOT/'reports'/('fixture_'+stamp+'.json')).write_text(json.dumps(report,indent=2,default=str))
  (ROOT/'reports/latest_fixture.json').write_text(json.dumps(report,indent=2,default=str))
  print(json.dumps({k:v for k,v in report.items() if k!='events'},indent=2,default=str))
 if report['status'] in ('stopped','cleanup_failed'):raise SystemExit(1)
if __name__=='__main__':main()



