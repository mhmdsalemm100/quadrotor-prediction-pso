"""Replay saved gains on custom, disarmed-only PX4 study firmware.
Pilot by default. Never arms, changes parameters, or falls back to software control.
"""
import os
os.environ['MAVLINK20']='1'
import argparse,ctypes,datetime,json,math,re,time,sys,faulthandler,io,gc
from contextlib import contextmanager

from pathlib import Path
import numpy as np
from numba import njit
from pymavlink import mavutil
from frozen_model import rhs,euler
ROOT=Path(__file__).resolve().parents[1]
@njit
def integrate(x,u,plant,elapsed):
 n=max(1,int(math.ceil(elapsed/.001)));h=elapsed/n
 k1=np.empty(11);k2=np.empty(11);k3=np.empty(11);k4=np.empty(11);d=np.zeros(3)
 for _ in range(n):
  rhs(x,u,plant,d,k1);rhs(x+h*k1/2,u,plant,d,k2);rhs(x+h*k2/2,u,plant,d,k3);rhs(x+h*k3,u,plant,d,k4)
  x=x+h*(k1+2*k2+2*k3+k4)/6
  x[:4]/=np.sqrt(np.sum(x[:4]**2))
 return x

def shell(c,command,x):
 data=(command+'\n').encode()
 for i in range(0,len(data),40):
  piece=data[i:i+40];c.mav.serial_control_send(10,6,0,0,len(piece),list(piece)+[0]*(70-len(piece)))
  until=time.perf_counter()+.006
  while time.perf_counter()<until:
   c.mav.hil_state_quaternion_send(time.time_ns()//1000,x[:4].tolist(),*x[4:7].tolist(),0,0,0,0,0,0,0,0,0,0,-1000)
   for _ in range(20):
    m=c.recv_match(blocking=False)
    if m is None: break
    if m.get_type()=='SERIAL_CONTROL': print(bytes(m.data[:m.count]).decode(errors='replace'),end='',flush=True)
   time.sleep(.001)

@contextmanager
def buffered_trace(path):
 buffer=io.StringIO()
 try: yield buffer
 finally:
  with path.open('x') as f: f.write(buffer.getvalue())
  buffer.close()

def scenario(c,item,dest):
 p=np.array(item['plant']);x=np.zeros(11);x[0]=1;x[7:]=p[11]*.609+p[12]
 u=np.full(4,.609);x=integrate(x,u,p,.001) # compile before hardware timing starts
 x[1:7]=0;u[:]=.609
 start=time.perf_counter();last=start;next_send=start;next_hb=start;last_hb=start
 sent=False;active=False;started=None;done=None;text='';records=[];intervals=[];ticks=[];last_output=None;first_tick=None
 with buffered_trace(dest/(item['id']+'.jsonl')) as log:
  while time.perf_counter()-start<12:
   now=time.perf_counter();elapsed=now-start
   if now-last_hb>4:raise RuntimeError('Heartbeat timeout')
   if active and last_output is not None and now-last_output>.05:raise RuntimeError('Study actuator stream lost')
   if now>=next_send:
    dt=now-last
    if active:
     if dt>.02:raise RuntimeError(f'Host integration gap {dt:.6f} s >20 ms')
     x=integrate(x,u,p,dt);intervals.append(dt)
    if not np.isfinite(x).all():raise RuntimeError('Nonfinite plant state')
    q=x[:4].tolist();rates=x[4:7].tolist()
    c.mav.hil_state_quaternion_send(time.time_ns()//1000,q,*rates,0,0,0,0,0,0,0,0,0,0,-1000)
    if active:
     row={'host_elapsed_s':now-started,'state':x.tolist(),'motors':u.tolist(),'latest_board_tick':ticks[-1]}
     records.append(row);log.write(json.dumps(row)+'\n')
    last=now;next_send=now+.001
   if now>=next_hb:c.mav.heartbeat_send(6,8,0,0,0);next_hb=now+1
   if elapsed>1 and not sent:
    cmd='psonn run '+' '.join(format(g,'.17g') for g in item['gains'])+f" {item['axis']} {item['amplitude_rad']:.17g}"
    shell(c,cmd,x);sent=True;now=time.perf_counter();last=now;next_send=now
   for _ in range(20):
    m=c.recv_match(blocking=False)
    if m is None:break
    if m.get_srcSystem()!=c.target_system:continue
    kind=m.get_type()
    if kind=='HEARTBEAT':
     last_hb=now
     if m.base_mode&128:raise RuntimeError('Unexpected armed board: aborting isolated study')
    elif kind=='HIL_ACTUATOR_CONTROLS' and m.flags==0x50534e4e:
     tick=int(round(m.controls[7]))
     if not active:
      active=True;started=now;first_tick=tick
      if tick>8:raise RuntimeError('Missed start of controller trial')
     if ticks and tick<=ticks[-1]:raise RuntimeError('Nonmonotonic board tick')
     if ticks and tick-ticks[-1]>20:raise RuntimeError('Excessive actuator stream gap')
     ticks.append(tick);u=np.array(m.controls[:4],dtype=float);last_output=now
     if not np.isfinite(u).all() or (u<0).any() or (u>1).any():raise RuntimeError('Invalid hardware motor command')
     log.write(json.dumps({'host_elapsed_s':now-started,'hardware_time_us':m.time_usec,'tick':tick,'motors':u.tolist(),'raw':list(m.controls[4:7])})+'\n')
    elif kind=='SERIAL_CONTROL':
     text+=bytes(m.data[:m.count]).decode('utf-8',errors='replace')
     found=re.search(r'PSNN_DONE code=(\d+) ticks=(\d+) max_late_us=(\d+) max_input_age_us=(\d+)',text)
     if found:done=[int(v) for v in found.groups()];break
   if done:break
   time.sleep(.0001)
 if done is None:raise RuntimeError('No PSNN_DONE from custom controller; firmware missing or trial incomplete. Shell: '+text[-600:])
 if done[0]!=0 or done[1]!=3201:raise RuntimeError('Board rejected trial: '+str(done))
 if not ticks or ticks[-1]<3190:raise RuntimeError('Incomplete hardware output coverage')
 result={'id':item['id'],'execution':'hardware','status':'recorded_requires_analysis','controller_completion':done,'first_board_tick':first_tick,'last_board_tick':ticks[-1],'gain_order':json.loads((ROOT/'configs/paper_config.json').read_text())['parameters'],'gains':item['gains'],'plant':item['plant'],'drone':item['drone'],'method':item['method'],'seed':item['seed'],'axis':item['axis'],'amplitude_rad':item['amplitude_rad'],'hardware_packets':len(ticks),'max_host_interval_s':max(intervals),'source':'ideal-state HIL; no estimator in feedback','note':'Transport checks only; scientific metrics require trace analysis.'}
 (dest/(item['id']+'.json')).write_text(json.dumps(result,indent=2));return result

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--port',default='COM4');ap.add_argument('--limit',type=int,default=1);ap.add_argument('--start',type=int,default=0);args=ap.parse_args()
 plan=json.loads((ROOT/'experiment/hil_validation_plan.json').read_text());items=plan['items'][args.start:args.start+args.limit]
 if not items or args.limit<1:raise ValueError('Empty validation selection')
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');dest=ROOT/'results'/('hardware_validation_'+stamp);dest.mkdir()
 c=None;winmm=ctypes.windll.winmm if os.name=='nt' else None
 if winmm:
  winmm.timeBeginPeriod(1)
  ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(),0x80)
 try:
  c=mavutil.mavlink_connection(args.port,baud=2000000,source_system=254,source_component=190)
  c.port.write_timeout=0.5
  h=c.wait_heartbeat(timeout=10)
  if not h or h.autopilot!=12 or h.base_mode&128 or not h.base_mode&32:raise RuntimeError('Disarmed PX4 HIL board required')
  c.target_system=h.get_srcSystem();c.target_component=h.get_srcComponent()
  c.mav.command_long_send(c.target_system,c.target_component,511,0,93,1000,0,0,0,0,0)
  gc.disable()
  for i,item in enumerate(items):
   result=scenario(c,item,dest);print(json.dumps({'completed':i+1,'requested':len(items),**result}),flush=True)
 except Exception as e:
  (dest/'STOPPED.json').write_text(json.dumps({'reason':str(e),'hardware_batch_complete':False},indent=2));raise
 finally:
  if c:
   c.mav.command_long_send(c.target_system,c.target_component,511,0,93,0,0,0,0,0,0)
   c.mav.serial_control_send(10,0,0,0,0,[0]*70);c.close()
  if winmm:winmm.timeEndPeriod(1)
if __name__=='__main__':main()


