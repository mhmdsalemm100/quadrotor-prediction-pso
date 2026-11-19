"""Disarmed HIL sensor-path probe. No arming, mode, parameter or firmware writes.
Stationary FRD body at sea level. This is an interface test, not a plant model.
"""
import os
os.environ['MAVLINK20'] = '1'
import argparse, collections, datetime, json, math, statistics, time
from pathlib import Path
from pymavlink import mavutil
ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port', default='COM4')
    p.add_argument('--seconds', type=float, default=30)
    p.add_argument('--self-test', action='store_true')
    a = p.parse_args()
    if not 1 <= a.seconds <= 60: p.error('Duration must be 1..60 seconds')
    if a.self_test:
        from pymavlink.dialects.v20 import common
        encoder = common.MAVLink(None)
        msg = common.MAVLink_hil_sensor_message(1000, 0,0,-9.80665, 0,0,0, .215,0,.427, 1013.25,0,0,20,8191)
        packet = msg.pack(encoder)
        decoded = common.MAVLink(None).parse_char(packet)
        assert decoded.get_type() == 'HIL_SENSOR'
        assert abs(decoded.zacc + 9.80665) < 1e-5 and decoded.fields_updated == 8191
        print('PASS: MAVLink 2 stationary sensor packet round trip; no hardware accessed')
        return
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    report = {'utc':stamp, 'port':a.port, 'test':'stationary_disarmed_sensor_path', 'status':'starting', 'performance_result':False}
    counts = collections.Counter(); samples = {}; sent_times=[]; c=None
    path = ROOT/'reports'/('sensor_probe_'+stamp+'.json')
    try:
        c=mavutil.mavlink_connection(a.port, baud=115200, source_system=254, source_component=190)
        deadline=time.monotonic()+15; hb=None
        while time.monotonic()<deadline:
            m=c.recv_match(type='HEARTBEAT',blocking=True,timeout=.5)
            if m and m.autopilot == 12: hb=m; break
        if hb is None: raise RuntimeError('PX4 heartbeat not received')
        if hb.base_mode & 128: raise RuntimeError('Board armed: refusing sensor injection')
        if not hb.base_mode & 32: raise RuntimeError('HIL heartbeat flag absent: refusing sensor injection')
        system,component=hb.get_srcSystem(),hb.get_srcComponent()
        report.update(system_id=system,component_id=component,initially_disarmed=True,hil_flag=True)
        start=time.monotonic(); next_sensor=start; next_request=start; last_hb=start
        with (ROOT/'logs'/('sensor_probe_'+stamp+'.jsonl')).open('w',encoding='utf-8') as log:
            while time.monotonic()-start < a.seconds:
                now=time.monotonic()
                if now-last_hb > 4: raise RuntimeError('Heartbeat lost: stopped sensor injection')
                if now >= next_sensor:
                    c.mav.hil_sensor_send(time.time_ns()//1000,0,0,-9.80665,0,0,0,.215,0,.427,1013.25,0,0,20,8191)
                    sent_times.append(now); next_sensor=now+.01
                if now >= next_request:
                    c.mav.heartbeat_send(0,8,0,0,0)
                    for mid in (30,31,105,230):
                        c.mav.command_long_send(system,component,512,0,mid,0,0,0,0,0,0)
                    next_request=now+1
                for _ in range(100):
                    m=c.recv_match(blocking=False)
                    if m is None: break
                    if m.get_srcSystem()!=system or m.get_type()=='BAD_DATA': continue
                    kind=m.get_type(); counts[kind]+=1; payload=m.to_dict()
                    samples[kind]=payload
                    log.write(json.dumps({'elapsed_s':time.monotonic()-start,'message':payload},default=str)+'\n')
                    if kind=='HEARTBEAT':
                        last_hb=time.monotonic()
                        if m.base_mode & 128: raise RuntimeError('Armed heartbeat detected: stopped sensor injection')
                time.sleep(.001)
        report['status']='completed_disarmed'
    except Exception as e:
        report.update(status='error',error=str(e))
    finally:
        if c: c.close()
        gaps=[1000*(b-a) for a,b in zip(sent_times,sent_times[1:])]
        report.update(sent_sensor_packets=len(sent_times),received_counts=dict(counts),latest_messages=samples)
        if gaps:
            report['send_timing']={'mean_interval_ms':statistics.mean(gaps),'max_interval_ms':max(gaps),'effective_hz':1000/statistics.mean(gaps)}
        report['interpretation']='Receiving actuator messages alone does not prove sensor acceptance. Inspect HIGHRES_IMU and estimator/attitude messages. No controller or optimization performance measured.'
        path.write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
        (ROOT/'reports/latest_sensor_probe.json').write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
        print(json.dumps({k:v for k,v in report.items() if k!='latest_messages'},indent=2))
        print('Saved:',path)
    if report['status']=='error': raise SystemExit(1)
if __name__=='__main__': main()
