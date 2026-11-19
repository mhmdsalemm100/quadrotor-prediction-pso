"""Read-only MAVLink inventory. Never arms, flashes, or changes parameters."""
from pathlib import Path
import argparse, json, time, datetime, collections, sys, struct
from serial.tools import list_ports
from pymavlink import mavutil
ROOT = Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--port',required=True); ap.add_argument('--baud',type=int,default=115200); ap.add_argument('--timeout',type=float,default=20); args=ap.parse_args()
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    report={'utc':stamp,'port':args.port,'baud':args.baud,'ports':[{'port':p.device,'description':p.description,'vid':p.vid,'pid':p.pid,'serial_number':p.serial_number,'manufacturer':p.manufacturer} for p in list_ports.comports()], 'status':'connecting','parameters':{},'messages':{},'commands_sent':[]}
    path=ROOT/'reports'/('device_'+stamp+'.json')
    conn=None
    try:
        conn=mavutil.mavlink_connection(args.port,baud=args.baud,source_system=254,source_component=190,autoreconnect=False)
        deadline=time.monotonic()+args.timeout; hb=None; counts=collections.Counter()
        with (ROOT/'logs'/('telemetry_'+stamp+'.jsonl')).open('w',encoding='utf-8') as log:
            def receive():
                msg=conn.recv_match(blocking=True,timeout=.5)
                if msg is None or msg.get_type()=='BAD_DATA':return None
                kind=msg.get_type();counts[kind]+=1
                entry={'time_monotonic':time.monotonic(),'system':msg.get_srcSystem(),'component':msg.get_srcComponent(),'message':msg.to_dict()}
                log.write(json.dumps(entry,default=str)+'\n')
                return msg
            while time.monotonic()<deadline:
                m=receive()
                if m and m.get_type()=='HEARTBEAT' and m.autopilot!=mavutil.mavlink.MAV_AUTOPILOT_INVALID:
                    hb=m;break
            if hb is None:
                report['status']='no_autopilot_heartbeat';return 2
            system,component=hb.get_srcSystem(),hb.get_srcComponent()
            report.update(system_id=system,component_id=component,heartbeat=hb.to_dict(),autopilot=mavutil.mavlink.enums['MAV_AUTOPILOT'][hb.autopilot].name,armed=bool(hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED))
            conn.target_system=system;conn.target_component=component
            conn.mav.command_long_send(system,component,mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE,0,148,0,0,0,0,0,0)
            report['commands_sent'].append('REQUEST_MESSAGE AUTOPILOT_VERSION (148)')
            names=['SYS_AUTOSTART','SYS_HITL','SYS_MC_EST_GROUP','MAV_SYS_ID'] if hb.autopilot==mavutil.mavlink.MAV_AUTOPILOT_PX4 else []
            for name in names:
                conn.mav.param_request_read_send(system,component,name.encode(),-1)
                report['commands_sent'].append('PARAM_REQUEST_READ '+name)
            deadline=time.monotonic()+12
            while time.monotonic()<deadline:
                m=receive()
                if not m or m.get_srcSystem()!=system:continue
                kind=m.get_type()
                if kind=='AUTOPILOT_VERSION':
                    report['version']=m.to_dict();v=m.flight_sw_version
                    report['firmware_version']='%d.%d.%d (type %d)'%((v>>24)&255,(v>>16)&255,(v>>8)&255,v&255)
                elif kind=='PARAM_VALUE':
                    name=m.param_id.decode(errors='replace') if isinstance(m.param_id,bytes) else m.param_id
                    value=m.param_value
                    if hb.autopilot==mavutil.mavlink.MAV_AUTOPILOT_PX4 and m.param_type in (1,2,3,4,5,6):
                        fmt={1:'B',2:'b',3:'H',4:'h',5:'I',6:'i'}[m.param_type]
                        raw=struct.pack('<f',m.param_value)
                        value=struct.unpack('<'+fmt,raw[:struct.calcsize(fmt)])[0]
                    report['parameters'][name.rstrip('\x00')]=value
            report['messages']=dict(counts);report['status']='identified';return 0
    except Exception as e:
        report.update(status='error',error=str(e));return 1
    finally:
        if conn:conn.close()
        path.write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
        (ROOT/'reports/latest_device.json').write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
        print(json.dumps(report,indent=2,default=str));print('Saved:',path)
if __name__=='__main__':sys.exit(main())

