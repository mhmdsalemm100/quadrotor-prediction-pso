"""Execute a fixed allowlist of read-only PX4 shell diagnostics over MAVLink."""
from pathlib import Path
from pymavlink import mavutil
import time,datetime,json
ROOT=Path(__file__).resolve().parents[1]
COMMANDS=('ver all','pwm_out_sim status','mavlink status','param show SYS_HITL','param show SYS_AUTOSTART','commander check','commander status')
def main():
    c=mavutil.mavlink_connection('COM4',baud=115200,source_system=254,source_component=190)
    output=[]
    try:
        c.mav.heartbeat_send(0,8,0,0,0)
        hb=c.wait_heartbeat(timeout=15)
        if hb is None or hb.autopilot!=mavutil.mavlink.MAV_AUTOPILOT_PX4:raise RuntimeError('PX4 heartbeat required')
        if hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:raise RuntimeError('Board is armed; stopping diagnostic')
        c.mav.serial_control_send(mavutil.mavlink.SERIAL_CONTROL_DEV_SHELL,6,0,0,1,[10]+[0]*69)
        time.sleep(.3)
        for cmd in COMMANDS:
            output.append('\n>>> '+cmd+'\n')
            data=(cmd+'\n').encode()
            c.mav.serial_control_send(mavutil.mavlink.SERIAL_CONTROL_DEV_SHELL,6,0,0,len(data),list(data)+[0]*(70-len(data)))
            end=time.monotonic()+3
            while time.monotonic()<end:
                m=c.recv_match(type='SERIAL_CONTROL',blocking=True,timeout=.25)
                if m and m.get_srcSystem()==hb.get_srcSystem():output.append(bytes(m.data[:m.count]).decode('utf-8',errors='replace'))
        text=''.join(output)
        stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        (ROOT/'reports'/('shell_'+stamp+'.txt')).write_text(text,encoding='utf-8')
        (ROOT/'reports/latest_shell.txt').write_text(text,encoding='utf-8')
        print(text)
    finally:
        try:c.mav.serial_control_send(mavutil.mavlink.SERIAL_CONTROL_DEV_SHELL,0,0,0,0,[0]*70)
        finally:c.close()
if __name__=='__main__':main()


