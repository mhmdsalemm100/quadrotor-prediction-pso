from pathlib import Path
import subprocess
R=Path(r'SOURCE_HIL/firmware/PX4-Autopilot')
files=subprocess.check_output(['git','-C',str(R),'ls-files','--recurse-submodules','-z']).decode().split('\0');n=0
for name in files:
 if not name or not(name.endswith(('.sh','.mk')) or Path(name).name.startswith('Makefile')):continue
 p=R/name
 if not p.is_file() or p.is_symlink():continue
 b=p.read_bytes()
 if b'\r\n' in b:p.write_bytes(b.replace(b'\r\n',b'\n'));n+=1
print('Normalized shell/Make files:',n,flush=True)
old=R/'build/px4_fmu-v6c_default';dest=R/'build/failed_windows_path_build'
if old.exists() and not dest.exists():old.rename(dest)
