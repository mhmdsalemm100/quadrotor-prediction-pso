from pathlib import Path
import json,hashlib
R=Path(r'SOURCE_HIL');S=R/'firmware/PX4-Autopilot'
files=['src/examples/psonn/psonn.cpp','src/examples/psonn/study_controller.hpp','src/examples/psonn/CMakeLists.txt','src/examples/psonn/Kconfig','msg/PsonnState.msg','msg/PsonnOutput.msg','msg/CMakeLists.txt','src/modules/mavlink/mavlink_receiver.h','src/modules/mavlink/mavlink_receiver.cpp','src/modules/mavlink/streams/HIL_ACTUATOR_CONTROLS.hpp','boards/px4/fmu-v6c/default.px4board','boards/px4/fmu-v6c/nuttx-config/nsh/defconfig']
report={'upstream_commit':'d6f12ad1c4f70ad3230afd7d86e971421e02fef4','target':'px4_fmu-v6c_default','interface':'psonn run; simulation-only output; disarmed HIL required','hardware_flashed':False,'source_hashes':{f:hashlib.sha256((S/f).read_bytes()).hexdigest() for f in files}}
(R/'firmware/study_firmware_manifest.json').write_text(json.dumps(report,indent=2))
p=R/'scripts/install_firmware_module.py';s=p.read_text(encoding='utf-8-sig');s=s.replace("shutil.copy2(R/'firmware/study_controller.hpp'", "if 'PsonnState.msg' in (S/'msg/CMakeLists.txt').read_text():raise RuntimeError('Patch already applied; refusing duplicate source modifications')\nshutil.copy2(R/'firmware/study_controller.hpp'",1);p.write_text(s)
print('Recorded 11 modified/new firmware source hashes')

