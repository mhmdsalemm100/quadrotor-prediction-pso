from pathlib import Path
import shutil
R=Path(r'SOURCE_HIL');S=R/'firmware/PX4-Autopilot';D=S/'src/examples/psonn';D.mkdir(exist_ok=True)
if 'PsonnState.msg' in (S/'msg/CMakeLists.txt').read_text():raise RuntimeError('Patch already applied; refusing duplicate source modifications')
if 'PsonnState.msg' in (S/'msg/CMakeLists.txt').read_text():raise RuntimeError('Patch already applied; refusing duplicate source modifications')
shutil.copy2(R/'firmware/study_controller.hpp',D/'study_controller.hpp')
(D/'CMakeLists.txt').write_text('px4_add_module(MODULE examples__psonn MAIN psonn STACK_MAIN 5000 SRCS psonn.cpp DEPENDS px4_work_queue)\n')
(D/'Kconfig').write_text('menuconfig EXAMPLES_PSONN\n\tbool "psonn study controller (simulation only)"\n\tdefault n\n')
(S/'msg/PsonnState.msg').write_text('uint64 timestamp\nuint64 source_time\nfloat32[4] q\nfloat32[3] rates\n')
(S/'msg/PsonnOutput.msg').write_text('uint64 timestamp\nuint32 tick\nfloat32[4] motors\nfloat32[3] raw\nbool running\n')
p=S/'msg/CMakeLists.txt';s=p.read_text();s=s.replace('set(msg_files\n','set(msg_files\n\tPsonnState.msg\n\tPsonnOutput.msg\n',1);p.write_text(s)
p=S/'boards/px4/fmu-v6c/default.px4board';s=p.read_text();p.write_text(s+'\nCONFIG_EXAMPLES_PSONN=y\n')
p=S/'src/modules/mavlink/mavlink_receiver.h';s=p.read_text();s=s.replace('#pragma once','#pragma once\n#include <uORB/topics/psonn_state.h>',1)
# This header uses an include guard in some releases.
if '#include <uORB/topics/psonn_state.h>' not in s:s='#include <uORB/topics/psonn_state.h>\n'+s
anchor='uORB::Publication<vehicle_attitude_s>'
pos=s.index(anchor);s=s[:pos]+'uORB::Publication<psonn_state_s> _psonn_state_pub{ORB_ID(psonn_state)};\n\t'+s[pos:];p.write_text(s)
p=S/'src/modules/mavlink/mavlink_receiver.cpp';s=p.read_text();anchor='mavlink_msg_hil_state_quaternion_decode(msg, &hil_state);';assert s.count(anchor)==1
s=s.replace(anchor,anchor+'\n\t// Ideal-state input for the isolated study controller; only dispatched in HIL.\n\tpsonn_state_s study_state{};\n\tstudy_state.timestamp = hrt_absolute_time();\n\tstudy_state.source_time = hil_state.time_usec;\n\tfor (int i=0;i<4;i++) { study_state.q[i]=hil_state.attitude_quaternion[i]; }\n\tstudy_state.rates[0]=hil_state.rollspeed; study_state.rates[1]=hil_state.pitchspeed; study_state.rates[2]=hil_state.yawspeed;\n\t_psonn_state_pub.publish(study_state);',1);p.write_text(s)
p=S/'src/modules/mavlink/streams/HIL_ACTUATOR_CONTROLS.hpp';s=p.read_text();s=s.replace('#include <uORB/topics/actuator_outputs.h>','#include <uORB/topics/actuator_outputs.h>\n#include <uORB/topics/psonn_output.h>')
s=s.replace('return _act_sub.advertised() ?', 'return (_act_sub.advertised() || _study_sub.advertised()) ?')
s=s.replace('uORB::Subscription _act_sub{','uORB::Subscription _study_sub{ORB_ID(psonn_output)};\n\tuORB::Subscription _act_sub{',1)
anchor='actuator_outputs_s act;'
s=s.replace(anchor,'''psonn_output_s study{};
        if (_study_sub.copy(&study) && study.running) {
            if (!_study_sub.updated()) { return false; }
            _study_sub.update(&study);
            mavlink_hil_actuator_controls_t msg{};
            msg.time_usec=study.timestamp;
            for(int i=0;i<4;i++) { msg.controls[i]=study.motors[i]; }
            for(int i=0;i<3;i++) { msg.controls[4+i]=study.raw[i]; }
            msg.controls[7]=study.tick;
            msg.mode=MAV_MODE_FLAG_HIL_ENABLED;
            msg.flags=0x50534e4e;
            mavlink_msg_hil_actuator_controls_send_struct(_mavlink->get_channel(), &msg);
            return true;
        }
        '''+anchor,1)
# Subscription::copy can consume update state: check updated before copy.
s=s.replace('if (_study_sub.copy(&study) && study.running) {\n            if (!_study_sub.updated()) { return false; }\n            _study_sub.update(&study);','const bool study_updated=_study_sub.updated();\n        if (_study_sub.copy(&study) && study.running) {\n            if (!study_updated) { return false; }')
p.write_text(s)
print('Installed custom messages, simulation-only MAVLink output and module build configuration')
