#include <px4_platform_common/px4_config.h>
#include <px4_platform_common/log.h>
#include <px4_platform_common/posix.h>
#include <drivers/drv_hrt.h>
#include <parameters/param.h>
#include <uORB/Publication.hpp>
#include <uORB/Subscription.hpp>
#include <uORB/topics/psonn_state.h>
#include <uORB/topics/psonn_output.h>
#include <uORB/topics/vehicle_status.h>
#include <cstdlib>
#include <cstring>
#include "study_controller.hpp"
extern "C" __EXPORT int psonn_main(int argc,char *argv[]);
int psonn_main(int argc,char *argv[]) {
 if(argc!=12 || strcmp(argv[1],"run")!=0){PX4_INFO("run g0 g1 g2 g3 g4 g5 g6 g7 axis amplitude_rad");return 1;}
 int32_t hil=0;param_get(param_find("SYS_HITL"),&hil);
 uORB::Subscription status_sub{ORB_ID(vehicle_status)};vehicle_status_s status{};
 if(hil!=1 || !status_sub.copy(&status) || status.arming_state==vehicle_status_s::ARMING_STATE_ARMED){PX4_ERR("requires HIL and DISARMED board");return 1;}
 const double lo[8]={1,.005,.0001,1,.005,.0001,.01,.0001};
 const double hi[8]={30,.35,1.5,30,.35,1.5,1,.5};
 StudyArray<double,8> gains{};
 for(int i=0;i<8;i++){char *end=nullptr;gains[i]=strtod(argv[2+i],&end);if(*end || !std::isfinite(gains[i]) || gains[i]<lo[i] || gains[i]>hi[i]){PX4_ERR("invalid gains");return 1;}}
 const int axis=atoi(argv[10]);char *end=nullptr;const double amplitude=strtod(argv[11],&end);
 if(axis<0 || axis>2 || *end || !std::isfinite(amplitude) || fabs(amplitude)>(axis==2?1.75:.35)){PX4_ERR("invalid target");return 1;}
 uORB::Subscription state_sub{ORB_ID(psonn_state)};psonn_state_s state{};
 if(!state_sub.copy(&state) || hrt_elapsed_time(&state.timestamp)>20000){PX4_ERR("fresh simulated state required");return 1;}
 StudyController controller;controller.reset(gains);
 uORB::Publication<psonn_output_s> pub{ORB_ID(psonn_output)};
 psonn_output_s out{};out.running=true;
 const uint64_t start=hrt_absolute_time();uint64_t max_late=0,max_age=0;int result=0;unsigned completed=0;
 PX4_INFO("PSNN_BEGIN axis=%d",axis);
 for(unsigned n=0;n<=3200;n++){
  const uint64_t deadline=start+1000ULL*n;uint64_t now=hrt_absolute_time();
  if(now<deadline){px4_usleep(deadline-now);now=hrt_absolute_time();}
  const uint64_t late=now>deadline?now-deadline:0;if(late>max_late)max_late=late;
  state_sub.copy(&state);status_sub.copy(&status);const uint64_t age=hrt_elapsed_time(&state.timestamp);if(age>max_age)max_age=age;
  if(age>20000 || late>5000 || status.arming_state==vehicle_status_s::ARMING_STATE_ARMED){result=2;break;}
  const double a=state.q[0],b=state.q[1],c=state.q[2],d=state.q[3];
  const double norm=a*a+b*b+c*c+d*d;
  if(!std::isfinite(norm) || fabs(norm-1)>0.01){result=2;break;}
  const double roll=atan2(2*(a*b+c*d),1-2*(b*b+c*c));
  const double pitch=asin(StudyController::clip(2*(a*c-b*d),-1,1));
  StudyArray<double,3> rates{{state.rates[0],state.rates[1],state.rates[2]}},target{};
  if(!std::isfinite(rates[0]) || !std::isfinite(rates[1]) || !std::isfinite(rates[2])){result=2;break;}
  if(n>=200)target[axis]=amplitude;
  controller.step(roll,pitch,rates,target);
  out.timestamp=hrt_absolute_time();out.tick=n;
  for(int i=0;i<4;i++)out.motors[i]=controller.motors[i];
  for(int i=0;i<3;i++)out.raw[i]=controller.raw[i];
  pub.publish(out);completed=n+1;
 }
 out.running=false;out.timestamp=hrt_absolute_time();pub.publish(out);
 PX4_INFO("PSNN_DONE code=%d ticks=%u max_late_us=%llu max_input_age_us=%llu",result,completed,(unsigned long long)max_late,(unsigned long long)max_age);
 return result;
}
