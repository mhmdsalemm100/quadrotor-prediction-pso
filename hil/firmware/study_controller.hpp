#pragma once

#include <math.h>
#include <stdint.h>
// Exact discrete controller from frozen study/model.py. No PX4 PID substitutions.
template<typename T, unsigned N> struct StudyArray {
 T data[N]{};
 T& operator[](unsigned i){return data[i];}
 const T& operator[](unsigned i)const{return data[i];}
 T* begin(){return data;} T* end(){return data+N;}
 const T* begin()const{return data;} const T* end()const{return data+N;}
 void fill(T value){for(unsigned i=0;i<N;i++)data[i]=value;}
};
struct StudyController {
 StudyArray<double,8> gain{};
 StudyArray<double,2> prev_error{},prev_raw{};
 StudyArray<double,3> raw{},control{};
 StudyArray<double,4> motors{};
 double yaw_integral=0.; uint64_t tick=0;
 static double clip(double x,double lo,double hi){return x<lo?lo:(x>hi?hi:x);}
 void reset(const StudyArray<double,8>& g,double throttle=.609){gain=g;prev_error={};prev_raw={};raw={};control={};motors.fill(throttle);yaw_integral=0;tick=0;}
 void step(double roll,double pitch,const StudyArray<double,3>& rates,const StudyArray<double,3>& target,double throttle=.609){
  constexpr double dt=.001,pole=.9997;
  const double angle[2]={roll,pitch};
  for(int j=0;j<2;j++){
   int i=3*j;double rate_ref=clip(gain[i]*(target[j]-angle[j]),-220.,220.);
   double error=rate_ref-rates[j];
   raw[j]=pole*prev_raw[j]+(gain[i+1]+gain[i+2]*dt)*error-pole*gain[i+1]*prev_error[j];
   prev_error[j]=error;prev_raw[j]=raw[j];control[j]=clip(raw[j],-1.,1.);
  }
  double ey=target[2]-rates[2];raw[2]=gain[6]*ey+yaw_integral;
  yaw_integral=clip(yaw_integral+dt*gain[7]*ey,-.2,.2);control[2]=clip(raw[2],-.5,.5);
  if(tick%4==0){double r=control[0],p=control[1],y=control[2];motors={clip(throttle-r+p+y,0.,1.),clip(throttle+r-p+y,0.,1.),clip(throttle+r+p-y,0.,1.),clip(throttle-r-p-y,0.,1.)};}
  ++tick;
 }
};
