#include "study_controller.hpp"
#include <iostream>
#include <iomanip>
#include <cstdlib>
int main(int argc,char**argv){
 if(argc!=9)return 2;
 StudyArray<double,8> g;for(int i=0;i<8;i++)g[i]=std::strtod(argv[i+1],nullptr);
 StudyController c;c.reset(g);double roll,pitch,p,q,r,tr,tp,ty,th;
 std::cout<<std::setprecision(17);
 while(std::cin>>roll>>pitch>>p>>q>>r>>tr>>tp>>ty>>th){
  c.step(roll,pitch,{p,q,r},{tr,tp,ty},th);
  for(double v:c.motors)std::cout<<v<<' ';
  for(double v:c.raw)std::cout<<v<<' ';std::cout<<'\n';
 }
}
