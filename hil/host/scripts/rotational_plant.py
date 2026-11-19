"""Fixed-position rotational HIL fixture, FRD body / NED world.
Study torque and motor equations; no translational dynamics. Not free flight.
Quaternion maps body to NED. Motors use PX4 HIL channel order 1..4.
"""
import math
J=(.0211,.0219,.0366); ARM=.225; CT=1.105e-5; CM=1.779e-7

def euler(q):
 a,b,c,d=q
 return (math.atan2(2*(a*b+c*d),1-2*(b*b+c*c)),math.asin(max(-1,min(1,2*(a*c-b*d)))),math.atan2(2*(a*d+b*c),1-2*(c*c+d*d)))
def body_vector(q,v):
 a,b,c,d=q
 R=((1-2*(c*c+d*d),2*(b*c-a*d),2*(b*d+a*c)),(2*(b*c+a*d),1-2*(b*b+d*d),2*(c*d-a*b)),(2*(b*d-a*c),2*(c*d+a*b),1-2*(b*b+c*c)))
 return [sum(R[j][i]*v[j] for j in range(3)) for i in range(3)]
def rhs(x,u):
 a,b,c,d,p,q,r,w1,w2,w3,w4=x
 arm=math.sqrt(.5)*ARM*CT; spin=w1+w2-w3-w4
 mx=arm*(-w1*w1+w2*w2+w3*w3-w4*w4)-.0035*p*abs(p)+.0001287*q*spin
 my=arm*(w1*w1-w2*w2+w3*w3-w4*w4)-.0039*q*abs(q)-.0001287*p*spin
 mz=CM*(w1*w1+w2*w2-w3*w3-w4*w4)-.0034*r*abs(r)
 return [-.5*(p*b+q*c+r*d),.5*(p*a+r*c-q*d),.5*(q*a-r*b+p*d),.5*(r*a+q*b-p*c),(mx-(J[2]-J[1])*q*r)/J[0],(my-(J[0]-J[2])*p*r)/J[1],(mz-(J[1]-J[0])*p*q)/J[2]]+[(1148*u[i]-141.4-x[7+i])/.02 for i in range(4)]
class Plant:
 def __init__(self):self.x=[1.,0,0,0,0,0,0,0,0,0,0]
 def step(self,u,dt):
  if len(u)!=4 or not all(math.isfinite(v) and 0<=v<=1 for v in u):raise ValueError('Invalid motor controls')
  if not 0<dt<=.1:raise ValueError('Integration interval out of range')
  n=math.ceil(dt/.001);h=dt/n
  for _ in range(n):
   x=self.x;k1=rhs(x,u);k2=rhs([v+h*k/2 for v,k in zip(x,k1)],u);k3=rhs([v+h*k/2 for v,k in zip(x,k2)],u);k4=rhs([v+h*k for v,k in zip(x,k3)],u)
   self.x=[v+h*(a+2*b+2*c+d)/6 for v,a,b,c,d in zip(x,k1,k2,k3,k4)]
   norm=math.sqrt(sum(v*v for v in self.x[:4]));self.x[:4]=[v/norm for v in self.x[:4]]
 def sensors(self):
  return body_vector(self.x[:4],(0,0,-9.80665)),self.x[4:7],body_vector(self.x[:4],(.215,0,.427))
if __name__=='__main__':
 p=Plant()
 for _ in range(1000):p.step([.609]*4,.001)
 assert max(abs(v) for v in p.x[1:7])<1e-10
 expected=1148*.609-141.4
 assert max(abs(v-expected) for v in p.x[7:])<1e-8
 # Independent right-hand rule expectations for each individual rotor.
 for rotor,signs in enumerate(((-1,1,1),(1,-1,1),(1,1,-1),(-1,-1,-1))):
  x=[1.,0,0,0,0,0,0,0,0,0,0];x[7+rotor]=500
  accel=rhs(x,[0]*4)[4:7]
  assert all(v*s>0 for v,s in zip(accel,signs))
 q=[math.cos(math.pi/4),math.sin(math.pi/4),0,0]
 assert abs(body_vector(q,(0,0,-9.80665))[1]+9.80665)<1e-9
 print('PASS: symmetric equilibrium, analytic motor response, four rotor torque signs, FRD gravity rotation')
