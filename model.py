"""Nonlinear rotational subsystem transcribed from the supplied SLX equations.

The rotation/motor subsystem is autonomous: translation does not feed its torque
equations in this source. Position is therefore unnecessary for attitude tuning.
No small-angle approximation is used in the optimization simulator.
"""
import math
import numpy as np
from numba import njit

# x = [quaternion w,x,y,z; body rates p,q,r; four motor angular speeds].
# plant = [Jxx,Jyy,Jzz,R,Ct,Cm,Cdm_x,Cdm_y,Cdm_z,Jrotor,tau,Cr,Wb]
NOMINAL = np.array([.0211, .0219, .0366, .225, 1.105e-5, 1.779e-7,
                    .0035, .0039, .0034, .0001287, .02, 1148., -141.4])
POLE = .9997
BASE_KP = .03223 / POLE

def baseline(controller_dt=.001):
    # Exactly reconstruct b=[.03248,-.03223], a=[1,-.9997].
    ki = (.03248 - BASE_KP) / controller_dt
    return np.array([5.0119, BASE_KP, ki, 5.0119, BASE_KP, ki, .5, .01])

def coefficients(gains, controller_dt=.001, pole=POLE):
    return {axis: {'numerator': [float(gains[i+1]+gains[i+2]*controller_dt),
                                float(-pole*gains[i+1])],
                   'denominator': [1.0, -pole]}
            for axis, i in [('roll', 0), ('pitch', 3)]}

def rc_condition(rc):
    """Source's four-channel RC clipping/dead zone, then angle/rate/throttle map."""
    rc = np.clip(np.asarray(rc, float), 1100., 1900.)
    v = np.where(rc > 1540., (rc-1540.)/360.,
                 np.where(rc < 1460., (rc-1460.)/360., 0.))
    t = (v[2]+1.)*.5
    t = 2*t*.609 if t < .5 else .609+2*(t-.5)*(1-.609)
    return np.array([v[0]*35*np.pi/180, v[1]*35*np.pi/180,
                     v[3]*200*np.pi/180, t])

@njit(cache=True)
def euler(x):
    a,b,c,d = x[0],x[1],x[2],x[3]
    roll = math.atan2(2*(a*b+c*d), 1-2*(b*b+c*c))
    pitch = math.asin(min(1., max(-1., 2*(a*c-b*d))))
    yaw = math.atan2(2*(a*d+b*c), 1-2*(c*c+d*d))
    return roll,pitch,yaw

@njit(cache=True)
def rhs(x, u, plant, disturbance, out):
    a,b,c,d = x[0],x[1],x[2],x[3]
    p,q,r = x[4],x[5],x[6]
    w1,w2,w3,w4 = x[7],x[8],x[9],x[10]
    Jx,Jy,Jz,R,Ct,Cm = plant[:6]
    arm = math.sqrt(.5)*R*Ct
    Mx = arm*(-w1*w1+w2*w2+w3*w3-w4*w4)
    My = arm*(w1*w1-w2*w2+w3*w3-w4*w4)
    Mz = Cm*(w1*w1+w2*w2-w3*w3-w4*w4)
    spin = w1+w2-w3-w4
    # Source Add1 adds propeller, aerodynamic and gyroscopic moments.
    Mx += -plant[6]*p*abs(p) + plant[9]*q*spin + disturbance[0]
    My += -plant[7]*q*abs(q) - plant[9]*p*spin + disturbance[1]
    Mz += -plant[8]*r*abs(r) + disturbance[2]
    out[0] = -.5*(p*b+q*c+r*d)
    out[1] = .5*(p*a+r*c-q*d)
    out[2] = .5*(q*a-r*b+p*d)
    out[3] = .5*(r*a+q*b-p*c)
    out[4] = (Mx-(Jz-Jy)*q*r)/Jx
    out[5] = (My-(Jx-Jz)*p*r)/Jy
    out[6] = (Mz-(Jy-Jx)*p*q)/Jz
    for j in range(4):
        # Deliberately preserve the source's negative low-throttle target.
        out[7+j] = (plant[11]*u[j]+plant[12]-x[7+j])/plant[10]

@njit(cache=True, nogil=True)
def simulate(gains, target, plant, throttle, dt, control_dt, mixer_dt,
             pre_step, horizon, initial_motors_zero=False, store=False,
             pulse_axis=-1, pulse_amplitude=0., pulse_start=1.2, pulse_end=1.3,
             noise_amplitude=0., pole=POLE, band=.02, final_hold=.25, inner=False):
    """RK4 plant with sampled controller and held mixer outputs.

    Returns metrics[3,10], diagnostics[7], trace[:,36].
    Metrics: rise,settle,overshoot,final_abs_error,IAE,ITAE,peak_abs,
             reached_90,settled,final_signed_error. Ratios normalize by
             command amplitude; inactive axes by 5 deg / 30 deg/s.
    Trace: t,roll,pitch,yaw,p,q,r,refs(3),rate_refs(3),raw_control(3),
           control(3),raw_motor(4),motor(4),min_motor_speed,motor_speed(4),quat(4).
    """
    nc = int(round(control_dt/dt)); nm = int(round(mixer_dt/dt))
    steps = int(round((pre_step+horizon)/dt))
    start = int(round(pre_step/dt))
    x = np.zeros(11); x[0] = 1.
    if not initial_motors_zero:
        x[7:] = plant[11]*throttle+plant[12]
    k1=np.empty(11); k2=np.empty(11); k3=np.empty(11); k4=np.empty(11)
    temp=np.empty(11); dist=np.zeros(3)
    ref=np.zeros(3); desired_rate=np.zeros(3)
    prev_error=np.zeros(2); prev_output=np.zeros(2)
    raw=np.zeros(3); ctrl=np.zeros(3)
    motors=np.full(4, throttle); rawmot=np.full(4, throttle)
    integral_yaw=0.
    metrics=np.zeros((3,10)); diag=np.zeros(7)
    t10=np.full(3,-1.); t90=np.full(3,-1.); lastoutside=np.full(3,-dt)
    peak=np.zeros(3); finalsum=np.zeros(3); tail_count=0
    denom=np.empty(3); active=np.zeros(3, dtype=np.bool_)
    for j in range(3):
        active[j] = abs(target[j]) > 1e-12
        denom[j] = abs(target[j]) if active[j] else (5*math.pi/180 if j<2 else 30*math.pi/180)
    trace=np.zeros((steps+1 if store else 1,36))
    count=0; clipcount=0; ctrlclip=0; rateclip=0; effort=0.; maxbank=0.
    minomega=1e9; failed=False
    for n in range(steps+1):
        t=n*dt
        roll,pitch,yaw=euler(x)
        if n>=start:
            ref[:] = target
        if n % nc == 0:
            # Fixed deterministic measurement perturbations used only in holdout.
            er0 = ref[0]-roll-noise_amplitude*math.sin(61*t)
            er1 = ref[1]-pitch-noise_amplitude*math.sin(73*t+.3)
            rp0 = gains[0]*er0; rp1 = gains[3]*er1
            if inner:
                rp0=ref[0]; rp1=ref[1]
            desired_rate[0]=min(220.,max(-220.,rp0))
            desired_rate[1]=min(220.,max(-220.,rp1))
            desired_rate[2]=ref[2]
            for j in range(2):
                idx=3*j
                error=desired_rate[j]-x[4+j]-noise_amplitude*math.sin(89*t+j)
                b0=gains[idx+1]+gains[idx+2]*control_dt
                b1=-pole*gains[idx+1]
                raw[j]=pole*prev_output[j]+b0*error+b1*prev_error[j]
                prev_error[j]=error
                prev_output[j]=raw[j]  # No invented anti-windup.
                ctrl[j]=min(1.,max(-1.,raw[j]))
            ey=ref[2]-x[6]-noise_amplitude*math.sin(97*t)
            # Source forward-Euler integrator outputs its OLD state this sample.
            raw[2]=gains[6]*ey+integral_yaw
            integral_yaw=min(.2,max(-.2,integral_yaw+control_dt*gains[7]*ey))
            ctrl[2]=min(.5,max(-.5,raw[2]))
        if n % nm == 0:
            R,P,Y=ctrl[0],ctrl[1],ctrl[2]
            rawmot[0]=throttle-R+P+Y
            rawmot[1]=throttle+R-P+Y
            rawmot[2]=throttle+R+P-Y
            rawmot[3]=throttle-R-P-Y
            for j in range(4):
                motors[j]=min(1.,max(0.,rawmot[j]))
        maxbank=max(maxbank,abs(roll),abs(pitch))
        minomega=min(minomega,x[7],x[8],x[9],x[10])
        if n>=start:
            tau=t-pre_step
            count+=1
            clipped=False
            for j in range(4):
                if rawmot[j]<-1e-10 or rawmot[j]>1+1e-10: clipped=True
                effort += (motors[j]-throttle)**2 / 4
            if clipped: clipcount+=1
            if abs(raw[0])>1 or abs(raw[1])>1 or abs(raw[2])>.5: ctrlclip+=1
            if abs(desired_rate[0])>=220 or abs(desired_rate[1])>=220: rateclip+=1
            ys=np.array([roll,pitch,x[6]])
            if inner:
                ys[0]=x[4]; ys[1]=x[5]
            for j in range(3):
                err=(target[j]-ys[j])/denom[j]
                ae=abs(err)
                metrics[j,4]+=ae*dt
                metrics[j,5]+=tau*ae*dt
                metrics[j,6]=max(metrics[j,6],abs(ys[j]))
                if ae>band: lastoutside[j]=tau
                if active[j]:
                    progress=ys[j]/target[j]
                    peak[j]=max(peak[j],progress)
                    if t10[j]<0 and progress>=.1: t10[j]=tau
                    if t90[j]<0 and progress>=.9: t90[j]=tau
                if tau>=horizon-final_hold:
                    finalsum[j]+=err
            if tau>=horizon-final_hold: tail_count+=1
        if store:
            trace[n,0]=t
            trace[n,1]=roll; trace[n,2]=pitch; trace[n,3]=yaw
            trace[n,4:7]=x[4:7]
            trace[n,7:10]=ref
            trace[n,10:13]=desired_rate
            trace[n,13:16]=raw
            trace[n,16:19]=ctrl
            trace[n,19:23]=rawmot
            trace[n,23:27]=motors
            trace[n,27]=min(x[7],x[8],x[9],x[10])
            trace[n,28:32]=x[7:11]
            trace[n,32:36]=x[:4]
        if max(abs(x[4]),abs(x[5]),abs(x[6]))>30 or abs(pitch)>1.4 or not np.isfinite(x).all():
            failed=True
            if store: trace[n+1:]=np.nan
            break
        if n==steps: break
        dist[:]=0.
        if pulse_axis>=0 and pulse_start<=t<pulse_end: dist[pulse_axis]=pulse_amplitude
        rhs(x,motors,plant,dist,k1)
        for j in range(11): temp[j]=x[j]+dt*.5*k1[j]
        rhs(temp,motors,plant,dist,k2)
        for j in range(11): temp[j]=x[j]+dt*.5*k2[j]
        rhs(temp,motors,plant,dist,k3)
        for j in range(11): temp[j]=x[j]+dt*k3[j]
        rhs(temp,motors,plant,dist,k4)
        for j in range(11): x[j]+=dt/6*(k1[j]+2*k2[j]+2*k3[j]+k4[j])
        norm=math.sqrt(x[0]**2+x[1]**2+x[2]**2+x[3]**2)
        x[:4]/=norm
    for j in range(3):
        metrics[j,0] = t90[j]-t10[j] if t90[j]>=0 and t10[j]>=0 else horizon
        settled = lastoutside[j] < horizon-final_hold
        metrics[j,1] = max(0.,lastoutside[j]+dt) if settled else horizon
        metrics[j,2] = max(0.,peak[j]-1.) if active[j] else 0.
        metrics[j,3] = abs(finalsum[j]/max(1,tail_count))
        metrics[j,7] = 1. if t90[j]>=0 else 0.
        metrics[j,8] = 1. if settled else 0.
        metrics[j,9] = finalsum[j]/max(1,tail_count)
    diag[0]=clipcount/max(1,count); diag[1]=ctrlclip/max(1,count)
    diag[2]=rateclip/max(1,count); diag[3]=effort/max(1,count)
    diag[4]=maxbank; diag[5]=1. if failed else 0.
    diag[6]=minomega
    return metrics,diag,trace

def run(gains, scenario, config, store=False):
    dt=float(scenario.get('dt', config['integration_dt_s']))
    tc=float(scenario.get('control_dt', config['controller_dt_s']))
    tm=float(scenario.get('mixer_dt', config['mixer_dt_s']))
    pre=scenario.get('pre_step',config['pre_step_s'])
    for value in [tc,tm,pre]:
        if abs(value/dt-round(value/dt))>1e-8 or value<dt:
            raise ValueError('Sample times and pre-step must be positive integer multiples of integration dt.')
    g=np.asarray(gains).copy()
    # Timing sensitivity keeps the actual exported numerator coefficients fixed.
    if tc != config['controller_dt_s']:
        g[2]*=config['controller_dt_s']/tc
        g[5]*=config['controller_dt_s']/tc
    return simulate(g, np.deg2rad(scenario['target']),
                    np.asarray(scenario.get('plant',NOMINAL)), scenario.get('throttle',.609),
                    dt,tc,tm,pre,scenario.get('horizon',config['response_horizon_s']),
                    scenario.get('startup',False),store,
                    scenario.get('pulse_axis',-1),scenario.get('pulse_amplitude',0.),
                    scenario.get('pulse_start',1.2),scenario.get('pulse_end',1.3),
                    scenario.get('noise_amplitude',0.),config['fixed_discrete_pole'],
                    config['settling_band_fraction'],config['minimum_final_hold_s'],
                    scenario.get('inner',False))

def rotation_matrix(q):
    a,b,c,d=q
    return np.array([[a*a+b*b-c*c-d*d,2*(b*c-a*d),2*(b*d+a*c)],
                     [2*(b*c+a*d),a*a-b*b+c*c-d*d,2*(c*d-a*b)],
                     [2*(b*d-a*c),2*(c*d+a*b),a*a-b*b-c*c+d*d]])

def full_rhs(state, motor_commands, plant=NOMINAL, mass=1.4, gravity=9.8, drag=.055):
    """Complete 17-state source equations, available for separate trajectory work.

    State: rotation/motors (11), body velocity (3), earth position NED (3).
    This confirms mathematically why translation is excluded from GA fitness.
    """
    out=np.zeros(17)
    rhs(state[:11], np.asarray(motor_commands), plant, np.zeros(3), out[:11])
    R=rotation_matrix(state[:4]); vb=state[11:14]
    fb=R.T@np.array([0.,0.,mass*gravity])-.5*drag*vb*np.abs(vb)
    fb[2]-=plant[4]*np.sum(state[7:11]**2)
    out[11:14]=fb/mass-np.cross(state[4:7],vb)
    out[14:17]=R@vb
    return out
