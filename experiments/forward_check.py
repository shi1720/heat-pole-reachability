"""Independent dealiased Fourier RK4 Burgers evolution on the real line."""
import sys,json,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from heatpoles import *
ROOT=Path(__file__).resolve().parents[1]
p=Potential((.8,),(0.,));data=np.load(ROOT/'results'/'witness.npz');theta=data['theta']
nu=.1;T=.2
out=[]
for n,dt in [(64,.0002),(128,.0002),(256,.0002),(256,.0001)]:
    x=np.arange(n)*2*np.pi/n
    def exact(t):
        f=p.value(x,nu*t);f1=p.value(x,nu*t,1)
        for j,z in enumerate(x):
            f[j]+=perturbation(theta,z,nu*t,1)
            f1[j]+=perturbation(theta,z,nu*t,1,derivative=1)
        return (-2*nu*f1/f).real
    k=np.fft.fftfreq(n,1/n);keep=abs(k)<n/3
    def rhs(v):
        vx=np.fft.ifft(1j*k*np.fft.fft(v)).real
        nonlinear=np.fft.fft(v*vx)*keep
        return np.fft.ifft(-nonlinear-nu*k*k*np.fft.fft(v)).real
    u=exact(0);steps=round(T/dt);t0=time.perf_counter()
    for _ in range(steps):
        a=rhs(u);b=rhs(u+dt*a/2);c=rhs(u+dt*b/2);d=rhs(u+dt*c)
        u+=dt*(a+2*b+2*c+d)/6
    ref=exact(T);err=np.max(abs(u-ref))
    out.append(dict(n=n,dt=dt,steps=steps,max_error=err,seconds=time.perf_counter()-t0))
assert out[-1]['max_error']<1e-8
(ROOT/'results'/'forward.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
