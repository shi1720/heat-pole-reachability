"""Continuous contour checks for asymmetric positive trigonometric centers."""
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from heatpoles import *
from verified import *
ROOT=Path(__file__).resolve().parents[1]
centers=[('two-mode',('1/2','3/20'),('0','2/25')),
         ('three-mode',('2/5','1/10','1/20'),('1/20','-2/25','3/100'))]
out=[]
for name,aa,bb in centers:
    a=tuple(float(A(v).mid()) for v in aa);b=tuple(float(A(v).mid()) for v in bb)
    p=Potential(a,b);vp=VerifiedPotential(aa,bb);M=len(a)
    for tau in [.02,.05,.1]:
        poly=np.zeros(2*M+1,dtype=complex);poly[M]=1
        for k,(ak,bk) in enumerate(zip(a,b),1):
            poly[M+k]=(ak-1j*bk)*np.exp(-tau*k*k)/2
            poly[M-k]=(ak+1j*bk)*np.exp(-tau*k*k)/2
        roots=np.roots(poly[::-1]);zs=[(np.angle(w)%(2*np.pi))-1j*np.log(abs(w)) for w in roots if abs(w)<1]
        z=min(zs,key=lambda z:z.imag)
        x,y=z.real,z.imag;d=.06
        rect=[(x-d,y-d),(x+d,y-d),(x+d,y+d),(x-d,y+d)]
        samples=[]
        for j in range(4):
            z1=complex(*rect[j]);z2=complex(*rect[(j+1)%4])
            samples.extend(radius(p,z1+(z2-z1)*s,tau,M,300) for s in np.linspace(0,1,40))
        eta=min(min(samples)*.2,p.positivity_lower/(2*np.sqrt(1/M)))
        eta=float(f'{eta:.7g}')
        cert=certify_contour(vp,rect,str(tau),M,str(eta),300)
        assert cert['certified'] and cert['zeros_with_multiplicity']==1
        out.append(dict(center=name,a=aa,b=bb,tau=tau,N=M,eta=eta,nominal_pole=[x,y],vertices=rect,
                        initial_positive_lower=p.positivity_lower-eta/np.sqrt(M),certificate=cert))
(ROOT/'results'/'general.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
