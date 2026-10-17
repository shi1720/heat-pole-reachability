import sys,json
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from heatpoles import *
from verified import *
ROOT=Path(__file__).resolve().parents[1]
p=Potential((.8,),(0.,));vp=VerifiedPotential(('4/5',),('0',));out=[]
for N in [1,16,32,64,96,128,160]:
    root=brentq(lambda y:radius(p,np.pi+1j*y,.005,N,400)-.01,.01,np.arccosh(np.exp(.005)/.8),xtol=1e-14)
    index=int(np.floor(root*1e6));lo_text=f'{index/1e6:.6f}';hi_text=f'{(index+1)/1e6:.6f}'
    lo=float(lo_text);hi=float(hi_text)
    rlo=radius_enclosure(vp,arb.pi(),lo_text,'.005',N,400)
    nominal=(A('.005').exp()/A('4/5')).acosh()
    if hi>float(nominal.upper()):
        rhi=A(0) # nominal potential has a zero below the stated upper height
    else:
        rhi=symmetric_witness_norm(hi_text,'.005',N,400)
    assert rlo>A('.01') and rhi<A('.01')
    out.append(dict(N=N,width=root,lower=lo,upper=hi,lower_radius=str(rlo),upper_witness_norm=str(rhi)))
(ROOT/'results/resolution.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
