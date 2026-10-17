"""Reproduce numerical tables and outward-rounded certificates."""
import sys,json,time,platform
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from heatpoles import *
from verified import *
import flint
ROOT=Path(__file__).resolve().parents[1]
p=Potential((.8,),(0.,));vp=VerifiedPotential(('4/5',),('0',))
results={'environment':{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'flint':flint.__version__,'precision_bits':ctx.prec},'widths':[],'joint':[],'contours':[],'asymptotics':[]}
start=time.perf_counter()
for tau in [.005,.01,.02,.05,.1,.2]:
    for eta in [.001,.01,.05,.1]:
        nominal=np.arccosh(np.exp(tau)/.8)
        root=brentq(lambda y:radius(p,np.pi+1j*y,tau,1,K=300)-eta,.00001,nominal,xtol=1e-14)
        index=int(np.floor(root*1e4));lo_text=f'{index/1e4:.4f}';hi_text=f'{(index+1)/1e4:.4f}'
        lo=float(lo_text);hi=float(hi_text)
        rlo=radius_enclosure(vp,arb.pi(),lo_text,str(tau),1,K=300)
        rhi=symmetric_witness_norm(hi_text,str(tau),1,K=300)
        assert rlo>A(str(eta)) and rhi<A(str(eta))
        # Global strip exclusion follows from the explicit monotonicity theorem;
        # run independent full-box certificate for representative cases.
        box=certify_box(vp,(0,2*arb.pi(),0,A(lo_text)),str(tau),1,str(eta),K=300)
        assert box['certified']
        results['widths'].append(dict(tau=tau,eta=eta,nominal=nominal,width=root,lower=lo,upper=hi,
            lower_radius=str(rlo),upper_witness_norm=str(rhi),box=box,
            positive_initial_lower=.2-eta*np.sqrt(np.pi**2/6-1)))
for y1,y2 in [(.55,.60),(.58,.60),(.50,.60),(.55,.65),(.58,.59)]:
    r1=radius_enclosure(vp,arb.pi(),str(y1),'.02',1,300)
    r2=radius_enclosure(vp,arb.pi(),str(y2),'.02',1,300)
    joint=symmetric_joint_norm([str(y1),str(y2)],'.02',1,300)
    results['joint'].append(dict(y1=y1,y2=y2,r1=str(r1),r2=str(r2),joint=str(joint),ratio=float(joint.mid())/max(float(r1.mid()),float(r2.mid()))))
for tau,rect,eta in [('.02',[(2.8,.68),(3.5,.68),(3.5,.78),(2.8,.78)],'.0001'),
                     ('.05',[(2.8,.71),(3.5,.71),(3.5,.84),(2.8,.84)],'.001'),
                     ('.1',[(2.8,.78),(3.5,.78),(3.5,.93),(2.8,.93)],'.001'),
                     ('.2',[(2.8,.91),(3.5,.91),(3.5,1.08),(2.8,1.08)],'.001')]:
    cert=certify_contour(vp,rect,tau,1,eta,300)
    assert cert['certified'] and cert['zeros_with_multiplicity']==1
    results['contours'].append(dict(tau=tau,vertices=rect,eta=eta,certificate=cert))
# Fixed off-axis point with nonzero p0; no fitting of the theoretical exponent.
z=2.+.6j
for tau in [.02,.01,.005,.0025,.00125,.000625]:
    K=int(np.ceil(.6/(2*tau)+15/np.sqrt(2*tau)))
    C,S=axes(.6,tau,1,K)
    saddle=(.25*(.6/(2*tau))**(-2)*np.exp(.6**2/(2*tau))*np.sqrt(np.pi/(2*tau)))
    # r asymptotic uses p(t,z), isolating the heat-kernel asymptotic.
    approx=abs(p.value(z,tau))/np.sqrt(saddle)
    r=radius(p,z,tau,1,K)
    results['asymptotics'].append(dict(tau=tau,C_ratio=C/saddle,S_ratio=S/saddle,radius=r,predicted=approx,ratio=r/approx,K=K))
# One explicit pole witness at pi+.59i, below nominal .7259i.
y=.59;theta,norm,res,rank=witness(p,[(np.pi+1j*y,0)],.02,1,180)
np.savez(ROOT/'results'/'witness.npz',theta=theta,N=1,tau=.02,y=y)
results['witness']={'y':y,'tau':.02,'norm':norm,'residual':res,'initial_positive_lower':.2-initial_sup_bound(theta,1),'rank':int(rank)}
results['elapsed_seconds']=time.perf_counter()-start
(ROOT/'results'/'study.json').write_text(json.dumps(results,indent=2))
print(json.dumps({'width_cases':len(results['widths']),'joint_cases':len(results['joint']),'contours':len(results['contours']),'witness':results['witness'],'asymptotics':results['asymptotics'],'seconds':results['elapsed_seconds']},indent=2))
