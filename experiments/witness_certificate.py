"""Ball arithmetic for the exact finite cosine pole-producing witness."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from verified import *
ROOT=Path(__file__).resolve().parents[1]
tau=A('.02');y=A('.59');K=180;N=1
f=1-A('4/5')*(-tau).exp()*y.cosh()
C=sum(((-2*tau*k*k).exp()*(k*y).cosh()**2/(k*k) for k in range(N+1,K+1)),A(0))
a=[-f*((-1)**k)*(-tau*k*k).exp()*(k*y).cosh()/(k*k*C) for k in range(N+1,K+1)]
norm=sum((k*k*q*q for k,q in zip(range(2,K+1),a)),A(0)).sqrt()
positive=A('1/5')-sum((abs(q) for q in a),A(0))
z=acb(arb.pi(),y);res=acb(1)+A('4/5')*(-tau).exp()*z.cos()
der=-A('4/5')*(-tau).exp()*z.sin()
for k,q in zip(range(2,K+1),a):
    res+=q*(-tau*k*k).exp()*(k*z).cos()
    der-=k*q*(-tau*k*k).exp()*(k*z).sin()
assert norm<A('.01') and positive>A('.1960') and res.contains(0) and abs(der)>0
out={'formula':'a_k=-p(pi+i*y,tau)*(-1)^k*exp(-tau*k^2)*cosh(k*y)/(k^2*C_K); b_k=0',
     'N':N,'K':K,'tau':'.02','y':'.59','norm_enclosure':str(norm),'initial_positive_lower':str(positive),
     'zero_residual_enclosure':str(res),'derivative_magnitude_enclosure':str(abs(der))}
(ROOT/'results/witness_certificate.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
