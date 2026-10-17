"""Real Sobolev-tail pseudozeros of periodic Cole--Hopf heat potentials.

Floating-point exploration. Rigorous enclosures live in verified.py.
Norm convention: ||h||_s^2 = sum k^(2s) (a_k^2+b_k^2).
"""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Potential:
    a: tuple
    b: tuple
    def value(self, z, tau, derivative=0):
        z = np.asarray(z, dtype=complex)
        ans = np.ones_like(z) if derivative == 0 else np.zeros_like(z)
        for k, (a,b) in enumerate(zip(self.a,self.b),1):
            ans = ans + np.exp(-tau*k*k)*k**derivative*(a*np.cos(k*z+derivative*np.pi/2)+b*np.sin(k*z+derivative*np.pi/2))
        return ans
    @property
    def positivity_lower(self):
        return 1-sum(np.hypot(a,b) for a,b in zip(self.a,self.b))


def evaluation(z, tau, N, K, s=1, derivative=0):
    k=np.arange(N+1,K+1,dtype=float)
    d=np.exp(-tau*k*k)*k**(derivative-s)
    return np.concatenate([d*np.cos(k*z+derivative*np.pi/2), d*np.sin(k*z+derivative*np.pi/2)])


def axes(y,tau,N,K=200,s=1):
    k=np.arange(N+1,K+1,dtype=float)
    q=k**(-2*s)
    plus=np.exp(-2*tau*k*k+2*k*y)
    minus=np.exp(-2*tau*k*k-2*k*y)
    center=np.exp(-2*tau*k*k)
    return np.sum(q*(plus+minus+2*center))/4, np.sum(q*(plus+minus-2*center))/4


def radius(p,z,tau,N,K=200,s=1):
    c,d=axes(abs(z.imag),tau,N,K,s)
    f=complex(p.value(z,tau))
    if d==0:
        return abs(f.real)/np.sqrt(c) if f.imag==0 else np.inf
    return np.sqrt(f.real**2/c+f.imag**2/d)


def circular_radius(p,z,tau,N,K=200,s=1):
    c,_=axes(abs(z.imag),tau,N,K,s)
    return abs(p.value(z,tau))/np.sqrt(c)


def witness(p, constraints, tau,N,K=200,s=1):
    """Least-norm real initial perturbation; (z,derivative) constraints.

    Independent row-based SVD handles redundant conjugate constraints.
    Returns normalized coefficients theta=(k^s a,k^s b).
    """
    rows=[]; target=[]
    for z,order in constraints:
        e=evaluation(z,tau,N,K,s,order)
        f=complex(p.value(z,tau,order))
        rows.extend([e.real,e.imag]);target.extend([-f.real,-f.imag])
    L=np.array(rows);v=np.array(target)
    theta,_,rank,_=np.linalg.lstsq(L,v,rcond=1e-13)
    return theta, np.linalg.norm(theta), np.linalg.norm(L@theta-v),rank


def perturbation(theta,z,tau,N,s=1,derivative=0):
    K=N+len(theta)//2
    return evaluation(z,tau,N,K,s,derivative)@theta


def initial_sup_bound(theta,N,s=1):
    K=N+len(theta)//2
    k=np.arange(N+1,K+1,dtype=float)
    return np.sum(np.hypot(theta[:K-N],theta[K-N:])/k**s)
