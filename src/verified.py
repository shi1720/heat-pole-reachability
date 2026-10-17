"""Outward-rounded certificates using FLINT/Arb, including infinite H1 tails.

All public scalar inputs may be exact rational/decimal strings. A float is
interpreted as its exact binary value, never as an exact decimal measurement.
"""
from flint import arb,acb,ctx
from dataclasses import dataclass
ctx.prec=160

def A(x):
    return x if isinstance(x,arb) else arb(x)

@dataclass(frozen=True)
class VerifiedPotential:
    a: tuple
    b: tuple
    def value(self,z,tau):
        tau=A(tau); f=acb(1)
        for k,(a,b) in enumerate(zip(self.a,self.b),1):
            f+=(-tau*k*k).exp()*(A(a)*(z*k).cos()+A(b)*(z*k).sin())
        return f


def axis_enclosures(y,tau,N,K=180):
    """C,S enclosed with geometric upper bound beyond K (H1 norm)."""
    tau=A(tau); y=abs(A(y))
    if not tau>0 or K<=N: raise ValueError('Require tau>0 and K>N')
    C=arb(0);S=arb(0)
    for k in range(N+1,K+1):
        d=(-2*tau*k*k).exp()/(k*k)
        C+=d*(y*k).cosh()**2
        S+=d*(y*k).sinh()**2
    L=K+1
    rho=(-2*tau*(2*L+1)+2*y).exp()
    if not rho<1: raise ValueError('Increase K beyond the Gaussian saddle')
    tail=(-2*tau*L*L+2*y*L).exp()/(L*L)/(1-rho)
    # The same upper bound applies to each nonnegative omitted sum.
    C+=arb(tail.upper()/2,tail.upper()/2)
    S+=arb(tail.upper()/2,tail.upper()/2)
    return C,S,tail


def radius_enclosure(p,x,y,tau,N,K=180):
    z=acb(A(x),A(y));f=p.value(z,tau)
    C,S,_=axis_enclosures(y,tau,N,K)
    if not S>0: raise ValueError('Use nonreal point')
    return (f.real**2/C+f.imag**2/S).sqrt()


def radius_lower_box(p,xa,xb,ya,yb,tau,N,K=180,circular=False):
    x=arb((xa+xb)/2,(xb-xa)/2)
    y=arb((ya+yb)/2,(yb-ya)/2)
    f=p.value(acb(x,y),tau)
    C,S,_=axis_enclosures(max(ya.abs_upper(),yb.abs_upper()),tau,N,K)
    r=f.real.abs_lower()**2/C.upper()
    if circular:
        r+=f.imag.abs_lower()**2/C.upper()
    elif S.upper()>0:
        r+=f.imag.abs_lower()**2/S.upper()
    return r.sqrt()


def certify_box(p,box,tau,N,eta,K=180,circular=False,maxdepth=24):
    """Prove every point of a closed rectangle is zero-free for every h.

    A depth limit returns unresolved, never a positive certificate.
    """
    eta=A(eta);stack=[(*map(A,box),0)];leaves=0;worst=None;max_depth=0
    while stack:
        xa,xb,ya,yb,depth=stack.pop()
        r=radius_lower_box(p,xa,xb,ya,yb,tau,N,K,circular)
        if r>eta:
            leaves+=1;worst=r.lower() if worst is None or r.lower()<worst else worst
            max_depth=max(max_depth,depth);continue
        if depth>=maxdepth:
            return {'certified':False,'leaves':leaves,'depth':depth}
        if xb-xa>=yb-ya:
            mid=(xa+xb)/2;stack.extend([(xa,mid,ya,yb,depth+1),(mid,xb,ya,yb,depth+1)])
        else:
            mid=(ya+yb)/2;stack.extend([(xa,xb,ya,mid,depth+1),(xa,xb,mid,yb,depth+1)])
    return {'certified':True,'leaves':leaves,'depth':max_depth,'radius_lower':str(worst)}


def certify_contour(p,vertices,tau,N,eta,K=180,circular=False,maxdepth=24):
    """Verify a continuous contour and nominal integer winding.

    Segment interval images exclude zero. Their convexity gives a homotopy
    to endpoint chords; principal argument increments count that polygon.
    Real-structure distance excludes zero for the entire perturbation homotopy.
    """
    v=[acb(A(x),A(y)) for x,y in vertices]
    stack=[(v[j],v[(j+1)%len(v)],0) for j in range(len(v))]
    phase=arb(0);leaves=0;min_r=None;max_depth=0
    while stack:
        za,zb,depth=stack.pop()
        xa=min(za.real,zb.real);xb=max(za.real,zb.real)
        ya=min(za.imag,zb.imag);yb=max(za.imag,zb.imag)
        box=acb(arb((xa+xb)/2,(xb-xa).abs_upper()/2),arb((ya+yb)/2,(yb-ya).abs_upper()/2))
        fbox=p.value(box,tau)
        r=radius_lower_box(p,xa,xb,ya,yb,tau,N,K,circular)
        fa=p.value(za,tau);fb=p.value(zb,tau)
        increment=(fb/fa).arg() if not fa.contains(0) else arb('nan')
        if not fbox.contains(0) and r>A(eta) and increment.is_finite() and increment.abs_upper()<arb.pi():
            phase+=increment;leaves+=1
            min_r=r.lower() if min_r is None or r.lower()<min_r else min_r
            max_depth=max(max_depth,depth)
        else:
            if depth>=maxdepth:
                return {'certified':False,'leaves':leaves,'depth':depth}
            mid=(za+zb)/2;stack.extend([(za,mid,depth+1),(mid,zb,depth+1)])
    wind=phase/(2*arb.pi());nearest=int(round(float(wind.mid())))
    if not (wind>A(nearest)-A('1/4') and wind<A(nearest)+A('1/4')):
        return {'certified':False,'winding_enclosure':str(wind)}
    return {'certified':True,'zeros_with_multiplicity':nearest,'winding_enclosure':str(wind),
            'radius_lower':str(min_r),'leaves':leaves,'depth':max_depth}


def symmetric_witness_norm(y,tau,N,K=180):
    """Exact finite cosine witness at pi+iy for center 1+(4/5)cos x."""
    tau=A(tau); y=A(y)
    f=1-A('4/5')*(-tau).exp()*y.cosh()
    C=sum(((-2*tau*k*k).exp()*(y*k).cosh()**2/(k*k) for k in range(N+1,K+1)),arb(0))
    return f.abs_upper()/C.sqrt() # upper enclosure on norm; f>0 below nominal root


def symmetric_joint_norm(ys,tau,N,K=180):
    """Infinite-tail exact Gram norm at pi+i*y_j; common observation time."""
    from flint import arb_mat
    tau=A(tau); ys=list(map(A,ys)); n=len(ys)
    _,_,tail=axis_enclosures(max(ys),tau,N,K)
    G=[]
    for y in ys:
        row=[]
        for z in ys:
            g=sum(((-2*tau*k*k).exp()*(y*k).cosh()*(z*k).cosh()/(k*k) for k in range(N+1,K+1)),arb(0))
            row.append(g+arb(tail.upper()/2,tail.upper()/2))
        G.append(row)
    v=arb_mat([[-(1-A('4/5')*(-tau).exp()*y.cosh())] for y in ys])
    q=(v.transpose()*arb_mat(G).solve(v))[0,0]
    return q.sqrt()
