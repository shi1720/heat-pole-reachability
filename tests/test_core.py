import unittest,sys
from pathlib import Path
import numpy as np
import mpmath as mp
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from heatpoles import *
from verified import *

class TestCore(unittest.TestCase):
    def test_distance_against_independent_svd(self):
        rng=np.random.default_rng(646)
        p=Potential((.4,.1),(.1,-.05))
        for _ in range(240):
            N=int(rng.integers(0,6));K=N+int(rng.integers(5,50))
            tau=float(rng.uniform(.005,.2));z=rng.uniform(0,6.2)+1j*rng.uniform(.03,.9)
            _,n,res,_=witness(p,[(z,0)],tau,N,K)
            self.assertLess(res,1e-10)
            self.assertAlmostEqual(n/radius(p,z,tau,N,K),1,places=10)

    def test_arb_infinite_tail_against_high_precision(self):
        mp.mp.dps=90
        rng=np.random.default_rng(2027)
        for _ in range(60):
            tau=str(rng.uniform(.005,.2));y=str(rng.uniform(.01,.9));N=int(rng.integers(0,8))
            # Deliberately small cutoffs exercise non-negligible tails.
            K=max(N+2,int(float(y)/(2*float(tau)))+8)
            C,S,_=axis_enclosures(y,tau,N,K)
            tt=mp.mpf(tau);yy=mp.mpf(y)
            c=mp.fsum(mp.exp(-2*tt*k*k)*mp.cosh(k*yy)**2/(k*k) for k in range(N+1,600))
            s=mp.fsum(mp.exp(-2*tt*k*k)*mp.sinh(k*yy)**2/(k*k) for k in range(N+1,600))
            self.assertTrue(C.contains(arb(mp.nstr(c,85))))
            self.assertTrue(S.contains(arb(mp.nstr(s,85))))

    def test_shared_constraints_and_conjugacy(self):
        p=Potential((.8,),(0.,));z=2.4+.6j
        a,n,res,rank=witness(p,[(z,0)],.02,1,100)
        b,m,res2,rank2=witness(p,[(z,0),(z.conjugate(),0)],.02,1,100)
        np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-12)
        self.assertEqual(rank,rank2);self.assertEqual(rank,2)
        _,j,r,_=witness(p,[(np.pi+.58j,0),(np.pi+.60j,0)],.02,1,100)
        self.assertGreater(j,.04);self.assertLess(r,1e-12)
        q=symmetric_joint_norm(['.58','.6'],'.02',1,180)
        self.assertLess(abs(float(q.mid())-j),1e-12)

    def test_derivative_constraints(self):
        p=Potential((.8,),(0.,));z=2.8+.7j
        theta,n,res,rank=witness(p,[(z,0),(z,1)],.02,1,100)
        self.assertLess(res,1e-11);self.assertEqual(rank,4)
        for j in [0,1]:
            self.assertLess(abs(p.value(z,.02,j)+perturbation(theta,z,.02,1,derivative=j)),1e-11)

    def test_certificate_refuses_ambiguous_region(self):
        p=VerifiedPotential(('4/5',),('0',))
        c=certify_box(p,(A('3.14'),A('3.15'),A('.58'),A('.60')),'.02',1,'.02',180,maxdepth=5)
        self.assertFalse(c['certified'])
        good=certify_contour(p,[(2.8,.68),(3.5,.68),(3.5,.78),(2.8,.78)],'.02',1,'.0001')
        self.assertTrue(good['certified']);self.assertEqual(good['zeros_with_multiplicity'],1)
        reverse=certify_contour(p,list(reversed([(2.8,.68),(3.5,.68),(3.5,.78),(2.8,.78)])),'.02',1,'.0001')
        self.assertEqual(reverse['zeros_with_multiplicity'],-1)

if __name__=='__main__':unittest.main()
