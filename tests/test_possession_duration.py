import unittest,itertools
import numpy as np
from fcb_external.temporal import decode as baseline

class DurationTests(unittest.TestCase):
    def test_optimum_and_disabled_equivalence(self):
        from fcb_external.possession_duration import decode
        p=np.random.default_rng(42).dirichlet([2,2,2],6);seq=np.zeros(6,int)
        np.testing.assert_array_equal(decode(p,seq,.5,3,0),baseline(p,seq,.5))
        labels=decode(p,seq,.5,3,2)
        def score(path):
            value=sum(np.log(p[i,s]) for i,s in enumerate(path));start=0
            for i in range(1,len(path)):
                if path[i]!=path[i-1]:
                    value-=.5
                    if start>0 and path[i-1]!=2 and i-start<3:value-=2
                    start=i
            return value
        path=[{'H':0,'A':1,'D':2}[x] for x in labels]
        self.assertAlmostEqual(score(path),max(score(x) for x in itertools.product(range(3),repeat=6)))
    def test_boundary_reset_and_team_symmetry(self):
        from fcb_external.possession_duration import decode
        p=np.array([[.98,.01,.01],[.01,.98,.01],[.01,.98,.01]])
        seq=np.array([0,1,1]);a=decode(p,seq,10,12,2)
        np.testing.assert_array_equal(a,list('HAA'))
        b=decode(p[:,[1,0,2]],seq,10,12,2)
        np.testing.assert_array_equal(b,[{'H':'A','A':'H','D':'D'}[x] for x in a])
