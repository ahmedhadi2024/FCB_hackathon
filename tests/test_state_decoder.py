import importlib
import itertools
import unittest
import numpy as np
from fcb_external.temporal import decode as uniform


class StateDecoderTests(unittest.TestCase):
    def decode(self,p,seq,owner,dead):
        try: fn=importlib.import_module('fcb_external.state_decoder').decode
        except ModuleNotFoundError: self.fail('Separate transition-cost decoder is missing')
        return fn(np.asarray(p),np.asarray(seq),owner,dead)

    def test_uniform_costs_exactly_match_released_decoder(self):
        rng=np.random.default_rng(3); p=rng.dirichlet([1,1,1],120)
        seq=np.repeat([0,1],60)
        for cost in (0,2,5,10):
            np.testing.assert_array_equal(self.decode(p,seq,cost,cost),uniform(p,seq,cost))

    def test_owner_and_dead_live_costs_have_different_effects(self):
        p=np.array([[.98,.01,.01],[.01,.98,.01],[.98,.01,.01]])
        np.testing.assert_array_equal(self.decode(p,[0,0,0],0,10),list('HAH'))
        np.testing.assert_array_equal(self.decode(p,[0,0,0],10,10),list('HHH'))
        p=p[:,[0,2,1]]
        np.testing.assert_array_equal(self.decode(p,[0,0,0],10,0),list('HDH'))

    def test_global_optimum_matches_exhaustive_path_score(self):
        p=np.random.default_rng(4).dirichlet([2,2,2],5)
        owner,dead=.7,1.3
        pred=self.decode(p,np.zeros(5,int),owner,dead)
        path=[{'H':0,'A':1,'D':2}[v] for v in pred]
        def score(states):
            result=sum(np.log(p[i,k]) for i,k in enumerate(states))
            for a,b in zip(states,states[1:]):
                if a!=b: result-=dead if 2 in (a,b) else owner
            return result
        self.assertAlmostEqual(score(path),max(map(score,itertools.product(range(3),repeat=5))),places=12)

    def test_sequence_reset_and_team_exchange(self):
        p=np.array([[.99,.005,.005],[.99,.005,.005],[.005,.99,.005]])
        seq=np.array([0,0,1])
        result=self.decode(p,seq,100,100)
        np.testing.assert_array_equal(result,list('HHA'))
        swapped=self.decode(p[:,[1,0,2]],seq,100,100)
        np.testing.assert_array_equal(swapped,np.array([{'H':'A','A':'H','D':'D'}[x] for x in result]))

    def test_invalid_costs_and_probabilities_rejected(self):
        for owner,dead in ((-1,0),(0,np.nan),(np.inf,1)):
            with self.assertRaises(ValueError): self.decode([[1,0,0]],[0],owner,dead)
        with self.assertRaises(ValueError): self.decode([[.1,.1,.1]],[0],1,1)
