"""Fictional tracking only: no competition fixtures or ground truth."""
import unittest
import numpy as np
from fcb_external import tracking_faults, team_features_v6
from fcb_external.schema import Observations
from fcb_external.possession_convention import shift_dead_edges


def scene(n=20000):
    rng=np.random.default_rng(0)
    players=rng.normal(0,15,(n,22,2)).astype(np.float32)
    return Observations('SYNTHETIC',np.ones(n,np.int8),np.arange(1,n+1),np.arange(n)/25,players,
        np.ones((n,22),bool),np.repeat([0,1],11).astype(np.int8),np.zeros((n,2),np.float32),
        np.ones(n,bool),np.zeros(n,np.int32),105.,68.,25.)


class TrackingFaultTests(unittest.TestCase):
    def test_profile_is_seeded_and_close_to_the_measured_one(self):
        obs=scene(); a=tracking_faults.apply(obs,3); b=tracking_faults.apply(obs,3)
        np.testing.assert_array_equal(a.player_observed,b.player_observed)
        self.assertEqual(a.player_xy_m.shape,(20000,24,2))
        for team in (0,1):
            count=a.player_observed[:,a.player_team==team].sum(axis=1)
            self.assertTrue(0.2<(count<11).mean()<0.5)
            self.assertTrue(0.01<(count>11).mean()<0.15)
            self.assertGreaterEqual(count.min(),5)
        np.testing.assert_array_equal(a.ball_xy_m,obs.ball_xy_m)
        self.assertTrue(np.isfinite(team_features_v6.extract(a)).all())

    def test_disabled_and_invalid(self):
        obs=scene(500); same=tracking_faults.apply(obs,1,missing_share=0,extra_share=0)
        np.testing.assert_array_equal(same.player_xy_m[:,:22],obs.player_xy_m)
        self.assertFalse(same.player_observed[:,22:].any())
        with self.assertRaises(ValueError):
            tracking_faults.apply(obs,1,missing_share=1.0)


class DeadEdgeTests(unittest.TestCase):
    def test_edges_grow_only_into_live_frames_of_the_same_sequence(self):
        labels=np.array(list('HHHHDDAAAAXHHDD')); seq=np.r_[np.zeros(11,np.int32),np.ones(4,np.int32)]
        out=shift_dead_edges(labels,seq,2,1)
        self.assertEqual(''.join(out),'HHDDDDDAAAXHHDD'.replace('XHHDD','XDDDD'))
        np.testing.assert_array_equal(shift_dead_edges(labels,seq,0,0),labels)
        with self.assertRaises(ValueError):
            shift_dead_edges(labels,seq,-1,0)


if __name__=='__main__':
    unittest.main()
