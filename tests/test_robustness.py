import importlib
import unittest
from dataclasses import replace
import numpy as np
from tests.test_temporal import observations


class RobustnessTests(unittest.TestCase):
    def corrupt(self,obs,cfg):
        try: fn=importlib.import_module('fcb_external.robustness').corrupt
        except ModuleNotFoundError: self.fail('Harder robustness corruption is missing')
        return fn(obs,cfg)

    def test_seeded_mask_fraction_immutability_and_sequence_keys(self):
        obs=observations(); original=obs.ball_xy_m.copy()
        cfg={'name':'fixture','seed':2,'missing_fraction':.4,'gap_min_frames':1,'gap_max_frames':5}
        a=self.corrupt(obs,cfg); b=self.corrupt(obs,cfg)
        np.testing.assert_array_equal(a.ball_observed,b.ball_observed)
        self.assertEqual(int((~a.ball_observed).sum()),13)
        np.testing.assert_array_equal(obs.ball_xy_m,original)
        np.testing.assert_array_equal(a.sequence_id,obs.sequence_id)
        a.validate()

    def test_sequence_masks_are_independent_and_crowd_branch_is_finite(self):
        obs=replace(observations(),sequence_id=np.repeat([0,1],16))
        cfg={'name':'crowded','seed':3,'missing_fraction':.5,'gap_min_frames':1,'gap_max_frames':4,
             'crowd_weight':9,'crowd_radius_m':10,'crowd_min_players':2}
        a=self.corrupt(obs,cfg)
        self.assertEqual((~a.ball_observed[:16]).sum(),8)
        self.assertEqual((~a.ball_observed[16:]).sum(),8)
        a.validate()

    def test_duplicates_preserve_source_team_and_missingness(self):
        obs=observations(); a=self.corrupt(obs,{'name':'duplicates','seed':1,'duplicate_fraction':.5})
        self.assertEqual(a.player_xy_m.shape,(32,6,2))
        np.testing.assert_array_equal(a.player_xy_m[:,:4],obs.player_xy_m)
        for i in (4,5):
            choices=np.flatnonzero(obs.player_team==a.player_team[i])
            self.assertTrue(any(np.array_equal(a.player_xy_m[:,i],obs.player_xy_m[:,j]) for j in choices))
        self.assertEqual(a.player_team[4:].tolist(),[0,1])

    def test_legacy_scenario_matches_existing_corruption(self):
        from fcb_external.features import corrupt
        cfg={'name':'legacy','seed':4,'ball_gap_frames':3,'ball_gap_period_frames':7}
        a=self.corrupt(observations(),cfg); b=corrupt(observations(),cfg)
        np.testing.assert_array_equal(a.ball_xy_m,b.ball_xy_m)

    def test_invalid_fraction_rejected(self):
        with self.assertRaises(ValueError):
            self.corrupt(observations(),{'seed':1,'missing_fraction':1.5})
