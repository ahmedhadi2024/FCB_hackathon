from dataclasses import replace
import importlib
import unittest
import numpy as np
from tests.test_temporal import observations
from fcb_external import team_features

class FeatureRecoveryTests(unittest.TestCase):
    def setUp(self):
        try: self.m=importlib.import_module('fcb_external.team_features_v3')
        except ModuleNotFoundError: self.fail('Versioned feature block missing')
        self.obs=observations()

    def test_legacy_prefix_and_missing_ball_geometry(self):
        o=replace(self.obs,ball_xy_m=np.full_like(self.obs.ball_xy_m,np.nan),ball_observed=np.zeros(32,bool))
        x=self.m.extract(o)
        self.assertEqual(x.shape,(32,52))
        np.testing.assert_array_equal(x[:,:40],team_features.extract(o))
        other=replace(o,player_xy_m=o.player_xy_m[:,[2,3,0,1]])
        self.assertFalse(np.array_equal(x[:,40:],self.m.extract(other)[:,40:]))
        self.assertTrue(np.isfinite(x).all())
        np.testing.assert_allclose(x[0,40:46],[.5/105,0,.5/np.hypot(105,68),0,1,0])

    def test_missing_and_one_player(self):
        p=self.obs.player_xy_m.copy(); p[:,:]=np.nan; p[:,0]=[1,2]
        x=self.m.extract(replace(self.obs,player_xy_m=p,player_observed=np.isfinite(p).all(-1)))
        np.testing.assert_array_equal(x[:,40:],0)

    def test_exchange_permutation_padding_and_reflection(self):
        o=self.obs;x=self.m.extract(o); order=[3,0,2,1]
        perm=replace(o,player_xy_m=o.player_xy_m[:,order],player_observed=o.player_observed[:,order],player_team=o.player_team[order])
        np.testing.assert_allclose(x,self.m.extract(perm))
        np.testing.assert_allclose(self.m.swap(x),self.m.extract(replace(o,player_team=1-o.player_team)))
        np.testing.assert_array_equal(self.m.swap(self.m.swap(x)),x)
        padded=replace(o,player_xy_m=np.concatenate([o.player_xy_m,np.full((32,2,2),np.nan)],1),player_observed=np.column_stack([o.player_observed,np.zeros((32,2),bool)]),player_team=np.r_[o.player_team,[0,1]])
        np.testing.assert_allclose(x,self.m.extract(padded))
        reflected=self.m.extract(replace(o,player_xy_m=-o.player_xy_m,ball_xy_m=-o.ball_xy_m))
        expected=x.copy();expected[:,[40,41,46,47]]*=-1
        np.testing.assert_allclose(expected,reflected)

    def test_motion_reset_and_invalid_endpoints(self):
        o=self.obs;p=o.player_xy_m.copy();p[:,:,0]+=np.arange(32)[:,None]*.04
        seq=np.repeat([0,1],16)
        x=self.m.extract(replace(o,player_xy_m=p,sequence_id=seq))
        np.testing.assert_allclose(x[5:16,43],.1,atol=1e-6)
        np.testing.assert_array_equal(x[16:21,43],0)
        p[0,:2]=np.nan
        x=self.m.extract(replace(o,player_xy_m=p,player_observed=np.isfinite(p).all(-1)))
        self.assertEqual(x[5,45],0)

    def test_duplicate_and_outlier_summary_is_finite(self):
        o=self.obs;p=o.player_xy_m.copy();p[:,0]=[1000,1000]
        x=self.m.extract(replace(o,player_xy_m=p))
        self.assertTrue(np.isfinite(x).all())
        d=replace(o,player_xy_m=np.concatenate([o.player_xy_m,o.player_xy_m[:,[0]]],1),player_observed=np.ones((32,5),bool),player_team=np.r_[o.player_team,0])
        np.testing.assert_allclose(self.m.extract(d)[:,40],0)
