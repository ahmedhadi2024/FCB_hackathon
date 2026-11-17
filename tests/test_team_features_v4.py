"""Fictional tracking only: no competition fixtures or ground truth."""
from dataclasses import replace
import unittest
import numpy as np
from tests.test_temporal import observations
from fcb_external import team_features_v3, team_features_v4 as v4

N=v4.NAMES.index


def with_ball(obs,ball):
    ball=np.asarray(ball,np.float32)
    return replace(obs,ball_xy_m=ball,ball_observed=np.isfinite(ball).all(-1))


def moving(n=32):
    return np.column_stack([np.arange(n)*.2,np.zeros(n)]).astype(np.float32)


class GapAwareFeatureTests(unittest.TestCase):
    def setUp(self):
        self.obs=with_ball(observations(),moving())

    def test_shape_names_and_fully_observed_prefix(self):
        x=v4.extract(self.obs)
        self.assertEqual(x.shape,(32,len(v4.NAMES)))
        self.assertEqual(len(set(v4.NAMES)),len(v4.NAMES))
        self.assertTrue(np.isfinite(x).all())
        np.testing.assert_array_equal(x[:,:v4.BASE],team_features_v3.extract(self.obs))
        np.testing.assert_array_equal(x[:,N('ball_raw_observed')],1)
        for name in ('ball_filled','gap_since','gap_until','gap_length'):
            np.testing.assert_array_equal(x[:,N(name)],0)

    def test_short_gap_is_interpolated_and_flagged(self):
        ball=moving(); ball[10:14]=np.nan
        obs=with_ball(self.obs,ball)
        filled,flag=v4.fill_ball(obs)
        np.testing.assert_allclose(filled,moving(),atol=1e-5)
        np.testing.assert_array_equal(np.flatnonzero(flag),[10,11,12,13])
        x=v4.extract(obs)
        np.testing.assert_array_equal(x[10:14,N('ball_raw_observed')],0)
        np.testing.assert_array_equal(x[10:14,N('ball_filled')],1)
        np.testing.assert_array_equal(x[10:14,N('gap_since')],0)
        # Distances survive the gap: identical to the fully observed track.
        full=v4.extract(self.obs)
        np.testing.assert_allclose(x[:,N('home_distance_1')],full[:,N('home_distance_1')],atol=1e-5)

    def test_long_gap_is_not_filled_and_holds_bordering_distances(self):
        ball=moving(200); ball[60:140]=np.nan
        n=200; base=observations()
        obs=replace(base,period=np.ones(n,np.int8),frame_id=np.arange(1,n+1),time_s=np.arange(n)/25,
                    player_xy_m=np.tile(base.player_xy_m[:1],(n,1,1)),player_observed=np.ones((n,4),bool),
                    sequence_id=np.zeros(n,np.int32))
        obs=with_ball(obs,ball)
        _,flag=v4.fill_ball(obs)
        self.assertFalse(flag.any())
        x=v4.extract(obs)
        np.testing.assert_array_equal(x[60:140,N('ball_observed')],0)
        self.assertAlmostEqual(x[60,N('gap_since')],1/250,places=6)
        self.assertAlmostEqual(x[139,N('gap_until')],1/250,places=6)
        np.testing.assert_allclose(x[60:140,N('gap_length')],80/250,atol=1e-6)
        # Held values equal the nearest distances on the last/next observed frames.
        np.testing.assert_allclose(x[60:140,N('home_prev_distance_1')],x[59,N('home_distance_1')],atol=1e-6)
        np.testing.assert_allclose(x[60:140,N('away_next_distance_1')],x[140,N('away_distance_1')],atol=1e-6)
        np.testing.assert_array_equal(x[60:140,N('home_prev_valid')],1)

    def test_no_fill_or_hold_across_sequences(self):
        ball=moving(); ball[14:18]=np.nan
        obs=replace(with_ball(self.obs,ball),sequence_id=np.repeat([0,1],16).astype(np.int32))
        _,flag=v4.fill_ball(obs)
        self.assertFalse(flag.any())
        x=v4.extract(obs)
        np.testing.assert_array_equal(x[14:16,N('home_next_valid')],0)
        np.testing.assert_array_equal(x[16:18,N('home_prev_valid')],0)
        np.testing.assert_array_equal(x[14:16,N('gap_until')],1)

    def test_no_ball_at_all_is_finite(self):
        x=v4.extract(with_ball(self.obs,np.full((32,2),np.nan)))
        self.assertTrue(np.isfinite(x).all())
        np.testing.assert_array_equal(x[:,N('gap_since')],1)
        np.testing.assert_array_equal(x[:,N('kinematics_valid')],0)

    def test_kinematics_detect_a_reversal(self):
        x=v4.extract(self.obs)
        np.testing.assert_allclose(x[5:27,N('turn_cos')],1,atol=1e-5)
        np.testing.assert_allclose(x[5:27,N('ball_accel')],0,atol=1e-5)
        np.testing.assert_array_equal(x[:5,N('kinematics_valid')],0)
        ball=moving(); ball[16:,0]=ball[16,0]-(np.arange(16)*.2)
        x=v4.extract(with_ball(self.obs,ball))
        self.assertAlmostEqual(float(x[16,N('turn_cos')]),-1,places=5)
        self.assertAlmostEqual(float(x[16,N('ball_accel')]),10/20,places=5)

    def test_exchange_permutation_and_reflection(self):
        ball=moving(); ball[10:14]=np.nan
        obs=with_ball(self.obs,ball); x=v4.extract(obs)
        np.testing.assert_allclose(v4.swap(x),v4.extract(replace(obs,player_team=1-obs.player_team)))
        np.testing.assert_array_equal(v4.swap(v4.swap(x)),x)
        order=[3,0,2,1]
        perm=replace(obs,player_xy_m=obs.player_xy_m[:,order],player_observed=obs.player_observed[:,order],
                     player_team=obs.player_team[order])
        np.testing.assert_allclose(x,v4.extract(perm))
        reflected=v4.extract(replace(obs,player_xy_m=-obs.player_xy_m,ball_xy_m=-obs.ball_xy_m))
        np.testing.assert_allclose(x[:,v4.BASE:],reflected[:,v4.BASE:],atol=1e-6)
        with self.assertRaises(ValueError):
            v4.swap(x[:,:-1])


if __name__=='__main__':
    unittest.main()
