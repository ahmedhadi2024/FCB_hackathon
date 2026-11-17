"""Fictional tracking only: no competition fixtures or ground truth."""
from dataclasses import replace
import json
from pathlib import Path
import unittest
import numpy as np
from fcb_external import team_features_v4, team_features_v6 as v6, retrain_v4
from fcb_external.schema import Observations

N=v6.NAMES.index
PROJECT=Path(__file__).resolve().parents[1]


def scene(n=200,speeds=(0.5,0.5,6.0,6.0)):
    """Four players moving along x at fixed speeds (m/s); two per team; static ball."""
    t=np.arange(n)/25
    players=np.zeros((n,4,2),np.float32)
    for i,v in enumerate(speeds):
        players[:,i,0]=(v*t)%40; players[:,i,1]=i*5
    ball=np.zeros((n,2),np.float32)
    return Observations('SYNTHETIC',np.ones(n,np.int8),np.arange(1,n+1),t,players,np.ones((n,4),bool),
        np.array([0,0,1,1],np.int8),ball,np.ones(n,bool),np.zeros(n,np.int32),105.,68.,25.)


class PaceFeatureTests(unittest.TestCase):
    def test_prefix_shape_and_team_pace(self):
        obs=scene(); x=v6.extract(obs)
        self.assertEqual(x.shape,(200,len(v6.NAMES)))
        self.assertEqual(len(set(v6.NAMES)),len(v6.NAMES))
        self.assertTrue(np.isfinite(x).all())
        np.testing.assert_array_equal(x[:,:v6.BASE],team_features_v4.extract(obs))
        mid=slice(40,150)
        np.testing.assert_allclose(x[mid,N('home_pace_median')],0.5/5,atol=1e-3)
        np.testing.assert_allclose(x[mid,N('away_pace_median')],6.0/5,atol=1e-2)
        np.testing.assert_allclose(x[mid,N('pace_walking_fraction')],0.5,atol=1e-6)
        np.testing.assert_array_equal(x[:v6.HALF,N('pace_valid')],0)
        np.testing.assert_array_equal(x[mid,N('pace_valid')],1)

    def test_track_jump_and_missing_slots_are_ignored(self):
        obs=scene(speeds=(1,1,1,1)); p=obs.player_xy_m.copy()
        p[100:,0,1]+=30                      # an identity swap: one slot teleports
        p[:,3]=np.nan                        # one slot never observed
        x=v6.extract(replace(obs,player_xy_m=p,player_observed=np.isfinite(p).all(-1)))
        self.assertTrue(np.isfinite(x).all())
        np.testing.assert_allclose(x[90:112,N('pace_median')],1/5,atol=1e-3)

    def test_no_motion_across_sequences_and_short_input(self):
        obs=replace(scene(60),sequence_id=np.repeat([0,1],30).astype(np.int32))
        x=v6.extract(obs)
        np.testing.assert_array_equal(x[30-v6.HALF:30+v6.HALF,N('pace_valid')],0)
        self.assertTrue(np.isfinite(v6.extract(scene(10))).all())

    def test_exchange_and_permutation(self):
        obs=scene(); x=v6.extract(obs)
        np.testing.assert_allclose(v6.swap(x),v6.extract(replace(obs,player_team=1-obs.player_team)))
        np.testing.assert_array_equal(v6.swap(v6.swap(x)),x)
        order=[3,0,2,1]
        perm=replace(obs,player_xy_m=obs.player_xy_m[:,order],player_observed=obs.player_observed[:,order],
                     player_team=obs.player_team[order])
        np.testing.assert_allclose(x,v6.extract(perm),atol=1e-5)  # float32 summation order in an inherited feature
        with self.assertRaises(ValueError):
            v6.swap(x[:,:-1])

    def test_pace_mask_prefers_slow_moments_and_is_seeded(self):
        n=4000; obs=scene(n,speeds=(4,4,4,4)); p=obs.player_xy_m.copy()
        p[2000:]=p[2000]                     # everyone stands still in the second half of the clip
        obs=replace(obs,player_xy_m=p)
        a=v6.pace_mask(obs,0.4,9,25,100,seed=1); b=v6.pace_mask(obs,0.4,9,25,100,seed=1)
        np.testing.assert_array_equal(a.ball_observed,b.ball_observed)
        missing=~a.ball_observed
        self.assertAlmostEqual(missing.mean(),0.4,places=2)
        self.assertGreater(missing[2100:].mean(),2*missing[:1900].mean())
        np.testing.assert_array_equal(a.player_xy_m,obs.player_xy_m)
        with self.assertRaises(ValueError):
            v6.pace_mask(obs,0,9,25,100,seed=1)

    def test_v6_config_and_runner_wiring(self):
        cfg=json.loads((PROJECT/'configs/retrain_v6.json').read_text())
        retrain_v4.validate_config(cfg)
        self.assertIs(retrain_v4.module_for(cfg),v6)
        self.assertIs(retrain_v4.module_for({}),team_features_v4)
        obs=scene(600)
        slow=next(s for s in cfg['scenarios'] if s['name']=='slow40')
        work=retrain_v4.changed(obs,slow,'train','A')
        self.assertAlmostEqual((~work.ball_observed).mean(),0.4,places=2)
        y=np.repeat([0,1,2],200)
        arrays=[(v6.extract(obs),y,np.asarray(obs.sequence_id))]
        training=dict(cfg['training'],epochs=1,steps_per_epoch=2,batch_size=2,core_frames=32,
                      scaling_sample_frames_per_condition=100,cpu_threads=1)
        checkpoint=retrain_v4.fit(arrays,1,{'channels':8,'dilations':[1,2],'kernel_size':3},training,v6)
        self.assertEqual(checkpoint['feature_module'],'v6')
        self.assertEqual(checkpoint['features'],list(v6.NAMES))
        p=retrain_v4.infer(checkpoint,obs,v6)
        self.assertEqual(p.shape,(600,3))


if __name__=='__main__':
    unittest.main()
