import importlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from tests.fixtures import write_bundle,frame,frameset
from fcb_external.idsse import convert
from fcb_external.schema import Observations
from download_development import development_ids


class LiveDeadTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)

    def mod(self,name):
        try:
            return importlib.import_module('fcb_external.'+name)
        except ModuleNotFoundError:
            self.fail(name+' not implemented')

    def obs(self):
        sets=[frameset('ball','ball',[frame(i,(-1)**i,0) for i in range(1,51)]),
              frameset('home','h',[frame(i,2,0) for i in range(1,51)]),
              frameset('away','a',[frame(i,-2,0) for i in range(1,51)])]
        meta,pos,_=write_bundle(self.root,sets)
        return convert(pos,meta,'TEST')[0]

    def test_split_has_no_overlap_and_never_downloads_tests(self):
        self.assertEqual(development_ids({'train':['a'],'validation':['b'],'sealed_test':['c']}),['a','b'])
        with self.assertRaises(ValueError):
            development_ids({'train':['a'],'validation':['a'],'sealed_test':['c']})

    def test_oracles_only_fix_the_named_error(self):
        gt=np.array(list('HHDD AA'.replace(' ','')))
        pred=np.array(list('AAAAHH'))
        state,team=self.mod('diagnostics').oracle_predictions(gt,pred)
        np.testing.assert_array_equal(state,list('AADDHH'))
        np.testing.assert_array_equal(team,list('HHAAAA'))

    def test_status_oracle_removes_false_dead_without_using_true_owner(self):
        gt=np.array(list('AA')); pred=np.array(list('HD'))
        state,team=self.mod('diagnostics').oracle_predictions(gt,pred)
        np.testing.assert_array_equal(state,list('HH'))
        np.testing.assert_array_equal(team,list('AD'))

    def test_atd_and_idsse_equivalent_geometry_and_features(self):
        meta,pos,_=write_bundle(self.root,[
            frameset('home','h',[frame(1,-42,0),frame(2,-31.5,0)]),
            frameset('away','a',[frame(1,42,0),frame(2,31.5,0)]),
            frameset('ball','ball',[frame(1,0,0),frame(2,0,0)])])
        external=convert(pos,meta,'TEST')[0]
        ameta=self.root/'atd.xml'
        ameta.write_text('<root><Team id="away"/><Team id="home"/>'
            '<Player teamId="home"/><Player teamId="home"/><Player teamId="away"/></root>')
        raw=self.root/'atd.txt'
        raw.write_text('1:0.1,0.5,0;NaN,NaN,NaN;0.9,0.5,0:0.5,0.5,0\n'
                       '2:0.2,0.5,0;NaN,NaN,NaN;0.8,0.5,0:0.5,0.5,0\n')
        atd=self.mod('atd').convert(raw,ameta,'TEST','home','away',105,68)
        np.testing.assert_allclose(atd.player_xy_m,external.player_xy_m,atol=1e-5,equal_nan=True)
        np.testing.assert_allclose(self.mod('features').extract(atd),
                                   self.mod('features').extract(external),atol=1e-5)

    def test_atd_partial_coordinates_and_frame_gap(self):
        meta=self.root/'atd.xml'
        meta.write_text('<root><Team id="h"/><Team id="a"/><Player teamId="h"/></root>')
        raw=self.root/'atd.txt'
        raw.write_text('1:0.1,0.5,0:0.2,NaN,0\n3:0.1,0.5,0:0.2,0.5,0\n')
        obs=self.mod('atd').convert(raw,meta,'TEST','h','a',105,68)
        self.assertFalse(obs.ball_observed[0])
        np.testing.assert_array_equal(obs.sequence_id,[0,1])

    def test_features_ignore_slot_permutation(self):
        from dataclasses import replace
        obs=self.obs()
        order=np.arange(len(obs.player_team))[::-1]
        perm=replace(obs,player_xy_m=obs.player_xy_m[:,order],
                     player_observed=obs.player_observed[:,order],player_team=obs.player_team[order])
        np.testing.assert_allclose(self.mod('features').extract(obs),self.mod('features').extract(perm),atol=1e-6)

    def test_corruption_deterministic_does_not_mutate_inputs(self):
        obs=self.obs(); before=obs.ball_xy_m.copy()
        cfg={'seed':1,'ball_gap_frames':5,'ball_gap_period_frames':10,
             'player_drop_probability':0.2,'coordinate_noise_std_m':0.5,'permute_slots_within_team':True}
        a=self.mod('features').corrupt(obs,cfg); b=self.mod('features').corrupt(obs,cfg)
        np.testing.assert_array_equal(obs.ball_xy_m,before)
        np.testing.assert_array_equal(a.ball_xy_m,b.ball_xy_m)
        self.assertLess(a.ball_observed.sum(),obs.ball_observed.sum())
        self.assertTrue(np.isfinite(self.mod('features').extract(a)).all())

    def test_temporal_means_reset_at_sequence_break(self):
        result=self.mod('features').window_mean(np.array([1.,1.,0.,0.]),np.array([0,0,1,1]),12)
        np.testing.assert_array_equal(result,[1,1,0,0])

    def test_logistic_learns_and_serializes_without_targets_at_inference(self):
        m=self.mod('model')
        x=np.array([[-2.],[-1.],[1.],[2.]])
        y=np.array([0.,0.,1.,1.])
        model=m.fit(x,y,l2=0.01,iterations=40)
        p=m.probability(model,x)
        self.assertTrue((p[:2]<0.5).all() and (p[2:]>0.5).all())
        path=self.root/'model.json'; path.write_text(json.dumps(model))
        np.testing.assert_array_equal(p,m.probability(json.loads(path.read_text()),x))

    def test_owner_propagation_stays_within_sequence(self):
        pred=np.array(list('HDDD'))
        out=self.mod('model').team_states(pred,np.array([0,0,1,1]))
        np.testing.assert_array_equal(out,list('HHXX'))

    def test_experiment_refuses_test_match_before_reading_files(self):
        protocol={'train':['train'],'validation':['val'],'sealed_test':['secret']}
        with self.assertRaisesRegex(ValueError,'sealed|allowed'):
            self.mod('experiment').load_match(self.root,'secret','validation',protocol)

    def test_input_fingerprint_includes_pitch_metadata(self):
        folder=self.root/'observations'; self.obs().save(folder)
        experiment=self.mod('experiment')
        self.assertTrue(hasattr(experiment,'input_fingerprint'),'Full input fingerprint is missing')
        before=experiment.input_fingerprint(folder)
        path=folder/'metadata.json'; meta=json.loads(path.read_text()); meta['pitch_length_m']=110
        path.write_text(json.dumps(meta))
        after=experiment.input_fingerprint(folder)
        self.assertNotEqual(before,after)

    def test_atd_cartesian_y_direction_is_explicit(self):
        np.testing.assert_allclose(self.mod('atd').xy('0.1,0.25,0',105,68),[-42,17])


if __name__=='__main__':
    unittest.main()
