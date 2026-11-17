import importlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
from tests.fixtures import frame, frameset, write_bundle
from fcb_external.idsse import convert
from fcb_external.schema import Observations, Targets


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)

    def module(self, name):
        try:
            return importlib.import_module('fcb_external.'+name)
        except ModuleNotFoundError:
            self.fail(name+' not implemented')

    def observation(self,sets=None):
        meta,pos,_=write_bundle(self.root,sets)
        return convert(pos,meta,'TEST')[0]

    def test_nearest_uses_metres_not_normalized_distances(self):
        # Home .02*105=2.1m; away .025*68=1.7m. Raw normalized picks home incorrectly.
        obs=self.observation([frameset('ball','ball',[frame(1),frame(2)]),
             frameset('home','h',[frame(1,2.1,0),frame(2,2.1,0)]),
             frameset('away','a',[frame(1,0,1.7),frame(2,0,1.7)])])
        np.testing.assert_array_equal(self.module('baseline').nearest_codes(obs),[1,1])

    def test_missing_ball_and_all_players_yield_dead_in_raw(self):
        obs=self.observation([frameset('ball','ball',[frame(1,'NaN'),frame(2)])])
        np.testing.assert_array_equal(self.module('baseline').predict(obs,0),['D','D'])

    def test_h_a_only_vote_and_home_tie(self):
        m=self.module('baseline')
        np.testing.assert_array_equal(m.smooth_codes(np.array([0,-1,-1,1]),1),
                                     ['H','H','A','A'])
        np.testing.assert_array_equal(m.smooth_codes(np.array([0,-1,1]),1),['H','H','A'])

    def test_no_smoothing_across_sequence_break(self):
        obs=self.observation([frameset('ball','ball',[frame(1,0),frame(2,0)]),
            frameset('home','h',[frame(1,0),frame(2,0)]),
            frameset('ball','ball',[frame(1,10),frame(2,10)],'secondHalf'),
            frameset('away','a',[frame(1,10),frame(2,10)],'secondHalf')])
        np.testing.assert_array_equal(self.module('baseline').predict(obs,125),['H','H','A','A'])

    def test_target_file_unnecessary_and_changes_do_not_affect_predictions(self):
        obs=self.observation()
        before=self.module('baseline').predict(obs,25)
        Targets(np.array(['X']*4),np.zeros(4,dtype=bool)).save(self.root/'targets',obs)
        np.testing.assert_array_equal(before,self.module('baseline').predict(obs,25))

    def test_smoothing_matches_official_on_synthetic_codes(self):
        m=self.module('metrics')
        official=m.load_official('starter/baseline.py')
        raw=np.random.default_rng(42).choice([-1,0,1],size=301)
        baseline=self.module('baseline')
        for width in [0,25,50,125]:
            np.testing.assert_array_equal(baseline.smooth_codes(raw,width),official.smooth(raw,width))

    def test_nearest_matches_official_on_synthetic_normalized_tracking(self):
        obs=self.observation([frameset('ball','ball',[frame(1),frame(2,'NaN'),frame(3)]),
             frameset('home','h',[frame(1,2.1,0),frame(2,2.1,0),frame(3,-1,0)]),
             frameset('away','a',[frame(1,0,1.7),frame(2,0,1.7),frame(3,2,0)])])
        official=self.module('metrics').load_official('starter/baseline.py')
        path=self.root/'official_fixture.txt'
        with path.open('w') as f:
            for i in range(3):
                groups=[]
                for x,y in obs.player_xy_m[i]:
                    groups.append(f'{x/105},{y/68},0')
                x,y=obs.ball_xy_m[i]
                ball='NaN,NaN,NaN' if not obs.ball_observed[i] else f'{x/105},{y/68},0'
                f.write(f'{i+1}:'+ ';'.join(groups)+':'+ball+'\n')
        expected=official.nearest_player_labels(path,obs.player_team,3,(105,68))
        np.testing.assert_array_equal(self.module('baseline').nearest_codes(obs),expected)

    def test_perfect_predictions_have_both_headline_scores_one(self):
        m=self.module('metrics')
        labels=np.array(list('HHHDDDAAAHHH'))
        result=m.evaluate_labels(labels,labels,np.zeros(len(labels),dtype=np.int32))
        self.assertEqual(result['metrics']['macro_f1'],1)
        self.assertEqual(result['metrics']['spell_f1_mean'],1)

    def test_halves_not_merged_into_one_spell(self):
        m=self.module('metrics')
        labels=np.array(list('HHHH'))
        result=m.evaluate_labels(labels,labels,np.array([0,0,1,1]))
        self.assertEqual(result['metrics']['n_gt_spells'],2)
        self.assertEqual(result['evaluation_separator_rows'],1)

    def test_unknown_targets_preserve_array_length_and_are_ungraded(self):
        m=self.module('metrics')
        gt=np.array(list('HHXXAA'))
        pred=np.array(list('HHDD AA'.replace(' ','')))
        result=m.evaluate_labels(gt,pred,np.zeros(len(gt),dtype=np.int32))
        self.assertEqual(result['graded_frames'],4)
        self.assertEqual(result['observation_rows'],6)
        self.assertEqual(result['metrics']['spell_f1_mean'],1)

    def test_false_dead_interruption_fragments_possession(self):
        m=self.module('metrics')
        gt=np.array(list('HHHHHHHHHHDDDAAAAA'))
        pred=gt.copy(); pred[4]='D'
        result=m.evaluate_labels(gt,pred,np.zeros(len(gt),dtype=np.int32))
        self.assertGreater(result['metrics']['n_pred_spells'],result['metrics']['n_gt_spells'])
        self.assertLess(result['metrics']['spell_f1_mean'],1)

    def test_observation_target_storage_roundtrip(self):
        obs=self.observation()
        obs.save(self.root/'observations')
        loaded=Observations.load(self.root/'observations')
        for name in obs.array_names():
            np.testing.assert_array_equal(getattr(obs,name),getattr(loaded,name))
        targets=Targets(np.array(['H','A','D','X']),np.array([True,True,True,False]))
        targets.save(self.root/'targets',obs)
        np.testing.assert_array_equal(Targets.load(self.root/'targets',loaded).label,targets.label)

    def test_invalid_observation_mask_rejected(self):
        obs=self.observation()
        obs.ball_observed[:]=False
        with self.assertRaisesRegex(ValueError,'mask'):
            obs.validate()

    def test_target_mismatch_rejected(self):
        obs=self.observation()
        targets=Targets(np.array(['H']),np.array([True]))
        with self.assertRaises(ValueError):
            targets.save(self.root/'targets',obs)


if __name__=='__main__':
    unittest.main()
