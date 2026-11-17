import unittest
import importlib
import numpy as np


class ErrorAnalysisTests(unittest.TestCase):
    def analyze(self,g,p,seq=None,seen=None):
        try:
            fn=importlib.import_module('fcb_external.error_analysis').analyze
        except ModuleNotFoundError:
            self.fail('External episode analysis is missing')
        n=len(g)
        return fn(np.array(list(g)),np.array(list(p)),
                  np.zeros(n,int) if seq is None else np.array(seq),
                  np.ones(n,bool) if seen is None else np.array(seen))

    def test_duration_bins_and_one_to_one_short_spell_recall(self):
        g='H'*25+'A'*75+'D'*25+'H'*150+'A'*400
        r=self.analyze(g,g)
        self.assertEqual([v['gt'] for v in r['duration_bins'].values()],[1,1,1,1])
        self.assertTrue(all(v['matched_iou_0.5']==1 for v in r['duration_bins'].values()))
        self.assertEqual(r['official']['metrics']['spell_f1_mean'],1)

    def test_merge_and_brief_stoppage_miss(self):
        r=self.analyze('H'*50+'D'*5+'H'*50,'H'*105)
        self.assertEqual(r['merged_predictions'],1)
        self.assertEqual(r['brief_dead_segments']['missed_majority'],1)
        self.assertEqual(r['transitions']['H->D']['missed'],1)

    def test_unknown_and_sequences_split_without_cross_boundary_matching(self):
        r=self.analyze('HHXHH','HHXHH',[0,0,0,1,1])
        self.assertEqual(r['gt_spells'],2)
        self.assertEqual(r['transitions'],{})
        self.assertEqual(r['official']['graded_frames'],4)

    def test_transition_pair_and_missingness_accuracy_are_separate(self):
        r=self.analyze('HHAA','HHHA',seen=[1,1,0,0])
        self.assertEqual(r['transitions']['H->A']['matched'],1)
        self.assertEqual(r['transitions']['H->A']['mean_abs_latency_frames'],1)
        self.assertEqual(r['visibility']['missing']['frame_accuracy'],.5)

    def test_spurious_and_missed_spells_and_reproducibility(self):
        r=self.analyze('HHDD','AAAA')
        self.assertEqual(r['missed_spells_iou_0.5'],1)
        self.assertEqual(r['spurious_spells_iou_0.5'],1)
        self.assertEqual(r,self.analyze('HHDD','AAAA'))
