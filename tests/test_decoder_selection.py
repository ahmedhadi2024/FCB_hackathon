import importlib
import unittest


class DecoderSelectionTests(unittest.TestCase):
    def select(self,records):
        try: fn=importlib.import_module('fcb_external.decoder_experiment').select_candidate
        except ModuleNotFoundError: self.fail('Decoder promotion selection is missing')
        return fn(records,{'primary_scenarios':['clean'],'baseline':{'owner':10,'dead_live':10},
                          'promotion':{'minimum_mean_spellF1_gain':.01,'maximum_condition_drop':.02,
                                       'minimum_short_recall_gain':.02}})

    def row(self,key,clean,stress,short):
        owner,dead=map(int,key.split('/'))
        return {'key':key,'owner':owner,'dead_live':dead,'conditions':[
            {'scenario':'clean','spell_f1_mean':clean,'short_recall':short},
            {'scenario':'stress','spell_f1_mean':stress,'short_recall':short}]}

    def test_gate_rejects_stress_regression_and_unchanged_short_recall(self):
        rows=[self.row('10/10',.5,.3,.1),self.row('2/5',.6,.25,.3),
              self.row('5/5',.53,.32,.11),self.row('5/10',.52,.31,.13)]
        result=self.select(rows)
        self.assertEqual(result['selected'],'5/10')
        self.assertTrue(result['promoted'])

    def test_no_eligible_candidate_preserves_baseline(self):
        result=self.select([self.row('10/10',.5,.3,.1),self.row('2/5',.505,.32,.3)])
        self.assertEqual(result['selected'],'10/10')
        self.assertFalse(result['promoted'])

    def test_incomplete_or_nonfinite_comparisons_fail(self):
        rows=[self.row('10/10',.5,.3,.1),self.row('5/10',.52,.31,.13)]
        rows[1]['conditions'].pop()
        with self.assertRaises(ValueError): self.select(rows)
        rows=[self.row('10/10',.5,.3,.1),self.row('5/10',float('nan'),.31,.13)]
        with self.assertRaises(ValueError): self.select(rows)
