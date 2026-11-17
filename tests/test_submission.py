"""Fictional tracking only: no competition fixtures or ground truth."""
import importlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from fcb_external import atd
from fcb_external.metrics import load_official


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def module(self):
        try:
            return importlib.import_module('fcb_external.submission')
        except ModuleNotFoundError:
            self.fail('Submission boundary is missing')

    def fixture(self):
        folder = self.root/'tactical'
        folder.mkdir()
        meta = folder/'synthetic_atd.xml'
        meta.write_text('<root><Team id="h"/><Team id="a"/>'
                        '<Player teamId="h"/><Player teamId="a"/></root>')
        raw = folder/'synthetic_atd.txt'
        raw.write_text(''.join(f'{i}:0.4,0.5,0;0.8,0.5,0:0.41,0.5,0\n'
                               for i in range(1,9)))
        return raw, meta

    def test_export_maps_one_based_ids_leaves_gaps_and_masks_outside_halves(self):
        path = self.root/'out.csv'
        result = self.module().export_labels(np.array([1,2,4,5,7,8]),
            np.array(list('HHADAA')), path, 8, [(1,1,3),(2,5,6)])
        self.assertEqual(path.read_text(),
            'frame,label\n0,X\n1,H\n2,X\n3,A\n4,X\n5,X\n6,A\n7,X\n')
        self.assertEqual(result['n_frames'],8)
        official = load_official('evaluation/possession_metrics.py')
        self.assertIsNone(official.validate_submission(official.read_label_csv(path),8))

    def test_rejects_bad_ids_labels_and_blocks_without_writing(self):
        cases = [([1,1],list('HA'),[(1,0,1)]),
                 ([0],['H'],[(1,0,1)]), ([9],['H'],[(1,0,1)]),
                 ([1.5],['H'],[(1,0,1)]), ([1],['HOME'],[(1,0,1)]),
                 ([1],['H'],[(1,0,4),(2,4,6)]),
                 ([1],['H'],[(1,0,8)])]
        for ids,labels,blocks in cases:
            with self.subTest(ids=ids,labels=labels,blocks=blocks):
                path = self.root/'bad.csv'
                with self.assertRaises(ValueError):
                    self.module().export_labels(np.array(ids),np.array(labels),path,8,blocks)
                self.assertFalse(path.exists())

    def test_export_will_not_overwrite(self):
        path = self.root/'existing.csv'
        path.write_text('keep me')
        with self.assertRaises(FileExistsError):
            self.module().export_labels(np.array([1]),np.array(['H']),path,8,[(1,0,7)])
        self.assertEqual(path.read_text(),'keep me')

    def test_atd_sorts_rows_but_rejects_duplicates(self):
        raw,meta = self.fixture()
        lines = raw.read_text().splitlines(True)
        raw.write_text(''.join(reversed(lines)))
        obs = atd.convert(raw,meta,'SYNTHETIC','h','a',105,68)
        np.testing.assert_array_equal(obs.frame_id,np.arange(1,9))
        raw.write_text(''.join(lines+lines[:1]))
        with self.assertRaisesRegex(ValueError,'duplicate'):
            atd.convert(raw,meta,'SYNTHETIC','h','a',105,68)

    def test_atd_ignores_empty_trailing_slot_delimiter(self):
        raw,meta = self.fixture()
        raw.write_text('1:0.4,0.5,0;0.8,0.5,0;:0.41,0.5,0\n')
        obs = atd.convert(raw,meta,'SYNTHETIC','h','a',105,68)
        self.assertEqual(obs.player_xy_m.shape,(1,2,2))

    def test_inference_is_target_free_and_deterministic(self):
        self.fixture()
        project = Path(__file__).resolve().parents[1]
        cfg = {'match_id':'SYNTHETIC','source':'tactical','stem':'synthetic',
               'n_frames':8,'pitch_m':[105,68], 'active_blocks':[[1,1,3],[2,5,6]],
               'model':str(project/'models/live_dead_v1.json'),
               'model_sha256':'42cd1665b01d1e1d4028e83f3ad283b666099901896d7f90f473680a0ac66589',
               'metadata_team_policy':'epts_first_team_is_home'}
        config = self.root/'config.json'
        config.write_text(json.dumps(cfg))
        a,b = self.root/'a.csv',self.root/'b.manifest.json'
        first = self.module().run(config,self.root,a)
        second = self.module().run(config,self.root,b)
        self.assertEqual(a.read_bytes(),b.read_bytes())
        self.assertEqual(first['csv_sha256'],second['csv_sha256'])
        self.assertEqual(first['validation'],'official_format_pass')
        self.assertTrue(b.with_name(b.name+'.manifest.json').exists())
        self.assertFalse((self.root/'targets').exists())
        data = a.read_text().splitlines()
        self.assertEqual([data[i+1].split(',')[1] for i in [0,4,7]],['X']*3)

    def test_wrong_model_hash_and_broadcast_rejected(self):
        cfg = {'source':'broadcast'}
        path = self.root/'config.json'
        path.write_text(json.dumps(cfg))
        with self.assertRaisesRegex(ValueError,'tactical'):
            self.module().run(path,self.root,self.root/'out.csv')

    def test_inference_rejects_wrong_model_hash_before_reading_inputs(self):
        project=Path(__file__).resolve().parents[1]
        cfg={'source':'tactical','metadata_team_policy':'epts_first_team_is_home',
             'n_frames':8,'active_blocks':[[1,0,7]],
             'model':str(project/'models/live_dead_v1.json'),'model_sha256':'wrong'}
        path=self.root/'config.json'; path.write_text(json.dumps(cfg))
        with self.assertRaisesRegex(ValueError,'checksum'):
            self.module().run(path,self.root,self.root/'out.csv')

    def test_neural_checkpoint_inference_exports_valid_csv_without_targets(self):
        import torch
        from fcb_external import temporal,team_features
        from fcb_external.schema import digest_file
        self.fixture()
        temporal.configure(1,1)
        net=temporal.TemporalNet(len(team_features.NAMES),8,[1,2])
        checkpoint={'kind':'pytorch_temporal_conv_v1','network':{'channels':8,'dilations':[1,2]},
                    'state_dict':net.state_dict(),'features':list(team_features.NAMES),
                    'mean':torch.zeros(len(team_features.NAMES)),
                    'scale':torch.ones(len(team_features.NAMES)), 'selected_transition_penalty':2}
        model=self.root/'weights.pt'; torch.save(checkpoint,model)
        cfg={'source':'tactical','stem':'synthetic','match_id':'SYNTHETIC',
             'metadata_team_policy':'epts_first_team_is_home','n_frames':8,
             'active_blocks':[[1,1,3],[2,5,6]],'pitch_m':[105,68],
             'model':str(model),'model_sha256':digest_file(model),'model_kind':'pytorch_temporal_conv_v1'}
        path=self.root/'config.json'; path.write_text(json.dumps(cfg))
        try:
            result=self.module().run(path,self.root,self.root/'neural.csv')
        except (ValueError,UnicodeError):
            self.fail('Neural checkpoint inference is missing')
        self.assertEqual(result['validation'],'official_format_pass')
        self.assertFalse(result['ground_truth_accessed'])

    def test_ensemble_bundle_averages_members_and_matches_single_model_when_identical(self):
        import torch
        from fcb_external import temporal,team_features_v6 as features
        from fcb_external.schema import digest_file
        self.fixture()
        temporal.configure(1,1)
        def member(seed):
            torch.manual_seed(seed)
            net=temporal.TemporalNet(len(features.NAMES),8,[1,2])
            return {'network':{'channels':8,'dilations':[1,2]},'state_dict':net.state_dict(),
                    'features':list(features.NAMES),'feature_module':'v6',
                    'mean':torch.zeros(len(features.NAMES)),'scale':torch.ones(len(features.NAMES))}
        shared={'feature_module':'v6','features':list(features.NAMES),
                'selected_transition_costs':[5,30],'min_controlled_frames':0}
        def run(name,checkpoint,kind):
            model=self.root/(name+'.pt'); torch.save(checkpoint,model)
            cfg={'source':'tactical','stem':'synthetic','match_id':'SYNTHETIC',
                 'metadata_team_policy':'epts_first_team_is_home','n_frames':8,
                 'active_blocks':[[1,1,3],[2,5,6]],'pitch_m':[105,68],
                 'model':str(model),'model_sha256':digest_file(model),'model_kind':kind}
            path=self.root/(name+'.json'); path.write_text(json.dumps(cfg))
            out=self.root/(name+'.csv')
            return self.module().run(path,self.root,out),out.read_bytes()
        first=member(1)
        single,single_csv=run('single',{'kind':'pytorch_temporal_conv_v4',**first,**shared},'pytorch_temporal_conv_v4')
        twin,twin_csv=run('twin',{'kind':'pytorch_temporal_ensemble_v1','members':[first,first],**shared},
                          'pytorch_temporal_ensemble_v1')
        self.assertEqual(twin_csv,single_csv)
        self.assertEqual(twin['ensemble_members'],2)
        self.assertEqual(twin['validation'],'official_format_pass')
        mixed={'kind':'pytorch_temporal_ensemble_v1','members':[first,member(2)],**shared}
        _,a=run('mixed_a',mixed,'pytorch_temporal_ensemble_v1')
        _,b=run('mixed_b',mixed,'pytorch_temporal_ensemble_v1')
        self.assertEqual(a,b)
        with self.assertRaisesRegex(ValueError,'two members'):
            run('lonely',{'kind':'pytorch_temporal_ensemble_v1','members':[first],**shared},
                'pytorch_temporal_ensemble_v1')

    def test_inference_rejects_incomplete_tactical_timeline(self):
        raw,meta=self.fixture()
        raw.write_text(raw.read_text().splitlines()[0]+'\n')
        project=Path(__file__).resolve().parents[1]
        cfg={'source':'tactical','stem':'synthetic','match_id':'SYNTHETIC',
             'metadata_team_policy':'epts_first_team_is_home', 'n_frames':8,
             'active_blocks':[[1,0,7]],'pitch_m':[105,68],
             'model':str(project/'models/live_dead_v1.json'),
             'model_sha256':'42cd1665b01d1e1d4028e83f3ad283b666099901896d7f90f473680a0ac66589'}
        path=self.root/'config.json'; path.write_text(json.dumps(cfg))
        with self.assertRaisesRegex(ValueError,'complete tactical'):
            self.module().run(path,self.root,self.root/'out.csv')
        self.assertFalse((self.root/'out.csv').exists())

    def test_periods_cover_outside_play_without_cross_half_context(self):
        raw,meta = self.fixture()
        blocks = self.module().timeline_blocks(8,[(1,1,3),(2,5,6)])
        obs = atd.convert(raw,meta,'SYNTHETIC','h','a',105,68,blocks)
        np.testing.assert_array_equal(obs.period,[0,1,1,1,0,2,2,0])
        np.testing.assert_array_equal(obs.sequence_id,[0,1,1,1,2,3,3,4])


if __name__ == '__main__':
    unittest.main()
