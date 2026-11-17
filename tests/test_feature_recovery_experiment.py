import importlib
import unittest
import numpy as np
from tests.test_temporal import observations

class RecoveryRunnerTests(unittest.TestCase):
    def setUp(self):
        try:self.m=importlib.import_module('fcb_external.feature_recovery_experiment')
        except ModuleNotFoundError:self.fail('Paired runner missing')
    def protocol(self):
        import json
        from pathlib import Path
        return json.loads(Path('configs/feature_recovery_v3.json').read_text())
    def named_scores(self,d):
        cfg=self.protocol();names=[s['name'] for s in cfg['scenarios']]+list(self.m.AUDITS)+['irregular40_seed301','irregular40_seed302','irregular60_seed303','mixed_stress_seed304']
        cells=[]
        for i,match in enumerate(cfg['matches']):
            fold=next(f['name'] for f in cfg['folds'] if match in f['validation'])
            for j,name in enumerate(names):
                for variant,value in [('baseline',0.),('geometry',float(d[i,j]))]:
                    cells.append(dict(variant=variant,fold=fold,match=match,scenario=name,spellF1=value))
        return {'cells':cells}
    def test_gates_reject_incomplete_and_single_winner(self):
        with self.assertRaises(ValueError):self.m.assess_gates({},self.protocol())
        s=np.full((5,18),.02)
        self.assertTrue(self.m.assess_gates(self.named_scores(s),self.protocol())['passed'])
        s[:4]=-.001;s[4]=.2
        self.assertFalse(self.m.assess_gates(self.named_scores(s),self.protocol())['passed'])
    def test_severe_cell_and_fold_loss_fail(self):
        s=np.full((5,18),.03);s[0,-1]=-.06
        self.assertFalse(self.m.assess_gates(self.named_scores(s),self.protocol())['passed'])
        s[:2]=-.001
        self.assertFalse(self.m.assess_gates(self.named_scores(s),self.protocol())['passed'])
    def test_seed_roles_and_pairing(self):
        a=self.m.seed_for('train','A','J03WMX','irregular40')
        self.assertEqual(a,self.m.seed_for('train','A','J03WMX','irregular40'))
        self.assertNotEqual(a,self.m.seed_for('validation','A','J03WMX','irregular40'))
    def test_fold_overlap_rejected(self):
        with self.assertRaises(ValueError):self.m.validate_fold({'train':['x'],'validation':['x']})
    def test_scaling_only_training_and_input_contract(self):
        x=np.arange(120,dtype=np.float32).reshape(3,40)
        mean,scale=self.m.scaling([x],self.m.team_features)
        self.assertEqual(mean.shape,(40,));self.assertTrue(np.all(scale>0))
        o=observations()
        c=self.m.fit([(o,np.tile(np.array([0,1,2,0]),8))],self.m.team_features,1,{'channels':4,'dilations':[1],'kernel_size':3},dict(epochs=1,steps_per_epoch=1,batch_size=2,core_frames=8,learning_rate=.001,weight_decay=.001,gradient_clip=1,team_exchange_probability=.5))
        p=self.m.infer(c,o,self.m.team_features,9)
        np.testing.assert_allclose(p.sum(1),1,atol=1e-6)
        with self.assertRaises(ValueError):self.m.infer(c,o,self.m.team_features_v3)


    def test_resume_rejects_changed_validation_inputs(self):
        report={'protocol_hash':'p','sources':{},'training_inputs':{},'validation_inputs':{'m':'old'},'model_hash':'h'}
        with self.assertRaises(ValueError):self.m.verify_resume(report,'p',{}, {},{'m':'new'},'h')

    def test_runner_passes_training_matches_only_and_rejects_incomplete_outputs(self):
        import json,tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        from dataclasses import replace
        cfg=json.loads(Path('configs/feature_recovery_v3.json').read_text())
        expected=[set(f['train']) for f in cfg['folds'] for _ in range(2)];seen=[]
        def fitting(cache,*args):
            seen.append({o.match_id for o,y in cache})
            self.assertEqual(seen[-1],expected[len(seen)-1])
            return {'training_loss':[1.]}
        def loading(root,match,role,protocol):
            self.assertIn(match,protocol[role])
            return replace(observations(),match_id=match),SimpleNamespace(label=np.tile(['H','A','D','H'],8))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'configs').mkdir();(root/'configs/feature_recovery_v3.json').write_text(json.dumps(cfg))
            with patch.object(self.m,'fit',side_effect=fitting),patch.object(self.m,'load_match',side_effect=loading),patch.object(self.m,'input_fingerprint',side_effect=lambda p:str(p)),patch.object(self.m,'infer',return_value=np.full((32,3),1/3)),patch.object(self.m,'evaluate_labels',return_value={'metrics':{'spell_f1_mean':.3}}),patch.object(self.m,'source_hashes',return_value={}):
                self.m.run(root)
                self.assertEqual(len(seen),6)
                (root/'reports/feature_recovery_v3/A_baseline.json').unlink()
                (root/'models/feature_recovery_v3/A_baseline.pt').write_bytes(b'incomplete')
                with self.assertRaises(ValueError):self.m.run(root)

    def test_validation_interruption_resumes_saved_fit_without_retraining(self):
        import json,tempfile,torch
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        from dataclasses import replace
        cfg=json.loads(Path('configs/feature_recovery_v3.json').read_text())
        def loading(root,match,role,protocol):
            return replace(observations(),match_id=match),SimpleNamespace(label=np.tile(['H','A','D','H'],8))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'configs').mkdir();(root/'configs/feature_recovery_v3.json').write_text(json.dumps(cfg))
            with patch.object(self.m,'fit',return_value={'training_loss':[1.]*8}) as fitting,patch.object(self.m,'load_match',side_effect=loading),patch.object(self.m,'input_fingerprint',side_effect=lambda p:str(p)),patch.object(self.m,'infer',side_effect=RuntimeError('evaluation interrupted')),patch.object(self.m,'source_hashes',return_value={}):
                with self.assertRaisesRegex(RuntimeError,'evaluation interrupted'):self.m.run(root)
                model=root/'models/feature_recovery_v3/A_baseline.pt'
                self.assertTrue(model.exists(),'Completed training must survive evaluation interruption')
                saved=torch.load(model,weights_only=True)
                self.assertEqual(saved['run_identity']['terminal_epoch'],8)
                with patch.object(self.m,'infer',return_value=np.full((32,3),1/3)),patch.object(self.m,'evaluate_labels',return_value={'metrics':{'spell_f1_mean':.3}}):self.m.run(root)
                self.assertEqual(fitting.call_count,6,'Only remaining five fits may run after resuming')

    def test_atomic_checkpoint_does_not_publish_failed_save(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.pt'
            with patch.object(self.m.torch,'save',side_effect=RuntimeError('disk error')):
                with self.assertRaisesRegex(RuntimeError,'disk error'):self.m.atomic_checkpoint(path,{})
            self.assertFalse(path.exists())

    def test_validation_cases_are_created_lazily(self):
        from unittest.mock import Mock,patch
        o=observations();change=Mock(return_value=o)
        with patch.object(self.m,'audit',return_value=o) as audit:
            cases=self.m.validation_cases(o,[{'name':'clean'}],[],change)
            change.assert_not_called();audit.assert_not_called()
            self.assertEqual(next(cases)[0],'clean')
            change.assert_called_once();audit.assert_not_called()
            self.assertEqual(next(cases)[0],self.m.AUDITS[0]);audit.assert_called_once()

    def test_trained_resume_rejects_mismatched_source_and_partial_epoch(self):
        import tempfile,torch
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'trained.pt';identity={'sources':{'runner':'old'},'terminal_epoch':8}
            torch.save({'run_identity':identity,'training_loss':[1.]*8},path)
            with self.assertRaises(ValueError):self.m.load_trained(path,{**identity,'sources':{'runner':'new'}})
            torch.save({'run_identity':identity,'training_loss':[1.]},path)
            with self.assertRaises(ValueError):self.m.load_trained(path,identity)

    def test_named_scores_are_order_independent_and_require_exact_cells(self):
        import copy
        scores=self.named_scores(np.full((5,18),.02));cfg=self.protocol()
        result=self.m.assess_gates(scores,cfg)
        scores['cells'].reverse()
        reordered=copy.deepcopy(cfg);reordered['matches'].reverse();reordered['folds'].reverse()
        self.assertEqual(result,self.m.assess_gates(scores,reordered))
        for mutation in ('duplicate','missing','wrong_fold','unknown','nonfinite'):
            bad=copy.deepcopy(scores)
            if mutation=='duplicate':bad['cells'].append(bad['cells'][0])
            elif mutation=='missing':bad['cells'].pop()
            elif mutation=='wrong_fold':bad['cells'][0]['fold']='wrong'
            elif mutation=='unknown':bad['cells'][0]['scenario']='unknown'
            else:bad['cells'][0]['spellF1']=float('nan')
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.m.assess_gates(bad,cfg)

    def test_unrounded_gate_boundaries(self):
        cfg=self.protocol()
        def gates(d):return self.m.assess_gates(self.named_scores(d),cfg)['gates']
        d=np.full((5,18),.01)
        self.assertTrue(gates(d)['mean_gain'])
        d[:,:4]=np.nextafter(.01,0)
        self.assertFalse(gates(d)['mean_gain'])
        d=np.full((5,18),.03);d[:,9]=-.02
        self.assertTrue(gates(d)['family_floor'])
        d[:,9]=np.nextafter(-.02,-1)
        self.assertFalse(gates(d)['family_floor'])
        d=np.full((5,18),.03);d[0,9]=-.05
        self.assertTrue(gates(d)['cell_floor'])
        d[0,9]=np.nextafter(-.05,-1)
        self.assertFalse(gates(d)['cell_floor'])
        d=np.full((5,18),.03);d[0,:4]=-.02
        self.assertTrue(gates(d)['match_floor'])
        d[0,:4]=np.nextafter(-.02,-1)
        self.assertFalse(gates(d)['match_floor'])
        d=np.full((5,18),.03);d[0,:4]=0
        self.assertTrue(gates(d)['four_matches'])
        d[1,:4]=0
        self.assertFalse(gates(d)['four_matches']);self.assertFalse(gates(d)['three_folds'])
        d=np.zeros((5,18));d[4]=.2
        self.assertFalse(gates(d)['without_best'])
        d=np.full((5,18),.03);d[:2,[4,5,13,14,15,16]]=0
        self.assertFalse(gates(d)['missing_four'])

    def test_protocol_rejects_overlap_and_unpartitioned_validation(self):
        cfg=self.protocol();cfg['folds'][0]['validation'].append('J03WQQ')
        with self.assertRaises(ValueError):self.m.validate_protocol(cfg)

    def test_real_fits_share_windows_scaling_prefix_and_class_weights(self):
        from unittest.mock import patch
        from dataclasses import replace
        o=observations();shifted=replace(o,player_xy_m=o.player_xy_m+2,ball_xy_m=o.ball_xy_m+2)
        y=np.tile(np.array([0,1,2,0]),8);cache=[(o,y),(shifted,y)]
        settings=dict(epochs=1,steps_per_epoch=3,batch_size=2,core_frames=8,learning_rate=.001,weight_decay=.001,gradient_clip=1,team_exchange_probability=.5,cpu_threads=1)
        original=self.m.temporal.sample_windows;criterion=self.m.torch.nn.CrossEntropyLoss
        records=[];weights=[];checkpoints=[]
        for module in (self.m.team_features,self.m.team_features_v3):
            windows=[]
            def sampled(*args,**kwargs):
                result=original(*args,**kwargs)
                windows.append(tuple(a.copy() for a in (result[0][:,:40],result[1],result[2])))
                return result
            def loss_constructor(*args,**kwargs):
                weights.append(kwargs['weight'].clone());return criterion(*args,**kwargs)
            with patch.object(self.m.temporal,'sample_windows',side_effect=sampled),patch.object(self.m.torch.nn,'CrossEntropyLoss',side_effect=loss_constructor):
                checkpoints.append(self.m.fit(cache,module,1701,{'channels':4,'dilations':[1],'kernel_size':3},settings))
            records.append(windows)
        for a,b in zip(records[0],records[1]):
            for x,z in zip(a,b):np.testing.assert_array_equal(x,z)
        np.testing.assert_array_equal(checkpoints[0]['mean'],checkpoints[1]['mean'][:40])
        np.testing.assert_array_equal(checkpoints[0]['scale'],checkpoints[1]['scale'][:40])
        self.m.torch.testing.assert_close(weights[0],weights[1],rtol=0,atol=0)
        np.testing.assert_allclose(weights[0].numpy(),[8/9,8/9,4/3],rtol=1e-6)
