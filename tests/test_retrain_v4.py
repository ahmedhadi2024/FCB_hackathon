"""Fictional external matches only: no competition fixtures or ground truth."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import numpy as np
from fcb_external import retrain_v4, state_decoder
from fcb_external.schema import Observations, Targets, digest_file

PROJECT=Path(__file__).resolve().parents[1]


def synthetic_match(root,name,seed,n=900):
    rng=np.random.default_rng(seed)
    owner=np.repeat(rng.integers(0,3,n//60+1),60)[:n]
    label=np.array(['H','A','D'])[owner]
    home=np.array([[-20.,0.],[-10.,5.]]); away=np.array([[20.,0.],[10.,-5.]])
    players=np.tile(np.concatenate([home,away])[None],(n,1,1)).astype(np.float32)
    players+=rng.normal(0,.2,players.shape).astype(np.float32)
    ball=np.where((owner==0)[:,None],home[0],away[0]).astype(np.float32)
    ball[owner==2]=[0,40]
    ball+=rng.normal(0,.3,ball.shape).astype(np.float32)
    obs=Observations(name,np.ones(n,np.int8),np.arange(1,n+1),np.arange(n)/25,players,
        np.ones((n,4),bool),np.array([0,0,1,1],np.int8),ball,np.ones(n,bool),
        np.zeros(n,np.int32),105.,68.,25.)
    folder=Path(root)/'artifacts'/name
    obs.save(folder/'observations')
    Targets(label,label!='X').save(folder/'targets',obs)


def tiny_config():
    return {'output_name':'tiny_v4','matches':['M1','M2','M3'],
        'folds':[{'name':'A','seed':1,'train':['M1','M2'],'validation':['M3']},
                 {'name':'B','seed':2,'train':['M3'],'validation':['M1','M2']}],
        'final_refit_seed':3,
        'network':{'channels':8,'dilations':[1,2],'kernel_size':3},
        'training':{'epochs':2,'steps_per_epoch':3,'batch_size':4,'core_frames':64,'learning_rate':.01,
            'weight_decay':.001,'gradient_clip':1.0,'scaling_sample_frames_per_condition':200,
            'cpu_threads':1,'team_exchange_probability':.5},
        'scenarios':[{'name':'clean'},
            {'name':'irregular40','missing_fraction':.4,'gap_min_frames':5,'gap_max_frames':60}],
        'training_scenarios':['clean','irregular40'],'evaluation_scenarios':['clean','irregular40'],
        'selection_scenarios':['irregular40'],
        'decoder_grid':{'owner':[2,5],'dead_live':[5,10]},
        'comparator':{'model':'models/temporal_v2.pt','model_sha256':digest_file(PROJECT/'models/temporal_v2.pt'),
            'unseen_matches':['M3'],'minimum_mean_gain':.02,'maximum_clean_loss':.02},
        'submission_template':'submission_selected.json'}


def cells(cfg,v4_score,v2_score):
    keys=[retrain_v4.pair_key(p) for p in retrain_v4.grid_pairs(cfg)]
    fold={m:f['name'] for f in cfg['folds'] for m in f['validation']}
    return [{'fold':fold[m],'match':m,'scenario':s,'v2':{'spellF1':v2_score(m,s)},
             'v4':{k:{'spellF1':v4_score(m,s,k)} for k in keys}}
            for m in cfg['matches'] for s in cfg['evaluation_scenarios']]


class RetrainTests(unittest.TestCase):
    def test_shipped_config_is_valid_and_pins_the_released_v2(self):
        cfg=json.loads((PROJECT/'configs/retrain_v4.json').read_text())
        retrain_v4.validate_config(cfg)
        self.assertEqual(cfg['comparator']['model_sha256'],digest_file(PROJECT/cfg['comparator']['model']))
        selected=json.loads((PROJECT/'configs/submission_selected.json').read_text())
        self.assertEqual(cfg['comparator']['model_sha256'],selected['model_sha256'])

    def test_v5_config_is_valid_and_relabels_targets(self):
        import numpy as np
        from fcb_external.schema import Targets
        cfg=json.loads((PROJECT/'configs/retrain_v5.json').read_text())
        retrain_v4.validate_config(cfg)
        self.assertEqual(cfg['output_name'],'temporal_v5')
        self.assertEqual(retrain_v4.min_controlled(cfg),50)
        label=np.array(['H']*100+['A']*20+['H']*100)
        class Obs: sequence_id=np.zeros(220,np.int32)
        out=retrain_v4.convention_labels(cfg,Targets(label,label!='X'),Obs)
        self.assertTrue((out=='H').all())
        self.assertIs(retrain_v4.convention_labels(tiny_config(),Targets(label,label!='X'),Obs).base,None)
        bad=dict(cfg,controlled_possession={'min_frames':-1})
        with self.assertRaises(ValueError):
            retrain_v4.validate_config(bad)

    def test_config_rejects_bad_folds_scenarios_and_costs(self):
        base=tiny_config(); retrain_v4.validate_config(base)
        def broken(change):
            cfg=copy.deepcopy(base); change(cfg); return cfg
        cases=[lambda c:c['folds'][0]['validation'].append('M1'),
               lambda c:c['folds'].pop(),
               lambda c:c['selection_scenarios'].append('unknown'),
               lambda c:c['evaluation_scenarios'].remove('irregular40'),
               lambda c:c['decoder_grid']['owner'].append(0),
               lambda c:c['comparator']['unseen_matches'].append('M9')]
        for change in cases:
            with self.assertRaises(ValueError):
                retrain_v4.validate_config(broken(change))

    def test_batched_decoder_equals_reference_decoder(self):
        rng=np.random.default_rng(7)
        p=rng.dirichlet([.4,.4,.4],size=400); seq=np.repeat([0,1,2],[150,50,200]).astype(np.int32)
        pairs=[(0.,0.),(1.,3.),(4.,1.),(10.,10.),(2.5,7.)]
        many=retrain_v4.decode_many(p,seq,pairs)
        names=np.array(['H','A','D'])
        for pair,row in zip(pairs,many):
            np.testing.assert_array_equal(names[row],state_decoder.decode(p,seq,*pair))

    def test_decision_selects_on_missing_ball_scenarios_and_gates_on_unseen_match(self):
        cfg=tiny_config()
        def v4(match,scenario,key):
            if scenario=='clean':
                return .9 if key=='2/5' else .5
            return .6 if key=='5/10' else .4
        result=retrain_v4.decide(cells(cfg,v4,lambda m,s:.5),cfg)
        self.assertEqual(result['selected_costs'],[5.,10.])
        self.assertTrue(result['promote'])
        self.assertAlmostEqual(result['unseen_mean_gain_vs_v2'],.1)
        self.assertEqual(result['selected_at_grid_edge'],
                         {'owner_at_grid_maximum':True,'owner_at_grid_minimum':False,
                          'dead_live_at_grid_maximum':True,'dead_live_at_grid_minimum':False})
        small=retrain_v4.decide(cells(cfg,v4,lambda m,s:.59),cfg)
        self.assertFalse(small['promote'])
        clean_loss=retrain_v4.decide(cells(cfg,v4,lambda m,s:.55 if s=='clean' else .3),cfg)
        self.assertFalse(clean_loss['promote'])
        with self.assertRaises(ValueError):
            retrain_v4.decide(cells(cfg,v4,lambda m,s:.5)[:-1],cfg)

    def test_cv_then_final_refit_then_submission_inference(self):
        import torch
        from fcb_external import submission
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); cfg=tiny_config()
            cfg['controlled_possession']={'min_frames':10}; cfg['kickoff_dead_rule']=True
            (root/'configs').mkdir(); (root/'models').mkdir()
            shutil.copy(PROJECT/'models/temporal_v2.pt',root/'models/temporal_v2.pt')
            template={'active_blocks':[[1,1,3],[2,5,6]],'match_id':'SYNTHETIC','n_frames':8,
                      'metadata_team_policy':'epts_first_team_is_home','pitch_m':[105,68],
                      'source':'tactical','stem':'synthetic','model':'x','model_sha256':'x',
                      'model_kind':'pytorch_temporal_conv_v1'}
            (root/'configs/submission_selected.json').write_text(json.dumps(template))
            (root/'configs/retrain_v4.json').write_text(json.dumps(cfg))
            for i,name in enumerate(cfg['matches']):
                synthetic_match(root,name,i)
            decision=retrain_v4.run_cv(root)
            self.assertIn(decision['selected_costs'][0],cfg['decoder_grid']['owner'])
            self.assertEqual(len(json.loads((root/'reports/tiny_v4/A.json').read_text())['cells']),2)
            with self.assertRaises(FileExistsError):
                retrain_v4.run_cv(root)
            path=root/'reports/tiny_v4/decision.json'
            saved=json.loads(path.read_text()); saved['promote']=False; path.write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError,'promote'):
                retrain_v4.run_final(root)
            saved['promote']=True; path.write_text(json.dumps(saved))
            retrain_v4.run_final(root)
            checkpoint=torch.load(root/'models/tiny_v4.pt',map_location='cpu',weights_only=True)
            self.assertEqual(checkpoint['kind'],retrain_v4.KIND)
            self.assertEqual(checkpoint['selected_transition_costs'],decision['selected_costs'])
            with self.assertRaises(FileExistsError):
                retrain_v4.run_final(root)
            # The written config must drive the ordinary submission exporter end to end.
            folder=root/'data'/'tactical'; folder.mkdir(parents=True)
            (folder/'synthetic_atd.xml').write_text('<root><Team id="h"/><Team id="a"/>'
                '<Player teamId="h"/><Player teamId="a"/></root>')
            (folder/'synthetic_atd.txt').write_text(''.join(
                f'{i}:0.4,0.5,0;0.8,0.5,0:0.41,0.5,0\n' for i in range(1,9)))
            (folder/'synthetic_smart_tagging.xml').write_text('<file><instance><start>0.08</start><end>0.2</end>'
                '<code>SET PIECES</code><label><group>Type</group><text>Kick Off (Start)</text></label>'
                '<label><group>Half</group><text>1st Half</text></label></instance></file>')
            self.assertEqual(checkpoint['min_controlled_frames'],10)
            result=submission.run(root/'configs/submission_tiny_v4.json',root/'data',root/'v4.csv')
            self.assertTrue(result['kickoff_dead_rule'])
            self.assertEqual(result['min_controlled_frames'],10)
            override=json.loads((root/'configs/submission_tiny_v4.json').read_text()); override['min_controlled_frames']=3
            (root/'configs/override.json').write_text(json.dumps(override))
            again=submission.run(root/'configs/override.json',root/'data',root/'override.csv')
            self.assertEqual(again['min_controlled_frames'],3)
            self.assertEqual((root/'v4.csv').read_text().splitlines()[2],'1,D')
            self.assertEqual(result['validation'],'official_format_pass')
            self.assertEqual(result['model_kind'],retrain_v4.KIND)
            rows=(root/'v4.csv').read_text().splitlines()
            self.assertEqual(len(rows),9)
            self.assertEqual(rows[1],'0,X')


if __name__=='__main__':
    unittest.main()
