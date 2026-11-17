"""Learning and sequence-boundary tests on fictional observations only."""
from dataclasses import replace
import importlib
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from fcb_external.schema import Observations


def observations():
    n=32
    players=np.tile(np.array([[0.,0.],[1.,0.],[10.,0.],[11.,0.]],np.float32),(n,1,1))
    return Observations('SYNTHETIC',np.ones(n,dtype=np.int8),np.arange(1,n+1),
        np.arange(n)/25,players,np.ones((n,4),bool),np.array([0,0,1,1],np.int8),
        np.zeros((n,2),np.float32),np.ones(n,bool),np.zeros(n,np.int32),105.,68.,25.)


class TemporalTests(unittest.TestCase):
    def mod(self,name):
        try:
            return importlib.import_module('fcb_external.'+name)
        except ModuleNotFoundError:
            self.fail(name+' is missing')

    def test_team_features_permutation_exchange_and_missingness(self):
        m=self.mod('team_features'); obs=observations()
        x=m.extract(obs)
        order=np.array([1,3,0,2])
        perm=replace(obs,player_xy_m=obs.player_xy_m[:,order],
            player_observed=obs.player_observed[:,order],player_team=obs.player_team[order])
        np.testing.assert_allclose(x,m.extract(perm))
        swapped=replace(obs,player_team=1-obs.player_team)
        np.testing.assert_allclose(m.swap(x),m.extract(swapped))
        np.testing.assert_allclose(m.swap(m.swap(x)),x)
        missing=replace(obs,ball_xy_m=np.full((32,2),np.nan,np.float32),ball_observed=np.zeros(32,bool))
        self.assertTrue(np.isfinite(m.extract(missing)).all())
        self.assertFalse(np.array_equal(x,m.extract(missing)))

    def test_network_chunking_and_period_isolation(self):
        m=self.mod('temporal'); m.configure(1,1)
        net=m.TemporalNet(4,8,[1,2,4])
        rng=np.random.default_rng(3)
        x=rng.normal(size=(80,4)).astype(np.float32); seq=np.repeat([0,1],40)
        whole=m.probabilities(net,x,seq,chunk_frames=1000)
        chunks=m.probabilities(net,x,seq,chunk_frames=9)
        np.testing.assert_allclose(whole,chunks,atol=1e-6)
        x[:40]=1000
        changed=m.probabilities(net,x,seq,chunk_frames=9)
        np.testing.assert_array_equal(changed[40:],chunks[40:])
        np.testing.assert_allclose(chunks.sum(axis=1),1,atol=1e-6)

    def test_network_learns_tiny_three_class_problem_and_reload_matches(self):
        m=self.mod('temporal'); m.configure(2,1)
        net=m.TemporalNet(3,12,[1,2])
        x=torch.zeros((3,3,64)); y=torch.arange(3)[:,None].repeat(1,64)
        for i in range(3):
            x[i,i]=2
        optimizer=torch.optim.AdamW(net.parameters(),lr=.02)
        criterion=torch.nn.CrossEntropyLoss()
        for _ in range(40):
            optimizer.zero_grad(); loss=criterion(net(x),y); loss.backward(); optimizer.step()
        self.assertGreater(float((net(x).argmax(1)==y).float().mean()),.98)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'weights.pt'
            torch.save(net.state_dict(),path)
            clone=m.TemporalNet(3,12,[1,2])
            clone.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
            torch.testing.assert_close(net(x),clone(x),rtol=0,atol=0)

    def test_training_padding_mask_matches_truncated_inference_at_sequence_edges(self):
        m=self.mod('temporal'); m.configure(3,1)
        net=m.TemporalNet(2,8,[1,2,4])
        x=torch.randn(1,2,12)
        padded=torch.nn.functional.pad(x,(7,7),value=5)
        mask=torch.zeros(1,1,26); mask[:,:,7:19]=1
        try:
            actual=net(padded,mask)[:,:,7:19]
        except TypeError:
            self.fail('Training boundary mask is missing')
        torch.testing.assert_close(actual,net(x),atol=1e-6,rtol=1e-6)

    def test_sampled_window_mask_marks_halo_as_observed_but_not_padding(self):
        m=self.mod('temporal')
        x=np.ones((4,2),np.float32); y=np.array([0,-100,1,2]); seq=np.zeros(4,int)
        try:
            windows,labels,mask=m.sample_windows(x,y,seq,np.random.default_rng(1),4,4,2,return_mask=True)
        except TypeError:
            self.fail('Observation boundary mask is missing')
        np.testing.assert_array_equal(mask[:,0,:],windows[:,0,:]!=0)
        self.assertTrue(np.all(mask.sum(axis=2)<=4))

    def test_decoder_stability_reset_and_invalid_probabilities(self):
        m=self.mod('temporal')
        p=np.array([[.9,.05,.05],[.4,.5,.1],[.9,.05,.05],[.05,.9,.05]])
        seq=np.array([0,0,0,1])
        np.testing.assert_array_equal(m.decode(p,seq,0),list('HAHA'))
        np.testing.assert_array_equal(m.decode(p,seq,2),list('HHHA'))
        with self.assertRaises(ValueError):
            m.decode(np.array([[np.nan,0,1]]),np.array([0]),2)

    def test_windows_and_labels_do_not_cross_sequence_or_include_unknown_targets(self):
        m=self.mod('temporal')
        x=np.r_[np.ones((4,2)),np.full((4,2),9)].astype(np.float32)
        y=np.array([0,0,-100,0,1,1,1,1]); seq=np.array([0]*4+[1]*4)
        windows,labels=m.sample_windows(x,y,seq,np.random.default_rng(1),20,2,1)
        self.assertTrue(np.isin(labels,[-100,0,1]).all())
        for window in windows:
            seen=np.unique(window[window!=0])
            self.assertEqual(len(seen),1)

    def test_checkpoint_predictions_are_team_equivariant_and_scaling_is_preserved(self):
        m=self.mod('temporal_experiment'); obs=observations()
        temporal=self.mod('temporal'); temporal.configure(1,1)
        features=self.mod('team_features')
        net=temporal.TemporalNet(len(features.NAMES),8,[1,2])
        checkpoint={'network':{'channels':8,'dilations':[1,2]},
                    'state_dict':net.state_dict(),'mean':torch.zeros(len(features.NAMES)),
                    'scale':torch.ones(len(features.NAMES)), 'features':list(features.NAMES)}
        before=m.infer(checkpoint,obs,7)
        changed=m.infer(checkpoint,replace(obs,player_team=1-obs.player_team),7)
        np.testing.assert_allclose(before,changed[:,[1,0,2]],atol=1e-6)

    def test_experiment_smoke_trains_external_only_and_saves_reproducible_checkpoint(self):
        import json
        from fcb_external.schema import Targets,write_json
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'configs').mkdir()
            obs=observations()
            for role in ('train','val'):
                item=replace(obs,match_id=role)
                item.save(root/'artifacts'/role/'observations')
                labels=np.array(list('H'*10+'A'*10+'D'*12))
                Targets(labels,np.ones(32,bool)).save(root/'artifacts'/role/'targets',item)
                write_json(root/'artifacts'/role/'manifest.json',{'fixture':True})
            protocol={'train':['train'],'validation':['val'],'sealed_test':['sealed'],
                      'robustness':[{'name':'clean','seed':1}]}
            write_json(root/'configs'/'split.json',protocol)
            project=Path(__file__).resolve().parents[1]
            cfg=json.loads((project/'configs/temporal_v1.json').read_text())
            cfg['split_protocol']='split.json'
            cfg['network']={'channels':8,'dilations':[1,2],'kernel_size':3}
            cfg['training'].update(epochs=1,steps_per_epoch=2,batch_size=2,core_frames=8,cpu_threads=1)
            cfg['selection']['transition_penalties']=[0,2]
            cfg['pilot_model']=str(project/'models/live_dead_v1.json')
            write_json(root/'configs/temporal_v1.json',cfg)
            report=self.mod('temporal_experiment').run(root)
            self.assertEqual(report['test_matches_evaluated'],[])
            self.assertEqual(set(report['training_inputs']),{'train'})
            self.assertEqual(set(report['validation_inputs']),{'val'})
            self.assertFalse((root/'artifacts/sealed').exists())
            saved=torch.load(root/'models/temporal_v1.pt',map_location='cpu',weights_only=True)
            a=self.mod('temporal_experiment').infer(saved,replace(obs,match_id='val'),9)
            b=self.mod('temporal_experiment').infer(saved,replace(obs,match_id='val'),100)
            np.testing.assert_allclose(a,b,atol=1e-6)


if __name__=='__main__':
    unittest.main()
