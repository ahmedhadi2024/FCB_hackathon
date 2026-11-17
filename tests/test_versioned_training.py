from pathlib import Path
import tempfile
import json
import unittest
from dataclasses import replace
import numpy as np
from tests.test_temporal import observations
from fcb_external.schema import Targets,write_json
from fcb_external.temporal_experiment import run
from fcb_external.experiment import input_fingerprint


class VersionedTrainingTests(unittest.TestCase):
    def test_promotion_reference_checks_model_scenarios_and_validation_fingerprints(self):
        import fcb_external.temporal_experiment as module
        self.assertTrue(hasattr(module,'verify_reference'),'Promotion reference provenance guard is missing')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'configs').mkdir(); (root/'models').mkdir()
            (root/'models/temporal_v1.pt').write_bytes(b'v1')
            write_json(root/'configs/stress.json',{'scenarios':[]})
            obs=replace(observations(),match_id='val')
            obs.save(root/'artifacts/val/observations')
            Targets(np.array(list('H'*32)),np.ones(32,bool)).save(root/'artifacts/val/targets',obs)
            from fcb_external.schema import digest_file
            ref={'model_sha256':digest_file(root/'models/temporal_v1.pt'),
                 'config_sha256':digest_file(root/'configs/stress.json'),
                 'inputs':{'val':{k:input_fingerprint(root/'artifacts/val'/k) for k in ('observations','targets')}}}
            cfg={'scenario_config':'stress.json'}; split={'validation':['val']}
            module.verify_reference(root,cfg,split,ref)
            for key in ('model_sha256','config_sha256','inputs'):
                wrong=dict(ref); wrong[key]={} if key=='inputs' else 'wrong'
                with self.assertRaises(ValueError): module.verify_reference(root,cfg,split,wrong)

    def test_default_training_will_not_overwrite_released_checkpoint(self):
        project=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'configs').mkdir(); (root/'models').mkdir()
            (root/'configs/temporal_v1.json').write_bytes((project/'configs/temporal_v1.json').read_bytes())
            released=root/'models/temporal_v1.pt'; released.write_bytes(b'keep-release')
            with self.assertRaises(FileExistsError): run(root)
            self.assertEqual(released.read_bytes(),b'keep-release')

    def test_new_augmentation_experiment_keeps_v1_release_and_reports_all_conditions(self):
        project=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'configs').mkdir(); (root/'models').mkdir()
            (root/'models/temporal_v1.pt').write_bytes(b'preserve-v1')
            for match in ('train','val'):
                obs=replace(observations(),match_id=match)
                obs.save(root/'artifacts'/match/'observations')
                labels=np.array(list('H'*10+'A'*10+'D'*12))
                Targets(labels,np.ones(32,bool)).save(root/'artifacts'/match/'targets',obs)
                write_json(root/'artifacts'/match/'manifest.json',{'fixture':True})
            split={'train':['train'],'validation':['val'],'sealed_test':['sealed'],
                   'robustness':[{'name':'clean','seed':1}]}
            write_json(root/'configs/split.json',split)
            write_json(root/'configs/stress.json',{'scenarios':[{'name':'clean','seed':1},
                {'name':'stress','seed':2,'missing_fraction':.6,'gap_min_frames':1,'gap_max_frames':5}]})
            cfg=json.loads((project/'configs/temporal_v1.json').read_text())
            cfg.update(split_protocol='split.json',scenario_config='stress.json',
                       output_name='temporal_v2',pilot_model=str(project/'models/live_dead_v1.json'))
            cfg['network']={'channels':8,'dilations':[1,2],'kernel_size':3}
            cfg['training'].update(epochs=1,steps_per_epoch=2,batch_size=2,core_frames=8,cpu_threads=1)
            cfg['selection']['transition_penalties']=[0,2]
            write_json(root/'configs/temporal_v2.json',cfg)
            try: report=run(root,'temporal_v2.json')
            except TypeError: self.fail('Versioned training configuration support is missing')
            self.assertEqual((root/'models/temporal_v1.pt').read_bytes(),b'preserve-v1')
            self.assertTrue((root/'models/temporal_v2.pt').exists())
            self.assertEqual([r['scenario'] for r in report['conditions']],['clean','stress'])
            self.assertFalse((root/'artifacts/sealed').exists())
