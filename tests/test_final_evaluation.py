import importlib
import json
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace
import numpy as np
from tests.test_temporal import observations
from fcb_external.schema import Targets,digest_file,write_json


class FinalEvaluationTests(unittest.TestCase):
    def frozen_fixture(self,root):
        from shutil import copyfile
        project=Path(__file__).resolve().parents[1]
        names=self.module().deployment_files(project)
        for name in names:
            path=root/name; path.parent.mkdir(parents=True,exist_ok=True)
            copyfile(project/name,path)
        model=root/'model.pt'; model.write_bytes(b'fixture')
        return {'selection_closed':True,'selected_model':'model.pt','model_sha256':digest_file(model),
                'frozen_files':{name:digest_file(root/name) for name in names}}

    def module(self):
        try: return importlib.import_module('fcb_external.final_evaluation')
        except ModuleNotFoundError: self.fail('Separate frozen final evaluation is missing')

    def test_frozen_hash_guard_rejects_changed_model_and_config_before_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); cfg=self.frozen_fixture(root); model=root/'model.pt'
            self.module().verify_frozen(root,cfg)
            model.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'checksum'):
                self.module().verify_frozen(root,cfg)

    def test_freeze_rejects_incomplete_manifest_and_changed_runtime_components(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); cfg=self.frozen_fixture(root)
            original=cfg['frozen_files']; cfg['frozen_files']={}
            with self.assertRaises(ValueError): self.module().verify_frozen(root,cfg)
            cfg['frozen_files']=original
            for name in ('run_final_evaluation.py','vendor/official/evaluation/possession_metrics.py',
                         'requirements-cpu-lock.txt','configs/robustness_v2.json','configs/experiment_v1.json'):
                path=root/name; content=path.read_bytes(); path.write_bytes(content+b'\n')
                with self.assertRaises(ValueError): self.module().verify_frozen(root,cfg)
                path.write_bytes(content)

    def test_final_loader_accepts_only_released_sealed_role_and_match_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); obs=replace(observations(),match_id='sealed')
            obs.save(root/'artifacts/sealed/observations')
            Targets(np.array(list('H'*32)),np.ones(32,bool)).save(root/'artifacts/sealed/targets',obs)
            cfg={'matches':['sealed']}
            protocol={'train':['train'],'validation':['val'],'sealed_test':['sealed']}
            a,b=self.module().load_final(root,'sealed',cfg,protocol)
            self.assertEqual(a.match_id,'sealed')
            with self.assertRaises(ValueError): self.module().load_final(root,'train',cfg,protocol)
            from fcb_external.experiment import load_match
            with self.assertRaises(ValueError): load_match(root,'sealed','train',protocol)
            del a,b

    def test_selection_must_be_closed(self):
        with self.assertRaisesRegex(ValueError,'closed'):
            self.module().verify_frozen(Path('.'),{'selection_closed':False})
