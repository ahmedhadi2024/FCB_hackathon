"""Synthetic checks for the two bounded development experiments."""
import copy
from pathlib import Path
import tempfile
import unittest

from fcb_external import bounded_experiment as bounded


SCENARIOS = [
    {'name': 'clean', 'seed': 100},
    {'name': 'short_occlusion', 'seed': 101, 'ball_gap_frames': 25,
     'ball_gap_period_frames': 250},
    {'name': 'long_occlusion', 'seed': 102, 'ball_gap_frames': 125,
     'ball_gap_period_frames': 500},
    {'name': 'mixed_noise', 'seed': 103, 'ball_gap_frames': 50,
     'ball_gap_period_frames': 250, 'player_drop_probability': .1,
     'coordinate_noise_std_m': .5, 'permute_slots_within_team': True},
]


class BoundedExperimentTests(unittest.TestCase):
    def test_reordered_definitions_use_frozen_order_and_copies(self):
        config = {'scenarios': [{'name': 'other', 'seed': 999}] + SCENARIOS[::-1]}
        selected = bounded.frozen_scenarios(config)
        self.assertEqual(selected, SCENARIOS)
        selected[0]['seed'] = -1
        self.assertEqual(bounded.frozen_scenarios(config), SCENARIOS)

    def test_missing_duplicate_and_changed_definitions_fail(self):
        changed = copy.deepcopy(SCENARIOS)
        changed[1]['ball_gap_frames'] = 26
        extra = copy.deepcopy(SCENARIOS)
        extra[0]['extra'] = True
        wrong_type = copy.deepcopy(SCENARIOS)
        wrong_type[3]['permute_slots_within_team'] = 1
        for scenarios in (SCENARIOS[1:], SCENARIOS + [SCENARIOS[0]], changed, extra, wrong_type):
            with self.subTest(scenarios=scenarios), self.assertRaises(ValueError):
                bounded.frozen_scenarios({'scenarios': scenarios})

    def test_provenance_detects_source_config_model_and_input_mutations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = ['runner.py', 'configs/experiment_v1.json',
                     'configs/robustness_v2.json', 'models/model.pt',
                     'artifacts/synthetic/observations/metadata.json',
                     'artifacts/synthetic/observations/frame_id.npy',
                     'artifacts/synthetic/targets/metadata.json',
                     'artifacts/synthetic/targets/label.npy',
                     'artifacts/synthetic/manifest.json']
            for name in paths:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'original')
            snapshot = bounded.capture_provenance(root, 'runner.py', 'models/model.pt', 'synthetic')
            self.assertIn('bounded_experiment.py', snapshot['source_sha256'])
            self.assertIn('runner.py', snapshot['source_sha256'])
            self.assertEqual(set(snapshot['external_inputs']), {'observations', 'targets'})
            bounded.verify_provenance(snapshot, root, 'runner.py', 'models/model.pt', 'synthetic')
            for name in paths:
                with self.subTest(path=name):
                    path = root / name
                    path.write_bytes(b'mutated')
                    with self.assertRaisesRegex(ValueError, 'provenance changed'):
                        bounded.verify_provenance(snapshot, root, 'runner.py', 'models/model.pt', 'synthetic')
                    path.write_bytes(b'original')


if __name__ == '__main__':
    unittest.main()
