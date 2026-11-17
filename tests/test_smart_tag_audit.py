"""Fictional audit fixtures; no match inputs or inference are accessed."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import run_smart_tag_audit as audit


class SmartTagAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'configs').mkdir()
        cfg = {'stem': 'fictional', 'match_id': 'fictional', 'n_frames': 100,
               'pitch_m': [100, 60], 'active_blocks': [[1, 0, 99]]}
        (self.root / 'configs/submission_selected.json').write_text(json.dumps(cfg))
        self.folder = self.root / 'tactical'
        self.folder.mkdir()
        self.metadata = self.folder / 'fictional_atd.xml'
        self.metadata.write_text('<root><Team id="a" name="Amber"/><Team id="v" name="Violet"/></root>')
        self.tracking = self.folder / 'fictional_atd.txt'
        self.tracking.write_text('fictional tracking; converter is mocked')
        self.tags = self.folder / 'fictional_smart_tagging.xml'
        self.write_tag()
        self.obs = SimpleNamespace(frame_id=np.arange(1, 101), period=np.ones(100, int),
                                   ball_observed=np.ones(100, bool))

    def write_tag(self, end='4', team='Amber'):
        self.tags.write_text('<file><instance><code>build up</code><start>0</start><end>' + end +
                            '</end><label><group>Team</group><text>' + team +
                            '</text></label><label><group>Half</group><text>1st half</text></label>'
                            '</instance></file>')

    def execute(self, converter=None):
        stream = io.StringIO()
        output = self.root / 'report.json'
        with patch.object(audit, '__file__', str(self.root / 'run_smart_tag_audit.py')), \
                patch.object(audit.atd, 'convert', side_effect=converter, return_value=self.obs), \
                contextlib.redirect_stdout(stream):
            success = audit.run(self.root, output)
        stdout = stream.getvalue()
        for private in ('fictional', 'Amber', 'Violet', str(self.root)):
            self.assertNotIn(private, stdout)
        return success, json.loads(output.read_text())

    def assert_contract(self, expected, report):
        self.assertTrue(report['audit_executed'])
        self.assertEqual(report['contract_passed'], expected)

    def test_successful_contract(self):
        success, report = self.execute()
        self.assertTrue(success)
        self.assert_contract(True, report)
        self.assertTrue(all(report['required_checks'].values()))

    def test_unresolved_mapping_fails_contract(self):
        self.metadata.write_text('<root><Team id="a"/><Team id="v"/></root>')
        success, report = self.execute()
        self.assertFalse(success)
        self.assert_contract(False, report)
        self.assertFalse(report['required_checks']['resolved_team_mapping'])

    def test_incomplete_timeline_fails_contract(self):
        self.obs.frame_id = np.arange(1, 100)
        self.obs.period = np.ones(99, int)
        self.obs.ball_observed = np.ones(99, bool)
        success, report = self.execute()
        self.assertFalse(success)
        self.assert_contract(False, report)
        self.assertFalse(report['required_checks']['complete_timeline'])

    def test_out_of_bounds_tag_fails_contract(self):
        self.write_tag(end='4.04')
        success, report = self.execute()
        self.assertFalse(success)
        self.assert_contract(False, report)
        self.assertFalse(report['required_checks']['tag_bounds'])

    def test_rounded_empty_interval_fails_bounds(self):
        self.write_tag(end='0.001')
        success, report = self.execute()
        self.assertFalse(success)
        self.assert_contract(False, report)
        self.assertFalse(report['required_checks']['tag_bounds'])

    def test_input_mutation_fails_contract(self):
        def mutate(*args):
            self.tracking.write_text('changed fictional tracking')
            return self.obs
        success, report = self.execute(mutate)
        self.assertFalse(success)
        self.assert_contract(False, report)
        self.assertFalse(report['required_checks']['input_immutability'])

    def test_kickoff_diagnostic_is_not_required(self):
        success, report = self.execute()
        self.assertTrue(success)
        self.assertEqual(report['kickoff_containment'], {'1': False, '2': False})
        self.assertNotIn('kickoff_containment', report['required_checks'])

    def test_unknown_team_is_neutral(self):
        self.write_tag(team='Unknown fictional club')
        self.obs.ball_observed = np.zeros(100, bool)
        success, report = self.execute()
        self.assertTrue(success)
        self.assert_contract(True, report)
        self.assertEqual(report['unknown_tag_teams'], ['unknown fictional club'])
        self.assertEqual(report['missing_ball_with_phase_evidence'], 0)

    def test_missing_inputs_marks_audit_not_executed(self):
        self.tracking.unlink()
        success, report = self.execute()
        self.assertFalse(success)
        self.assertFalse(report['audit_executed'])
        self.assertFalse(report['contract_passed'])
        self.assertIn('local_error', report)


if __name__ == '__main__':
    unittest.main()
