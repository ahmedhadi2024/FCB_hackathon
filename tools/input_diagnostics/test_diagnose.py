import contextlib
import importlib
import io
import json
from pathlib import Path
import tempfile
import unittest


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def module(self):
        try:
            return importlib.import_module('diagnose')
        except ModuleNotFoundError:
            self.fail('The diagnostic module has not been implemented')

    def tracking(self, text, n=4):
        meta = self.root / 'meta.xml'
        meta.write_text('<root><Team id="h"/><Team id="a"/>'
                        '<Player teamId="h"/><Player teamId="a"/></root>')
        raw = self.root / 'raw.txt'
        raw.write_text(text)
        return self.module().audit_tracking(raw, meta, n, 25)

    def tags(self, instances, n=4):
        path = self.root / 'tags.xml'
        path.write_text('<root>' + instances + '</root>')
        return self.module().audit_tags(path, n, 25)

    def test_missing_ball_runs_do_not_infer_dead_ball(self):
        r = self.tracking('1:0,0,0;1,1,0:0,0,0\n'
                          '2:0,0,0;NaN,NaN,NaN:NaN,NaN,NaN\n'
                          '3:0,0,0;1,1,0:NaN,NaN,NaN\n'
                          '4:0,0,0;1,1,0:1,1,0\n')
        self.assertTrue(r['frame_alignment_ok'])
        self.assertEqual(r['ball_observed_frames'], 2)
        self.assertEqual(r['ball_missing_runs']['max_frames'], 2)
        self.assertEqual(r['ball_missing_runs']['count'], 1)
        self.assertEqual(r['player_observations_by_team'], [4, 3])
        self.assertNotIn('dead_ball', r)

    def test_duplicate_and_gap_cannot_cancel(self):
        r = self.tracking('1:0,0,0;1,1,0:0,0,0\n'
                          '2:0,0,0;1,1,0:0,0,0\n'
                          '2:0,0,0;1,1,0:0,0,0\n'
                          '4:0,0,0;1,1,0:0,0,0\n')
        self.assertFalse(r['frame_alignment_ok'])
        self.assertEqual(r['duplicate_frame_ids'], 1)
        self.assertEqual(r['missing_frame_ids'], 1)

    def test_malformed_and_partial_positions(self):
        r = self.tracking('1:0,0,0;1,1,0:0,NaN,0\n'
                          'broken\n3:0,0,0:inf,0,0\n5:0,0,0;1,1,0:0,0,0\n')
        self.assertFalse(r['frame_alignment_ok'])
        self.assertEqual(r['malformed_rows'], 1)
        self.assertEqual(r['out_of_range_ids'], 1)
        self.assertEqual(r['slot_count_mismatch_rows'], 1)
        self.assertEqual(r['ball_observed_frames'], 0)

    def test_empty_tracking_reports_missing_ids(self):
        r = self.tracking('')
        self.assertEqual(r['missing_frame_ids'], 4)
        self.assertFalse(r['frame_alignment_ok'])
        self.assertEqual(r['ball_missing_runs']['max_frames'], 4)

    def test_tags_half_open_overlap_clipping_and_invalid(self):
        r = self.tags('<instance><start>-0.04</start><end>0.08</end><code>Build Up</code>'
                      '<label><group>Team</group><text>Home</text></label></instance>'
                      '<instance><start>0.04</start><end>0.16</end><code>Defending</code>'
                      '<label><group>Team</group><text>Away</text></label></instance>'
                      '<instance><start>1</start><end>0</end></instance>')
        self.assertEqual(r['covered_frames'], 4)
        self.assertEqual(r['multiple_team_tag_frames'], 1)
        self.assertEqual(r['clipped_intervals'], 1)
        self.assertEqual(r['invalid_intervals'], 1)
        self.assertEqual(r['coverage_by_code_frames'], {'Build Up': 2, 'Defending': 3})

    def test_empty_tags(self):
        r = self.tags('')
        self.assertEqual(r['covered_frames'], 0)
        self.assertEqual(r['coverage_by_code_frames'], {})

    def test_bad_metadata_is_rejected(self):
        meta = self.root / 'bad.xml'
        meta.write_text('<root><Team id="h"/><Team id="a"/><Player teamId="other"/></root>')
        raw = self.root / 'raw.txt'
        raw.write_text('')
        with self.assertRaises(ValueError):
            self.module().audit_tracking(raw, meta, 4, 25)

    def test_reordered_ids_are_not_valid_alignment(self):
        r = self.tracking('2:0,0,0;1,1,0:0,0,0\n'
                          '1:0,0,0;1,1,0:0,0,0\n', n=2)
        self.assertEqual(r['missing_frame_ids'], 0)
        self.assertFalse(r['frame_alignment_ok'])

    def test_unknown_team_and_nonfinite_tag_times(self):
        r = self.tags('<instance><start>0</start><end>0.04</end>'
                      '<label><group>Team</group><text>N/A</text></label></instance>'
                      '<instance><start>NaN</start><end>1</end></instance>')
        self.assertEqual(r['instances_without_known_team'], 1)
        self.assertEqual(r['coverage_by_team_tag_frames'], {})
        self.assertEqual(r['invalid_intervals'], 1)

    def test_cli_writes_report_using_only_synthetic_assets(self):
        folder = self.root / 'tactical'
        folder.mkdir()
        (folder / 'BAR-ATH_tactical_atd.xml').write_text(
            '<root><Team id="h"/><Team id="a"/><Player teamId="h"/></root>')
        (folder / 'BAR-ATH_tactical_atd.txt').write_text('1:0,0,0:NaN,NaN,NaN\n')
        (folder / 'BAR-ATH_tactical_smart_tagging.xml').write_text('<root/>')
        out = self.root / 'report.json'
        with contextlib.redirect_stdout(io.StringIO()):
            code = self.module().main(['--data-dir', str(self.root), '--out', str(out),
                                      '--expected-frames', '1'])
        self.assertEqual(code, 0)
        report = json.loads(out.read_text())
        self.assertTrue(report['tracking']['frame_alignment_ok'])
        self.assertEqual(report['smart_tagging']['covered_frames'], 0)


if __name__ == '__main__':
    unittest.main()
