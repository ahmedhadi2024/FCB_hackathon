import importlib
from pathlib import Path
import tempfile
import unittest
import numpy as np
from tests.fixtures import frame, frameset, write_bundle, META


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def convert(self, sets=None, meta=META):
        try:
            module = importlib.import_module('fcb_external.idsse')
        except ModuleNotFoundError:
            self.fail('IDSSE adapter not implemented')
        metadata, positions, _ = write_bundle(self.root, sets, meta)
        return module.convert(positions, metadata, 'TEST')

    def test_metadata_controls_team_membership_and_metres(self):
        obs, targets, audit = self.convert()
        np.testing.assert_array_equal(obs.player_team, [0,0,1])
        np.testing.assert_array_equal(obs.player_xy_m[0,0], [0,0])
        np.testing.assert_array_equal(obs.player_xy_m[0,2], [10,0])
        self.assertEqual((obs.pitch_length_m, obs.pitch_width_m), (105,68))
        np.testing.assert_array_equal(targets.label, ['H','H','A','D'])
        self.assertEqual(audit['status_domains'], {'1':3,'0':1})

    def test_entity_order_does_not_change_alignment(self):
        sets=[frameset('ball','ball',[frame(1),frame(2)]),
              frameset('home','h',[frame(2,2),frame(1,1)])]
        obs, _, _ = self.convert(sets)
        np.testing.assert_array_equal(obs.frame_id, [1,2])
        np.testing.assert_array_equal(obs.player_xy_m[:,0,0], [1,2])

    def test_partial_nonfinite_and_missing_positions(self):
        obs, _, _ = self.convert([frameset('ball','ball',[frame(1,'NaN',0),frame(2,0,'inf')]),
                                  frameset('home','h',[frame(1,1,2)])])
        self.assertFalse(obs.ball_observed.any())
        self.assertTrue(np.isnan(obs.ball_xy_m).all())
        self.assertFalse(obs.player_observed[1].any())

    def test_conflicting_duplicates_rejected(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            self.convert([frameset('ball','ball',[frame(1),frame(1,2)])])

    def test_periods_and_gaps_break_sequences(self):
        obs, _, _ = self.convert([frameset('ball','ball',[frame(1),frame(2),frame(4)]),
                                  frameset('ball','ball',[frame(1),frame(2)],'secondHalf')])
        np.testing.assert_array_equal(obs.sequence_id,[0,0,1,2,2])
        np.testing.assert_array_equal(obs.period,[1,1,1,2,2])

    def test_unknown_targets_excluded_not_dead(self):
        obs, target, _ = self.convert([frameset('ball','ball',[
            frame(1,status='0',possession='garbage'),frame(2,possession='2'),
            frame(3,status='9'),frame(4,possession='0')])])
        np.testing.assert_array_equal(target.label,['D','A','X','X'])
        np.testing.assert_array_equal(target.valid,[True,True,False,False])

    def test_targets_do_not_change_observations(self):
        first, _, _ = self.convert([frameset('ball','ball',[frame(1),frame(2)])])
        second, _, _ = self.convert([frameset('ball','ball',[
            frame(1,status='0',possession='2'),frame(2,status='9')])])
        for name in first.array_names():
            np.testing.assert_array_equal(getattr(first,name),getattr(second,name))

    def test_substitute_slot_is_representable(self):
        obs, _, _ = self.convert([frameset('home','sub',[frame(1,3,4),frame(2,4,5)])])
        self.assertTrue(obs.player_observed[:,1].all())

    def test_wrong_match_rejected(self):
        with self.assertRaisesRegex(ValueError,'match'):
            self.convert(meta=META.replace('DFL-MAT-TEST','DFL-MAT-WRONG'))

    def test_inconsistent_timestamps_rejected(self):
        with self.assertRaisesRegex(ValueError,'timestamp'):
            self.convert([frameset('ball','ball',[frame(1),frame(2)]),
                          frameset('home','h',[frame(1,t='2023-01-01T00:00:01+00:00')])])

    def test_10hz_rejected(self):
        with self.assertRaisesRegex(ValueError,'25 Hz'):
            self.convert([frameset('ball','ball',[frame(1),
                frame(2,t='2023-01-01T00:00:00.100+00:00')])])


if __name__ == '__main__':
    unittest.main()
