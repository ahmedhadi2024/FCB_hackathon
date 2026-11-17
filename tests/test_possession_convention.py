"""Fictional labels and tags only: no competition fixtures or ground truth."""
import unittest
import numpy as np
from fcb_external.possession_convention import absorb_short_spells
from fcb_external.smart_tagging import Tag, kickoff_dead_mask


def expand(spec):
    return np.array([label for label,count in spec for _ in range(count)])


def tag(start,end,half,kind='kick off (start)',code='set pieces'):
    return Tag(code,start/25,end/25,start,end,None,half,kind,())


class ConventionTests(unittest.TestCase):
    def absorb(self,spec,min_frames=50,seq=None):
        labels=expand(spec)
        seq=np.zeros(len(labels),np.int32) if seq is None else seq
        return absorb_short_spells(labels,seq,min_frames)

    def test_uncontrolled_touch_stays_with_last_team(self):
        out=self.absorb([('H',100),('A',20),('H',100)])
        np.testing.assert_array_equal(out,expand([('H',220)]))

    def test_controlled_change_is_kept_at_its_first_frame(self):
        spec=[('H',100),('A',50),('H',100)]
        np.testing.assert_array_equal(self.absorb(spec),expand(spec))

    def test_tiny_interruption_does_not_break_a_real_spell(self):
        out=self.absorb([('H',100),('A',40),('H',3),('A',40),('H',100)])
        np.testing.assert_array_equal(out,expand([('H',100),('A',83),('H',100)]))

    def test_restart_possessor_dead_ball_and_unknown_are_kept(self):
        spec=[('D',30),('A',10),('H',100),('X',5),('A',10),('D',2),('H',5)]
        np.testing.assert_array_equal(self.absorb(spec),expand(spec))

    def test_no_carry_across_sequences(self):
        labels=expand([('H',100),('A',20)]); seq=np.r_[np.zeros(100,np.int32),np.ones(20,np.int32)]
        np.testing.assert_array_equal(absorb_short_spells(labels,seq,50),labels)

    def test_disabled_and_invalid_inputs(self):
        labels=expand([('H',5),('A',1),('H',5)]); seq=np.zeros(11,np.int32)
        np.testing.assert_array_equal(absorb_short_spells(labels,seq,0),labels)
        for bad in (-1,2.5,True):
            with self.assertRaises(ValueError):
                absorb_short_spells(labels,seq,bad)
        with self.assertRaises(ValueError):
            absorb_short_spells(np.array(['H','Q']),np.zeros(2,np.int32),5)
        with self.assertRaises(ValueError):
            absorb_short_spells(labels,seq[:-1],5)

    def test_result_never_adds_possessions(self):
        rng=np.random.default_rng(3)
        labels=np.array(['H','A','D'])[np.repeat(rng.integers(0,3,300),rng.integers(1,40,300))]
        seq=np.zeros(len(labels),np.int32)
        out=absorb_short_spells(labels,seq,50)
        np.testing.assert_array_equal(out=='D',labels=='D')
        changes=lambda v:int(np.sum(v[1:]!=v[:-1]))
        self.assertLessEqual(changes(out),changes(labels))


class KickoffRuleTests(unittest.TestCase):
    def setUp(self):
        self.frames=np.arange(1000)
        self.period=np.r_[np.zeros(100),np.ones(300),np.zeros(100),np.full(400,2),np.zeros(100)].astype(np.int8)

    def test_marks_only_frames_before_the_tag_inside_its_half(self):
        mask=kickoff_dead_mask([tag(90,110,'1st half'),tag(620,650,'2nd half')],self.frames,self.period)
        np.testing.assert_array_equal(np.flatnonzero(mask),np.arange(500,620))

    def test_ambiguous_missing_late_or_wrong_tags_change_nothing(self):
        cases=[[],
               [tag(620,650,'2nd half'),tag(700,720,'2nd half')],
               [tag(620,650,'2nd half',kind='kick off (after goal)')],
               [tag(620,650,'2nd half',code='build up')],
               [tag(950,960,'2nd half')],
               [tag(620,650,None)]]
        for tags in cases:
            self.assertFalse(kickoff_dead_mask(tags,self.frames,self.period).any())
        late=kickoff_dead_mask([tag(620,650,'2nd half')],self.frames,self.period,max_lead_frames=50)
        self.assertFalse(late.any())


if __name__=='__main__':
    unittest.main()
