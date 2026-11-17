import unittest
import numpy as np
from types import SimpleNamespace

class GapEvidenceTests(unittest.TestCase):
    def test_bridge_requires_agreement_and_sequence_continuity(self):
        from fcb_external.gap_evidence import evidence
        o=SimpleNamespace(ball_observed=np.array([1,0,0,1],bool),
            ball_xy_m=np.array([[0.,0.],[np.nan,np.nan],[np.nan,np.nan],[0.,0.]]),
            player_xy_m=np.tile([[1.,0.],[8.,0.]],(4,1,1)),
            player_observed=np.ones((4,2),bool),player_team=np.array([0,1]),sequence_id=np.zeros(4,int))
        np.testing.assert_array_equal(evidence(o),[-1,0,0,-1])
        o.sequence_id=np.array([0,0,1,1])
        np.testing.assert_array_equal(evidence(o),-1)
        o.sequence_id[:]=0;o.player_xy_m[-1]=[[8.,0.],[1.,0.]]
        np.testing.assert_array_equal(evidence(o),-1)
    def test_ambiguous_and_unbounded_gaps_are_neutral(self):
        from fcb_external.gap_evidence import evidence
        o=SimpleNamespace(ball_observed=np.array([1,0,1],bool),ball_xy_m=np.zeros((3,2)),
            player_xy_m=np.tile([[1.,0.],[1.5,0.]],(3,1,1)),
            player_observed=np.ones((3,2),bool),player_team=np.array([0,1]),sequence_id=np.zeros(3,int))
        np.testing.assert_array_equal(evidence(o),-1)
        o.player_xy_m[:,1,0]=8
        np.testing.assert_array_equal(evidence(o,max_gap=0),-1)
        o.ball_observed[:]=False
        np.testing.assert_array_equal(evidence(o),-1)
