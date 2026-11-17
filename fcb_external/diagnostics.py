"""External-only oracle diagnostics, deliberately separate from model inference."""
import numpy as np
from .model import team_states


def oracle_predictions(gt,pred,sequence_id=None):
    # Status oracle replaces only known live/dead states; it never supplies team ownership.
    status=np.array(pred,dtype='<U1').copy()
    seq=np.zeros(len(gt),dtype=np.int32) if sequence_id is None else sequence_id
    owners=team_states(pred,seq)
    live=np.isin(gt,['H','A'])
    status[live]=owners[live]
    status[gt=='D']='D'
    # A live prediction with the wrong team gets the correct external team.
    # The predicted D mask and predictions during true D remain unchanged.
    team=np.array(pred,dtype='<U1').copy()
    live=np.isin(gt,['H','A'])&(pred!='D')
    team[live]=gt[live]
    return status,team
