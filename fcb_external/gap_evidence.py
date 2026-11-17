"""Observation-only gap endpoint clue; phase/possession labels are never inputs."""
import numpy as np
from .features import sequence_spans


def evidence(obs,max_gap=125):
    if not isinstance(max_gap,int) or max_gap<0:raise ValueError('Nonnegative integer gap required')
    n=len(obs.ball_observed);result=np.full(n,-1,np.int8)
    distance=np.linalg.norm(obs.player_xy_m-obs.ball_xy_m[:,None,:],axis=-1)
    distance[~obs.player_observed|~obs.ball_observed[:,None]]=np.inf
    closest=[]
    for team in (0,1):
        block=distance[:,obs.player_team==team]
        closest.append(block.min(1) if block.shape[1] else np.full(n,np.inf))
    d=np.column_stack(closest);side=d.argmin(1);best=d[np.arange(n),side];other=d[np.arange(n),1-side]
    confident=obs.ball_observed&(best<=2)&(other>=best+1)
    for lo,hi in sequence_spans(obs.sequence_id):
        seen=np.flatnonzero(obs.ball_observed[lo:hi])+lo
        for left,right in zip(seen[:-1],seen[1:]):
            gap=right-left-1
            if 0<gap<=max_gap and confident[left] and confident[right] and side[left]==side[right]:
                result[left+1:right]=side[left]
    return result
