"""Player-tracking faults shaped like the competition tracking's measured, label-free profile.

Measured on the competition ATD over graded frames (observations only, no answers):
  - a team has fewer than 11 tracked players on about 36% of frames (10 players 22%, 9 players
    5.5%, 8 players 2.5%, 7 or fewer 2%), in episodes of median about 4 s (p75 9-15 s, p90 35 s);
  - a team has more than 11 tracked players (a duplicate track) on about 5-8% of frames.
Clean external tracking shows all 22 players on every frame, so the network would otherwise
meet these player counts only at inference.

apply() reproduces both properties on a clean external match. It uses a seed and the
observations only, never a label.
"""
from dataclasses import replace
import numpy as np
from .features import sequence_spans

MISSING_SHARE=0.36
EXTRA_SHARE=0.065
LOG_MEDIAN_S=np.log(4.0)
SIGMA=1.69                     # gives p75 about 12 s and p90 about 35 s
MIN_S,MAX_S=0.4,120.0
DROP_COUNTS=(1,2,3,4,5)
DROP_PROBABILITY=(0.61,0.15,0.07,0.035,0.035)
DUPLICATE_OFFSET_M=1.0


def _episodes(n,share,rng):
    """Boolean mask and episode list covering about `share` of n frames with log-normal lengths."""
    covered=np.zeros(n,bool); episodes=[]; target=int(n*share); guard=0
    while covered.sum()<target and guard<100000:
        guard+=1
        length=max(1,int(np.clip(np.exp(rng.normal(LOG_MEDIAN_S,SIGMA)),MIN_S,MAX_S)*25))
        start=int(rng.integers(0,n)); end=min(n,start+length)
        if covered[start:end].any():
            continue
        covered[start:end]=True; episodes.append((start,end))
    return episodes


def apply(obs,seed,missing_share=MISSING_SHARE,extra_share=EXTRA_SHARE):
    if not 0<=missing_share<1 or not 0<=extra_share<1:
        raise ValueError('Invalid fault shares')
    rng=np.random.default_rng(seed)
    players=np.array(obs.player_xy_m,dtype=np.float32); teams=np.array(obs.player_team)
    n=len(players)
    extra=np.full((n,2,2),np.nan,np.float32)        # one spare slot per team for duplicate tracks
    probability=np.array(DROP_PROBABILITY)/sum(DROP_PROBABILITY)
    for team in (0,1):
        slots=np.flatnonzero(teams==team)
        if not len(slots):
            continue
        for lo,hi in sequence_spans(obs.sequence_id):
            for start,end in _episodes(hi-lo,missing_share,rng):
                count=min(len(slots),int(rng.choice(DROP_COUNTS,p=probability)))
                players[lo+start:lo+end,rng.choice(slots,count,replace=False)]=np.nan
            for start,end in _episodes(hi-lo,extra_share,rng):
                source=int(rng.choice(slots))
                offset=rng.normal(0,DUPLICATE_OFFSET_M,2).astype(np.float32)
                extra[lo+start:lo+end,team]=players[lo+start:lo+end,source]+offset
    players=np.concatenate([players,extra],axis=1); teams=np.r_[teams,[0,1]].astype(teams.dtype)
    return replace(obs,player_xy_m=players,player_observed=np.isfinite(players).all(axis=-1),
                   player_team=teams).validate()
