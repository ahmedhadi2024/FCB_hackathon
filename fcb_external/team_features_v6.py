"""Version 6 features: v4 plus player pace, the dead-ball cue that survives a missing ball.

During a stoppage players walk; in live play they run. Pace is measured per track slot as the
displacement over a centred one-second window, so no player identity is needed: a slot that
vanishes or jumps (an identity swap) is simply left out of that frame's statistics.

pace_mask() hides the ball in contiguous gaps that start preferentially while players are
slow. It conditions only on observed player motion, never on targets, and mimics automatic
tracking, which loses the ball far more often out of play.
"""
from dataclasses import replace
import numpy as np
from . import team_features_v4
from .features import sequence_spans, window_mean

HALF=12                 # centred window of 2*HALF frames, about one second at 25 fps
MAX_SPEED=12.0          # m/s; faster apparent motion is a track jump, not a player
WALKING=2.0             # m/s
SLOW_MEDIAN=1.5         # m/s; pooled median pace below this counts as a slow frame
SYMMETRIC=('pace_median','pace_p75','pace_walking_fraction','pace_valid','pace_median_1s','pace_median_3s')
TEAM=('pace_median','pace_valid')
BASE=len(team_features_v4.NAMES)
NAMES=tuple(team_features_v4.NAMES)+SYMMETRIC+tuple(
    f'{side}_{name}' for side in ('home','away') for name in TEAM)


def slot_speeds(obs):
    """(frames, slots) speed in m/s with NaN where a slot has no trustworthy motion."""
    xy=np.asarray(obs.player_xy_m,dtype=np.float64); seen=np.asarray(obs.player_observed)
    n=len(xy); speed=np.full(xy.shape[:2],np.nan)
    if n<=2*HALF:
        return speed
    seq=np.asarray(obs.sequence_id)
    same=(seq[2*HALF:]==seq[:-2*HALF])[:,None]
    ok=seen[2*HALF:]&seen[:-2*HALF]&same
    with np.errstate(invalid='ignore'):
        value=np.linalg.norm(xy[2*HALF:]-xy[:-2*HALF],axis=-1)*25/(2*HALF)
    value=np.where(ok&(value<=MAX_SPEED),value,np.nan)
    speed[HALF:n-HALF]=value
    return speed


def _summary(speed):
    count=np.isfinite(speed).sum(axis=1); valid=count>=2
    filled=np.where(np.isfinite(speed),speed,np.inf)
    ordered=np.sort(filled,axis=1)
    rows=np.arange(len(speed)); safe=np.maximum(count,1)
    median=np.where(valid,ordered[rows,(safe-1)//2],0.)
    upper=np.where(valid,ordered[rows,np.minimum(safe-1,(3*safe)//4)],0.)
    walking=np.where(valid,(filled<WALKING).sum(axis=1)/safe,0.)
    return median,upper,walking,valid


def pooled_pace(obs):
    """Median player pace per frame and its validity; used for features and for masking."""
    median,_,_,valid=_summary(slot_speeds(obs))
    return median,valid


def extract(obs):
    speed=slot_speeds(obs)
    median,upper,walking,valid=_summary(speed)
    seq=np.asarray(obs.sequence_id)
    blocks=[team_features_v4.extract(obs),
            np.column_stack([median/5,upper/5,walking,valid,
                             window_mean(median/5,seq,12),window_mean(median/5,seq,37)])]
    for team in (0,1):
        slots=np.asarray(obs.player_team)==team
        if slots.any():
            team_median,_,_,team_valid=_summary(speed[:,slots])
        else:
            team_median=np.zeros(len(speed)); team_valid=np.zeros(len(speed),bool)
        blocks.append(np.column_stack([team_median/5,team_valid]))
    x=np.column_stack(blocks).astype(np.float32)
    if x.shape!=(len(seq),len(NAMES)) or not np.isfinite(x).all():
        raise ValueError('Invalid v6 feature matrix')
    return x


def swap(x):
    x=np.asarray(x)
    if x.shape[-1]!=len(NAMES):
        raise ValueError('Unexpected v6 feature count')
    start=BASE+len(SYMMETRIC); width=len(TEAM)
    return np.concatenate([team_features_v4.swap(x[...,:BASE]),x[...,BASE:start],
                           x[...,start+width:start+2*width],x[...,start:start+width]],axis=-1)


def pace_mask(obs,fraction,slow_weight,gap_min,gap_max,seed):
    """Hide the ball on about `fraction` of frames, in gaps that favour slow-player moments."""
    if not 0<fraction<=1 or slow_weight<0 or not 1<=gap_min<=gap_max:
        raise ValueError('Invalid pace-mask parameters')
    rng=np.random.default_rng(seed)
    median,valid=pooled_pace(obs)
    weight=1+slow_weight*(valid&(median<SLOW_MEDIAN))
    ball=np.array(obs.ball_xy_m,dtype=np.float32)
    for lo,hi in sequence_spans(obs.sequence_id):
        n=hi-lo; target=int(np.ceil(n*fraction)); mask=np.zeros(n,bool)
        while mask.sum()<target:
            start_weights=weight[lo:hi]*(~mask)
            start=int(rng.choice(n,p=start_weights/start_weights.sum()))
            end=min(n,start+int(rng.integers(gap_min,gap_max+1)))
            fresh=np.flatnonzero(~mask[start:end])+start
            mask[fresh[:target-int(mask.sum())]]=True
        ball[lo:hi][mask]=np.nan
    return replace(obs,ball_xy_m=ball,ball_observed=np.isfinite(ball).all(axis=-1)).validate()
