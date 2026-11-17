"""Ball-gap-aware features, version 4. Observation-only: no targets, identities or events.

Three additions on top of the v3 geometry block:

1. Short ball gaps (<= MAX_FILL_FRAMES) are linearly interpolated inside a sequence before
   the v3 features are computed, so team-to-ball distances survive brief occlusions.
2. Gap timing (frames since / until the ball was last / next available, gap length) and each
   team's nearest-player distance held from the frames bordering the gap. On a long gap the
   network still sees who was at the ball when it vanished and who is there when it returns.
3. Ball kinematics that can reveal a touch: direction change and velocity change across a
   +-KIN_LAG window.

The first 52 columns keep the v3 names but are computed on the gap-filled ball; the true
tracker flag is preserved separately as 'ball_raw_observed'.
"""
from dataclasses import replace
import numpy as np
from . import team_features_v3
from .features import sequence_spans

MAX_FILL_FRAMES=50
GAP_SCALE_FRAMES=250
KIN_LAG=5
MIN_DIRECTION_SPEED=0.5

SYMMETRIC=('ball_raw_observed','ball_filled','gap_since','gap_until','gap_length',
           'turn_cos','ball_accel','kinematics_valid')
TEAM=('prev_distance_1','next_distance_1','prev_valid','next_valid')
BASE=len(team_features_v3.NAMES)
NAMES=tuple(team_features_v3.NAMES)+SYMMETRIC+tuple(
    f'{side}_{name}' for side in ('home','away') for name in TEAM)


def fill_ball(obs,max_gap=MAX_FILL_FRAMES):
    """Linearly interpolate interior ball gaps of at most max_gap frames, per sequence."""
    ball=np.array(obs.ball_xy_m,dtype=np.float32)
    seen=np.array(obs.ball_observed,dtype=bool)
    filled=np.zeros(len(seen),bool)
    for lo,hi in sequence_spans(obs.sequence_id):
        idx=np.flatnonzero(seen[lo:hi])
        if len(idx)<2:
            continue
        pos=np.arange(idx[0],idx[-1]+1)
        right=np.searchsorted(idx,pos,side='left')
        following=idx[right]; previous=idx[np.maximum(right-1,0)]
        short=(following!=pos)&(following-previous-1<=max_gap)
        target=pos[short]
        for axis in (0,1):
            ball[lo+target,axis]=np.interp(target,idx,ball[lo+idx,axis])
        filled[lo+target]=True
    return ball,filled


def neighbours(have,sequence_id):
    """Index of the last/next frame with a ball in the same sequence, or -1."""
    n=len(have); frames=np.arange(n); seq=np.asarray(sequence_id)
    previous=np.maximum.accumulate(np.where(have,frames,-1))
    following=np.minimum.accumulate(np.where(have,frames,n)[::-1])[::-1]
    previous[(previous>=0)&(seq[np.maximum(previous,0)]!=seq)]=-1
    following[following==n]=-1
    following[(following>=0)&(seq[np.maximum(following,0)]!=seq)]=-1
    return previous,following


def kinematics(ball,have,sequence_id,lag=KIN_LAG):
    n=len(have); seq=np.asarray(sequence_id)
    turn=np.zeros(n); accel=np.zeros(n); valid=np.zeros(n,bool)
    if n<=2*lag:
        return turn,accel,valid
    mid=slice(lag,n-lag)
    ok=have[mid]&have[:-2*lag]&have[2*lag:]&(seq[:-2*lag]==seq[2*lag:])
    b=np.nan_to_num(ball.astype(np.float64),nan=0.)
    before=(b[mid]-b[:-2*lag])*25/lag
    after=(b[2*lag:]-b[mid])*25/lag
    s1=np.linalg.norm(before,axis=1); s2=np.linalg.norm(after,axis=1)
    moving=(s1>MIN_DIRECTION_SPEED)&(s2>MIN_DIRECTION_SPEED)
    cosine=np.ones(len(s1))
    cosine[moving]=np.clip((before[moving]*after[moving]).sum(axis=1)/(s1[moving]*s2[moving]),-1,1)
    turn[mid]=np.where(ok,cosine,0)
    accel[mid]=np.where(ok,np.minimum(np.linalg.norm(after-before,axis=1)/20,5),0)
    valid[mid]=ok
    return turn,accel,valid


def extract(obs):
    n=len(obs.frame_id)
    ball,filled=fill_ball(obs)
    have=np.isfinite(ball).all(axis=-1)
    work=replace(obs,ball_xy_m=ball,ball_observed=have)
    blocks=[team_features_v3.extract(work)]
    previous,following=neighbours(have,obs.sequence_id)
    frames=np.arange(n)
    since=np.where(previous>=0,np.minimum((frames-previous)/GAP_SCALE_FRAMES,1),1)
    until=np.where(following>=0,np.minimum((following-frames)/GAP_SCALE_FRAMES,1),1)
    length=np.where((previous>=0)&(following>=0),
                    np.minimum((following-previous-1)/GAP_SCALE_FRAMES,1),1)
    since[have]=0; until[have]=0; length[have]=0
    turn,accel,kin_valid=kinematics(ball,have,obs.sequence_id)
    blocks.append(np.column_stack([np.asarray(obs.ball_observed,dtype=float),filled,since,until,
                                   length,turn,accel,kin_valid]))
    for team in (0,1):
        slots=obs.player_team==team
        distance=np.linalg.norm(obs.player_xy_m[:,slots]-ball[:,None,:],axis=-1)
        distance=np.where(obs.player_observed[:,slots]&have[:,None],distance,np.inf)
        nearest=distance.min(axis=1) if distance.shape[1] else np.full(n,np.inf)
        held=[]
        for index in (previous,following):
            value=np.where(index>=0,nearest[np.maximum(index,0)],np.inf)
            ok=np.isfinite(value)
            held.append((np.where(ok,np.minimum(value/10,10),0),ok))
        blocks.append(np.column_stack([held[0][0],held[1][0],held[0][1],held[1][1]]))
    x=np.column_stack(blocks).astype(np.float32)
    if x.shape!=(n,len(NAMES)) or not np.isfinite(x).all():
        raise ValueError('Invalid v4 feature matrix')
    return x


def swap(x):
    """Exchange home/away channels; gap timing and kinematics are team-symmetric."""
    x=np.asarray(x)
    if x.shape[-1]!=len(NAMES):
        raise ValueError('Unexpected v4 feature count')
    start=BASE+len(SYMMETRIC); width=len(TEAM)
    return np.concatenate([team_features_v3.swap(x[...,:BASE]),x[...,BASE:start],
                           x[...,start+width:start+2*width],x[...,start:start+width]],axis=-1)
