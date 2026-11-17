"""Reference nearest-player baseline, with H/A-only smoothing per sequence.

Smoothing semantics reproduce metrica-sports/more-than-a-hack-2027 (MIT).
"""
import numpy as np

VARIANTS = {'baseline_raw':0,'baseline_smooth25':25,'baseline_smooth50':50,
            'baseline_smooth125':125}


def nearest_codes(obs):
    obs.validate()
    out = np.full(len(obs.frame_id),-1,dtype=np.int8)
    # Limit temporary distance matrices. Stable slot order determines equal-distance ties.
    for lo in range(0,len(out),4096):
        hi = min(lo+4096,len(out))
        delta = obs.player_xy_m[lo:hi].astype(np.float64)-obs.ball_xy_m[lo:hi,None,:]
        distance = np.sum(delta*delta,axis=-1)
        usable = obs.player_observed[lo:hi] & obs.ball_observed[lo:hi,None]
        distance[~usable] = np.inf
        nearest = np.argmin(distance,axis=1)
        valid = usable.any(axis=1)
        out[lo:hi] = np.where(valid,obs.player_team[nearest],-1)
    return out


def smooth_codes(codes, half_width):
    if half_width < 0:
        raise ValueError('Half-width cannot be negative')
    if half_width == 0:
        return np.where(codes==0,'H',np.where(codes==1,'A','D'))
    h = np.concatenate(([0],np.cumsum(codes==0)))
    a = np.concatenate(([0],np.cumsum(codes==1)))
    i = np.arange(len(codes))
    lo,hi = np.maximum(0,i-half_width),np.minimum(len(codes),i+half_width+1)
    home,away = h[hi]-h[lo],a[hi]-a[lo]
    return np.where(home+away>0,np.where(home>=away,'H','A'),'D')


def sequence_smooth(codes, sequence_id, half_width):
    out = np.full(len(codes),'D',dtype='<U1')
    edges = np.r_[0,np.flatnonzero(np.diff(sequence_id))+1,len(codes)]
    for lo,hi in zip(edges[:-1],edges[1:]):
        out[lo:hi] = smooth_codes(codes[lo:hi],half_width)
    return out


def predict(obs, half_width):
    return sequence_smooth(nearest_codes(obs),obs.sequence_id,half_width)


def predict_all(obs):
    codes = nearest_codes(obs)
    return {name:sequence_smooth(codes,obs.sequence_id,width) for name,width in VARIANTS.items()}
