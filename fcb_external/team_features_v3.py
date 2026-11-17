"""Versioned geometry available independently of the observed ball."""
import warnings
import numpy as np
from . import team_features

EXTRA=('centre_x','centre_y','spread','speed','geometry_valid','motion_valid')
NAMES=team_features.NAMES+tuple(f'{team}_{name}' for team in ('home','away') for name in EXTRA)

def extract(obs):
    blocks=[team_features.extract(obs)]
    n=len(obs.frame_id)
    for team in (0,1):
        slots=obs.player_team==team
        points=obs.player_xy_m[:,slots]
        valid=obs.player_observed[:,slots].sum(axis=1)>=2
        centre=np.zeros((n,2),np.float64); spread=np.zeros(n)
        if points.shape[1]:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                centre[valid]=np.nanmedian(points[valid],axis=1)
                spread[valid]=np.nanmedian(np.linalg.norm(points[valid]-centre[valid,None,:],axis=-1),axis=1)
        motion=np.zeros(n,bool);speed=np.zeros(n)
        dt=obs.time_s[5:]-obs.time_s[:-5]
        motion[5:]=valid[5:]&valid[:-5]&(obs.sequence_id[5:]==obs.sequence_id[:-5])&(dt>0)
        indices=np.flatnonzero(motion)
        speed[indices]=np.linalg.norm(centre[indices]-centre[indices-5],axis=1)/(obs.time_s[indices]-obs.time_s[indices-5])
        blocks.append(np.column_stack([centre[:,0]/obs.pitch_length_m,centre[:,1]/obs.pitch_width_m,
            spread/np.hypot(obs.pitch_length_m,obs.pitch_width_m),speed/10,valid,motion]))
    return np.column_stack(blocks).astype(np.float32)

def swap(x):
    x=np.asarray(x)
    if x.shape[-1]!=len(NAMES): raise ValueError('Unexpected v3 feature count')
    return np.concatenate([team_features.swap(x[...,:40]),x[...,46:52],x[...,40:46]],axis=-1)
