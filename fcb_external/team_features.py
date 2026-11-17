"""Team-aware geometric features with no player identities or target fields."""
import numpy as np
from .features import extract as symmetric_extract, NAMES as BASE_NAMES

TEAM_NAMES=('distance_1','distance_2','distance_3','distance_1_valid','distance_2_valid',
            'distance_3_valid','within_2m','within_5m','within_10m','observed_players')
NAMES=tuple(BASE_NAMES)+tuple(f'{side}_{name}' for side in ('home','away') for name in TEAM_NAMES)


def extract(obs):
    data=[symmetric_extract(obs)]
    for team in (0,1):
        slots=obs.player_team==team
        points=obs.player_xy_m[:,slots]
        distance=np.linalg.norm(points-obs.ball_xy_m[:,None,:],axis=-1)
        usable=obs.player_observed[:,slots]&obs.ball_observed[:,None]
        distance=np.where(usable,distance,np.inf)
        nearest=np.full((len(distance),3),np.inf)
        k=min(3,distance.shape[1])
        if k:
            nearest[:,:k]=np.sort(distance,axis=1)[:,:k]
        valid=np.isfinite(nearest)
        scaled=np.where(valid,np.minimum(nearest/10,10),0)
        counts=np.column_stack([(distance<r).sum(axis=1)/4 for r in (2,5,10)])
        observed=obs.player_observed[:,slots].sum(axis=1,keepdims=True)/11
        data.append(np.column_stack([scaled,valid.astype(float),counts,observed]))
    return np.column_stack(data).astype(np.float32)


def swap(x):
    """Exchange team channels; the first 20 identity-free features are symmetric."""
    if x.shape[-1]!=len(NAMES):
        raise ValueError('Unexpected temporal feature count')
    base=len(BASE_NAMES); width=len(TEAM_NAMES)
    order=list(range(base))+list(range(base+width,base+2*width))+list(range(base,base+width))
    return np.asarray(x)[...,order]
