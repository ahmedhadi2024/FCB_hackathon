"""Seeded observation-only stress corruption; no possession targets accepted."""
from dataclasses import replace
import numpy as np
from .features import corrupt as legacy_corrupt, sequence_spans


def corrupt(obs,scenario):
    fraction=scenario.get('missing_fraction',0.)
    duplicate=scenario.get('duplicate_fraction',0.)
    if not 0<=fraction<=1 or not 0<=duplicate<=1:
        raise ValueError('Corruption fractions must be in [0,1]')
    rng=np.random.default_rng(scenario['seed'])
    players=obs.player_xy_m.copy(); teams=obs.player_team.copy(); ball=obs.ball_xy_m.copy()
    if fraction:
        lower=scenario.get('gap_min_frames',25); upper=scenario.get('gap_max_frames',375)
        if not isinstance(lower,int) or not isinstance(upper,int) or not 1<=lower<=upper:
            raise ValueError('Positive integer gap bounds required')
        weight=np.ones(len(ball),float)
        if scenario.get('crowd_weight',0):
            distance=np.linalg.norm(players-ball[:,None,:],axis=-1)
            counts=np.sum(distance<scenario['crowd_radius_m'],axis=1)
            weight+=scenario['crowd_weight']*(counts>=scenario['crowd_min_players'])
        for lo,hi in sequence_spans(obs.sequence_id):
            n=hi-lo; target=int(np.ceil(n*fraction)); mask=np.zeros(n,bool)
            # Sample only not-yet-masked starts: guaranteed progress, bounded by n iterations.
            while mask.sum()<target:
                start_weights=weight[lo:hi]*(~mask)
                start=int(rng.choice(n,p=start_weights/start_weights.sum()))
                end=min(n,start+int(rng.integers(lower,upper+1)))
                fresh=np.flatnonzero(~mask[start:end])+start
                mask[fresh[:target-int(mask.sum())]]=True
            ball[lo:hi][mask]=np.nan
    if duplicate:
        selected=[]
        for team in (0,1):
            slots=np.flatnonzero(teams==team)
            if len(slots):
                selected.extend(rng.choice(slots,int(np.ceil(len(slots)*duplicate)),replace=False).tolist())
        players=np.concatenate([players,players[:,selected]],axis=1)
        teams=np.r_[teams,teams[selected]]
    changed=replace(obs,player_xy_m=players,player_observed=np.isfinite(players).all(axis=-1),
                    player_team=teams,ball_xy_m=ball,ball_observed=np.isfinite(ball).all(axis=-1)).validate()
    return legacy_corrupt(changed,scenario)
