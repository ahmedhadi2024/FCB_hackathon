"""Observation-only, identity-free features and deterministic external robustness tests."""
from dataclasses import replace
import numpy as np

NAMES=('ball_observed','players_present_per_22','min_team_players_per_11','max_team_players_per_11',
       'ball_abs_x','ball_abs_y','distance_to_boundary','nearest_team_distance','farther_team_distance',
       'players_within_5m_per_4','speed_1','speed_1_valid','speed_5','speed_5_valid',
       'speed_25','speed_25_valid','player_centroid_speed','ball_seen_1s','ball_seen_5s','ball_speed_mean_1s')


def sequence_spans(sequence_id):
    edges=np.r_[0,np.flatnonzero(np.diff(sequence_id))+1,len(sequence_id)]
    return zip(edges[:-1],edges[1:])


def window_mean(values,sequence_id,half_width):
    result=np.empty(len(values),dtype=np.float64)
    for lo,hi in sequence_spans(sequence_id):
        v=np.asarray(values[lo:hi],dtype=np.float64)
        c=np.r_[0,np.cumsum(v)]
        i=np.arange(len(v)); left=np.maximum(0,i-half_width); right=np.minimum(len(v),i+half_width+1)
        result[lo:hi]=(c[right]-c[left])/(right-left)
    return result


def speed(points,mask,seq,lag):
    result=np.zeros(len(points)); valid=np.zeros(len(points),dtype=bool)
    for lo,hi in sequence_spans(seq):
        if hi-lo<=lag:
            continue
        okay=mask[lo+lag:hi]&mask[lo:hi-lag]
        delta=points[lo+lag:hi]-points[lo:hi-lag]
        value=np.linalg.norm(delta,axis=1)*25/lag
        result[lo+lag:hi]=np.where(okay,np.minimum(value,100),0)
        valid[lo+lag:hi]=okay
    return result,valid


def extract(obs):
    obs.validate()
    n=len(obs.frame_id)
    counts=obs.player_observed.sum(axis=1)
    h=obs.player_observed[:,obs.player_team==0].sum(axis=1)
    a=obs.player_observed[:,obs.player_team==1].sum(axis=1)
    ball=np.nan_to_num(obs.ball_xy_m,nan=0.).astype(np.float64)
    distance=np.linalg.norm(obs.player_xy_m-ball[:,None,:],axis=-1)
    usable=obs.player_observed&obs.ball_observed[:,None]
    distance[~usable]=np.inf
    team_dist=[]
    for team in (0,1):
        block=distance[:,obs.player_team==team]
        team_dist.append(block.min(axis=1) if block.shape[1] else np.full(n,np.inf))
    nearest=np.minimum(*team_dist); farther=np.maximum(*team_dist)
    nearest=np.where(np.isfinite(nearest),np.minimum(nearest/10,10),0)
    farther=np.where(np.isfinite(farther),np.minimum(farther/10,10),0)
    bx=np.where(obs.ball_observed,np.abs(ball[:,0])/(obs.pitch_length_m/2),0)
    by=np.where(obs.ball_observed,np.abs(ball[:,1])/(obs.pitch_width_m/2),0)
    boundary=np.where(obs.ball_observed,np.minimum(obs.pitch_length_m/2-np.abs(ball[:,0]),
                    obs.pitch_width_m/2-np.abs(ball[:,1]))/10,0)
    data=[obs.ball_observed.astype(float),np.minimum(counts/22,2),np.minimum(h,a)/11,
          np.maximum(h,a)/11,bx,by,boundary,nearest,farther,(distance<5).sum(axis=1)/4]
    speeds=[]
    for lag in (1,5,25):
        value,valid=speed(obs.ball_xy_m,obs.ball_observed,obs.sequence_id,lag)
        data.extend([value/20,valid.astype(float)]); speeds.append(value)
    centroid=np.nansum(obs.player_xy_m,axis=1)/np.maximum(counts[:,None],1)
    player_speed,_=speed(centroid,counts>0,obs.sequence_id,5)
    data.extend([player_speed/10,
        window_mean(obs.ball_observed,obs.sequence_id,25),
        window_mean(obs.ball_observed,obs.sequence_id,125),
        window_mean(speeds[0]/20,obs.sequence_id,12)])
    x=np.column_stack(data).astype(np.float64)
    if x.shape!=(n,len(NAMES)) or not np.isfinite(x).all():
        raise ValueError('Invalid feature matrix')
    return x


def corrupt(obs,scenario):
    rng=np.random.default_rng(scenario['seed'])
    players=obs.player_xy_m.copy(); ball=obs.ball_xy_m.copy()
    length=scenario.get('ball_gap_frames',0); every=scenario.get('ball_gap_period_frames',0)
    if length and every:
        for lo,hi in sequence_spans(obs.sequence_id):
            offset=int(rng.integers(0,every))
            for start in range(lo+offset,hi,every):
                ball[start:min(start+length,hi)]=np.nan
    drop=scenario.get('player_drop_probability',0)
    if drop:
        players[rng.random(players.shape[:2])<drop]=np.nan
    noise=scenario.get('coordinate_noise_std_m',0)
    if noise:
        players+=rng.normal(0,noise,players.shape)
        ball+=rng.normal(0,noise,ball.shape)
    if scenario.get('permute_slots_within_team'):
        for lo,hi in sequence_spans(obs.sequence_id):
            for start in range(lo,hi,25):
                end=min(start+25,hi)
                for team in (0,1):
                    slots=np.flatnonzero(obs.player_team==team)
                    players[start:end,slots]=players[start:end,rng.permutation(slots)]
    return replace(obs,player_xy_m=players,player_observed=np.isfinite(players).all(axis=-1),
                   ball_xy_m=ball,ball_observed=np.isfinite(ball).all(axis=-1)).validate()
