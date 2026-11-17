"""SkillCorner open data (github.com/SkillCorner/opendata, MIT): broadcast-derived tracking.

Twenty A-League 2024/25 matches of real automatic tracking: players are flagged detected or
extrapolated and the ball detected or not, so the faults are natural, not simulated. These are
other games, so usable as external data.

Conversion to the common observation contract:
  - 10 fps is resampled to 25 fps by linear interpolation between neighbouring samples; a
    value is produced only where both neighbours are observed.
  - natural_ball=True treats an undetected ball as missing (the provider's own ball faults).
  - natural_players=True also treats extrapolated players as missing. The default keeps the
    extrapolated positions: a wide tactical camera, unlike a broadcast camera, sees nearly
    every player, so the default is the closer match to tactical tracking. Either way, on
    frames with no tracking at all (replays, close-ups; about 40% of frames) everything is
    missing, which tactical tracking does not do.
  - labels come from the provider's phases-of-play file, which exists only while the ball is
    in play and names the team in possession: inside a phase H or A, outside every phase D.
    This dead-ball truth is the provider's in-play segmentation, not a referee signal.
"""
import csv
import json
from pathlib import Path
import numpy as np
from .schema import Observations

BASE='https://raw.githubusercontent.com/SkillCorner/opendata/master/data'
MEDIA='https://media.githubusercontent.com/media/SkillCorner/opendata/master/data'


def files(match_id):
    return {f'{match_id}_match.json':f'{BASE}/matches/{match_id}/{match_id}_match.json',
            f'{match_id}_phases_of_play.csv':f'{BASE}/matches/{match_id}/{match_id}_phases_of_play.csv',
            f'{match_id}_tracking_extrapolated.jsonl':f'{MEDIA}/matches/{match_id}/{match_id}_tracking_extrapolated.jsonl'}


def _resample(values,ok,positions):
    """Linear interpolation of 10 Hz samples at fractional sample positions; NaN unless both ends are observed."""
    lower=np.floor(positions).astype(int); upper=np.minimum(lower+1,len(values)-1); weight=(positions-lower)
    shape=(len(positions),)+(1,)*(values.ndim-1)
    out=values[lower]*(1-weight.reshape(shape))+values[upper]*weight.reshape(shape)
    good=ok[lower]&ok[upper]
    out[~good]=np.nan
    return out.astype(np.float32)


def load_match(folder,match_id,natural_ball=True,natural_players=False):
    """Return (Observations at 25 fps, labels H/A/D) for one SkillCorner match."""
    folder=Path(folder)
    meta=json.loads((folder/f'{match_id}_match.json').read_text(encoding='utf-8'))
    home=meta['home_team']['id']; away=meta['away_team']['id']
    team_of={p['id']:(0 if p['team_id']==home else 1) for p in meta['players'] if p['team_id'] in (home,away)}
    ids=sorted(team_of,key=lambda i:(team_of[i],i)); slot={pid:i for i,pid in enumerate(ids)}
    periods={p['period']:(p['start_frame'],p['end_frame']) for p in meta['match_periods'] if p['period'] in (1,2)}
    span={p:(e-s+1) for p,(s,e) in periods.items()}
    players={p:np.full((n,len(ids),2),np.nan,np.float32) for p,n in span.items()}
    ball={p:np.full((n,2),np.nan,np.float32) for p,n in span.items()}
    with (folder/f'{match_id}_tracking_extrapolated.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            row=json.loads(line); period=row.get('period')
            if period not in periods:
                continue
            start,end=periods[period]
            if not start<=row['frame']<=end:
                continue
            i=row['frame']-start
            b=row['ball_data']
            if b['x'] is not None and (b['is_detected'] or not natural_ball):
                ball[period][i]=(b['x'],b['y'])
            for p in row['player_data']:
                j=slot.get(p['player_id'])
                if j is not None and (p['is_detected'] or not natural_players):
                    players[period][i,j]=(p['x'],p['y'])
    owner={p:np.full(n,-1,np.int8) for p,n in span.items()}
    with (folder/f'{match_id}_phases_of_play.csv').open(newline='',encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            period=int(row['period'])
            if period not in periods:
                continue
            start=periods[period][0]; a=max(0,int(row['frame_start'])-start); b=min(span[period],int(row['frame_end'])-start)
            team=int(float(row['team_in_possession_id']))
            if team in (home,away) and b>a:
                owner[period][a:b]=0 if team==home else 1
    xy=[];bxy=[];label=[];period_id=[];time=[];sequence=[]
    for index,period in enumerate(sorted(periods)):
        n=span[period]; m=int((n-1)*2.5)+1; positions=np.arange(m)/2.5
        p=players[period]; b=ball[period]
        xy.append(_resample(np.nan_to_num(p),np.isfinite(p).all(-1)[...,None].repeat(2,-1),positions))
        bxy.append(_resample(np.nan_to_num(b),np.isfinite(b).all(-1)[:,None].repeat(2,-1),positions))
        nearest=owner[period][np.minimum(np.rint(positions).astype(int),n-1)]
        label.append(np.where(nearest==0,'H',np.where(nearest==1,'A','D')))
        period_id.append(np.full(m,period,np.int8)); time.append(np.arange(m)/25.); sequence.append(np.full(m,index,np.int32))
    xy=np.concatenate(xy); bxy=np.concatenate(bxy)
    xy[~np.isfinite(xy).all(-1)]=np.nan; bxy[~np.isfinite(bxy).all(-1)]=np.nan
    n=len(xy)
    obs=Observations(f'SC{match_id}',np.concatenate(period_id),np.arange(1,n+1),np.concatenate(time),xy,
        np.isfinite(xy).all(-1),np.array([team_of[i] for i in ids],np.int8),bxy,np.isfinite(bxy).all(-1),
        np.concatenate(sequence),float(meta['pitch_length']),float(meta['pitch_width']),25.).validate()
    return obs,np.concatenate(label)
