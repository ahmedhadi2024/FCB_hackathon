"""Metrica Sports public sample matches (github.com/metrica-sports/sample-data).

Three anonymised matches of the organiser's own corrected tracking with synchronised event
data: other games, so usable as external data. There is no possession field, so labels are
derived mechanically:

  dead  : the corrected ball position is missing (the provider blanks it out of play)
  owner : the team of the most recent on-ball event (pass, carry, shot, recovery, set piece,
          ball lost), carried forward; a loose ball therefore stays with the last team.

These labels follow the provider's event definitions, not the competition ground truth.
"""
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from . import atd
from .schema import Observations

OWNER_EVENTS=('PASS','CARRY','SHOT','RECOVERY','SET PIECE','BALL LOST')
PITCH=(105.,68.)


def _xy(x,y):
    return x*PITCH[0]-PITCH[0]/2,PITCH[1]/2-y*PITCH[1]


def _read_tracking_csv(path):
    with Path(path).open(newline='') as handle:
        rows=list(csv.reader(handle))[3:]
    data=np.array([[float(v) if v not in ('','NaN') else np.nan for v in row] for row in rows])
    return data[:,0].astype(np.int8),data[:,1].astype(np.int64),data[:,3:-2],data[:,-2:]


def load_csv_game(folder,game):
    """Sample Game 1 or 2: (Observations, events) with events as (team, type, period, start_frame)."""
    folder=Path(folder); name=f'Sample_Game_{game}'
    period,frame,home,ball=_read_tracking_csv(folder/f'{name}_RawTrackingData_Home_Team.csv')
    period_b,frame_b,away,_=_read_tracking_csv(folder/f'{name}_RawTrackingData_Away_Team.csv')
    if not np.array_equal(frame,frame_b) or not np.array_equal(period,period_b):
        raise ValueError('Home and away tracking timelines differ')
    raw=np.concatenate([home,away],axis=1).reshape(len(frame),-1,2)
    players=np.stack(_xy(raw[...,0],raw[...,1]),axis=-1).astype(np.float32)
    ball_xy=np.stack(_xy(ball[:,0],ball[:,1]),axis=-1).astype(np.float32)
    teams=np.r_[np.zeros(home.shape[1]//2,np.int8),np.ones(away.shape[1]//2,np.int8)]
    sequence=np.r_[0,np.cumsum((np.diff(frame)!=1)|(np.diff(period)!=0))].astype(np.int32)
    obs=Observations(name,period,frame,frame/25.,players,np.isfinite(players).all(-1),teams,
                     ball_xy,np.isfinite(ball_xy).all(-1),sequence,PITCH[0],PITCH[1],25.).validate()
    events=[]
    with (folder/f'{name}_RawEventsData.csv').open(newline='') as handle:
        for row in csv.DictReader(handle):
            events.append(({'Home':0,'Away':1}[row['Team']],row['Type'],int(row['Period']),int(row['Start Frame'])))
    return obs,events


def load_epts_game(folder):
    """Sample Game 3 (EPTS tracking + JSON events).

    The EPTS metadata lists a different set of on-pitch players for each frame range
    (substitutions), so every row is mapped through its range's channel list to one fixed
    slot per player.
    """
    folder=Path(folder); name='Sample_Game_3'
    root=ET.parse(folder/f'{name}_metadata.xml').getroot()
    values={p.findtext('Name'):p.findtext('Value') for p in root.iter('ProviderParameter')}
    halves=[(1,int(values['first_half_start']),int(values['first_half_end'])),
            (2,int(values['second_half_start']),int(values['second_half_end']))]
    score=next(root.iter('Score'))
    ids=(score.get('idLocalTeam'),score.get('idVisitingTeam'))
    team_of={p.get('id'):(0 if p.get('teamId')==ids[0] else 1) for p in root.iter('Player') if p.get('teamId') in ids}
    order=sorted(team_of,key=lambda pid:(team_of[pid],pid)); slot={pid:i for i,pid in enumerate(order)}
    channel={c.get('id'):c.get('playerId') for c in root.iter('PlayerChannel') if c.get('channelId')=='x'}
    specs=[]
    for spec in root.iter('DataFormatSpecification'):
        refs=[r.get('playerChannelId') for r in spec.iter('PlayerChannelRef') if r.get('playerChannelId').endswith('_x')]
        specs.append((int(spec.get('startFrame')),int(spec.get('endFrame')),[slot[channel[r]] for r in refs]))
    specs.sort()
    last=halves[-1][2]
    players=np.full((last,len(order),2),np.nan,np.float32); ball=np.full((last,2),np.nan,np.float32)
    current=0
    with (folder/f'{name}_tracking.txt').open(encoding='utf-8-sig') as handle:
        for line in handle:
            parts=line.strip().split(':')
            frame=int(parts[0])
            if frame<1 or frame>last:
                continue
            while specs[current][1]<frame:
                current+=1
            groups=parts[1].rstrip(';').split(';'); slots=specs[current][2]
            if len(groups)!=len(slots):
                raise ValueError('EPTS row does not match its format specification')
            for group,j in zip(groups,slots):
                players[frame-1,j]=atd.xy(group,PITCH[0],PITCH[1])
            ball[frame-1]=atd.xy(parts[2],PITCH[0],PITCH[1])
    frame_id=np.arange(1,last+1); period=np.zeros(last,np.int8)
    for number,start,end in halves:
        period[start-1:end]=number
    keep=period>0
    frame_id=frame_id[keep]; period=period[keep]; players=players[keep]; ball=ball[keep]
    sequence=np.r_[0,np.cumsum((np.diff(frame_id)!=1)|(np.diff(period)!=0))].astype(np.int32)
    obs=Observations(name,period,frame_id,frame_id/25.,players,np.isfinite(players).all(-1),
                     np.array([team_of[pid] for pid in order],np.int8),ball,np.isfinite(ball).all(-1),
                     sequence,PITCH[0],PITCH[1],25.).validate()
    events=[]
    for item in json.loads((folder/f'{name}_events.json').read_text())['data']:
        events.append(({ids[0]:0,ids[1]:1}[item['team']['id']],item['type']['name'],
                       int(item['period']),int(item['start']['frame'])))
    return obs,events


def event_labels(obs,events):
    """Per-frame H/A/D labels derived from on-ball events and ball availability."""
    n=len(obs.frame_id); owner=np.full(n,-1,np.int8)
    for period in np.unique(obs.period):
        rows=np.flatnonzero(obs.period==period)
        marks=sorted((start,team) for team,kind,p,start in events if p==period and kind in OWNER_EVENTS)
        if not marks:
            continue
        starts=np.array([m[0] for m in marks]); teams=np.array([m[1] for m in marks],np.int8)
        index=np.searchsorted(starts,obs.frame_id[rows],side='right')-1
        owner[rows]=teams[np.maximum(index,0)]   # frames before the first event take its team
    label=np.where(owner==0,'H',np.where(owner==1,'A','X'))
    label[~obs.ball_observed]='D'
    return label


def load_all(folder,include_game_3=False):
    """{match name: (Observations, labels)} for every sample match present in folder."""
    folder=Path(folder); out={}
    for game in (1,2):
        if (folder/f'Sample_Game_{game}_RawEventsData.csv').exists():
            obs,events=load_csv_game(folder,game); out[obs.match_id]=(obs,event_labels(obs,events))
    # Sample Game 3 is loaded only on request so earlier two-game experiments stay reproducible.
    if include_game_3 and (folder/'Sample_Game_3_events.json').exists():
        obs,events=load_epts_game(folder); out[obs.match_id]=(obs,event_labels(obs,events))
    return out
