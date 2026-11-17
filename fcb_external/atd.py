"""Two-pass EPTS raw-ATD adapter; no labels, video or ground truth inputs."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from .schema import Observations


def xy(group,length,width):
    try:
        parts=group.split(','); x,y=float(parts[0]),float(parts[1])
        if np.isfinite([x,y]).all():
            # Metrica normalized image-style axes -> centred Cartesian axes.
            return x*length-length/2,width/2-y*width
    except (ValueError,IndexError):
        pass
    return np.nan,np.nan


def convert(raw_path,metadata_path,match_id,home_team_id,away_team_id,length,width,period_blocks=None):
    if not home_team_id or not away_team_id or home_team_id==away_team_id:
        raise ValueError('Explicit distinct home/away IDs are required')
    root=ET.parse(metadata_path).getroot()
    slots=[p.get('teamId') for p in root.iter('Player')]
    if not slots or any(t not in (home_team_id,away_team_id) for t in slots):
        raise ValueError('Invalid ATD track team membership')
    keys=[]
    with Path(raw_path).open(encoding='utf-8-sig') as handle:
        for line in handle:
            parts=line.strip().split(':')
            if len(parts)!=3:
                raise ValueError('Malformed ATD row')
            frame_id=int(parts[0])
            if frame_id<1:
                raise ValueError('Invalid ATD frame ID')
            keys.append(frame_id)
    if len(set(keys))!=len(keys):
        raise ValueError('Invalid/duplicate ATD frame ID')
    keys.sort()
    if not keys:
        raise ValueError('Empty ATD')
    periods=np.ones(len(keys),dtype=np.int8)
    if period_blocks:
        for i,key in enumerate(keys):
            matches=[p for p,start,end in period_blocks if start<=key-1<end]
            if len(matches)!=1:
                raise ValueError('ATD row outside an unambiguous explicit period block')
            periods[i]=matches[0]
    starts={}
    times=np.zeros(len(keys),dtype=np.float64); seq=np.zeros(len(keys),dtype=np.int32)
    for i,key in enumerate(keys):
        starts.setdefault(int(periods[i]),key)
        times[i]=(key-starts[int(periods[i])])/25
        if i:
            seq[i]=seq[i-1]+int(periods[i]!=periods[i-1] or key!=keys[i-1]+1)
    # Store only frame->row lookup in Python; coordinates go straight to arrays.
    index={key:i for i,key in enumerate(keys)}
    players=np.full((len(keys),len(slots),2),np.nan,dtype=np.float32)
    ball=np.full((len(keys),2),np.nan,dtype=np.float32)
    with Path(raw_path).open(encoding='utf-8-sig') as handle:
        for line in handle:
            parts=line.strip().split(':')
            groups=parts[1].rstrip(';').split(';')
            if len(groups)!=len(slots):
                raise ValueError('ATD track-slot count mismatch')
            i=index[int(parts[0])]
            players[i]=[xy(group,length,width) for group in groups]
            ball[i]=xy(parts[2],length,width)
    return Observations(match_id,periods,np.array(keys,dtype=np.int64),times,players,
        np.isfinite(players).all(axis=-1),np.array([0 if t==home_team_id else 1 for t in slots],dtype=np.int8),
        ball,np.isfinite(ball).all(axis=-1),seq,float(length),float(width),25.).validate()
