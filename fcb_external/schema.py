"""Canonical observations contain no target fields; targets are persisted separately."""
from dataclasses import dataclass, fields
import hashlib
import json
from pathlib import Path
import numpy as np

VERSION = 1


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n',
                    encoding='utf-8')


@dataclass(frozen=True)
class Observations:
    match_id: str
    period: np.ndarray
    frame_id: np.ndarray
    time_s: np.ndarray
    player_xy_m: np.ndarray
    player_observed: np.ndarray
    player_team: np.ndarray
    ball_xy_m: np.ndarray
    ball_observed: np.ndarray
    sequence_id: np.ndarray
    pitch_length_m: float
    pitch_width_m: float
    fps: float

    @staticmethod
    def array_names():
        return ('period','frame_id','time_s','player_xy_m','player_observed',
                'player_team','ball_xy_m','ball_observed','sequence_id')

    def validate(self):
        n, p = len(self.frame_id), len(self.player_team)
        if not n or not p:
            raise ValueError('Empty observation timeline or player roster')
        shapes = {'period':(n,), 'frame_id':(n,), 'time_s':(n,),
                  'player_xy_m':(n,p,2), 'player_observed':(n,p), 'player_team':(p,),
                  'ball_xy_m':(n,2), 'ball_observed':(n,), 'sequence_id':(n,)}
        for name, shape in shapes.items():
            if getattr(self,name).shape != shape:
                raise ValueError(f'{name}: incorrect shape')
        if self.fps != 25 or not all(np.isfinite(v) and v > 0 for v in (
                self.pitch_length_m, self.pitch_width_m)):
            raise ValueError('Requires 25 Hz tracking and positive finite pitch dimensions')
        if not np.isin(self.player_team,[0,1]).all():
            raise ValueError('Team codes must be 0=home, 1=away')
        if not np.isfinite(self.time_s).all():
            raise ValueError('Nonfinite timestamps')
        for points, mask in ((self.player_xy_m,self.player_observed),
                             (self.ball_xy_m,self.ball_observed)):
            if mask.dtype != np.bool_ or not np.array_equal(np.isfinite(points).all(axis=-1),mask):
                raise ValueError('Observation masks disagree with finite coordinate pairs')
            if not np.isnan(points[~mask]).all():
                raise ValueError('Missing coordinates must be NaN pairs')
        if self.sequence_id[0] != 0 or not np.isin(np.diff(self.sequence_id),[0,1]).all():
            raise ValueError('Sequence IDs must be consecutive, nondecreasing and start at 0')
        same_period = self.period[1:] == self.period[:-1]
        dn = np.diff(self.frame_id)
        dt = np.diff(self.time_s)
        bad_grid=(dn<=0)|~np.isclose(dt,dn/25,rtol=0,atol=1e-6)
        if np.any(same_period&bad_grid):
            raise ValueError('Frame IDs/timestamps are not a consistent 25 Hz grid')
        same_sequence=np.diff(self.sequence_id)==0
        if np.any(same_sequence&(~same_period|(dn!=1))):
            raise ValueError('Sequence crosses a period or missing-frame gap')
        return self

    def timeline_hash(self):
        digest = hashlib.sha256(self.match_id.encode())
        for name in ('period','frame_id','time_s','sequence_id'):
            digest.update(np.ascontiguousarray(getattr(self,name)).tobytes())
        return digest.hexdigest()

    def save(self, directory):
        self.validate()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name in self.array_names():
            np.save(directory / (name+'.npy'), getattr(self,name), allow_pickle=False)
        scalars = {f.name:getattr(self,f.name) for f in fields(self) if f.name not in self.array_names()}
        write_json(directory/'metadata.json', {'schema_version':VERSION, **scalars,
                    'timeline_sha256':self.timeline_hash()})

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        meta = json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
        if meta.pop('schema_version') != VERSION:
            raise ValueError('Unsupported observation schema version')
        expected = meta.pop('timeline_sha256')
        obj = cls(**meta, **{name:np.load(directory/(name+'.npy'), mmap_mode='r',
                       allow_pickle=False) for name in cls.array_names()}).validate()
        if obj.timeline_hash() != expected:
            raise ValueError('Observation timeline checksum mismatch')
        return obj


@dataclass(frozen=True)
class Targets:
    label: np.ndarray
    valid: np.ndarray

    def validate(self, n):
        if self.label.shape != (n,) or self.valid.shape != (n,) or self.valid.dtype != np.bool_:
            raise ValueError('Target shape or validity dtype mismatch')
        if not np.isin(self.label,['H','A','D','X']).all():
            raise ValueError('Unsupported target label')
        if not np.array_equal(self.valid,self.label!='X'):
            raise ValueError('Target validity disagrees with labels')
        return self

    def save(self, directory, obs):
        self.validate(len(obs.frame_id))
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory/'label.npy',self.label,allow_pickle=False)
        np.save(directory/'valid.npy',self.valid,allow_pickle=False)
        write_json(directory/'metadata.json',{'schema_version':VERSION,
                    'timeline_sha256':obs.timeline_hash()})

    @classmethod
    def load(cls, directory, obs):
        directory = Path(directory)
        meta = json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
        if meta != {'schema_version':VERSION,'timeline_sha256':obs.timeline_hash()}:
            raise ValueError('Targets do not belong to this observation timeline')
        return cls(np.load(directory/'label.npy',allow_pickle=False),
                   np.load(directory/'valid.npy',allow_pickle=False)).validate(len(obs.frame_id))
