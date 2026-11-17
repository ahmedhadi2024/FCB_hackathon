"""Ball-loss scenario shaped like the competition tracking's measured, label-free gap profile.

Measured on the competition ATD on graded frames (observations only, no answers):
  - ball missing on about 41% of frames, in few long gaps (median ~10.7 s, p25 ~6 s, p75 ~21 s,
    p90 ~34 s, ~75% of missing frames in gaps longer than 15 s);
  - the ball is missing on about 58% of slow-player frames against about 13% of other frames;
  - about 78% of gaps begin within 3 m of a touchline or goal line.

mask() reproduces those three properties on clean external matches. It conditions only on
observed player pace and ball position, never on any label.
"""
from dataclasses import replace
import numpy as np
from .features import sequence_spans
from .team_features_v6 import pooled_pace, SLOW_MEDIAN

LOG_MEDIAN_S = np.log(10.7)
SIGMA = 0.92            # from p25/p75 of the measured gap lengths
MIN_S, MAX_S = 0.4, 110.0
NEAR_LINE_M = 3.0


def mask(obs, fraction, slow_weight, line_weight, seed):
    if not 0 < fraction <= 1 or slow_weight < 0 or line_weight < 0:
        raise ValueError('Invalid target-like mask parameters')
    rng = np.random.default_rng(seed)
    median, valid = pooled_pace(obs)
    weight = 1 + slow_weight * (valid & (median < SLOW_MEDIAN))
    xy = np.asarray(obs.ball_xy_m, dtype=np.float64)
    L, W = obs.pitch_length_m, obs.pitch_width_m
    with np.errstate(invalid='ignore'):
        near = (np.minimum(xy[:, 0], L - xy[:, 0]) < NEAR_LINE_M) | (np.minimum(xy[:, 1], W - xy[:, 1]) < NEAR_LINE_M)
    weight = weight * (1 + line_weight * np.nan_to_num(near.astype(float)))
    ball = np.array(obs.ball_xy_m, dtype=np.float32)
    for lo, hi in sequence_spans(obs.sequence_id):
        n = hi - lo; target = int(np.ceil(n * fraction)); hidden = np.zeros(n, bool)
        guard = 0
        while hidden.sum() < target and guard < 100000:
            guard += 1
            w = weight[lo:hi] * (~hidden)
            if w.sum() <= 0:
                break
            start = int(rng.choice(n, p=w / w.sum()))
            seconds = float(np.clip(np.exp(rng.normal(LOG_MEDIAN_S, SIGMA)), MIN_S, MAX_S))
            end = min(n, start + max(1, int(seconds * 25)))
            fresh = np.flatnonzero(~hidden[start:end]) + start
            hidden[fresh[:target - int(hidden.sum())]] = True
        ball[lo:hi][hidden] = np.nan
    return replace(obs, ball_xy_m=ball, ball_observed=np.isfinite(ball).all(axis=-1)).validate()
