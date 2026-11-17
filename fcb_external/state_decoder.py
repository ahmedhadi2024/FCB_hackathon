"""Three-state decoding with separate symmetric owner and dead/live costs."""
import numpy as np
from .features import sequence_spans


def decode(probabilities,sequence_id,owner_penalty,dead_live_penalty):
    p=np.asarray(probabilities,dtype=float)
    costs=np.asarray([owner_penalty,dead_live_penalty],dtype=float)
    if p.shape!=(len(sequence_id),3) or not len(p) or not np.isfinite(p).all() or np.any(p<0) or \
            not np.allclose(p.sum(axis=1),1,atol=1e-5) or not np.isfinite(costs).all() or np.any(costs<0):
        raise ValueError('Aligned finite probabilities and nonnegative finite transition costs required')
    seq=np.asarray(sequence_id)
    if seq.ndim!=1 or not np.issubdtype(seq.dtype,np.integer) or seq[0]!=0 or not np.isin(np.diff(seq),[0,1]).all():
        raise ValueError('Consecutive nondecreasing integer sequence IDs required')
    transition=-np.array([[0,owner_penalty,dead_live_penalty],
                          [owner_penalty,0,dead_live_penalty],
                          [dead_live_penalty,dead_live_penalty,0]],dtype=float)
    result=np.empty(len(p),np.int8)
    for lo,hi in sequence_spans(seq):
        emission=np.log(np.maximum(p[lo:hi],1e-8))
        score=emission[0]; back=np.empty((hi-lo,3),np.int8)
        for i in range(1,hi-lo):
            options=score[:,None]+transition
            back[i]=options.argmax(axis=0)
            score=options.max(axis=0)+emission[i]
        state=int(score.argmax())
        for i in range(hi-lo-1,-1,-1):
            result[lo+i]=state
            if i: state=int(back[i,state])
    return np.array(['H','A','D'])[result]
