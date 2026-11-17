"""Small regularized logistic live/dead model and sequence-safe offline decoding."""
import numpy as np
from .features import sequence_spans,window_mean


def sigmoid(z):
    return 1/(1+np.exp(-np.clip(z,-35,35)))


def fit(x,y,l2=0.01,iterations=40):
    x=np.asarray(x,dtype=float); y=np.asarray(y,dtype=float)
    if x.ndim!=2 or y.shape!=(len(x),) or not np.isfinite(x).all() or set(np.unique(y))!={0.,1.}:
        raise ValueError('Training requires finite features and both binary classes')
    mean=x.mean(axis=0); scale=x.std(axis=0); scale[scale<1e-8]=1
    design=np.column_stack([np.ones(len(x)),(x-mean)/scale])
    weights=np.zeros(design.shape[1])
    class_weight=np.where(y==1,0.5/y.mean(),0.5/(1-y.mean()))
    penalty=np.eye(len(weights))*l2; penalty[0,0]=1e-8
    for _ in range(iterations):
        prob=sigmoid(design@weights)
        grad=design.T@(class_weight*(prob-y))/len(x)+penalty@weights
        curve=class_weight*np.maximum(prob*(1-prob),1e-8)
        hessian=(design.T*curve)@design/len(x)+penalty
        step=np.linalg.solve(hessian,grad)
        magnitude=np.linalg.norm(step)
        if magnitude>10:
            step*=10/magnitude
        weights-=step
        if np.linalg.norm(step)<1e-7:
            break
    return {'kind':'numpy_logistic_regression','mean':mean.tolist(),'scale':scale.tolist(),
            'weights':weights.tolist(),'l2':l2,'iterations_budget':iterations,
            'training_rows':len(x),'training_dead_fraction':float(y.mean())}


def probability(model,x):
    z=(np.asarray(x,dtype=float)-np.array(model['mean']))/np.array(model['scale'])
    return sigmoid(np.column_stack([np.ones(len(z)),z])@np.array(model['weights']))


def team_states(labels,sequence_id):
    result=np.array(labels,dtype='<U1').copy()
    for lo,hi in sequence_spans(sequence_id):
        last='X'
        for i in range(lo,hi):
            if result[i] in ('H','A'):
                last=result[i]
            else:
                result[i]=last
        first=next((v for v in result[lo:hi] if v in ('H','A')),'X')
        for i in range(lo,hi):
            if result[i]!='X':
                break
            result[i]=first
    return result


def decode(owners,dead_probability,sequence_id,threshold,half_width):
    p=window_mean(dead_probability,sequence_id,half_width)
    return np.where(p>=threshold,'D',owners)
