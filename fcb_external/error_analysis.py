"""Supplementary external episode diagnostics; headline scoring stays official."""
import numpy as np
from .metrics import load_official, evaluate_labels
from .features import sequence_spans


def analyze(gt,pred,sequence_id,ball_observed):
    gt=np.asarray(gt); pred=np.asarray(pred); seq=np.asarray(sequence_id)
    seen=np.asarray(ball_observed,dtype=bool)
    if gt.ndim!=1 or pred.shape!=gt.shape or seq.shape!=gt.shape or seen.shape!=gt.shape or not len(gt):
        raise ValueError('Nonempty aligned analysis arrays required')
    pm=load_official('evaluation/possession_metrics.py')
    g=pm.label_codes(gt); p=pm.label_codes(pred)
    gs=[]; ps=[]; transition={}; brief=[]
    for lo,hi in sequence_spans(seq):
        runs=pm.valid_runs(g[lo:hi])
        filled=pm.fill_abstentions(p[lo:hi],runs)
        gs.extend((a+lo,b+lo,k) for a,b,k in pm.possession_spells(g[lo:hi],runs))
        ps.extend((a+lo,b+lo,k) for a,b,k in pm.possession_spells(filled,runs))
        for a,b,k in pm.segments(g[lo:hi],runs):
            if k==pm.D and b-a<50:
                brief.append(float(np.mean(filled[a:b]==pm.D))<.5)
        # Match within each valid run; never across unknown gaps or sequences.
        for run in runs:
            a=pm.transitions(g[lo:hi],[run]); b=pm.transitions(filled,[run])
            for origin in range(3):
                for dest in range(3):
                    if origin==dest: continue
                    aa=[t for t in a if t[1:]==(origin,dest)]
                    bb=[t for t in b if t[1:]==(origin,dest)]
                    if not aa and not bb: continue
                    key=f'{"HAD"[origin]}->{"HAD"[dest]}'
                    item=transition.setdefault(key,{'matched':0,'missed':0,'spurious':0,'latencies':[]})
                    pairs,missed,spurious=pm.match_transitions(aa,bb,25,typed='pair')
                    item['matched']+=len(pairs); item['missed']+=len(missed); item['spurious']+=len(spurious)
                    item['latencies'].extend(abs(y[0]-x[0]) for x,y in pairs)
    overlaps=pm.spell_overlaps(gs,ps)
    matched=pm.matched_gt_flags(overlaps,len(gs),.5)
    tp,fp,fn=pm.match_spells(overlaps,len(gs),len(ps),.5)
    bins={name:{'gt':0,'matched_iou_0.5':0} for name in ('under_2s','2_to_5s','5_to_15s','15s_plus')}
    for i,(a,b,_) in enumerate(gs):
        name=list(bins)[int(b-a>=50)+int(b-a>=125)+int(b-a>=375)]
        bins[name]['gt']+=1; bins[name]['matched_iou_0.5']+=int(matched[i])
    for item in bins.values():
        item['recall_iou_0.5']=item['matched_iou_0.5']/item['gt'] if item['gt'] else None
    merged=sum(sum(k==pk and max(0,min(b,pb)-max(a,pa))>=.25*(b-a)
                   for a,b,k in gs)>=2 for pa,pb,pk in ps)
    for item in transition.values():
        lat=item.pop('latencies')
        item['mean_abs_latency_frames']=float(np.mean(lat)) if lat else None
    visibility={}
    for name,mask in [('observed',seen),('missing',~seen)]:
        valid=mask&(g!=pm.X)
        visibility[name]={'graded_frames':int(valid.sum()),
                          'frame_accuracy':float(np.mean(g[valid]==p[valid])) if valid.any() else None}
    return {'official':evaluate_labels(gt,pred,seq),'duration_bins':bins,
            'gt_spells':len(gs),'pred_spells':len(ps),'missed_spells_iou_0.5':fn,
            'spurious_spells_iou_0.5':fp,'merged_predictions':int(merged),
            'merge_definition':'Prediction overlaps at least 25% of each of two same-team reference spells; diagnostic, not causal proof.',
            'brief_dead_segments':{'count':len(brief),'missed_majority':int(sum(brief))},
            'transitions':transition,'visibility':visibility,
            'interpretation':'Supplementary external diagnostics; duration bins do not replace full official scoring.'}
