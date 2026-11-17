"""External development error decomposition. Never a deployable predictor."""
from pathlib import Path
import numpy as np
from fcb_external.schema import Observations,Targets,write_json
from fcb_external.baseline import predict_all
from fcb_external.diagnostics import oracle_predictions
from fcb_external.metrics import evaluate_labels

root=Path(__file__).resolve().parent
obs=Observations.load(root/'artifacts/J03WMX/observations')
gt=Targets.load(root/'artifacts/J03WMX/targets',obs)
report={'purpose':'EXTERNAL ORACLE DIAGNOSTIC ONLY; uses labels and cannot be deployed',
        'development_match':'J03WMX','variants':{}}
for name,pred in predict_all(obs).items():
    status,team=oracle_predictions(gt.label,pred,obs.sequence_id)
    report['variants'][name]={}
    for kind,labels in [('reference',pred),('oracle_status',status),('oracle_team',team)]:
        result=evaluate_labels(gt.label,labels,obs.sequence_id)
        report['variants'][name][kind]=result
        print(name,kind,'spellF1=',round(result['metrics']['spell_f1_mean'],4),
              'macroF1=',round(result['metrics']['macro_f1'],4))
write_json(root/'reports/oracles/J03WMX.json',report)
