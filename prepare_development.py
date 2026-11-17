"""Convert only protocol-authorized training/validation files already downloaded."""
import json
from pathlib import Path
from download_development import development_ids
from fcb_external.cli import prepare

root=Path(__file__).resolve().parent
protocol=json.loads((root/'configs/experiment_v1.json').read_text())
for match in development_ids(protocol):
    config=root/'configs'/f'idsse_{match}.json'
    if not config.exists():
        raise SystemExit(f'Download not complete for {match}; run download_development.py')
    if (root/'artifacts'/match/'manifest.json').exists():
        print('Already prepared:',match,flush=True)
        continue
    print('Preparing:',match,flush=True)
    prepare(config)
