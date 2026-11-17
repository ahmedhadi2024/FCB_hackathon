"""Fetch the external matches the v4-v6 experiments need beyond download_development.py.

1. DFL/IDSSE matches J03WN1 and J03WR9 (CC BY 4.0). They were the final test of the v2
   release and are ordinary development data from v4 on. Checksums are pinned in
   configs/idsse_J03WN1.json and configs/idsse_J03WR9.json and enforced by the prepare step.
2. Metrica Sports public sample games (github.com/metrica-sports/sample-data), used only to
   choose between pipelines. Please acknowledge Metrica Sports when using them.

Neither source contains the competition match.

    python download_external_extra.py
"""
import json
from pathlib import Path
import urllib.request
from fcb_external.cli import prepare

ROOT=Path(__file__).resolve().parent
METRICA='https://raw.githubusercontent.com/metrica-sports/sample-data/master/data'
METRICA_FILES=['Sample_Game_1/Sample_Game_1_RawEventsData.csv',
               'Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv',
               'Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv',
               'Sample_Game_2/Sample_Game_2_RawEventsData.csv',
               'Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv',
               'Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv']


def fetch(url,path):
    path=Path(path)
    if path.exists():
        print('Already present:',path.name,flush=True); return
    path.parent.mkdir(parents=True,exist_ok=True)
    partial=path.with_name(path.name+'.partial')
    print('Downloading',path.name,flush=True)
    with urllib.request.urlopen(url,timeout=120) as response,partial.open('wb') as out:
        for block in iter(lambda:response.read(1024*1024),b''):
            out.write(block)
    partial.replace(path)


def main():
    for match in ('J03WN1','J03WR9'):
        config=ROOT/'configs'/f'idsse_{match}.json'
        cfg=json.loads(config.read_text())
        base=f'https://huggingface.co/datasets/pysport/idsse-data/resolve/{cfg["source_revision"]}'
        for kind in ('metadata','events','positions'):
            relative=cfg[kind]
            fetch(f'{base}/{Path(relative).name}',ROOT/relative)
        if (ROOT/'artifacts'/match/'manifest.json').exists():
            print('Already prepared:',match,flush=True)
        else:
            prepare(config)        # verifies the pinned SHA256 of every file
    for relative in METRICA_FILES:
        fetch(f'{METRICA}/{relative}',ROOT/'external-data/metrica_sample'/Path(relative).name)
    print('External extras ready.')


if __name__=='__main__':
    main()
