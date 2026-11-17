"""Fetch the two external test sets used by run_bench.py. Neither contains the competition match.

1. Metrica Sports public sample game 3 (github.com/metrica-sports/sample-data): EPTS tracking,
   metadata and JSON events. Please acknowledge Metrica Sports when using it.
2. SkillCorner open data (github.com/SkillCorner/opendata, MIT): every match listed in the
   provider's matches.json, with match metadata, phases of play and extrapolated tracking.

No shipped model was trained on either set.

    python download_bench.py
"""
import json
from pathlib import Path
from download_external_extra import METRICA, fetch
from fcb_external import skillcorner

ROOT=Path(__file__).resolve().parent
GAME_3=['Sample_Game_3_events.json','Sample_Game_3_metadata.xml','Sample_Game_3_tracking.txt']


def main():
    for name in GAME_3:
        fetch(f'{METRICA}/Sample_Game_3/{name}',ROOT/'external-data/metrica_sample'/name)
    folder=ROOT/'external-data/skillcorner'
    fetch(f'{skillcorner.BASE}/matches.json',folder/'matches.json')
    for meta in json.loads((folder/'matches.json').read_text(encoding='utf-8')):
        for name,url in skillcorner.files(meta['id']).items():
            fetch(url,folder/name)
    print('Bench data ready.')


if __name__=='__main__':
    main()
