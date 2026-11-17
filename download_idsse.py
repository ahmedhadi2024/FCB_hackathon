"""Download only the three external J03WMX XML assets. Standard library only."""
import hashlib
import json
from pathlib import Path
import sys
import urllib.request
from fcb_external.xmlio import validate_xml

REVISION = 'a715a38dfbaf5f58e431727c2b78d174101a703c'
FILES = [
    'DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-J03WMX.xml',
    'DFL_03_02_events_raw_DFL-COM-000001_DFL-MAT-J03WMX.xml',
    'DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-J03WMX.xml',
]
TRACKING_SHA256 = 'd3dd92e509421a6466acb0a083c009035f1751bc5fd9ef42ffa8f20d01cf7d49'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def download(destination):
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {'revision': REVISION, 'licence': 'CC BY 4.0', 'files': []}
    for name in FILES:
        path = destination / name
        required = 'Positions' if name.startswith('DFL_04') else (
            'MatchInformation' if name.startswith('DFL_02') else 'Event')
        url = f'https://huggingface.co/datasets/pysport/idsse-data/resolve/{REVISION}/{name}?download=true'
        if not path.exists():
            temp = path.with_suffix('.xml.partial')
            print(f'Downloading {name}', flush=True)
            request = urllib.request.Request(url, headers={'User-Agent': 'FCB-external-benchmark/1'})
            with urllib.request.urlopen(request, timeout=60) as response, temp.open('wb') as handle:
                expected = response.headers.get('Content-Length')
                for chunk in iter(lambda: response.read(1024 * 1024), b''):
                    handle.write(chunk)
            if expected and temp.stat().st_size != int(expected):
                raise ValueError('Incomplete transfer; partial file retained, not accepted')
            if name == FILES[2] and sha256(temp) != TRACKING_SHA256:
                raise ValueError('Tracking checksum differs from the published SHA256')
            validate_xml(temp,'J03WMX',required)
            temp.replace(path)
        digest = sha256(path)
        if name == FILES[2] and digest != TRACKING_SHA256:
            raise ValueError('Existing tracking file fails published checksum; not overwriting it')
        validate_xml(path,'J03WMX',required)
        manifest['files'].append({'name': name, 'url': url, 'sha256': digest,
                                  'size_bytes': path.stat().st_size})
        print(f'Verified {name} ({path.stat().st_size:,} bytes)', flush=True)
    (destination / 'download_manifest.json').write_text(
        json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print('External download complete. No competition files were accessed.')


if __name__ == '__main__':
    try:
        download(Path(__file__).resolve().parent / 'external-data' / 'J03WMX')
    except Exception as exc:
        print(f'Download failed: {exc}', file=sys.stderr)
        raise SystemExit(1)
