"""Acquire only the frozen training/validation matches; sealed tests are excluded."""
import hashlib
import json
from pathlib import Path
import sys
import urllib.request
from download_idsse import REVISION, sha256
from fcb_external.xmlio import validate_xml

ROOT=Path(__file__).resolve().parent
BASE='https://huggingface.co'


def development_ids(protocol):
    train,val,test=map(set,(protocol['train'],protocol['validation'],protocol['sealed_test']))
    if train & val or train & test or val & test:
        raise ValueError('Train/validation/test overlap is forbidden')
    return list(dict.fromkeys(protocol['train']+protocol['validation']))


def main():
    protocol=json.loads((ROOT/'configs/experiment_v1.json').read_text())
    matches=development_ids(protocol)
    url=f'{BASE}/api/datasets/pysport/idsse-data/tree/{REVISION}?recursive=false&expand=false'
    with urllib.request.urlopen(url,timeout=60) as response:
        listing={entry['path']:entry for entry in json.load(response)}
    for match in matches:
        division='000001' if match=='J03WMX' else '000002'
        paths={kind:f'DFL_{prefix}_DFL-COM-{division}_DFL-MAT-{match}.xml'
               for kind,prefix in [('metadata','02_01_matchinformation'),
                                   ('events','03_02_events_raw'),
                                   ('positions','04_03_positions_raw_observed')]}
        destination=ROOT/'external-data'/match
        destination.mkdir(parents=True,exist_ok=True)
        expected_sha={}
        downloads=[]
        for kind,name in paths.items():
            info=listing[name]
            path=destination/name
            source=f'{BASE}/datasets/pysport/idsse-data/resolve/{REVISION}/{name}?download=true'
            if not path.exists():
                print(f'Downloading {match}: {kind} ({info["size"]:,} bytes)',flush=True)
                partial=path.with_suffix('.xml.partial')
                with urllib.request.urlopen(source,timeout=60) as response, partial.open('wb') as out:
                    for chunk in iter(lambda:response.read(1024*1024),b''):
                        out.write(chunk)
                candidate=partial
            else:
                candidate=path
            if candidate.stat().st_size != info['size']:
                raise ValueError(f'{name}: incomplete/wrong-size file')
            digest=sha256(candidate)
            if info.get('lfs'):
                if digest != info['lfs']['oid']:
                    raise ValueError(f'{name}: SHA256 mismatch')
            else:
                content=candidate.read_bytes()
                blob=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
                if blob != info['oid']:
                    raise ValueError(f'{name}: Git blob checksum mismatch')
            validate_xml(candidate,match,{'metadata':'MatchInformation','events':'Event',
                                          'positions':'Positions'}[kind])
            if candidate != path:
                candidate.replace(path)
            expected_sha[kind]=digest
            downloads.append({'name':name,'url':source,'size_bytes':info['size'],'sha256':digest})
            print(f'Verified {match}: {kind}',flush=True)
        (destination/'download_manifest.json').write_text(json.dumps(
            {'revision':REVISION,'licence':'CC BY 4.0','files':downloads},indent=2)+'\n')
        config={'project_root':'..','match_id':match,'coordinate_system':'dfl-centred-metres',
                **{kind:f'external-data/{match}/{name}' for kind,name in paths.items()},
                'output':f'artifacts/{match}','source_revision':REVISION,
                'source_urls':[f'{BASE}/datasets/pysport/idsse-data/tree/{REVISION}'],
                'expected_sha256':expected_sha}
        (ROOT/'configs'/f'idsse_{match}.json').write_text(json.dumps(config,indent=2)+'\n')
    print('Training/validation download complete. Sealed test matches were not acquired.')


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        print(f'Download failed: {exc}',file=sys.stderr)
        raise SystemExit(1)
