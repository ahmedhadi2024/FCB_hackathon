"""Acquire only explicitly frozen final external matches, then verify/prepare locally."""
from pathlib import Path
import json
import urllib.request
import hashlib
from fcb_external.final_evaluation import verify_frozen
from fcb_external.schema import write_json
from fcb_external.xmlio import validate_xml
from fcb_external.cli import prepare
from download_idsse import REVISION,sha256


def resolve_assets(listing,match):
    names={}
    for kind,prefix in [('metadata','02_01_matchinformation'),('events','03_02_events_raw'),
                        ('positions','04_03_positions_raw_observed')]:
        candidates=[name for name in listing if name.startswith(f'DFL_{prefix}_DFL-COM-')
                    and name.endswith(f'_DFL-MAT-{match}.xml')]
        if len(candidates)!=1:
            raise ValueError('Missing or ambiguous listed external asset: '+match+' '+kind)
        names[kind]=candidates[0]
    return names


def main(root,listing_path=None):
    root=Path(root)
    cfg=json.loads((root/'configs/final_evaluation_v1.json').read_text())
    verify_frozen(root,cfg)  # Must pass before network or test asset access.
    split=json.loads((root/'configs/experiment_v1.json').read_text())
    if set(cfg['matches'])!=set(split['sealed_test']) or set(cfg['matches'])&set(split['train']+split['validation']):
        raise ValueError('Final acquisition roles do not match frozen split')
    url=f'https://huggingface.co/api/datasets/pysport/idsse-data/tree/{REVISION}?recursive=false&expand=false'
    if listing_path:
        listing={item['path']:item for item in json.loads(Path(listing_path).read_text())}
    else:
        with urllib.request.urlopen(url,timeout=60) as response:
            listing={item['path']:item for item in json.load(response)}
    for match in cfg['matches']:
        names=resolve_assets(listing,match)
        hashes={}; assets=[]; folder=root/'external-data'/match; folder.mkdir(parents=True,exist_ok=True)
        for kind,name in names.items():
            info=listing[name]; path=folder/name
            source=f'https://huggingface.co/datasets/pysport/idsse-data/resolve/{REVISION}/{name}?download=true'
            if not path.exists():
                if listing_path:
                    raise ValueError('Verified-local mode requires complete local asset: '+name)
                partial=path.with_suffix('.xml.partial')
                print('Downloading external final asset',match,kind,info['size'],flush=True)
                with urllib.request.urlopen(source,timeout=60) as response,partial.open('wb') as out:
                    for block in iter(lambda:response.read(1024*1024),b''): out.write(block)
                candidate=partial
            else: candidate=path
            if candidate.stat().st_size!=info['size']: raise ValueError('Incomplete external final asset')
            digest=sha256(candidate)
            if info.get('lfs'):
                if digest!=info['lfs']['oid']: raise ValueError('External final SHA256 mismatch')
            else:
                content=candidate.read_bytes()
                if hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()!=info['oid']:
                    raise ValueError('External final Git checksum mismatch')
            validate_xml(candidate,match,{'metadata':'MatchInformation','events':'Event','positions':'Positions'}[kind])
            if candidate!=path: candidate.replace(path)
            hashes[kind]=digest; assets.append({'name':name,'url':source,'size_bytes':info['size'],'sha256':digest})
        write_json(folder/'download_manifest.json',{'revision':REVISION,'licence':'CC BY 4.0','files':assets})
        prepared={'project_root':'..','match_id':match,'coordinate_system':'dfl-centred-metres',
                  **{k:f'external-data/{match}/{n}' for k,n in names.items()},'output':f'artifacts/{match}',
                  'source_revision':REVISION,'source_urls':[url],'expected_sha256':hashes}
        config=root/'configs'/f'idsse_{match}.json'; write_json(config,prepared)
        prepare(config)
        print('External final match verified/prepared:',match,flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--listing',type=Path,help='Pinned listing for verification-only local preparation; no network')
    args=parser.parse_args()
    main(Path(__file__).resolve().parent,args.listing)
