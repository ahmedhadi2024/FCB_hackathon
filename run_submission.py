"""Run locally. Redirect output to an ignored local log when using provided data."""
import argparse
from pathlib import Path
from fcb_external.submission import run


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--config',type=Path,
        default=Path(__file__).resolve().parent/'configs/submission_final.json')
    args=parser.parse_args()
    run(args.config,args.data_dir,args.out)
    print('Submission generated and passed official format validation. No performance measured.')
