"""Train and compare the frozen external temporal experiment; no competition inputs."""
from pathlib import Path
import argparse
from fcb_external.temporal_experiment import run

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='temporal_v1.json')
    args=parser.parse_args()
    run(Path(__file__).resolve().parent,args.config)
