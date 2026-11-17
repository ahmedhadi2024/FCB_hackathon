"""Evaluate explicitly released external final matches after selection closes."""
from pathlib import Path
from fcb_external.final_evaluation import run

if __name__=='__main__':
    root=Path(__file__).resolve().parent
    run(root,root/'configs/final_evaluation_v1.json')
