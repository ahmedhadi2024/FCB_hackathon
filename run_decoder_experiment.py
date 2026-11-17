"""Run only the predeclared decoder-cost validation experiment; no retraining."""
from pathlib import Path
from fcb_external.decoder_experiment import run

if __name__=='__main__': run(Path(__file__).resolve().parent)
