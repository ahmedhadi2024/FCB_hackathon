"""Run the predeclared external-only model experiment; sealed test matches stay closed."""
from pathlib import Path
from fcb_external.experiment import run

try:
    run(Path(__file__).resolve().parent)
except (ValueError,OSError) as exc:
    raise SystemExit(str(exc))
