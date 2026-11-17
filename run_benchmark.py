"""Reproduce the one-match external benchmark from downloaded raw XML."""
import os
from pathlib import Path
import sys
from fcb_external.cli import main

root=Path(__file__).resolve().parent
os.chdir(root)
for args in (
    ['prepare','--config','configs/idsse_J03WMX.json'],
    ['baseline','--observations','artifacts/J03WMX/observations','--out','artifacts/J03WMX/predictions'],
    ['evaluate','--observations','artifacts/J03WMX/observations','--targets','artifacts/J03WMX/targets',
     '--predictions','artifacts/J03WMX/predictions','--out','reports/J03WMX'],
):
    code=main(args)
    if code:
        raise SystemExit(code)
print((root/'reports/J03WMX/scoreboard.csv').read_text(encoding='utf-8'))
