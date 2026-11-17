# J03WMX external development benchmark

This is ONE externally sourced development match, not the competition match and not a held-out generalisation test. No model was trained and no baseline parameter was selected.

## Verified conversion

- Match: DFL-MAT-J03WMX (1. FC Koeln vs FC Bayern Muenchen).
- Source: IDSSE / DFL, CC BY 4.0; PySport mirror revision a715a38dfbaf5f58e431727c2b78d174101a703c.
- Download sizes: metadata 11,930 bytes; events 776,578 bytes; tracking 418,524,833 bytes.
- Tracking SHA256 matches the published digest.
- 145,967 observation frames, two periods/sequences, verified 25 Hz timestamps, 40 roster slots.
- Team roles come from metadata home/guest IDs, despite reversed XML team ordering.
- All external target fields map to documented domains; no unknown target frames.
- Target counts: H 39,111; A 46,258; D 60,598.
- Ball x/y coordinates are finite on all these frames. This does not establish visual detection certainty; coordinates can remain available during stoppages.

## Fixed reference results

| Baseline | macroF1 | spellF1 | Predicted spells | External target spells |
|---|---:|---:|---:|---:|
| Raw | 0.3826 | 0.1043 | 2,240 | 511 |
| +/-25 frames | 0.4106 | 0.3025 | 560 | 511 |
| +/-50 frames | 0.4150 | 0.3072 | 360 | 511 |
| +/-125 frames | 0.4065 | 0.2193 | 224 | 511 |

Reported in predefined variant order, not a model-selection ranking. Official metric SHA256: `341bad48904b489d6e875397b27687f5d6a4d22a397f3dc8af3a00d48f16874d`.

## Finding and next experiment

Every variant has D-class F1=0. The nearest-player reference uses missing coordinates as its only route to D; the external ball coordinates remain finite, so it labels stoppages as team possession. The smooth50 variant's correctness on externally labelled live frames is approximately 0.8455, but it still misses all dead-ball states. This separates a stoppage failure from team inference quality.

The next justified experiment is an explicit live/dead detector based on geometry and motion, combined with the fixed possession baseline. Train and select it only after defining external train/validation/test match assignments. J03WMX is already examined development data and cannot be called untouched testing. Do not introduce competition-only Smart Tagging as a required training feature without a defensible external analogue and evaluation protocol.

DFL possession conventions may differ from Metrica's. The clean external coordinates also differ from the competition's published missing-ball problem. Future external tests must include predefined occlusion/noise conditions; the external scores above are not a prediction of leaderboard performance.

## Verification

The complete synthetic suite currently contains 31 tests, including wrong-match/truncated downloads, exact target mapping, target isolation, entity joins, metre-scaled distances, direct reference equivalence for raw nearest-player logic and smoothing, gap/period separation, official spell metrics, CLI integration and synthetic deterministic reruns. The real-match deterministic rerun is checked separately in `docs/reproduction_check.json` when available.

Peak process memory was not measured: the execution environment did not expose the running Python process to `Get-Process`. The converter uses two streaming XML passes and bounded arrays; no measured memory-performance claim is made.
