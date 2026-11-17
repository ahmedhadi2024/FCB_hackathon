# External DFL target mapping

Schema paths: positions XML `FrameSet[TeamId=ball]/Frame`, attributes `BallStatus` and `BallPossession`. Frame keys are GameSection + N. The DFL label fields are external supervision only; no target fields enter observation arrays, temporal segmentation or baseline inference.

| BallStatus | BallPossession | Target |
|---|---|---|
| literal 0 | anything | D |
| literal 1 | literal 1 | H |
| literal 1 | literal 2 | A |
| otherwise/missing | anything | X |

Unknown targets are not dead ball and are never filled. A frame may have a valid target even when its ball position is missing, or a valid ball observation but unknown target. Slots and time keys are constructed independently of target validity. All inferred labels cover all observation rows before evaluation masking.

Source: Bassek et al. 2025 and floodlight's DFL reader document BallStatus 0=Dead,1=Alive and BallPossession 1=Home,2=Away. Do not infer an undocumented code's meaning from geometry or the eventual score. Treat numeric strings in their actual exported form; nonliteral unsupported representations stay X rather than guessing.

Real-file conversion succeeded on J03WMX. `BallStatus` has exactly 0 and 1, with counts 60,598 and 85,369. `BallPossession` has exactly 1 and 2, with counts 74,558 and 71,409. There are 145,967 ball-frame rows and 145,967 canonical observation rows, across two sequences/halves; all have valid H/A/D targets. After inactive status overrides ownership, counts are H=39,111, A=46,258, D=60,598. These are external-match statistics, not competition data.

Full observed domains and coverage are recorded in `artifacts/J03WMX/manifest.json` under `audit`. Label-level human/algorithmic/mixed provenance is not established by this implementation; the source is official DFL data, and this distinction remains a documentation limitation. DFL targets are not asserted equivalent to Metrica's controlled-possession convention.
