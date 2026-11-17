# Possession-duration layer result

Implemented an optional hybrid TCN plus soft duration-aware sequence decoder. Expanded decoder states retain class and capped spell age; leaving an interior H/A spell shorter than 12 frames adds cost 2 on top of transition cost 10. D spells and censored sequence-edge spells have no duration penalty. This is an alternative decoder, not a second smoothing pass. Setting the extra cost to zero returns the existing Viterbi output exactly.

The fixed candidate was evaluated on the original four scenarios of the reused external validation match J03WQQ. No public five-minute labels were read or used to choose parameters in this experiment. No neural retraining or competition inference occurred.

| Scenario | v2 spellF1 | Duration layer spellF1 |
|---|---:|---:|
| Clean | 0.607038 | 0.607038 |
| Short occlusion | 0.611387 | 0.611387 |
| Long occlusion | 0.537705 | 0.538507 |
| Mixed noise | 0.540588 | 0.540588 |

Mean improvement: 0.000201. Worst scenario change: zero. This is far below the predeclared 0.01 improvement threshold, so it is not selected. Tiny development-score differences are not evidence of a meaningful gain. All 124 main tests passed, including exhaustive optimal-path comparison on a tiny example, baseline parity when disabled, sequence reset and team symmetry. Author review only.

The public-sample score identifies neither the cause of errors nor a suitable duration prior. Do not increase the duration penalty based on public answers. A stronger duration model would require a separately specified external experiment. Existing v2 remains the selected submission.

Source: fcb_external/possession_duration.py. Reproduction: python run_possession_duration.py; the runner refuses to overwrite reports/possession_duration_v1.json. Full report stays local. The image accompanying this checkpoint is explanatory artwork; source and this document define the exact architecture and experiment status.


Provenance correction: this page describes the historical v1 run. The corrected runner now targets a separate v2 report with pinned scenario definitions and recorded source/config/input fingerprints. No v2 experiment has been run as part of that correction, and no v1 scores have been rewritten.
