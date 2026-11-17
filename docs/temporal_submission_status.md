# Temporal model and submission milestone — 8 November 2026

Historical v1 milestone. The selected v2 release and subsequent robustness/final-evaluation work are documented in `robustness_results.md` and `technical_report.md`; original v1 weights/config remain preserved.

The completed external challenger is a PyTorch temporal convolutional network with 20,035 parameters, six residual convolutions at dilations 1,2,4,8,16,32, and a 127-frame receptive field (5.08 seconds). Its geometry/missingness features also include predefined centred windows. It jointly predicts H/A/D. A sequence-local Viterbi decoder with transition penalty 5 reduces flickering. X is reserved for abstention and ungraded video intervals.

## External validation

| Scenario | Frozen logistic/rule pilot | Neural without decoder | Neural with selected decoder |
|---|---:|---:|---:|
| Clean | 0.4208 | 0.3505 | 0.5822 |
| Short ball occlusion | 0.3208 | 0.3324 | 0.5746 |
| Long ball occlusion | 0.2985 | 0.2084 | 0.5068 |
| Mixed noise | 0.2854 | 0.1374 | 0.5307 |
| Mean spellF1 | 0.3314 | 0.2572 | 0.5486 |

The neural system passes the predeclared promotion gate: improved mean and no scenario regressing more than .02. The raw network alone does not beat the pilot; the temporal decoder is essential. These are correlated conditions on one reused validation match, not independent tests or competition scores.

Training used J03WMX/J03WOH/J03WOY/J03WPY only; validation used J03WQQ. J03WN1/J03WR9 remain unacquired and sealed. Eight fixed epochs, 128 steps/epoch, batch 8, 256-frame supervision cores, AdamW at .001. Team exchange augmentation and two-pass team-swapped probability averaging prevent a fixed home preference. Scaling is fitted to external training features only. Full settings were committed before scoring.

The reviewed training-window boundary inconsistency was reproduced in failing tests and fixed using a per-layer observation mask. The original pre-review checkpoint/report remain locally under ignored artifacts for audit. The corrected experiment was retrained with the same budget and seed; this table reports only the corrected result. The corrected run took approximately 53 seconds in the verified environment.

## Delivery

The pipeline reads tactical ATD plus its metadata, converts coordinates to metres, resets context at each half, predicts H/A/D, and writes exactly 199319 video-frame rows. It uses ATD ID minus one, not the legacy corrected-tracking offsets. Outside the public half intervals it writes X. It rejects incomplete source timelines, invalid IDs/labels, unknown team memberships, changed inputs, model checksum mismatches and existing output paths. The official unchanged validator checks output format without ground truth.

`configs/submission_temporal.json` selects the neural checkpoint; `configs/submission_tactical.json` preserves the frozen logistic fallback. Neither video, broadcast tracking, homography nor Smart Tagging is used. Competition-derived outputs and logs stay local and ignored; no competition metrics, prediction visual review or video comparison is performed.

Model SHA256: `7e485627fc8d672147fc62c8407be81fc581bc7b27ec3ec090b5fb60a683fdf5`.

## Verification completed after reconnection

- Fresh full suite: 64 main tests and 10 input-diagnostic tests passed. The Windows sandbox default temporary folder denied writes; setting TEMP/TMP to a writable workspace directory resolved the environmental failure without changing model code.
- Saved external validation report, checkpoint, training/split configs and submission config hashes agree. The declared promotion gate passes; no sealed test match was evaluated.
- Two complete neural submission runs passed the official format validator and produced byte-identical CSVs. CSVs, provenance and execution logs remain local under ignored `artifacts/competition/neural_verification_20261108/`. No predictions, input contents or competition diagnostics were inspected through the assistant.
- Installed package versions match the CPU dependency lock. A fresh read-only review found no material code defect; its receptive-field wording clarification is incorporated in the technical report.
- The neural checkpoint is the default exporter model; the logistic pilot remains selectable through its original config. No submission or publication was performed.

## Next milestone

- Assess generalisation beyond the single validation match. Keep final tests sealed until the model-selection process is complete.
- Investigate automatic Smart Tagging as additional evidence, with external validation before parameter changes; broadcast requires separate automatic alignment.
- Finalise team naming, public reproducible repository and technical document before submission. Nothing has been published or submitted.
- Confirm the organiser's exact November 17 cutoff time/timezone from participant communication.

## Operational notes

PyTorch could not be installed through the agent environment; the user installed 2.15.0+cpu successfully. No custom training framework was substituted. The local execution connection temporarily dropped during training and recovered; checkpoint/report status was checked before continuing, with no duplicate run launched as a workaround. The only retraining was the deliberate boundary correctness fix.

uv is optional. Existing pip/venv is retained; exact CPU dependency versions are recorded in requirements-cpu-lock.txt. Source, model, protocol and tests are versioned in local Git. No remote repository exists yet.
