# First model execution plan

Approved scope: freeze external splits/inputs, run external oracle error decomposition, prove IDSSE/ATD adapter equivalence using fictional data, train one live/dead model and compare combined pipelines on external validation.

## Decisions frozen before further data inspection

`configs/experiment_v1.json` is the executable experiment protocol. Train: J03WMX/J03WOH/J03WOY/J03WPY. Validate: J03WQQ. Sealed final test: J03WN1/J03WR9. J03WMX has already been examined, so belongs to development/training. Test assets will not be downloaded. The other unseen first-division match is retained for final testing. Shared teams across second-division matches remain a limitation; identity/name/clock features are excluded.

Use NumPy logistic regression so no new dependency installation blocks progress. L2, sample budget, seed and iteration budget are fixed, not tuned. Fit standardization on sampled training features only. Training rows cover all four predefined observation-corruption scenarios; target labels are unchanged and never drive corruption. Select only the finite six-combination probability threshold/context grid using combined pipeline spellF1 on external validation across the same fixed conditions. Also report every clean/per-condition score; no claim of independent generalisation.

The ownership component is the published baseline_smooth50 reference, fixed in advance as a reference, with within-sequence propagation through missing observations for model-combined variants. Report that propagation as a separately measured control, so an apparent gain cannot be attributed silently to it.

## Implementation order

1. Record protocol and local Git checkpoint; extend acquisition to train/validation matches only, validating pinned-revision downloads and XML identity.
2. Write tests for external-only oracle diagnostic, synthetic ATD adapter alignment/team/units/missingness, cross-format equivalence and model-input invariance.
3. Implement shared features and deterministic corruption; test no half/gap crossing, within-team slot permutation invariance and unchanged labels.
4. Implement logistic model and offline decoding; test learning on fictional separable features, serialization, target-independent inference and sequence boundaries.
5. Convert only training/validation external matches and run frozen protocol, record metrics/hashes/config/Git version. Do not use sealed test data or competition assets.
6. Independently review source, fix meaningful defects with regression tests, run whole suite, save model/source checkpoint and human-readable finished/remaining summary.

No competition training, tuning, video checking, or hosted data exposure. No dashboard, video model, harness installation or unrestricted parameter search.
