# First model experiment — scope and rollback

## Completed foundations

- Original public baseline reproduced and validated.
- External IDSSE adapter and official metric integration verified on one real match.
- External split frozen in Git before further label inspection: training WMX/WOH/WOY/WPY; validation WQQ; final test WN1/WR9.
- Only training/validation assets downloaded, checksum-checked and converted.
- Synthetic ATD/IDSSE geometry/feature equivalence, explicit team mapping and within-sequence motion/decoding checks implemented.
- External oracle diagnostic confirms team-transition errors remain important even with perfect stoppage information.

## Model protocol

One NumPy logistic live/dead model; identity-free geometry/motion/missingness features, training-only scaling, fixed sample budget/regularization. Four predefined corruption conditions: clean, short ball occlusion, long ball occlusion and mixed player dropout/coordinate noise/slot switches. Six fixed decoding choices (three probability thresholds, two temporal widths) selected using external validation combined-pipeline spellF1. An ownership-propagation-only control is separately measured.

This is pilot development, not an independent test or a competition score. Synthetic occlusion conditions do not establish transfer to the competition's missing-data mechanism.

## Review and implementation checks

An independent source reviewer found no source-level target leakage or sealed-test access. A P2 provenance gap was reproduced with a failing test: pitch metadata could change without changing the recorded array hashes. The fix fingerprints observation/target metadata and prepared manifests alongside the arrays. The full main suite then passed 44 tests. The earlier input diagnostic suite passed 10 tests separately.

Timeline validation was vectorized without changing its invariants to reduce repeated feature-extraction overhead. A complete second fit/validation run checks numerical reproducibility after that refactor. Oracle status correction uses baseline-derived ownership propagation to remove false-D outputs; on J03WMX the four reference variants have no D outputs, so this propagation is identity. The diagnostic never supplies the true team in its status-oracle branch.

Final result: the complete second fit/validation run reproduced all model weights/scaling, decoder selection, candidate scores and per-condition metrics exactly. Metadata fingerprint presence and sealed-test absence were explicitly checked. The trained model release is versioned in `models/live_dead_v1.json`; see `docs/pilot_results.md` for the uneven validation gains and remaining work.

## Rollback and ownership

All project-owned scripts, tests, configs and docs are consolidated in the local `fcb-external-benchmark` Git repository, including a copy of the earlier diagnostic source. The original downloaded folders are not moved or modified. Raw data, videos, generated arrays, temporary logs and environment files are excluded. Small trained model releases and selected decoder configs can be versioned after training. No remote repository is configured and nothing is pushed.

Key commits: c7305ed (verified external benchmark), 573ec41 (frozen experiment protocol), 80312c0 (first pilot implementation), a3e4cf0 (earlier diagnostics/specification consolidated). Use a new branch at an old commit for inspection; do not reset or delete current work to inspect history.

## Remaining after this pilot

- Improve team-control transition inference, which the oracle diagnostic shows is still limiting.
- Check robustness and transfer on additional providers if time allows.
- Freeze the complete candidate pipeline, then perform one explicitly released evaluation on sealed external test matches.
- Build/run the final competition inference/export runner locally under the organiser's data restrictions.
- Validate the exact 199,319-row CSV; reproduce the file; finish the technical document and submission packaging.
- Confirm exact submission cutoff/timezone and team registration details.

Competition video inspection against predictions, target tuning on competition ground truth, and uploading restricted assets/extracts to hosted AI remain prohibited.
