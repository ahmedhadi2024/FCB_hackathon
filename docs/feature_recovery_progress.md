# Execution ledger: 2026-11-08-feature-recovery-and-release.md

- Base: 3690fad. Dedicated branch feat/ball-independent-recovery.
- Ruling: use the existing feature checkout and a separate protected release directory rather than another worktree; this preserves prepared local data and avoids duplicated environments. Only this executor edits the branch.
- Ruling: skill shell helper scripts are replaced by this native Windows ledger and saved command logs; equivalent progress is retained without shell-porting work.
- Preflight: feature extractor -> paired runner -> candidate exporter share versioned names/scalers. Legacy inference remains isolated.
- Ruling: the plan mistakenly calls clean/irregular40/irregular60/mixed_stress the historical original four. Actual original four are clean/short_occlusion/long_occlusion/mixed_noise. Preserve the actual original four as primary; all irregular conditions remain stress gates. Otherwise the historical comparison claim would be false.
- Task 1: protected release copied from prior verified clean tracked checkout; model SHA256 verified. Fresh environment rehearsal and synthetic tests pending.
- Task 1: complete. Protected release suite passed; fresh install blocked by network policy. Existing environment remains usable.
- Task 2: protocol frozen with original primary four and 14 stress/audit conditions including four additional seeds. No new immediately usable independent source established; existing five-match CV used. Official source https://github.com/spoho-datascience/idsse-data documents seven matches, already consumed in this project.
- Task 3: complete. Five new feature tests failed before implementation, then passed; legacy prefix and sequence validity verified.
- Review: validation fingerprints and official metric/runtime provenance now checked on resume; regression RED to GREEN. Synthetic orchestration test exercises actual runner role boundaries.

- Verification: current main suite 102 passed; diagnostics 10 passed; protected historical v2 release 82 passed.
- Task 4: INCOMPLETE. Training launch reached `fit A_baseline`, but no epoch, checkpoint or scored report is saved. Unified exec session 6210 became unavailable after automatic review rejected an incidental external telemetry connection with unspecified payload/untrusted destination. No retry or indirect bypass attempted. Local process inventory is unavailable; do not assume training completed or relaunch blindly.
- Task 5: protected v2 release manifest saved locally; no candidate was promoted and no new competition inference ran. Experimental six-fit comparison and final selection remain blocked pending a reliable local execution connection.
- Ruling: do not request blanket telemetry authorization for an unknown payload. Preserve logs and source, continue local read-only checks, and report the execution blocker accurately.

## Recovery step 1: local execution verification

- Rechecked original log: unchanged `fit A_baseline`; no model/report artifacts. No Python/Conda process visible through Get-Process after the smoke job exited. This supports treating the earlier local job as inactive; it is not an inventory of hidden/remote workers.
- Desktop thread terminal diagnostics failed with a Codex app tool error. No destination/payload/source detail for the historical telemetry rejection is available, so its cause remains unidentified. No rejected action was retried.
- A materially restricted synthetic smoke check completed in 3.883 seconds in the existing environment: tiny TemporalNet, one AdamW optimizer step, checkpoint save/reload, bit-identical predictions. Python audit hook rejected socket connection/DNS/sendto events and child-process execution for this check. No network event was observed; three subprocess attempts were blocked while the test still completed. This is a Python-level diagnostic restriction, not a guarantee about host/native traffic.
- Evidence: ../work/feature-recovery/local-smoke.json and local-smoke.pt. Python 3.13.13, NumPy 2.5.4, CPU PyTorch 2.15.0+cpu.
- Recovery Task 1 outcome: local synthetic training/save/reload is usable; the historical telemetry root cause is unresolved. Full training has NOT restarted. Next is Task 2 (progress logging and checkpoint-before-validation), then named protocol validation before expensive fits. Any recurrence of the rejected external action must stop execution rather than trigger an alternate launch.

## Recovery Task 2: complete

- Added preparation/feature/epoch/checkpoint/validation progress with elapsed times and feature-array byte counts; no raw tracking/labels are logged.
- Completed eight-epoch fits are atomically saved before validation with run identity, protocol, sources/runtime/scorer and train-input fingerprints. Trained-only artifacts resume evaluation; invalid/partial/mismatched artifacts are preserved and rejected. Completed reports still require validation fingerprints and model hash.
- Validation scenarios are generated lazily and released after each score, rather than retaining all 18 observation variants at once.
- RED: simulated evaluation interruption demonstrated missing checkpoint before the implementation; atomic-save API was absent. GREEN: resume completes six total fits across interruption and recovery, not seven; failed save publishes no checkpoint. Additional tests reject mismatched source/incomplete epoch and exercise lazy scenario creation.
- Verification: 106 main tests passed (4.427 seconds), 10 diagnostics passed (.062 seconds). Logs: durable-red.log, durable-green.log, durable-suite.log, durable-diagnostics.log under ../work/feature-recovery.
- No external training was restarted, no competition input opened, no selected model changed. Next: recovery Task 3, name-based protocol and promotion-gate checks before expensive fits.

## Recovery Task 3: complete

- Protocol now names primary/missing-ball/stress families, exact gate thresholds, extra evaluation seeds and output order; previous protocol hash preserved. Freeze record: docs/feature_recovery_protocol_freeze.json.
- Gate input is a complete named baseline/candidate match/scenario/fold record set. No positional grouping remains; duplicate/missing/unknown/nonfinite/wrong-fold cells are rejected. Protocol checks require unique folds and exactly one held-out role per match.
- RED: new named API/protocol validation tests failed before implementation. GREEN: order-invariance and exact unrounded threshold tests pass. Direct real-fit tracing verifies paired windows, labels/masks, scaler legacy prefix and actual analytic class weights; existing orchestration tests exclude held-out matches from fit cache.
- Fresh source review found no material blockers; its pairing-evidence gap was addressed by the direct fit test.
- Verification: 110 main tests passed (4.614 seconds), 10 diagnostic tests passed (.080 seconds); 15 focused runner tests pass. Logs named-red.log, named-green.log, pairing-check.log, named-suite.log and named-diagnostics.log.
- No external fits restarted or scores generated in this step; all numeric score fixtures were synthetic. Next: recovery Task 4, six sequential paired external fits under the frozen protocol. V2 stays selected pending complete gates.

## Recovery Task 4: complete; candidate rejected

- Verified no visible old Python job and frozen protocol/source/scorer hashes before launch. Ran six sequential paired fits under local-only Python audit restrictions; completed normally.
- All six reports/checkpoints saved, eight epochs each; all 180 named scores present. Recomputed gates from named cells independently; decision matches saved output. Verified model hashes and source/protocol identities. V2 checkpoint hash unchanged.
- Primary CV mean: retrained baseline 0.509309, geometry 0.499780, delta -0.009529. Missing-ball delta +0.047993, positive on 5/5 matches. Primary improves on only 1/5; all folds negative. Home-visibility stress loss exceeds .02.
- Gate decision FAIL. Retain v2; no refit, promotion or further automatic search. Full recorded result: docs/feature_recovery_result.md; named evidence docs/feature_recovery_decision.json. No competition data/inference used.
- Task 5 skipped by the declared failure branch. Next task is Task 6: final retained-v2 release checkpoint and reproduction documentation.
