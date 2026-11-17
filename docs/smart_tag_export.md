# Experimental Smart Tag submission integration

The frozen Smart Tagging prototype is connected to the existing temporal exporter through configs/submission_smart_tag_experimental.json. The selected configuration remains unchanged. Run locally:

    python run_submission.py --config configs/submission_smart_tag_experimental.json --data-dir <local-dataset-directory> --out local-reports/<new-candidate-name>.csv

Redirect stdout/stderr to a local log when processing restricted assets. The command refuses to overwrite an existing export or manifest. It reads tactical tracking, metadata and automatic Smart Tags; it does not read possession ground truth or video. Candidate configuration and tag file checksums join the tracking checksums in the local manifest, and inputs are checked for replacement before export.

The adapter enforces the previously fixed policy: Build Up/Progression only, 25-frame edge margin, missing-ball-only fusion, odds multiplier 1.5, unchanged dead probability, neutral unknown/conflicting evidence. An invalid or missing tag file fails the explicitly enabled candidate rather than silently producing a baseline artifact. Ordinary v2 configuration does not load tags and passes original probabilities directly to the existing decoder.

Validation: the new configured-adapter test failed before implementation; afterward all 122 main tests passed. Existing parser/fusion tests cover absent evidence, conflicts, shifts, wrong-team effects, half isolation and probability preservation. Preserving dead probability does not guarantee unchanged final dead labels after sequence decoding.

Selection decision: keep v2 selected. Format validation, repeatability and input compatibility establish operational correctness only. Competition accuracy and rank are unknown. The rejected endpoint tracking proxy is not evidence for automatic Smart Tag accuracy. There is no equivalent external automatic-tag benchmark in this work so far; this limits confidence in promoting this experimental output. No public-answer fitting or manual match-specific corrections were used.

Review: author source review; independent review remains unavailable after the earlier reviewer usage-limit failure.

Execution blocker: automatic approval review rejected local inference because of possible telemetry to an external Visual Studio endpoint with unknown payload. No retry or alternate execution was attempted. The two-export validation and repeatability check is unverified; do not treat any partial artifacts as a completed candidate.

Follow-up execution: a fresh no-profile shell inherited POWERSHELL_TELEMETRY_OPTOUT=1 and POWERSHELL_UPDATECHECK=Off before startup, with the Python local-only audit guard installed before inference imports. Automatic approval review again rejected execution, reporting a Visual Studio telemetry attempt despite those controls. The payload and emitting component remain unverified. This disproves treating the PowerShell opt-out alone as a sufficient resolution. No further inference retry or alternate execution was attempted. Export validation and repeatability remain unverified. Further work requires identifying the emitter using a data-free diagnostic or establishing enforced network isolation through an approved environment.
