# FCB local input diagnostics

Python 3.9+; standard library only. This is an input-quality audit, not a possession model or accuracy evaluation.

## Run

From this directory in PowerShell:

```powershell
python -B diagnose.py --data-dir <data-dir> --out "local-reports\tactical.json"
python -B -m unittest -v
```

Use `--source broadcast` to audit its separate timeline. Smart Tagging is optional; its absence is recorded as null. Neither video nor homography is opened. Frame counts default to the published tactical/broadcast counts; `--expected-frames` and `--fps` support synthetic or external fixtures. They are format settings, not fitted parameters.

## What it checks

- ATD IDs are exactly 1 through the expected count, in order; malformed, missing, duplicate and out-of-range IDs are counted separately.
- Observed x/y must both be finite. Counts, missing-ball run lengths, player counts, per-team observations and positions outside the normalized pitch are measured. Outside-pitch positions are reported, never automatically discarded.
- Metadata team membership and slot count are checked; slots are not player identities.
- Smart Tagging intervals use rounded seconds-to-frames, half-open boundaries and clipping. Coverage is the union of intervals, not their summed durations. Unknown team tags are counted separately. Opposing team tags can legitimately overlap, especially attacking versus defending phases.

Statistics cover the ENTIRE recording, including pre-match, halftime and post-match. Missing rows contribute to missing-ball runs. A missing observation is never treated as dead ball. Malformed coordinate strings are treated as unobserved; malformed row structure is separately reported.

Exit 0 means tracking alignment/slot checks pass; 2 means a report was written but those checks failed; 1 means a file/metadata/XML failure. Tag problems are report fields, not proof of invalid tracking. No data quality result measures possession accuracy.

## Competition restrictions

No network calls; no ground-truth access; no video reads; no output/video comparison; no training, fitting, calibration or parameter selection. All tests generate fictional data. Reports contain aggregate information derived from the restricted match: keep them local and do not upload them, or match assets/extracts, to hosted AI services. Only source code, synthetic tests and documentation may be shared from this package.

Next: use this audit to verify input handling. Train/tune any later possession model on appropriately licensed external data, never this match's video or possession ground truth.
