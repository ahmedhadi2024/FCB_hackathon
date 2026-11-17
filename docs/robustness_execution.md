# Robustness execution ledger

Plan: docs/superpowers/plans/2026-11-08-robustness-and-final-evaluation.md

Existing release a88e6e0 preserved. Work on feat/robustness-final-evaluation; source-only local checkpoint and existing dependencies retained.

Ruling: work in the existing writable checkout on a new feature branch. This preserves the released commits and avoids duplicating external data or changing path-bound configs.

Pre-flight: error analysis consumes inference labels and official metrics; corruption consumes observations only; revised training must write a new checkpoint; final evaluation needs a separate loader without weakening development guards. Competition exporter remains unchanged unless a new candidate is promoted.

Task 1 complete: five new episode tests observed failing before implementation, then passed; external four-scenario analysis completed without retraining. Whole main suite 74/74 passed in 10.881 seconds. Clean reference spell recall at IoU .5: <2s 10/101, 2-5s 46/111, 5-15s 94/129, >=15s 54/60.

Task 2: five new corruption tests observed failing before implementation. Exact scenario config committed in cd10580 before any stress scores. The suite includes source immutability, deterministic masks, per-sequence coverage, duplicate team/coordinate preservation and legacy equivalence.

Task 2 benchmark complete: neural spellF1 clean .5822, irregular40 .4130, irregular60 .3057, crowded40 .4515, duplicates .4874, mixed stress .3148. The logistic + same-penalty Viterbi control is weaker than neural on all nine conditions; the improvement is not solely attaching that decoder to the pilot.

Task 3 decision: one harder-augmentation training experiment, same architecture/budget/seed, all nine fixed conditions. Decoder sweep remains 0/2/5/10, selection primary is the original four-scenario mean. Promotion additionally requires no condition losing more than .02 against frozen v1. New model/report paths are temporal_v2; v1 files remain untouched. Do not combine feature deduplication or a new decoder with this experiment.

Task 3 complete: eight epochs trained once. Selected penalty 10; primary mean .5742 versus frozen v1 .5486. Every original and added condition improves; promotion gate passes. The architecture/features remain unchanged. New selected submission config points to temporal_v2; original neural/logistic releases remain unchanged.

Final review fixed P1 release-overwrite guard: existing model/report outputs now fail before training. Default-training sentinel test observed failing, then passing. A clean release cannot be retrained in place accidentally; use a fresh experiment output name.

Final review fixed P1 incomplete final-freeze guard: require the complete deployment inventory, including entrypoints, all package/vendor files, selected config, split/stress configs, pilot model and dependency lock; verify installed pinned versions. Incomplete-manifest/component-change tests failed before the fix, then passed.

Final review fixed P2 promotion-reference provenance: require matching frozen v1 hash, scenario config and current validation input fingerprints before training and selection, and reject benchmark replacement during the run. Regression failed before the fix. Applied these assertions retrospectively to the completed v2 run; they pass. Numerical training/selection was not changed or repeated; its recorded source hashes remain those of the original training run.

Whole suite after guard fixes: 81 main tests passed in 11.544 seconds; 10 diagnostic tests also passed. Final tests remain unaccessed at this point.

Existing metric caveat documented: official transition diagnostics assume widely separated valid runs; one-X sequence separators can permit cross-sequence transition pairing. Possession/headline scores remain correctly separated; the new supplementary transition analysis matches within each valid run. No ranking-metric implementation was rewritten.

Ruling: pin LF checkout endings for project text while preserving vendored official bytes and binary model files. Git core.autocrlf=true would otherwise change source/config checksums on clean checkout. Current project source/config files are already LF; this does not change model mathematics or stored training configuration hashes.

Checkout verification refinement before any final-data acquisition: preserve the legacy logistic JSON model's original CRLF bytes explicitly, and normalize the newly generated selected config to LF. Refresh the freeze manifest for these transport-only changes; selection, weights and test protocol are unchanged and the sealed matches remain unaccessed.

Final acquisition routing: Python DNS failed for both Hugging Face and alternate Figshare API. The native Node runtime can access the same pinned Hugging Face revision. Its public listing shows J03WN1 uses competition division 000001, while J03WR9 uses 000002. Correct the downloader to resolve unique listed names rather than assume division, verified with a failing fictional test. Add a local verification/preparation mode for checksummed files obtained through the working native download route. Refresh the source freeze before acquiring any raw final files; no model/decoder/selection changes follow from this source metadata.

Fresh checkout revealed vendor/official/source_manifest.json working bytes differed from its older indexed newline form. Re-index only that metadata file with its existing -text attribute, retaining all working bytes and the unchanged official scorer. No metric code or model changes.

Task 4 complete: verified both real external matches against upstream size and Git/LFS hashes, internal XML identity and canonical timeline checks; evaluated exactly clean/irregular40/irregular60/mixed_stress once after freeze. Selected model beats pilot in all eight comparisons. Clean scores .5104/.5650. No model changes after final scoring; test matches are consumed.

Task 5 complete: fresh tracked-file checkout passes full frozen deployment hashes and 82 main tests. 10 diagnostics pass. Promoted local full-match export and fresh-checkout export each pass official format validation and their CSV bytes match exactly. No competition contents or output diagnostics were displayed. Updated technical document and external-results report. Local Git clone was blocked by the host's MSYS process restrictions; git checkout-index provided a clean checkout without that helper. Same machine/environment only; no claim of independent-machine validation.

Follow-up user requested architectural assessment before further modelling: no additional training or architecture work started. Recommendation is retain the small selected release, acknowledge weak short-spell recall and heavy-occlusion/domain-transfer limits, and consider only a separately planned decoder-cost experiment if further iteration is desired. No GitHub/team details are needed to finish this technical milestone.
