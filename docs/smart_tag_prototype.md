> Historical checkpoint at commit 26fdc39. Export integration was subsequently added in 101dd4e; see [current integration status](smart_tag_export.md). Statements below about a disconnected exporter describe that earlier checkpoint. The candidate remains experimental and unselected.

# Smart Tagging prototype checkpoint

## Result

The optional evidence adapter and local input audit are implemented. The supplied tracking and tag audit completed, explicit metadata team mapping resolved, and the complete timeline check passed. Detailed match-derived results remain in ignored local-reports/smart-tag-audit-20261109.json. This is an input-contract result, not an accuracy result.

The selected v2 model and submission configuration are unchanged. The candidate is not connected to the submission exporter and has not generated a candidate submission.

## Design

Only Build Up and Progression phase teams supply evidence. One second is excluded at both interval edges. Evidence applies only when the ball is missing, in the matching half, with an explicit known team. Duplicate tags do not amplify evidence; opposing tags, unknown teams and unsupported phases provide no evidence. Home/away odds receive a fixed 1.5 multiplier while their combined mass and the dead-ball probability are preserved. The configuration records these design choices; it is not yet an exporter input or calibrated confidence.

Preserving dead-ball probability does not guarantee unchanged final dead-ball labels after decoding: changing competing home/away evidence can change the selected path.

## Verification and review

The original parser implementation passed 115 main-suite tests. Additional synthetic stress tests cover absent and shifted tags, unsupported phases, wrong-team evidence, unchanged inputs, and integer one-hot probabilities. Wrong tags can flip a close home/away decision; bounding the weight does not prove safety or improvement.

Direct review found and fixed an integer-array fusion crash and audit CLI failure reporting. Reproduction tests failed before the fixes. A separate reviewer was attempted but could not run because of a service usage limit; this checkpoint has author review only.

## Plan progress and decisions

Task 1: parser and local audit complete. Task 2: conservative prototype and fixed candidate design complete. Task 3: synthetic stress checks implemented; external proxy evaluation is still pending. Tasks 4 and 5: candidate inference and final release decision remain pending.

Ruling: retain v2 as selected and keep fusion disconnected from the exporter until the evidence experiment is assessed. Synthetic correctness and input compatibility do not establish a possession-score gain. Cost: any real benefit from tags is deferred until evaluated.

Next implementation step: examine development-only external event inputs for an independently constructed proxy, then test sensitivity to incorrect and shifted evidence. Do not derive proxy tags from possession targets, tune against public competition answers, or claim external proxies measure automatic tag accuracy. If no defensible proxy exists, document that limitation before deciding whether to generate a separate experimental candidate.

No new training, ground-truth access, video processing, public upload, or dependency installation was needed for this checkpoint.

Final verification: python -m unittest discover -s tests passed all 119 tests after the review fixes.
