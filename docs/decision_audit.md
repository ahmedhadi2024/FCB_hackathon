> **Rules provenance update (9 November 2026):** See `docs/rules_reverification.md`. External-only/inference-only restrictions below describe historical project scope and an unverified claimed organiser clarification; they are not a complete statement of the verified public rules. Public labels still cannot be training/tuning data. Historical experiment choices/results remain unchanged.

# Decision audit and new stress results — 8 November 2026

## Recommendation

Preserve v2 as the working submission candidate. The next modelling experiment should recover ball-independent team geometry, using the existing small temporal network. Do not spend another round searching decoder costs or scaling up the network. This is a proposed next experiment, not a claim that new features will necessarily improve possession inference.

The earlier description that “only delivery remains” was too strong. Engineering readiness and research confidence are separate. A working reproducible CSV does not establish competitive accuracy. The earlier 85% completion estimate was subjective and should be replaced by explicit gates below.

## Fresh checks actually run

- Main suite: 90 tests passed in 5.264 seconds. Input diagnostics: 10 passed in .116 seconds.
- Ten additional synthetic end-to-end audit checks with the actual released checkpoint pass: slot permutation, team exchange, pitch reflection, empty-slot padding, time-origin invariance, chunking equivalence, period isolation and rejection of three invalid observation contracts.
- Additional external stress scenarios were run on J03WQQ only. The model and decoder were fixed; no parameter selection, retraining, competition data access or new final-match evaluation occurred.

### A demonstrated representation limitation

When the ball is absent, current team-specific distance/radius features become zero. Only observed-player counts remain in those team channels; the existing player-centroid speed pools both teams.

A synthetic probe changed the two teams' spatial layouts while retaining equal team counts and the same overall set of positions. With ball observations missing, the two scenes produced exactly identical 40-dimensional features (maximum difference 0). The team-symmetrized network returned equal H/A probabilities (maximum difference 0). The final probe used static integer coordinates to avoid irrelevant floating-point summation-order differences in the pooled centroid.

This proves an information bottleneck. It does not prove that formation alone identifies possession, that all missing-ball frames are unknowable, or that the model ignores context near visible observations. The temporal network and decoder still carry surrounding evidence, but cannot reconstruct distinctions already discarded by the feature extractor.

### New fixed-model stress results

All values are official external-validation spellF1. These are deliberately artificial sensitivity probes, not measured competition error rates or independent test results.

| Scenario | spellF1 | Interpretation |
|---|---:|---|
| Clean replay | .6070 | Matches the existing v2 clean result |
| Irregular 40% missing, new seed 301 | .4583 | Similar to original .4585 |
| Irregular 40% missing, new seed 302 | .4498 | Modest realization sensitivity |
| Irregular 60% missing, new seed 303 | .3687 | Above original .3364; seed changes do not uniformly worsen performance |
| Combined stress, new seed 304 | .4201 | Similar to original .4222 |
| False stationary ball | .4500 | Ball set to observed pitch centre for 10-second bursts every 60 seconds |
| Ball-only constant +3 m x bias | .2658 | Strong sensitivity to a persistently wrong ball track; players remain unshifted |
| Three extra home-team tracks | .5774 | Additional asymmetric duplication of highly observed slots |
| Home-team visibility loss | .4999 | All home players hidden in 10-second bursts every 60 seconds |
| No ball in first period | .3640 | Extreme information-loss test; second period remains unmodified |

The +3 m probe shifts the ball only. It is not a common coordinate shift of all players and ball, and its severity does not establish that this error occurs in the competition. The burst corruption begins 10 seconds into each observation sequence and is independent of possession targets. True labels are used only for evaluation.

New-seed results give no evidence of catastrophic dependence on the original random seed. They do not validate new missingness mechanisms. Incorrect but observed ball positions and asymmetric player visibility deserve more attention than another seed sweep.

Maximum softmax values were retained only as diagnostics. They are not calibrated confidence scores. No calibration or abstention policy was introduced from these results.

## Decisions: keep, correct, or watch

| Decision | Judgment | Action |
|---|---|---|
| Small CPU TCN with H/A/D output | Keep | Its size is not the demonstrated bottleneck |
| Avoid stable player identities | Keep | Correct for unstable track slots; retain permutation tests |
| External-only fitting and selection | Keep | Preserve competition data restrictions and target-free inference |
| Explicit sequence boundaries and metric reuse | Keep | Fresh invariance checks and unchanged official scoring support these choices |
| Temporal decoding | Keep, with limitations | It helps coherent spells but cannot create missing spatial evidence |
| Stop the two-cost decoder experiment after its fixed gate failed | Keep | A small mean gain with worse stress behavior is insufficient; do not loosen gates afterwards |
| Initial clean external data plus light synthetic gaps | Corrected late | v2 improved this, but the false-observed-ball mechanism was not tested early enough |
| Aggressive geometry aggregation around the ball | Revisit first | It discards most team geometry precisely when the ball disappears |
| One reused validation match | Evidence limit | Use paired match-level evaluation for further feature decisions; frame counts are not independent samples |
| One fixed corruption realization per training condition | Limited support, not leakage | Use separate train/test seeds and report generator assumptions; avoid claiming nine conditions cover real tracking failures |
| Consume the final reserve after freezing v2 | Valid for v2 | Those results still support v2, but do not lend independent-test credibility to future candidates |
| Root-wide freeze inventory | Safe but awkward | Isolate a production release so later experiments do not break its historical replay |
| Defer Smart Tagging production fusion | Reasonable release decision | Lack of validated benefit is not evidence of no benefit. A proxy study can support limited conclusions; require honest uncertainty |
| Describe readiness with “85%” or “only delivery left” | Regrettable communication | Replace with software, model-evidence and delivery gates; no winning probability is established |

## What we could regret later

1. Improving a metric on the same provider while misunderstanding controlled-possession semantics.
2. Treating a finite, normalized softmax output as reliable confidence, especially because weighted training loss and decoder cost interact with emission scale.
3. Testing only missing observations while real errors include confidently wrong observations and asymmetric visibility.
4. Presenting future selected candidates as independently tested after reusing known matches.
5. Having several checkpoint archives and research branches without one unmistakable production release and reproduction command.

These are risks to manage, not proof of poor competition performance. The current model's official competition score and other teams' performance remain unknown.

## Next plan in order

1. Isolate and label the existing v2 production release; keep its working CSV and fallback models. Confirm the exact runnable release and dependency-install limitations. No GitHub account or publication is required for this local step.
2. Freeze the evaluation plan before further fitting. Prefer reserving genuinely additional external matches if available. Otherwise use explicit paired grouped development evaluation on existing data and disclose that it is not a fresh independent final test.
3. Run one feature-recovery experiment: add a small number of team-centre/spread/motion measurements that exist without ball coordinates. Keep the original feature contract and v2 checkpoint untouched. Use the same network, bounded budget, paired baseline and candidate training, and distinct train/test corruption seeds.
4. Keep the new hostile-input cases as fixed audits. Do not patch predictions to hit a desired possession count or introduce a competition-specific displacement correction. Consider augmentation for false observed balls only as a separate evidence-driven follow-up, not bundled with the feature change.
5. Promote only under predeclared multi-match and stress gates. Report the evidence level honestly; unchanged v2 remains available if the experiment fails. Validate/reproduce any selected export locally before final delivery.

Detailed implementation proposal: `docs/superpowers/plans/2026-11-08-feature-recovery-and-release.md`. No new model or production change was implemented in this audit.
