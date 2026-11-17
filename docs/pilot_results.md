# First live/dead pilot — understandable result

We now have a working, reproducible model trained on four external matches. One separate external match was used to choose its decoding settings. Two further matches remain sealed: they have not been downloaded, opened or evaluated. No competition data was used in this milestone.

The model estimates whether play is live or stopped. It uses geometry, motion and missing-observation indicators; team possession during live play still comes from the fixed nearest-player reference with temporal smoothing/ownership propagation.

| Validation condition | Original reference spellF1 | Propagation-only control | Model spellF1 | Model dead-ball F1 |
|---|---:|---:|---:|---:|
| Clean observations | 0.3608 | 0.3608 | 0.4208 | 0.9205 |
| Short ball gaps | 0.3446 | 0.3446 | 0.3208 | 0.8780 |
| Long ball gaps | 0.2321 | 0.2792 | 0.2985 | 0.8006 |
| Mixed noise/dropout/slot switches | 0.3046 | 0.3046 | 0.2854 | 0.7439 |

Selected within the six predeclared choices: probability threshold 0.65 and centred probability smoothing half-width 12 frames. Mean spellF1 over the four validation conditions is approximately 0.3314, compared with 0.3105 for the original reference and 0.3223 for the propagation-only control. These conditions are correlated versions of ONE match and were used for selection, so this is not an independent test or a statistical significance claim.

The gain is uneven: clean and long-gap results improve; short-gap and mixed-noise results worsen. The pilot is versioned for learning and rollback, not declared ready for competition deployment.

The oracle diagnostic on the earlier development match explains the remaining problem. With smooth50, reference spellF1 is 0.3072, perfect external stoppage information gives 0.4613, while correcting team ownership on live frames gives 0.7835. Neither oracle result is achievable performance. Both are error-decomposition diagnostics, and team transitions clearly deserve further work.

## What is finished

- Reliable data conversion, official reference baselines and official metric integration.
- Frozen external split, deterministic corruption scenarios and model-selection objective.
- Synthetic ATD/IDSSE input and feature equivalence; no competition ATD was read.
- First logistic live/dead model, validation comparison and separate ownership-propagation control.
- Local Git history covering all project code/configs/docs and the small trained checkpoint.
- 44 main tests plus 10 input diagnostic tests passed. The complete second fit/validation run reproduced fitted parameters, selected decoder and all validation metrics exactly. Metadata and prepared-manifest hashes now join array hashes in input provenance.

## What remains

1. Improve team-control transition reasoning and reduce fragility under observation corruption.
2. Freeze a complete candidate, then explicitly release a one-time evaluation on the sealed external tests.
3. Run final competition inference locally, preserving data restrictions and exact frame alignment.
4. Validate/reproduce the final CSV; finish technical documentation and submission.

Code commit for the reproduced fit and selected model release digest are in `docs/pilot_reproduction_check.json`. No remote Git repository is configured. `models/live_dead_v1.json` is a small external-data model checkpoint; raw datasets/videos/generated frame arrays remain outside Git.
