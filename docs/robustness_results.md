# External robustness milestone — development evidence

This report covers external IDSSE data only. No competition score has been measured.

## Frozen neural v1

| Validation condition | Ball missing | Neural spellF1 | Logistic pilot | Pilot + Viterbi control |
|---|---:|---:|---:|---:|
| Clean | 0% | .5822 | .4208 | .4068 |
| Short occlusion | 10% | .5746 | .3208 | .3797 |
| Long occlusion | 25% | .5068 | .2985 | .3168 |
| Mixed noise | 20% | .5307 | .2854 | .3435 |
| Irregular 40% | 40% | .4130 | .2742 | .2663 |
| Irregular 60% | 60% | .3057 | .2309 | .1620 |
| Crowded-play proxy | 40% | .4515 | .2557 | .2007 |
| Duplicate tracks | 0% | .4874 | .4435 | .4678 |
| Combined stress | 40% | .3148 | .2278 | .3034 |

All conditions are correlated versions of development validation match J03WQQ. Masks are observation-only simulations, not a claim to reproduce the competition's occlusion process. Crowded masking biases interval starts using local player density; it is not an aerial-ball detector. Duplicates copy sampled team slots rather than synthesising all possible tracker failures. Exact seeds and policies are fixed in configs/robustness_v2.json, committed before scoring.

The Viterbi control converts logistic dead probability plus propagated nearest-player ownership into three emissions and uses the same transition penalty 5. It is weaker than the neural system on every condition. This supports the neural emission model contributing beyond the pilot; it is not an architecture-only ablation or a probability calibration comparison.

## Episode error analysis

Clean validation has 401 true possession spells and 286 predicted spells. At official IoU .5, recall by reference duration is:

| Reference duration | Recovered | Total | Recall |
|---|---:|---:|---:|
| <2 seconds | 10 | 101 | 9.9% |
| 2–5 seconds | 46 | 111 | 41.4% |
| 5–15 seconds | 94 | 129 | 72.9% |
| >=15 seconds | 54 | 60 | 90.0% |

Duration-bin recall and overlap-based merge counts are supplementary diagnostics, not alternative headline metrics or proof of why a possession was missed. Short-spell suppression, heavy missingness and duplicate-track sensitivity are all measured weaknesses. External label conventions may themselves include micro-spells differing from the competition's controlled-possession semantics.

## Bounded response

Train one temporal_v2 candidate with harder observation augmentation, retaining the v1 architecture and eight-epoch budget. Keep decoder sweep 0/2/5/10 and selection primary on the original four conditions. Promotion requires beating v1's original four-condition mean with no original or added condition regressing by more than .02. Preserve v1 checkpoint and exporter config regardless of the outcome. Do not also add deduplication or change decoder semantics in this experiment.

## Completed bounded experiment

One eight-epoch run selected transition penalty 10 from the original 0/2/5/10 sweep. Architecture and 40-feature interface are unchanged. All nine conditions improved, so the candidate passes the fixed gate and is selected before final-test access.

| Condition | Frozen v1 | Selected v2 |
|---|---:|---:|
| Clean | .5822 | .6070 |
| Short occlusion | .5746 | .6114 |
| Long occlusion | .5068 | .5377 |
| Mixed noise | .5307 | .5406 |
| Original four mean | .5486 | .5742 |
| Irregular 40% | .4130 | .4585 |
| Irregular 60% | .3057 | .3364 |
| Crowded proxy | .4515 | .4578 |
| Duplicate tracks | .4874 | .6011 |
| Combined stress | .3148 | .4222 |

This improves robustness but does not solve heavy occlusion. The raw v2 network remains weak without temporal decoding. Selection still uses one reused validation match. Training quantities: four external matches, 571,928 frames (6.355 hours at 25 Hz), H=167,241, A=147,819, D=256,868. Supervision is provider-supplied DFL ownership/status; exact annotator/process provenance is not established by this project and no equivalence with Metrica control conventions is asserted.

## Frozen final external evaluation

Model, decoder and stress protocol were fixed before acquiring J03WN1/J03WR9. No model changes or selection followed these scores. These matches are now consumed final tests and must not be called untouched in a future iteration.

| Condition | J03WN1 selected | J03WN1 pilot | J03WR9 selected | J03WR9 pilot |
|---|---:|---:|---:|---:|
| Clean | .5104 | .3825 | .5650 | .3900 |
| Irregular 40% | .3584 | .2450 | .4230 | .2846 |
| Irregular 60% | .2811 | .1991 | .3657 | .2244 |
| Combined stress | .3153 | .1773 | .4132 | .2053 |

The selected model beats the logistic pilot in all eight comparisons. This is independent external evidence for the selected system, with only two final matches from the same provider, potentially repeated teams and synthetic stress. It does not establish competition ranking or transfer to all tracking/annotation domains. Raw per-match official evaluations and input provenance remain in local reports/final_evaluation_v1/results.json.

## Release verification and recommendation

82 main tests and 10 input diagnostics pass. A fresh tracked-file checkout passed the complete deployment-hash check and all 82 main tests. Both working-checkout and fresh-checkout local inference passed official format validation and produced byte-identical CSVs. Competition-derived files and logs remain local and excluded from the source/model checkpoint. The rehearsal used the same Windows machine and existing environment; independent-machine reproduction remains a later release check.

Keep the 20,035-parameter, identity-free feature/TCN/Viterbi architecture. Do not add a larger network, a new tracker, broadcast alignment or an LLM inference layer for this release. Main unresolved modelling weaknesses are short-spell suppression, imperfect possession-control semantics, provider/domain shift and information loss through extended occlusions. The next small modelling experiment, if a separate iteration is chosen, should test separate owner-change and dead/live transition costs using the same frozen emission model, with a small predeclared validation sweep and fresh final evidence for any newly selected system. Do not tune possession counts to competition statistics or treat these final matches as untouched again. Smart Tagging is a possible evidence source only with a credible external validation strategy.
