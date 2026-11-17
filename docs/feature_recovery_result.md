# Feature recovery: completed experiment and decision

**Decision: retain the existing v2 release. Do not refit or promote the geometry candidate.**

Three grouped folds, six paired CPU fits, eight terminal epochs each, fixed decoder penalty 10. All 180 named variant/match/scenario scores are present. Original final matches and competition data were not used.

The comparator below is a retrained original-feature baseline within each fold, not the released v2 checkpoint. These development-CV scores are not directly comparable with the earlier single-validation-match scores or competition results.

| Match | Baseline primary spellF1 | Geometry primary spellF1 | Delta | Missing-ball delta |
|---|---:|---:|---:|---:|
| J03WMX | 0.450800 | 0.438489 | -0.012311 | +0.056668 |
| J03WOH | 0.523459 | 0.492931 | -0.030528 | +0.071592 |
| J03WOY | 0.505546 | 0.510518 | +0.004972 | +0.031189 |
| J03WPY | 0.509682 | 0.503990 | -0.005692 | +0.047113 |
| J03WQQ | 0.557056 | 0.552971 | -0.004085 | +0.033401 |

Equal-match primary mean: baseline 0.509309; geometry 0.499780; delta -0.009529.
Equal-match missing-ball delta: +0.047993, positive on all five matches.

| Gate | Pass |
|---|---|
| cell_floor | True |
| family_floor | False |
| four_matches | False |
| match_floor | False |
| mean_gain | False |
| missing_four | True |
| missing_mean | True |
| three_folds | False |
| without_best | False |

| Condition/family | Mean paired delta |
|---|---:|
| ball_offset_3m | +0.026240 |
| clean | -0.000307 |
| crowded40 | -0.003487 |
| duplicates | -0.013601 |
| false_stationary_ball | +0.002262 |
| home_extra_tracks | -0.003084 |
| home_visibility_loss | -0.024434 |
| irregular40 | +0.039062 |
| irregular40_seed301 | +0.034498 |
| irregular40_seed302 | +0.029756 |
| irregular60 | +0.059489 |
| irregular60_seed303 | +0.073089 |
| long_occlusion | -0.008139 |
| mixed_noise | -0.010149 |
| mixed_stress | -0.000988 |
| mixed_stress_seed304 | +0.015116 |
| no_ball_first_period | +0.052062 |
| short_occlusion | -0.019521 |

## Interpretation

The representation change recovered useful signal under ball loss, but its benefits did not generalize to the primary conditions consistently. Only one of five primary match deltas was positive; all three fold means fell. Home visibility stress fell by 0.024434, beyond the 0.02 limit. The experiment therefore fails the predeclared overall and consistency requirements despite improved missing-ball scores. Do not weaken gates or start another search from this result.

This does not establish that team geometry is intrinsically unhelpful or that a larger model would fix the trade-off. One paired seed per fold and five matches limit certainty. All comparisons are development evidence, without fresh independent final confirmation.

## Execution and provenance

Protocol SHA256: `74a1f52b7962f61fa5c74dea882e9cdd25eef0347d213d7701a403106a742531`. Full condition definitions and thresholds: configs/feature_recovery_v3.json. Source hashes match docs/feature_recovery_protocol_freeze.json.

| Run | Seconds including fit and validation | Scored conditions | Checkpoint SHA256 |
|---|---:|---:|---|
| A_baseline | 113.46 | 36 | `e3d9258d9373eac398ba25eb0cb24958c6a7f8344474aa49f13ebec53f3adf4e` |
| A_geometry | 148.91 | 36 | `8928ba219d9170a82ebec89808a0e600ad0be2c96e65d1b335b0a9b31abb2813` |
| B_baseline | 90.56 | 36 | `f37aef1ce4b373bb041474e50d7888f256470175b45a0df79efb3d1241287e41` |
| B_geometry | 182.30 | 36 | `87f15e44cf53b0976b4aff28c332bc32ca5d7ad1c0b14ab4b754ac87af0f855a` |
| C_baseline | 92.37 | 18 | `df5fdb263ad815f9b293ca4ce30ba269b7d22fb99619cbf7cd7ad2311e9fbeaf` |
| C_geometry | 193.26 | 18 | `1848846833b4d3e27a5481cbfba8c0cb083bc22bab6d97e7f9b1478fcb1011b4` |

Sum of six reported fit/validation times: 820.9 seconds; excludes observation-cache preparation between runs. All reports/checkpoints saved locally under reports/feature_recovery_v3 and models/feature_recovery_v3.

Run used Python-level audit restrictions on network and child processes. The rejected historical telemetry action was not retried; full comparison completed without a connection rejection. This does not establish the earlier rejection's source. Fresh dependency installation remains unverified; existing environment was used.

V2 SHA256 verified unchanged: `85f2006b010672063758259d367428f8e4a73706251e8842ac0f993f0b7782d7`. Selected submission configuration unchanged. No new competition inference ran.

## Next release step

Skip conditional final refit (recovery Task 5). Complete recovery Task 6 for the retained v2 release: verify its manifest and reproduction command, include the negative feature result in the technical account, package the selected checkpoint and preserve the fallback. Do not rerun full competition inference solely for packaging.
