# Decoder-only experiment: keep the released v2 candidate

Nine owner/dead-live transition-cost pairs were fixed in configs/decoder_v1.json and committed in b008ab6 before scoring. The temporal_v2 network, scaling, features and weights were unchanged. All candidates were evaluated on the same nine observation conditions of external validation match J03WQQ. Previously evaluated test matches and competition assets were not read by this experiment.

## Predeclared gate and outcome

Promotion required at least .01 original-four mean spellF1 gain, at least .02 mean sub-two-second IoU-.5 recall gain, and no condition dropping by more than .02. These are release decision thresholds, not statistical significance claims. Ranking among eligible candidates was also fixed before evaluation.

No candidate passes all gates. Keep uniform costs 10/10 and the existing v2 submission candidate. No thresholds were loosened, no further grid was run, and no new model was trained.

| Owner/dead cost | Original-four spellF1 mean | Short-spell recall mean | Worst condition delta vs v2 |
|---|---:|---:|---:|
| 10/10 (v2) | .57418 | .07673 | .00000 |
| 2/5 | .57506 | .10644 | -.02903 |
| 2/10 | .57880 | .10644 | -.02266 |
| 2/15 | .58102 | .10644 | -.02166 |
| 5/5 | .57250 | .07921 | -.02163 |
| 5/10 | .57666 | .07921 | -.01468 |
| 5/15 | .57926 | .07921 | -.01353 |
| 10/5 | .57010 | .07673 | -.01051 |
| 10/15 | .57617 | .07673 | -.00442 |

The highest primary mean, 2/15, improves .57418 to .58102 (only .00684), while combined-stress spellF1 drops .42220 to .40054. Its mean short-spell recall rises by .02970, but the overall and robustness gates fail. 5/15 avoids a substantial stress drop but improves the primary mean only .00508 and short recall only .00248. Retaining v2 is the predeclared result, not a claim that every alternative is worse on every metric.

## What the diagnostic tells us

Lowering owner-change costs can recover some short episodes but also admit noisy handovers. The tested two-cost decoder does not deliver a clear balanced improvement. Do not proceed to larger networks or repeated threshold searches merely to manufacture a release.

Of the 101 short reference spells on clean validation, 1 lasts one frame, 5 last 2–4 frames, 8 last 5–12 frames, 23 last 13–24 frames and 64 last 25–49 frames. Fifty are immediately flanked by the other same team. These are automatic external-label diagnostics, not proof that an episode is a deflection or controlled possession; the DFL/Metrica convention mismatch remains unresolved.

## Verification and preserved release

Five decoder tests and three selection tests first failed, then passed. Full main suite: 90/90; input diagnostics: 10/10. Tests include exhaustive maximum-score agreement, exact uniform-decoder equivalence, separate costs, sequence isolation, team exchange and invalid inputs. A fresh read-only reviewer independently ran the eight focused tests and recomputed the saved selection decision exactly, finding no material defect.

models/temporal_v2.pt retains SHA256 85f2006b010672063758259d367428f8e4a73706251e8842ac0f993f0b7782d7. No default exporter, selected submission config, checkpoint, competition CSV or prior final result was changed. No unnecessary competition rerun was performed.

The added experiment code expands the working tree's deployment inventory. Historical frozen final evaluation should be replayed from commit 6b3dfb6 or FCB_Robust_Final_Checkpoint.zip; its freeze manifest was not rewritten to disguise later development. Normal v2 submission inference remains unchanged. No new independent final evidence is claimed.

## Nonblocking audit notes

- Future experiment reports can retain the metric-source hash and runtime versions directly instead of relying on the pinned project context.
- Explicit threshold-equality and ranking-tie tests would strengthen the existing gate tests.
- End-of-run split and data-fingerprint rechecks could supplement the implemented model/config/scenario/source checks. No concurrent input change was observed, and the completed experiment decision was independently verified.

Recommendation: retain the measured, reproducible v2 release. Further modelling should be driven by genuinely new external evidence or a validated additional signal; this bounded decoder experiment is complete.
