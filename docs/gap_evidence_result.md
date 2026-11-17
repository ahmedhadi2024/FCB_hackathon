# Gap evidence experiment result

Decision: reject the tracking-endpoint challenger. Retain selected v2. No automatic Smart Tagging performance claim follows from this experiment.

The architecture inspection found that temporal.decode already performs offline three-state Viterbi with full traceback and sequence isolation. It already carries competing paths and uses future emissions. A new hypothesis tracker was therefore unnecessary; the experiment reused this decoder and changed only optional evidence.

The single frozen candidate supplied weak same-team evidence inside gaps of at most 125 frames when adjacent observed endpoints agreed, were within 2m of the ball and had a 1m distance advantage. Candidate and deliberately inverted evidence were generated exclusively from observations. Possession targets were used only afterward for scoring.

| External development scenario | Frozen v2 spellF1 | Endpoint candidate spellF1 |
|---|---:|---:|
| Clean | 0.607038 | 0.607038 |
| Short occlusion | 0.611387 | 0.611387 |
| Long occlusion | 0.537705 | 0.537705 |
| Mixed noise | 0.540588 | 0.540588 |

Mean change: 0.000000. Required improvement: 0.01, with no scenario drop over 0.02. Gate failed. Identical spell scores do not establish identical output labels. Inverted evidence changed long-occlusion spellF1 to 0.533135 and mixed-noise spellF1 to 0.542353; this is a sensitivity check, not a proposed alternative.

Validation used the already reused J03WQQ development match and original four scenarios, frozen temporal_v2.pt and transition penalty 10. No training, independent final-test evaluation or competition inference occurred. Full external report: reports/gap_evidence_v1.json (ignored locally). Command: python run_gap_evidence.py; refuses to overwrite the existing report.

Verification: gap tests failed before module implementation, then all 121 main tests passed. Tests cover agreeing endpoints, disagreement, ambiguous nearest teams, all-missing ball, disabled bridging and sequence boundaries. The run completed under the Python local-only audit hook used in previous experiments. This hook is not a host-level network sandbox.

Review: author review only. The earlier independent reviewer could not run because of a service usage limit. The adapter assumes validated canonical observations, as supplied by load_match/corrupt; it does not revalidate the complete schema internally.

Ruling: reuse the existing offline decoder rather than implement another path tracker; duplicating Viterbi would not add the proposed capability. Cost if wrong: a richer latent state model may have benefits not explored here.

Ruling: stop this endpoint candidate after its failed gate; do not increase the evidence weight against the same scores. Cost: a stronger variant could improve, but searching this reused validation match would weaken the evidence.

Next useful work remains the actual automatic-tag candidate. This negative tracking proxy does not settle whether Smart Tags contain complementary information. Keep that candidate separate and report its evidence limits explicitly.


Provenance correction: this page describes the historical v1 run. The corrected runner now targets a separate v2 report with pinned scenario definitions and recorded source/config/input fingerprints. No v2 experiment has been run as part of that correction, and no v1 scores have been rewritten.
