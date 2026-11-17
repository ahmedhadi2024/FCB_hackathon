# Review of recent experiments and possession terminology

Scope: git diff a5de105...9159bcb. Two independent source-only review axes. No restricted assets or match-derived reports were supplied to reviewers. Specs: supplied-evidence-next-step, gap_evidence_plan, possession_duration_plan. No repository coding-standards or issue-tracker configuration was found; local specs were sufficient, so no tracker setup was needed.

## Standards

No documented hard violations. Two heuristic findings:

1. Duplicated experiment orchestration: run_gap_evidence.py:16-43 and run_possession_duration.py:15-40 repeat loading, inference, scoring, output protection and gates. Both use positional scenario selection. A common small experiment helper plus explicit scenario identities would reduce drift; this is a judgment call, not a requirement to build a framework.
2. Conflicting status documentation: docs/smart_tag_prototype.md:7,11 says the candidate is disconnected, while docs/smart_tag_export.md:3 documents the later integration. The prototype page needs an explicit historical/superseded notice.

Standards total: 2 heuristic findings, 0 hard violations. Worst in this axis: duplicated orchestration with mutable scenario selection.

## Spec

1. Medium: audit execution success is conflated with contract readiness. run_smart_tag_audit.py:23-25 permits unresolved team mapping, :31 assigns complete, :32-40 records potentially failed checks, and :47 returns success based on execution status. Separate audit_executed from contract_passed, define required vs informational checks, and test unresolved mapping/incomplete timeline. Coarse kickoff containment is diagnostic, not proof of exact synchronization. This finding does not establish that the actual supplied bundle failed its contract.
2. Medium: run_gap_evidence.py:21 and run_possession_duration.py:20 select scenarios with [:4], despite the frozen/original-four experiment requirement. Explicitly pin expected names and parameters and save source/config/input provenance. The existing model checksum alone does not establish experiment identity.

Core semantics otherwise agree with the specs: unknown/conflicting team evidence is neutral; phase tags are not possession labels; D probability is preserved without guaranteeing unchanged decoded D; gaps respect sequences; duration cost is soft and sequence edges are censored. Smart Tag actual export remains openly blocked, not verified.

Spec total: 2 findings. Worst in this axis: audit success can be mistaken for readiness.

## Domain modeling

CONTEXT.md establishes a glossary, not an implementation spec. Key distinctions: possession spell vs tactical phase; missing observation vs dead ball; reference validity vs prediction abstention; frame-level probability vs decoded possession state; external development evidence vs competition evaluation. Existing DFL target mapping remains an external supervision convention, not proof of equivalence to Metrica's control convention.

## Evidence already present

Before commissioning more diagnostics, use docs/decision_audit.md: it already demonstrates loss of team-specific spatial distinctions during missing-ball intervals and sensitivity to incorrectly observed ball coordinates. The ball-only +3m displacement is a synthetic hostile probe, not evidence that this bias occurs in the competition. The failed geometry experiment does not erase the representation limitation; it shows that the particular remedy failed its gate.

run_error_analysis.py currently loads temporal_v1.pt, so its output must not be casually attributed to v2. That is a pre-existing item outside this diff, requiring provenance verification before reusing the report as current-model evidence.

## Disposition

This was a review, not a code-fix pass. No model, exporter selection, experiment score or restricted asset changed. The glossary and this review are the only additions. Fix audit readiness and experiment provenance before relying on these tools for a new candidate decision; then resolve historical status notices. No claim of a fresh independent test or improved competition score is supported by this review.


## Follow-up correction scope

The audit readiness and scenario/provenance findings were corrected in a separate fix pass. The historical prototype page now explicitly points to the later integration status. Prior experimental scores and their original provenance are not retroactively rewritten. The selected v2 model, configuration and CSV remain unchanged.

Verification of the integrated fixes: 136 main tests and 10 input-diagnostic tests passed. Audit completion and contract readiness now have separate booleans; required checks gate CLI success. Both bounded experiment runners pin named scenario definitions, record source/config/model/input fingerprints and verify them before saving new v2 reports. Existing v1 reports were not altered or rerun. Remaining duplicated scoring loops were intentionally retained to avoid building an experiment framework; shared scenario/provenance handling now lives in one small helper.
