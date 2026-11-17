# Rules re-verification — 9 November 2026

## Finding

The blanket claim that every kind of training, adaptation or feature development must use external data is broader than the public organiser rules. External supervised fitting remains valid, but it is not the only permitted approach. The public five-minute answer key remains expressly forbidden for training and tuning.

## Primary sources checked

Downloaded RULES.md (especially lines 54–80), FAQ.md (training, Smart Tagging and hosted-LLM sections), CHALLENGE.md, DATA.md, EVALUATION.md and SUBMISSION.md. Cross-checked live organiser GitHub RULES.md and FAQ.md on 9 November 2026:

- https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/RULES.md
- https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/FAQ.md

No restricted match asset, video, ground-truth array or generated prediction was opened during this review.

## Verified public rule boundaries

| Activity | Public documentation |
|---|---|
| Train/tune on the public five-minute possession labels | Explicitly forbidden; correctness/alignment checks only |
| Use private answers or human-produced annotations about this same match | Forbidden |
| Manually label/correct the supplied match or tracking | Forbidden |
| Learn from this match without human labels | Explicitly allowed: unsupervised/self-supervised methods and algorithm-produced pseudo-labels with no human label decisions |
| Inspect data/video to understand/debug/design features | Allowed in public RULES.md; cannot turn observations into hand labels/corrections or constants tied to particular frames |
| Use supplied automatic Smart Tagging | Explicitly encouraged; noisy phase intervals are not possession answers |
| Improve tracking algorithmically | Allowed |
| Upload restricted assets/extracts to hosted LLMs | Forbidden in FAQ terms summary; keep processing local and data out of tool/chat outputs |
| Train on other games | Allowed subject to access, explanation and data licensing |

Public permission to inspect video does not authorize displaying restricted video frames to a hosted assistant. Local automated processing and human local inspection are different from uploading material.

## Provenance conflict

`docs/specs/external-benchmark-prompt.md`, section 3, says an organiser's latest clarification overrides older documentation, then imposes external-only training and no video comparison. The brief supplies no original message, date, author quotation or source link for that clarification. An identical generated output copy exists; it is not independent corroboration. A newer direct organiser instruction could override public documentation, but its existence/content has not been verified here. Do not claim the brief itself is an organiser source or silently erase its historical scope.

Previous assistant responses made two contradictory errors: presenting the broad external-only policy as established organiser law, then suggesting public-label fitting might be permitted. The verified public rules support neither statement: label-free learning is allowed, public-label training/tuning is not.

## Development decisions versus requirements

The small CPU network, eight epochs, one-feature experiment, external validation promotion gates and requirement for external evidence before production fusion were our engineering choices. These helped prevent optimistic selection but are not organiser mandates. The completed feature experiment remains a valid negative result under its declared protocol; no gates or records should be changed retrospectively.

External fitting has produced a working v2 fallback. However, treating the supplied match as inference-only and deferring all Smart Tagging work until an equivalent independently labelled external tag dataset exists may have excluded useful permitted development. No improvement is guaranteed by relaxing those internal choices.

## Corrected next technical direction

1. Preserve v2 and the completed negative experiment; stop additional generic geometry/decoder searches.
2. Build a local-only input/Smart Tagging contract audit: verify timeline, units, team mapping, tag semantics, parser behavior and missingness using deterministic scripts; keep match-derived details local. Use published documentation and synthetic fixtures for assistant review. Never use possession answers to pick a variant.
3. Design one bounded tag-assisted restart/owner-evidence candidate, testing its mechanics on synthetic cases and transfer behavior with clearly labelled external proxies where practical. Identical external tags are useful evidence, not a prerequisite imposed by organisers. Do not describe proxy tests as measured competition gains.
4. Consider label-free adaptation only with a specified objective, drift safeguards and reproducible local execution. Do not start broad pseudo-label training solely because it is permitted; it may reinforce the model's errors.
5. If original newer organiser guidance is supplied, reconcile it before any action that conflicts with it. No blanket new permission request is needed to continue source/synthetic development.

This review corrects policy documentation. It does not train on the supplied match, inspect restricted assets, promote a candidate or alter v2.
