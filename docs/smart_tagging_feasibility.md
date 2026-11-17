# Smart Tagging feasibility: retain the current v2 release

Date: 8 November 2026. Decision: **no production integration on the evidence established by this study**. This is a validation decision, not a finding that Smart Tagging is useless.

## What was checked

- The public challenge documentation and supplied starter-loader source, without opening the competition Smart Tagging asset, videos, ground truth or predictions.
- The IDSSE publication and authors' repository, plus event XML from the four existing external training matches and the validation match. No held-out/regression match was needed.
- Public Metrica sample-data documentation, the previous hackathon's README and a vendor demo description. No gated bundle or account demo was downloaded.

## Why production fusion is not justified yet

Smart Tagging represents automatically detected intervals, which can overlap, have unknown teams and contain false detections or timing errors. An interval is not a possession and its endpoint is not necessarily the restart instant. The public loader exposes no event-confidence field. A restart team must not be invented when it is N/A. See the supplied public `DATA.md` and `starter/load_smart_tagging.py`; the fuller primary-source review is in `smart_tagging_sources.md`.

Our external DFL events are manual action annotations, not outputs of the same automatic interval generator. The [IDSSE paper](https://doi.org/10.1038/s41597-025-04505-y) documents event-timestamp errors and synchronization of passes/shots; that does not establish exact timing for all restart subtypes. Using clean point events, or manufacturing arbitrary windows around them, could produce an optimistic proxy result without validating actual Smart Tagging fusion.

The [previous hackathon's public README](https://github.com/metrica-sports/more-than-a-hack/blob/main/README.md) advertises automatic tags plus tracking, but the inspected documentation does not establish an independent possession/dead-ball reference, match identities excluding the current competition match, or dataset reuse terms. The vendor demo is likewise an acquisition lead, not a verified validation dataset. No claim is made that suitable data cannot exist; the bounded search did not establish it.

## Concrete external schema audit

| External match | Explicit restart records | Deleted/void event records | Restart records with CalculatedTimestamp |
|---|---:|---:|---:|
| J03WMX | 101 | 108 | 0 |
| J03WOH | 97 | 52 | 0 |
| J03WOY | 108 | 75 | 1 |
| J03WPY | 93 | 60 | 0 |
| J03WQQ | 98 | 80 | 1 |

The audit counted direct KickOff/ThrowIn/CornerKick/FreeKick/GoalKick wrappers, verified event match IDs and unique event IDs, and kept EventTime separate from DecisionTimestamp. It did not generate possession labels or inspect target transitions. CalculatedTimestamp presence is only a schema observation; absence does not prove a record was never synchronized.

DecisionTimestamp and EventTime often differ by tens of seconds. On J03WQQ the median differences are 9.41 seconds for throw-ins, 28.66 for free kicks, 18.57 for goal kicks and 25.39 for corners. **These differences are not measured annotation errors:** the decision and the subsequent action are distinct events. Treating one field as the other, or treating their span as the supplied Smart Tagging interval, would be unjustified. Local aggregate audit and source hashes are recorded in `reports/restart_feasibility/audit.json`.

## What would reopen this path

Establish a permitted other-match dataset with automatic Smart Tagging from a relevant generator/version, matched tracking, verified time alignment and independent possession/dead-ball targets. Then test one small optional fusion branch against v2, including missing/wrong teams, timing errors and contradictory tags. Keep clean human events and simulated errors explicitly labeled as proxy conditions. Do not calibrate on the competition match or claim reused matches are untouched tests.

## Work completed and release impact

The approved milestone was a go/no-go feasibility study. The external-validation condition was not met, so no fusion parser, fusion weight, larger model or new training run was added. v2 weights, submission configuration, exporter and existing local CSV remain intact. No new model-performance result is claimed and no competition rerun was necessary. The prior 100 passing tests describe the unchanged code; this documentation-only study does not add artificial tests to increase that count.

Next: use the preserved v2 release for final delivery work unless genuinely new validation evidence becomes available. Publication and submission still require their own later action; this study did not publish or submit anything.
