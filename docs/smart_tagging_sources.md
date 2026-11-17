# Smart Tagging restart evidence: external-validation feasibility

Research date: 2026-11-08. Scope: public first-party documentation, the IDSSE publication and authors' repository. No competition match assets, videos, ground-truth files, predictions, or evaluation results were opened or executed. The local competition `DATA.md` and `starter/load_smart_tagging.py` were read as documentation/source only. No bulk downloads, account access, messages, or model runs were performed.

## Decision

**No-go for claiming externally validated Smart Tagging fusion, and no-go for promoting this branch into the production decoder on the evidence currently established.** IDSSE can support a deliberately limited proxy experiment. It does not supply the same automatic tag generator, interval semantics, or error distribution. A positive proxy result would justify further research, not a claim of competition improvement.

There are first-party routes to external Metrica automatic tags and tracking: the previous hackathon's documented download and a vendor demo. It would be incorrect to assert that no external automatic tags exist. The bounded lookup did **not** establish an externally accessible package containing those tags, independently labeled frame targets, verified alignment, and a documented generator/version correspondence to the 2027 release. Availability of tags alone does not meet the validation requirement.

## What the sources establish

### Competition input contract

The public [2027 DATA.md](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/DATA.md) describes automatic tactical phases and set pieces in SportsCode XML, derived from the same video/tracking and subject to false positives and negatives. It explicitly distinguishes phase intervals from possessions. `Team=N/A` includes every throw-in in the documented release. This creates correlated evidence and missing-side problems; it does not provide an independent answer key.

The public [loader](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/starter/load_smart_tagging.py), also read from the allowed local source path, exposes `code`, `start_s`, `end_s`, `start_frame`, `end_frame`, and tag groups. There is **no per-instance confidence field in this documented loader contract**. That is a contract observation, not proof that Metrica's internal system has no confidence estimates.

Both documents put timestamps on the tactical video's clock: nearest-frame rounding of seconds multiplied by fps. The loader treats intervals as right-open, describes tag accuracy as approximately one second, and describes kickoff tags as roughly five-second windows. Its time-base check tests whether kickoff lies inside a window; it does not identify an exact restart instant. The check's `<= end_frame` comparison is also less strict than the stated right-open convention. Do not execute the loader in this study because it opens competition assets.

Consequences for an adapter: preserve both endpoints and missing teams, accept overlapping tags, and record uncertainty. Do not turn interval starts or ends into certain dead-to-live boundaries, label an entire set-piece interval dead, infer a throw-in side from `N/A`, or invent confidence values. A matching XML container does not establish matching event semantics.

### IDSSE provenance, targets, and timing

The [authors' repository](https://github.com/spoho-datascience/idsse-data) identifies seven official DFL matches, TRACAB position data, and CC-BY 4.0 with DFL attribution and publication citation required. The [publication](https://doi.org/10.1038/s41597-025-04505-y) ([open-access text](https://pmc.ncbi.nlm.nih.gov/articles/PMC11787359/)) describes human event analysts and supervisor review. Its table includes restart variants such as ThrowIn, FreeKick, Kickoff, CornerKick and GoalKick. Ball frames carry `BallStatus` (inactive/active) and `BallPossession` (home/away control). These are provider reference fields, not independently established equivalence to the challenge's possession definition.

Both modalities have ISO timestamps, but manual events have unsystematic timing errors. The published synchronization applies DataBallPy to 6,251 passes/shots; that does not establish synchronization of every restart subtype. Timestamp corrections average -0.37 +/- 1.82 seconds and reach 27.34 seconds. The authors did not evaluate synchronization against video ground truth. Thus neither original timestamps nor synchronized point events establish exact automatic interval endpoints. A synchronization certainty score concerns that procedure; it is not Smart Tagging detection confidence. [Publication, event-data and synchronicity sections](https://pmc.ncbi.nlm.nih.gov/articles/PMC11787359/).

### External Metrica candidates

| Candidate and primary source | Established by documentation | Missing for this validation |
|---|---|---|
| [Metrica sample-data](https://github.com/metrica-sports/sample-data) | Vendor sample tracking and events; synchronized streams; third sample uses FIFA EPTS and event JSON | README does not identify these as the same automatic Smart Tagging phase intervals or provide independent frame possession/dead-ball targets. Same vendor is insufficient. |
| [More than a Hack 2026 README](https://github.com/metrica-sports/more-than-a-hack/blob/main/README.md) | Automatic Smart Tagging SportsCode XML plus broadcast ATD; documentation supplies a Drive link and a tracking-and-events-only option; tags may have false positives/negatives | No independent frame targets documented; broadcast setting differs; precise version, actual accessible contents, alignment and overlap with competition match not verified. Documentation only was inspected. |
| [Vendor Smart Tagging article](https://www.metrica-sports.com/blog/smart-tagging-football-analysis), dated 2024-03-21 | Advertises a game demo via an account, Play application, Cloud Engine and key `SMART`; describes automated tactical and set-piece tags | No independent labeled frame target, research export/license, generator-version match or exact timing contract established. Account/demo was not accessed. |

These are acquisition leads, not verified validation datasets. The previous hackathon's documented download is the strongest lead for actual automatic tags plus tracking. A future acquisition must exclude the competition match and verify rights, assets and alignment before any training or scoring. The [2027 rules](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/RULES.md) allow other-game labels and prohibit human-produced labels about the provided game. This study's stricter prohibition on reading competition assets remains in force.

**Identity and rights check:** The 2026 README gives only `<GAME_NAME>` placeholders, with no match list. It does not exclude BAR-ATH or state a dataset reuse license; the visible repository root has no LICENSE entry. The demo article does not identify its match or grant external training/redistribution rights. Therefore neither candidate is cleared for acquisition/use here. Public download advertising and free product access are not established dataset permissions. IDSSE's explicit CC-BY license is a separate, stronger provenance case. No legal conclusion about an unseen download's terms is implied.

## Conditional proxy experiment

The following is a proposed design, not an executed experiment or a calibrated model:

1. Audit the external XML hierarchy and per-match clock correspondence first. Extract only explicit restart actions and known team attributes. Treat point-action time as an uncertain action observation, never as a measured Smart Tagging window boundary. Exclude malformed, ambiguous, deleted or unresolved events and report exclusions by subtype.
2. Build restart evidence independently of target access. Hold `BallStatus` and `BallPossession` exclusively in evaluation code. Do not generate candidate times, widths, teams or confidence from target transitions, and do not use targets to select an event's nearest boundary.
3. Compare a frozen tracking-only baseline with a bounded, optional restart-evidence branch. Use leave-one-match-out or a fixed match holdout. Choose all parameters without the held-out match. Seven matches permit only limited estimates of transfer.
4. Keep a clean manually annotated point-event condition separate from simulated conditions. Predeclare timing jitter, global offset, synthetic window widening, event deletion, extra events, wrong teams, missing teams (including a missing-team throw-in condition), duplicates, overlaps and conflicting tags. Treat these as sensitivity axes; their rates are assumed and **not measured Metrica error rates**. An arbitrary symmetric window is only a stress-test construction.
5. Report live/dead and team performance separately, boundary errors, possession fragmentation and match-level results. Include baseline differences, zero-evidence behavior, and worst-case degradation. Report how many eligible events/teams/timing matches survived the audit. Do not use only an oracle event condition to select a production weight.

This can establish whether a decoder safely accepts imperfect restart evidence and identify failure modes. It cannot establish the benefit, calibrated weight, false-positive cost or boundary placement of the actual automatic Smart Tagging feed. Tracking quality, event generation and label policy all shift between sources. In particular, simulating independent event noise cannot recreate errors shared by Metrica tagging and tracking.

## Evidence required to change the decision

Obtain other-match automatic Smart Tagging exports from the relevant generator/version, matched tracking and a verified timeline, plus independent labels with an explicit possession/dead-ball policy and usable provenance. Measure error by restart type, endpoint and team availability; confirm no test-match overlap. Then freeze a fusion rule and evaluate on held-out matches, including no-tag and contradictory-tag cases. Until then, retain any adapter as research-only with neutral default behavior. No numerical Smart Tagging validation result is asserted here.

## Lookup boundary and reproducibility

Checked the paper/DOI, author repository, 2027 public docs/loader, vendor sample-data README, 2026 public README and vendor Smart Tagging article. Search was restricted to those primary-source families. Nature/PMC direct rendering intermittently failed, so indexed primary-paper text was used to verify the collection and synchronization sections. No dataset inventory beyond the documentation, Drive contents, proprietary demo, or generator implementation was inspected. Pin source revisions and content hashes before making a reproducible acquisition manifest; URLs to moving `main` branches alone are not version pins.
