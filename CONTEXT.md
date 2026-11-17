# Football possession estimation

Estimate team possession and stoppages on the tactical-video timeline. Evaluation distinguishes correct frame labels from correctly identified possession intervals.

## Language

**Possession state**:
The inferred home-control (H), away-control (A), or dead-ball (D) state at a frame. A nearby player is evidence of control, not its definition.
_Avoid_: nearest-player ownership, possession answer

**Possession spell**:
A maximal continuous interval of H or A in the evaluated timeline; D intervals separate possessions but are not possessions themselves.
_Avoid_: phase, arbitrary segment

**Dead ball**:
A stoppage in play, represented by D. A ball missing from tracking does not establish a stoppage.
_Avoid_: missing ball, ungraded frame

**Abstention**:
An X prediction. On graded frames it counts as a miss for macroF1; temporal scoring fills it from nearby labels within the scorer's permitted runs.
_Avoid_: dead ball, harmless unknown

**Ungraded frame**:
A frame excluded by reference-label validity. A model's uncertainty or missing observations do not determine whether a frame is graded.
_Avoid_: predicted X, missing observation

**Phase tag**:
An automatically inferred interval describing a tactical phase or set piece, potentially with a team. It can overlap possessions and is uncertain evidence rather than a possession annotation.
_Avoid_: possession label, ground truth

**Track slot**:
An automatically maintained object track with team membership whose association with a real player can change.
_Avoid_: player identity, shirt number

**Missing observation**:
An unavailable tracked position. It is independent of whether the ball is in play and whether a reference possession label exists.
_Avoid_: dead ball, unknown reference

**Tactical frame**:
A zero-based frame of the reference tactical video, also used to identify submission rows. ATD frame identifiers use a different one-based convention.
_Avoid_: broadcast frame, unspecified frame number

**Sequence boundary**:
A boundary beyond which temporal evidence must not propagate, such as a half boundary or an observation discontinuity defined by the input contract.
_Avoid_: possession boundary

**Reference label**:
A possession or live/dead annotation used to score or supervise an algorithm. External DFL reference fields and organiser possession references may embody different control conventions.
_Avoid_: universal football truth

**Public correctness sample**:
The five-minute organiser reference provided to check pipeline correctness; it is not a training, tuning, or variant-selection set.
_Avoid_: validation set, leaderboard

**External development evaluation**:
Evaluation on noncompetition matches used to develop or select the system. Repeated use means it is not independent final evidence.
_Avoid_: competition score, unseen final test

**SpellF1**:
Possession-interval matching F1 averaged over the official temporal overlap thresholds. It is the competition ranking metric and is distinct from frame accuracy.
_Avoid_: accuracy percentage

**Selected submission**:
The particular frozen algorithm output chosen for delivery. An experimental candidate or correctly formatted file need not be selected or competitively better.
_Avoid_: latest experiment, best model without a stated evaluation basis
