# Frame-level ball possession from automatic tracking

Team Ball Knowledge. More than a Hack 2027, Phase 1 (Metrica Sports and FC Barcelona possession challenge). Submitted model: v10e. Submitted file: `Ball Knowledge.csv`, SHA256 `e947dbddce6566e3766756d87214015ef977f4c1fcc2f096a29e0221a86aaa68`.

## 1. Executive summary

The task is to label every frame of the 199,319-frame tactical video as home possession (`H`), away possession (`A`) or dead ball (`D`), with `X` outside the two graded halves. Submissions are ranked by spellF1, which matches whole possession spells between prediction and truth by temporal overlap, so a model must produce clean intervals, not just accurate frames.

Our system, v10e, works from the provided automatic tracking only. For each frame it computes 78 features that do not depend on player identity. Twelve identically configured temporal convolutional networks (TCNs) turn these into class probabilities, which are averaged and decoded into spells by a three-state Viterbi decoder. A 1 s controlled-possession filter and a kickoff rule taken from the automatic Smart Tagging finish the labelling.

The networks are trained on ten external matches: seven DFL Bundesliga matches with provider possession labels and the three public Metrica Sports sample games, with labels derived from tracking and event data. The competition tracking loses the ball on 40.7% of graded frames and drops players often. We measured these fault patterns on the match without using labels, and reproduced them on the training matches.

On 20 SkillCorner matches of broadcast-derived tracking, which no model trained on, v10e reaches a mean spellF1 of 0.548, against 0.540 for the previous five-network model and 0.483 for v7. SkillCorner was also used to decide between model variants, so it is a selection benchmark rather than an untouched test. Adding the third Metrica game to training raised held-out Metrica scores by 0.005 and 0.012. These differences are small relative to run-to-run variation. v10e has no score on the competition match: it was neither evaluated nor selected on the public five-minute window.

The repository regenerates the submitted file byte for byte in under a minute on a CPU (section 9).

## 2. Problem definition

The organisers provide automatic tactical tracking (players and ball at 25 frames per second), automatic Smart Tagging events, two videos and a homography for the competition match. A submission is one row per tactical-video frame with a label in `H`, `A`, `D` or `X`. Frames 11185 to 90651 and 110808 to 190197 (zero-based video frames) are graded; all others must be `X`.

The ground truth follows a controlled-possession convention: possession changes only when a team gains control of the ball, a loose ball stays with the last team in control, and dead-ball time is its own state. The metric, spellF1, extracts maximal runs of `H` and `A` (possession spells) from prediction and truth, matches them one to one by temporal intersection over union at thresholds from 0.3 to 0.7, and averages the resulting F1 scores. A prediction with high frame accuracy can still score poorly if it breaks one true spell into several short ones or merges two spells across a short stoppage. The design therefore puts as much weight on temporal structure (decoding, minimum spell length, stoppage edges) as on per-frame classification.

## 3. System overview

```
automatic tactical tracking (ATD: player tracks and the ball, 25 fps)
  -> 78 identity-invariant features per frame              fcb_external/team_features_v6.py
  -> 12 temporal convolutional networks, each averaged
     over the original and the home/away-exchanged input   fcb_external/temporal.py
  -> mean of the 12 probability vectors P(H), P(A), P(D)
  -> three-state Viterbi decoder, reset at each half       fcb_external/state_decoder.py
  -> controlled-possession filter (spells under 1 s
     return to the team already in possession)             fcb_external/possession_convention.py
  -> kickoff rule: dead ball before each half's
     Smart Tagging "Kick Off (Start)"                      fcb_external/smart_tagging.py
  -> frame,label CSV, validated with the official checker  fcb_external/submission.py
```

| Component | Setting in the submitted checkpoint |
|---|---|
| Inputs read at inference | `BAR-ATH_tactical_atd.txt`, `BAR-ATH_tactical_atd.xml`, `BAR-ATH_tactical_smart_tagging.xml` |
| Features | 78 per frame, feature module `v6` |
| Network | 64 channels, 7 residual dilated convolutions (dilations 1 to 64), receptive field 255 frames (10.2 s), 91,715 parameters |
| Ensemble | 12 networks, seeds 9300 to 9311, equal weights |
| Decoder costs | owner change 5, dead/live change 30 |
| Controlled-possession filter | 25 frames (1 s) |
| Checkpoint | `models/temporal_v10e.pt`, kind `pytorch_temporal_ensemble_v1`, SHA256 `490d00f5099359ae254f9636598e831ca884e7286f7e0b204318d9d8164e690d` |

Team roles follow the metadata team order (the first team is home). ATD frame identifiers are one-based, so video frame = ATD frame − 1. The pitch is 105.3 × 68.0 m.

## 4. Data and preprocessing

### 4.1 Data-use boundaries

Training and model selection use external matches only. Nothing about the competition match that carries a label (the public five-minute answers, event feeds, commentary or any hand annotation) was used for training, tuning or selection. Two kinds of label-free information about the competition tracking were used, as the rules allow: the length and timing profile of its ball-visibility gaps, and the frequency and duration of its player-tracking faults (section 4.4). At inference the pipeline reads the tactical tracking, its metadata and the Smart Tagging file. It does not read the video, the broadcast data or the homography. A development experiment that processed the tactical video is described in section 9.3.

### 4.2 Training data and labels

| Dataset | Matches | Frames (25 fps) | Labels | Role in v10e |
|---|---|---|---|---|
| DFL / IDSSE (Bassek et al. 2025), CC BY 4.0 | 7 Bundesliga matches: J03WMX, J03WOH, J03WOY, J03WPY, J03WQQ, J03WN1, J03WR9 | 1,002,644 (H 293,931, A 259,174, D 449,539) | Provider fields: `BallStatus` 0 gives `D`, otherwise `BallPossession` 1 or 2 gives `H` or `A` | Training |
| Metrica Sports public sample games 1 and 2 | 2 anonymised matches | 145,006 and 141,156 | Derived: dead where the corrected ball position is missing; otherwise the team of the most recent on-ball event, carried forward | Training (counted three times each in sampling) |
| Metrica Sports public sample game 3 (EPTS format) | 1 anonymised match | 143,761 (H 49,808, A 47,628, D 46,325) | Derived in the same way from EPTS tracking and JSON events | Training (counted three times); a held-out benchmark for models up to v9e |

The DFL labels are the provider's own; we did not label anything and do not know their annotation process in detail. Download sources and checksums are pinned in `configs/idsse_*.json`. The Metrica games come from the organiser's corrected tracking, so their dead-ball definition is close to the competition's, but their possession labels are derived from events, which is a different convention from the organisers' controlled-possession truth. In the Metrica data the ball position is blank exactly while the ball is dead. No automatic tracking behaves this way, and a model trained on it would learn that a missing ball means a dead ball. For training, the ball is therefore held at its last position through each blank run before any simulated fault is applied (`run_retrain_v8.py`).

### 4.3 External benchmarks

| Benchmark | Data | Labels | Status for v10e |
|---|---|---|---|
| Held-out Metrica game (leave one game out) | Metrica game 1 or 2, under five simulated ball-loss conditions | Derived from events, as above | Ablation only: the compared models are trained without the held-out game; the final v10e checkpoint trained on all three games and has no Metrica test score |
| SkillCorner open data, MIT | 20 A-League 2024/25 matches, broadcast-derived automatic tracking resampled from 10 to 25 Hz (`fcb_external/skillcorner.py`) | Provider phases of play: inside a phase, the team in possession; outside every phase, `D` | Never used for training; used to decide between model variants (selection benchmark) |
| Metrica game 3 under 24 fault conditions | Six ball-loss scenarios × ball held or blank during stoppages × all players or simulated player faults | Derived from events | Test for v7, v9 and v9e; training data for v10e |

SkillCorner differs from the competition data in camera (broadcast rather than a wide tactical view, with about 40% of frames having no tracking at all), in fault type (the ball is used only where the provider detected it, so ball faults are real) and in labelling convention. Absolute SkillCorner scores are therefore lower than on the other datasets and are useful mainly for comparing models.

### 4.4 Robustness to automatic-tracking errors

The external training data are clean, while the competition tracking is not. Measured over graded frames without using labels, the competition tracking has the following profile:

| Property | Competition tracking |
|---|---|
| Ball missing | 40.7% of frames, in about 150 gaps; median gap 10.7 s (p25 6.1 s, p75 21 s, p90 34 s); 75% of missing frames lie in gaps over 15 s |
| Where gaps occur | ball missing on 58% of frames with slow players against 13% otherwise; 78% of gaps start within 3 m of a touchline or goal line |
| Player-tracking faults | a team has fewer than 11 tracked players on about 36% of frames (10 players 22%, 9 players 5.5%, 8 players 2.5%), in episodes of median about 4 s; more than 11 on 5 to 8% of frames |

Two simulators reproduce this profile on the external matches using observations only. `fcb_external/target_like.py` hides the ball in log-normal gaps (median 10.7 s), placed preferentially when players are slow and the ball is near a line. `fcb_external/tracking_faults.py` removes one to five players per team in episodes covering 36% of frames and adds duplicate tracks on 6.5%. Their parameters were fixed from the measurements before any training that used them. Each training match is presented clean and under six corruptions (ball loss in irregular, crowded, slow-player and target-like patterns, plus a mixed stress test), with player faults added to every corrupted copy. Mask seeds differ by role, fold, match and scenario, so training and validation never share a mask. Appendix A lists the scenarios.

A label-free check (`run_adversarial_check.py`, two-sample Kolmogorov-Smirnov statistics on each of the 78 features) motivated the player-fault simulator. Against the competition tracking the target-like scenario was the closest overall (mean KS 0.072 against 0.216 for clean data), and the largest remaining mismatches were the player-count features (KS 0.34 to 0.56).

## 5. Model and training

### 5.1 Features

The 78 features (`fcb_external/team_features_v6.py`) never use player identity, shirt number or track-slot index. Players are pooled within their team, so swapped or reordered tracks do not change the features, and the home and away blocks can be exchanged exactly.

| Group | Count | Content |
|---|---|---|
| Ball and geometry | 20 | ball visibility, position, distance to the boundary, speed at three lags, nearest and second team distance, player counts, visibility over 1 s and 5 s |
| Per team, relative to the ball | 2 × 10 | three nearest distances, counts within 2, 5 and 10 m |
| Per team, without the ball | 2 × 6 | centre, spread, centre speed |
| Ball-visibility gaps | 8 + 2 × 4 | time since and until the ball is seen, gap length, change of direction and velocity across the gap, each team's nearest distance held from the frames bordering the gap |
| Player pace | 6 + 2 × 2 | median and 75th-percentile speed over a centred 1 s window, share of players walking, smoothed versions, per-team medians |

Ball gaps up to 2 s are interpolated linearly before the other features are computed. The decoder is offline, so features may use the far end of a gap. Pace features exist because automatic tracking often loses the ball when play stops, which removes the most direct dead-ball cue; slow movement is the next best signal. A track that moves faster than 12 m/s between frames is treated as an identity swap and excluded from pace.

### 5.2 Network and training

Each network is a 1 × 1 input convolution, seven residual dilated convolutions (kernel 3, 64 channels, dilations 1 to 64) and a 1 × 1 output layer, with 91,715 parameters and a receptive field of 255 frames (10.2 s). Training uses class-weighted cross-entropy, AdamW (learning rate 0.002, weight decay 0.001, cosine decay to 5%), and 16,000 steps of 16 windows with 256 supervised frames each, drawn at random across matches and scenarios. Home and away are exchanged at random during training, and at inference each network's probabilities for the original and the exchanged input are averaged. The v10e networks were trained on a GPU with `run_step4.py final 12`.

### 5.3 Ensemble

v10e averages the probabilities of twelve networks that differ only in random seed, then decodes once. Single networks trained this way scored between about 0.48 and 0.53 on SkillCorner depending only on the seed, which is larger than most differences between model variants. Averaging reduces that spread. Twenty networks scored 0.5481 against 0.5484 for twelve, so twelve was kept; we do not claim that further networks would add accuracy.

### 5.4 Decoding and possession rules

The Viterbi decoder runs over log-probabilities with one cost for a change of owner (5) and one for a change between live and dead (30), reset at each half. The costs were chosen by cross-validation on the DFL matches for v7 and kept since. The decoder implements the loose-ball rule: without evidence of a change, the owner is held.

After decoding, an opposing spell shorter than 25 frames is returned to the team already in possession, which implements "a change needs controlled possession". The first spell after a stoppage is always kept, since a restart has a clear possessor. The 1 s threshold was chosen on Metrica games 1 and 2, where it was best or tied for best under three readings of the rule; 2 s was worse under the loosest reading.

In each half, frames before the start of the Smart Tagging `Kick Off (Start)` window are set to `D`. The rule applies only if the half has exactly one such tag, opening inside the half and within ten minutes of its first frame; no frame number is hard-coded. Dead ball is otherwise predicted by the networks. Restart ownership rules (goal kick, corner, throw-in), set-piece tags as restart anchors and phase tags are not implemented.

## 6. Experimental evaluation

All scores are spellF1 computed with the organisers' scoring code (`vendor/official/evaluation/possession_metrics.py`, vendored unmodified) against the provider labels described in section 4. Scores on different datasets, label conventions or fault conditions are not comparable with each other, and none of them is comparable with scores reported by other teams on their own benchmarks.

### 6.1 Comparison of final checkpoints on SkillCorner

Twenty SkillCorner matches; mean spellF1; each row is a final checkpoint as shipped at the time; no model trained on SkillCorner.

| Checkpoint | Description | SkillCorner spellF1 |
|---|---|---|
| v7 | one network, DFL only | 0.483 |
| v9 | one network, DFL and Metrica games 1 and 2, player faults | 0.534 |
| v9e | five v9 networks averaged | 0.540 |
| v10e (submitted) | twelve networks, DFL and Metrica games 1 to 3 | 0.548 |

v10e is better than v9e on 17 of the 20 matches. SkillCorner influenced the choice between v9 and v9e, between v10e and the larger network, and between twelve and twenty networks (section 7), so these figures are selection evidence. The gain from v9 to v10e (0.014) is of the same order as the seed-to-seed spread of single networks.

### 6.2 Earlier checkpoints on Metrica game 3

Metrica game 3 was a held-out game until v9e. Mean spellF1 over five ball-loss scenarios, with the ball held at its last position or blanked during stoppages, and with simulated player faults or all players tracked. v10e trained on this game and has no entry.

| Checkpoint | held, player faults | held, all players | blank, player faults | blank, all players | clean, held, faults |
|---|---|---|---|---|---|
| v7 | 0.661 | 0.687 | 0.606 | 0.630 | 0.775 |
| v9 | 0.707 | 0.713 | 0.633 | 0.645 | 0.860 |
| v9e | 0.708 | 0.717 | 0.637 | 0.647 | 0.852 |

The step from v7 to v9 (+0.046 on the main condition) is the largest measured improvement on a match neither model trained on.

### 6.3 Ablation: Metrica game 3 as training data

To test whether game 3 should be added to training, we trained three networks per condition with and without it, leaving out Metrica game 1 or game 2 in turn, and scored the three-network average on the held-out game (`run_step4.py logo`). The adoption rule was fixed before the run: a mean gain of at least 0.005 on held ball with player faults, no held-out game below −0.005, and SkillCorner not lower by more than 0.005.

| Condition (held-out game) | without game 3 | with game 3 | change |
|---|---|---|---|
| Game 1, held ball, player faults | 0.712 | 0.717 | +0.005 |
| Game 2, held ball, player faults | 0.664 | 0.676 | +0.012 |
| Game 1, blank ball, player faults | 0.657 | 0.666 | +0.009 |
| Game 2, blank ball, player faults | 0.568 | 0.600 | +0.033 |
| SkillCorner, 20 matches (six-network averages) | 0.522 | 0.545 | +0.023 |

The rule was met and every cell moved in the same direction. These are ablation models, not the submitted checkpoint, and three networks per cell leave the individual differences within noise.

### 6.4 Checks on the submitted file

The file passes the official format validation. Two independent runs from the repository produced the same SHA256. All frames outside the graded halves are `X`, and the kickoff rule marks the leading frames of each half as dead. Over graded frames v10e labels 35.7% as dead (v9e 34.8%, v9 33.8%, v7 36.9%) and contains 323 possession spells (v9e 340, v9 348, v7 452); the v9e and v10e files agree on 96.8% of graded frames.

### 6.5 What the evidence establishes

The results establish that the v7-to-v9 changes (Metrica training data and simulated player faults) improved held-out performance on matches neither model trained on, and that adding game 3 and averaging networks did not reduce performance on any benchmark we have. They do not establish an absolute score on the competition match, they do not show that the gains after v9 exceed run-to-run noise, and they cannot show how well the controlled-possession convention of our training labels matches the organisers' truth.

Historical models v2 to v7 were scored on the public five-minute window as pipeline checks (frame alignment and label meaning): v2 0.526, v4 0.474, v6 0.584, v7 0.541. Those are not v10e results. The window holds 17 possessions and four stoppages, so these scores differ by less than one possession and cannot rank versions. Neither v9 nor any later model was scored on the window.

## 7. Design decisions and ablations

### 7.1 Evolution of the model

| Step | Change | Evidence |
|---|---|---|
| v2 to v4 | ball-gap features, longer training, all seven DFL matches | about +0.19 on DFL cross-validation |
| v4 to v6 | player-pace features, ball loss biased to slow play | about +0.04 on Metrica games 1 and 2 |
| v6 to v7 | target-like ball loss from the measured gap profile | +0.007 masked mean on DFL folds; +0.020 on the target-like condition, better on all seven matches |
| v7 to v9 | Metrica games in training, simulated player faults | +0.039 and +0.043 in the Metrica swap test; +0.046 on game 3 and +0.051 on SkillCorner |
| v9 to v9e | five-network average | +0.001 on game 3, +0.005 on SkillCorner |
| v9e to v10e | Metrica game 3 in training, twelve networks | ablation in section 6.3; +0.008 on SkillCorner |

Appendix B gives the full tables behind each step.

### 7.2 Decision record

| Candidate | Outcome | How the criterion was set |
|---|---|---|
| v5: relabel targets as 2 s controlled possession | dropped: worse than v4 on Metrica games 1 and 2 | compared after the run |
| v7: target-like ball loss | adopted on DFL cross-validation; decoder costs were tuned on the same held-out cells, which is mildly optimistic | cross-validation protocol fixed beforehand; no numeric threshold |
| v8: Metrica games as training data | did not qualify | rule written in `run_retrain_v8.py` before the run |
| v9: Metrica games plus player faults | adopted; the only candidate to qualify | rule written before the run, then restated in a combined form after the v8 and first v9 held-ball results had been seen, but before any blank-ball result |
| v9e: five-network average | adopted | bar (beat v9 on game 3 held with faults and on SkillCorner, lose no more than 0.01 elsewhere) set after an earlier check that a three-network average scored 0.538 on ten SkillCorner matches |
| Self-supervised ball imputer as extra input; teacher-student variant; control-event memory features | dropped: tied with v9 or failed the v9e bar | v9e bar |
| Dead-ball edge shift; output-weight edit; averaging v6 fold models | dropped: no gain beyond +0.005 | compared after the run |
| Metrica game 3 in training (v10e) | adopted | rule written before the run (section 6.3) |
| Larger network: 96 channels, dilations to 256 (receptive field 41 s), 32,000 steps | not adopted: +0.008 to +0.022 on the four Metrica cells, −0.013 on SkillCorner | rule written before the run |
| Twenty networks instead of twelve | not adopted: 0.5481 against 0.5484 on SkillCorner | rule written before the run |
| Ball detector trained on the tactical video | abandoned at its accuracy gate (section 9.3) | gates written before the run |

The larger network is the most informative negative result. It improved every Metrica cell but lost on SkillCorner, which suggests it fitted the Metrica conventions more closely without carrying over to other data. With only three Metrica games we could not tell which effect would dominate on the competition match, and the predeclared rule decided against it.

## 8. Limitations

Label convention. The DFL labels follow the provider's possession definition and the Metrica labels are derived from events; neither is the organisers' controlled-possession truth. The 1 s filter and the decoder approximate that convention but cannot reproduce it.

Short stoppages. Recall of dead-ball frames is high, but short stoppages are rarely found. For v7 on Metrica games 1 and 2 (clean input; `reports/dead_ball_analysis/v7.json`), dead-frame recall was 0.88 in both games and precision 0.89 and 0.97, yet 69 of 71 and 85 of 86 stoppages shorter than 5 s were missed entirely, while no stoppage longer than 20 s was missed. Detected stoppages start a median of 0.6 to 0.7 s late and end about 0.4 s early. These measurements are for v7; the same analysis was not repeated for v10e. Because a missed short stoppage merges two possession spells, this is likely a material source of spellF1 loss.

Missing ball. Inside long ball-visibility gaps the model relies mainly on the state at both ends. In a simulation on Metrica game 3, restoring 30% or 50% of the hidden ball frames (with 1 m noise) did not improve the v9e model, and restoring all of them added 0.024. Better ball recovery alone would therefore help only modestly.

Provider shift and simulated faults. The simulators reproduce the measured gap and player-fault profile of the competition tracking, fitted to this one match. Real automatic tracking also produces confidently wrong ball positions and team mislabels, which we do not simulate, and its ball track is noisier when visible (ball-speed features still differ, KS about 0.2).

Evidence base. Training uses ten matches from two sources. The only remaining external benchmark, SkillCorner, comes from a different camera and convention and was used for selection. Differences of a few hundredths between variants are within the run-to-run spread.

Reproduction environment. PyTorch results can change across CPU types and library versions; a few labels near decision boundaries could then differ (section 9.1).

## 9. Reproducibility and data-use compliance

### 9.1 Reproducing the submitted file

Reference environment: Windows 11 x64, Python 3.14.2, numpy 2.4.4, pandas 3.1.1, torch 2.11.0+cpu, 4 CPU threads, deterministic algorithms enabled.

```
pip install -r requirements-submission.txt
python run_submission.py --data-dir <bundle> --out "Ball Knowledge.csv"
```

`<bundle>` must contain `tactical/BAR-ATH_tactical_atd.txt`, `tactical/BAR-ATH_tactical_atd.xml` and `tactical/BAR-ATH_tactical_smart_tagging.xml`. The default configuration `configs/submission_final.json` pins the checkpoint by SHA256. The script refuses to overwrite existing files, validates the output with the official format check and writes a manifest of input, model and source hashes next to the CSV.

| Artifact | SHA256 |
|---|---|
| Submitted CSV (v10e) | `e947dbddce6566e3766756d87214015ef977f4c1fcc2f096a29e0221a86aaa68` |
| `models/temporal_v10e.pt` | `490d00f5099359ae254f9636598e831ca884e7286f7e0b204318d9d8164e690d` |
| v9e file (`configs/submission_temporal_v9e.json`) | `e47a1b9576ddfa2e3c41511adb7a6e71523051f68b4a669e050290cdbb1c8b4b` |
| v9 file (`configs/submission_temporal_v9.json`) | `8bea28e5eee180ce7d86c71d6d55fbd985be6066dacc497c626ba91ec0205f99` |

Two runs gave the submitted hash on the reference machine; v9, v9e and v10e have been run on that machine only (earlier v6 and v7 files were reproduced on two machines). Inference runs on the CPU. The v10e networks were trained on a GPU (torch 2.11.0+cu128), which is not bit-reproducible, so retraining reproduces the method but not the exact checkpoint; the shipped checkpoint is pinned for that reason. Retraining commands are listed in Appendix D. Unit tests: `python -m unittest discover -s tests` runs 168 tests; 167 pass, and `test_frozen_hash_guard_rejects_changed_model_and_config_before_data` fails outside the environment of the earlier v2 release because it checks package versions against that release's lock file.

### 9.2 Data use

| Activity | What we did |
|---|---|
| Public five-minute labels | used only as pipeline checks for v2 to v7 (scores in section 6.5); never for training, tuning or selection; not used from v9 onwards |
| Human-made data about the competition match | none used |
| Hand labels or corrections on the competition match | none |
| Label-free statistics of the competition tracking | gap profile and player-fault profile used to set the simulators (section 4.4); feature distributions compared in `run_adversarial_check.py` |
| Smart Tagging | only the two kickoff tags, for the kickoff rule |
| Tactical video | not read by the submitted pipeline; processed locally in an abandoned experiment (section 9.3) |
| Broadcast video, broadcast tracking, homography | not used by the submitted pipeline; the tactical homography was used only in the abandoned experiment |

### 9.3 Abandoned video experiment and an open compliance question

During development we tested whether the ball could be recovered from the tactical video. A small detector was trained locally on short three-frame crops of the competition's tactical video, with targets taken from the tracking's own visible ball positions projected through the supplied homography; no human labelled anything. Gates were written before the run. The detector failed the first gate on held-out time blocks (57 to 73% of accepted detections within 1 m, against 90% required), the experiment was stopped, and the decoded crops were deleted. No detector, detection or derived value is in the repository or the submitted model. A preceding simulation had already shown that even perfect ball recovery would add at most about 0.024 (section 8).

We flag this for the organisers rather than resolve it ourselves. The public rules as we read them on 9 November 2026 (`docs/rules_reverification.md`) allow inspecting the video and allow algorithmic pseudo-labels without human label decisions, which would cover this experiment. However, an earlier project brief recorded an organiser clarification requiring external-only training and no video comparison; we could not find the original of that clarification, and our team had separately adopted a no-video policy. If such a clarification applies, the experiment conflicts with it, even though nothing from it reached the submission. We disclose it here so the organisers can judge.

### 9.4 Use of AI assistants

AI coding assistants (OpenAI Codex, and Anthropic Claude models Sonnet 5.5, Fable 5.2 and Opus 5.5 through Claude Code) wrote and reviewed code, ran experiments on external data and drafted documentation. Local scripts processed all competition data; the hosted assistants launched those scripts and saw only aggregate outputs: the official scorer's summary metrics and confusion counts for the v4, v6 and v7 files on the public window; the label-free gap and player-fault statistics in section 4.4; per-feature summary statistics from the adversarial check; the number of tagged set pieces with a predicted stoppage nearby; aggregate properties of generated files (label shares, possession counts, agreement between files, hashes); and, for the video experiment, aggregate statistics only (homography stability, the share of projected positions inside the frame, alignment contrast ratios, the share of frames with a recorded ball height, and detector error percentiles). No tracking values, video frames, crops, label files or per-frame predictions were given to a hosted model.

### 9.5 Licences and acknowledgements

DFL / Bassek, Rein, Weber and Memmert (2025), *An integrated dataset of spatiotemporal and event data in elite soccer*, Scientific Data 12, 195 (https://doi.org/10.1038/s41597-025-04505-y), CC BY 4.0. Metrica Sports for the public sample data (https://github.com/metrica-sports/sample-data) and the challenge scorer (MIT), vendored unmodified in `vendor/official`. SkillCorner and PySport for the SkillCorner open data (https://github.com/SkillCorner/opendata, MIT), used for evaluation only. The competition match data are licensed for the competition and are not in the repository.

## Appendix A. Simulated tracking faults

The external tracking is clean, so each training match is presented under up to eight corruptions that use observations only, never labels:

| Scenario | What it does |
|---|---|
| `irregular40`, `irregular60` | Ball hidden on 40% / 60% of frames in gaps of 1-15 s |
| `crowded40` | 40% hidden, gaps starting preferentially when 4+ players are within 10 m of the ball |
| `slow40`, `slow60` | 40% / 60% hidden, gaps starting preferentially while median player pace is under 1.5 m/s |
| `mixed_stress` | 40% hidden, plus 25% duplicated tracks, 10% dropped player observations, 0.5 m coordinate noise |
| `target_like40`, `target_like55` (new in v7) | 41% / 55% hidden in long gaps with log-normal length (median 10.7 s, spread 0.92, 0.4-110 s), starting preferentially while players are slow (weight 4) and when the ball is within 3 m of a line (weight 15) |

Corruption seeds are derived per role, fold, match and scenario, so training and validation
never share a mask. v9 trains on seven of these scenarios (`irregular60` and `target_like55`
are left out to fit nine matches in memory) and, new in v9, adds player faults on top of every
scenario except `clean`.


The v9, v9e and v10e networks train on `clean` and six corrupted scenarios (`irregular40`, `crowded40`, `mixed_stress`, `slow40`, `slow60`, `target_like40`), with player faults added to every corrupted copy.

Why `target_like` exists. Measured on the competition ATD over graded frames, using only
observations: the ball is missing on 40.7% of frames, in about 150 gaps (median 10.7 s, p25 6.1 s,
p75 21 s, p90 34 s; 75% of missing frames lie in gaps over 15 s); it is missing on 58% of
slow-player frames against 13% of the others; and 78% of gaps begin within 3 m of a touchline or
goal line. The earlier scenarios capped gaps at 15 s. `fcb_external/target_like.py` reproduces
these three properties on clean external matches (checked on one DFL match: median 11.4 s, 72% of
missing frames in gaps over 15 s). The scenario parameters were fixed from these statistics before
any training. No labels were involved.

Why player faults exist (new in v9). `run_adversarial_check.py` compares each of the 78
input features on the competition tracking with the same feature on the simulated external
matches (two-sample KS statistic; no labels). `target_like40` is the closest scenario overall
(mean KS 0.072 against 0.216 for clean data), and the largest remaining mismatches are the
player-count features (KS 0.34-0.56). Measured on the competition ATD over graded frames: a
team has fewer than 11 tracked players on about 36% of frames (10 players 22%, 9 players 5.5%,
8 players 2.5%), in episodes of median about 4 s (p75 9-15 s, p90 35 s), and more than 11 on
5-8% of frames; clean external data shows 22 players on every frame.
`fcb_external/tracking_faults.py` reproduces that profile: per team, missing-player episodes
covering 36% of frames with log-normal length (median 4 s, spread 1.69, 0.4-120 s) that hide
one to five players, and duplicate-track episodes on 6.5% of frames. The parameters were fixed
from these statistics before any v9 training. On the Metrica sample games the shipped v7 loses
0.03-0.07 spellF1 when these faults are applied, which is what v9 is meant to recover.

## Appendix B. Development evidence

This appendix keeps the detailed tables behind section 7.1, in the form they were recorded when each version was chosen. Version labels refer to the model that was current at the time, not to v10e.

Cross-validation on the seven DFL matches (each match held out once; mean over matches;
provider labels as given):

| Scenario | v2 (earlier release) | v4 | v6 (predecessor of v7) |
|---|---|---|---|
| clean | 0.51-0.57* | 0.778 | 0.762 |
| irregular40 | 0.36-0.42* | 0.635 | 0.630 |
| crowded40 | | 0.598 | 0.595 |
| mixed_stress | | 0.549 | 0.594 |
| slow40 | | | 0.667 |

\* v2 was trained on five of these matches; its figures are for the two it never saw.
On those two matches, under the masked scenarios, v6 scores 0.558 and 0.652 against v2's
0.362 and 0.406.

v7 against v6 on the seven DFL matches (fold models; each match is scored by the fold model that
never trained on it, with the same masks for both; predictions use the shipped 1 s filter; the
truth is the provider's labels as given, which is what the official scorer sees; mean over matches):

| Scenario | v6 | v7 | change | v7 better on |
|---|---|---|---|---|
| clean | 0.741 | 0.751 | +0.009 | 6/7 |
| irregular40 | 0.618 | 0.631 | +0.013 | 6/7 |
| crowded40 | 0.592 | 0.599 | +0.006 | 5/7 |
| mixed_stress | 0.587 | 0.581 | -0.006 | 1/7 |
| slow40 | 0.653 | 0.654 | +0.001 | 3/7 |
| target_like40 | 0.620 | 0.641 | +0.020 | 7/7 |
| masked mean (5 conditions) | 0.614 | 0.621 | +0.007 | |

v6 here uses its own decoder costs (5, 10) and v7 its cross-validated costs (5, 30). Rerunning v6 with
(5, 30) gave nearly the same differences (target_like40 +0.019, irregular40 +0.013, clean +0.009,
crowded40 +0.007, slow40 +0.001, mixed_stress -0.004), so the gain does not come from the decoder
setting. The gains are small and mostly within the organisers' stated noise band of about 0.04;
the direction is consistent, and v7 is better on the target-like condition on all seven matches.
v7's costs were chosen on the same held-out cells, which is mildly optimistic. mixed_stress is
slightly worse. (An earlier comparison in our notes scored against truth that also had short spells
absorbed; it was wrong for this purpose and is not used.)

The two checkpoints that would ship, on the Metrica games (final v6 against final v7; neither
trained on these two games; each model with the costs stored in its own checkpoint):

| Scenario | v6 | v7 | change | v7 better on |
|---|---|---|---|---|
| clean | 0.752 | 0.748 | -0.004 | 1/2 |
| irregular40 | 0.641 | 0.646 | +0.004 | 2/2 |
| crowded40 | 0.621 | 0.599 | -0.021 | 0/2 |
| mixed_stress | 0.556 | 0.559 | +0.003 | 1/2 |
| slow40 | 0.659 | 0.669 | +0.011 | 1/2 |
| target_like40 | 0.612 | 0.627 | +0.015 | 2/2 |
| masked mean (5 conditions) | 0.618 | 0.620 | +0.002 | |

On these two games the two models are close: v7 is ahead on the irregular, mixed, slow and
target-like conditions, behind on clean and crowded, and the masked mean differs by 0.002. With two games this
is a tie, not a confirmation. The final v7 checkpoint is refit on all seven DFL matches, so no
held-out DFL score exists for it. Per-cell results: `reports/temporal_v7/compare_v6_rawtruth.json`,
`compare_v6_rawtruth_cost30.json`, `final_metrica_compare.json`.

v8 and v9: swap test on the Metrica games. v8 adds the Metrica sample games to the training
data; v9 adds player faults as well. Each candidate is trained on the seven DFL matches plus
one sample game and scored on the other, in both directions; the shipped v7, which trained on
neither, is scored on the same cells. Scores are masked means over five ball-loss scenarios
(irregular40, crowded40, mixed_stress, slow40, target_like40), game 1 / game 2. Four input
conditions: the ball held at its last position through stoppages or blank during them, and
all players tracked or the measured player faults applied. "w1"/"w3" is how many times the
Metrica game is counted in the sampling.

| Model | held, player faults | held, all players | blank, player faults | blank, all players | qualifies |
|---|---|---|---|---|---|
| v7 | 0.648 / 0.625 | 0.679 / 0.651 | 0.613 / 0.559 | 0.647 / 0.590 | |
| v8 w1 | 0.661 / 0.638 | 0.700 / 0.678 | 0.611 / 0.546 | 0.646 / 0.581 | no |
| v8 w3 | 0.673 / 0.652 | 0.709 / 0.692 | 0.605 / 0.545 | 0.632 / 0.583 | no |
| v9 w1 | 0.679 / 0.658 | 0.696 / 0.672 | 0.618 / 0.565 | 0.633 / 0.576 | no |
| v9 w3 | 0.687 / 0.668 | 0.704 / 0.683 | 0.625 / 0.573 | 0.642 / 0.588 | yes |

Rule: a candidate qualifies only if, in both directions, it beats v7 on held + player faults
(the condition closest to the competition tracking) and is not more than 0.01 below v7 on each
of the other three. v9 w3 is the only one that qualifies; it is ahead of v7 by 0.039 and 0.043
on the main condition and on all six scenarios there, including clean (0.820 against 0.772).
Caveats: the gains are at the edge of the organisers' noise band of about 0.04; there
are only two Metrica games; the rule was written down in this combined form after the v8 and
first v9 results on the held-ball conditions had been seen (earlier, separate rules had been
fixed before each run), though before any v9 result on the blank-ball conditions; and the
"held" scores here are higher than the Metrica numbers elsewhere in this section because those
use the blank ball. The submitted checkpoint is refit on all nine matches, so no held-out
score exists for it. Per-cell results: `reports/temporal_v8/swap.json`,
`reports/temporal_v9/swap_w1.json`, `swap_w3.json`, `reports/final_selection/results.json`;
scripts `run_retrain_v8.py`, `run_retrain_v9.py`, `run_final_selection.py`.

v9e against v9 on the test bench. Neither model trained on any bench match. Metrica game 3
figures are means over the five ball-loss scenarios (clean shown separately); SkillCorner is
the mean over 20 matches.

| Model | Game 3: held, player faults | held, all players | blank, player faults | blank, all players | clean (held, faults) | SkillCorner |
|---|---|---|---|---|---|---|
| v7 | 0.661 | 0.687 | 0.606 | 0.630 | 0.775 | 0.483 |
| v9 | 0.707 | 0.713 | 0.633 | 0.645 | 0.860 | 0.534 |
| v9e | 0.708 | 0.717 | 0.637 | 0.647 | 0.852 | 0.540 |

Rule for replacing the submitted model: beat it on game 3 held + player faults and on
SkillCorner, and lose no more than 0.01 on any other condition. v9e passes, narrowly. The differences from v9 (+0.001 and +0.005 on the two deciding columns) are far
inside the noise; v9e is 0.008 lower on clean input and 0.007 lower on the `target_like40`
scenario of the main condition (0.699 against 0.706), and it is better on 15 of the 20
SkillCorner matches and worse on 5. We do not claim v9e is more accurate than v9. The reason to
submit it is variance: single networks trained this way scored between 0.48 and 0.53 on
SkillCorner depending only on the seed, and the average varies much less from run to run
without a measured cost. The bench also confirms the v7 to v9
step on matches neither saw (+0.046 on game 3 held + player faults, +0.051 on SkillCorner).
Caveats: one Metrica game and one provider's 20 matches; the bench was built after v9 was
chosen, so it did not select v9, but it did select v9e. Per-cell results:
`reports/bench/temporal_v7_temporal_v9.json`, `temporal_v9_temporal_v9e.json`.

Metrica sample games 1 and 2 (never used for training up to v7; mean of the two) for the earlier versions:

| Pipeline | Clean | Masked (4 scenarios) | Masked, truth with <1 s spells absorbed | <2 s absorbed |
|---|---|---|---|---|
| v2 | 0.486 | 0.383 | 0.423 | 0.457 |
| v4 + 1 s filter | 0.740 | 0.579 | 0.630 | 0.628 |
| v5 (trained on relabelled targets, 2 s filter) | 0.638 | 0.504 | 0.559 | 0.606 |
| v6 + 1 s filter | 0.752 | 0.619 | 0.670 | 0.672 |
| v6 + 2 s filter | 0.681 | 0.581 | 0.639 | 0.680 |

v6 is ahead of v4 on both sample games. These Metrica numbers were produced before v7 and were used to choose v6 and the 1 s filter; v7 was selected on the DFL folds above, not on the Metrica games.


## Appendix C. Negative results

- *Averaging v6 models.* Averaging the four v6 fold models, or fold models with the final model,
  gave no gain on the Metrica games (masked spellF1 0.6193 against 0.6187 for the final model
  alone). The v9e average likewise shows no clear accuracy gain; it is kept for stability.
- *Three further model variants after v9* (extra inputs from a self-supervised model that fills
  in the hidden ball position, trained on tracking only with no labels; a teacher-student
  variant of it; and features remembering the last clear ball contact) each tied with v9 in
  swap tests or failed the bench rule (the first scored 0.704 on game 3 held + player faults
  and 0.493 on SkillCorner). They are not in the submission and their code is not in this
  repository.
- *Weight-space edit.* We tried a gradient-free edit: on each fold network, the D-row output weights
  of the top-k last-layer channels (ranked by dead-versus-live activation difference times the
  D output weight, measured on one training match) were multiplied by a gain g (k in 4, 8, 16;
  g in 1.25, 1.5, 2). Scored on fold A's held-out matches (against truth with spells under 1 s
  absorbed, applied equally to every setting), none of the nine settings improved masked
  spellF1; most were slightly worse (`reports/temporal_v7/weight_edit_select.json`,
  `run_weight_edit.py`). It is not in the submission.
- *Larger network.* A 96-channel network with dilations to 256 (receptive field about 41 s) and twice the
  training steps beat the v9 size by 0.008 to 0.022 on the four Metrica fold cells and lost 0.013 on
  SkillCorner (six-network averages). The rule required SkillCorner not to fall, so it is not in the
  submission.
- *Twenty networks.* Adding eight more seeds to the twelve gave 0.5481 on SkillCorner against 0.5484.
  The average stops improving well before twelve networks.
- *Dead-ball edge shift.* On the Metrica games v7's stoppages start a median of about 0.6 s late
  and end about 0.3 s early. Moving every dead run's start earlier and end later by fixed
  offsets gained at most 0.005 masked spellF1 (`run_dead_edge_calibration.py`,
  `reports/dead_ball_analysis/`). It is not in the submission.
- *Relabelled training targets (v5)* and *more Metrica weight without player faults (v8)* did
  not qualify under the rules above.
- *Video ball recovery.* Section 9.3. Before the experiment, a simulation on Metrica game 3 with the v9e model (`run_diag3.py`, `reports/diagnostics/diag3.txt`) restored the true ball, with 1 m noise, on 0%, 30%, 50% and 100% of the hidden frames: mean spellF1 0.714, 0.703, 0.713 and 0.738 over three masks. An oracle that replaced each matched predicted spell and its true counterpart with the true labels over their combined span reached 0.94 with the same model. The oracle uses the answers, so it is an upper bound, not an achievable score; it indicates that much of the remaining error lies in spell boundaries and spell matching rather than in ball position. No label-free way to capture that gain was found; the one tried, a fixed dead-ball edge shift, gave at most +0.005.

## Appendix D. Retraining and repository layout

To retrain from scratch (external data only; the v10e steps need a CUDA GPU with `FCB_DEVICE=cuda`):

```
python download_development.py && python prepare_development.py
python download_external_extra.py
python run_retrain_v7.py cv       # v7 folds and decoder costs, about 70 minutes
python run_retrain_v7.py final    # v7, the comparator, about 12 minutes
python run_retrain_v8.py prepare  # Metrica sample games as training data
python run_retrain_v8.py swap     # v8 swap test, both weightings
python run_retrain_v9.py swap 3   # v9 swap test (and `swap 1`)
python run_final_selection.py     # all candidates, four conditions, the rule
python run_retrain_v9.py final 3  # refit on all nine matches, about 10 minutes
python run_ensemble_v9.py train 4 # four more seeds and the v9e bundle, about 45 minutes
python download_bench.py          # Metrica game 3 and SkillCorner open data
python run_bench.py temporal_v9 temporal_v9e
FCB_DEVICE=cuda python run_step4.py feats      # cache features on disk (about 3 GB)
FCB_DEVICE=cuda python run_step4.py logo       # with / without game 3, leave-one-game-out
FCB_DEVICE=cuda python run_step4.py final 12   # the v10e bundle
python run_bench_sk.py temporal_v9e temporal_v10e   # SkillCorner comparison
```

`final` and `train` refuse to overwrite `models/temporal_v9.pt`, `models/temporal_v9e.pt` and
the seed networks under `models/temporal_v9e/`; move the shipped ones aside first (`temporal_v10e.pt` likewise). A
retrained model is bit-identical to the shipped one only in a matching environment.
`run_adversarial_check.py --data-dir <bundle>` reproduces the label-free feature comparison.

| Path | Content |
|---|---|
| `run_submission.py`, `configs/submission_final.json` | regenerate the submitted CSV |
| `fcb_external/` | tracking reader, features, network, decoder, fault simulators, submission export |
| `models/temporal_v10e.pt` | submitted checkpoint; earlier checkpoints are kept for comparison |
| `run_retrain_v7.py` to `run_retrain_v9.py`, `run_ensemble_v9.py`, `run_step4.py` | training and selection runs |
| `run_bench.py`, `run_bench_sk.py`, `download_bench.py` | external benchmarks |
| `reports/` | recorded results on external data |
| `vendor/official/` | organisers' scorer, unmodified (MIT) |
| `tests/` | unit tests |
| `docs/` | this report (`TECHNICAL_REPORT.md`) and earlier development notes, which may describe versions v1 to v6 |
