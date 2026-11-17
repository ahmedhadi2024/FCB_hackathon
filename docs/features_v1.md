# Model inputs and units (v1)

Only common observation arrays enter feature extraction. Targets, events, names, shirt numbers, stable identities, period clock and ball height are excluded. Every temporal operation resets at observation sequence breaks. Context is offline/centred where specified.

| Feature | Definition |
|---|---|
| ball_observed | 1 only if both x/y are finite |
| players_present_per_22 | Observed slots / 22, capped at 2; not divided by provider roster size |
| min/max_team_players_per_11 | Smaller/larger observed team count / 11; home/away symmetric |
| ball_abs_x/y | Absolute centred coordinate / pitch half-length or half-width |
| distance_to_boundary | Minimum signed distance to any rectangular pitch edge / 10 m |
| nearest/farther_team_distance | Minimum distance to a player in each team, sorted, / 10 m capped at 10 |
| players_within_5m_per_4 | Count of observed players less than 5 m from observed ball / 4 |
| speed_1/5/25 | Ball displacement / elapsed time at lag 1/5/25 frames, capped at 100 m/s then divided by 20 |
| speed_*_valid | Both endpoints observed and in the same contiguous sequence |
| player_centroid_speed | Displacement of observed-player centroid over 5 frames, in m/s / 10 |
| ball_seen_1s | Observed fraction in centred +/-25-frame context (1-second HALF-width) |
| ball_seen_5s | Observed fraction in centred +/-125-frame context (5-second HALF-width) |
| ball_speed_mean_1s | Mean scaled one-frame speed in centred +/-12-frame context (25-frame full window) |

Missing geometry uses finite zero placeholders accompanied by observation/validity/context flags. No ball or player trajectory is reconstructed from labels. The model's centring and scaling parameters are fitted only to sampled training features, including predefined label-independent observation corruptions.

DFL is already centred in metres. The synthetic ATD adapter explicitly converts normalized top-left image axes to centred Cartesian coordinates (x*length-length/2, width/2-y*width). The dead-state features use magnitudes, distances and symmetric team counts rather than provider-specific attack direction. The ATD adapter is tested on fictional files only in this milestone, not on competition files.
