# Common observation contract, schema 1

`Observations` contains match ID, pitch dimensions (metres), fps=25, period, original frame ID, time in seconds from the first observed frame of each period, player x/y (T,P,2), player-observed mask (T,P), fixed slot-team array (P), ball x/y (T,2), ball-observed mask (T), and observation sequence ID (T).

Coordinates are DFL's centred pitch coordinates in metres; X spans pitch length and Y spans width. No per-half flip or attack-direction normalization is performed. Home/away is read from `MatchInformation/General` HomeTeamId and GuestTeamId/AwayTeamId; player membership comes from Teams/Team/Players/Player. Team and entity XML order cannot assign roles. Stable slots sort home first then away then PersonId; IDs are storage alignment, not permitted model identity features.

Observations are keyed by (GameSection,N), derived from the UNION of known team-player and ball records, independently of target values. Reordered records are joined by keys. Explicit non-team objects such as referees are excluded. Repeated entity/frame keys fail. Conflicting timestamps for a shared key fail. Matching source MatchId fields are checked when present.

UTC timestamp T must carry an offset. Within a period the difference between adjacent observed keys must equal the frame-ID difference / 25 within one microsecond; otherwise conversion fails. Seconds are relative to the first observation, not claimed to be the exact kickoff time. Period changes and nonconsecutive frame IDs start new sequences. No interpolation or imputation is performed. Removed observation rows are never relabelled or filled using targets.

NaN/nonfinite or incomplete coordinate pairs become NaN pairs with False observation masks. Zero is a valid coordinate. Observation masks must exactly match finite coordinate pairs. Source speeds, ball height, status, ownership, event tags and target validity are excluded from observations.

One `.npy` per observation array supports memory-mapped reads. `metadata.json` records the schema version and timeline hash. Targets are separate label/validity arrays; their metadata must bind to the same timeline. Prediction provenance binds BOTH timeline and complete observation-array hashes, so changed coordinates cannot silently reuse earlier predictions.

Official evaluation uses 25 fps durations and frame tolerances. X separators are inserted between observation sequences only in scoring arrays; neither canonical files nor their row numbers change. Unknown target rows stay in place and ungraded. Predicted X follows the original official abstention-filling policy.
