# Bounded gap-evidence experiment

Reuse v2's existing offline Viterbi decoder. Test one observation-only clue: bridge a missing-ball interval only when the immediately adjacent observed frames have the same nearest team, nearest distance <=2m, a >=1m advantage over the other team, and gap <=125 frames. Never cross a sequence boundary. Fuse with the previously fixed 1.5 owner-odds multiplier, preserving dead probability. These constants are design choices, frozen before this evaluation.

Compare frozen v2 with this candidate on the original four external validation scenarios of WQQ. This is reused development evidence, not an independent test. Also evaluate inverted evidence to expose sensitivity. No possession targets enter evidence generation. No training or competition data access. Promotion is not automatic: require mean spellF1 improvement >=0.01 and no scenario drop >0.02, then broader review. Preserve v2 selection regardless of this exploratory run.

Smart Tags are covered separately by synthetic tests. Endpoint-derived evidence is a tracking proxy, not a measurement of automatic Smart Tag accuracy. Do not label this experiment as a real-tag evaluation.
