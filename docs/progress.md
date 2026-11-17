# Execution ledger — FCB next-step implementation prompt

Spec: `outputs/FCB_Next_Step_Implementation_Prompt.md` (Library version 0).

Workspace is a new standalone directory; no existing Git branch or tracked source is modified. Use source-only checkpoints rather than creating a worktree without an existing repository.

Pre-flight: adapter produces Observations and separate Targets; baseline consumes Observations only; evaluator consumes aligned targets/predictions plus explicit sequence breaks. No shared target dependency in inference.

Acquisition: Python DNS failed for Hugging Face and the alternate Figshare API. Public web retrieval verified metadata schema and tracking SHA256; shell cannot download bytes. Building source/synthetic tests while preparing a user-runnable downloader. Real-match verification remains pending until bytes arrive.

Update: user executed the downloader in their own PowerShell. All three assets arrived; tracking matched published SHA256. Real conversion and all four baseline/evaluation variants succeeded. Actual status domains are 0/1 and ownership 1/2, with 145,967 valid targets and two 25 Hz sequences.

Tasks complete: source verification/acquisition; external adapter; fixed baseline variants; unchanged official metric adapter; CLI; 31 synthetic tests passed. Full real-data deterministic rerun completed: 17 canonical/target/prediction/report files matched SHA256 exactly, with no mismatches. Details are in `docs/reproduction_check.json`.

Independent review: no P0/P1 findings. Two P2 provenance defects were found: incomplete XML could be promoted after a prefix check, and event XML from a different match could be accepted. Both received failing regression tests, streaming completeness/match validation fixes, and a green whole suite. Added direct raw-nearest-player equivalence to the official implementation as well as smoothing equivalence.

No Git repository was modified. The final source-only archive is the checkpoint; official public code licence is retained. No competition data/outputs/videos/ground truth have been accessed in this milestone.

Ruling: use J03WMX metadata plus entity/frame structure documented in floodlight's DFL parser as the synthetic contract. Do not claim real label domains or end-to-end success until actual external tracking is parsed. A schema difference must fail visibly, not be silently guessed.

Ruling: stream XML Frame records in two passes, clearing each processed element; use a union of observation frame keys independent of targets. Two passes avoid assuming ball-first ordering. Observation timestamps and frame IDs both determine discontinuities.
