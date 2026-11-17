# Ball Knowledge: frame-level ball possession (More than a Hack 2027, Phase 1)

Labels every frame of the tactical video `H`, `A`, `D` (or `X` outside the graded halves) from the
provided automatic tracking only. Twelve small temporal convolutional networks trained on other matches
(seven DFL/IDSSE Bundesliga matches, CC BY 4.0, and Metrica Sports' three public sample games) give
per-frame probabilities, which are averaged; a Viterbi decoder, a 1 s controlled-possession filter and a Smart Tagging
kick-off rule produce the possession file. The submitted pipeline reads only the tactical tracking, its
metadata and the Smart Tagging file; it does not read video, audio, broadcast data or the homography, and
no public ground-truth labels were used for training, tuning or selection. An abandoned development
experiment that processed the tactical video is disclosed in section 9.3 of the technical report.

The technical document is `docs/TECHNICAL_REPORT.md` (also submitted as `Ball Knowledge.md`).
Older documents under `docs/` are historical development notes and may describe earlier versions
(v1 to v6); the submitted model is v10e, the average of twelve networks trained on ten external matches (seven DFL, three Metrica).

## Reproduce the submitted file

```
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-submission.txt
.venv/Scripts/python run_submission.py --data-dir <bundle> --out "Ball Knowledge.csv"
```

`<bundle>` is the organisers' data folder (`tactical/BAR-ATH_tactical_atd.txt`,
`..._atd.xml`, `..._smart_tagging.xml`). The match data is not in this repository and must not be
added. The default config `configs/submission_final.json` pins the model
(`models/temporal_v10e.pt`) by SHA256. Expected SHA256 of the output:

```
e947dbddce6566e3766756d87214015ef977f4c1fcc2f096a29e0221a86aaa68
```

Reference environment: Windows 11 x64, Python 3.14.2, numpy 2.4.4, torch 2.11.0+cpu. PyTorch
results can differ across CPU types and library versions.

## Retrain (external data only, a few hours on a laptop CPU with 16 GB of memory)

```
python download_development.py && python prepare_development.py
python download_external_extra.py
python run_retrain_v7.py cv
python run_retrain_v7.py final
python run_retrain_v8.py prepare
python run_retrain_v8.py swap
python run_retrain_v9.py swap 3
python run_final_selection.py
python run_retrain_v9.py final 3
python run_ensemble_v9.py train 4
python download_bench.py
python run_bench.py temporal_v9 temporal_v9e
FCB_DEVICE=cuda python run_step4.py feats
FCB_DEVICE=cuda python run_step4.py logo
FCB_DEVICE=cuda python run_step4.py final 12
python run_bench_sk.py temporal_v9e temporal_v10e
```

`final` and `train` refuse to overwrite a shipped model; move `models/temporal_v7.pt`,
`models/temporal_v9.pt`, `models/temporal_v9e.pt`, `models/temporal_v9e/` and `models/temporal_v10e.pt` aside first if you retrain. The external data (DFL/IDSSE, Metrica public
samples, SkillCorner open data) is downloaded by the scripts to `external-data/` and `artifacts/`, which are git-ignored.

## Layout

- `fcb_external/` source: tracking reader, features, network, decoder and fault simulations
  (`target_like.py` models the ball-gap profile and `tracking_faults.py` the missing-player
  profile measured on the competition tracking, both label-free).
- `run_retrain_v8.py`, `run_retrain_v9.py`, `run_final_selection.py` the swap tests and the
  selection rule behind v9; `run_ensemble_v9.py` the four extra seeds and the v9e bundle;
  `run_bench.py` the test bench on matches no model trained on (Metrica sample game 3 and
  SkillCorner open data), which selected v9e; `run_step4.py` the leave-one-game-out test of
  adding game 3 and the v10e bundle; `run_diag3.py` the ball-recovery simulation; `run_adversarial_check.py` the label-free comparison
  of inputs between the competition tracking and the simulated training data.
- `configs/` run configurations; `models/` frozen checkpoints; `reports/` external-data evidence.
- `vendor/official/` the organisers' scorer, vendored unmodified (MIT).
- `tests/` unit tests.

Licences: DFL/IDSSE data CC BY 4.0 (Bassek et al. 2025); Metrica sample data used under its public
terms, with acknowledgement; SkillCorner open data MIT, used for testing only. The competition match data is licensed for the competition only and
is not included.
