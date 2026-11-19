# Quadrotor prediction-initialized PSO: code and data

Completed simulation and processor-in-the-loop records. This branch adds the seven-method benchmark, its new-configuration validation, and original hardware replay. No manuscript or submission material is included in this update.

## Contents

- `simulation/`: plant, controller, objective, PSO and shared configuration.
- `predictors/`: frozen NN ensemble and GP, training code and recorded training results.
- `configurations/`: original 20 training and 10 test plant files. New Drone_41--50 parameters are in `rico_benchmark/new_configurations/plants.json`.
- `original_study/`: original five-method cohort, 150 controllers and evaluation ledgers.
- `rico_benchmark/`: first seven-method benchmark, 210 runs, histories, gains, statistics and figures.
- `rico_benchmark/new_configurations/`: 30 new long references, 210 validation runs, histories, audit and comparison tables.
- `hil/firmware/`: custom PX4 controller, modified source and firmware version manifests.
- `hil/host/`: host scripts, experimental configurations and frozen gains.
- `hil/replay_logs/`: all recorded attempts and traces for 900 planned scenarios (892 timing-valid responses).
- `hil/analysis/`: cohort/roll analysis and saved numerical outputs.

## Use

Clone this branch, inspect the CSV/JSON results directly, and use the recorded requirements for the relevant environment. Root requirements describe the benchmark runtime; `predictors/requirements-original.txt` specifies the historical predictor environment and `hil/host/requirements-lock.txt` records hardware-host dependencies.

Historical scripts retain their original path assumptions, with personal prefixes replaced by `USER_HOME`, `WORKSPACE`, `SOURCE_STUDY` or `SOURCE_HIL`. Configure those paths before rerunning execution scripts. No training, simulation, or hardware run is required to inspect the archived results. `RELEASE_MANIFEST.json` records SHA-256 hashes of added release data and code. Numerical records were retained; personal path strings were sanitized.

The new validation audit passed all 210 exact 1800-evaluation budgets, including OBL screening, and 30 exact 3264-evaluation references. Final passes were random 13/30, LHS 15/30, OBL 15/30, chaotic 8/30, prediction 22/30, library 24/30 and hybrid 25/30. Hybrid reached random PSO's final median cost after 402 evaluations. Hardware evidence covers only the original five-method cohort; no new hardware experiments were run.

Original code: MIT (`LICENSE`). Study data: CC BY 4.0 (`LICENSE-DATA`). PX4 and other third-party code retain upstream licenses. No publication DOI is claimed.
