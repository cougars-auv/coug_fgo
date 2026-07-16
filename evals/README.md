# 📊 Evals

Offline evaluation and tuning tools for `coug_fgo`. Requires a built `ros2_ws` (for the
`coug_fgo_py` bindings) and the Python packages in `requirements.txt`.

Each `.sh` script is an interactive [gum](https://github.com/charmbracelet/gum) picker for
its matching `.py` script; the `.py` scripts take the same options as `argparse` flags for
scripted use. Shared defaults (namespaces, bag paths, evo flags) live in `config.py`.

## 🏃 Workflows

- **`run_offline_fgo.sh` / `run_offline_fgo.py`:** Re-run the factor graph on recorded bags
  through the `coug_fgo_py` bindings, deterministically and faster than real time. Saves a
  TUM trajectory and evo results under each bag's `evo/<namespace>/offline/` directory and
  opens state plots against ground truth.

- **`eval_bags.sh` / `eval_bags.py`:** Score the estimator topics *recorded in* bags against
  ground truth with evo (APE and RPE), then render the trajectory, timing, benchmark, and
  lag plots. Results and figures are saved under each bag's `evo/` directory.

- **`tune_scalars.py`:** Optuna sweep over sensor covariance scalars, scored by APE RMSE
  across the configured bags. Trials are stored in `scalar_tuning.db`; view them with
  **`optuna_dashboard.sh`**.

## 🗂️ Layout

- **`offline/`:** Replays bags through the offline factor graph (`coug_fgo_py` wrapper,
  message extractors, URDF transform lookups).
- **`scoring/`:** Turns trajectories into metrics (TUM file handling, evo APE/RPE runs,
  the estimator registry, and metric/param readers for recorded bags).
- **`plots/`:** Renders figures from results produced by the other two.
