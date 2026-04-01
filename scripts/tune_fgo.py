#!/usr/bin/env python3
# Copyright (c) 2026 BYU FROST Lab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import atexit
import subprocess
from datetime import datetime
from pathlib import Path

import optuna

import fgo_utils

NAMESPACE = "bluerov2"
BAG_PATH = str(Path.home() / "cougars-dev/bags/bluerov2_dropout_2026-03-31-11-05-33")
FLEET_CONFIG_PATH = str(Path.home() / "cougars-dev/config/fleet/coug_fgo_params.yaml")
AUV_CONFIG_PATH = str(Path.home() / f"cougars-dev/config/{NAMESPACE}_params.yaml")
SCRIPTS_PATH = str(Path.home() / "cougars-dev/ros2_ws/src/coug_fgo/scripts")
EVO_FLAGS = ["--align", "--project_to_plane", "xy"]

DB_URL = f"sqlite:///{SCRIPTS_PATH}/optuna_fgo.db"
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
STUDY_NAME = f"{NAMESPACE}_scalar_sweep_{timestamp}"
SCALARS_TO_TUNE = ["dvl", "const_vel"]
N_OPTUNA_TRIALS = 50
MIN_SCALAR = 0.5
MAX_SCALAR = 100


def main() -> None:
    print("\n--- Ground Truth ---")
    pose_gt, vel_gt, bias_gt = fgo_utils.load_ground_truth(BAG_PATH, NAMESPACE)

    if not pose_gt:
        raise RuntimeError("No ground truth found in bag. Aborting.")

    def objective(trial: optuna.Trial) -> float:
        scalars = {
            s: trial.suggest_float(s, MIN_SCALAR, MAX_SCALAR, log=True)
            for s in SCALARS_TO_TUNE
        }
        with fgo_utils.param_override_file(scalars) as override_path:
            results, crashed = fgo_utils.run_factor_graph(
                BAG_PATH,
                [FLEET_CONFIG_PATH, AUV_CONFIG_PATH, override_path],
                NAMESPACE,
                verbose=False,
            )
        return fgo_utils.compute_ape_rmse(pose_gt, results, crashed)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=STUDY_NAME, storage=DB_URL, direction="minimize"
    )

    print("\n--- Starting Optuna Dashboard ---\n")
    subprocess.run(["pkill", "-9", "-f", "optuna-dashboard"], capture_output=True)
    dashboard_process = subprocess.Popen(
        ["optuna-dashboard", DB_URL, "--host", "0.0.0.0", "--port", "9000"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    atexit.register(dashboard_process.terminate)
    print("Dashboard running at http://localhost:9000.")

    print("\n--- Running Optuna Trials ---\n")
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)
    print(f"\nResults: {study.best_params}")

    print("\n--- Factor Graph Optimization ---\n")
    with fgo_utils.param_override_file(study.best_params) as best_override_path:
        results, _ = fgo_utils.run_factor_graph(
            BAG_PATH,
            [FLEET_CONFIG_PATH, AUV_CONFIG_PATH, best_override_path],
            NAMESPACE,
        )

    print("\n--- Saving TUM Files ---\n")
    evo_dir = Path(BAG_PATH) / "evo" / NAMESPACE / "odometry" / "tuned"
    evo_dir.mkdir(parents=True, exist_ok=True)
    if results:
        fgo_utils.write_tum(evo_dir / "tuned.tum", results)
    if pose_gt:
        fgo_utils.write_tum(evo_dir / "ground_truth.tum", pose_gt)

    print("\n--- Evo Evaluation ---")
    if results and pose_gt:
        fgo_utils.run_evo_evaluations(
            str(evo_dir / "ground_truth.tum"),
            str(evo_dir / "tuned.tum"),
            evo_dir,
            EVO_FLAGS,
        )

    print("\n--- Plotting ---\n")
    if results:
        fgo_utils.plot_results(results, pose_gt, vel_gt, bias_gt)


if __name__ == "__main__":
    main()
