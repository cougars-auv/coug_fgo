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
import math
import subprocess
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import optuna

import fgo_utils

NAMESPACE = "bluerov2"
BAG_PATHS = [
    str(Path.home() / "cougars-dev/bags/dropout_1.0_2026-04-01-15-47-59"),
    str(Path.home() / "cougars-dev/bags/dropout_5.0_2026-04-01-15-42-34"),
    str(Path.home() / "cougars-dev/bags/dropout_7.0_2026-04-01-15-36-25"),
]
FLEET_CONFIG_PATH = str(Path.home() / "cougars-dev/config/fleet/coug_fgo_params.yaml")
AUV_CONFIG_PATH = str(Path.home() / f"cougars-dev/config/{NAMESPACE}_params.yaml")
SCRIPTS_PATH = str(Path.home() / "cougars-dev/ros2_ws/src/coug_fgo/scripts")
EVO_FLAGS = ["--align", "--project_to_plane", "xy"]

DB_URL = f"sqlite:///{SCRIPTS_PATH}/optuna_fgo.db"
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
STUDY_NAME = f"{NAMESPACE}_scalar_sweep_{timestamp}"
SCALARS_TO_TUNE = ["dvl"]
N_OPTUNA_TRIALS = 30
MIN_SCALAR = 0.01
MAX_SCALAR = 100


def main() -> None:
    print("\n--- Ground Truth ---")
    ground_truths = []
    for bag_path in BAG_PATHS:
        if not Path(bag_path).exists():
            print(f"\nBag not found: {bag_path}")
            ground_truths.append((None, None, None))
            continue
        pose_gt, vel_gt, bias_gt = fgo_utils.load_ground_truth(bag_path, NAMESPACE)
        if not pose_gt:
            raise RuntimeError(f"No ground truth found in {bag_path}. Aborting.")
        ground_truths.append((pose_gt, vel_gt, bias_gt))

    def objective(trial: optuna.Trial) -> float:
        scalars = {
            s: trial.suggest_float(s, MIN_SCALAR, MAX_SCALAR, log=True)
            for s in SCALARS_TO_TUNE
        }
        rmses = []
        with fgo_utils.param_override_file(scalars) as override_path:
            for bag_path, (pose_gt, _, _) in zip(BAG_PATHS, ground_truths):
                if pose_gt is None:
                    continue
                results, crashed = fgo_utils.run_factor_graph(
                    bag_path,
                    [FLEET_CONFIG_PATH, AUV_CONFIG_PATH, override_path],
                    NAMESPACE,
                    verbose=False,
                )
                rmses.append(fgo_utils.compute_ape_rmse(pose_gt, results, crashed))
        if not rmses:
            return float("inf")
        return math.sqrt(sum(r**2 for r in rmses) / len(rmses))

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

    plot_args = []
    print("\n--- Factor Graph Optimization ---")
    with fgo_utils.param_override_file(study.best_params) as best_override_path:
        for bag_path, (pose_gt, vel_gt, bias_gt) in zip(BAG_PATHS, ground_truths):
            if pose_gt is None:
                continue
            print(f"\nProcessing bag: {bag_path}\n")
            results, _ = fgo_utils.run_factor_graph(
                bag_path,
                [FLEET_CONFIG_PATH, AUV_CONFIG_PATH, best_override_path],
                NAMESPACE,
            )

            print("\n--- Saving TUM Files ---\n")
            evo_dir = Path(bag_path) / "evo" / NAMESPACE / "odometry" / "tuned"
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

            if results:
                plot_args.append(
                    (results, pose_gt, vel_gt, bias_gt, Path(bag_path).name)
                )

    print("\n--- Plotting ---\n")
    for results, pose_gt, vel_gt, bias_gt, label in plot_args:
        fgo_utils.plot_results(results, pose_gt, vel_gt, bias_gt, label)
    print("Displaying plots...")
    plt.show()


if __name__ == "__main__":
    main()
