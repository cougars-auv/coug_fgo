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

from pathlib import Path

import fgo_utils

NAMESPACE = "bluerov2"
BAG_PATHS = [
    str(Path.home() / "cougars-dev/bags/bluerov2_dropout_2026-03-31-12-00-00"),
    str(Path.home() / "cougars-dev/bags/bluerov2_dropout_2026-03-31-11-05-33"),
]
FLEET_CONFIG_PATH = str(Path.home() / "cougars-dev/config/fleet/coug_fgo_params.yaml")
AUV_CONFIG_PATH = str(Path.home() / f"cougars-dev/config/{NAMESPACE}_params.yaml")
EVO_FLAGS = ["--align", "--project_to_plane", "xy"]


def process_bag(bag_path: str) -> None:
    print(f"\nProcessing bag: {bag_path}")

    print("\n--- Ground Truth ---")
    pose_gt, vel_gt, bias_gt = fgo_utils.load_ground_truth(bag_path, NAMESPACE)

    print("\n--- Factor Graph Optimization ---\n")
    results, _ = fgo_utils.run_factor_graph(
        bag_path, [FLEET_CONFIG_PATH, AUV_CONFIG_PATH], NAMESPACE
    )

    print("\n--- Saving TUM Files ---\n")
    evo_dir = Path(bag_path) / "evo" / NAMESPACE / "odometry" / "batch"
    evo_dir.mkdir(parents=True, exist_ok=True)
    if results:
        fgo_utils.write_tum(evo_dir / "batch.tum", results)
    if pose_gt:
        fgo_utils.write_tum(evo_dir / "ground_truth.tum", pose_gt)

    print("\n--- Evo Evaluation ---")
    if results and pose_gt:
        fgo_utils.run_evo_evaluations(
            str(evo_dir / "ground_truth.tum"),
            str(evo_dir / "batch.tum"),
            evo_dir,
            EVO_FLAGS,
        )

    print("\n--- Plotting ---\n")
    if results:
        fgo_utils.plot_results(results, pose_gt, vel_gt, bias_gt)


def main() -> None:
    for bag in BAG_PATHS:
        if not Path(bag).exists():
            print(f"\nBag not found: {bag}")
            continue

        process_bag(bag)


if __name__ == "__main__":
    main()
