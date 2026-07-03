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

import matplotlib.pyplot as plt
import fgo_utils

NAMESPACE = "turtlmap"
BAG_PATHS = [
    str(Path.home() / "cougars-dev/bags/turtlmap_batch/log1_batch_2026-05-05-11-16-18"),
    str(Path.home() / "cougars-dev/bags/turtlmap_batch/log2_batch_2026-05-05-11-22-42"),
]
FLEET_CONFIG_PATH = str(Path.home() / "cougars-dev/config/fleet/coug_fgo_params.yaml")
AUV_CONFIG_PATH = str(Path.home() / f"cougars-dev/config/{NAMESPACE}_params.yaml")
EVO_FLAGS = ["--align"]  # , "--project_to_plane", "xy"]


def main() -> None:
    plot_args = []
    for bag in BAG_PATHS:
        results, pose_gt = fgo_utils.evaluate_and_save(
            bag, [FLEET_CONFIG_PATH, AUV_CONFIG_PATH], NAMESPACE, "batch", EVO_FLAGS
        )
        if results:
            plot_args.append((results, pose_gt, Path(bag).name))

    for results, pose_gt, label in plot_args:
        fgo_utils.plot_results(results, pose_gt, label)
    plt.show()


if __name__ == "__main__":
    main()
