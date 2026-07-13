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

import logging
from pathlib import Path

import matplotlib.pyplot as plt
from tqdm.contrib.logging import logging_redirect_tqdm

from plots import state_plots
from utils import pipeline
from utils.log_setup import setup_logging

logger = logging.getLogger(__name__)

NAMESPACE = "turtlmap"
BAG_PATHS = [
    str(
        Path.home()
        / "cougars-dev/bags/turtlmap_offline/log1_offline_2026-07-06-16-08-28"
    ),
    str(
        Path.home()
        / "cougars-dev/bags/turtlmap_offline/log2_offline_2026-07-06-16-15-19"
    ),
]
CONFIG_PATHS = [
    str(Path.home() / "cougars-dev/config/fleet/coug_fgo_params.yaml"),
    str(Path.home() / f"cougars-dev/config/{NAMESPACE}_params.yaml"),
]
EVO_FLAGS = ["--align"]  # , "--project_to_plane", "xy"]


def main() -> None:
    setup_logging()

    plot_args = []
    with logging_redirect_tqdm():
        for bag in BAG_PATHS:
            result = pipeline.process_and_evaluate(
                bag, CONFIG_PATHS, NAMESPACE, "offline", EVO_FLAGS
            )
            if result is not None:
                plot_args.append(result)

    for results, pose_gt, label in plot_args:
        state_plots.plot_results(results, pose_gt, label)
    plt.show()


if __name__ == "__main__":
    main()
