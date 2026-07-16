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

NAMESPACE = "turtlmap"
BAG_PATHS = [
    str(
        Path.home()
        / "cougars-dev/bags/turtlmap_ros2_eval/log1_ros2_eval_2026-07-16-11-57-41"
    ),
    str(
        Path.home()
        / "cougars-dev/bags/turtlmap_ros2_eval/log2_ros2_eval_2026-07-16-12-04-30"
    ),
]
EVO_FLAGS = ["--align"]  # , "--project_to_plane", "xy"]

TARGET_DIR = Path.home() / "cougars-dev/bags/turtlmap_ros2_eval"
AGENTS = ["coug1sim", "coug2sim", "coug3sim", "blue1sim", "bluerov2", "turtlmap"]


def config_paths(namespace: str) -> list[str]:
    """
    Build the layered config paths for an agent namespace.

    :param namespace: AUV namespace whose params override the fleet config.
    :return: Fleet and namespace config file paths, in override order.
    """
    return [
        str(Path.home() / "cougars-dev/config/fleet/coug_fgo_params.yaml"),
        str(Path.home() / f"cougars-dev/config/{namespace}_params.yaml"),
    ]
