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

from dataclasses import dataclass


@dataclass(frozen=True)
class Estimator:
    """
    Registry entry linking one estimator's folder, label, color, and topics.

    :param key: Evo output folder name, and the key plots look estimates up by.
    :param label: Short algorithm name shown in plot legends and tables.
    :param color: Hex color used for this estimator in every plot.
    :param topic: Odometry topic suffix exported by ``eval_bags.py``, or None.
    :param node: Metrics node base name for solver timing/lag, or None if not published.
    """

    key: str
    label: str
    color: str
    topic: str | None = None
    node: str | None = None


TRUTH_TOPIC = "odometry/truth"
GROUND_TRUTH_COLOR = "#000000"

ESTIMATORS: list[Estimator] = [
    Estimator(
        "global", "FL-B", "#55A868", topic="odometry/global", node="factor_graph_node"
    ),
    Estimator(
        "global_isam2",
        "iS2-B",
        "#DD8452",
        topic="odometry/global_isam2",
        node="factor_graph_node_isam2",
    ),
    Estimator(
        "global_lpi",
        "FL-LPI",
        "#4C72B0",
        topic="odometry/global_lpi",
        node="factor_graph_node_lpi",
    ),
    Estimator(
        "global_tpi",
        "FL-TPI",
        "#C44E52",
        topic="odometry/global_tpi",
        node="factor_graph_node_tpi",
    ),
    Estimator("global_iekf", "IEKF", "#8172B2", topic="odometry/global_iekf"),
    # Estimator("global_ukf", "UKF", "#937860", topic="odometry/global_ukf"),
    # Estimator("global_ekf", "EKF", "#DA8BC3", topic="odometry/global_ekf"),
    Estimator("global_tm", "TM", "#8C8C8C"),
    Estimator("imu", "SBG", "#CCB974", topic="imu/odometry"),
    Estimator("dvl", "DVL", "#64B5CD", topic="dvl/odometry"),
]


def exported_estimators() -> list[Estimator]:
    """Return the estimators that ``eval_bags.py`` exports from bags (have a topic)."""
    return [e for e in ESTIMATORS if e.topic is not None]


def timed_estimators() -> list[Estimator]:
    """Return the estimators that publish solver timing metrics (have a node)."""
    return [e for e in ESTIMATORS if e.node is not None]


def label_for_folder(folder: str) -> str | None:
    """Return the algorithm label for an evo output folder name, or None."""
    return next((e.label for e in ESTIMATORS if e.key == folder), None)


def label_for_row(row_key: str) -> str | None:
    """Return the label for a benchmark CSV row key by longest matching folder key."""
    for est in sorted(ESTIMATORS, key=lambda e: len(e.key), reverse=True):
        if est.key in str(row_key):
            return est.label
    return None


def color_map() -> dict[str, str]:
    """Map each algorithm label (plus ``GT``) to its plot color."""
    return {e.label: e.color for e in ESTIMATORS} | {"GT": GROUND_TRUTH_COLOR}


def labels() -> list[str]:
    """Return the algorithm labels in plot order."""
    return [e.label for e in ESTIMATORS]
