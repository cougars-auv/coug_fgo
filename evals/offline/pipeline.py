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

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from tqdm import tqdm

from offline.extractors import EXTRACTORS
from offline.graph import OfflineFactorGraph
from offline.urdf import UrdfTree, resolve_urdf_path

logger = logging.getLogger(__name__)


def process_bag_offline(
    bag_path: str,
    config_paths: list[str],
    namespace: str,
    urdf_path: str | None = None,
) -> tuple[dict | None, bool]:
    """
    Replay a bag through the offline factor graph.

    :param bag_path: Path to the ROS 2 bag directory.
    :param config_paths: Parameter YAML files, in increasing priority.
    :param namespace: AUV namespace used for topics and parameters.
    :param urdf_path: Optional URDF path, resolved from config files if omitted.
    :return: Result arrays keyed by state name (or None), and a crash flag.
    """
    if urdf_path is None:
        urdf_path = resolve_urdf_path(namespace, config_paths)

    cfg_str = "\n".join(f"  - {p}" for p in config_paths)
    logger.info(f"Loaded config files:\n{cfg_str}")
    if urdf_path:
        logger.info(f"Loaded URDF: {urdf_path}")
    else:
        logger.warning("No URDF found. Sensor TFs must come from parameters.")

    urdf = UrdfTree(urdf_path) if urdf_path else None
    graph = OfflineFactorGraph(config_paths, namespace, urdf)

    topic_to_sensors = graph.topic_map

    crashed = False
    with AnyReader(
        [Path(bag_path)], default_typestore=get_typestore(Stores.ROS2_JAZZY)
    ) as reader:
        matched_conns = [c for c in reader.connections if c.topic in topic_to_sensors]

        if matched_conns:
            conn_str = "\n".join(
                f"  - {c.topic} ({topic_to_sensors[c.topic]})" for c in matched_conns
            )
            logger.info(f"Matched sensor topics:\n{conn_str}")

            pbar = tqdm(
                reader.messages(connections=matched_conns),
                total=sum(c.msgcount for c in matched_conns),
                disable=not logger.isEnabledFor(logging.INFO),
            )

            for conn, _, rawdata in pbar:
                msg = reader.deserialize(rawdata, conn.msgtype)
                try:
                    for sensor in topic_to_sensors[conn.topic]:
                        key = (
                            "multiagent" if sensor.startswith("multiagent_") else sensor
                        )
                        frame_id, measurement = EXTRACTORS[key](msg)
                        graph.add_message(sensor, frame_id, measurement)
                except Exception as e:
                    logger.error(f"Factor graph optimization failed: {e}")
                    crashed = True
                    break
        else:
            logger.error("No matching sensor topics found in the bag.")

    if not crashed:
        try:
            graph.finalize()
        except Exception as e:
            logger.error(f"Final optimization failed: {e}")
            crashed = True
    results = graph.get_results()

    if results is None:
        if not graph.is_initialized:
            missing = graph.pending_init_sensors()
            if missing:
                logger.error(
                    f"Graph never initialized. No data received for: {', '.join(missing)}."
                )
            else:
                logger.error(
                    "Graph never initialized. Not enough sensor data in the bag."
                )
        else:
            logger.error("Graph initialized but produced no results.")

    return results, crashed
