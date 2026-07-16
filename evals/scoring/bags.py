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

import pandas as pd
import yaml
from rosbags.highlevel import AnyReader

from scoring import estimators

logger = logging.getLogger(__name__)

NAME_MAPPING = {e.node: e.label for e in estimators.timed_estimators()}
FGO_TOPIC = f"{estimators.timed_estimators()[0].node}/metrics"


def get_smoother_lag(bag_dir: Path, agent_name: str) -> float | None:
    """
    Read the smoother lag parameter from a bag's saved config files.

    :param bag_dir: Path to the ROS 2 bag directory.
    :param agent_name: Agent namespace used to select namespaced parameters.
    :return: The smoother lag in seconds, or None if it was not found.
    """
    config_paths = [
        bag_dir / "config" / "fleet" / "coug_fgo_params.yaml",
        bag_dir / "config" / f"{agent_name}_params.yaml",
    ]

    params: dict = {}
    for path in config_paths:
        try:
            with open(path) as f:
                config = yaml.safe_load(f)
        except (OSError, yaml.YAMLError):
            continue
        try:
            params.update(config["/**"]["ros__parameters"])
        except (KeyError, TypeError):
            pass
        try:
            params.update(config[f"/{agent_name}"]["**"]["ros__parameters"])
        except (KeyError, TypeError):
            pass

    if "smoother_lag" not in params:
        return None
    return float(params["smoother_lag"])


def read_timing_metrics(bag_dir: Path, agent_name: str) -> pd.DataFrame:
    """
    Read the solver timing metrics for one agent from a bag.

    :param bag_dir: Path to the ROS 2 bag directory.
    :param agent_name: Agent namespace to read the metrics topics for.
    :return: DataFrame of per-message timing durations by algorithm.
    """
    timing_data = []
    try:
        with AnyReader([bag_dir]) as reader:
            available_topics = {c.topic: c for c in reader.connections}
            topic_to_algo = {
                f"/{agent_name}/{node}/metrics": algo
                for node, algo in NAME_MAPPING.items()
            }
            topics_to_read = [
                available_topics[t] for t in topic_to_algo if t in available_topics
            ]
            if not topics_to_read:
                return pd.DataFrame()
            for connection, _, rawdata in reader.messages(connections=topics_to_read):
                msg = reader.deserialize(rawdata, connection.msgtype)
                timing_data.append(
                    {
                        "Algorithm": topic_to_algo[connection.topic],
                        "Total": msg.total_duration,
                        "Smoother": msg.smoother_duration,
                        "Covariance": msg.cov_duration,
                    }
                )
    except Exception as e:
        logger.warning(f"Could not read {bag_dir}: {e}")

    return pd.DataFrame(timing_data)


def read_bag_durations(bag_dir: Path, agent_name: str) -> list[float]:
    """
    Read the total solver durations from a bag's metrics topic.

    :param bag_dir: Path to the ROS 2 bag directory.
    :param agent_name: Agent namespace to read the metrics topic for.
    :return: Total optimization durations in seconds.
    """
    topic = f"/{agent_name}/{FGO_TOPIC}"
    try:
        with AnyReader([bag_dir]) as reader:
            connections = [c for c in reader.connections if c.topic == topic]
            if not connections:
                return []
            return [
                float(reader.deserialize(rawdata, c.msgtype).total_duration)
                for c, _, rawdata in reader.messages(connections=connections)
            ]
    except Exception as e:
        logger.warning(f"Could not read {bag_dir}: {e}")
        return []
