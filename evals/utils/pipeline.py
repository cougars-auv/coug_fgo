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
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import yaml
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from tqdm import tqdm

from utils import evo_tools
from utils.factor_graph import OfflineFactorGraph
from utils.urdf import UrdfTree, resolve_urdf_path

logger = logging.getLogger(__name__)


def _stamp(m) -> float:
    """
    Return a message header stamp as seconds since the epoch.

    :param m: Message with a std_msgs/Header field.
    :return: Timestamp in seconds.
    """
    return m.header.stamp.sec + m.header.stamp.nanosec * 1e-9


def _vec3(v) -> np.ndarray:
    """
    Return a Vector3-like message as a numpy array.

    :param v: Message with x, y, and z fields.
    :return: The values as a length-3 numpy array.
    """
    return np.array([v.x, v.y, v.z])


def _quat(q) -> np.ndarray:
    """
    Return a Quaternion message as an xyzw numpy array.

    :param q: Message with x, y, z, and w fields.
    :return: The values as an xyzw numpy array.
    """
    return np.array([q.x, q.y, q.z, q.w])


def _cov(arr, n: int) -> np.ndarray:
    """
    Return a flat covariance field as an n-by-n numpy array.

    :param arr: Flat row-major covariance values.
    :param n: Side length of the square covariance matrix.
    :return: The values reshaped to an n-by-n numpy array.
    """
    return np.array(arr).reshape(n, n)


EXTRACTORS = {
    "imu": lambda m: (
        m.header.frame_id,
        (
            _stamp(m),
            _vec3(m.linear_acceleration),
            _vec3(m.angular_velocity),
            _cov(m.linear_acceleration_covariance, 3),
            _cov(m.angular_velocity_covariance, 3),
        ),
    ),
    "gps": lambda m: (
        m.child_frame_id,
        (_stamp(m), _vec3(m.pose.pose.position), _cov(m.pose.covariance, 6)),
    ),
    "depth": lambda m: (
        m.child_frame_id,
        (_stamp(m), m.pose.pose.position.z, _cov(m.pose.covariance, 6)),
    ),
    "mag": lambda m: (
        m.header.frame_id,
        (_stamp(m), _vec3(m.magnetic_field), _cov(m.magnetic_field_covariance, 3)),
    ),
    "ahrs": lambda m: (
        m.header.frame_id,
        (_stamp(m), _quat(m.orientation), _cov(m.orientation_covariance, 3)),
    ),
    "dvl": lambda m: (
        m.header.frame_id,
        (_stamp(m), _vec3(m.twist.twist.linear), _cov(m.twist.covariance, 6)),
    ),
    "wrench": lambda m: (
        m.header.frame_id,
        (_stamp(m), np.concatenate([_vec3(m.wrench.force), _vec3(m.wrench.torque)])),
    ),
}


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
        else:
            logger.error("No matching sensor topics found in the bag.")

        pbar = tqdm(
            reader.messages(connections=matched_conns),
            total=sum(c.msgcount for c in matched_conns),
            disable=not logger.isEnabledFor(logging.INFO),
        )

        for conn, _, rawdata in pbar:
            msg = reader.deserialize(rawdata, conn.msgtype)
            try:
                for sensor in topic_to_sensors[conn.topic]:
                    frame_id, measurement = EXTRACTORS[sensor](msg)
                    graph.add_message(sensor, frame_id, measurement)
            except Exception as e:
                logger.error(f"Factor graph optimization failed: {e}")
                crashed = True
                break

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
                    f"Graph never initialized. No data received for: {', '.join(missing)}"
                )
            else:
                logger.error(
                    "Graph never initialized. Not enough sensor data in the bag."
                )
        else:
            logger.error("Graph initialized but produced no results.")

    return results, crashed


def process_and_evaluate(
    bag_path: str,
    config_paths: list[str],
    namespace: str,
    tag: str,
    evo_flags: list[str],
    **kwargs,
) -> tuple[dict, dict, str] | None:
    """
    Run the full offline pipeline for one bag: load truth, process, save, evaluate.

    :param bag_path: Path to the ROS 2 bag directory.
    :param config_paths: Parameter YAML files, in increasing priority.
    :param namespace: AUV namespace used for topics and parameters.
    :param tag: Subdirectory and file suffix for this run (e.g. ``offline``).
    :param evo_flags: Extra evo flags forwarded to the APE and RPE runs.
    :param kwargs: Extra keyword arguments forwarded to ``process_bag_offline``.
    :return: ``(results, pose_gt, label)`` tuple, or None if no results.
    """
    logger.info(f"Processing bag: {bag_path}")
    pose_gt = evo_tools.load_ground_truth(bag_path, namespace)
    results, _ = process_bag_offline(bag_path, config_paths, namespace, **kwargs)
    if not results:
        return None

    evo_dir = evo_tools.evo_agent_dir(bag_path, namespace) / tag
    evo_dir.mkdir(parents=True, exist_ok=True)
    est_path = evo_dir / f"{namespace}_{tag}.tum"
    evo_tools.save_tum(est_path, results)

    gt_path = evo_tools.ensure_ground_truth(bag_path, namespace)
    if pose_gt and gt_path is not None:
        evo_tools.run_evo_evaluations(gt_path, est_path, evo_dir, evo_flags)

    return results, pose_gt, Path(bag_path).name


@contextmanager
def covariance_override_file(scalars: dict[str, float]):
    """
    Yield a temporary params file overriding sensor covariance scalars.

    :param scalars: Mapping of sensor name to covariance scalar value.
    """
    override = {
        "/**": {
            "ros__parameters": {s: {"covariance_scalar": v} for s, v in scalars.items()}
        }
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(override, f)
        override_path = f.name
    try:
        yield override_path
    finally:
        Path(override_path).unlink()
