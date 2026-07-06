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

import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from pathlib import Path

import logging

import numpy as np
import yaml
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation
from tqdm import tqdm

FGO_LIB_PATH = str(
    Path.home()
    / "cougars-dev/ros2_ws/install/coug_fgo/lib"
    / f"python{sys.version_info.major}.{sys.version_info.minor}"
    / "site-packages"
)
sys.path.insert(0, FGO_LIB_PATH)
import pybind11fgo  # noqa: E402

logger = logging.getLogger(__name__)

SENSORS = ("imu", "gps", "depth", "mag", "ahrs", "dvl", "wrench")


class UrdfTree:
    def __init__(self, urdf_path: str):
        path = Path(urdf_path)
        if path.suffix == ".xacro":
            import xacro

            xml_text = xacro.process_file(str(path)).toxml()
        else:
            xml_text = path.read_text()

        self._joints: dict[str, tuple[str, np.ndarray, Rotation]] = {}
        self._links: set[str] = set()
        for joint in ET.fromstring(xml_text).findall("joint"):
            parent = joint.find("parent").attrib["link"]
            child = joint.find("child").attrib["link"]
            origin = joint.find("origin")
            attrib = origin.attrib if origin is not None else {}
            pos = np.array([float(v) for v in attrib.get("xyz", "0 0 0").split()])
            rpy = [float(v) for v in attrib.get("rpy", "0 0 0").split()]
            self._joints[child] = (parent, pos, Rotation.from_euler("xyz", rpy))
            self._links.update((parent, child))

    def lookup(
        self, target_frame: str, source_frame: str
    ) -> tuple[np.ndarray, np.ndarray]:
        target_pos, target_rot = self._root_tf(target_frame)
        source_pos, source_rot = self._root_tf(source_frame)
        pos = target_rot.inv().apply(source_pos - target_pos)
        rot = target_rot.inv() * source_rot
        return pos, rot.as_quat()

    def _root_tf(self, frame: str) -> tuple[np.ndarray, Rotation]:
        link = frame.split("/")[-1]  # Strip robot_state_publisher frame_prefix
        if link not in self._links:
            raise KeyError(f"Frame '{frame}' not found in the URDF!")

        pos, rot = np.zeros(3), Rotation.identity()
        while link in self._joints:
            link, j_pos, j_rot = self._joints[link]
            pos = j_pos + j_rot.apply(pos)
            rot = j_rot * rot
        return pos, rot


def resolve_urdf_path(namespace: str, config_paths: list[str]) -> str | None:
    def read_urdf_file(yaml_path: Path, top_keys: list[str]) -> str | None:
        try:
            data = yaml.safe_load(yaml_path.read_text())
        except OSError:
            return None
        for top_key in top_keys:
            try:
                return data[top_key]["coug_description_launch"]["ros__parameters"][
                    "urdf_file"
                ]
            except (KeyError, TypeError):
                continue
        return None

    urdf_file = None
    for path in map(Path, config_paths):
        urdf_file = read_urdf_file(path, [f"/{namespace}", "/**"]) or urdf_file
    if urdf_file is None:
        for path in map(Path, config_paths):
            for config_dir in (path.parent / "fleet", path.parent):
                fleet_path = config_dir / "coug_description_params.yaml"
                urdf_file = urdf_file or read_urdf_file(fleet_path, ["/**"])
    if urdf_file is None:
        urdf_file = "couguv_holoocean.urdf.xacro"

    urdf_dirs = [
        Path.home() / "cougars-dev/ros2_ws/src/coug_description/coug_description/urdf",
        Path.home()
        / "cougars-dev/ros2_ws/install/coug_description/share/coug_description/urdf",
    ]
    try:
        from ament_index_python.packages import get_package_share_directory

        urdf_dirs.insert(
            0, Path(get_package_share_directory("coug_description")) / "urdf"
        )
    except Exception:
        pass

    for urdf_dir in urdf_dirs:
        if (urdf_dir / urdf_file).is_file():
            return str(urdf_dir / urdf_file)
    return None


# =============================================================================
# ROS DATA EXTRACTION
# =============================================================================


def _stamp(m) -> float:
    return m.header.stamp.sec + m.header.stamp.nanosec * 1e-9


def _vec3(v) -> np.ndarray:
    return np.array([v.x, v.y, v.z])


def _quat(q) -> np.ndarray:
    return np.array([q.x, q.y, q.z, q.w])


def _cov(arr, n: int) -> np.ndarray:
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


# =============================================================================
# OFFLINE FACTOR GRAPH PIPELINE
# =============================================================================


class OfflineFactorGraph:
    # IMPORTANT! Offline, the timer relies on IMU message stamps instead of the ROS 2 clock
    SOURCE_SENSORS = {"DVL": "dvl", "Depth": "depth", "Timer": "imu"}

    def __init__(
        self, config_paths: list[str], namespace: str = "", urdf: UrdfTree | None = None
    ):
        self.fg = pybind11fgo.FactorGraphPy(config_paths, namespace)
        self.params = self.fg.get_params()
        self.urdf = urdf

        sensors = self.params["sensors"]
        self.loose_preint = self.params["comparison"]["enable_loose_dvl_preintegration"]

        def subscribed(key: str) -> bool:
            return sensors[key]["enable"] or sensors[key]["enable_extra_only"]

        self.enabled = {
            "imu": True,
            "gps": subscribed("gps"),
            "depth": subscribed("depth"),
            "mag": subscribed("mag"),
            "ahrs": subscribed("ahrs") or self.loose_preint,
            "dvl": subscribed("dvl"),
            "wrench": subscribed("dynamics"),
        }

        self.is_lm = self.params["solver_type"] == "LevenbergMarquardt"
        if self.is_lm and not self.params["publish_smoothed_path"]:
            raise RuntimeError(
                "LevenbergMarquardt requires publish_smoothed_path to be set to true."
            )
        self.kf_source = self.params["keyframe_source"]
        self.kf_backup = self.params["backup_keyframe_source"]
        self.kf_timeout = self.params["keyframe_timeout_sec"]
        self.kf_period = 1.0 / self.params["keyframe_timer_hz"]

        for source in (self.kf_source, self.kf_backup):
            sensor = self.SOURCE_SENSORS.get(source)
            if sensor in ("dvl", "depth") and not self.enabled[sensor]:
                raise ValueError(
                    f"Keyframe source '{self.kf_source}' or backup "
                    f"'{self.kf_backup}' references a disabled sensor!"
                )

        self.results: list[dict] = []
        self.base_tf: tuple[np.ndarray, Rotation] | None = None
        self.tfs_resolved: set[str] = set()
        self._last_msg_time: dict[str, float] = {}
        self._stream_time = 0.0
        self._reset_state()

    def reset(self) -> None:
        self.fg.reset()
        self._reset_state()

    def _reset_state(self) -> None:
        self.is_initialized = False
        self.queues: dict[str, list[tuple]] = {sensor: [] for sensor in SENSORS}
        self._last_target_time: float | None = None
        self._last_update_time: float | None = None
        self._last_opt_time: float | None = None
        self._last_timer_time: float | None = None

    def add_message(self, sensor: str, frame_id: str, sample: tuple) -> None:
        self._resolve_sensor_tf(sensor, frame_id)
        self.queues[sensor].append(sample)
        self._last_msg_time[sensor] = sample[0]
        self._stream_time = max(self._stream_time, sample[0])

        sources = (self.kf_source, self.kf_backup)
        if (sensor == "dvl" and "DVL" in sources) or (
            sensor == "depth" and "Depth" in sources
        ):
            self._process_frontend()
        if "Timer" in sources:
            if self._last_timer_time is None:
                self._last_timer_time = self._stream_time
            elif self._stream_time - self._last_timer_time >= self.kf_period:
                self._last_timer_time = self._stream_time
                self._process_frontend()

    def _process_frontend(self) -> None:
        if not self.is_initialized:
            self._initialize_graph()
        elif self._check_and_update_rate_limit(
            "_last_update_time", "max_update_rate_hz"
        ):
            self._update_graph()
            self._optimize_graph()

    def _initialize_graph(self) -> None:
        for sensor in ("imu", "gps", "depth", "mag", "ahrs", "dvl"):
            if self.enabled[sensor] and sensor not in self.tfs_resolved:
                return

        if self.base_tf is None:
            base = self.params["sensors"]["base"]
            pos, quat = self._load_or_lookup_tf(base, self.params["base_frame"])
            self.fg.set_tf("base", pos, quat)
            self.base_tf = (pos, Rotation.from_quat(quat))

        newest_stamp = max((q[-1][0] for q in self.queues.values() if q), default=0.0)
        if newest_stamp <= 0.0:
            return

        batches = self.queues
        self.queues = {sensor: [] for sensor in SENSORS}
        if self.fg.initialize(newest_stamp, **batches):
            self.is_initialized = True

    def _update_graph(self) -> bool:
        active = self.kf_source
        if active != "Timer":
            last_received = self._last_msg_time.get(
                "dvl" if active == "DVL" else "depth"
            )
            newest_stamp = self._last_msg_time.get("imu")
            timed_out = (
                newest_stamp is not None
                and last_received is not None
                and newest_stamp - last_received > self.kf_timeout
            )
            if (last_received is None or timed_out) and self.kf_backup != "None":
                active = self.kf_backup

        src = self.SOURCE_SENSORS.get(active)
        target_time = self._last_msg_time.get(src) if src and self.queues[src] else None
        if target_time is None or (
            self._last_target_time is not None and target_time <= self._last_target_time
        ):
            return False
        self._last_target_time = target_time

        leftover = self.fg.update(target_time, **self.queues)
        if leftover is None:
            return False
        self.queues = leftover
        return True

    def _optimize_graph(self) -> None:
        if self.is_lm:
            return  # LevenbergMarquardt uses finalize()
        if not self._check_and_update_rate_limit("_last_opt_time", "max_opt_rate_hz"):
            return
        if result := self.fg.optimize():
            self.results.append(result)

    def finalize(self) -> list[dict]:
        if self.is_lm and self.is_initialized:
            result = self.fg.optimize()
            self.results = list(result.get("smoothed_path", [])) if result else []
        return self.results

    def _check_and_update_rate_limit(self, last_attr: str, rate_param: str) -> bool:
        max_rate_hz = self.params[rate_param]
        if max_rate_hz <= 0.0:
            return True
        last_time = getattr(self, last_attr)
        if last_time is not None and self._stream_time - last_time < 1.0 / max_rate_hz:
            return False
        setattr(self, last_attr, self._stream_time)
        return True

    def _resolve_sensor_tf(self, sensor: str, frame_id: str) -> None:
        if sensor in self.tfs_resolved:
            return
        cfg = self.params["sensors"]["dynamics" if sensor == "wrench" else sensor]
        frame = cfg["parameter_frame"] if cfg["use_parameter_frame"] else frame_id
        pos, quat = self._load_or_lookup_tf(cfg, frame)
        self.fg.set_tf("com" if sensor == "wrench" else sensor, pos, quat)
        self.tfs_resolved.add(sensor)

    def _load_or_lookup_tf(
        self, cfg: dict, frame: str
    ) -> tuple[np.ndarray, np.ndarray]:
        if cfg["use_parameter_tf"]:
            return np.array(cfg["tf_position"]), np.array(cfg["tf_orientation"])
        if self.urdf is None:
            raise RuntimeError(
                f"use_parameter_tf is false for frame '{frame}' but no URDF was found!"
            )
        return self.urdf.lookup(self.params["target_frame"], frame)


# =============================================================================
# BAG PROCESSING
# =============================================================================


def process_bag_offline(
    bag_path: str,
    config_paths: list[str],
    namespace: str,
    urdf_path: str | None = None,
    verbose: bool = True,
) -> tuple[dict | None, bool]:
    if urdf_path is None:
        urdf_path = resolve_urdf_path(namespace, config_paths)

    urdf = UrdfTree(urdf_path) if urdf_path else None
    pipeline = OfflineFactorGraph(config_paths, namespace, urdf)

    if verbose:
        cfg_str = "\n".join(f"  - {p}" for p in config_paths)
        logger.info(f"Loaded configs:\n{cfg_str}")
        if urdf_path:
            logger.info(f"Loaded URDF: {urdf_path}")

    topic_to_sensors: dict[str, list[str]] = {}
    for sensor in SENSORS:
        if not pipeline.enabled[sensor]:
            continue
        topic = pipeline.params["topics"][sensor]
        if not topic.startswith("/"):
            topic = f"/{namespace}/{topic}" if namespace else f"/{topic}"
        topic_to_sensors.setdefault(topic, []).append(sensor)

    crashed = False
    with AnyReader(
        [Path(bag_path)], default_typestore=get_typestore(Stores.ROS2_HUMBLE)
    ) as reader:
        matched_conns = [c for c in reader.connections if c.topic in topic_to_sensors]

        if verbose:
            conn_str = "\n".join(
                f"  - {c.topic} ({topic_to_sensors[c.topic]})" for c in matched_conns
            )
            logger.info(f"Matched Connections:\n{conn_str}")

        pbar = tqdm(
            reader.messages(connections=matched_conns),
            total=sum(c.msgcount for c in matched_conns),
            disable=not verbose,
        )

        for conn, _, rawdata in pbar:
            msg = reader.deserialize(rawdata, conn.msgtype)
            try:
                for sensor in topic_to_sensors[conn.topic]:
                    frame_id, sample = EXTRACTORS[sensor](msg)
                    pipeline.add_message(sensor, frame_id, sample)
            except Exception as e:
                if verbose:
                    logger.error(f"FGO Error: {e}")
                crashed = True
                break

    if not crashed:
        try:
            pipeline.finalize()
        except Exception as e:
            if verbose:
                logger.error(f"FGO Error: {e}")
            crashed = True
    raw_results = pipeline.results

    if verbose and not pipeline.is_initialized:
        missing = [
            s
            for s in ("imu", "gps", "depth", "mag", "ahrs", "dvl")
            if pipeline.enabled[s] and s not in pipeline.tfs_resolved
        ]
        logger.error(
            f"FGO never initialized! No data received for: {', '.join(missing)}"
        )
    if not raw_results:
        return None, crashed

    results = {
        k: np.array([r[k] for r in raw_results])
        for k in raw_results[0].keys()
        if k != "smoothed_path"
    }

    # IMPORTANT! Pose covariance is just left at the target frame here
    base_pos, base_rot = pipeline.base_tf
    target_positions = np.column_stack((results["x"], results["y"], results["z"]))
    target_quats = np.column_stack(
        (results["qx"], results["qy"], results["qz"], results["qw"])
    )

    map_R_target = Rotation.from_quat(target_quats)
    map_R_base = map_R_target * base_rot
    map_t_base = map_R_target.apply(base_pos) + target_positions

    results["x"], results["y"], results["z"] = map_t_base.T
    results["qx"], results["qy"], results["qz"], results["qw"] = map_R_base.as_quat().T
    results["roll"], results["pitch"], results["yaw"] = map_R_base.as_euler("xyz").T

    return results, crashed


@contextmanager
def covariance_override_file(scalars: dict[str, float]):
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
        os.unlink(override_path)
