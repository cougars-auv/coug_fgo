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
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .urdf import UrdfTree

FGO_LIB_PATH = str(
    Path.home()
    / "cougars-dev/ros2_ws/install/coug_fgo/lib"
    / f"python{sys.version_info.major}.{sys.version_info.minor}"
    / "site-packages"
)
sys.path.insert(0, FGO_LIB_PATH)
import coug_fgo_py  # noqa: E402

logger = logging.getLogger(__name__)

SENSORS = ("imu", "gps", "depth", "mag", "ahrs", "dvl", "wrench")


class OfflineFactorGraph:
    """
    Offline factor graph using the FactorGraphPy Python wrapper.

    Should match the ROS 2 framework in ``factor_graph.cpp`` as closely as possible.

    :author: Nelson Durrant
    :date: July 2026
    """

    # IMPORTANT! Offline, the timer relies on IMU message stamps instead of the ROS 2 clock
    SOURCE_SENSORS = {"DVL": "dvl", "Depth": "depth", "Timer": "imu"}
    INIT_SENSORS = ("imu", "gps", "depth", "mag", "ahrs", "dvl")

    def __init__(
        self,
        config_paths: list[str],
        namespace: str = "",
        urdf: UrdfTree | None = None,
    ):
        """
        Load the parameters and prepare the sensor queues.

        :param config_paths: Parameter YAML files, in increasing priority.
        :param namespace: AUV namespace used to select parameters.
        :param urdf: Parsed URDF tree for transform lookups, if available.
        :raises RuntimeError: If LevenbergMarquardt lacks publish_smoothed_path.
        :raises ValueError: If a keyframe source is unknown or its sensor is disabled.
        """
        self.fg = coug_fgo_py.FactorGraphPy(config_paths, namespace)
        self.params = self.fg.get_params()
        self.namespace = namespace
        self.urdf = urdf

        sensors = self.params["sensors"]
        self.loose_preint = self.params["comparison"]["enable_loose_dvl_preintegration"]

        def subscribed(key: str) -> bool:
            """
            Return whether a sensor is used for factors or initialization.

            :param key: Sensor key in the parameters.
            :return: True if the sensor is enabled in any mode.
            """
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

        valid_sources = {"None", *self.SOURCE_SENSORS}
        for source in (self.kf_source, self.kf_backup):
            if source not in valid_sources:
                raise ValueError(f"Unknown keyframe source: {source}")
            sensor = self.SOURCE_SENSORS.get(source)
            if sensor in ("dvl", "depth") and not self.enabled[sensor]:
                raise ValueError(
                    f"Keyframe source '{self.kf_source}' or backup "
                    f"'{self.kf_backup}' references a disabled sensor."
                )

        self.results: list[dict] = []
        self.base_tf: tuple[np.ndarray, Rotation] | None = None
        self.tfs_resolved: set[str] = set()
        self._last_msg_time: dict[str, float] = {}
        self._stream_time = 0.0
        self._reset_state()

    def reset(self) -> None:
        """Clear the graph and all factor graph state for a fresh run."""
        self.fg.reset()
        self._reset_state()

    def _reset_state(self) -> None:
        """Reset the queues and bookkeeping shared between graph runs."""
        self.is_initialized = False
        self.queues: dict[str, list[tuple]] = {sensor: [] for sensor in SENSORS}
        self._using_backup = False
        self._last_target_time: float | None = None
        self._last_update_time: float | None = None
        self._last_opt_time: float | None = None
        self._last_timer_time: float | None = None

    @property
    def topic_map(self) -> dict[str, list[str]]:
        """
        Map each namespace-resolved topic to the sensors that read from it.

        :return: Resolved topic names to lists of sensor keys.
        """
        topics: dict[str, list[str]] = {}
        for sensor in SENSORS:
            if not self.enabled[sensor]:
                continue
            topic = self.params["topics"][sensor]
            if not topic.startswith("/"):
                topic = f"/{self.namespace}/{topic}" if self.namespace else f"/{topic}"
            topics.setdefault(topic, []).append(sensor)
        return topics

    def pending_init_sensors(self) -> list[str]:
        """
        Return enabled sensors still missing a resolved transform.

        :return: Sensor keys blocking initialization, in priority order.
        """
        return [
            s
            for s in self.INIT_SENSORS
            if self.enabled[s] and s not in self.tfs_resolved
        ]

    def add_message(self, sensor: str, frame_id: str, measurement: tuple) -> None:
        """
        Queue one sensor measurement and trigger keyframe processing when due.

        :param sensor: Sensor key from SENSORS.
        :param frame_id: ROS frame the measurement was reported in.
        :param measurement: Extracted measurement tuple, with the stamp first.
        """
        self._resolve_sensor_tf(sensor, frame_id)
        self.queues[sensor].append(measurement)
        self._last_msg_time[sensor] = measurement[0]
        self._stream_time = max(self._stream_time, measurement[0])

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
        """Initialize the graph or run a rate-limited update and optimize."""
        if not self.is_initialized:
            self._initialize_graph()
        elif self._check_and_update_rate_limit(
            "_last_update_time", "max_update_rate_hz"
        ):
            self._update_graph()
            self._optimize_graph()

    def _initialize_graph(self) -> None:
        """Attempt initialization once all enabled sensor TFs are resolved."""
        if self.pending_init_sensors():
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
            logger.info("Graph initialized successfully.")

    def _update_graph(self) -> None:
        """Advance the graph to the newest stamp from the active source."""
        active_source = self.kf_source
        if active_source != "Timer":
            last_received = self._last_msg_time.get(
                "dvl" if active_source == "DVL" else "depth"
            )
            newest_stamp = self._last_msg_time.get("imu")
            timed_out = (
                newest_stamp is not None
                and last_received is not None
                and newest_stamp - last_received > self.kf_timeout
            )
            if last_received is None or timed_out:
                if self.kf_backup != "None":
                    active_source = self.kf_backup
                    if not self._using_backup:
                        logger.warning(
                            f"Primary keyframe source '{self.kf_source}' timed out. "
                            f"Using backup '{self.kf_backup}'."
                        )
                else:
                    if not self._using_backup:
                        logger.error(
                            f"Primary keyframe source '{self.kf_source}' timed out and no backup is configured. "
                            "No new keyframes will be created."
                        )
                self._using_backup = True
            else:
                self._using_backup = False

        src = self.SOURCE_SENSORS.get(active_source)
        target_time = self._last_msg_time.get(src) if src and self.queues[src] else None
        if target_time is None or (
            self._last_target_time is not None and target_time <= self._last_target_time
        ):
            return
        self._last_target_time = target_time

        leftover = self.fg.update(target_time, **self.queues)
        if leftover is None:
            return
        self.queues = leftover

    def _optimize_graph(self) -> None:
        """Run a rate-limited optimization and record the result."""
        if self.is_lm:
            return  # LevenbergMarquardt uses finalize()
        if not self._check_and_update_rate_limit("_last_opt_time", "max_opt_rate_hz"):
            return
        if result := self.fg.optimize():
            self.results.append(result)

    def finalize(self) -> None:
        """Run the final batch optimization for LevenbergMarquardt solvers."""
        if self.is_lm and self.is_initialized:
            result = self.fg.optimize()
            self.results = list(result.get("smoothed_path", [])) if result else []

    def get_results(self) -> dict | None:
        """
        Assemble the optimized results, re-expressed at the base frame.

        :return: Result arrays keyed by state name, or None if there are none.
        """
        if not self.results:
            return None

        results = {
            k: np.array([r[k] for r in self.results])
            for k in self.results[0].keys()
            if k != "smoothed_path"
        }

        # IMPORTANT! Pose covariance is just left at the target frame here
        base_pos, base_rot = self.base_tf
        target_positions = np.column_stack((results["x"], results["y"], results["z"]))
        target_quats = np.column_stack(
            (results["qx"], results["qy"], results["qz"], results["qw"])
        )

        map_R_target = Rotation.from_quat(target_quats)
        map_R_base = map_R_target * base_rot
        map_t_base = map_R_target.apply(base_pos) + target_positions

        results["x"], results["y"], results["z"] = map_t_base.T
        results["qx"], results["qy"], results["qz"], results["qw"] = (
            map_R_base.as_quat().T
        )
        results["roll"], results["pitch"], results["yaw"] = map_R_base.as_euler("xyz").T

        return results

    def _check_and_update_rate_limit(self, last_attr: str, rate_param: str) -> bool:
        """
        Check a rate limit against stream time, updating it when passed.

        :param last_attr: Attribute holding the last accepted stream time.
        :param rate_param: Parameter holding the maximum rate in Hz.
        :return: True if the action is allowed at the current stream time.
        """
        max_rate_hz = self.params[rate_param]
        if max_rate_hz <= 0.0:
            return True
        last_time = getattr(self, last_attr)
        if last_time is not None and self._stream_time - last_time < 1.0 / max_rate_hz:
            return False
        setattr(self, last_attr, self._stream_time)
        return True

    def _resolve_sensor_tf(self, sensor: str, frame_id: str) -> None:
        """
        Resolve and register a sensor's static transform once.

        :param sensor: Sensor key from SENSORS.
        :param frame_id: Frame reported by the sensor's messages.
        """
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
        """
        Load a transform from parameters or look it up in the URDF.

        :param cfg: Sensor parameter dictionary.
        :param frame: Source frame to look up when parameters are not used.
        :return: Position and xyzw quaternion in the target frame.
        :raises RuntimeError: If a URDF lookup is required but no URDF was found.
        """
        if cfg["use_parameter_tf"]:
            return np.array(cfg["tf_position"]), np.array(cfg["tf_orientation"])
        if self.urdf is None:
            raise RuntimeError(
                f"use_parameter_tf is false for frame '{frame}' but no URDF was found."
            )
        return self.urdf.lookup(self.params["target_frame"], frame)
