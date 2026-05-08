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

import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from evo.core import metrics, sync
from evo.core.trajectory import PoseTrajectory3D
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation
from tqdm import tqdm

FGO_LIB_PATH = str(
    Path.home() / "cougars-dev/ros2_ws/install/coug_fgo/lib/python3.10/site-packages"
)
sys.path.insert(0, FGO_LIB_PATH)
import pybind11fgo  # noqa: E402

# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class FGOConfig:
    topic_map: dict[str, list[str]]
    required_sensors: set[str]
    kf_source: str
    kf_topic: str | None
    kf_backup_source: str
    kf_backup_topic: str | None
    kf_timeout: float
    kf_period: float
    base_pos: np.ndarray
    base_rot: Rotation
    solver_type: str

    @property
    def is_lm(self) -> bool:
        return self.solver_type == "LevenbergMarquardt"


def _deep_merge(base: dict, override: dict) -> None:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def _parse_config(
    config_paths: list[str], namespace: str, verbose: bool = True
) -> FGOConfig:
    params = {}
    for path in config_paths:
        if verbose:
            print(f"Loading config: {path}")
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        for key in ["/**", f"/{namespace}/**"]:
            _deep_merge(params, config.get(key, {}).get("ros__parameters", {}))

    sensor_topics = {
        "imu": params["imu_topic"],
        "gps": params["gps_odom_topic"],
        "depth": params["depth_odom_topic"],
        "mag": params["mag_topic"],
        "ahrs": params["ahrs_topic"],
        "dvl": params["dvl_topic"],
        "wrench": params["wrench_topic"],
    }

    topic_map: dict[str, list[str]] = {}
    for sensor, topic in sensor_topics.items():
        topic_map.setdefault(topic, []).append(sensor)

    required_sensors = {"imu"}
    for s in ["gps", "depth", "mag", "ahrs", "dvl"]:
        if params[s].get(f"enable_{s}") or params[s].get(f"enable_{s}_init_only"):
            required_sensors.add(s)

    source_to_topic_key = {"DVL": "dvl_topic", "Depth": "depth_odom_topic"}
    kf_source = params["keyframe_source"]
    kf_topic = None if kf_source == "Timer" else params[source_to_topic_key[kf_source]]

    backup_source = params["backup_keyframe_source"]
    backup_topic = (
        None
        if backup_source in ("None", "Timer")
        else params[source_to_topic_key[backup_source]]
    )

    base_tf = params["base"]["parameter_tf"]

    return FGOConfig(
        topic_map=topic_map,
        required_sensors=required_sensors,
        kf_source=kf_source,
        kf_topic=kf_topic,
        kf_backup_source=backup_source,
        kf_backup_topic=backup_topic,
        kf_timeout=params["keyframe_timeout"],
        kf_period=1.0 / max(params["keyframe_timer_hz"], 0.1),
        base_pos=np.array(base_tf["position"]),
        base_rot=Rotation.from_quat(base_tf["orientation"]),
        solver_type=params["solver_type"],
    )


@contextmanager
def param_override_file(scalars: dict[str, float]):
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


# =============================================================================
# ROS DATA EXTRACTION
# =============================================================================

EXTRACTORS = {
    "imu": lambda m: (
        np.array(
            [m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z]
        ),
        np.array([m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z]),
        np.array(m.linear_acceleration_covariance).reshape(3, 3),
        np.array(m.angular_velocity_covariance).reshape(3, 3),
    ),
    "dvl": lambda m: (
        np.array(
            [m.twist.twist.linear.x, m.twist.twist.linear.y, m.twist.twist.linear.z]
        ),
        np.array(m.twist.covariance).reshape(6, 6),
    ),
    "ahrs": lambda m: (
        np.array([m.orientation.x, m.orientation.y, m.orientation.z, m.orientation.w]),
        np.array(m.orientation_covariance).reshape(3, 3),
    ),
    "depth": lambda m: (
        m.pose.pose.position.z,
        np.array(m.pose.covariance).reshape(6, 6),
    ),
    "gps": lambda m: (
        np.array(
            [m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z]
        ),
        np.array(m.pose.covariance).reshape(6, 6),
    ),
    "mag": lambda m: (
        np.array([m.magnetic_field.x, m.magnetic_field.y, m.magnetic_field.z]),
        np.array(m.magnetic_field_covariance).reshape(3, 3),
    ),
    "wrench": lambda m: (
        np.array(
            [
                m.wrench.force.x,
                m.wrench.force.y,
                m.wrench.force.z,
                m.wrench.torque.x,
                m.wrench.torque.y,
                m.wrench.torque.z,
            ]
        ),
    ),
}


# =============================================================================
# FACTOR GRAPH
# =============================================================================


def run_factor_graph(
    bag_path: str, config_paths: list[str], namespace: str, verbose: bool = True
) -> tuple[dict | None, bool]:
    cfg = _parse_config(config_paths, namespace, verbose)
    fg = pybind11fgo.FactorGraphPy(config_paths, namespace)
    raw_results = []

    with AnyReader(
        [Path(bag_path)], default_typestore=get_typestore(Stores.ROS2_HUMBLE)
    ) as reader:
        conns = [
            c
            for c in reader.connections
            if not namespace or c.topic.startswith(f"/{namespace}/")
        ]

        topic_to_sensors: dict[str, list[str]] = {}
        for c in conns:
            for suffix, sensors in cfg.topic_map.items():
                if c.topic.endswith(suffix):
                    topic_to_sensors.setdefault(c.topic, []).extend(sensors)

        matched_conns = [c for c in conns if c.topic in topic_to_sensors]

        if verbose:
            print("\nMatched Connections:")
            for c in matched_conns:
                print(f"- {c.topic} ({topic_to_sensors[c.topic]})")
            print()

        sensors_seen: set[str] = set()
        is_running, crashed = False, False
        last_kf_time, current_time, last_kf_received = 0.0, 0.0, 0.0

        pbar = tqdm(
            reader.messages(connections=matched_conns),
            total=sum(c.msgcount for c in matched_conns),
            desc="Processing FGO",
            disable=not verbose,
        )

        for conn, _, rawdata in pbar:
            msg = reader.deserialize(rawdata, conn.msgtype)
            sensors = topic_to_sensors[conn.topic]
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

            for sensor in sensors:
                getattr(fg, f"add_{sensor}")(t, *EXTRACTORS[sensor](msg))
                sensors_seen.add(sensor)

            if not is_running:
                if cfg.required_sensors.issubset(sensors_seen) and fg.initialize_graph(
                    t
                ):
                    is_running = True
                    last_kf_time = t
                continue

            current_time = max(t, current_time)
            if cfg.kf_topic and conn.topic.endswith(cfg.kf_topic):
                last_kf_received = max(t, last_kf_received)

            use_backup = cfg.kf_backup_source != "None" and (
                last_kf_received == 0.0
                or (current_time - last_kf_received) > cfg.kf_timeout
            )
            active_source = cfg.kf_backup_source if use_backup else cfg.kf_source
            active_topic = cfg.kf_backup_topic if use_backup else cfg.kf_topic

            trigger = False
            if (
                active_source == "Timer"
                and "imu" in sensors
                and t >= last_kf_time + cfg.kf_period
            ):
                trigger = True
            elif active_topic and conn.topic.endswith(active_topic):
                trigger = True

            if trigger:
                last_kf_time = (
                    t if active_source != "Timer" else last_kf_time + cfg.kf_period
                )
                try:
                    fg.update_graph(t)
                    if not cfg.is_lm:
                        # ISAM2, IncrementalFixedLagSmoother
                        if res := fg.optimize_graph():
                            raw_results.append(res)
                except Exception as e:
                    if verbose:
                        tqdm.write(f"FGO Error: {e}\n")
                    crashed = True
                    break

        if cfg.is_lm and is_running:
            # LevenbergMarquardt
            if res := fg.optimize_graph():
                raw_results = list(res.get("smoothed_path", []))

    if not raw_results:
        return None, False

    results = {
        k: np.array([r[k] for r in raw_results])
        for k in raw_results[0].keys()
        if k != "smoothed_path"
    }

    # Transform poses from target_frame (e.g. dvl_link) to base_link
    target_positions = np.column_stack((results["x"], results["y"], results["z"]))
    target_quats = np.column_stack(
        (results["qx"], results["qy"], results["qz"], results["qw"])
    )

    map_R_target = Rotation.from_quat(target_quats)
    map_R_base = map_R_target * cfg.base_rot
    map_t_base = map_R_target.apply(cfg.base_pos) + target_positions

    results["x"], results["y"], results["z"] = map_t_base.T
    results["qx"], results["qy"], results["qz"], results["qw"] = map_R_base.as_quat().T
    results["roll"], results["pitch"], results["yaw"] = map_R_base.as_euler("xyz").T

    return results, crashed


# =============================================================================
# EVALUATION & METRICS
# =============================================================================


def load_ground_truth(bag_path: str, namespace: str) -> dict:
    tum_path = (
        Path(bag_path)
        / "evo"
        / namespace
        / "odometry"
        / f"{namespace}_odometry_truth.tum"
    )
    print(f"Loading GT: {tum_path}")
    try:
        data = np.loadtxt(tum_path, comments="#")
        if data.ndim == 1:
            data = data.reshape(1, -1)
    except Exception:
        return {}

    if data.size == 0:
        return {}

    keys = ["time", "x", "y", "z", "qx", "qy", "qz", "qw"]
    pose = {k: data[:, i] for i, k in enumerate(keys)}
    r, p, yaw = Rotation.from_quat(data[:, 4:8]).as_euler("xyz").T
    pose.update({"roll": r, "pitch": p, "yaw": yaw})
    return pose


def compute_ape_rmse(
    gt: dict | None, est: dict | None, crashed: bool = False, max_diff: float = 0.05
) -> float:
    if not gt or not est or crashed:
        return float("inf")

    gt_traj = PoseTrajectory3D(
        positions_xyz=np.column_stack([gt["x"], gt["y"], gt["z"]]),
        orientations_quat_wxyz=np.column_stack(
            [gt["qw"], gt["qx"], gt["qy"], gt["qz"]]
        ),
        timestamps=gt["time"],
    )
    est_traj = PoseTrajectory3D(
        positions_xyz=np.column_stack([est["x"], est["y"], est["z"]]),
        orientations_quat_wxyz=np.column_stack(
            [est["qw"], est["qx"], est["qy"], est["qz"]]
        ),
        timestamps=est["time"],
    )

    gt_sync, est_sync = sync.associate_trajectories(
        gt_traj, est_traj, max_diff=max_diff
    )
    est_sync.align(gt_sync)

    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data((gt_sync, est_sync))
    return ape.get_statistic(metrics.StatisticsType.rmse)


def run_evo_evaluations(
    gt_file: str, est_file: str, evo_dir: Path, evo_flags: list[str]
) -> None:
    evo_base = ["--t_max_diff", "0.05", "--no_warnings"]
    plane_flags = [
        f for f in evo_flags if f.startswith("--project") or f in ("xy", "xz", "yz")
    ]
    non_plane_flags = [f for f in evo_flags if f not in plane_flags]

    for metric, cmd in [("APE", "evo_ape"), ("RPE", "evo_rpe")]:
        for mode, pose_relation, suffix in [
            ("Translation", "trans_part", "trans"),
            ("Rotation", "angle_deg", "rot"),
        ]:
            # Only apply xy projection flags to translation (no depth GT)
            flags = evo_flags if pose_relation == "trans_part" else non_plane_flags
            args = (
                [cmd, "tum", gt_file, est_file, "-r", pose_relation] + evo_base + flags
            )
            args += ["--save_results", str(evo_dir / f"{metric.lower()}_{suffix}.zip")]

            if metric == "RPE":
                args += ["--delta", "1", "--delta_unit", "m", "--all_pairs"]

            print(f"\n{metric} ({mode}):")
            subprocess.run(args)


def evaluate_and_save(
    bag_path: str,
    config_paths: list[str],
    namespace: str,
    evo_dir_name: str,
    evo_flags: list[str],
    verbose: bool = True,
) -> tuple:
    print(f"\nProcessing bag: {bag_path}")
    pose_gt = load_ground_truth(bag_path, namespace)
    results, crashed = run_factor_graph(bag_path, config_paths, namespace, verbose)

    evo_dir = Path(bag_path) / "evo" / namespace / "odometry" / evo_dir_name
    evo_dir.mkdir(parents=True, exist_ok=True)
    if results:
        out_path = evo_dir / f"{namespace}_odometry_{evo_dir_name}.tum"
        arr = np.column_stack(
            [results[k] for k in ["time", "x", "y", "z", "qx", "qy", "qz", "qw"]]
        )
        np.savetxt(out_path, arr, fmt="%.9f")
        print(f"Saved: {out_path}")

    if results and pose_gt:
        gt_tum_path = (
            Path(bag_path)
            / "evo"
            / namespace
            / "odometry"
            / f"{namespace}_odometry_truth.tum"
        )
        run_evo_evaluations(
            str(gt_tum_path),
            str(evo_dir / f"{namespace}_odometry_{evo_dir_name}.tum"),
            evo_dir,
            evo_flags,
        )

    return results, pose_gt


# =============================================================================
# PLOTTING
# =============================================================================


def plot_results(results: dict, pose_gt: dict, label: str = "") -> None:
    t0 = results["time"][0]
    t_fgo = results["time"] - t0

    layout = [
        (["x", "y", "z"], ["X (m)", "Y (m)", "Z (m)"], pose_gt),
        (["roll", "pitch", "yaw"], ["Roll (rad)", "Pitch (rad)", "Yaw (rad)"], pose_gt),
        (["vx", "vy", "vz"], ["Vx (m/s)", "Vy (m/s)", "Vz (m/s)"], None),
        (
            ["bias_accel_x", "bias_accel_y", "bias_accel_z"],
            ["Accel Bias X", "Accel Bias Y", "Accel Bias Z"],
            None,
        ),
        (
            ["bias_gyro_x", "bias_gyro_y", "bias_gyro_z"],
            ["Gyro Bias X", "Gyro Bias Y", "Gyro Bias Z"],
            None,
        ),
    ]

    _, axes = plt.subplots(5, 3, figsize=(12, 10), num=label or None)
    for row, (keys, labels, gt_data) in enumerate(layout):
        for col, (key, axis_label) in enumerate(zip(keys, labels)):
            ax = axes[row, col]
            if gt_data:
                gt_t = gt_data["time"] - t0
                gt_vals = gt_data[key]
                if len(gt_t) >= 2:
                    dts = np.diff(gt_t)
                    threshold = max(float(np.median(dts)) * 5.0, 0.5)
                    gap_idx = np.where(dts > threshold)[0] + 1
                    if len(gap_idx):
                        gt_t = np.insert(gt_t, gap_idx, np.nan)
                        gt_vals = np.insert(gt_vals, gap_idx, np.nan)
                ax.plot(gt_t, gt_vals, "-k", label="GT")
            if key in results:
                ax.plot(t_fgo, results[key], "-r", label="FGO")
            ax.set_ylabel(axis_label)
            if row == 4:
                ax.set_xlabel("Time (s)")

    plt.tight_layout()
