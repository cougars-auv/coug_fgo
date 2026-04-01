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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

FGO_LIB_PATH = str(
    Path.home() / "cougars-dev/ros2_ws/install/coug_fgo/lib/python3.10/site-packages"
)
EVO_LIB_PATH = str(Path.home() / ".local/pipx/venvs/evo/lib/python3.10/site-packages")
sys.path.insert(0, FGO_LIB_PATH)
sys.path.insert(0, EVO_LIB_PATH)
import pybind11fgo  # noqa: E402
from evo.core import metrics, sync  # noqa: E402
from evo.core.trajectory import PoseTrajectory3D  # noqa: E402

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


def _deep_merge(base: dict, override: dict) -> None:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def _parse_config(
    config_paths: list[str], namespace: str, verbose: bool = True
) -> tuple[dict, set[str], dict, tuple, str]:
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

    # IMU is always required, other sensors required only when enabled
    required_sensors = {"imu"}
    for s in ["gps", "depth", "mag", "ahrs", "dvl"]:
        sensor_params = params[s]
        if sensor_params.get(f"enable_{s}") or sensor_params.get(
            f"enable_{s}_init_only"
        ):
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

    kf_config = {
        "source": kf_source,
        "topic": kf_topic,
        "backup_source": backup_source,
        "backup_topic": backup_topic,
        "timeout": params["keyframe_timeout"],
        "period": 1.0 / max(params["keyframe_timer_hz"], 0.1),
    }

    base_tf = params["base"]["parameter_tf"]
    target_T_base = (np.array(base_tf["position"]), R.from_quat(base_tf["orientation"]))

    return topic_map, required_sensors, kf_config, target_T_base, params["solver_type"]


def run_factor_graph(
    bag_path: str, config_paths: list[str], namespace: str, verbose: bool = True
) -> tuple[dict | None, bool]:
    topic_map, required_sensors, kf_config, target_T_base, solver_type = _parse_config(
        config_paths, namespace, verbose
    )

    is_lm = solver_type == "LevenbergMarquardt"
    fg = pybind11fgo.FactorGraphPy(config_paths, namespace)
    raw_results = []

    if verbose:
        print(f"Opening bag: {bag_path}")
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
            for suffix, sensors in topic_map.items():
                if c.topic.endswith(suffix):
                    topic_to_sensors.setdefault(c.topic, []).extend(sensors)

        matched_conns = [c for c in conns if c.topic in topic_to_sensors]

        if verbose:
            print("\nMatched Connections:")
            for c in matched_conns:
                print(f"- {c.topic} ({topic_to_sensors[c.topic]})")
            print()

        sensors_seen: set[str] = set()
        is_running = False
        crashed = False
        last_kf_time = None
        current_time = 0.0
        last_kf_received = 0.0
        kf_topic = kf_config["topic"]

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

            # Wait until all required sensors have been seen before initializing
            if not is_running:
                if required_sensors.issubset(sensors_seen) and fg.initialize_graph(t):
                    is_running = True
                    last_kf_time = t
                continue

            current_time = max(t, current_time)

            if kf_topic and conn.topic.endswith(kf_topic):
                last_kf_received = max(t, last_kf_received)

            # Switch to backup keyframe source if primary has timed out
            active_topic = kf_topic
            if kf_config["source"] != "Timer" and kf_config["backup_source"] != "None":
                if (
                    last_kf_received == 0.0
                    or (current_time - last_kf_received) > kf_config["timeout"]
                ):
                    active_topic = kf_config["backup_topic"]

            trigger = False
            if (
                kf_config["source"] == "Timer"
                and "imu" in sensors
                and t >= last_kf_time + kf_config["period"]
            ):
                last_kf_time += kf_config["period"]
                trigger = True
            elif active_topic and conn.topic.endswith(active_topic):
                trigger = True
            elif (
                kf_config["backup_source"] == "Timer"
                and active_topic != kf_topic
                and "imu" in sensors
                and t >= last_kf_time + kf_config["period"]
            ):
                last_kf_time += kf_config["period"]
                trigger = True

            if trigger:
                try:
                    fg.update_graph(t)
                    if not is_lm:
                        # Incremental solvers (ISAM2, FixedLagSmoother) optimize each keyframe
                        if res := fg.optimize_graph():
                            raw_results.append(res)
                except Exception as e:
                    if verbose:
                        tqdm.write(f"{e}\n")
                    crashed = True
                    break

        if is_lm and is_running:
            # LevenbergMarquardt: run one full batch optimization over all keyframes
            if res := fg.optimize_graph():
                raw_results = list(res.get("smoothed_path", []))

    if not is_running and verbose:
        print(
            f"\nGraph never initialized! Missing sensors: {required_sensors - sensors_seen}"
        )

    if not raw_results:
        return None, False

    results = {
        k: np.array([r[k] for r in raw_results])
        for k in raw_results[0].keys()
        if k != "smoothed_path"
    }

    # Transform poses from target_frame (e.g. dvl_link) to base_link
    base_pos, base_rot = target_T_base
    rolls, pitches, yaws = [], [], []
    for i in range(len(raw_results)):
        map_R_target = R.from_quat(
            [results["qx"][i], results["qy"][i], results["qz"][i], results["qw"][i]]
        )
        map_R_base = map_R_target * base_rot
        map_t_base = map_R_target.apply(base_pos) + np.array(
            [results["x"][i], results["y"][i], results["z"][i]]
        )

        results["x"][i], results["y"][i], results["z"][i] = map_t_base
        q = map_R_base.as_quat()
        results["qx"][i], results["qy"][i], results["qz"][i], results["qw"][i] = q

        r, p, y = map_R_base.as_euler("xyz")
        rolls.append(r)
        pitches.append(p)
        yaws.append(y)

    results["roll"] = np.array(rolls)
    results["pitch"] = np.array(pitches)
    results["yaw"] = np.array(yaws)
    return results, crashed


def load_ground_truth(bag_path: str, namespace: str) -> tuple[dict, dict, dict]:
    pose_dict: dict[str, list] = {
        "time": [],
        "x": [],
        "y": [],
        "z": [],
        "qx": [],
        "qy": [],
        "qz": [],
        "qw": [],
        "roll": [],
        "pitch": [],
        "yaw": [],
    }
    vel_dict: dict[str, list] = {"time": [], "vx": [], "vy": [], "vz": []}
    bias_dict: dict[str, list] = {
        "time": [],
        "bias_accel_x": [],
        "bias_accel_y": [],
        "bias_accel_z": [],
        "bias_gyro_x": [],
        "bias_gyro_y": [],
        "bias_gyro_z": [],
    }

    with AnyReader(
        [Path(bag_path)], default_typestore=get_typestore(Stores.ROS2_HUMBLE)
    ) as reader:
        conns = [
            c
            for c in reader.connections
            if not namespace or c.topic.startswith(f"/{namespace}/")
        ]

        gt_topic_map = {
            "odometry/truth": "pose gt",
            "VelocitySensor": "vel gt",
            f"{namespace}/imu_bias": "imu bias gt",
        }
        gt_conns = [c for c in conns if any(c.topic.endswith(k) for k in gt_topic_map)]
        gt_labels = {
            c.topic: next(v for k, v in gt_topic_map.items() if c.topic.endswith(k))
            for c in gt_conns
        }

        print("\nMatched GT Connections:")
        for c in gt_conns:
            print(f"- {c.topic} ({gt_labels[c.topic]})")
        print()

        pbar = tqdm(
            reader.messages(connections=gt_conns),
            total=sum(c.msgcount for c in gt_conns),
            desc="Processing GT",
        )

        for conn, _, rawdata in pbar:
            msg = reader.deserialize(rawdata, conn.msgtype)
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

            if conn.topic.endswith("odometry/truth"):
                q, pos = msg.pose.pose.orientation, msg.pose.pose.position
                roll, pitch, yaw = R.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")
                pose_dict["time"].append(t)
                pose_dict["x"].append(pos.x)
                pose_dict["y"].append(pos.y)
                pose_dict["z"].append(pos.z)
                pose_dict["qx"].append(q.x)
                pose_dict["qy"].append(q.y)
                pose_dict["qz"].append(q.z)
                pose_dict["qw"].append(q.w)
                pose_dict["roll"].append(roll)
                pose_dict["pitch"].append(pitch)
                pose_dict["yaw"].append(yaw)

            elif conn.topic.endswith("VelocitySensor"):
                v = msg.twist.twist.linear
                vel_dict["time"].append(t)
                vel_dict["vx"].append(v.x)
                vel_dict["vy"].append(v.y)
                vel_dict["vz"].append(v.z)

            elif conn.topic.endswith(f"{namespace}/imu_bias"):
                lin, ang = msg.twist.twist.linear, msg.twist.twist.angular
                bias_dict["time"].append(t)
                bias_dict["bias_accel_x"].append(lin.x)
                bias_dict["bias_accel_y"].append(lin.y)
                bias_dict["bias_accel_z"].append(lin.z)
                bias_dict["bias_gyro_x"].append(ang.x)
                bias_dict["bias_gyro_y"].append(ang.y)
                bias_dict["bias_gyro_z"].append(ang.z)

    pose = {k: np.array(v) for k, v in pose_dict.items()} if pose_dict["time"] else {}
    vel = {k: np.array(v) for k, v in vel_dict.items()} if vel_dict["time"] else {}
    bias = {k: np.array(v) for k, v in bias_dict.items()} if bias_dict["time"] else {}
    return pose, vel, bias


def write_tum(filepath: str | Path, data: dict) -> None:
    with open(filepath, "w") as f:
        for t, x, y, z, qx, qy, qz, qw in zip(
            data["time"],
            data["x"],
            data["y"],
            data["z"],
            data["qx"],
            data["qy"],
            data["qz"],
            data["qw"],
        ):
            f.write(
                f"{t:.9f} {x:.9f} {y:.9f} {z:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n"
            )
    print(f"Saved: {filepath}")


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


def plot_results(results: dict, pose_gt: dict, vel_gt: dict, bias_gt: dict) -> None:
    """Plot FGO trajectory, velocity, and IMU bias estimates against ground truth."""
    t0 = results["time"][0]
    t_fgo = results["time"] - t0

    layout = [
        (["x", "y", "z"], ["X (m)", "Y (m)", "Z (m)"], pose_gt),
        (["roll", "pitch", "yaw"], ["Roll (rad)", "Pitch (rad)", "Yaw (rad)"], pose_gt),
        (["vx", "vy", "vz"], ["Vx (m/s)", "Vy (m/s)", "Vz (m/s)"], vel_gt),
        (
            ["bias_accel_x", "bias_accel_y", "bias_accel_z"],
            ["Accel Bias X", "Accel Bias Y", "Accel Bias Z"],
            bias_gt,
        ),
        (
            ["bias_gyro_x", "bias_gyro_y", "bias_gyro_z"],
            ["Gyro Bias X", "Gyro Bias Y", "Gyro Bias Z"],
            bias_gt,
        ),
    ]

    _, axes = plt.subplots(5, 3, figsize=(12, 10))
    for row, (keys, labels, gt_data) in enumerate(layout):
        for col, (key, label) in enumerate(zip(keys, labels)):
            ax = axes[row, col]
            if gt_data:
                ax.plot(gt_data["time"] - t0, gt_data[key], "-k", label="GT")
            ax.plot(t_fgo, results[key], "-r", label="FGO")
            ax.set_ylabel(label)
            if row == 4:
                ax.set_xlabel("Time (s)")

    plt.tight_layout()
    print("Displaying plots...")
    plt.show()
