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

# %%
import sys
from pathlib import Path

import numpy as np
import yaml
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from tqdm import tqdm

FGO_LIB_PATH = str(
    Path.home() / "cougars-dev/ros2_ws/install/coug_fgo/lib/python3.10/site-packages"
)
sys.path.insert(0, FGO_LIB_PATH)
import pybind11fgo  # noqa: E402

NAMESPACE = "bluerov2"
BAG_PATH = str(
    Path.home() / "cougars-dev/bags/batch_ul_surface_1.0_2026-03-25-14-46-57"
)
FLEET_CONFIG_PATH = str(Path.home() / "cougars-dev/config/fleet/coug_fgo_params.yaml")
AUV_CONFIG_PATH = str(Path.home() / "cougars-dev/config/bluerov2_params.yaml")
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


# %%
def load_config(config_paths, namespace):
    params = {}
    for path in config_paths:
        print(f"Loading config: {path}")
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        for key in ["/**", f"/{namespace}/**"]:
            layer = config.get(key, {}).get("ros__parameters", {})

            def merge(base, override):
                for k, v in override.items():
                    if isinstance(v, dict) and isinstance(base.get(k), dict):
                        merge(base[k], v)
                    else:
                        base[k] = v

            merge(params, layer)

    sensor_topics = {
        "imu": params.get("imu_topic", "imu/data"),
        "gps": params.get("gps_odom_topic", "odometry/gps"),
        "depth": params.get("depth_odom_topic", "odometry/depth"),
        "mag": params.get("mag_topic", "imu/mag"),
        "ahrs": params.get("ahrs_topic", "imu/ahrs"),
        "dvl": params.get("dvl_topic", "dvl/twist"),
        "wrench": params.get("wrench_topic", "cmd_wrench"),
    }

    topic_map = {}
    for sensor, topic in sensor_topics.items():
        if topic not in topic_map:
            topic_map[topic] = []
        topic_map[topic].append(sensor)

    required_sensors = {"imu"}
    for s in ["gps", "depth", "mag", "ahrs", "dvl"]:
        sensor_params = params.get(s, {})
        if sensor_params.get(f"enable_{s}") or sensor_params.get(
            f"enable_{s}_init_only"
        ):
            required_sensors.add(s)

    kf_source = params.get("keyframe_source", "DVL")
    if kf_source == "Timer":
        kf_topic = None
    else:
        topic_key = {"DVL": "dvl_topic", "Depth": "depth_odom_topic"}.get(
            kf_source, "dvl_topic"
        )
        kf_topic = params.get(topic_key, "dvl/twist")

    kf_period = 1.0 / max(params.get("keyframe_timer_hz", 10.0), 0.1)

    base_tf = params.get("base", {}).get("parameter_tf", {})
    base_pos = np.array(base_tf.get("position", [0.0, 0.0, 0.0]))  # [x, y, z]
    base_quat = base_tf.get("orientation", [0.0, 0.0, 0.0, 1.0])  # [qx, qy, qz, qw]
    target_T_base = (base_pos, R.from_quat(base_quat))

    return topic_map, required_sensors, kf_source, kf_topic, kf_period, target_T_base


def process_bag(bag_path, config_paths, namespace):
    topic_map, required_sensors, kf_source, kf_topic, kf_period, target_T_base = (
        load_config(config_paths, namespace)
    )

    fg = pybind11fgo.FactorGraphPy(config_paths, namespace)
    raw_results = []

    print(f"Opening bag: {bag_path}")
    with AnyReader(
        [Path(bag_path)], default_typestore=get_typestore(Stores.ROS2_HUMBLE)
    ) as reader:
        conns = []
        for c in reader.connections:
            if not namespace or c.topic.startswith(f"/{namespace}/"):
                conns.append(c)

        topic_to_sensors = {}
        for c in conns:
            for suffix, sensors in topic_map.items():
                if c.topic.endswith(suffix):
                    if c.topic not in topic_to_sensors:
                        topic_to_sensors[c.topic] = []
                    topic_to_sensors[c.topic].extend(sensors)

        matched_conns = []
        for c in conns:
            if c.topic in topic_to_sensors:
                matched_conns.append(c)

        print("\nMatched Connections:")
        for c in matched_conns:
            print(f"- {c.topic} ({topic_to_sensors[c.topic]})")
        print()

        sensors_seen = set()
        is_running = False
        last_kf_time = None

        total_msgs = sum(c.msgcount for c in matched_conns)
        pbar = tqdm(
            reader.messages(connections=matched_conns),
            total=total_msgs,
            desc="Processing FGO",
        )

        for conn, _, rawdata in pbar:
            msg = reader.deserialize(rawdata, conn.msgtype)

            sensors = topic_to_sensors[conn.topic]
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

            for sensor in sensors:
                getattr(fg, f"add_{sensor}")(t, *EXTRACTORS[sensor](msg))
                sensors_seen.add(sensor)

            if not is_running:
                if required_sensors.issubset(sensors_seen) and fg.initialize_graph(t):
                    is_running = True
                    last_kf_time = t
                continue

            trigger = False
            if (
                kf_source == "Timer"
                and "imu" in sensors
                and (t >= last_kf_time + kf_period)
            ):
                last_kf_time += kf_period
                trigger = True
            elif kf_topic and conn.topic.endswith(kf_topic):
                trigger = True

            if trigger:
                try:
                    fg.update_graph(t)
                    if res := fg.optimize_graph():
                        raw_results.append(res)
                except Exception as e:
                    tqdm.write(f"{e}\n")
                    break

    if not is_running:
        missing = required_sensors - sensors_seen
        print(f"\nGraph never initialized! Missing sensors: {missing}")

    if raw_results:
        results = {
            k: np.array([r[k] for r in raw_results]) for k in raw_results[0].keys()
        }

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

            results["x"][i] = map_t_base[0]
            results["y"][i] = map_t_base[1]
            results["z"][i] = map_t_base[2]

            q = map_R_base.as_quat()
            results["qx"][i] = q[0]
            results["qy"][i] = q[1]
            results["qz"][i] = q[2]
            results["qw"][i] = q[3]

            r, p, y = map_R_base.as_euler("xyz")
            rolls.append(r)
            pitches.append(p)
            yaws.append(y)

        results["roll"] = np.array(rolls)
        results["pitch"] = np.array(pitches)
        results["yaw"] = np.array(yaws)

        return results


def read_ground_truth(bag_path, namespace):
    pose_dict = {
        "time": [],
        "x": [],
        "y": [],
        "z": [],
        "roll": [],
        "pitch": [],
        "yaw": [],
    }
    vel_dict = {"time": [], "vx": [], "vy": [], "vz": []}
    bias_dict = {
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
        conns = []
        for c in reader.connections:
            if not namespace or c.topic.startswith(f"/{namespace}/"):
                conns.append(c)

        target_conns = []
        gt_labels = {}
        for c in conns:
            if c.topic.endswith("odometry/truth"):
                target_conns.append(c)
                gt_labels[c.topic] = "pose gt"
            elif c.topic.endswith("VelocitySensor"):
                target_conns.append(c)
                gt_labels[c.topic] = "vel gt"
            elif c.topic.endswith(f"{namespace}/imu_bias"):
                target_conns.append(c)
                gt_labels[c.topic] = "imu bias gt"

        print("\nMatched GT Connections:")
        for c in target_conns:
            print(f"- {c.topic} ({gt_labels[c.topic]})")
        print()

        total_msgs = sum(c.msgcount for c in target_conns)
        pbar = tqdm(
            reader.messages(connections=target_conns),
            total=total_msgs,
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


# %%
results = process_bag(BAG_PATH, [FLEET_CONFIG_PATH, AUV_CONFIG_PATH], NAMESPACE)
pose_gt, vel_gt, bias_gt = read_ground_truth(BAG_PATH, NAMESPACE)

# %%
if results:
    t0 = results["time"][0]
    t_fgo = results["time"] - t0

    def t_gt(data):
        return data["time"] - t0 if data else []

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

    fig, axes = plt.subplots(5, 3, figsize=(12, 10))

    for row, (keys, labels, gt_data) in enumerate(layout):
        for col, (key, label) in enumerate(zip(keys, labels)):
            ax = axes[row, col]

            if gt_data:
                ax.plot(
                    t_gt(gt_data),
                    gt_data[key],
                    "-k",
                    label="GT",
                )

            ax.plot(
                t_fgo,
                results[key],
                "-r",
                label="FGO",
            )

            ax.set_ylabel(label)
            if row == 4:
                ax.set_xlabel("Time (s)")

    plt.tight_layout()
    plt.show()
