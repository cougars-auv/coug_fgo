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

FGO_LIB_PATH = str(
    Path.home() / "cougars-dev/ros2_ws/install/coug_fgo/lib/python3.10/site-packages"
)
sys.path.insert(0, FGO_LIB_PATH)
import pybind11fgo  # noqa: E402

NAMESPACE = "coug0sim"
BAG_PATH = str(Path.home() / "cougars-dev/bags/launch_2026-03-24-11-26-15")
CONFIG_PATH = str(
    Path.home() / "cougars-dev/ros2_ws/src/coug_fgo/scripts/batch_params.yaml"
)
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
def load_config(config_path):
    print(f"Loading configuration from: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    params = config.get("/**", {}).get("ros__parameters", config)

    topic_map = {
        params.get("imu_topic", "imu/data"): "imu",
        params.get("gps_odom_topic", "odometry/gps"): "gps",
        params.get("depth_odom_topic", "odometry/depth"): "depth",
        params.get("mag_topic", "imu/mag"): "mag",
        params.get("ahrs_topic", "imu/ahrs"): "ahrs",
        params.get("dvl_topic", "dvl/twist"): "dvl",
        params.get("wrench_topic", "cmd_wrench"): "wrench",
    }

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

    return topic_map, required_sensors, kf_source, kf_topic, kf_period


def process_bag(bag_path, config_path, namespace):
    topic_map, required_sensors, kf_source, kf_topic, kf_period = load_config(
        config_path
    )

    fg = pybind11fgo.FactorGraphPy(config_path)
    raw_results = []

    print(f"Opening bag: {bag_path}")
    with AnyReader(
        [Path(bag_path)], default_typestore=get_typestore(Stores.ROS2_HUMBLE)
    ) as reader:
        conns = []
        for c in reader.connections:
            if not namespace or c.topic.startswith(f"/{namespace}/"):
                conns.append(c)

        topic_to_sensor = {}
        for c in conns:
            for suffix, sensor in topic_map.items():
                if c.topic.endswith(suffix):
                    topic_to_sensor[c.topic] = sensor

        matched_conns = []
        for c in conns:
            if c.topic in topic_to_sensor:
                matched_conns.append(c)

        print("\nMatched Connections:")
        for c in matched_conns:
            print(f"- {c.topic} ({topic_to_sensor[c.topic]})")
        print()

        sensors_seen = set()
        is_running = False
        last_kf_time = None
        last_print_time = None

        for conn, _, rawdata in reader.messages(connections=matched_conns):
            msg = reader.deserialize(rawdata, conn.msgtype)

            sensor = topic_to_sensor[conn.topic]
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            getattr(fg, f"add_{sensor}")(t, *EXTRACTORS[sensor](msg))
            sensors_seen.add(sensor)

            if not is_running:
                if required_sensors.issubset(sensors_seen) and fg.initialize_graph(t):
                    is_running = True
                    last_kf_time = t
                    print(f"{t:.3f} s: Initialized with sensors: {sensors_seen}")
                continue

            trigger = False
            if (
                kf_source == "Timer"
                and sensor == "imu"
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
                    print(f"{e}\n")
                    break

            if last_print_time is None or t - last_print_time >= 30.0:
                print(f"{t:.3f} s: Processed {len(raw_results)} keyframes...")
                last_print_time = t

    print(f"Finished processing bag. Total keyframes: {len(raw_results)}")

    if raw_results:
        results = {
            k: np.array([r[k] for r in raw_results]) for k in raw_results[0].keys()
        }

        rolls, pitches, yaws = [], [], []
        for i in range(len(raw_results)):
            r, p, y = R.from_quat(
                [results["qx"][i], results["qy"][i], results["qz"][i], results["qw"][i]]
            ).as_euler("xyz")
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

        for conn, _, rawdata in reader.messages(connections=target_conns):
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
results = process_bag(BAG_PATH, CONFIG_PATH, NAMESPACE)
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
    print("Displaying results...")
    plt.show()
