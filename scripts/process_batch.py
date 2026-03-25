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
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

sys.path.insert(
    0,
    str(
        Path.home()
        / "cougars-dev/ros2_ws/install/coug_fgo/lib/python3.10/site-packages"
    ),
)
import pybind11fgo

# %%

BAG_PATH = str(Path.home() / "cougars-dev/bags/launch_2026-03-24-11-26-15")
CONFIG_PATH = str(
    Path.home() / "cougars-dev/ros2_ws/src/coug_fgo/coug_fgo/config/batch_params.yaml"
)
NAMESPACE = "coug0sim"
KEYFRAME_TOPIC = "dvl/twist"
TOPIC_MAP = {
    "imu/data": ("sensor_msgs/msg/Imu", "imu"),
    "odometry/gps": ("nav_msgs/msg/Odometry", "gps"),
    "odometry/depth": ("nav_msgs/msg/Odometry", "depth"),
    "imu/mag": ("sensor_msgs/msg/MagneticField", "mag"),
    "imu/ahrs": ("sensor_msgs/msg/Imu", "ahrs"),
    "dvl/twist": ("geometry_msgs/msg/TwistWithCovarianceStamped", "dvl"),
    "cmd_wrench": ("geometry_msgs/msg/WrenchStamped", "wrench"),
}

# %% Open bag and build topic mapping

typestore = get_typestore(Stores.ROS2_HUMBLE)
fg = pybind11fgo.FactorGraphPy(str(CONFIG_PATH))

topic_sensor_map = {}
results = []

reader = AnyReader([Path(BAG_PATH)], default_typestore=typestore)
reader.open()

for conn in reader.connections:
    if NAMESPACE and not conn.topic.startswith(f"/{NAMESPACE}/"):
        continue
    for suffix, (msg_type, sensor) in TOPIC_MAP.items():
        if conn.topic.endswith(suffix) and conn.msgtype == msg_type:
            topic_sensor_map[conn.topic] = sensor
            break

print(f"Mapped topics: {topic_sensor_map}")

keyframe_full_topic = next(
    (t for t in topic_sensor_map if t.endswith(KEYFRAME_TOPIC)), None
)
if keyframe_full_topic is None:
    print(f"Warning: keyframe topic '{KEYFRAME_TOPIC}' not found in bag.")

# %% Process all messages

matched_conns = [c for c in reader.connections if c.topic in topic_sensor_map]

for conn, _, rawdata in reader.messages(connections=matched_conns):
    msg = reader.deserialize(rawdata, conn.msgtype)
    sensor = topic_sensor_map[conn.topic]
    timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    if sensor == "imu":
        fg.add_imu(
            timestamp,
            np.array(
                [
                    msg.linear_acceleration.x,
                    msg.linear_acceleration.y,
                    msg.linear_acceleration.z,
                ]
            ),
            np.array(
                [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
            ),
            np.array(msg.linear_acceleration_covariance).reshape(3, 3),
            np.array(msg.angular_velocity_covariance).reshape(3, 3),
        )

    elif sensor == "dvl":
        fg.add_dvl(
            timestamp,
            np.array(
                [
                    msg.twist.twist.linear.x,
                    msg.twist.twist.linear.y,
                    msg.twist.twist.linear.z,
                ]
            ),
            np.array(msg.twist.covariance).reshape(6, 6),
        )

    elif sensor == "ahrs":
        fg.add_ahrs(
            timestamp,
            np.array(
                [
                    msg.orientation.x,
                    msg.orientation.y,
                    msg.orientation.z,
                    msg.orientation.w,
                ]
            ),
            np.array(msg.orientation_covariance).reshape(3, 3),
        )

    elif sensor == "depth":
        fg.add_depth(
            timestamp,
            msg.pose.pose.position.z,
            np.array(msg.pose.covariance).reshape(6, 6),
        )

    elif sensor == "gps":
        fg.add_gps(
            timestamp,
            np.array(
                [
                    msg.pose.pose.position.x,
                    msg.pose.pose.position.y,
                    msg.pose.pose.position.z,
                ]
            ),
            np.array(msg.pose.covariance).reshape(6, 6),
        )

    elif sensor == "mag":
        fg.add_mag(
            timestamp,
            np.array(
                [msg.magnetic_field.x, msg.magnetic_field.y, msg.magnetic_field.z]
            ),
            np.array(msg.magnetic_field_covariance).reshape(3, 3),
        )

    elif sensor == "wrench":
        fg.add_wrench(
            timestamp,
            np.array(
                [
                    msg.wrench.force.x,
                    msg.wrench.force.y,
                    msg.wrench.force.z,
                    msg.wrench.torque.x,
                    msg.wrench.torque.y,
                    msg.wrench.torque.z,
                ]
            ),
        )

    if not fg.initialize_graph(timestamp):
        continue

    if conn.topic == keyframe_full_topic:
        fg.update_graph(timestamp)
        result = fg.optimize_graph()
        if result:
            results.append(result)

reader.close()
print(f"Processed {len(results)} keyframes.")

# %% Plot results

if results:
    import matplotlib.pyplot as plt

    x = [r["x"] for r in results]
    y = [r["y"] for r in results]
    z = [r["z"] for r in results]
    t = [r["time"] - results[0]["time"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    sc = ax1.scatter(x, y, c=t, cmap="viridis", s=5)
    ax1.plot(x[0], y[0], "go", markersize=10, label="Start")
    ax1.plot(x[-1], y[-1], "ro", markersize=10, label="End")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_title("Top-Down Trajectory")
    ax1.legend()
    ax1.set_aspect("equal")
    fig.colorbar(sc, ax=ax1, label="Time (s)")

    ax2.plot(t, z)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Z (m)")
    ax2.set_title("Depth Profile")

    plt.tight_layout()
    plt.show()
