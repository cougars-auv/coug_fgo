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
import yaml
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

FGO_LIB_PATH = str(
    Path.home() / "cougars-dev/ros2_ws/install/coug_fgo/lib/python3.10/site-packages"
)
sys.path.insert(0, FGO_LIB_PATH)
import pybind11fgo  # noqa: E402

# %%
BAG_PATH = str(Path.home() / "cougars-dev/bags/launch_2026-03-24-11-26-15")
CONFIG_PATH = str(
    Path.home() / "cougars-dev/ros2_ws/src/coug_fgo/coug_fgo/config/batch_params.yaml"
)
NAMESPACE = "coug0sim"

with open(CONFIG_PATH, "r") as file:
    config_yaml = yaml.safe_load(file)

params = config_yaml.get("/**", {}).get("ros__parameters", config_yaml)

kf_source = params.get("keyframe_source", "DVL")
KEYFRAME_TOPIC = {
    "DVL": params.get("dvl_topic", "dvl/twist"),
    "Depth": params.get("depth_odom_topic", "odometry/depth"),
}.get(kf_source, "dvl/twist")

TOPIC_MAP = {
    params.get("imu_topic", "imu/data"): ("sensor_msgs/msg/Imu", "imu"),
    params.get("gps_odom_topic", "odometry/gps"): ("nav_msgs/msg/Odometry", "gps"),
    params.get("depth_odom_topic", "odometry/depth"): (
        "nav_msgs/msg/Odometry",
        "depth",
    ),
    params.get("mag_topic", "imu/mag"): ("sensor_msgs/msg/MagneticField", "mag"),
    params.get("ahrs_topic", "imu/ahrs"): ("sensor_msgs/msg/Imu", "ahrs"),
    params.get("dvl_topic", "dvl/twist"): (
        "geometry_msgs/msg/TwistWithCovarianceStamped",
        "dvl",
    ),
    params.get("wrench_topic", "cmd_wrench"): (
        "geometry_msgs/msg/WrenchStamped",
        "wrench",
    ),
}

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
typestore = get_typestore(Stores.ROS2_HUMBLE)
fg = pybind11fgo.FactorGraphPy(CONFIG_PATH)

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
if not keyframe_full_topic:
    print(f"Warning: keyframe topic '{KEYFRAME_TOPIC}' not found in bag.")

# %%
matched_conns = [c for c in reader.connections if c.topic in topic_sensor_map]
for conn, _, rawdata in reader.messages(connections=matched_conns):
    msg = reader.deserialize(rawdata, conn.msgtype)
    sensor = topic_sensor_map[conn.topic]
    timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    args = EXTRACTORS[sensor](msg)
    getattr(fg, f"add_{sensor}")(timestamp, *args)

    if not fg.initialize_graph(timestamp):
        continue

    if conn.topic == keyframe_full_topic:
        fg.update_graph(timestamp)
        if result := fg.optimize_graph():
            results.append(result)

reader.close()
print(f"Processed {len(results)} keyframes.")
