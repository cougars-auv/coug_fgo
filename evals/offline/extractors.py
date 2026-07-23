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

import numpy as np


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
    "multiagent": lambda m: (
        m.header.frame_id,
        (
            _stamp(m),
            _vec3(m.local_odometry.position),
            _quat(m.local_odometry.orientation),
            _cov(m.odometry_covariance, 6),
            m.pressure_depth,
            _quat(m.imu_orientation),
            m.includes_range,
            m.range_dist,
            m.includes_usbl,
            m.usbl_azimuth,
            m.usbl_elevation,
            m.includes_position,
            m.position_depth,
        ),
    ),
}
