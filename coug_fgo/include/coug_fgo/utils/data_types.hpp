// Copyright (c) 2026 BYU FROST Lab
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @file data_types.hpp
 * @brief Shared data types for the factor graph core.
 * @author Nelson Durrant
 * @date Jan 2026
 */

#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <Eigen/Dense>
#include <deque>
#include <memory>

namespace coug_fgo::utils {

/**
 * @struct TfBundle
 * @brief Container for static TF sensor transformations.
 */
struct TfBundle {
  gtsam::Pose3 target_T_imu;
  gtsam::Pose3 target_T_gps;
  gtsam::Pose3 target_T_depth;
  gtsam::Pose3 target_T_mag;
  gtsam::Pose3 target_T_ahrs;
  gtsam::Pose3 target_T_dvl;
  gtsam::Pose3 target_T_base;
  gtsam::Pose3 target_T_com;
};

/**
 * @struct ImuData
 * @brief C++ container for IMU data.
 */
struct ImuData {
  double timestamp;
  gtsam::Vector3 linear_acceleration;
  gtsam::Vector3 angular_velocity;
  gtsam::Matrix3 linear_acceleration_covariance;
  gtsam::Matrix3 angular_velocity_covariance;
};

/**
 * @struct OdometryData
 * @brief C++ container for Odometry data.
 */
struct OdometryData {
  double timestamp;
  gtsam::Pose3 pose;
  gtsam::Matrix66 pose_covariance;
};

/**
 * @struct MagneticFieldData
 * @brief C++ container for Magnetometer data.
 */
struct MagneticFieldData {
  double timestamp;
  gtsam::Vector3 magnetic_field;
  gtsam::Matrix3 magnetic_field_covariance;
};

/**
 * @struct AhrsData
 * @brief C++ container for AHRS data.
 */
struct AhrsData {
  double timestamp;
  gtsam::Rot3 orientation;
  gtsam::Matrix3 orientation_covariance;
};

/**
 * @struct TwistData
 * @brief C++ container for Twist data.
 */
struct TwistData {
  double timestamp;
  gtsam::Vector3 linear_velocity;
  gtsam::Matrix66 twist_covariance;
};

/**
 * @struct WrenchData
 * @brief C++ container for Wrench data.
 */
struct WrenchData {
  double timestamp;
  gtsam::Vector3 force;
  gtsam::Vector3 torque;
};

/**
 * @struct QueueBundle
 * @brief Container for sensor data queues.
 */
struct QueueBundle {
  std::deque<std::shared_ptr<ImuData>> imu;
  std::deque<std::shared_ptr<OdometryData>> gps;
  std::deque<std::shared_ptr<OdometryData>> depth;
  std::deque<std::shared_ptr<MagneticFieldData>> mag;
  std::deque<std::shared_ptr<AhrsData>> ahrs;
  std::deque<std::shared_ptr<TwistData>> dvl;
  std::deque<std::shared_ptr<WrenchData>> wrench;
};

}  // namespace coug_fgo::utils
