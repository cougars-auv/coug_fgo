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
 * @file types.hpp
 * @brief Shared types for the factor graph system.
 * @author Nelson Durrant
 * @date Jan 2026
 */

#pragma once

#include <gtsam/geometry/Pose3.h>

#include <deque>
#include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>
#include <memory>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/magnetic_field.hpp>

namespace coug_fgo::utils {

/**
 * @struct TfBundle
 * @brief Container for TF sensor transformations.
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
 * @struct QueueBundle
 * @brief Container for drained sensor message deques.
 */
struct QueueBundle {
  std::deque<sensor_msgs::msg::Imu::SharedPtr> imu;
  std::deque<nav_msgs::msg::Odometry::SharedPtr> gps;
  std::deque<nav_msgs::msg::Odometry::SharedPtr> depth;
  std::deque<sensor_msgs::msg::MagneticField::SharedPtr> mag;
  std::deque<sensor_msgs::msg::Imu::SharedPtr> ahrs;
  std::deque<geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr> dvl;
  std::deque<geometry_msgs::msg::WrenchStamped::SharedPtr> wrench;
};

}  // namespace coug_fgo::utils
