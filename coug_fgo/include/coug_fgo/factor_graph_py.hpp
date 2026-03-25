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
 * @file factor_graph_py.hpp
 * @brief Python bindings wrapper for the FactorGraphCore.
 * @author Nelson Durrant
 * @date Jan 2026
 */

#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>
#include <memory>
#include <string>

#include "coug_fgo/factor_graph_core.hpp"
#include "coug_fgo/factor_graph_parameters.hpp"
#include "coug_fgo/utils/state_initializer.hpp"

/**
 * @class FactorGraphPy
 * @brief Standalone wrapper around FactorGraphCore for use from Python via pybind11.
 */
class FactorGraphPy {
 public:
  /**
   * @brief Constructs the wrapper, loading parameters from a ROS 2 YAML config file.
   * @param config_path Path to the ROS 2 parameter YAML file.
   */
  explicit FactorGraphPy(const std::string& config_path);

  // --- Sensor Input ---
  /**
   * @brief Queues an IMU measurement.
   * @param timestamp Message timestamp in seconds.
   * @param accel Linear acceleration (x, y, z) in m/s^2.
   * @param gyro Angular velocity (x, y, z) in rad/s.
   * @param accel_cov 3x3 linear acceleration covariance.
   * @param gyro_cov 3x3 angular velocity covariance.
   */
  void add_imu(double timestamp, const Eigen::Vector3d& accel, const Eigen::Vector3d& gyro,
               const Eigen::Matrix3d& accel_cov, const Eigen::Matrix3d& gyro_cov);

  /**
   * @brief Queues a DVL velocity measurement.
   * @param timestamp Message timestamp in seconds.
   * @param velocity Linear velocity (x, y, z) in m/s.
   * @param twist_cov 6x6 twist covariance.
   */
  void add_dvl(double timestamp, const Eigen::Vector3d& velocity,
               const Eigen::Matrix<double, 6, 6>& twist_cov);

  /**
   * @brief Queues an AHRS orientation measurement.
   * @param timestamp Message timestamp in seconds.
   * @param quat_xyzw Orientation quaternion (x, y, z, w).
   * @param orientation_cov 3x3 orientation covariance.
   */
  void add_ahrs(double timestamp, const Eigen::Vector4d& quat_xyzw,
                const Eigen::Matrix3d& orientation_cov);

  /**
   * @brief Queues a depth measurement.
   * @param timestamp Message timestamp in seconds.
   * @param depth_z Depth in meters (z-axis position).
   * @param pose_cov 6x6 pose covariance.
   */
  void add_depth(double timestamp, double depth_z, const Eigen::Matrix<double, 6, 6>& pose_cov);

  /**
   * @brief Queues a GPS position measurement.
   * @param timestamp Message timestamp in seconds.
   * @param position Position (x, y, z) in meters.
   * @param pose_cov 6x6 pose covariance.
   */
  void add_gps(double timestamp, const Eigen::Vector3d& position,
               const Eigen::Matrix<double, 6, 6>& pose_cov);

  /**
   * @brief Queues a magnetometer measurement.
   * @param timestamp Message timestamp in seconds.
   * @param mag_field Magnetic field (x, y, z) in Tesla.
   * @param mag_cov 3x3 magnetic field covariance.
   */
  void add_mag(double timestamp, const Eigen::Vector3d& mag_field, const Eigen::Matrix3d& mag_cov);

  /**
   * @brief Queues a wrench (force/torque) measurement.
   * @param timestamp Message timestamp in seconds.
   * @param force_torque 6-vector (fx, fy, fz, tx, ty, tz).
   */
  void add_wrench(double timestamp, const Eigen::VectorXd& force_torque);

  // --- Main Logic ---
  /**
   * @brief Initializes the factor graph using averaged sensor data.
   * @param current_time Current timestamp in seconds.
   * @return True if the graph is initialized (or was already initialized).
   */
  bool initialize_graph(double current_time);

  /**
   * @brief Drains queues and delegates factor building to the core.
   * @param target_time Keyframe timestamp in seconds.
   * @return True if the update succeeded.
   */
  bool update_graph(double target_time);

  /**
   * @brief Delegates optimization to the core and returns the result.
   * @return Dict with pose (x, y, z, qx, qy, qz, qw), velocity, biases, and time.
   *         Empty dict if the graph is not yet initialized.
   */
  pybind11::dict optimize_graph();

 private:
  /**
   * @brief Extracts sensor-to-target transforms from the parameter struct.
   * @param p The loaded parameters.
   * @return Bundle of GTSAM Pose3 transforms.
   */
  static coug_fgo::utils::TfBundle extract_tfs(const factor_graph_node::Params& p);

  // --- Core ---
  factor_graph_node::Params params_;
  std::unique_ptr<coug_fgo::FactorGraphCore> core_;
  std::unique_ptr<coug_fgo::utils::StateInitializer> state_init_;

  // --- State ---
  coug_fgo::utils::TfBundle tfs_;
  coug_fgo::utils::QueueBundle queues_;
  bool is_initialized_{false};
};
