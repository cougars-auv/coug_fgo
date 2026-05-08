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
 * @file state_initializer.hpp
 * @brief Utility for initializing state priors from sensor data.
 * @author Nelson Durrant
 * @date May 2026
 */

#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/navigation/ImuBias.h>

#include <memory>

#include "coug_fgo/factor_graph_parameters.hpp"
#include "coug_fgo/utils/data_types.hpp"

namespace coug_fgo {

/**
 * @class StateInitializer
 * @brief Computes initial state priors for factor graph initialization.
 */
class StateInitializer {
 public:
  /**
   * @brief Constructor for StateInitializer.
   * @param params Node parameters; must outlive this object.
   */
  explicit StateInitializer(const factor_graph_node::Params& params);

  /**
   * @brief Updates the running averages with new data from sensor queues.
   * @param current_time Current time in seconds.
   * @param queues Bundle of sensor message queues.
   * @return True if initialization averaging is complete.
   */
  bool update(double current_time, utils::QueueBundle& queues);

  /**
   * @brief Computes initial pose, velocity, and bias.
   * @param tfs Bundle of core sensor transformations.
   */
  void compute(const utils::TfBundle& tfs);

  const gtsam::Pose3& getPose() const { return pose_; }
  const gtsam::Vector3& getVelocity() const { return velocity_; }
  const gtsam::imuBias::ConstantBias& getBias() const { return bias_; }
  double getTime() const { return time_; }

  const std::shared_ptr<utils::ImuData>& getInitialImu() const { return initial_imu_; }
  const std::shared_ptr<utils::OdometryData>& getInitialGps() const { return initial_gps_; }
  const std::shared_ptr<utils::OdometryData>& getInitialDepth() const { return initial_depth_; }
  const std::shared_ptr<utils::AhrsData>& getInitialAhrs() const { return initial_ahrs_; }
  const std::shared_ptr<utils::MagneticFieldData>& getInitialMag() const { return initial_mag_; }
  const std::shared_ptr<utils::TwistData>& getInitialDvl() const { return initial_dvl_; }

 private:
  /**
   * @brief Accumulates running averages from drained sensor message deques.
   * @param queues Bundle of drained sensor message deques to process.
   */
  void incrementAverages(utils::QueueBundle& queues);

  /**
   * @brief Computes initial orientation from accelerometer tilt and heading sensors.
   * @param tfs SE(3) sensor transforms for lever arm compensation.
   * @return Initial rotation of the target frame in the map frame.
   */
  gtsam::Rot3 computeInitialOrientation(const utils::TfBundle& tfs);

  /**
   * @brief Computes initial position using GPS and depth with lever arm compensation.
   * @param map_R_target The computed initial rotation.
   * @param tfs SE(3) sensor transforms for lever arm compensation.
   * @return Initial position of the target frame in the map frame.
   */
  gtsam::Point3 computeInitialPosition(const gtsam::Rot3& map_R_target, const utils::TfBundle& tfs);

  /**
   * @brief Computes initial map-frame velocity from DVL body-frame measurements.
   * @param map_R_target The computed initial rotation.
   * @param tfs SE(3) sensor transforms for lever arm compensation.
   * @return Initial velocity of the target frame in the map frame.
   */
  gtsam::Vector3 computeInitialVelocity(const gtsam::Rot3& map_R_target,
                                        const utils::TfBundle& tfs);

  /**
   * @brief Computes initial IMU bias from averaged gyroscope readings.
   * @return Initial accelerometer and gyroscope bias estimate.
   */
  gtsam::imuBias::ConstantBias computeInitialBias();

  const factor_graph_node::Params& params_;
  double start_avg_time_{0.0};
  size_t imu_count_ = 0, gps_count_ = 0, depth_count_ = 0, mag_count_ = 0, ahrs_count_ = 0,
         dvl_count_ = 0;
  gtsam::Rot3 ahrs_ref_;
  gtsam::Vector3 ahrs_log_sum_ = gtsam::Vector3::Zero();

  gtsam::Pose3 pose_;
  gtsam::Vector3 velocity_;
  gtsam::imuBias::ConstantBias bias_;
  double time_{0.0};

  std::shared_ptr<utils::ImuData> initial_imu_;
  std::shared_ptr<utils::OdometryData> initial_gps_;
  std::shared_ptr<utils::OdometryData> initial_depth_;
  std::shared_ptr<utils::AhrsData> initial_ahrs_;
  std::shared_ptr<utils::MagneticFieldData> initial_mag_;
  std::shared_ptr<utils::TwistData> initial_dvl_;
};

}  // namespace coug_fgo
