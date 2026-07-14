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
 * @brief Initializer for state priors from sensor data.
 * @author Nelson Durrant
 * @date May 2026
 */

#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/navigation/ImuBias.h>

#include <memory>
#include <optional>

#include "coug_fgo/factor_graph_parameters.hpp"
#include "coug_fgo/utils/data_types.hpp"

namespace coug_fgo::utils {

/**
 * @class StateInitializer
 * @brief Initializer for state priors from sensor data.
 */
class StateInitializer {
 public:
  /**
   * @brief Constructs the initializer with node parameters.
   * @param params Node parameters (held by reference; must outlive this object).
   */
  explicit StateInitializer(const factor_graph_node::Params& params);

  /**
   * @brief Accumulates sensor data and computes the initial state once ready.
   * @param current_time Current time in seconds.
   * @param queues Bundle of drained sensor message deques.
   * @param tfs Bundle of core sensor transformations.
   * @return The computed initial state once ready, or nullopt while collecting.
   */
  std::optional<InitialState> update(double current_time, utils::QueueBundle& queues,
                                     const utils::TfBundle& tfs);

 private:
  /**
   * @brief Collects sensor samples until enough data is in hand.
   * @param current_time Current time in seconds.
   * @param queues Bundle of drained sensor message deques to process.
   * @return True once the collected samples can seed the initial state.
   */
  bool accumulate(double current_time, utils::QueueBundle& queues);

  /**
   * @brief Computes the initial pose, velocity, bias, and start time from the collected data.
   * @param tfs Bundle of core sensor transformations.
   * @return The computed initial state and the averaged samples behind it.
   */
  InitialState compute(const utils::TfBundle& tfs) const;

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
  gtsam::Rot3 computeInitialOrientation(const utils::TfBundle& tfs) const;

  /**
   * @brief Computes initial position using GPS and depth with lever arm compensation.
   * @param map_R_target The computed initial rotation.
   * @param tfs SE(3) sensor transforms for lever arm compensation.
   * @return Initial position of the target frame in the map frame.
   */
  gtsam::Point3 computeInitialPosition(const gtsam::Rot3& map_R_target,
                                       const utils::TfBundle& tfs) const;

  /**
   * @brief Computes initial map-frame velocity from DVL body-frame measurements.
   * @param map_R_target The computed initial rotation.
   * @param tfs SE(3) sensor transforms for lever arm compensation.
   * @return Initial velocity of the target frame in the map frame.
   */
  gtsam::Vector3 computeInitialVelocity(const gtsam::Rot3& map_R_target,
                                        const utils::TfBundle& tfs) const;

  /**
   * @brief Computes initial IMU bias from averaged gyroscope readings.
   * @return Initial accelerometer and gyroscope bias estimate.
   */
  gtsam::imuBias::ConstantBias computeInitialBias() const;

  const factor_graph_node::Params& params_;
  double start_avg_time_{0.0};
  size_t imu_count_ = 0, gps_count_ = 0, depth_count_ = 0, mag_count_ = 0, ahrs_count_ = 0,
         dvl_count_ = 0;
  gtsam::Rot3 ahrs_ref_;
  gtsam::Vector3 ahrs_log_sum_ = gtsam::Vector3::Zero();

  std::shared_ptr<utils::ImuData> initial_imu_;
  std::shared_ptr<utils::OdometryData> initial_gps_;
  std::shared_ptr<utils::OdometryData> initial_depth_;
  std::shared_ptr<utils::AhrsData> initial_ahrs_;
  std::shared_ptr<utils::MagneticFieldData> initial_mag_;
  std::shared_ptr<utils::TwistData> initial_dvl_;
};

}  // namespace coug_fgo::utils
