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
 * @file dvl_preintegrator.hpp
 * @brief Utility for preintegrating DVL velocity measurements into relative translation.
 * @author Nelson Durrant
 * @date Jan 2026
 */

#pragma once

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>

namespace coug_fgo::utils
{

/**
 * @class DvlPreintegrator
 * @brief Utility for preintegrating DVL velocity measurements into relative translation.
 */
class DvlPreintegrator
{
public:
  /**
   * @brief Constructor for DvlPreintegrator.
   */
  DvlPreintegrator() {reset(gtsam::Rot3());}

  /**
   * @brief Resets the preintegrator state.
   * @param initial_orientation The orientation at the start of the integration period.
   */
  void reset(const gtsam::Rot3 & initial_orientation)
  {
    world_R_i_ = initial_orientation;
    i_p_k_ = gtsam::Vector3::Zero();
    covariance_ = gtsam::Matrix3::Zero();
  }

  /**
   * @brief Integrates a new DVL velocity measurement.
   * @param measured_vel The velocity measurement in the body frame.
   * @param measured_orientation The current orientation estimate.
   * @param dt The time delta since the last measurement.
   * @param measured_cov The measurement noise covariance.
   */
  void integrateMeasurement(
    const gtsam::Vector3 & measured_vel,
    const gtsam::Rot3 & measured_orientation, double dt,
    const gtsam::Matrix3 & measured_cov)
  {
    // Relative rotation from the integration start frame
    gtsam::Rot3 i_R_k = world_R_i_.inverse() * measured_orientation;

    // Accumulate the position change in the start frame
    gtsam::Vector3 p_i = i_R_k.rotate(measured_vel);
    i_p_k_ += p_i * dt;

    // Propagate measurement uncertainty into the covariance
    gtsam::Matrix3 J = i_R_k.matrix() * dt;
    covariance_ += J * measured_cov * J.transpose();
  }

  /**
   * @brief Gets the preintegrated translation delta.
   * @return The translation delta in the starting frame.
   */
  gtsam::Vector3 delta() const {return i_p_k_;}

  /**
   * @brief Gets the accumulated translation covariance.
   * @return The 3x3 covariance matrix.
   */
  gtsam::Matrix3 covariance() const {return covariance_;}

private:
  gtsam::Rot3 world_R_i_;
  gtsam::Vector3 i_p_k_;
  gtsam::Matrix3 covariance_;
};

}  // namespace coug_fgo::utils
