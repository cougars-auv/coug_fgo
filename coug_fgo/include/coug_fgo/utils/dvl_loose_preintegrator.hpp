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
 * @file dvl_loose_preintegrator.hpp
 * @brief Utility for preintegrating loosely-coupled DVL velocities into relative translation.
 * @author Nelson Durrant
 * @date May 2026
 */

#pragma once

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Rot3.h>

namespace coug_fgo::utils {

/**
 * @class DvlLoosePreintegrator
 * @brief Utility for preintegrating loosely-coupled DVL velocities into relative translation.
 */
class DvlLoosePreintegrator {
 public:
  /**
   * @brief Constructor for DvlLoosePreintegrator.
   */
  DvlLoosePreintegrator() { reset(gtsam::Rot3()); }

  /**
   * @brief Resets the preintegrator state.
   * @param initial_orientation The orientation at the start of the integration period.
   * @param target_R_ahrs Static extrinsic rotation from the AHRS to the target frame.
   * @param target_R_dvl Static extrinsic rotation from the DVL to the target frame.
   * @param ahrs_cov Constant (bias-like) AHRS attitude error covariance in the AHRS frame.
   */
  void reset(const gtsam::Rot3& initial_orientation,
             const gtsam::Rot3& target_R_ahrs = gtsam::Rot3(),
             const gtsam::Rot3& target_R_dvl = gtsam::Rot3(),
             const gtsam::Matrix3& ahrs_cov = gtsam::Matrix3::Zero()) {
    map_R_i_ = initial_orientation;
    target_R_ahrs_ = target_R_ahrs.matrix();
    dvl_R_ahrs_ = (target_R_dvl.inverse() * target_R_ahrs).matrix();
    ahrs_cov_ = ahrs_cov;
    measured_translation_ = gtsam::Vector3::Zero();
    covariance_ = gtsam::Matrix3::Zero();
    J_ahrs_ = gtsam::Matrix3::Zero();
  }

  /**
   * @brief Integrates a new DVL velocity measurement.
   * @param measured_vel The velocity measurement in the DVL sensor frame.
   * @param measured_orientation The current DVL orientation estimate (map_R_dvl).
   * @param dt The time delta since the last measurement.
   * @param measured_cov The measurement noise covariance.
   */
  void integrateMeasurement(const gtsam::Vector3& measured_vel,
                            const gtsam::Rot3& measured_orientation, double dt,
                            const gtsam::Matrix3& measured_cov) {
    // Relative rotation from the integration start frame
    gtsam::Rot3 i_R_k = map_R_i_.between(measured_orientation);

    // Accumulate the position change in the start frame
    gtsam::Vector3 vel_in_i = i_R_k.rotate(measured_vel);
    measured_translation_ += vel_in_i * dt;

    // Propagate measurement uncertainty into the covariance
    gtsam::Matrix3 J = i_R_k.matrix() * dt;
    covariance_ += J * measured_cov * J.transpose();

    // A constant AHRS attitude error perturbs the start-frame reference and each step:
    // d(delta_p) = sum_k dt * ([i_R_k v_k]x target_R_ahrs - i_R_k [v_k]x dvl_R_ahrs) delta_theta
    J_ahrs_ += dt * (gtsam::skewSymmetric(vel_in_i) * target_R_ahrs_ -
                     i_R_k.matrix() * gtsam::skewSymmetric(measured_vel) * dvl_R_ahrs_);
  }

  /**
   * @brief Gets the preintegrated translation delta.
   * @return The translation delta in the starting frame.
   */
  gtsam::Vector3 delta() const { return measured_translation_; }

  /**
   * @brief Gets the accumulated translation covariance (DVL noise + AHRS attitude error).
   * @return The 3x3 covariance matrix.
   */
  gtsam::Matrix3 covariance() const {
    return covariance_ + J_ahrs_ * ahrs_cov_ * J_ahrs_.transpose();
  }

 private:
  gtsam::Rot3 map_R_i_;
  gtsam::Matrix3 target_R_ahrs_;
  gtsam::Matrix3 dvl_R_ahrs_;
  gtsam::Matrix3 ahrs_cov_;
  gtsam::Vector3 measured_translation_;
  gtsam::Matrix3 covariance_;
  gtsam::Matrix3 J_ahrs_;
};

}  // namespace coug_fgo::utils
