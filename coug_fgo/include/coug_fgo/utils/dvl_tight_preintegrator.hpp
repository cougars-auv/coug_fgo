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
 * @file dvl_tight_preintegrator.hpp
 * @brief Utility for preintegrating tightly-coupled DVL velocity measurements into relative
 * translation.
 * @author Nelson Durrant
 * @date May 2026
 */

#pragma once

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>

namespace coug_fgo::utils {

/**
 * @class DvlTightPreintegrator
 * @brief Utility for preintegrating tightly-coupled DVL velocity measurements into relative
 * translation.
 */
class DvlTightPreintegrator {
 public:
  /**
   * @brief Constructor for DvlTightPreintegrator.
   */
  DvlTightPreintegrator() { reset(); }

  /**
   * @brief Resets the preintegrator state.
   */
  void reset() {
    measured_translation_ = gtsam::Vector3::Zero();
    covariance_ = gtsam::Matrix3::Zero();
    d_translation_d_bias_ = gtsam::Matrix3::Zero();
    cross_cov_rot_trans_ = gtsam::Matrix3::Zero();
    prev_delta_R_ik_ = gtsam::Rot3();
  }

  /**
   * @brief Integrates a new DVL velocity measurement with joint covariance propagation (Forster et
   * al., TRO 2017)
   *
   * @param measured_vel The velocity measurement in the DVL sensor frame.
   * @param delta_R_ik Relative target-frame rotation from interval start (i) to measurement (k).
   * @param target_R_dvl Static extrinsic rotation from the DVL to the target frame.
   * @param dt The time delta since the last measurement.
   * @param measured_cov The measurement noise covariance.
   * @param rot_cov_k Right-perturbation covariance of delta_R_ik from the IMU preintegrator.
   * @param J_bg_k Jacobian of delta_R_ik w.r.t. the gyro bias, in the manifold convention
   * delta_R_ik(b + db) ~= delta_R_ik(b) * Exp(J_bg_k * db).
   */
  void integrateMeasurement(const gtsam::Vector3& measured_vel, const gtsam::Rot3& delta_R_ik,
                            const gtsam::Rot3& target_R_dvl, double dt,
                            const gtsam::Matrix3& measured_cov, const gtsam::Matrix3& rot_cov_k,
                            const gtsam::Matrix3& J_bg_k) {
    // Transport the rotation error frame from the previous step to this one:
    // delta_phi_k = delta_R_{k-1,k}^T * delta_phi_{k-1} + (new gyro noise)
    gtsam::Matrix3 A = (prev_delta_R_ik_.inverse() * delta_R_ik).matrix().transpose();
    cross_cov_rot_trans_ = A * cross_cov_rot_trans_;

    // Calculate velocity in the target frame at time k
    gtsam::Vector3 vel_in_target = target_R_dvl.rotate(measured_vel);

    // Rotate into the target frame at the start of the interval (time i)
    gtsam::Vector3 vel_in_i = delta_R_ik.rotate(vel_in_target);
    measured_translation_ += vel_in_i * dt;

    // Compute Jacobians for uncertainty and bias correction
    // J_vel maps DVL velocity noise to translation noise
    gtsam::Matrix3 J_vel = delta_R_ik.matrix() * target_R_dvl.matrix() * dt;

    // J_rot maps Gyro rotation noise to translation noise
    gtsam::Matrix3 J_rot = -delta_R_ik.matrix() * gtsam::skewSymmetric(vel_in_target) * dt;

    // Joint propagation: delta_p_new = delta_p + J_rot * delta_phi + J_vel * delta_v
    covariance_ += (J_vel * measured_cov * J_vel.transpose()) +
                   (J_rot * rot_cov_k * J_rot.transpose()) + (J_rot * cross_cov_rot_trans_) +
                   (J_rot * cross_cov_rot_trans_).transpose();
    cross_cov_rot_trans_ += rot_cov_k * J_rot.transpose();

    // Accumulate first-order Jacobian w.r.t Gyro Bias
    d_translation_d_bias_ += J_rot * J_bg_k;

    prev_delta_R_ik_ = delta_R_ik;
  }

  /**
   * @brief Gets the preintegrated translation delta.
   * @return The translation delta in the target frame at the start of the interval (i).
   */
  gtsam::Vector3 delta() const { return measured_translation_; }

  /**
   * @brief Gets the accumulated translation covariance.
   * @return The 3x3 covariance matrix.
   */
  gtsam::Matrix3 covariance() const { return covariance_; }

  /**
   * @brief Gets the first-order derivative of the preintegrated measurement w.r.t gyro bias.
   * @return The 3x3 Jacobian matrix.
   */
  gtsam::Matrix3 preintMeasDerivativeWrtBias() const { return d_translation_d_bias_; }

 private:
  gtsam::Vector3 measured_translation_;
  gtsam::Matrix3 covariance_;
  gtsam::Matrix3 d_translation_d_bias_;
  gtsam::Matrix3 cross_cov_rot_trans_;
  gtsam::Rot3 prev_delta_R_ik_;
};

}  // namespace coug_fgo::utils
