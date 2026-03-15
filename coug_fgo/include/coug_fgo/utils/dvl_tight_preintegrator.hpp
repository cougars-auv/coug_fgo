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
 * @brief Utility for preintegrating tightly-coupled DVL velocity measurements into relative translation.
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
 * @class DvlTightPreintegrator
 * @brief Utility for preintegrating tightly-coupled DVL velocity measurements into relative translation.
 */
class DvlTightPreintegrator
{
public:
  /**
   * @brief Constructor for DvlTightPreintegrator.
   */
  DvlTightPreintegrator() {reset();}

  /**
   * @brief Resets the preintegrator state.
   */
  void reset()
  {
    i_p_j_ = gtsam::Vector3::Zero();
    covariance_ = gtsam::Matrix3::Zero();
    dp_ij_dbias_ = gtsam::Matrix3::Zero();
  }

  /**
   * @brief Integrates a new DVL velocity measurement using the AQUA-SLAM formulation.
   * @param measured_vel The velocity measurement in the DVL sensor frame.
   * @param delta_R_ik Relative IMU rotation from start of interval (i) to current measurement (k).
   * @param imu_R_dvl Static extrinsic rotation from the DVL to the IMU frame.
   * @param dt The time delta since the last measurement.
   * @param measured_cov The measurement noise covariance of the DVL.
   * @param rot_cov_k Current rotation covariance from the IMU preintegrator at step k.
   * @param J_bg_k Jacobian of delta_R_ik w.r.t. the gyro bias (from IMU preintegrator).
   */
  void integrateMeasurement(
    const gtsam::Vector3 & measured_vel,
    const gtsam::Rot3 & delta_R_ik,
    const gtsam::Rot3 & imu_R_dvl,
    double dt,
    const gtsam::Matrix3 & measured_cov,
    const gtsam::Matrix3 & rot_cov_k,
    const gtsam::Matrix3 & J_bg_k)
  {
    // Calculate velocity in the IMU frame at time k
    gtsam::Vector3 v_Ik = imu_R_dvl.rotate(measured_vel);

    // Rotate into the IMU frame at the start of the interval (time i)
    gtsam::Vector3 p_i = delta_R_ik.rotate(v_Ik);
    i_p_j_ += p_i * dt;

    // Compute Jacobians for uncertainty and bias correction
    // J_vel maps DVL velocity noise to translation noise
    gtsam::Matrix3 J_vel = delta_R_ik.matrix() * imu_R_dvl.matrix() * dt;

    // J_rot maps Gyro rotation noise to translation noise
    gtsam::Matrix3 J_rot = -delta_R_ik.matrix() * gtsam::skewSymmetric(v_Ik) * dt;

    // Propagate combined DVL and Gyroscope uncertainty
    covariance_ += (J_vel * measured_cov * J_vel.transpose()) +
      (J_rot * rot_cov_k * J_rot.transpose());

    // Accumulate first-order Jacobian w.r.t Gyro Bias
    dp_ij_dbias_ += J_rot * J_bg_k;
  }

  /**
   * @brief Gets the preintegrated translation delta.
   * @return The translation delta in the IMU frame at the start of the interval (i).
   */
  gtsam::Vector3 delta() const {return i_p_j_;}

  /**
   * @brief Gets the accumulated translation covariance.
   * @return The 3x3 covariance matrix.
   */
  gtsam::Matrix3 covariance() const {return covariance_;}

  /**
   * @brief Gets the first-order derivative of the preintegrated measurement w.r.t gyro bias.
   * @return The 3x3 Jacobian matrix.
   */
  gtsam::Matrix3 preintMeasDerivativeWrtBias() const {return dp_ij_dbias_;}

private:
  gtsam::Vector3 i_p_j_;
  gtsam::Matrix3 covariance_;
  gtsam::Matrix3 dp_ij_dbias_;
};

}  // namespace coug_fgo::utils
