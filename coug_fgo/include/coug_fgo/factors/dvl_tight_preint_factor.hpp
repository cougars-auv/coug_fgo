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
 * @file dvl_tight_preint_factor.hpp
 * @brief GTSAM factor for tightly-coupled preintegrated DVL measurements with a lever arm.
 * @author Nelson Durrant
 * @date Jan 2026
 */

#pragma once

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

using gtsam::symbol_shorthand::X;  // Pose3 (x,y,z,r,p,y)
using gtsam::symbol_shorthand::B;  // ImuBias (ax,ay,az,gx,gy,gz)

namespace coug_fgo::factors
{

/**
 * @class DvlTightPreintFactorArm
 * @brief GTSAM factor for tightly-coupled preintegrated DVL translation measurements
 * with a lever arm.
 */
class DvlTightPreintFactorArm : public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3,
    gtsam::imuBias::ConstantBias>
{
  gtsam::Pose3 target_P_imu_;
  gtsam::Pose3 target_P_dvl_;
  gtsam::Pose3 imu_P_dvl_;
  gtsam::Vector3 i_p_j_measured_;
  gtsam::Matrix3 dp_ij_dbias_;
  gtsam::Vector3 gyro_bias_hat_;

public:
  /**
   * @brief Constructor for DvlTightPreintFactorArm.
   * @param pose_key_i GTSAM key for the starting AUV pose.
   * @param pose_key_j GTSAM key for the ending AUV pose.
   * @param bias_key_i GTSAM key for the IMU bias at the start of the interval.
   * @param target_T_imu The static transformation from target (Base) to IMU.
   * @param target_T_dvl The static transformation from target (Base) to DVL.
   * @param measured_translation The preintegrated translation measurement (in IMU frame).
   * @param dp_ij_dbias Jacobian mapping changes in gyro bias to changes in the measurement.
   * @param gyro_bias_hat The gyro bias estimate used during pre-integration.
   * @param noise_model The noise model for the measurement.
   */
  DvlTightPreintFactorArm(
    gtsam::Key pose_key_i, gtsam::Key pose_key_j, gtsam::Key bias_key_i,
    const gtsam::Pose3 & target_T_imu,
    const gtsam::Pose3 & target_T_dvl,
    const gtsam::Vector3 & measured_translation,
    const gtsam::Matrix3 & dp_ij_dbias,
    const gtsam::Vector3 & gyro_bias_hat,
    gtsam::SharedNoiseModel noise_model)
  : gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::imuBias::ConstantBias>(
      noise_model, pose_key_i, pose_key_j, bias_key_i),
    i_p_j_measured_(measured_translation),
    dp_ij_dbias_(dp_ij_dbias),
    gyro_bias_hat_(gyro_bias_hat)
  {
    target_P_imu_ = target_T_imu;
    target_P_dvl_ = target_T_dvl;
    imu_P_dvl_ = target_P_imu_.between(target_P_dvl_);
  }

  /**
   * @brief Evaluates the error and Jacobians for the factor.
   * @param pose_i Starting AUV pose estimate in the base frame.
   * @param pose_j Ending AUV pose estimate in the base frame.
   * @param bias_i Starting IMU bias estimate.
   * @param H_pose_i Optional Jacobian matrix with respect to pose_i.
   * @param H_pose_j Optional Jacobian matrix with respect to pose_j.
   * @param H_bias_i Optional Jacobian matrix with respect to bias_i.
   * @return The 3D error vector (predicted - measured).
   */
  gtsam::Vector evaluateError(
    const gtsam::Pose3 & pose_i,
    const gtsam::Pose3 & pose_j,
    const gtsam::imuBias::ConstantBias & bias_i,
    gtsam::OptionalMatrixType H_pose_i = nullptr,
    gtsam::OptionalMatrixType H_pose_j = nullptr,
    gtsam::OptionalMatrixType H_bias_i = nullptr) const override
  {
    gtsam::Vector3 gyro_bias_update = bias_i.gyroscope() - gyro_bias_hat_;
    gtsam::Vector3 corrected_i_p_j = i_p_j_measured_ + (dp_ij_dbias_ * gyro_bias_update);

    // Predict the translation measurement
    gtsam::Matrix66 H_posej_compose;
    gtsam::Pose3 pose_dvl_j = pose_j.compose(target_P_dvl_, H_pose_j ? &H_posej_compose : nullptr);

    gtsam::Matrix36 H_tj_posej = gtsam::Matrix36::Zero();
    gtsam::Point3 t_j = pose_dvl_j.translation(H_pose_j ? &H_tj_posej : nullptr);

    gtsam::Matrix66 H_posei_compose;
    gtsam::Pose3 pose_imu_i = pose_i.compose(target_P_imu_, H_pose_i ? &H_posei_compose : nullptr);

    gtsam::Matrix36 H_pij_posei = gtsam::Matrix36::Zero();
    gtsam::Matrix33 H_pij_tj = gtsam::Matrix33::Zero();
    gtsam::Point3 p_ij = pose_imu_i.transformTo(
      t_j,
      H_pose_i ? &H_pij_posei : nullptr,
      H_pose_j ? &H_pij_tj : nullptr);

    gtsam::Point3 p_ID = imu_P_dvl_.translation();
    gtsam::Vector3 pred_p_ij = p_ij - p_ID;

    // 3D translation residual
    gtsam::Vector3 error = pred_p_ij - corrected_i_p_j;

    if (H_pose_i) {
      // Jacobian with respect to starting pose (3x6)
      *H_pose_i = H_pij_posei * H_posei_compose;
    }

    if (H_pose_j) {
      // Jacobian with respect to ending pose (3x6)
      *H_pose_j = H_pij_tj * H_tj_posej * H_posej_compose;
    }

    if (H_bias_i) {
      // Jacobian with respect to the bias state (3x6)
      gtsam::Matrix36 H_bias = gtsam::Matrix36::Zero();
      H_bias.block<3, 3>(0, 3) = -dp_ij_dbias_;
      *H_bias_i = H_bias;
    }

    return error;
  }
};

}  // namespace coug_fgo::factors
