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
 * @file dvl_factor.hpp
 * @brief GTSAM factor for DVL velocity measurements with a lever arm.
 * @author Nelson Durrant
 * @date May 2026
 */

#pragma once

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace coug_fgo::factors {

/**
 * @class DvlFactorArm
 * @brief GTSAM factor for DVL velocity measurements with a lever arm.
 */
class DvlFactorArm
    : public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Vector3, gtsam::imuBias::ConstantBias> {
  gtsam::Vector3 measured_velocity_;
  gtsam::Vector3 measured_gyro_;
  gtsam::Rot3 target_R_sensor_;
  gtsam::Point3 target_p_sensor_;
  gtsam::Rot3 target_R_imu_;

 public:
  /**
   * @brief Constructor for DvlFactorArm.
   * @param pose_key GTSAM key for the starting AUV pose.
   * @param vel_key GTSAM key for the AUV map-frame velocity.
   * @param bias_key GTSAM key for the IMU bias.
   * @param target_T_sensor The static transformation from target to sensor.
   * @param target_T_imu The static transformation from target to IMU.
   * @param measured_velocity The velocity measurement in the sensor frame.
   * @param measured_gyro The raw gyro measurement (IMU frame) at the keyframe time.
   * @param noise_model The noise model for the measurement.
   */
  DvlFactorArm(gtsam::Key pose_key, gtsam::Key vel_key, gtsam::Key bias_key,
               const gtsam::Pose3& target_T_sensor, const gtsam::Pose3& target_T_imu,
               const gtsam::Vector3& measured_velocity, const gtsam::Vector3& measured_gyro,
               const gtsam::SharedNoiseModel& noise_model)
      : NoiseModelFactor3<gtsam::Pose3, gtsam::Vector3, gtsam::imuBias::ConstantBias>(
            noise_model, pose_key, vel_key, bias_key),
        measured_velocity_(measured_velocity),
        measured_gyro_(measured_gyro),
        target_R_sensor_(target_T_sensor.rotation()),
        target_p_sensor_(target_T_sensor.translation()),
        target_R_imu_(target_T_imu.rotation()) {}

  /**
   * @brief Evaluates the error and Jacobians for the factor.
   * @param pose The AUV pose estimate.
   * @param vel_map The AUV map-frame velocity estimate.
   * @param bias The IMU bias estimate.
   * @param H_pose Optional Jacobian matrix with respect to pose.
   * @param H_vel Optional Jacobian matrix with respect to velocity.
   * @param H_bias Optional Jacobian matrix with respect to bias.
   * @return The 3D error vector (predicted - measured).
   */
  gtsam::Vector evaluateError(const gtsam::Pose3& pose, const gtsam::Vector3& vel_map,
                              const gtsam::imuBias::ConstantBias& bias,
                              gtsam::OptionalMatrixType H_pose = nullptr,
                              gtsam::OptionalMatrixType H_vel = nullptr,
                              gtsam::OptionalMatrixType H_bias = nullptr) const override {
    gtsam::Matrix33 H_unrotate_R = gtsam::Matrix33::Zero();
    gtsam::Matrix33 H_unrotate_v = gtsam::Matrix33::Zero();

    gtsam::Vector3 vel_target = pose.rotation().unrotate(vel_map, H_pose ? &H_unrotate_R : nullptr,
                                                         H_vel ? &H_unrotate_v : nullptr);

    gtsam::Vector3 omega_target = target_R_imu_.rotate(measured_gyro_ - bias.gyroscope());
    gtsam::Vector3 lever_arm_vel = omega_target.cross(target_p_sensor_);

    gtsam::Vector3 predicted_velocity = target_R_sensor_.unrotate(vel_target + lever_arm_vel);

    // 3D velocity residual
    gtsam::Vector3 error = predicted_velocity - measured_velocity_;

    if (H_pose) {
      // Jacobian with respect to pose (3x6)
      H_pose->setZero(3, 6);
      H_pose->block<3, 3>(0, 0) = target_R_sensor_.transpose() * H_unrotate_R;
    }

    if (H_vel) {
      // Jacobian with respect to velocity (3x3)
      *H_vel = target_R_sensor_.transpose() * H_unrotate_v;
    }

    if (H_bias) {
      // Jacobian with respect to bias (3x6)
      H_bias->setZero(3, 6);
      H_bias->block<3, 3>(0, 3) = target_R_sensor_.transpose() *
                                  gtsam::skewSymmetric(target_p_sensor_) * target_R_imu_.matrix();
    }

    return error;
  }
};

}  // namespace coug_fgo::factors
