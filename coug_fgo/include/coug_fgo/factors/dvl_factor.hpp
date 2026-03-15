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
 * @date Jan 2026
 */

#pragma once

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

using gtsam::symbol_shorthand::V;  // Velocity (x,y,z)
using gtsam::symbol_shorthand::X;  // Pose3 (x,y,z,r,p,y)

namespace coug_fgo::factors
{

/**
 * @class DvlFactorArm
 * @brief GTSAM factor for DVL velocity measurements with a lever arm.
 */
class DvlFactorArm : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Vector3>
{
  gtsam::Pose3 target_P_sensor_;
  gtsam::Vector3 sensor_v_measured_;

public:
  /**
   * @brief Constructor for DvlFactorArm.
   * @param pose_key GTSAM key for the starting AUV pose.
   * @param vel_key GTSAM key for the AUV world-frame velocity.
   * @param target_T_sensor The static transformation from target to sensor.
   * @param measured_velocity_sensor The velocity measurement in the sensor frame.
   * @param noise_model The noise model for the measurement.
   */
  DvlFactorArm(
    gtsam::Key pose_key, gtsam::Key vel_key,
    const gtsam::Pose3 & target_T_sensor,
    const gtsam::Vector3 & measured_velocity_sensor,
    const gtsam::SharedNoiseModel & noise_model)
  : NoiseModelFactor2<gtsam::Pose3, gtsam::Vector3>(noise_model, pose_key, vel_key),
    sensor_v_measured_(measured_velocity_sensor)
  {
    target_P_sensor_ = target_T_sensor;
  }

  /**
   * @brief Evaluates the error and Jacobians for the factor.
   * @param pose The AUV pose estimate.
   * @param vel_world The AUV world-frame velocity estimate.
   * @param H_pose Optional Jacobian matrix with respect to pose.
   * @param H_vel Optional Jacobian matrix with respect to velocity.
   * @return The 3D error vector (measured - predicted).
   */
  gtsam::Vector evaluateError(
    const gtsam::Pose3 & pose,
    const gtsam::Vector3 & vel_world,
    gtsam::OptionalMatrixType H_pose = nullptr,
    gtsam::OptionalMatrixType H_vel = nullptr) const override
  {
    // Predict the velocity measurement
    gtsam::Matrix33 H_unrotate_R = gtsam::Matrix33::Zero();
    gtsam::Matrix33 H_unrotate_v = gtsam::Matrix33::Zero();

    gtsam::Vector3 vel_target = pose.rotation().unrotate(
      vel_world, H_pose ? &H_unrotate_R : nullptr, H_vel ? &H_unrotate_v : nullptr);

    gtsam::Rot3 R_target_sensor = target_P_sensor_.rotation();
    gtsam::Vector3 predicted_vel_sensor = R_target_sensor.unrotate(vel_target);

    // 3D velocity residual
    gtsam::Vector3 error = predicted_vel_sensor - sensor_v_measured_;

    if (H_pose) {
      // Jacobian with respect to pose (3x6)
      H_pose->setZero(3, 6);
      H_pose->block<3, 3>(0, 0) = R_target_sensor.transpose() * H_unrotate_R;
    }

    if (H_vel) {
      // Jacobian with respect to velocity (3x3)
      *H_vel = R_target_sensor.transpose() * H_unrotate_v;
    }

    return error;
  }
};

}  // namespace coug_fgo::factors
