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
 * @file ahrs_factor.hpp
 * @brief GTSAM factor for AHRS attitude measurements with a lever arm.
 * @author Nelson Durrant
 * @date May 2026
 */

#pragma once

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace coug_fgo::factors {

/**
 * @class AhrsFactorArm
 * @brief GTSAM factor for AHRS attitude measurements with a lever arm.
 */
class AhrsFactorArm : public gtsam::NoiseModelFactor1<gtsam::Pose3> {
  gtsam::Rot3 measured_orientation_;
  gtsam::Rot3 target_R_sensor_;

 public:
  /**
   * @brief Constructor for AhrsFactorArm.
   * @param pose_key GTSAM key for the AUV pose.
   * @param measured_orientation The measured orientation of the sensor in the map frame.
   * @param target_T_sensor The static transformation from target to sensor.
   * @param mag_declination Magnetic declination to add to the measurement [rad].
   * @param noise_model The noise model for the measurement.
   */
  AhrsFactorArm(gtsam::Key pose_key, const gtsam::Rot3& measured_orientation,
                const gtsam::Pose3& target_T_sensor, double mag_declination,
                const gtsam::SharedNoiseModel& noise_model)
      : NoiseModelFactor1<gtsam::Pose3>(noise_model, pose_key),
        measured_orientation_(gtsam::Rot3::Yaw(mag_declination) * measured_orientation),
        target_R_sensor_(target_T_sensor.rotation()) {}

  /**
   * @brief Evaluates the error and Jacobians for the factor.
   * @param pose The AUV pose estimate.
   * @param H Optional Jacobian matrix.
   * @return The 3D error vector (predicted - measured).
   */
  gtsam::Vector evaluateError(const gtsam::Pose3& pose,
                              gtsam::OptionalMatrixType H = nullptr) const override {
    gtsam::Matrix33 H_compose = gtsam::Matrix33::Zero();
    gtsam::Rot3 predicted_orientation =
        pose.rotation().compose(target_R_sensor_, H ? &H_compose : nullptr);

    // 3D orientation residual (Lie algebra)
    gtsam::Matrix33 H_between = gtsam::Matrix33::Zero();
    gtsam::Rot3 R_err =
        measured_orientation_.between(predicted_orientation, nullptr, H ? &H_between : nullptr);

    gtsam::Matrix33 H_logmap = gtsam::Matrix33::Zero();
    gtsam::Vector3 error = gtsam::Rot3::Logmap(R_err, H ? &H_logmap : nullptr);

    if (H) {
      // Jacobian with respect to pose (3x6)
      H->setZero(3, 6);
      H->block<3, 3>(0, 0) = H_logmap * H_between * H_compose;
    }

    return error;
  }
};

}  // namespace coug_fgo::factors
