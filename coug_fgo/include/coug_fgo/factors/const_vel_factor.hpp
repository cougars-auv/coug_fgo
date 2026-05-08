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
 * @file const_vel_factor.hpp
 * @brief GTSAM factor for constant target-frame velocity.
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
 * @class ConstVelFactor
 * @brief GTSAM factor for constant body-frame velocity.
 */
class ConstVelFactor
    : public gtsam::NoiseModelFactor4<gtsam::Pose3, gtsam::Vector3, gtsam::Pose3, gtsam::Vector3> {
 public:
  /**
   * @brief Constructor for ConstVelFactor.
   * @param pose_key_i GTSAM key for the starting AUV pose.
   * @param vel_key_i GTSAM key for the starting AUV velocity.
   * @param pose_key_j GTSAM key for the ending AUV pose.
   * @param vel_key_j GTSAM key for the ending AUV velocity.
   * @param noise_model The noise model for the constraint.
   */
  ConstVelFactor(gtsam::Key pose_key_i, gtsam::Key vel_key_i, gtsam::Key pose_key_j,
                 gtsam::Key vel_key_j, const gtsam::SharedNoiseModel& noise_model)
      : NoiseModelFactor4<gtsam::Pose3, gtsam::Vector3, gtsam::Pose3, gtsam::Vector3>(
            noise_model, pose_key_i, vel_key_i, pose_key_j, vel_key_j) {}

  /**
   * @brief Evaluates the error and Jacobians for the factor.
   * @param pose_i Starting AUV pose estimate.
   * @param vel_i Starting AUV map-frame velocity estimate.
   * @param pose_j Ending AUV pose estimate.
   * @param vel_j Ending AUV map-frame velocity estimate.
   * @param H_pose_i Optional Jacobian matrix with respect to pose_i.
   * @param H_vel_i Optional Jacobian matrix with respect to vel_i.
   * @param H_pose_j Optional Jacobian matrix with respect to pose_j.
   * @return The 3D error vector (measured - predicted).
   */
  gtsam::Vector evaluateError(const gtsam::Pose3& pose_i, const gtsam::Vector3& vel_i,
                              const gtsam::Pose3& pose_j, const gtsam::Vector3& vel_j,
                              gtsam::OptionalMatrixType H_pose_i = nullptr,
                              gtsam::OptionalMatrixType H_vel_i = nullptr,
                              gtsam::OptionalMatrixType H_pose_j = nullptr,
                              gtsam::OptionalMatrixType H_vel_j = nullptr) const override {
    gtsam::Matrix33 H_unrotate_Ri = gtsam::Matrix33::Zero();
    gtsam::Matrix33 H_unrotate_vi = gtsam::Matrix33::Zero();
    gtsam::Matrix33 H_unrotate_Rj = gtsam::Matrix33::Zero();
    gtsam::Matrix33 H_unrotate_vj = gtsam::Matrix33::Zero();
    gtsam::Vector3 v_target_i = pose_i.rotation().unrotate(
        vel_i, H_pose_i ? &H_unrotate_Ri : nullptr, H_vel_i ? &H_unrotate_vi : nullptr);
    gtsam::Vector3 v_target_j = pose_j.rotation().unrotate(
        vel_j, H_pose_j ? &H_unrotate_Rj : nullptr, H_vel_j ? &H_unrotate_vj : nullptr);

    // 3D velocity difference residual
    gtsam::Vector3 error = v_target_i - v_target_j;

    if (H_pose_i) {
      // Jacobian with respect to pose_i (3x6)
      H_pose_i->setZero(3, 6);
      H_pose_i->block<3, 3>(0, 0) = H_unrotate_Ri;
    }
    if (H_vel_i) {
      // Jacobian with respect to vel_i (3x3)
      *H_vel_i = H_unrotate_vi;
    }
    if (H_pose_j) {
      // Jacobian with respect to pose_j (3x6)
      H_pose_j->setZero(3, 6);
      H_pose_j->block<3, 3>(0, 0) = -H_unrotate_Rj;
    }
    if (H_vel_j) {
      // Jacobian with respect to vel_j (3x3)
      *H_vel_j = -H_unrotate_vj;
    }

    return error;
  }
};

}  // namespace coug_fgo::factors
