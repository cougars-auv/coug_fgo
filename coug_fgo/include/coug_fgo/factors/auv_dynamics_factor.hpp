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
 * @file auv_dynamics_factor.hpp
 * @brief GTSAM factor for a simplified version of Fossen's equations with a lever arm.
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
 * @class AuvDynamicsFactorArm
 * @brief GTSAM factor for a simplified version of Fossen's equations with a lever arm.
 */
class AuvDynamicsFactorArm
    : public gtsam::NoiseModelFactor4<gtsam::Pose3, gtsam::Vector3, gtsam::Pose3, gtsam::Vector3> {
 private:
  double dt_;
  gtsam::Vector3 target_f_;
  gtsam::Matrix33 mass_;
  gtsam::Matrix33 linear_drag_;
  gtsam::Matrix33 quad_drag_;
  gtsam::Matrix33 mass_inv_;

 public:
  /**
   * @brief Constructor for AuvDynamicsFactorArm.
   * @param pose_key_i GTSAM key for the starting AUV pose.
   * @param vel_key_i GTSAM key for the starting AUV velocity.
   * @param pose_key_j GTSAM key for the ending AUV pose.
   * @param vel_key_j GTSAM key for the ending AUV velocity.
   * @param dt The time interval between the two states.
   * @param control_force The sensor-frame force vector from thrusters.
   * @param target_T_sensor The static transformation from target to sensor.
   * @param mass Combined mass (Rigid body + Added mass).
   * @param linear_drag Linear damping coefficient.
   * @param quad_drag Quadratic damping coefficient.
   * @param noise_model The noise model for the constraint.
   */
  AuvDynamicsFactorArm(gtsam::Key pose_key_i, gtsam::Key vel_key_i, gtsam::Key pose_key_j,
                       gtsam::Key vel_key_j, double dt, const gtsam::Vector3& control_force,
                       const gtsam::Pose3& target_T_sensor, const gtsam::Matrix33& mass,
                       const gtsam::Matrix33& linear_drag, const gtsam::Matrix33& quad_drag,
                       const gtsam::SharedNoiseModel& noise_model)
      : NoiseModelFactor4<gtsam::Pose3, gtsam::Vector3, gtsam::Pose3, gtsam::Vector3>(
            noise_model, pose_key_i, vel_key_i, pose_key_j, vel_key_j),
        dt_(dt),
        target_f_(target_T_sensor.rotation().rotate(control_force)),
        mass_(mass),
        linear_drag_(linear_drag),
        quad_drag_(quad_drag),
        mass_inv_(mass.inverse()) {}

  /**
   * @brief Evaluates the error and Jacobians for the factor.
   * @param pose_i Starting AUV pose estimate.
   * @param vel_i Starting AUV map-frame velocity estimate.
   * @param pose_j Ending AUV pose estimate.
   * @param vel_j Ending AUV map-frame velocity estimate.
   * @param H_pose_i Optional Jacobian matrix with respect to pose_i.
   * @param H_vel_i Optional Jacobian matrix with respect to vel_i.
   * @param H_pose_j Optional Jacobian matrix with respect to pose_j.
   * @param H_vel_j Optional Jacobian matrix with respect to vel_j.
   * @return The 3D error vector (predicted - measured).
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

    gtsam::Vector3 abs_v_target_i = v_target_i.cwiseAbs();
    gtsam::Matrix33 J_drag_v = gtsam::Matrix33::Zero();
    gtsam::Vector3 drag_force =
        -(linear_drag_ * v_target_i + quad_drag_ * abs_v_target_i.asDiagonal() * v_target_i);

    if (H_pose_i || H_vel_i) {
      J_drag_v = -(linear_drag_ + 2.0 * quad_drag_ * abs_v_target_i.asDiagonal());
    }

    gtsam::Vector3 accel_target = mass_inv_ * (target_f_ + drag_force);
    gtsam::Vector3 v_target_pred = v_target_i + accel_target * dt_;

    // 3D velocity residual
    gtsam::Vector3 error = v_target_j - v_target_pred;

    if (H_pose_i) {
      // Jacobian with respect to pose_i (3x6)
      gtsam::Matrix33 J_scale = gtsam::Matrix33::Identity() + dt_ * mass_inv_ * J_drag_v;

      H_pose_i->setZero(3, 6);
      H_pose_i->block<3, 3>(0, 0) = -J_scale * H_unrotate_Ri;
    }

    if (H_vel_i) {
      // Jacobian with respect to vel_i (3x3)
      gtsam::Matrix33 J_scale = gtsam::Matrix33::Identity() + dt_ * mass_inv_ * J_drag_v;

      *H_vel_i = -J_scale * H_unrotate_vi;
    }

    if (H_pose_j) {
      // Jacobian with respect to pose_j (3x6)
      H_pose_j->setZero(3, 6);
      H_pose_j->block<3, 3>(0, 0) = H_unrotate_Rj;
    }

    if (H_vel_j) {
      // Jacobian with respect to vel_j (3x3)
      *H_vel_j = H_unrotate_vj;
    }

    return error;
  }
};

}  // namespace coug_fgo::factors
