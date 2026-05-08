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
 * @file test_const_vel_factor.cpp
 * @brief Unit tests for const_vel_factor.hpp.
 * @author Nelson Durrant (w Gemini 3 Pro)
 * @date May 2026
 */

#include <gtest/gtest.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/inference/Symbol.h>

#include <functional>
#include <optional>

#include "coug_fgo/factors/const_vel_factor.hpp"

/**
 * @brief Verify Jacobians against numerical differentiation.
 */
TEST(ConstVelFactorTest, Jacobians) {
  gtsam::Key pose_key_i = gtsam::symbol_shorthand::X(1);
  gtsam::Key vel_key_i = gtsam::symbol_shorthand::V(1);
  gtsam::Key pose_key_j = gtsam::symbol_shorthand::X(2);
  gtsam::Key vel_key_j = gtsam::symbol_shorthand::V(2);
  gtsam::SharedNoiseModel model = gtsam::noiseModel::Isotropic::Sigma(3, 0.1);

  coug_fgo::factors::ConstVelFactor factor(pose_key_i, vel_key_i, pose_key_j, vel_key_j, model);

  gtsam::Pose3 pose_i(gtsam::Rot3::Ypr(0.1, 0.2, 0.3), gtsam::Point3(1.0, 2.0, 4.0));
  gtsam::Vector3 vel_i(1.0, 0.5, 0.0);
  gtsam::Pose3 pose_j(gtsam::Rot3::Ypr(0.4, -0.1, 0.2), gtsam::Point3(2.0, 3.0, 4.0));
  gtsam::Vector3 vel_j(1.1, 0.4, 0.1);

  auto evalFunc = [&](const gtsam::Pose3& pi, const gtsam::Vector3& vi, const gtsam::Pose3& pj,
                      const gtsam::Vector3& vj) {
    return factor.evaluateError(pi, vi, pj, vj, nullptr, nullptr, nullptr, nullptr);
  };

  gtsam::Matrix expectedH_pose_i =
      gtsam::numericalDerivative41<gtsam::Vector, gtsam::Pose3, gtsam::Vector3, gtsam::Pose3,
                                   gtsam::Vector3>(evalFunc, pose_i, vel_i, pose_j, vel_j, 1e-5);

  gtsam::Matrix expectedH_vel_i =
      gtsam::numericalDerivative42<gtsam::Vector, gtsam::Pose3, gtsam::Vector3, gtsam::Pose3,
                                   gtsam::Vector3>(evalFunc, pose_i, vel_i, pose_j, vel_j, 1e-5);

  gtsam::Matrix expectedH_pose_j =
      gtsam::numericalDerivative43<gtsam::Vector, gtsam::Pose3, gtsam::Vector3, gtsam::Pose3,
                                   gtsam::Vector3>(evalFunc, pose_i, vel_i, pose_j, vel_j, 1e-5);

  gtsam::Matrix expectedH_vel_j =
      gtsam::numericalDerivative44<gtsam::Vector, gtsam::Pose3, gtsam::Vector3, gtsam::Pose3,
                                   gtsam::Vector3>(evalFunc, pose_i, vel_i, pose_j, vel_j, 1e-5);

  gtsam::Matrix actualH_pose_i, actualH_vel_i, actualH_pose_j, actualH_vel_j;
  factor.evaluateError(pose_i, vel_i, pose_j, vel_j, &actualH_pose_i, &actualH_vel_i,
                       &actualH_pose_j, &actualH_vel_j);

  EXPECT_TRUE(gtsam::assert_equal(expectedH_pose_i, actualH_pose_i, 1e-5));
  EXPECT_TRUE(gtsam::assert_equal(expectedH_vel_i, actualH_vel_i, 1e-5));
  EXPECT_TRUE(gtsam::assert_equal(expectedH_pose_j, actualH_pose_j, 1e-5));
  EXPECT_TRUE(gtsam::assert_equal(expectedH_vel_j, actualH_vel_j, 1e-5));
}
