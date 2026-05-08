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
 * @file test_dvl_tight_preint_factor.cpp
 * @brief Unit tests for dvl_tight_preint_factor.hpp.
 * @author Nelson Durrant (w Antigravity)
 * @date May 2026
 */

#include <gtest/gtest.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/ImuBias.h>

#include <functional>
#include <optional>

#include "coug_fgo/factors/dvl_tight_preint_factor.hpp"

/**
 * @brief Verify Jacobians against numerical differentiation.
 */
TEST(DvlTightPreintFactorArmTest, Jacobians) {
  gtsam::Key pose_key_i = gtsam::symbol_shorthand::X(1);
  gtsam::Key pose_key_j = gtsam::symbol_shorthand::X(2);
  gtsam::Key bias_key_i = gtsam::symbol_shorthand::B(1);
  gtsam::SharedNoiseModel model = gtsam::noiseModel::Isotropic::Sigma(3, 0.1);
  gtsam::Pose3 target_T_imu(gtsam::Rot3::Ypr(0.1, -0.1, 0.1), gtsam::Point3(0.1, 0.2, 0.3));
  gtsam::Pose3 target_T_dvl(gtsam::Rot3::Ypr(-0.1, 0.1, -0.1), gtsam::Point3(0.5, 0.5, 0.5));
  gtsam::Vector3 measured_translation(1.0, 0.5, -0.2);
  gtsam::Matrix3 d_translation_d_bias = gtsam::Matrix3::Identity() * 0.01;
  gtsam::Vector3 gyro_bias_hat(0.01, -0.02, 0.005);

  coug_fgo::factors::DvlTightPreintFactorArm factor(
      pose_key_i, pose_key_j, bias_key_i, target_T_imu, target_T_dvl, measured_translation,
      d_translation_d_bias, gyro_bias_hat, model);

  gtsam::Pose3 pose_i(gtsam::Rot3::Ypr(0.1, 0.2, 0.3), gtsam::Point3(1.0, 2.0, 4.0));
  gtsam::Pose3 pose_j(gtsam::Rot3::Ypr(-0.2, 0.4, 0.1), gtsam::Point3(2.0, 3.0, 2.5));
  gtsam::imuBias::ConstantBias bias_i(gtsam::Vector3(0.0, 0.0, 0.0),
                                      gtsam::Vector3(0.02, -0.01, 0.01));

  auto evalFunc = [&](const gtsam::Pose3& pi, const gtsam::Pose3& pj,
                      const gtsam::imuBias::ConstantBias& bi) {
    return factor.evaluateError(pi, pj, bi, nullptr, nullptr, nullptr);
  };

  gtsam::Matrix expectedH_pose_i =
      gtsam::numericalDerivative31<gtsam::Vector, gtsam::Pose3, gtsam::Pose3,
                                   gtsam::imuBias::ConstantBias>(evalFunc, pose_i, pose_j, bias_i,
                                                                 1e-5);
  gtsam::Matrix expectedH_pose_j =
      gtsam::numericalDerivative32<gtsam::Vector, gtsam::Pose3, gtsam::Pose3,
                                   gtsam::imuBias::ConstantBias>(evalFunc, pose_i, pose_j, bias_i,
                                                                 1e-5);
  gtsam::Matrix expectedH_bias_i =
      gtsam::numericalDerivative33<gtsam::Vector, gtsam::Pose3, gtsam::Pose3,
                                   gtsam::imuBias::ConstantBias>(evalFunc, pose_i, pose_j, bias_i,
                                                                 1e-5);

  gtsam::Matrix actualH_pose_i, actualH_pose_j, actualH_bias_i;
  factor.evaluateError(pose_i, pose_j, bias_i, &actualH_pose_i, &actualH_pose_j, &actualH_bias_i);

  EXPECT_TRUE(gtsam::assert_equal(expectedH_pose_i, actualH_pose_i, 1e-5));
  EXPECT_TRUE(gtsam::assert_equal(expectedH_pose_j, actualH_pose_j, 1e-5));
  EXPECT_TRUE(gtsam::assert_equal(expectedH_bias_i, actualH_bias_i, 1e-5));
}
