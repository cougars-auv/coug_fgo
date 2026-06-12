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
 * @file test_dvl_factor.cpp
 * @brief Unit tests for dvl_factor.hpp.
 * @author Nelson Durrant (w Gemini 3 Pro)
 * @date May 2026
 */

#include <gtest/gtest.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/ImuBias.h>

#include <functional>
#include <optional>

#include "coug_fgo/factors/dvl_factor.hpp"

/**
 * @brief Verify Jacobians against numerical differentiation.
 */
TEST(DvlFactorArmTest, Jacobians) {
  gtsam::Key poseKey = gtsam::symbol_shorthand::X(1);
  gtsam::Key velKey = gtsam::symbol_shorthand::V(1);
  gtsam::Key biasKey = gtsam::symbol_shorthand::B(1);
  gtsam::SharedNoiseModel model = gtsam::noiseModel::Isotropic::Sigma(3, 0.1);
  gtsam::Pose3 target_T_sensor(gtsam::Rot3::Ypr(0.1, -0.1, 0.1), gtsam::Point3(0.5, 0.5, 0.5));
  gtsam::Pose3 target_T_imu(gtsam::Rot3::Ypr(-0.2, 0.1, 0.3), gtsam::Point3(0.1, 0.2, 0.3));
  gtsam::Vector3 measured_vel(1.0, 0.5, -0.2);
  gtsam::Vector3 measured_gyro(0.1, -0.3, 0.2);

  coug_fgo::factors::DvlFactorArm factor(poseKey, velKey, biasKey, target_T_sensor, target_T_imu,
                                         measured_vel, measured_gyro, model);

  gtsam::Pose3 pose(gtsam::Rot3::Ypr(0.1, 0.2, 0.3), gtsam::Point3(1.0, 2.0, 4.0));
  gtsam::Vector3 vel_map(1.5, -0.5, 0.2);
  gtsam::imuBias::ConstantBias bias(gtsam::Vector3(0.01, -0.02, 0.03),
                                    gtsam::Vector3(0.02, -0.01, 0.01));

  auto evalFunc = [&](const gtsam::Pose3& p, const gtsam::Vector3& v,
                      const gtsam::imuBias::ConstantBias& b) {
    return factor.evaluateError(p, v, b, nullptr, nullptr, nullptr);
  };

  gtsam::Matrix expectedH1 =
      gtsam::numericalDerivative31<gtsam::Vector, gtsam::Pose3, gtsam::Vector3,
                                   gtsam::imuBias::ConstantBias>(evalFunc, pose, vel_map, bias,
                                                                 1e-5);
  gtsam::Matrix expectedH2 =
      gtsam::numericalDerivative32<gtsam::Vector, gtsam::Pose3, gtsam::Vector3,
                                   gtsam::imuBias::ConstantBias>(evalFunc, pose, vel_map, bias,
                                                                 1e-5);
  gtsam::Matrix expectedH3 =
      gtsam::numericalDerivative33<gtsam::Vector, gtsam::Pose3, gtsam::Vector3,
                                   gtsam::imuBias::ConstantBias>(evalFunc, pose, vel_map, bias,
                                                                 1e-5);

  gtsam::Matrix actualH1, actualH2, actualH3;
  factor.evaluateError(pose, vel_map, bias, &actualH1, &actualH2, &actualH3);

  EXPECT_TRUE(gtsam::assert_equal(expectedH1, actualH1, 1e-5));
  EXPECT_TRUE(gtsam::assert_equal(expectedH2, actualH2, 1e-5));
  EXPECT_TRUE(gtsam::assert_equal(expectedH3, actualH3, 1e-5));
}
