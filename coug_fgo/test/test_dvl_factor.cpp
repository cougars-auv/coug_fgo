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
 * @date Jan 2026
 */

#include <gtest/gtest.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/inference/Symbol.h>

#include <functional>
#include <optional>

#include "coug_fgo/factors/dvl_factor.hpp"

/**
 * @brief Verify Jacobians against numerical differentiation.
 */
TEST(DvlFactorArmTest, Jacobians) {
  gtsam::Key poseKey = gtsam::symbol_shorthand::X(1);
  gtsam::Key velKey = gtsam::symbol_shorthand::V(1);
  gtsam::SharedNoiseModel model = gtsam::noiseModel::Isotropic::Sigma(3, 0.1);
  gtsam::Pose3 target_P_sensor(gtsam::Rot3::Ypr(0.1, -0.1, 0.1), gtsam::Point3(0.5, 0.5, 0.5));
  gtsam::Vector3 measured_vel(1.0, 0.5, -0.2);

  coug_fgo::factors::DvlFactorArm factor(poseKey, velKey, target_P_sensor, measured_vel, model);

  gtsam::Pose3 pose(gtsam::Rot3::Ypr(0.1, 0.2, 0.3), gtsam::Point3(1.0, 2.0, 4.0));
  gtsam::Vector3 vel_world(1.5, -0.5, 0.2);

  gtsam::Matrix expectedH1 =
      gtsam::numericalDerivative21<gtsam::Vector, gtsam::Pose3, gtsam::Vector3>(
          [&](const gtsam::Pose3& p, const gtsam::Vector3& v) {
            return factor.evaluateError(p, v, nullptr, nullptr);
          },
          pose, vel_world, 1e-5);
  gtsam::Matrix expectedH2 =
      gtsam::numericalDerivative22<gtsam::Vector, gtsam::Pose3, gtsam::Vector3>(
          [&](const gtsam::Pose3& p, const gtsam::Vector3& v) {
            return factor.evaluateError(p, v, nullptr, nullptr);
          },
          pose, vel_world, 1e-5);

  gtsam::Matrix actualH1, actualH2;
  factor.evaluateError(pose, vel_world, &actualH1, &actualH2);

  EXPECT_TRUE(gtsam::assert_equal(expectedH1, actualH1, 1e-5));
  EXPECT_TRUE(gtsam::assert_equal(expectedH2, actualH2, 1e-5));
}
