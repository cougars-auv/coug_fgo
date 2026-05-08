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
 * @file test_gps_factor.cpp
 * @brief Unit tests for gps_factor.hpp.
 * @author Nelson Durrant (w Gemini 3 Pro)
 * @date May 2026
 */

#include <gtest/gtest.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/inference/Symbol.h>

#include <functional>
#include <optional>

#include "coug_fgo/factors/gps_factor.hpp"

/**
 * @brief Verify Jacobians against numerical differentiation.
 */
TEST(Gps2dFactorArmTest, Jacobians) {
  gtsam::Key poseKey = gtsam::symbol_shorthand::X(1);
  gtsam::SharedNoiseModel model = gtsam::noiseModel::Isotropic::Sigma(2, 0.1);
  gtsam::Pose3 target_T_sensor(gtsam::Rot3::Ypr(0.1, -0.1, 0.1), gtsam::Point3(0.5, 0.5, 0.5));
  gtsam::Point3 measured_point(5.0, 5.0, 5.0);

  coug_fgo::factors::Gps2dFactorArm factor(poseKey, measured_point, target_T_sensor, model);

  gtsam::Pose3 pose(gtsam::Rot3::Ypr(0.1, 0.2, 0.3), gtsam::Point3(1.0, 2.0, 4.0));

  gtsam::Matrix expectedH = gtsam::numericalDerivative11<gtsam::Vector, gtsam::Pose3>(
      [&](const gtsam::Pose3& p) { return factor.evaluateError(p, nullptr); }, pose, 1e-5);

  gtsam::Matrix actualH;
  factor.evaluateError(pose, &actualH);

  EXPECT_TRUE(gtsam::assert_equal(expectedH, actualH, 1e-5));
  EXPECT_EQ(actualH.rows(), 2);
  EXPECT_EQ(actualH.cols(), 6);
}
