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
 * @file test_dvl_loose_preint_factor.hpp
 * @brief Unit tests for dvl_loose_preint_factor.hpp.
 * @author Nelson Durrant (w Gemini 3 Pro)
 * @date Jan 2026
 */

#include <gtest/gtest.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/inference/Symbol.h>

#include <functional>
#include <optional>

#include "coug_fgo/factors/dvl_loose_preint_factor.hpp"

/**
 * @brief Verify error evaluation logic and lever arm correction.
 */
TEST(DvlLoosePreintFactorArmTest, ErrorEvaluation) {
  gtsam::Key poseIKey = gtsam::symbol_shorthand::X(1);
  gtsam::Key poseJKey = gtsam::symbol_shorthand::X(2);
  gtsam::Vector3 measured_translation(1.0, 0.0, 0.0);
  gtsam::SharedNoiseModel model = gtsam::noiseModel::Isotropic::Sigma(3, 0.1);
  gtsam::Pose3 target_P_sensor = gtsam::Pose3::Identity();
  coug_fgo::factors::DvlLoosePreintFactorArm factor(
    poseIKey, poseJKey, target_P_sensor, measured_translation, model);

  // Non-zero error when no translation occurred but measurement is non-zero
  EXPECT_TRUE(
    gtsam::assert_equal(
      gtsam::Vector3(-1, 0, 0),
      factor.evaluateError(gtsam::Pose3::Identity(), gtsam::Pose3::Identity()), 1e-9));

  // Zero error when body-frame translation matches the measurement
  gtsam::Pose3 pose_i = gtsam::Pose3::Identity();
  gtsam::Pose3 pose_j = gtsam::Pose3(gtsam::Rot3::Identity(), gtsam::Point3(1, 0, 0));
  EXPECT_TRUE(
    gtsam::assert_equal(
      gtsam::Vector3::Zero(),
      factor.evaluateError(pose_i, pose_j), 1e-9));

  // Rotation is accounted for when computing relative displacement
  pose_i = gtsam::Pose3(gtsam::Rot3::Yaw(M_PI_2), gtsam::Point3(0, 0, 0));
  pose_j = gtsam::Pose3(gtsam::Rot3::Identity(), gtsam::Point3(0, 1, 0));
  EXPECT_TRUE(
    gtsam::assert_equal(
      gtsam::Vector3::Zero(),
      factor.evaluateError(pose_i, pose_j), 1e-9));

  // Sensor mounting transform is compensated in displacement prediction
  gtsam::Pose3 target_P_sensor_arm(gtsam::Rot3::Yaw(M_PI_2), gtsam::Point3(0, 0, 1));
  coug_fgo::factors::DvlLoosePreintFactorArm factor_arm(
    poseIKey, poseJKey, target_P_sensor_arm, gtsam::Vector3(0, -1, 0), model);

  pose_i = gtsam::Pose3::Identity();
  pose_j = gtsam::Pose3(gtsam::Rot3::Identity(), gtsam::Point3(1, 0, 0));
  EXPECT_TRUE(
    gtsam::assert_equal(
      gtsam::Vector3::Zero(),
      factor_arm.evaluateError(pose_i, pose_j), 1e-9));

  // Repeated baseline check confirms consistent residual
  EXPECT_TRUE(
    gtsam::assert_equal(
      gtsam::Vector3(-1, 0, 0),
      factor.evaluateError(gtsam::Pose3::Identity(), gtsam::Pose3::Identity()), 1e-9));
}

/**
 * @brief Verify Jacobians against numerical differentiation.
 */
TEST(DvlLoosePreintFactorArmTest, Jacobians) {
  gtsam::Key poseIKey = gtsam::symbol_shorthand::X(1);
  gtsam::Key poseJKey = gtsam::symbol_shorthand::X(2);
  gtsam::Vector3 measured_translation(1.0, 0.5, -0.2);
  gtsam::SharedNoiseModel model = gtsam::noiseModel::Isotropic::Sigma(3, 0.1);
  gtsam::Pose3 target_P_sensor_2(gtsam::Rot3::Ypr(0.1, 0, 0), gtsam::Point3(0.5, 0.5, 0.5));
  coug_fgo::factors::DvlLoosePreintFactorArm factor(
    poseIKey, poseJKey, target_P_sensor_2, measured_translation, model);

  gtsam::Pose3 pose_i = gtsam::Pose3(gtsam::Rot3::Ypr(0.1, 0.2, 0.3), gtsam::Point3(1, 2, 3));
  gtsam::Pose3 pose_j = gtsam::Pose3(gtsam::Rot3::Ypr(-0.2, 0.4, 0.1), gtsam::Point3(2, 3, 2.5));

  gtsam::Matrix expectedH1 = gtsam::numericalDerivative21<gtsam::Vector, gtsam::Pose3,
      gtsam::Pose3>(
    [&](const gtsam::Pose3 & pi, const gtsam::Pose3 & pj) {
      return factor.evaluateError(pi, pj, nullptr, nullptr);
    },
    pose_i, pose_j, 1e-5);
  gtsam::Matrix expectedH2 = gtsam::numericalDerivative22<gtsam::Vector, gtsam::Pose3,
      gtsam::Pose3>(
    [&](const gtsam::Pose3 & pi, const gtsam::Pose3 & pj) {
      return factor.evaluateError(pi, pj, nullptr, nullptr);
    },
    pose_i, pose_j, 1e-5);

  gtsam::Matrix actualH1, actualH2;
  factor.evaluateError(pose_i, pose_j, &actualH1, &actualH2);
  EXPECT_TRUE(gtsam::assert_equal(expectedH1, actualH1, 1e-5));
  EXPECT_TRUE(gtsam::assert_equal(expectedH2, actualH2, 1e-5));
}
