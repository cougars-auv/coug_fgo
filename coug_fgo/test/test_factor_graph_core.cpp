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
 * @file test_factor_graph_core.cpp
 * @brief Unit tests for factor_graph_core.hpp.
 * @author Nelson Durrant (w Gemini 3 Pro)
 * @date Jan 2026
 */

#include <gtest/gtest.h>

#include "coug_fgo/factor_graph_core.hpp"

using coug_fgo::FactorGraphCore;
using coug_fgo::utils::StateInitializer;
using coug_fgo::utils::QueueBundle;
using coug_fgo::utils::TfBundle;

/**
 * @class FactorGraphCoreTest
 * @brief Test fixture for FactorGraphCore tests.
 */
class FactorGraphCoreTest : public ::testing::Test
{
protected:
  factor_graph_node::Params params_;

  void SetUp() override
  {
    params_.gps.enable_gps = false;
    params_.gps.enable_gps_init_only = false;
    params_.mag.enable_mag = false;
    params_.mag.enable_mag_init_only = false;
    params_.ahrs.enable_ahrs = false;
    params_.ahrs.enable_ahrs_init_only = false;
    params_.prior.use_parameter_priors = true;
  }

  sensor_msgs::msg::Imu::SharedPtr createImuMsg(double t, double az = 9.81)
  {
    auto msg = std::make_shared<sensor_msgs::msg::Imu>();
    msg->header.stamp.sec = static_cast<int32_t>(t);
    msg->header.stamp.nanosec = static_cast<uint32_t>((t - std::floor(t)) * 1e9);
    msg->header.frame_id = "imu_link";
    msg->linear_acceleration.z = az;
    msg->orientation.w = 1.0;
    msg->linear_acceleration_covariance[0] = 0.1;
    msg->linear_acceleration_covariance[4] = 0.1;
    msg->linear_acceleration_covariance[8] = 0.1;
    msg->angular_velocity_covariance[0] = 0.01;
    msg->angular_velocity_covariance[4] = 0.01;
    msg->angular_velocity_covariance[8] = 0.01;
    msg->orientation_covariance[0] = 0.01;
    msg->orientation_covariance[4] = 0.01;
    msg->orientation_covariance[8] = 0.01;
    return msg;
  }

  nav_msgs::msg::Odometry::SharedPtr createOdomMsg(double t, double z = -5.0)
  {
    auto msg = std::make_shared<nav_msgs::msg::Odometry>();
    msg->header.stamp.sec = static_cast<int32_t>(t);
    msg->header.stamp.nanosec = static_cast<uint32_t>((t - std::floor(t)) * 1e9);
    msg->child_frame_id = "base_link";
    msg->pose.pose.position.z = z;
    msg->pose.pose.orientation.w = 1.0;
    msg->pose.covariance[0] = 1.0;
    msg->pose.covariance[7] = 1.0;
    msg->pose.covariance[14] = 1.0;
    return msg;
  }

  geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr createDvlMsg(
    double t, double vx = 1.0)
  {
    auto msg = std::make_shared<geometry_msgs::msg::TwistWithCovarianceStamped>();
    msg->header.stamp.sec = static_cast<int32_t>(t);
    msg->header.stamp.nanosec = static_cast<uint32_t>((t - std::floor(t)) * 1e9);
    msg->header.frame_id = "dvl_link";
    msg->twist.twist.linear.x = vx;
    msg->twist.covariance[0] = 0.01;
    msg->twist.covariance[7] = 0.01;
    msg->twist.covariance[14] = 0.01;
    return msg;
  }

  TfBundle createIdentityTfs()
  {
    return TfBundle{
      gtsam::Pose3(), gtsam::Pose3(), gtsam::Pose3(), gtsam::Pose3(),
      gtsam::Pose3(), gtsam::Pose3(), gtsam::Pose3(), gtsam::Pose3()};
  }

  std::unique_ptr<FactorGraphCore> initializeCore()
  {
    StateInitializer init(params_);
    QueueBundle q;
    q.imu.push_back(createImuMsg(1.0));
    q.depth.push_back(createOdomMsg(1.0));
    q.dvl.push_back(createDvlMsg(1.0));
    rclcpp::Time now(1, 0, RCL_ROS_TIME);
    init.update(now, q);

    TfBundle tfs = createIdentityTfs();
    init.compute(tfs);

    auto core = std::make_unique<FactorGraphCore>(params_);
    core->initialize(init, tfs);
    return core;
  }
};

/**
 * @brief Verify initialize sets the correct initial timestamp.
 */
TEST_F(FactorGraphCoreTest, Initialize) {
  auto core = initializeCore();
  EXPECT_EQ(core->prev_time().seconds(), 1.0);
}

/**
 * @brief Verify update with a stale timestamp returns nullopt.
 */
TEST_F(FactorGraphCoreTest, UpdateStaleTime) {
  auto core = initializeCore();

  QueueBundle msgs;
  msgs.imu.push_back(createImuMsg(0.5));
  msgs.depth.push_back(createOdomMsg(0.5));
  msgs.dvl.push_back(createDvlMsg(0.5));

  rclcpp::Time stale_time(0, 500000000, RCL_ROS_TIME);
  auto result = core->update(stale_time, msgs);
  EXPECT_FALSE(result.has_value());
}

/**
 * @brief Verify update advances prev_time and produces an UpdateResult.
 */
TEST_F(FactorGraphCoreTest, Update) {
  auto core = initializeCore();

  QueueBundle msgs;
  msgs.imu.push_back(createImuMsg(1.1));
  msgs.imu.push_back(createImuMsg(1.2));
  msgs.imu.push_back(createImuMsg(1.3));
  msgs.depth.push_back(createOdomMsg(1.3));
  msgs.dvl.push_back(createDvlMsg(1.3));

  rclcpp::Time target_time(1, 300000000, RCL_ROS_TIME);
  auto result = core->update(target_time, msgs);
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(core->prev_time().seconds(), 1.3);
}

/**
 * @brief Verify optimize returns a valid pose near the initial state.
 */
TEST_F(FactorGraphCoreTest, Optimize) {
  auto core = initializeCore();

  QueueBundle msgs;
  msgs.imu.push_back(createImuMsg(1.1));
  msgs.imu.push_back(createImuMsg(1.2));
  msgs.imu.push_back(createImuMsg(1.3));
  msgs.depth.push_back(createOdomMsg(1.3));
  msgs.dvl.push_back(createDvlMsg(1.3));

  rclcpp::Time target_time(1, 300000000, RCL_ROS_TIME);
  core->update(target_time, msgs);

  auto result = core->optimize();
  EXPECT_TRUE(result.has_value());

  // After one short step from identity, the pose should still be near identity
  EXPECT_NEAR(result->pose.translation().norm(), 0.0, 1.0);
  EXPECT_NEAR(result->velocity.norm(), 0.0, 2.0);
}

/**
 * @brief Verify optimize returns nullopt when no update has been buffered.
 */
TEST_F(FactorGraphCoreTest, OptimizeWithoutUpdate) {
  auto core = initializeCore();
  auto result = core->optimize();
  EXPECT_FALSE(result.has_value());
}
