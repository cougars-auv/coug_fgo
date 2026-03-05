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
 * @file test_state_initializer.cpp
 * @brief Unit tests for state_initializer.hpp.
 * @author Nelson Durrant
 * @date Mar 2026
 */

#include <gtest/gtest.h>

#include "coug_fgo/utils/state_initializer.hpp"

using coug_fgo::utils::StateInitializer;
using coug_fgo::utils::QueueBundle;
using coug_fgo::utils::TfBundle;

/**
 * @class StateInitializerTest
 * @brief Test fixture for StateInitializer tests.
 */
class StateInitializerTest : public ::testing::Test
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
  }

  sensor_msgs::msg::Imu::SharedPtr createImuMsg(
    double t, double ax = 0.0, double ay = 0.0, double az = 9.81,
    double gx = 0.0, double gy = 0.0, double gz = 0.0)
  {
    auto msg = std::make_shared<sensor_msgs::msg::Imu>();
    msg->header.stamp.sec = static_cast<int32_t>(t);
    msg->header.stamp.nanosec = static_cast<uint32_t>((t - std::floor(t)) * 1e9);
    msg->header.frame_id = "imu_link";
    msg->linear_acceleration.x = ax;
    msg->linear_acceleration.y = ay;
    msg->linear_acceleration.z = az;
    msg->angular_velocity.x = gx;
    msg->angular_velocity.y = gy;
    msg->angular_velocity.z = gz;
    msg->orientation.w = 1.0;
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
    double t, double vx = 1.0, double vy = 0.0, double vz = 0.0)
  {
    auto msg = std::make_shared<geometry_msgs::msg::TwistWithCovarianceStamped>();
    msg->header.stamp.sec = static_cast<int32_t>(t);
    msg->header.stamp.nanosec = static_cast<uint32_t>((t - std::floor(t)) * 1e9);
    msg->header.frame_id = "dvl_link";
    msg->twist.twist.linear.x = vx;
    msg->twist.twist.linear.y = vy;
    msg->twist.twist.linear.z = vz;
    msg->twist.covariance[0] = 0.01;
    msg->twist.covariance[7] = 0.01;
    msg->twist.covariance[14] = 0.01;
    return msg;
  }

  QueueBundle createMinimalQueues(double t)
  {
    QueueBundle q;
    q.imu.push_back(createImuMsg(t));
    q.depth.push_back(createOdomMsg(t));
    q.dvl.push_back(createDvlMsg(t));
    return q;
  }

  TfBundle createIdentityTfs()
  {
    return TfBundle{
      gtsam::Pose3(), gtsam::Pose3(), gtsam::Pose3(), gtsam::Pose3(),
      gtsam::Pose3(), gtsam::Pose3(), gtsam::Pose3(), gtsam::Pose3()};
  }
};

/**
 * @brief Verify parameter-based update stores the latest messages and returns true.
 */
TEST_F(StateInitializerTest, UpdateWithParameterPriors) {
  params_.prior.use_parameter_priors = true;
  StateInitializer init(params_);

  QueueBundle q = createMinimalQueues(1.0);
  rclcpp::Time now(1, 0, RCL_ROS_TIME);

  EXPECT_TRUE(init.update(now, q));
  EXPECT_NE(init.initial_imu_, nullptr);
  EXPECT_NE(init.initial_depth_, nullptr);
  EXPECT_NE(init.initial_dvl_, nullptr);
}

/**
 * @brief Verify averaging-based update returns false until the duration elapses.
 */
TEST_F(StateInitializerTest, UpdateWithAveraging) {
  params_.prior.use_parameter_priors = false;
  params_.prior.initialization_duration = 1.0;
  StateInitializer init(params_);

  QueueBundle q1 = createMinimalQueues(1.0);
  rclcpp::Time t1(1, 0, RCL_ROS_TIME);
  EXPECT_FALSE(init.update(t1, q1));

  QueueBundle q2 = createMinimalQueues(2.5);
  rclcpp::Time t2(2, 500000000, RCL_ROS_TIME);
  EXPECT_TRUE(init.update(t2, q2));
}

/**
 * @brief Verify averaging produces the mean of multiple IMU samples.
 */
TEST_F(StateInitializerTest, IncrementAverages) {
  params_.prior.use_parameter_priors = false;
  params_.prior.initialization_duration = 0.0;
  StateInitializer init(params_);

  QueueBundle q;
  q.imu.push_back(createImuMsg(1.0, 0.0, 0.0, 9.0));
  q.imu.push_back(createImuMsg(1.1, 0.0, 0.0, 11.0));
  q.depth.push_back(createOdomMsg(1.0, -3.0));
  q.depth.push_back(createOdomMsg(1.1, -7.0));
  q.dvl.push_back(createDvlMsg(1.0, 2.0));
  q.dvl.push_back(createDvlMsg(1.1, 4.0));

  rclcpp::Time now(1, 0, RCL_ROS_TIME);
  init.update(now, q);

  EXPECT_NEAR(init.initial_imu_->linear_acceleration.z, 10.0, 1e-9);
  EXPECT_NEAR(init.initial_depth_->pose.pose.position.z, -5.0, 1e-9);
  EXPECT_NEAR(init.initial_dvl_->twist.twist.linear.x, 3.0, 1e-9);
}

/**
 * @brief Verify compute produces correct initial state with parameter priors.
 */
TEST_F(StateInitializerTest, Compute) {
  params_.prior.use_parameter_priors = true;
  StateInitializer init(params_);

  QueueBundle q = createMinimalQueues(1.0);
  rclcpp::Time now(1, 0, RCL_ROS_TIME);
  init.update(now, q);

  TfBundle tfs = createIdentityTfs();
  init.compute(tfs);

  // Default parameter priors: orientation=[0,0,0], position=[0,0,0], velocity=[0,0,0]
  // With identity TFs, pose should be identity rotation at origin
  EXPECT_TRUE(gtsam::assert_equal(gtsam::Rot3::Identity(), init.pose_.rotation(), 1e-9));
  EXPECT_TRUE(gtsam::assert_equal(gtsam::Point3(gtsam::Point3::Zero()), gtsam::Point3(init.pose_.translation()), 1e-9));
  EXPECT_TRUE(gtsam::assert_equal(gtsam::Vector3(gtsam::Vector3::Zero()), gtsam::Vector3(init.velocity_), 1e-9));
  EXPECT_NEAR(init.bias_.accelerometer().norm(), 0.0, 1e-9);
  EXPECT_NEAR(init.bias_.gyroscope().norm(), 0.0, 1e-9);
  EXPECT_EQ(init.time_.seconds(), 1.0);
}

/**
 * @brief Verify gyro bias is set from averaged angular velocity in sensor mode.
 */
TEST_F(StateInitializerTest, ComputeBiasFromGyro) {
  params_.prior.use_parameter_priors = false;
  params_.prior.initialization_duration = 0.0;
  StateInitializer init(params_);

  QueueBundle q;
  q.imu.push_back(createImuMsg(1.0, 0.0, 0.0, 9.81, 0.01, -0.02, 0.03));
  q.depth.push_back(createOdomMsg(1.0));
  q.dvl.push_back(createDvlMsg(1.0));

  rclcpp::Time now(1, 0, RCL_ROS_TIME);
  init.update(now, q);

  TfBundle tfs = createIdentityTfs();
  init.compute(tfs);

  EXPECT_NEAR(init.bias_.gyroscope().x(), 0.01, 1e-9);
  EXPECT_NEAR(init.bias_.gyroscope().y(), -0.02, 1e-9);
  EXPECT_NEAR(init.bias_.gyroscope().z(), 0.03, 1e-9);
}
