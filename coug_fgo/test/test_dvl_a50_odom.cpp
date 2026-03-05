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
 * @file test_dvl_a50_odom.cpp
 * @brief Unit tests for dvl_a50_odom.hpp.
 * @author Nelson Durrant (w Gemini 3 Pro)
 * @date Mar 2026
 */

#include <gtest/gtest.h>
#include <tf2_ros/static_transform_broadcaster.h>

#include <dvl_msgs/msg/dvldr.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>

#include "coug_fgo/dvl_a50_odom.hpp"

using coug_fgo::DvlA50OdomNode;

/**
 * @class TestDvlA50OdomNode
 * @brief Harness to expose protected DvlA50OdomNode members for testing.
 */
class TestDvlA50OdomNode : public DvlA50OdomNode
{
public:
  using DvlA50OdomNode::DvlA50OdomNode;

  // Expose protected methods for direct unit testing
  using DvlA50OdomNode::dvlCallback;

  // Expose protected state for assertion checking
  using DvlA50OdomNode::params_;
  using DvlA50OdomNode::last_dvl_time_;
};

/**
 * @class DvlA50OdomNodeTest
 * @brief Test fixture for DvlA50OdomNode tests.
 */
class DvlA50OdomNodeTest : public ::testing::Test
{
protected:
  static void SetUpTestSuite()
  {
    if (!rclcpp::ok()) {
      rclcpp::init(0, nullptr);
    }
  }

  void SetUp() override
  {
    rclcpp::NodeOptions options;
    options.append_parameter_override("use_parameter_frame", true);
    options.append_parameter_override("parameter_frame", "dvl_link");
    options.append_parameter_override("dvl_odom_frame", "dvl_odom");
    options.append_parameter_override("base_frame", "base_link");
    options.append_parameter_override("publish_local_tf", false);
    options.append_parameter_override("publish_diagnostics", false);
    node = std::make_shared<TestDvlA50OdomNode>(options);

    // Publish a static identity transform from dvl_link -> base_link
    static_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(node);
    geometry_msgs::msg::TransformStamped tf;
    tf.header.stamp = node->get_clock()->now();
    tf.header.frame_id = "dvl_link";
    tf.child_frame_id = "base_link";
    tf.transform.translation.x = 0.0;
    tf.transform.translation.y = 0.0;
    tf.transform.translation.z = 0.0;
    tf.transform.rotation.w = 1.0;
    static_broadcaster_->sendTransform(tf);

    // Subscribe to the output topic
    received_odom_ = false;
    odom_sub_ = node->create_subscription<nav_msgs::msg::Odometry>(
      "odometry/dvl", rclcpp::SystemDefaultsQoS(),
      [this](const nav_msgs::msg::Odometry::SharedPtr msg) {
        last_odom_ = *msg;
        received_odom_ = true;
      });
  }

  /**
   * @brief Spin the node until a condition is met or timeout expires.
   */
  bool spinUntilOdom(double timeout_sec = 2.0)
  {
    auto start = std::chrono::steady_clock::now();
    while (!received_odom_) {
      rclcpp::spin_some(node);
      if (std::chrono::steady_clock::now() - start >
        std::chrono::duration<double>(timeout_sec))
      {
        return false;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return true;
  }

  std::shared_ptr<TestDvlA50OdomNode> node;
  std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_broadcaster_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  nav_msgs::msg::Odometry last_odom_;
  bool received_odom_ = false;
};

/**
 * @brief Verify that a DVLDR message produces a correct odometry output.
 *
 * With an identity TF (dvl_link == base_link), the output pose should
 * match the DVL DR position and orientation directly.
 */
TEST_F(DvlA50OdomNodeTest, DvlCallbackPublishesOdom) {
  // Allow the static TF to propagate
  rclcpp::spin_some(node);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  rclcpp::spin_some(node);

  auto msg = std::make_shared<dvl_msgs::msg::DVLDR>();
  msg->header.frame_id = "dvl_link";
  msg->position.x = 1.0;
  msg->position.y = 2.0;
  msg->position.z = 3.0;
  msg->roll = 0.0;
  msg->pitch = 0.0;
  msg->yaw = 0.0;
  msg->pos_std = 0.5;
  msg->time = 10.5;

  node->dvlCallback(msg);
  ASSERT_TRUE(spinUntilOdom());

  // Frame IDs
  EXPECT_STREQ(last_odom_.header.frame_id.c_str(), "dvl_odom");
  EXPECT_STREQ(last_odom_.child_frame_id.c_str(), "base_link");

  // Timestamp: 10.5 seconds -> sec=10, nanosec=500000000
  EXPECT_EQ(last_odom_.header.stamp.sec, 10);
  EXPECT_EQ(last_odom_.header.stamp.nanosec, static_cast<uint32_t>(500000000));

  // Position should pass through with identity TF
  EXPECT_NEAR(last_odom_.pose.pose.position.x, 1.0, 1e-6);
  EXPECT_NEAR(last_odom_.pose.pose.position.y, 2.0, 1e-6);
  EXPECT_NEAR(last_odom_.pose.pose.position.z, 3.0, 1e-6);

  // Covariance diagonal = pos_std^2 = 0.25
  EXPECT_DOUBLE_EQ(last_odom_.pose.covariance[0], 0.25);
  EXPECT_DOUBLE_EQ(last_odom_.pose.covariance[7], 0.25);
  EXPECT_DOUBLE_EQ(last_odom_.pose.covariance[14], 0.25);
}

/**
 * @brief Verify the diagnostic state updates after receiving a message.
 */
TEST_F(DvlA50OdomNodeTest, LastDvlTimeUpdated) {
  // Allow the static TF to propagate
  rclcpp::spin_some(node);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  rclcpp::spin_some(node);

  EXPECT_DOUBLE_EQ(node->last_dvl_time_.load(), 0.0);

  auto msg = std::make_shared<dvl_msgs::msg::DVLDR>();
  msg->header.frame_id = "dvl_link";
  msg->position.x = 0.0;
  msg->position.y = 0.0;
  msg->position.z = 0.0;
  msg->roll = 0.0;
  msg->pitch = 0.0;
  msg->yaw = 0.0;
  msg->pos_std = 1.0;
  msg->time = 1.0;

  node->dvlCallback(msg);

  EXPECT_GT(node->last_dvl_time_.load(), 0.0);
}
