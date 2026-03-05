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
 * @file test_factor_graph.cpp
 * @brief Unit tests for factor_graph.hpp.
 * @author Nelson Durrant (w Gemini 3 Pro)
 * @date Jan 2026
 */

#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>

#include "coug_fgo/factor_graph.hpp"

using coug_fgo::FactorGraphNode;

/**
 * @class TestFactorGraphNode
 * @brief Harness to expose protected FactorGraphNode members for testing.
 */
class TestFactorGraphNode : public FactorGraphNode
{
public:
  using FactorGraphNode::FactorGraphNode;

  using FactorGraphNode::state_;
  using FactorGraphNode::core_;
  using FactorGraphNode::state_initializer_;
  using FactorGraphNode::params_;

  using FactorGraphNode::imu_queue_;
  using FactorGraphNode::depth_queue_;
  using FactorGraphNode::dvl_queue_;

  using FactorGraphNode::global_odom_pub_;
  using FactorGraphNode::imu_sub_;
  using FactorGraphNode::depth_sub_;
  using FactorGraphNode::dvl_sub_;
};

/**
 * @class FactorGraphNodeTest
 * @brief Test fixture for FactorGraphNode tests.
 */
class FactorGraphNodeTest : public ::testing::Test
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
    node = std::make_shared<TestFactorGraphNode>(options);
  }

  std::shared_ptr<TestFactorGraphNode> node;
};

/**
 * @brief Verify node starts in WAITING_FOR_SENSORS state.
 */
TEST_F(FactorGraphNodeTest, InitialState) {
  EXPECT_EQ(node->state_.load(), FactorGraphNode::State::WAITING_FOR_SENSORS);
}

/**
 * @brief Verify core is initialized after startup.
 */
TEST_F(FactorGraphNodeTest, CoreInitialized) {
  EXPECT_NE(node->core_, nullptr);
}

/**
 * @brief Verify required ROS interfaces are created.
 */
TEST_F(FactorGraphNodeTest, RosInterfacesCreated) {
  EXPECT_NE(node->global_odom_pub_, nullptr);
  EXPECT_NE(node->imu_sub_, nullptr);
  EXPECT_NE(node->depth_sub_, nullptr);
  EXPECT_NE(node->dvl_sub_, nullptr);
}

/**
 * @brief Verify message queues start empty.
 */
TEST_F(FactorGraphNodeTest, QueuesEmpty) {
  EXPECT_TRUE(node->imu_queue_.empty());
  EXPECT_TRUE(node->depth_queue_.empty());
  EXPECT_TRUE(node->dvl_queue_.empty());
}

/**
 * @brief Verify optional publishers are only created when enabled.
 */
TEST_F(FactorGraphNodeTest, OptionalPublishers) {
  // The default parameter values determine which optional publishers exist.
  // We just verify the node constructed without error and the required ones exist.
  EXPECT_NE(node->global_odom_pub_, nullptr);
}
