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
 * @file odom_to_tf.hpp
 * @brief ROS 2 node that broadcasts a TF transform from an odometry topic.
 * @author Nelson Durrant
 * @date May 2026
 */

#pragma once

#include <tf2_ros/transform_broadcaster.h>

#include <memory>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>

#include "coug_fgo/odom_to_tf_parameters.hpp"

namespace coug_fgo {

/**
 * @class OdomToTfNode
 * @brief ROS 2 node that broadcasts a TF transform from an odometry topic.
 */
class OdomToTfNode : public rclcpp::Node {
 public:
  /**
   * @brief Constructs the node and sets up the odometry-to-TF broadcaster.
   * @param options The node options.
   */
  explicit OdomToTfNode(const rclcpp::NodeOptions& options);

 protected:
  /**
   * @brief Callback for receiving new odometry data.
   * @param msg The incoming Odometry message.
   */
  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);

  // --- ROS Interfaces ---
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  // --- Parameters ---
  std::shared_ptr<odom_to_tf_node::ParamListener> param_listener_;
  odom_to_tf_node::Params params_;
};

}  // namespace coug_fgo
