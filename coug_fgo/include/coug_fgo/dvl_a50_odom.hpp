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
 * @file dvl_a50_odom.hpp
 * @brief ROS 2 node that converts DVL A50 dead-reckoning data to an odometry message.
 * @author Nelson Durrant
 * @date May 2026
 */

#pragma once

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include <diagnostic_updater/diagnostic_updater.hpp>
#include <dvl_msgs/msg/dvldr.hpp>
#include <memory>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <string>

#include "coug_fgo/dvl_a50_odom_parameters.hpp"

namespace coug_fgo {

/**
 * @class DvlA50OdomNode
 * @brief ROS 2 node that converts DVL A50 dead-reckoning data to an odometry message.
 */
class DvlA50OdomNode : public rclcpp::Node {
 public:
  /**
   * @brief Constructs the node and sets up DVL dead-reckoning conversion.
   * @param options The node options.
   */
  explicit DvlA50OdomNode(const rclcpp::NodeOptions& options);

 protected:
  /**
   * @brief Callback for receiving new DVL DR data.
   * @param msg The incoming DVLDR message.
   */
  void dvlCallback(const dvl_msgs::msg::DVLDR::SharedPtr msg);

  /**
   * @brief Diagnostic task to report the status of the DVL data.
   * @param stat The diagnostic status wrapper.
   */
  void checkDvlStatus(diagnostic_updater::DiagnosticStatusWrapper& stat);

  // --- ROS Interfaces ---
  rclcpp::Subscription<dvl_msgs::msg::DVLDR>::SharedPtr dvl_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  diagnostic_updater::Updater diagnostic_updater_;

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  // --- Parameters ---
  std::shared_ptr<dvl_a50_odom_node::ParamListener> param_listener_;
  dvl_a50_odom_node::Params params_;

  // --- State ---
  double last_dvl_time_{0.0};
};

}  // namespace coug_fgo
