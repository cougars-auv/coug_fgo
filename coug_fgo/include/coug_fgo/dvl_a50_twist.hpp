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
 * @file dvl_a50_twist.hpp
 * @brief ROS 2 node that converts DVL A50 velocity data to a twist message.
 * @author Nelson Durrant
 * @date May 2026
 */

#pragma once

#include <dvl_msgs/msg/dvl.hpp>
#include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <string>

#include "coug_fgo/dvl_a50_twist_parameters.hpp"

namespace coug_fgo {

/**
 * @class DvlA50TwistNode
 * @brief ROS 2 node that converts DVL A50 velocity data to a twist message.
 */
class DvlA50TwistNode : public rclcpp::Node {
 public:
  /**
   * @brief Constructs the node and sets up DVL velocity conversion.
   * @param options The node options.
   */
  explicit DvlA50TwistNode(const rclcpp::NodeOptions& options);

 protected:
  /**
   * @brief Gates DVL samples on validity/FOM (and simulated dropout), then publishes the twist.
   * @param msg The incoming DVL message.
   */
  void dvlCallback(const dvl_msgs::msg::DVL::SharedPtr msg);

  /**
   * @brief Converts a DVL report to a stamped twist with FOM- or message-derived covariance.
   * @param msg The incoming DVL message.
   * @return The converted TwistWithCovarianceStamped message (DVL time-of-validity stamp).
   */
  geometry_msgs::msg::TwistWithCovarianceStamped convertToTwist(
      const dvl_msgs::msg::DVL::SharedPtr msg);

  // --- ROS Interfaces ---
  rclcpp::Subscription<dvl_msgs::msg::DVL>::SharedPtr dvl_sub_;
  rclcpp::Publisher<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr twist_pub_;

  // --- Parameters ---
  std::shared_ptr<dvl_a50_twist_node::ParamListener> param_listener_;
  dvl_a50_twist_node::Params params_;

  // --- State ---
  double last_dvl_time_{0.0};
  bool last_velocity_valid_{false};
  bool is_simulating_dropout_{false};
};

}  // namespace coug_fgo
