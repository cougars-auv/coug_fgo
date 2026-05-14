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
 * @file odom_ned_to_enu.cpp
 * @brief Implementation of the OdomNedToEnuNode.
 * @author Nelson Durrant
 * @date May 2026
 */

#include "coug_fgo/odom_ned_to_enu.hpp"

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Vector3.h>

#include <rclcpp_components/register_node_macro.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

namespace coug_fgo {

OdomNedToEnuNode::OdomNedToEnuNode(const rclcpp::NodeOptions& options)
    : Node("odom_ned_to_enu_node", options) {
  RCLCPP_INFO(get_logger(), "Starting Odom NED to ENU Node...");

  param_listener_ =
      std::make_shared<odom_ned_to_enu_node::ParamListener>(get_node_parameters_interface());
  params_ = param_listener_->get_params();

  // --- ROS Interfaces ---
  odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      params_.input_topic, rclcpp::SensorDataQoS(),
      std::bind(&OdomNedToEnuNode::odomCallback, this, std::placeholders::_1));

  odom_pub_ =
      create_publisher<nav_msgs::msg::Odometry>(params_.output_topic, rclcpp::SystemDefaultsQoS());

  RCLCPP_INFO(get_logger(), "Startup complete! Waiting for NED odometry data...");
}

void OdomNedToEnuNode::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
  odom_pub_->publish(convertToEnu(msg));
}

nav_msgs::msg::Odometry OdomNedToEnuNode::convertToEnu(
    const nav_msgs::msg::Odometry::SharedPtr msg) {
  static const tf2::Quaternion q_enu_ned(M_SQRT1_2, M_SQRT1_2, 0.0, 0.0);

  nav_msgs::msg::Odometry out = *msg;

  // Rotate position vector from NED to ENU
  tf2::Vector3 pos_ned(msg->pose.pose.position.x, msg->pose.pose.position.y,
                       msg->pose.pose.position.z);
  tf2::Vector3 pos_enu = tf2::quatRotate(q_enu_ned, pos_ned);
  out.pose.pose.position.x = pos_enu.x();
  out.pose.pose.position.y = pos_enu.y();
  out.pose.pose.position.z = pos_enu.z();

  // Rotate orientation from NED to ENU
  tf2::Quaternion q_ned_b;
  tf2::fromMsg(msg->pose.pose.orientation, q_ned_b);
  tf2::Quaternion q_enu_b = q_enu_ned * q_ned_b;
  q_enu_b.normalize();
  out.pose.pose.orientation = tf2::toMsg(q_enu_b);

  return out;
}

}  // namespace coug_fgo

RCLCPP_COMPONENTS_REGISTER_NODE(coug_fgo::OdomNedToEnuNode)
