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
 * @file imu_ned_to_enu.cpp
 * @brief Implementation of the ImuNedToEnuNode.
 * @author Nelson Durrant
 * @date May 2026
 */

#include "coug_fgo/imu_ned_to_enu.hpp"

#include <cmath>
#include <rclcpp_components/register_node_macro.hpp>

namespace coug_fgo {

ImuNedToEnuNode::ImuNedToEnuNode(const rclcpp::NodeOptions& options)
    : Node("imu_ned_to_enu_node", options) {
  RCLCPP_INFO(get_logger(), "Starting IMU NED to ENU Node...");

  param_listener_ =
      std::make_shared<imu_ned_to_enu_node::ParamListener>(get_node_parameters_interface());
  params_ = param_listener_->get_params();

  // --- ROS Interfaces ---
  imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
      params_.input_topic, rclcpp::SensorDataQoS(),
      std::bind(&ImuNedToEnuNode::imuCallback, this, std::placeholders::_1));

  imu_pub_ =
      create_publisher<sensor_msgs::msg::Imu>(params_.output_topic, rclcpp::SystemDefaultsQoS());

  RCLCPP_INFO(get_logger(), "Startup complete! Waiting for NED IMU data...");
}

void ImuNedToEnuNode::imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg) {
  imu_pub_->publish(convertToEnu(msg));
}

sensor_msgs::msg::Imu ImuNedToEnuNode::convertToEnu(const sensor_msgs::msg::Imu::SharedPtr msg) {
  sensor_msgs::msg::Imu out = *msg;

  const auto& q = msg->orientation;
  static constexpr double s = M_SQRT1_2;
  out.orientation.w = -s * (q.x + q.y);
  out.orientation.x = s * (q.w + q.z);
  out.orientation.y = s * (q.w - q.z);
  out.orientation.z = s * (q.y - q.x);

  return out;
}

}  // namespace coug_fgo

RCLCPP_COMPONENTS_REGISTER_NODE(coug_fgo::ImuNedToEnuNode)
