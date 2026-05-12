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
 * @file imu_ned_to_enu.hpp
 * @brief ROS 2 node that converts AHRS IMU orientation from a NED to an ENU world convention.
 * @author Nelson Durrant
 * @date May 2026
 */

#pragma once

#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include "coug_fgo/imu_ned_to_enu_parameters.hpp"

namespace coug_fgo {

/**
 * @class ImuNedToEnuNode
 * @brief ROS 2 node that converts AHRS IMU orientation from a NED to an ENU world convention.
 */
class ImuNedToEnuNode : public rclcpp::Node {
 public:
  /**
   * @brief Constructs the node and sets up the NED to ENU orientation conversion.
   * @param options The node options.
   */
  explicit ImuNedToEnuNode(const rclcpp::NodeOptions& options);

 protected:
  /**
   * @brief Callback for receiving new IMU data.
   * @param msg The incoming IMU message.
   */
  void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg);

  /**
   * @brief Converts an IMU message from a NED to an ENU world convention.
   * @param msg The incoming IMU message.
   * @return The converted IMU message.
   */
  sensor_msgs::msg::Imu convertToEnu(const sensor_msgs::msg::Imu::SharedPtr msg);

  // --- ROS Interfaces ---
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;

  // --- Parameters ---
  std::shared_ptr<imu_ned_to_enu_node::ParamListener> param_listener_;
  imu_ned_to_enu_node::Params params_;
};

}  // namespace coug_fgo
