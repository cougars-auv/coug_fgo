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
 * @file seatrac_x150_imu.hpp
 * @brief ROS 2 node that converts a SeaTrac ModemStatus message to IMU and MagneticField messages.
 * @author Nelson Durrant
 * @date May 2026
 */

#pragma once

#include <diagnostic_updater/diagnostic_updater.hpp>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <seatrac_interfaces/msg/modem_status.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/magnetic_field.hpp>

#include "coug_fgo/seatrac_x150_imu_parameters.hpp"

namespace coug_fgo {

/**
 * @class SeatracX150ImuNode
 * @brief ROS 2 node that converts a SeaTrac ModemStatus message to IMU and MagneticField messages.
 */
class SeatracX150ImuNode : public rclcpp::Node {
 public:
  /**
   * @brief Constructs the node and sets up ModemStatus conversion.
   * @param options The node options.
   */
  explicit SeatracX150ImuNode(const rclcpp::NodeOptions& options);

 protected:
  /**
   * @brief Callback for receiving new ModemStatus data.
   * @param msg The incoming ModemStatus message.
   */
  void modemStatusCallback(const seatrac_interfaces::msg::ModemStatus::SharedPtr msg);

  /**
   * @brief Converts a ModemStatus message to an Imu message.
   * @param msg The incoming ModemStatus message.
   * @return The converted Imu message.
   */
  sensor_msgs::msg::Imu convertToImu(const seatrac_interfaces::msg::ModemStatus::SharedPtr msg);

  /**
   * @brief Converts a ModemStatus message to a MagneticField message.
   * @param msg The incoming ModemStatus message.
   * @return The converted MagneticField message.
   */
  sensor_msgs::msg::MagneticField convertToMag(
      const seatrac_interfaces::msg::ModemStatus::SharedPtr msg);

  /**
   * @brief Diagnostic task to report the status of the modem data.
   * @param stat The diagnostic status wrapper.
   */
  void checkModemStatus(diagnostic_updater::DiagnosticStatusWrapper& stat);

  // --- ROS Interfaces ---
  rclcpp::Subscription<seatrac_interfaces::msg::ModemStatus>::SharedPtr modem_sub_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;
  rclcpp::Publisher<sensor_msgs::msg::MagneticField>::SharedPtr mag_pub_;
  diagnostic_updater::Updater diagnostic_updater_;

  // --- Parameters ---
  std::shared_ptr<seatrac_x150_imu_node::ParamListener> param_listener_;
  seatrac_x150_imu_node::Params params_;

  // --- State ---
  double last_modem_time_{0.0};
};

}  // namespace coug_fgo
