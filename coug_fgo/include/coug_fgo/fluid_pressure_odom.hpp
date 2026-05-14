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
 * @file fluid_pressure_odom.hpp
 * @brief ROS 2 node that converts fluid pressure data to a depth odometry message.
 * @author Nelson Durrant
 * @date May 2026
 */

#pragma once

#include <diagnostic_updater/diagnostic_updater.hpp>
#include <memory>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/fluid_pressure.hpp>
#include <string>

#include "coug_fgo/fluid_pressure_odom_parameters.hpp"

namespace coug_fgo {

/**
 * @class FluidPressureOdomNode
 * @brief ROS 2 node that converts fluid pressure data to a depth odometry message.
 */
class FluidPressureOdomNode : public rclcpp::Node {
 public:
  /**
   * @brief Constructs the node and sets up fluid pressure conversion.
   * @param options The node options.
   */
  explicit FluidPressureOdomNode(const rclcpp::NodeOptions& options);

 protected:
  /**
   * @brief Callback for receiving new fluid pressure data.
   * @param msg The incoming FluidPressure message.
   */
  void pressureCallback(const sensor_msgs::msg::FluidPressure::SharedPtr msg);

  /**
   * @brief Diagnostic task to report the status of the pressure sensor.
   * @param stat The diagnostic status wrapper.
   */
  void checkPressureStatus(diagnostic_updater::DiagnosticStatusWrapper& stat);

  // --- ROS Interfaces ---
  rclcpp::Subscription<sensor_msgs::msg::FluidPressure>::SharedPtr pressure_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  diagnostic_updater::Updater diagnostic_updater_;

  // --- Parameters ---
  std::shared_ptr<fluid_pressure_odom_node::ParamListener> param_listener_;
  fluid_pressure_odom_node::Params params_;

  // --- State ---
  double last_pressure_time_{0.0};
  double last_depth_{0.0};
  double last_pressure_{-1.0};
};

}  // namespace coug_fgo
