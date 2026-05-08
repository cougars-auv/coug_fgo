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
 * @file navsat_odom.hpp
 * @brief ROS 2 node that converts GPS NavSatFix data to an ENU odometry message.
 * @author Nelson Durrant
 * @date May 2026
 */

#pragma once

#include <GeographicLib/LocalCartesian.hpp>
#include <diagnostic_updater/diagnostic_updater.hpp>
#include <memory>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <string>

#include "coug_fgo/navsat_odom_parameters.hpp"

namespace coug_fgo {

/**
 * @class NavsatOdomNode
 * @brief ROS 2 node that converts GPS NavSatFix data to an ENU odometry message.
 */
class NavsatOdomNode : public rclcpp::Node {
 public:
  /**
   * @brief Constructs the node and sets up the GPS to ENU odometry conversion.
   * @param options The node options.
   */
  explicit NavsatOdomNode(const rclcpp::NodeOptions& options);

 protected:
  /**
   * @brief Callback for receiving new GPS NavSatFix data.
   * @param msg The incoming NavSatFix message.
   */
  void navsatCallback(const sensor_msgs::msg::NavSatFix::SharedPtr msg);

  /**
   * @brief Callback for receiving an origin from an external source.
   * @param msg The incoming origin NavSatFix message.
   */
  void originCallback(const sensor_msgs::msg::NavSatFix::SharedPtr msg);

  /**
   * @brief Sets the ENU origin and seeds the LocalCartesian projection.
   * @param msg The NavSatFix to use as the origin.
   */
  void setOrigin(const sensor_msgs::msg::NavSatFix& msg);

  /**
   * @brief Diagnostic task to report the status of the ENU origin.
   * @param stat The diagnostic status wrapper.
   */
  void checkOriginStatus(diagnostic_updater::DiagnosticStatusWrapper& stat);

  /**
   * @brief Diagnostic task to report the status of the GPS receiver.
   * @param stat The diagnostic status wrapper.
   */
  void checkNavSatFix(diagnostic_updater::DiagnosticStatusWrapper& stat);

  // --- ROS Interfaces ---
  rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr navsat_sub_;
  rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr origin_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<sensor_msgs::msg::NavSatFix>::SharedPtr origin_pub_;
  rclcpp::TimerBase::SharedPtr origin_timer_;
  diagnostic_updater::Updater diagnostic_updater_;

  // --- Parameters ---
  std::shared_ptr<navsat_odom_node::ParamListener> param_listener_;
  navsat_odom_node::Params params_;

  // --- State ---
  GeographicLib::LocalCartesian local_cartesian_;
  sensor_msgs::msg::NavSatFix origin_navsat_;
  bool origin_set_{false};
  double last_navsat_time_{0.0};
  int last_fix_status_{sensor_msgs::msg::NavSatStatus::STATUS_NO_FIX};
};

}  // namespace coug_fgo
