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
 * @file fluid_pressure_odom.cpp
 * @brief Implementation of the FluidPressureOdomNode.
 * @author Nelson Durrant
 * @date May 2026
 */

#include "coug_fgo/fluid_pressure_odom.hpp"

#include <rclcpp_components/register_node_macro.hpp>

namespace coug_fgo {

FluidPressureOdomNode::FluidPressureOdomNode(const rclcpp::NodeOptions& options)
    : Node("fluid_pressure_odom_node", options), diagnostic_updater_(this) {
  RCLCPP_INFO(get_logger(), "Starting Fluid Pressure Odom Node...");

  param_listener_ =
      std::make_shared<fluid_pressure_odom_node::ParamListener>(get_node_parameters_interface());
  params_ = param_listener_->get_params();

  // --- ROS Interfaces ---
  pressure_sub_ = create_subscription<sensor_msgs::msg::FluidPressure>(
      params_.input_topic, rclcpp::SensorDataQoS(),
      std::bind(&FluidPressureOdomNode::pressureCallback, this, std::placeholders::_1));

  odom_pub_ =
      create_publisher<nav_msgs::msg::Odometry>(params_.output_topic, rclcpp::SystemDefaultsQoS());

  // --- ROS Diagnostics ---
  if (params_.publish_diagnostics) {
    std::string ns = this->get_namespace();
    std::string clean_ns = (ns == "/") ? "" : ns;
    diagnostic_updater_.setHardwareID(clean_ns + "/fluid_pressure_odom_node");

    std::string prefix = clean_ns.empty() ? "" : "[" + clean_ns + "] ";
    std::string pressure_task = prefix + "Pressure Sensor Status";
    diagnostic_updater_.add(pressure_task, this, &FluidPressureOdomNode::checkPressureStatus);
  }

  RCLCPP_INFO(get_logger(), "Startup complete! Waiting for pressure data...");
}

void FluidPressureOdomNode::pressureCallback(const sensor_msgs::msg::FluidPressure::SharedPtr msg) {
  last_pressure_time_ = this->get_clock()->now().seconds();

  nav_msgs::msg::Odometry odom_msg;
  odom_msg.header.stamp = msg->header.stamp;
  odom_msg.header.frame_id = params_.map_frame;

  if (params_.use_parameter_child_frame) {
    odom_msg.child_frame_id = params_.parameter_child_frame;
  } else {
    odom_msg.child_frame_id = msg->header.frame_id;
  }

  // depth [m] = (pressure [Pa] - atmospheric_pressure [Pa]) / (water_density [kg/m^3] * g [m/s^2])
  double pressure_to_depth = 1.0 / (params_.water_density * params_.gravity);
  double gauge_pressure = msg->fluid_pressure - params_.atmospheric_pressure;
  odom_msg.pose.pose.position.z = gauge_pressure * pressure_to_depth;

  // var_depth = var_pressure / (rho*g)^2
  double var_depth = (msg->variance > 0.0)
                         ? msg->variance * pressure_to_depth * pressure_to_depth
                         : params_.fallback_depth_noise_sigma * params_.fallback_depth_noise_sigma;
  odom_msg.pose.covariance[14] = var_depth;

  last_depth_ = odom_msg.pose.pose.position.z;
  odom_pub_->publish(odom_msg);
}

void FluidPressureOdomNode::checkPressureStatus(diagnostic_updater::DiagnosticStatusWrapper& stat) {
  stat.summary(diagnostic_msgs::msg::DiagnosticStatus::OK, "Pressure data acquired.");

  double time_since = (last_pressure_time_ > 0.0)
                          ? (this->get_clock()->now().seconds() - last_pressure_time_)
                          : -1.0;
  stat.add("Time Since Last (s)", time_since);
  stat.add("Last Depth (m)", last_depth_);

  if (time_since > params_.diagnostic_timeout || last_pressure_time_ == 0.0) {
    stat.mergeSummary(diagnostic_msgs::msg::DiagnosticStatus::ERROR, "Pressure sensor is offline.");
  }
}

}  // namespace coug_fgo

RCLCPP_COMPONENTS_REGISTER_NODE(coug_fgo::FluidPressureOdomNode)
