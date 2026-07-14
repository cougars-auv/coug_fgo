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

#include <cmath>
#include <rclcpp_components/register_node_macro.hpp>

namespace coug_fgo {

FluidPressureOdomNode::FluidPressureOdomNode(const rclcpp::NodeOptions& options)
    : Node("fluid_pressure_odom_node", options) {
  param_listener_ =
      std::make_shared<fluid_pressure_odom_node::ParamListener>(get_node_parameters_interface());
  params_ = param_listener_->get_params();

  // --- ROS Interfaces ---
  pressure_sub_ = create_subscription<sensor_msgs::msg::FluidPressure>(
      params_.input_topic, rclcpp::SensorDataQoS(),
      std::bind(&FluidPressureOdomNode::pressureCallback, this, std::placeholders::_1));

  odom_pub_ =
      create_publisher<nav_msgs::msg::Odometry>(params_.output_topic, rclcpp::SystemDefaultsQoS());

  calibrate_srv_ = create_service<std_srvs::srv::Trigger>(
      params_.calibrate_service, std::bind(&FluidPressureOdomNode::calibrateCallback, this,
                                           std::placeholders::_1, std::placeholders::_2));

  RCLCPP_INFO(get_logger(), "Initialization complete.");
}

void FluidPressureOdomNode::calibrateCallback(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
  (void)request;

  if (last_pressure_ < 0.0) {
    response->success = false;
    response->message = "No pressure data.";
    return;
  }

  calibrated_pressure_ = last_pressure_;
  calibrated_ = true;
  rejected_count_ = 0;

  response->success = true;
  response->message = "Depth calibrated.";
  RCLCPP_INFO(get_logger(), "Depth calibrated: zero reference set to %.1f Pa.",
              calibrated_pressure_);
}

void FluidPressureOdomNode::pressureCallback(const sensor_msgs::msg::FluidPressure::SharedPtr msg) {
  double pressure = msg->fluid_pressure * params_.pressure_scale;

  if (params_.max_pressure_delta > 0.0 && last_pressure_ >= 0.0 &&
      std::abs(pressure - last_pressure_) > params_.max_pressure_delta) {
    rejected_count_++;
    if (rejected_count_ <= params_.max_consecutive_rejections) {
      RCLCPP_WARN(get_logger(), "Rejected pressure spike.");
      return;
    }
    RCLCPP_WARN(get_logger(), "Accepting pressure step after %d consecutive rejections.",
                rejected_count_);
  }
  rejected_count_ = 0;
  last_pressure_ = pressure;

  nav_msgs::msg::Odometry odom_msg;
  odom_msg.header.stamp = msg->header.stamp;
  odom_msg.header.frame_id = params_.map_frame;

  odom_msg.child_frame_id =
      params_.use_parameter_child_frame ? params_.parameter_child_frame : msg->header.frame_id;

  // depth [m] = (pressure [Pa] - reference_pressure [Pa]) / (water_density [kg/m^3] * g [m/s^2])
  double reference_pressure = calibrated_ ? calibrated_pressure_ : params_.atmospheric_pressure;
  double pressure_to_depth = 1.0 / (params_.water_density * params_.gravity);
  double gauge_pressure = pressure - reference_pressure;
  odom_msg.pose.pose.position.z = -gauge_pressure * pressure_to_depth;
  odom_msg.pose.pose.orientation.w = 1.0;

  // var_depth = var_pressure / (rho*g)^2
  double var_pressure = msg->variance * params_.pressure_scale * params_.pressure_scale;
  double var_depth = var_pressure * pressure_to_depth * pressure_to_depth;
  odom_msg.pose.covariance[14] = var_depth;

  odom_pub_->publish(odom_msg);
}

}  // namespace coug_fgo

RCLCPP_COMPONENTS_REGISTER_NODE(coug_fgo::FluidPressureOdomNode)
