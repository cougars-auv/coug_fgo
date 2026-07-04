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
 * @file dvl_a50_twist.cpp
 * @brief Implementation of the DvlA50TwistNode.
 * @author Nelson Durrant
 * @date May 2026
 */

#include "coug_fgo/dvl_a50_twist.hpp"

#include <cmath>
#include <rclcpp_components/register_node_macro.hpp>

namespace coug_fgo {

DvlA50TwistNode::DvlA50TwistNode(const rclcpp::NodeOptions& options)
    : Node("dvl_a50_twist_node", options), diagnostic_updater_(this) {
  RCLCPP_INFO(get_logger(), "Starting DVL A50 Twist Node...");

  param_listener_ =
      std::make_shared<dvl_a50_twist_node::ParamListener>(get_node_parameters_interface());
  params_ = param_listener_->get_params();

  // --- ROS Interfaces ---
  dvl_sub_ = create_subscription<dvl_msgs::msg::DVL>(
      params_.input_topic, rclcpp::SensorDataQoS(),
      std::bind(&DvlA50TwistNode::dvlCallback, this, std::placeholders::_1));

  twist_pub_ = create_publisher<geometry_msgs::msg::TwistWithCovarianceStamped>(
      params_.output_topic, rclcpp::SystemDefaultsQoS());

  // --- ROS Diagnostics ---
  if (params_.publish_diagnostics) {
    std::string ns = this->get_namespace();
    std::string clean_ns = (ns == "/") ? "" : ns;
    diagnostic_updater_.setHardwareID(clean_ns + "/dvl_a50_twist_node");

    std::string prefix = clean_ns.empty() ? "" : "[" + clean_ns + "] ";
    std::string dvl_task = prefix + "DVL Status";
    diagnostic_updater_.add(dvl_task, this, &DvlA50TwistNode::checkDvlStatus);
  }

  RCLCPP_INFO(get_logger(), "Startup complete! Waiting for DVL data...");
}

void DvlA50TwistNode::dvlCallback(const dvl_msgs::msg::DVL::SharedPtr msg) {
  const auto now = this->get_clock()->now();
  last_dvl_time_ = now.seconds();
  last_velocity_valid_ = msg->velocity_valid;

  if (params_.simulate_dropout && params_.dropout_frequency_hz > 0.0) {
    double cycle_period = 1.0 / params_.dropout_frequency_hz;
    if (std::fmod(last_dvl_time_, cycle_period) < params_.dropout_duration_sec) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), static_cast<int>(cycle_period * 1000),
                           "Simulating DVL dropout...");
      return;
    }
  }

  if (!msg->velocity_valid && msg->fom > params_.fom_valid_threshold) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "Received invalid DVL velocity.");
    return;
  }

  geometry_msgs::msg::TwistWithCovarianceStamped twist_msg = convertToTwist(msg);
  twist_pub_->publish(twist_msg);
}

geometry_msgs::msg::TwistWithCovarianceStamped DvlA50TwistNode::convertToTwist(
    const dvl_msgs::msg::DVL::SharedPtr msg) {
  geometry_msgs::msg::TwistWithCovarianceStamped twist_msg;
  twist_msg.header.frame_id =
      params_.use_parameter_frame ? params_.parameter_frame : msg->header.frame_id;

  uint64_t sec = msg->time_of_validity / 1000000;
  uint64_t nanosec = (msg->time_of_validity % 1000000) * 1000;
  twist_msg.header.stamp = rclcpp::Time(sec, nanosec, RCL_ROS_TIME);

  twist_msg.twist.twist.linear.x = msg->velocity.x;
  twist_msg.twist.twist.linear.y = msg->velocity.y;
  twist_msg.twist.twist.linear.z = msg->velocity.z;

  if (params_.use_fom_covariance) {
    double cov_val = msg->fom * params_.fom_covariance_scale;
    twist_msg.twist.covariance[0] = cov_val;
    twist_msg.twist.covariance[7] = cov_val;
    twist_msg.twist.covariance[14] = cov_val;
  } else {
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        twist_msg.twist.covariance[i * 6 + j] = msg->covariance[i * 3 + j];
      }
    }
  }
  return twist_msg;
}

void DvlA50TwistNode::checkDvlStatus(diagnostic_updater::DiagnosticStatusWrapper& stat) {
  stat.add("Velocity Valid", last_velocity_valid_);

  double time_since =
      (last_dvl_time_ > 0.0) ? (this->get_clock()->now().seconds() - last_dvl_time_) : -1.0;
  stat.add("Time Since Last (s)", time_since);

  if (time_since > params_.diagnostic_timeout_sec || last_dvl_time_ == 0.0) {
    stat.summary(diagnostic_msgs::msg::DiagnosticStatus::ERROR, "DVL is offline.");
  } else if (!last_velocity_valid_) {
    stat.summary(diagnostic_msgs::msg::DiagnosticStatus::WARN, "DVL velocity is invalid.");
  } else {
    stat.summary(diagnostic_msgs::msg::DiagnosticStatus::OK, "DVL data acquired.");
  }
}

}  // namespace coug_fgo

RCLCPP_COMPONENTS_REGISTER_NODE(coug_fgo::DvlA50TwistNode)
