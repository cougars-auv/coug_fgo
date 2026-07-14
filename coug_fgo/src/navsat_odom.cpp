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
 * @file navsat_odom.cpp
 * @brief Implementation of the NavsatOdomNode.
 * @author Nelson Durrant
 * @date May 2026
 */

#include "coug_fgo/navsat_odom.hpp"

#include <algorithm>
#include <chrono>
#include <rclcpp_components/register_node_macro.hpp>

namespace coug_fgo {

NavsatOdomNode::NavsatOdomNode(const rclcpp::NodeOptions& options)
    : Node("navsat_odom_node", options),
      diagnostic_updater_(this),
      local_cartesian_(0.0, 0.0, 0.0, GeographicLib::Geocentric::WGS84()) {
  param_listener_ =
      std::make_shared<navsat_odom_node::ParamListener>(get_node_parameters_interface());
  params_ = param_listener_->get_params();

  // --- ROS Interfaces ---
  navsat_sub_ = create_subscription<sensor_msgs::msg::NavSatFix>(
      params_.input_topic, rclcpp::SensorDataQoS(),
      std::bind(&NavsatOdomNode::navsatCallback, this, std::placeholders::_1));

  odom_pub_ =
      create_publisher<nav_msgs::msg::Odometry>(params_.output_topic, rclcpp::SystemDefaultsQoS());

  if (params_.set_origin) {
    origin_pub_ = create_publisher<sensor_msgs::msg::NavSatFix>(params_.origin_topic,
                                                                rclcpp::SystemDefaultsQoS());
    origin_timer_ = create_wall_timer(
        std::chrono::milliseconds(static_cast<int>(1000.0 / params_.origin_pub_rate_hz)), [this]() {
          if (origin_set_) {
            origin_pub_->publish(origin_navsat_);
          }
        });
  } else {
    origin_sub_ = create_subscription<sensor_msgs::msg::NavSatFix>(
        params_.origin_topic, rclcpp::SystemDefaultsQoS(),
        std::bind(&NavsatOdomNode::originCallback, this, std::placeholders::_1));
  }

  if (params_.set_origin && params_.use_parameter_origin) {
    sensor_msgs::msg::NavSatFix origin;
    origin.header.frame_id = params_.map_frame;
    origin.status.status = sensor_msgs::msg::NavSatStatus::STATUS_FIX;
    origin.latitude = params_.origin_latitude;
    origin.longitude = params_.origin_longitude;
    origin.altitude = params_.origin_altitude;
    setOrigin(origin);
    RCLCPP_INFO(get_logger(), "Parameter origin set: Lat %.6f, Lon %.6f, Alt %.2f",
                origin_navsat_.latitude, origin_navsat_.longitude, origin_navsat_.altitude);
  }

  // --- ROS Diagnostics ---
  if (params_.publish_diagnostics) {
    std::string ns = this->get_namespace();
    std::string clean_ns = (ns == "/") ? "" : ns;
    diagnostic_updater_.setHardwareID(clean_ns + "/navsat_odom_node");

    std::string prefix = clean_ns.empty() ? "" : "[" + clean_ns + "] ";

    std::string origin_task = prefix + "Origin Status";
    diagnostic_updater_.add(origin_task, this, &NavsatOdomNode::checkOriginStatus);
  }

  RCLCPP_INFO(get_logger(), "Initialization complete.");
}

void NavsatOdomNode::setOrigin(const sensor_msgs::msg::NavSatFix& msg) {
  local_cartesian_.Reset(msg.latitude, msg.longitude, msg.altitude);
  origin_navsat_ = msg;
  if (origin_navsat_.header.frame_id.empty()) {
    origin_navsat_.header.frame_id = params_.map_frame;
  }
  origin_set_ = true;
}

void NavsatOdomNode::originCallback(const sensor_msgs::msg::NavSatFix::SharedPtr msg) {
  if (!origin_set_ && msg->status.status >= sensor_msgs::msg::NavSatStatus::STATUS_FIX) {
    setOrigin(*msg);
    RCLCPP_INFO(get_logger(), "GPS origin received: Lat %.6f, Lon %.6f, Alt %.2f",
                origin_navsat_.latitude, origin_navsat_.longitude, origin_navsat_.altitude);
  }
}

void NavsatOdomNode::navsatCallback(const sensor_msgs::msg::NavSatFix::SharedPtr msg) {
  if (msg->status.status == sensor_msgs::msg::NavSatStatus::STATUS_NO_FIX) {
    RCLCPP_WARN(get_logger(), "Received NavSatFix with no fix.");
    return;
  }

  if (!origin_set_) {
    if (!params_.set_origin) {
      return;
    }
    setOrigin(*msg);
    RCLCPP_INFO(get_logger(), "GPS origin set: Lat %.6f, Lon %.6f, Alt %.2f",
                origin_navsat_.latitude, origin_navsat_.longitude, origin_navsat_.altitude);
    return;
  }

  if (msg->position_covariance_type == sensor_msgs::msg::NavSatFix::COVARIANCE_TYPE_UNKNOWN) {
    RCLCPP_ERROR(get_logger(), "Unknown covariance type.");
    return;
  }

  // Reject degraded fixes (e.g. when using GPS as ground truth)
  if (params_.position_covariance_threshold > 0.0 &&
      std::max(msg->position_covariance[0], msg->position_covariance[4]) >
          params_.position_covariance_threshold) {
    RCLCPP_WARN(get_logger(), "Rejected high-covariance fix.");
    return;
  }

  nav_msgs::msg::Odometry odom_msg;
  odom_msg.header.stamp = msg->header.stamp;
  odom_msg.header.frame_id = params_.map_frame;
  odom_msg.child_frame_id =
      params_.use_parameter_child_frame ? params_.parameter_child_frame : msg->header.frame_id;

  double east, north, up;
  local_cartesian_.Forward(msg->latitude, msg->longitude, msg->altitude, east, north, up);

  odom_msg.pose.pose.position.x = east;
  odom_msg.pose.pose.position.y = north;
  odom_msg.pose.pose.position.z = up;

  odom_msg.pose.pose.orientation.w = 1.0;

  const auto& cov = msg->position_covariance;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      odom_msg.pose.covariance[i * 6 + j] = cov[i * 3 + j];
    }
  }

  odom_msg.pose.covariance[21] = 1e9;
  odom_msg.pose.covariance[28] = 1e9;
  odom_msg.pose.covariance[35] = 1e9;
  odom_msg.twist.covariance[0] = -1.0;

  odom_pub_->publish(odom_msg);
}

void NavsatOdomNode::checkOriginStatus(diagnostic_updater::DiagnosticStatusWrapper& stat) {
  if (origin_set_) {
    stat.add("Origin Latitude", origin_navsat_.latitude);
    stat.add("Origin Longitude", origin_navsat_.longitude);
    stat.add("Origin Altitude", origin_navsat_.altitude);
    stat.summary(diagnostic_msgs::msg::DiagnosticStatus::OK, "Origin successfully set.");
  } else {
    stat.summary(diagnostic_msgs::msg::DiagnosticStatus::WARN, "Waiting for origin.");
  }
}

}  // namespace coug_fgo

RCLCPP_COMPONENTS_REGISTER_NODE(coug_fgo::NavsatOdomNode)
