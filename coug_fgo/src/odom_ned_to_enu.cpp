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

#include <Eigen/Core>
#include <rclcpp_components/register_node_macro.hpp>

namespace coug_fgo {

OdomNedToEnuNode::OdomNedToEnuNode(const rclcpp::NodeOptions& options)
    : Node("odom_ned_to_enu_node", options) {
  param_listener_ =
      std::make_shared<odom_ned_to_enu_node::ParamListener>(get_node_parameters_interface());
  params_ = param_listener_->get_params();

  // --- ROS Interfaces ---
  odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      params_.input_topic, rclcpp::SensorDataQoS(),
      std::bind(&OdomNedToEnuNode::odomCallback, this, std::placeholders::_1));

  odom_pub_ =
      create_publisher<nav_msgs::msg::Odometry>(params_.output_topic, rclcpp::SystemDefaultsQoS());

  RCLCPP_INFO(get_logger(), "Initialization complete.");
}

void OdomNedToEnuNode::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
  odom_pub_->publish(convertToEnu(msg));
}

nav_msgs::msg::Odometry OdomNedToEnuNode::convertToEnu(
    const nav_msgs::msg::Odometry::SharedPtr msg) {
  nav_msgs::msg::Odometry out = *msg;

  // Rotate position vector from NED to ENU
  out.pose.pose.position.x = msg->pose.pose.position.y;
  out.pose.pose.position.y = msg->pose.pose.position.x;
  out.pose.pose.position.z = -msg->pose.pose.position.z;

  // Rotate orientation from NED to ENU
  const auto& q = msg->pose.pose.orientation;
  static constexpr double s = M_SQRT1_2;
  out.pose.pose.orientation.w = -s * (q.x + q.y);
  out.pose.pose.orientation.x = s * (q.w + q.z);
  out.pose.pose.orientation.y = s * (q.w - q.z);
  out.pose.pose.orientation.z = s * (q.y - q.x);

  if (out.pose.covariance[0] >= 0.0) {
    static const Eigen::Matrix<double, 6, 6> T = []() {
      static const Eigen::Matrix3d M = (Eigen::Matrix3d() << 0, 1, 0, 1, 0, 0, 0, 0, -1).finished();
      Eigen::Matrix<double, 6, 6> t = Eigen::Matrix<double, 6, 6>::Zero();
      t.block<3, 3>(0, 0) = M;
      t.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();
      return t;
    }();
    Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> cov(out.pose.covariance.data());
    cov = (T * cov * T.transpose()).eval();
  }

  return out;
}

}  // namespace coug_fgo

RCLCPP_COMPONENTS_REGISTER_NODE(coug_fgo::OdomNedToEnuNode)
