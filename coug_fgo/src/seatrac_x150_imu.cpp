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
 * @file seatrac_x150_imu.cpp
 * @brief Implementation of the SeatracX150ImuNode.
 * @author Nelson Durrant
 * @date May 2026
 */

#include "coug_fgo/seatrac_x150_imu.hpp"

#include <tf2/LinearMath/Quaternion.h>

#include <cmath>
#include <rclcpp_components/register_node_macro.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

namespace coug_fgo {

SeatracX150ImuNode::SeatracX150ImuNode(const rclcpp::NodeOptions& options)
    : Node("seatrac_x150_imu_node", options), diagnostic_updater_(this) {
  RCLCPP_INFO(get_logger(), "Starting Seatrac x150 IMU Node...");

  param_listener_ =
      std::make_shared<seatrac_x150_imu_node::ParamListener>(get_node_parameters_interface());
  params_ = param_listener_->get_params();

  // --- ROS Interfaces ---
  modem_sub_ = create_subscription<seatrac_interfaces::msg::ModemStatus>(
      params_.input_topic, rclcpp::SensorDataQoS(),
      std::bind(&SeatracX150ImuNode::modemStatusCallback, this, std::placeholders::_1));

  imu_pub_ = create_publisher<sensor_msgs::msg::Imu>(params_.imu_output_topic,
                                                     rclcpp::SystemDefaultsQoS());

  mag_pub_ = create_publisher<sensor_msgs::msg::MagneticField>(params_.mag_output_topic,
                                                               rclcpp::SystemDefaultsQoS());

  // --- ROS Diagnostics ---
  if (params_.publish_diagnostics) {
    std::string ns = this->get_namespace();
    std::string clean_ns = (ns == "/") ? "" : ns;
    diagnostic_updater_.setHardwareID(clean_ns + "/seatrac_x150_imu_node");

    std::string prefix = clean_ns.empty() ? "" : "[" + clean_ns + "] ";
    diagnostic_updater_.add(prefix + "Modem Status", this, &SeatracX150ImuNode::checkModemStatus);
  }

  RCLCPP_INFO(get_logger(), "Startup complete! Waiting for ModemStatus data...");
}

void SeatracX150ImuNode::modemStatusCallback(
    const seatrac_interfaces::msg::ModemStatus::SharedPtr msg) {
  last_modem_time_ = this->get_clock()->now().seconds();

  if (msg->includes_local_attitude || msg->includes_comp_ahrs) {
    imu_pub_->publish(convertToImu(msg));
  }

  if (msg->includes_comp_ahrs) {
    mag_pub_->publish(convertToMag(msg));
  }
}

sensor_msgs::msg::Imu SeatracX150ImuNode::convertToImu(
    const seatrac_interfaces::msg::ModemStatus::SharedPtr msg) {
  sensor_msgs::msg::Imu imu_msg;
  imu_msg.header = msg->header;
  if (params_.use_parameter_frame) {
    imu_msg.header.frame_id = params_.parameter_frame;
  }

  // Convert to quaternion
  if (msg->includes_local_attitude) {
    double roll_rad = (msg->attitude_roll / 10.0) * M_PI / 180.0;
    double pitch_rad = (msg->attitude_pitch / 10.0) * M_PI / 180.0;
    double yaw_rad = (msg->attitude_yaw / 10.0) * M_PI / 180.0;

    tf2::Quaternion q_ned_b;
    q_ned_b.setRPY(roll_rad, pitch_rad, yaw_rad);
    q_ned_b.normalize();
    imu_msg.orientation = tf2::toMsg(q_ned_b);

    auto& s = params_.orientation_noise_sigmas;
    imu_msg.orientation_covariance[0] = s[0] * s[0];
    imu_msg.orientation_covariance[4] = s[1] * s[1];
    imu_msg.orientation_covariance[8] = s[2] * s[2];
  } else {
    imu_msg.orientation_covariance[0] = -1.0;
  }

  if (msg->includes_comp_ahrs) {
    // Convert raw units and correct for 45° physical mounting offset of IMU chip
    constexpr double kAccScale = 9.80665 / 250.0;
    constexpr double kDegToRad = M_PI / 180.0;

    imu_msg.linear_acceleration.x = kAccScale * M_SQRT1_2 * (msg->acc_x - msg->acc_y);
    imu_msg.linear_acceleration.y = -kAccScale * M_SQRT1_2 * (msg->acc_x + msg->acc_y);
    imu_msg.linear_acceleration.z = kAccScale * msg->acc_z;

    imu_msg.angular_velocity.x = kDegToRad * M_SQRT1_2 * (msg->gyro_y - msg->gyro_x);
    imu_msg.angular_velocity.y = kDegToRad * M_SQRT1_2 * (msg->gyro_x + msg->gyro_y);
    imu_msg.angular_velocity.z = -kDegToRad * msg->gyro_z;

    auto& a = params_.accel_noise_sigmas;
    imu_msg.linear_acceleration_covariance[0] = a[0] * a[0];
    imu_msg.linear_acceleration_covariance[4] = a[1] * a[1];
    imu_msg.linear_acceleration_covariance[8] = a[2] * a[2];

    auto& g = params_.gyro_noise_sigmas;
    imu_msg.angular_velocity_covariance[0] = g[0] * g[0];
    imu_msg.angular_velocity_covariance[4] = g[1] * g[1];
    imu_msg.angular_velocity_covariance[8] = g[2] * g[2];
  } else {
    imu_msg.linear_acceleration_covariance[0] = -1.0;
    imu_msg.angular_velocity_covariance[0] = -1.0;
  }

  return imu_msg;
}

sensor_msgs::msg::MagneticField SeatracX150ImuNode::convertToMag(
    const seatrac_interfaces::msg::ModemStatus::SharedPtr msg) {
  sensor_msgs::msg::MagneticField mag_msg;
  mag_msg.header = msg->header;
  if (params_.use_parameter_frame) {
    mag_msg.header.frame_id = params_.parameter_frame;
  }

  mag_msg.magnetic_field.x = msg->mag_x;
  mag_msg.magnetic_field.y = msg->mag_y;
  mag_msg.magnetic_field.z = msg->mag_z;

  auto& m = params_.magnetic_field_noise_sigmas;
  mag_msg.magnetic_field_covariance[0] = m[0] * m[0];
  mag_msg.magnetic_field_covariance[4] = m[1] * m[1];
  mag_msg.magnetic_field_covariance[8] = m[2] * m[2];

  return mag_msg;
}

void SeatracX150ImuNode::checkModemStatus(diagnostic_updater::DiagnosticStatusWrapper& stat) {
  stat.summary(diagnostic_msgs::msg::DiagnosticStatus::OK, "Modem data acquired.");

  double time_since =
      (last_modem_time_ > 0.0) ? (this->get_clock()->now().seconds() - last_modem_time_) : -1.0;
  stat.add("Time Since Last (s)", time_since);

  if (time_since > params_.diagnostic_timeout || last_modem_time_ == 0.0) {
    stat.mergeSummary(diagnostic_msgs::msg::DiagnosticStatus::ERROR, "Modem is offline.");
  }
}

}  // namespace coug_fgo

RCLCPP_COMPONENTS_REGISTER_NODE(coug_fgo::SeatracX150ImuNode)
