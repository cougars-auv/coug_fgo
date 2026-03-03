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
 * @file dvl_a50_odom_node.cpp
 * @brief Implementation of the DvlA50OdomNode.
 * @author Nelson Durrant
 * @date Jan 2026
 */

#include "coug_fgo/dvl_a50_odom_node.hpp"

#include <cmath>

#include <rclcpp_components/register_node_macro.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

namespace coug_fgo
{

DvlA50OdomNode::DvlA50OdomNode(const rclcpp::NodeOptions & options)
: Node("dvl_a50_odom_node", options),
  diagnostic_updater_(this)
{
  RCLCPP_INFO(get_logger(), "Starting DVL A50 Odom Node...");

  param_listener_ = std::make_shared<dvl_a50_odom_node::ParamListener>(
    get_node_parameters_interface());
  params_ = param_listener_->get_params();

  // --- ROS Interfaces ---
  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

  dvl_sub_ = create_subscription<dvl_msgs::msg::DVLDR>(
    params_.input_topic, rclcpp::SensorDataQoS(),
    std::bind(&DvlA50OdomNode::dvlCallback, this, std::placeholders::_1));

  odom_pub_ = create_publisher<nav_msgs::msg::Odometry>(
    params_.output_topic, 10);

  // --- ROS Diagnostics ---
  if (params_.publish_diagnostics) {
    std::string ns = this->get_namespace();
    std::string clean_ns = (ns == "/") ? "" : ns;
    diagnostic_updater_.setHardwareID(clean_ns + "/dvl_a50_odom_node");

    std::string prefix = clean_ns.empty() ? "" : "[" + clean_ns + "] ";
    std::string dvl_task = prefix + "DVL DR Status";
    diagnostic_updater_.add(dvl_task, this, &DvlA50OdomNode::checkDvlStatus);
  }

  RCLCPP_INFO(get_logger(), "Startup complete! Waiting for DVL DR data...");
}

void DvlA50OdomNode::dvlCallback(const dvl_msgs::msg::DVLDR::SharedPtr msg)
{
  last_dvl_time_ = this->get_clock()->now().seconds();

  std::string current_dvl_frame = params_.use_parameter_frame ? params_.parameter_frame : msg->header.frame_id;

  geometry_msgs::msg::TransformStamped dvl_T_base_tf;
  try {
    dvl_T_base_tf = tf_buffer_->lookupTransform(
      current_dvl_frame, params_.base_frame, tf2::TimePointZero);
  } catch (const tf2::TransformException & ex) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 1000,
      "Could not transform %s to %s: %s",
      current_dvl_frame.c_str(), params_.base_frame.c_str(), ex.what());
    return;
  }

  geometry_msgs::msg::PoseStamped p_base_in_dvl;
  p_base_in_dvl.header.frame_id = current_dvl_frame;
  p_base_in_dvl.pose.position.x = dvl_T_base_tf.transform.translation.x;
  p_base_in_dvl.pose.position.y = dvl_T_base_tf.transform.translation.y;
  p_base_in_dvl.pose.position.z = dvl_T_base_tf.transform.translation.z;
  p_base_in_dvl.pose.orientation = dvl_T_base_tf.transform.rotation;

  geometry_msgs::msg::TransformStamped odom_T_dvl_tf;
  odom_T_dvl_tf.header.frame_id = params_.dvl_odom_frame;
  odom_T_dvl_tf.child_frame_id = current_dvl_frame;
  odom_T_dvl_tf.transform.translation.x = msg->position.x;
  odom_T_dvl_tf.transform.translation.y = msg->position.y;
  odom_T_dvl_tf.transform.translation.z = msg->position.z;

  tf2::Quaternion q;
  q.setRPY(
      msg->roll * M_PI / 180.0,
      msg->pitch * M_PI / 180.0,
      msg->yaw * M_PI / 180.0);
  odom_T_dvl_tf.transform.rotation = tf2::toMsg(q);

  geometry_msgs::msg::Pose p_base_in_odom;
  tf2::doTransform(p_base_in_dvl.pose, p_base_in_odom, odom_T_dvl_tf);

  nav_msgs::msg::Odometry odom;
  odom.header.frame_id = params_.dvl_odom_frame;

  odom.child_frame_id = params_.base_frame;

  uint64_t sec = static_cast<uint64_t>(msg->time);
  uint64_t nanosec = static_cast<uint64_t>((msg->time - sec) * 1e9);
  odom.header.stamp = rclcpp::Time(sec, nanosec, RCL_ROS_TIME);

  odom.pose.pose = p_base_in_odom;

  double var = msg->pos_std * msg->pos_std;
  odom.pose.covariance[0] = var;
  odom.pose.covariance[7] = var;
  odom.pose.covariance[14] = var;

  odom_pub_->publish(odom);

  if (params_.publish_local_tf) {
    geometry_msgs::msg::TransformStamped ts;
    ts.header = odom.header;
    ts.child_frame_id = odom.child_frame_id;
    ts.transform.translation.x = p_base_in_odom.position.x;
    ts.transform.translation.y = p_base_in_odom.position.y;
    ts.transform.translation.z = p_base_in_odom.position.z;
    ts.transform.rotation = p_base_in_odom.orientation;
    tf_broadcaster_->sendTransform(ts);
  }
}

void DvlA50OdomNode::checkDvlStatus(diagnostic_updater::DiagnosticStatusWrapper & stat)
{
  stat.summary(diagnostic_msgs::msg::DiagnosticStatus::OK, "DVL DR data acquired.");

  double time_since =
    (last_dvl_time_ > 0.0) ? (this->get_clock()->now().seconds() - last_dvl_time_) : -1.0;
  stat.add("Time Since Last (s)", time_since);

  if (time_since > params_.timeout_threshold || last_dvl_time_ == 0.0) {
    stat.mergeSummary(diagnostic_msgs::msg::DiagnosticStatus::ERROR, "DVL DR is offline.");
  }
}

}  // namespace coug_fgo

RCLCPP_COMPONENTS_REGISTER_NODE(coug_fgo::DvlA50OdomNode)
