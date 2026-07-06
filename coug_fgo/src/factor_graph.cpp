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
 * @file factor_graph.cpp
 * @brief Implementation of the FactorGraphNode.
 * @author Nelson Durrant
 * @date May 2026
 */

#include "coug_fgo/factor_graph.hpp"

#include <algorithm>
#include <rclcpp_components/register_node_macro.hpp>
#include <stdexcept>

#include "coug_fgo/utils/param_enums.hpp"
#include "coug_fgo/utils/ros_conversions.hpp"

namespace coug_fgo {

using utils::KeyframeSource;
using utils::parseKeyframeSource;
using utils::parseSolverType;
using utils::SolverType;
using utils::StateInitializer;
using utils::toCovariance36Msg;
using utils::toGtsam;
using utils::toPoseCovarianceMsg;
using utils::toPoseMsg;
using utils::toQuatMsg;
using utils::toVectorMsg;

namespace {

/**
 * @brief Maps a row-major ROS covariance array onto an N x N Eigen matrix.
 * @param arr The flat covariance array from a ROS message.
 * @return The covariance as a column-major Eigen matrix.
 */
template <int N, typename Array>
Eigen::Matrix<double, N, N> toCovMatrix(const Array& arr) {
  return Eigen::Map<const Eigen::Matrix<double, N, N, Eigen::RowMajor>>(arr.data());
}

}  // namespace

void FactorGraphNode::setupRosInterfaces() {
  // --- ROS TF Interfaces ---
  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_, this);
  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

  // --- ROS Publishers ---
  global_odom_pub_ = create_publisher<nav_msgs::msg::Odometry>(params_.global_odom_topic,
                                                               rclcpp::SystemDefaultsQoS());
  if (params_.publish_smoothed_path) {
    smoothed_path_pub_ = create_publisher<nav_msgs::msg::Path>(params_.smoothed_path_topic,
                                                               rclcpp::SystemDefaultsQoS());
  }
  if (params_.publish_velocity) {
    velocity_pub_ = create_publisher<geometry_msgs::msg::TwistWithCovarianceStamped>(
        params_.velocity_topic, rclcpp::SystemDefaultsQoS());
  }
  if (params_.publish_imu_bias) {
    imu_bias_pub_ = create_publisher<geometry_msgs::msg::TwistWithCovarianceStamped>(
        params_.imu_bias_topic, rclcpp::SystemDefaultsQoS());
  }
  if (params_.publish_graph_metrics) {
    graph_metrics_pub_ = create_publisher<coug_interfaces::msg::GraphMetrics>(
        params_.graph_metrics_topic, rclcpp::SystemDefaultsQoS());
  }

  // --- ROS Services ---
  reset_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  reset_srv_ = create_service<std_srvs::srv::Trigger>(
      params_.reset_service,
      [this](const std_srvs::srv::Trigger::Request::SharedPtr req,
             std::shared_ptr<std_srvs::srv::Trigger::Response> res) { resetGraph(req, res); },
      rclcpp::ServicesQoS(), reset_cb_group_);

  // --- ROS Callback Groups ---
  sensor_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
  auto sensor_options = rclcpp::SubscriptionOptions();
  sensor_options.callback_group = sensor_cb_group_;

  // --- ROS Subscribers ---
  imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
      params_.imu_topic, rclcpp::SensorDataQoS().keep_last(200),
      [this](const sensor_msgs::msg::Imu::SharedPtr msg) {
        std::string child =
            params_.imu.use_parameter_frame ? params_.imu.parameter_frame : msg->header.frame_id;
        loadOrLookupTf(target_T_imu_tf_, child, params_.imu.use_parameter_tf,
                       params_.imu.parameter_tf.position, params_.imu.parameter_tf.orientation);
        {
          std::scoped_lock lock(tf_mutex_);
          if (target_T_imu_tf_.header.frame_id.empty()) {
            return;
          }
          imu_frame_ = child;
        }
        auto data = std::make_shared<utils::ImuData>();
        data->timestamp = rclcpp::Time(msg->header.stamp).seconds();
        data->linear_acceleration = toGtsam(msg->linear_acceleration);
        data->angular_velocity = toGtsam(msg->angular_velocity);
        data->linear_acceleration_covariance = toCovMatrix<3>(msg->linear_acceleration_covariance);
        data->angular_velocity_covariance = toCovMatrix<3>(msg->angular_velocity_covariance);
        imu_queue_.push(data);
      },
      sensor_options);

  if (params_.gps.enable_gps || params_.gps.enable_gps_init_only) {
    gps_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        params_.gps_odom_topic, rclcpp::SensorDataQoS(),
        [this](const nav_msgs::msg::Odometry::SharedPtr msg) {
          std::string child =
              params_.gps.use_parameter_frame ? params_.gps.parameter_frame : msg->child_frame_id;
          loadOrLookupTf(target_T_gps_tf_, child, params_.gps.use_parameter_tf,
                         params_.gps.parameter_tf.position, params_.gps.parameter_tf.orientation);
          auto data = std::make_shared<utils::OdometryData>();
          data->timestamp = rclcpp::Time(msg->header.stamp).seconds();
          data->pose = toGtsam(msg->pose.pose);
          data->pose_covariance = toCovMatrix<6>(msg->pose.covariance);
          gps_queue_.push(data);
        },
        sensor_options);
  }

  if (params_.depth.enable_depth || params_.depth.enable_depth_init_only) {
    depth_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        params_.depth_odom_topic, rclcpp::SensorDataQoS(),
        [this](const nav_msgs::msg::Odometry::SharedPtr msg) {
          std::string child = params_.depth.use_parameter_frame ? params_.depth.parameter_frame
                                                                : msg->child_frame_id;
          loadOrLookupTf(target_T_depth_tf_, child, params_.depth.use_parameter_tf,
                         params_.depth.parameter_tf.position,
                         params_.depth.parameter_tf.orientation);
          auto data = std::make_shared<utils::OdometryData>();
          data->timestamp = rclcpp::Time(msg->header.stamp).seconds();
          data->pose = toGtsam(msg->pose.pose);
          data->pose_covariance = toCovMatrix<6>(msg->pose.covariance);
          depth_queue_.push(data);

          if (keyframe_source_ == KeyframeSource::kDepth ||
              backup_keyframe_source_ == KeyframeSource::kDepth) {
            notifyFrontend();
          }
        },
        sensor_options);
  }

  if (params_.mag.enable_mag || params_.mag.enable_mag_init_only) {
    mag_sub_ = create_subscription<sensor_msgs::msg::MagneticField>(
        params_.mag_topic, rclcpp::SensorDataQoS(),
        [this](const sensor_msgs::msg::MagneticField::SharedPtr msg) {
          std::string child =
              params_.mag.use_parameter_frame ? params_.mag.parameter_frame : msg->header.frame_id;
          loadOrLookupTf(target_T_mag_tf_, child, params_.mag.use_parameter_tf,
                         params_.mag.parameter_tf.position, params_.mag.parameter_tf.orientation);
          auto data = std::make_shared<utils::MagneticFieldData>();
          data->timestamp = rclcpp::Time(msg->header.stamp).seconds();
          data->magnetic_field = toGtsam(msg->magnetic_field);
          data->magnetic_field_covariance = toCovMatrix<3>(msg->magnetic_field_covariance);
          mag_queue_.push(data);
        },
        sensor_options);
  }

  if (params_.ahrs.enable_ahrs || params_.ahrs.enable_ahrs_init_only ||
      params_.comparison.enable_loose_dvl_preintegration) {
    ahrs_sub_ = create_subscription<sensor_msgs::msg::Imu>(
        params_.ahrs_topic, rclcpp::SensorDataQoS().keep_last(200),
        [this](const sensor_msgs::msg::Imu::SharedPtr msg) {
          std::string child = params_.ahrs.use_parameter_frame ? params_.ahrs.parameter_frame
                                                               : msg->header.frame_id;
          loadOrLookupTf(target_T_ahrs_tf_, child, params_.ahrs.use_parameter_tf,
                         params_.ahrs.parameter_tf.position, params_.ahrs.parameter_tf.orientation);
          auto data = std::make_shared<utils::AhrsData>();
          data->timestamp = rclcpp::Time(msg->header.stamp).seconds();
          data->orientation = toGtsam(msg->orientation);
          data->orientation_covariance = toCovMatrix<3>(msg->orientation_covariance);
          ahrs_queue_.push(data);
        },
        sensor_options);
  }

  if (params_.dvl.enable_dvl || params_.dvl.enable_dvl_init_only) {
    dvl_sub_ = create_subscription<geometry_msgs::msg::TwistWithCovarianceStamped>(
        params_.dvl_topic, rclcpp::SensorDataQoS(),
        [this](const geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr msg) {
          std::string child =
              params_.dvl.use_parameter_frame ? params_.dvl.parameter_frame : msg->header.frame_id;
          loadOrLookupTf(target_T_dvl_tf_, child, params_.dvl.use_parameter_tf,
                         params_.dvl.parameter_tf.position, params_.dvl.parameter_tf.orientation);
          auto data = std::make_shared<utils::TwistData>();
          data->timestamp = rclcpp::Time(msg->header.stamp).seconds();
          data->linear_velocity = toGtsam(msg->twist.twist.linear);
          data->twist_covariance = toCovMatrix<6>(msg->twist.covariance);
          dvl_queue_.push(data);

          if (keyframe_source_ == KeyframeSource::kDvl ||
              backup_keyframe_source_ == KeyframeSource::kDvl) {
            notifyFrontend();
          }
        },
        sensor_options);
  }

  if (params_.dynamics.enable_dynamics || params_.dynamics.enable_dynamics_dropout_only) {
    wrench_sub_ = create_subscription<geometry_msgs::msg::WrenchStamped>(
        params_.wrench_topic, rclcpp::SensorDataQoS(),
        [this](const geometry_msgs::msg::WrenchStamped::SharedPtr msg) {
          std::string child = params_.dynamics.use_parameter_frame
                                  ? params_.dynamics.parameter_frame
                                  : msg->header.frame_id;
          loadOrLookupTf(target_T_com_tf_, child, params_.dynamics.use_parameter_tf,
                         params_.dynamics.parameter_tf.position,
                         params_.dynamics.parameter_tf.orientation);
          auto data = std::make_shared<utils::WrenchData>();
          data->timestamp = rclcpp::Time(msg->header.stamp).seconds();
          data->force = toGtsam(msg->wrench.force);
          data->torque = toGtsam(msg->wrench.torque);
          wrench_queue_.push(data);
        },
        sensor_options);
  }

  if (keyframe_source_ == KeyframeSource::kTimer ||
      backup_keyframe_source_ == KeyframeSource::kTimer) {
    double period = 1.0 / params_.keyframe_timer_hz;
    keyframe_timer_ =
        create_wall_timer(std::chrono::duration<double>(period), [this]() { notifyFrontend(); });
  }

  // --- ROS Diagnostics ---
  if (params_.publish_diagnostics) {
    std::string ns = this->get_namespace();
    std::string clean_ns = (ns == "/") ? "" : ns;
    diagnostic_updater_.setHardwareID(clean_ns + "/factor_graph_node");

    std::string prefix = clean_ns.empty() ? "" : "[" + clean_ns + "] ";

    std::string suffix;
    if (params_.comparison.enable_loose_dvl_preintegration) {
      suffix = " (FL-LPI)";
    } else if (params_.comparison.enable_tight_dvl_preintegration) {
      suffix = " (FL-TPI)";
    } else if (parseSolverType(params_.solver_type) == SolverType::kIsam2) {
      suffix = " (iS2-B)";
    } else {
      suffix = "";
    }

    std::string sensor_task = prefix + "Sensor Inputs" + suffix;
    diagnostic_updater_.add(sensor_task, this, &FactorGraphNode::checkSensorInputs);

    std::string state_task = prefix + "Graph State" + suffix;
    diagnostic_updater_.add(state_task, this, &FactorGraphNode::checkGraphState);

    std::string overflow_task = prefix + "Processing Overflow" + suffix;
    diagnostic_updater_.add(overflow_task, this, &FactorGraphNode::checkProcessingOverflow);
  }
}

FactorGraphNode::FactorGraphNode(const rclcpp::NodeOptions& options)
    : Node("factor_graph_node", options), diagnostic_updater_(this) {
  RCLCPP_INFO(get_logger(), "Starting Factor Graph Node...");

  param_listener_ =
      std::make_shared<factor_graph_node::ParamListener>(get_node_parameters_interface());
  params_ = param_listener_->get_params();

  keyframe_source_ = parseKeyframeSource(params_.keyframe_source);
  backup_keyframe_source_ = parseKeyframeSource(params_.backup_keyframe_source);

  // Ensure the keyframe sources are valid
  auto source_enabled = [this](KeyframeSource source) {
    switch (source) {
      case KeyframeSource::kDvl:
        return params_.dvl.enable_dvl || params_.dvl.enable_dvl_init_only;
      case KeyframeSource::kDepth:
        return params_.depth.enable_depth || params_.depth.enable_depth_init_only;
      default:
        return true;
    }
  };
  if (!source_enabled(keyframe_source_) || !source_enabled(backup_keyframe_source_)) {
    RCLCPP_FATAL(get_logger(),
                 "Keyframe source '%s' or backup '%s' references a disabled sensor! Shutting down.",
                 params_.keyframe_source.c_str(), params_.backup_keyframe_source.c_str());
    throw std::runtime_error("Invalid keyframe source configuration.");
  }

  setupRosInterfaces();
  core_ = std::make_unique<FactorGraphCore>(params_);
  state_init_ = std::make_unique<StateInitializer>(params_);
  frontend_thread_ = std::thread(&FactorGraphNode::frontendThreadLoop, this);
  backend_thread_ = std::thread(&FactorGraphNode::backendThreadLoop, this);

  RCLCPP_INFO(get_logger(), "Startup complete! Waiting for sensor data...");
}

FactorGraphNode::~FactorGraphNode() {
  is_running_.store(false);
  notifyFrontend();
  notifyBackend();
  if (frontend_thread_.joinable()) {
    frontend_thread_.join();
  }
  if (backend_thread_.joinable()) {
    backend_thread_.join();
  }
}

void FactorGraphNode::notifyFrontend() {
  {
    std::scoped_lock lock(frontend_trigger_mutex_);
    frontend_trigger_ = true;
  }
  frontend_cv_.notify_one();
}

void FactorGraphNode::notifyBackend() {
  {
    std::scoped_lock lock(backend_trigger_mutex_);
    backend_trigger_ = true;
  }
  backend_cv_.notify_one();
}

bool FactorGraphNode::checkAndUpdateRateLimit(rclcpp::Time& last_time, double max_rate_hz) {
  if (max_rate_hz <= 0.0) {
    return true;
  }
  rclcpp::Time now = get_clock()->now();
  if (now - last_time < rclcpp::Duration::from_seconds(1.0 / max_rate_hz)) {
    return false;
  }
  last_time = now;
  return true;
}

void FactorGraphNode::loadOrLookupTf(geometry_msgs::msg::TransformStamped& tf_out,
                                     const std::string& child, bool use_parameter_tf,
                                     const std::vector<double>& pos,
                                     const std::vector<double>& quat) {
  std::scoped_lock lock(tf_mutex_);
  if (!tf_out.header.frame_id.empty()) {
    return;
  }

  if (use_parameter_tf) {
    tf_out.header.stamp = get_clock()->now();
    tf_out.header.frame_id = params_.target_frame;
    tf_out.child_frame_id = child;
    tf_out.transform.translation.x = pos[0];
    tf_out.transform.translation.y = pos[1];
    tf_out.transform.translation.z = pos[2];
    tf_out.transform.rotation.x = quat[0];
    tf_out.transform.rotation.y = quat[1];
    tf_out.transform.rotation.z = quat[2];
    tf_out.transform.rotation.w = quat[3];
  } else {
    try {
      if (!params_.target_frame.empty()) {
        tf_out = tf_buffer_->lookupTransform(params_.target_frame, child, tf2::TimePointZero);
      }
    } catch (const tf2::TransformException& ex) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "Could not transform %s to %s: %s",
                           params_.target_frame.c_str(), child.c_str(), ex.what());
    }
  }
}

utils::QueueBundle FactorGraphNode::drainAllQueues() {
  utils::QueueBundle queues;
  queues.imu = imu_queue_.drain();
  queues.gps = gps_queue_.drain();
  queues.depth = depth_queue_.drain();
  queues.mag = mag_queue_.drain();
  queues.ahrs = ahrs_queue_.drain();
  queues.dvl = dvl_queue_.drain();
  queues.wrench = wrench_queue_.drain();
  return queues;
}

void FactorGraphNode::restoreAllQueues(const utils::QueueBundle& queues) {
  imu_queue_.restore(queues.imu);
  gps_queue_.restore(queues.gps);
  depth_queue_.restore(queues.depth);
  mag_queue_.restore(queues.mag);
  ahrs_queue_.restore(queues.ahrs);
  dvl_queue_.restore(queues.dvl);
  wrench_queue_.restore(queues.wrench);
}

void FactorGraphNode::publishGlobalOdom(const gtsam::Pose3& current_pose,
                                        const gtsam::Matrix& pose_covariance,
                                        const rclcpp::Time& timestamp) {
  gtsam::Pose3 target_T_base;
  {
    std::scoped_lock lock(tf_mutex_);
    target_T_base = toGtsam(target_T_base_tf_.transform);
  }
  gtsam::Pose3 map_T_base = current_pose * target_T_base;

  nav_msgs::msg::Odometry odom_msg;
  odom_msg.header.stamp = timestamp;
  odom_msg.header.frame_id = params_.map_frame;
  odom_msg.child_frame_id = params_.base_frame;
  odom_msg.pose.pose = toPoseMsg(map_T_base);

  gtsam::Matrix cov_to_pub = pose_covariance;

  if (params_.publish_pose_cov) {
    gtsam::Rot3 map_R_base = map_T_base.rotation();
    gtsam::Matrix66 Rot = gtsam::Matrix66::Zero();
    Rot.block<3, 3>(0, 0) = map_R_base.matrix();
    Rot.block<3, 3>(3, 3) = map_R_base.matrix();

    gtsam::Matrix66 warped_covariance = target_T_base.inverse().AdjointMap() * pose_covariance *
                                        target_T_base.inverse().AdjointMap().transpose();
    cov_to_pub = Rot * warped_covariance * Rot.transpose();
  }

  odom_msg.pose.covariance = toPoseCovarianceMsg(gtsam::Matrix66(cov_to_pub));
  odom_msg.twist.covariance[0] = -1.0;
  global_odom_pub_->publish(odom_msg);
}

void FactorGraphNode::broadcastGlobalTf(const gtsam::Pose3& current_pose,
                                        const rclcpp::Time& timestamp) {
  try {
    gtsam::Pose3 target_T_base;
    {
      std::scoped_lock lock(tf_mutex_);
      target_T_base = toGtsam(target_T_base_tf_.transform);
    }
    gtsam::Pose3 map_T_base = current_pose * target_T_base;

    gtsam::Pose3 odom_T_base = toGtsam(
        tf_buffer_->lookupTransform(params_.odom_frame, params_.base_frame, tf2::TimePointZero)
            .transform);
    gtsam::Pose3 map_T_odom = map_T_base * odom_T_base.inverse();

    geometry_msgs::msg::TransformStamped tf_msg;
    tf_msg.header.stamp = timestamp;
    tf_msg.header.frame_id = params_.map_frame;
    tf_msg.child_frame_id = params_.odom_frame;
    tf_msg.transform.translation = toVectorMsg(map_T_odom.translation());
    tf_msg.transform.rotation = toQuatMsg(map_T_odom.rotation());
    tf_broadcaster_->sendTransform(tf_msg);
  } catch (const tf2::TransformException& ex) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "Could not transform %s to %s: %s",
                         params_.odom_frame.c_str(), params_.base_frame.c_str(), ex.what());
  }
}

void FactorGraphNode::publishSmoothedPath(const gtsam::Values& values,
                                          const rclcpp::Time& timestamp) {
  nav_msgs::msg::Path path_msg;
  path_msg.header.stamp = timestamp;
  path_msg.header.frame_id = params_.map_frame;

  gtsam::Pose3 target_T_base;
  {
    std::scoped_lock lock(tf_mutex_);
    target_T_base = toGtsam(target_T_base_tf_.transform);
  }

  auto keys_snapshot = core_->snapshotTimeKeys();

  for (const auto& pair : keys_snapshot) {
    if (values.exists(pair.second)) {
      geometry_msgs::msg::PoseStamped ps;
      ps.header.frame_id = params_.map_frame;
      ps.header.stamp = rclcpp::Time(pair.first);
      ps.pose = toPoseMsg(values.at<gtsam::Pose3>(pair.second) * target_T_base);
      path_msg.poses.push_back(ps);
    }
  }
  smoothed_path_pub_->publish(path_msg);
}

void FactorGraphNode::publishVelocity(const gtsam::Vector3& current_vel,
                                      const gtsam::Matrix& vel_covariance,
                                      const rclcpp::Time& timestamp) {
  geometry_msgs::msg::TwistWithCovarianceStamped vel_msg;
  vel_msg.header.stamp = timestamp;

  // IMPORTANT! This is the velocity of the target frame with respect to the map frame.
  vel_msg.header.frame_id = params_.map_frame;
  vel_msg.twist.twist.linear = toVectorMsg(current_vel);
  vel_msg.twist.covariance = toCovariance36Msg(gtsam::Matrix33(vel_covariance));

  for (int i = 3; i < 6; ++i) {
    vel_msg.twist.covariance[i * 6 + i] = -1.0;
  }
  velocity_pub_->publish(vel_msg);
}

void FactorGraphNode::publishImuBias(const gtsam::imuBias::ConstantBias& current_imu_bias,
                                     const gtsam::Matrix& imu_bias_covariance,
                                     const rclcpp::Time& timestamp) {
  geometry_msgs::msg::TwistWithCovarianceStamped imu_bias_msg;
  imu_bias_msg.header.stamp = timestamp;
  {
    std::scoped_lock lock(tf_mutex_);
    imu_bias_msg.header.frame_id = imu_frame_;
  }

  // IMPORTANT! We use 'linear' for accelerometer bias and 'angular' for gyroscope bias.
  imu_bias_msg.twist.twist.linear = toVectorMsg(current_imu_bias.accelerometer());
  imu_bias_msg.twist.twist.angular = toVectorMsg(current_imu_bias.gyroscope());
  imu_bias_msg.twist.covariance = toCovariance36Msg(gtsam::Matrix66(imu_bias_covariance));

  imu_bias_pub_->publish(imu_bias_msg);
}

void FactorGraphNode::publishGraphMetrics(const rclcpp::Time& timestamp) {
  coug_interfaces::msg::GraphMetrics metrics_msg;
  metrics_msg.header.stamp = timestamp;

  metrics_msg.total_duration = last_total_duration_.load();
  metrics_msg.smoother_duration = last_smoother_duration_.load();
  metrics_msg.cov_duration = last_cov_duration_.load();
  metrics_msg.new_factors = static_cast<uint32_t>(new_factors_.load());
  metrics_msg.total_factors = static_cast<uint32_t>(total_factors_.load());
  metrics_msg.total_variables = static_cast<uint32_t>(total_variables_.load());

  graph_metrics_pub_->publish(metrics_msg);
}

void FactorGraphNode::frontendThreadLoop() {
  while (is_running_.load()) {
    std::unique_lock<std::mutex> lock(frontend_trigger_mutex_);
    frontend_cv_.wait(lock, [this] { return frontend_trigger_ || !is_running_.load(); });
    frontend_trigger_ = false;

    if (!is_running_.load()) {
      break;
    }

    lock.unlock();
    {
      std::shared_lock reset_lock(reset_mutex_);

      if (!is_initialized_.load()) {
        initializeGraph();
      } else if (checkAndUpdateRateLimit(last_update_time_, params_.max_update_rate_hz)) {
        updateGraph();
        notifyBackend();
      }
    }
  }
}

void FactorGraphNode::backendThreadLoop() {
  while (is_running_.load()) {
    std::unique_lock<std::mutex> lock(backend_trigger_mutex_);
    backend_cv_.wait(lock, [this] { return backend_trigger_ || !is_running_.load(); });
    backend_trigger_ = false;

    if (!is_running_.load()) {
      break;
    }

    if (is_initialized_.load()) {
      lock.unlock();

      std::shared_lock reset_lock(reset_mutex_);
      if (!is_initialized_.load()) {
        continue;
      }

      if (checkAndUpdateRateLimit(last_opt_time_, params_.max_opt_rate_hz)) {
        optimizeGraph();
      }
    }
  }
}

void FactorGraphNode::initializeGraph() {
  // --- Wait for Sensor TFs ---
  bool imu_ok, gps_ok, depth_ok, mag_ok, ahrs_ok, dvl_ok;
  {
    std::scoped_lock lock(tf_mutex_);
    imu_ok = !target_T_imu_tf_.header.frame_id.empty();
    gps_ok = !(params_.gps.enable_gps || params_.gps.enable_gps_init_only) ||
             !target_T_gps_tf_.header.frame_id.empty();
    depth_ok = !(params_.depth.enable_depth || params_.depth.enable_depth_init_only) ||
               !target_T_depth_tf_.header.frame_id.empty();
    mag_ok = !(params_.mag.enable_mag || params_.mag.enable_mag_init_only) ||
             !target_T_mag_tf_.header.frame_id.empty();
    ahrs_ok = !(params_.ahrs.enable_ahrs || params_.ahrs.enable_ahrs_init_only ||
                params_.comparison.enable_loose_dvl_preintegration) ||
              !target_T_ahrs_tf_.header.frame_id.empty();
    dvl_ok = !(params_.dvl.enable_dvl || params_.dvl.enable_dvl_init_only) ||
             !target_T_dvl_tf_.header.frame_id.empty();
  }

  if (!(imu_ok && gps_ok && depth_ok && mag_ok && ahrs_ok && dvl_ok)) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "Waiting for sensors: %s%s%s%s%s%s",
                         !imu_ok ? "[IMU] " : "",
                         (!gps_ok && params_.gps.enable_gps) ? "[GPS] " : "",
                         (!mag_ok && params_.mag.enable_mag) ? "[Mag] " : "",
                         (!ahrs_ok && params_.ahrs.enable_ahrs) ? "[AHRS] " : "",
                         (!depth_ok && params_.depth.enable_depth) ? "[Depth] " : "",
                         (!dvl_ok && params_.dvl.enable_dvl) ? "[DVL] " : "");
    return;
  }

  loadOrLookupTf(target_T_base_tf_, params_.base_frame, params_.base.use_parameter_tf,
                 params_.base.parameter_tf.position, params_.base.parameter_tf.orientation);
  {
    std::scoped_lock lock(tf_mutex_);
    if (target_T_base_tf_.header.frame_id.empty()) {
      return;
    }
  }

  // --- Compute Initial State ---
  utils::QueueBundle init_queues = drainAllQueues();

  auto back_time = [](const auto& q) { return q.empty() ? 0.0 : q.back()->timestamp; };
  double newest_stamp = std::max({back_time(init_queues.imu), back_time(init_queues.gps),
                                  back_time(init_queues.depth), back_time(init_queues.mag),
                                  back_time(init_queues.ahrs), back_time(init_queues.dvl),
                                  back_time(init_queues.wrench)});
  if (newest_stamp <= 0.0 || !state_init_->update(newest_stamp, init_queues)) {
    return;
  }

  utils::TfBundle tfs;
  {
    std::scoped_lock lock(tf_mutex_);
    tfs = {toGtsam(target_T_imu_tf_.transform),   toGtsam(target_T_gps_tf_.transform),
           toGtsam(target_T_depth_tf_.transform), toGtsam(target_T_mag_tf_.transform),
           toGtsam(target_T_ahrs_tf_.transform),  toGtsam(target_T_dvl_tf_.transform),
           toGtsam(target_T_base_tf_.transform),  toGtsam(target_T_com_tf_.transform)};
  }
  state_init_->compute(tfs);
  core_->initialize(*state_init_, tfs);

  is_initialized_.store(true);
  RCLCPP_INFO(get_logger(), "Graph initialized successfully!");
  // TODO: Print priors
}

void FactorGraphNode::updateGraph() {
  KeyframeSource active_source = keyframe_source_;
  if (active_source != KeyframeSource::kTimer) {
    std::optional<double> last_received = (active_source == KeyframeSource::kDvl)
                                              ? dvl_queue_.getLastTime()
                                              : depth_queue_.getLastTime();

    std::optional<double> newest_stamp = imu_queue_.getLastTime();
    if (!last_received.has_value() ||
        (newest_stamp.has_value() &&
         (*newest_stamp - *last_received) > params_.keyframe_timeout_sec)) {
      if (backup_keyframe_source_ != KeyframeSource::kNone) {
        active_source = backup_keyframe_source_;
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                             "Primary keyframe source '%s' timed out! Using backup '%s'.",
                             params_.keyframe_source.c_str(),
                             params_.backup_keyframe_source.c_str());
      }
    }
  }

  // Extract target time
  std::optional<double> target_time;
  if (active_source == KeyframeSource::kDvl && !dvl_queue_.empty()) {
    target_time = dvl_queue_.getLastTime();
  } else if (active_source == KeyframeSource::kDepth && !depth_queue_.empty()) {
    target_time = depth_queue_.getLastTime();
  } else if (active_source == KeyframeSource::kTimer && !imu_queue_.empty()) {
    target_time = imu_queue_.getLastTime();
  }

  if (!target_time.has_value() ||
      (last_target_time_.has_value() && *target_time <= *last_target_time_)) {
    return;
  }
  last_target_time_ = *target_time;

  // --- Update Request ---
  utils::QueueBundle queues = drainAllQueues();
  auto leftover = core_->update(*target_time, queues);
  restoreAllQueues(leftover ? *leftover : queues);
}

void FactorGraphNode::optimizeGraph() {
  // --- Optimization Request ---
  try {
    auto result = core_->optimize();
    if (!result) {
      return;
    }

    // --- Update Diagnostics State ---
    last_total_duration_.store(result->total_duration);
    last_smoother_duration_.store(result->smoother_duration);
    last_cov_duration_.store(result->cov_duration);
    new_factors_.store(result->new_factors);
    total_factors_.store(result->total_factors);
    total_variables_.store(result->total_variables);
    processing_overflow_.store(result->processing_overflow);

    if (result->processing_overflow) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                           "Processing overflow. Batching %zu keyframes.", result->num_keyframes);
    }

    // --- Publish Results ---
    const rclcpp::Time stamp(static_cast<int64_t>(result->target_time * 1e9));
    publishGlobalOdom(result->pose, result->pose_cov, stamp);

    if (params_.publish_global_tf) {
      broadcastGlobalTf(result->pose, stamp);
    }

    if (params_.publish_smoothed_path) {
      publishSmoothedPath(result->all_estimates, stamp);
    }

    if (params_.publish_velocity) {
      publishVelocity(result->velocity, result->vel_cov, stamp);
    }

    if (params_.publish_imu_bias) {
      publishImuBias(result->imu_bias, result->bias_cov, stamp);
    }

    if (params_.publish_graph_metrics) {
      publishGraphMetrics(stamp);
    }
  } catch (const std::exception& e) {
    RCLCPP_FATAL(get_logger(), "%s", e.what());
    rclcpp::shutdown();
  }
}

void FactorGraphNode::checkSensorInputs(diagnostic_updater::DiagnosticStatusWrapper& stat) {
  bool any_critical_offline = false;
  std::vector<std::string> offline_sensors;

  auto check_queue = [&](const std::string& name, size_t size, std::optional<double> since_arrival,
                         bool enabled, bool is_critical, double timeout) {
    if (!enabled) {
      return;
    }

    double time_since = since_arrival.value_or(-1.0);

    stat.add(name + " Queue Size", size);
    stat.add(name + " Time Since Last (s)", time_since);

    if (time_since > timeout || (!since_arrival.has_value() && size == 0)) {
      offline_sensors.push_back(name + " is offline.");
      if (is_critical) {
        any_critical_offline = true;
      }
    }
  };

  check_queue("IMU", imu_queue_.size(), imu_queue_.secondsSinceLastArrival(), true, true,
              params_.imu.diagnostic_timeout_sec);
  check_queue("GPS", gps_queue_.size(), gps_queue_.secondsSinceLastArrival(),
              params_.gps.enable_gps, false, params_.gps.diagnostic_timeout_sec);
  check_queue("Depth", depth_queue_.size(), depth_queue_.secondsSinceLastArrival(),
              params_.depth.enable_depth, params_.depth.enable_depth,
              params_.depth.diagnostic_timeout_sec);
  check_queue("Mag", mag_queue_.size(), mag_queue_.secondsSinceLastArrival(),
              params_.mag.enable_mag, false, params_.mag.diagnostic_timeout_sec);
  check_queue("AHRS", ahrs_queue_.size(), ahrs_queue_.secondsSinceLastArrival(),
              params_.ahrs.enable_ahrs, false, params_.ahrs.diagnostic_timeout_sec);
  check_queue("DVL", dvl_queue_.size(), dvl_queue_.secondsSinceLastArrival(),
              params_.dvl.enable_dvl, params_.dvl.enable_dvl, params_.dvl.diagnostic_timeout_sec);
  check_queue("Wrench", wrench_queue_.size(), wrench_queue_.secondsSinceLastArrival(),
              params_.dynamics.enable_dynamics, false, params_.dynamics.diagnostic_timeout_sec);

  if (!offline_sensors.empty()) {
    std::string msg = "";
    for (size_t i = 0; i < offline_sensors.size(); ++i) {
      msg += (i > 0 ? " " : "") + offline_sensors[i];
    }
    auto level = any_critical_offline ? diagnostic_msgs::msg::DiagnosticStatus::ERROR
                                      : diagnostic_msgs::msg::DiagnosticStatus::WARN;
    stat.summary(level, msg);
  } else {
    stat.summary(diagnostic_msgs::msg::DiagnosticStatus::OK, "All requested sensors online.");
  }
}

void FactorGraphNode::checkGraphState(diagnostic_updater::DiagnosticStatusWrapper& stat) {
  if (is_initialized_.load()) {
    stat.summary(diagnostic_msgs::msg::DiagnosticStatus::OK, "Optimizing factor graph.");
  } else {
    stat.summary(diagnostic_msgs::msg::DiagnosticStatus::WARN, "Waiting for sensor data.");
  }
}

void FactorGraphNode::checkProcessingOverflow(diagnostic_updater::DiagnosticStatusWrapper& stat) {
  if (processing_overflow_.load()) {
    stat.summary(diagnostic_msgs::msg::DiagnosticStatus::WARN,
                 "Processing overflow detected. Batching keyframes.");
  } else {
    stat.summary(diagnostic_msgs::msg::DiagnosticStatus::OK, "No processing overflow detected.");
  }
  stat.add("Total Duration (s)", last_total_duration_.load());
  stat.add("Smoother Duration (s)", last_smoother_duration_.load());
  stat.add("Covariance Duration (s)", last_cov_duration_.load());

  stat.add("New Factors", new_factors_.load());
  stat.add("Total Factors", total_factors_.load());
  stat.add("Total Variables", total_variables_.load());
}

void FactorGraphNode::resetGraph(const std_srvs::srv::Trigger::Request::SharedPtr,
                                 std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
  RCLCPP_WARN(get_logger(), "Reset requested!");

  std::unique_lock reset_lock(reset_mutex_);

  // Discard data and reset estimator state
  drainAllQueues();

  core_ = std::make_unique<FactorGraphCore>(params_);
  state_init_ = std::make_unique<StateInitializer>(params_);

  is_initialized_.store(false);
  last_target_time_.reset();
  last_update_time_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
  last_opt_time_ = rclcpp::Time(0, 0, RCL_ROS_TIME);

  last_total_duration_.store(0.0);
  last_smoother_duration_.store(0.0);
  last_cov_duration_.store(0.0);
  processing_overflow_.store(false);
  new_factors_.store(0);
  total_factors_.store(0);
  total_variables_.store(0);

  RCLCPP_INFO(get_logger(), "Graph reset successfully. Waiting for sensor data...");
  response->success = true;
  response->message = "Factor graph reset successfully.";
}

}  // namespace coug_fgo

RCLCPP_COMPONENTS_REGISTER_NODE(coug_fgo::FactorGraphNode)
