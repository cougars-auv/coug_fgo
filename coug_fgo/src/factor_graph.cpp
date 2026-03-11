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
 * @date Jan 2026
 */

#include "coug_fgo/factor_graph.hpp"

#include <rclcpp_components/register_node_macro.hpp>

#include "coug_fgo/utils/conversions.hpp"

using coug_fgo::utils::toGtsam;
using coug_fgo::utils::toQuatMsg;
using coug_fgo::utils::toVectorMsg;
using coug_fgo::utils::toPoseMsg;
using coug_fgo::utils::toCovariance36Msg;
using coug_fgo::utils::toPoseCovarianceMsg;

namespace coug_fgo
{

void FactorGraphNode::setupRosInterfaces()
{
  // --- ROS TF Interfaces ---
  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_, this);
  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

  // --- ROS Publishers ---
  global_odom_pub_ = create_publisher<nav_msgs::msg::Odometry>(
    params_.global_odom_topic,
    rclcpp::SystemDefaultsQoS());
  if (params_.publish_smoothed_path) {
    smoothed_path_pub_ = create_publisher<nav_msgs::msg::Path>(
      params_.smoothed_path_topic,
      rclcpp::SystemDefaultsQoS());
  }
  if (params_.publish_velocity) {
    velocity_pub_ =
      create_publisher<geometry_msgs::msg::TwistWithCovarianceStamped>(
      params_.velocity_topic,
      rclcpp::SystemDefaultsQoS());
  }
  if (params_.publish_imu_bias) {
    imu_bias_pub_ =
      create_publisher<geometry_msgs::msg::TwistWithCovarianceStamped>(
      params_.imu_bias_topic,
      rclcpp::SystemDefaultsQoS());
  }
  if (params_.publish_graph_metrics) {
    graph_metrics_pub_ =
      create_publisher<coug_fgo_msgs::msg::GraphMetrics>(
      params_.graph_metrics_topic,
      rclcpp::SystemDefaultsQoS());
  }

  // --- ROS Callback Groups ---
  sensor_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
  auto sensor_options = rclcpp::SubscriptionOptions();
  sensor_options.callback_group = sensor_cb_group_;

  auto try_lookup_tf =
    [this](geometry_msgs::msg::TransformStamped & tf_out,
      const std::string & child, const std::string & sensor_name) {
      if (!tf_out.header.frame_id.empty()) {return;}
      try {
        if (!params_.target_frame.empty()) {
          tf_out = tf_buffer_->lookupTransform(params_.target_frame, child, tf2::TimePointZero);
        }
      } catch (const tf2::TransformException & ex) {
        RCLCPP_ERROR(
          get_logger(), "Failed to lookup %s to target transform: %s",
          sensor_name.c_str(), ex.what());
      }
    };

  // --- ROS Subscribers ---
  imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
    params_.imu_topic, rclcpp::SensorDataQoS().keep_last(200),
    [this, try_lookup_tf](const sensor_msgs::msg::Imu::SharedPtr msg) {
      std::string child = params_.imu.use_parameter_frame ?
      params_.imu.parameter_frame : msg->header.frame_id;
      try_lookup_tf(target_T_imu_tf_, child, "IMU");
      if (target_T_imu_tf_.header.frame_id.empty()) {return;}
      imu_frame_ = child;
      imu_queue_.push(msg);
    },
    sensor_options);

  if (params_.gps.enable_gps || params_.gps.enable_gps_init_only) {
    gps_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      params_.gps_odom_topic, rclcpp::SensorDataQoS(),
      [this, try_lookup_tf](const nav_msgs::msg::Odometry::SharedPtr msg) {
        std::string child = params_.gps.use_parameter_frame ?
        params_.gps.parameter_frame : msg->child_frame_id;
        try_lookup_tf(target_T_gps_tf_, child, "GPS");
        gps_queue_.push(msg);
      },
      sensor_options);
  }

  depth_sub_ = create_subscription<nav_msgs::msg::Odometry>(
    params_.depth_odom_topic, rclcpp::SensorDataQoS(),
    [this, try_lookup_tf](const nav_msgs::msg::Odometry::SharedPtr msg) {
      std::string child = params_.depth.use_parameter_frame ?
      params_.depth.parameter_frame : msg->child_frame_id;
      try_lookup_tf(target_T_depth_tf_, child, "depth");
      depth_queue_.push(msg);

      if (params_.experimental.enable_loose_dvl_preintegration) {
        {
          std::scoped_lock lock(frontend_trigger_mutex_);
          frontend_trigger_ = true;
        }
        frontend_cv_.notify_one();
      } else {
        double time_since_dvl = (get_clock()->now() - dvl_queue_.getLastTime()).seconds();
        if (state_.load() == State::RUNNING && time_since_dvl > params_.dvl.timeout_threshold) {
          RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 5000,
            "DVL timed out (%.2fs)! Using depth sensor to trigger keyframes.",
            time_since_dvl);
          {
            std::scoped_lock lock(frontend_trigger_mutex_);
            frontend_trigger_ = true;
          }
          frontend_cv_.notify_one();
        }
      }
    },
    sensor_options);

  if (params_.mag.enable_mag || params_.mag.enable_mag_init_only) {
    mag_sub_ = create_subscription<sensor_msgs::msg::MagneticField>(
      params_.mag_topic, rclcpp::SensorDataQoS(),
      [this, try_lookup_tf](const sensor_msgs::msg::MagneticField::SharedPtr msg) {
        std::string child = params_.mag.use_parameter_frame ?
        params_.mag.parameter_frame : msg->header.frame_id;
        try_lookup_tf(target_T_mag_tf_, child, "mag");
        mag_queue_.push(msg);
      },
      sensor_options);
  }

  if (params_.ahrs.enable_ahrs || params_.ahrs.enable_ahrs_init_only ||
    params_.experimental.enable_loose_dvl_preintegration)
  {
    ahrs_sub_ = create_subscription<sensor_msgs::msg::Imu>(
      params_.ahrs_topic, rclcpp::SensorDataQoS(),
      [this, try_lookup_tf](const sensor_msgs::msg::Imu::SharedPtr msg) {
        std::string child = params_.ahrs.use_parameter_frame ?
        params_.ahrs.parameter_frame : msg->header.frame_id;
        try_lookup_tf(target_T_ahrs_tf_, child, "AHRS");
        ahrs_queue_.push(msg);
      },
      sensor_options);
  }

  dvl_sub_ = create_subscription<geometry_msgs::msg::TwistWithCovarianceStamped>(
    params_.dvl_topic, rclcpp::SensorDataQoS(),
    [this, try_lookup_tf](const geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr msg) {
      std::string child = params_.dvl.use_parameter_frame ?
      params_.dvl.parameter_frame : msg->header.frame_id;
      try_lookup_tf(target_T_dvl_tf_, child, "DVL");
      dvl_queue_.push(msg);

      if (!params_.experimental.enable_loose_dvl_preintegration) {
        {
          std::scoped_lock lock(frontend_trigger_mutex_);
          frontend_trigger_ = true;
        }
        frontend_cv_.notify_one();
      }
    },
    sensor_options);

  if (params_.dynamics.enable_dynamics || params_.dynamics.enable_dynamics_dropout_only) {
    wrench_sub_ = create_subscription<geometry_msgs::msg::WrenchStamped>(
      params_.wrench_topic, rclcpp::SensorDataQoS(),
      [this, try_lookup_tf](const geometry_msgs::msg::WrenchStamped::SharedPtr msg) {
        std::string child = params_.dynamics.use_parameter_frame ?
        params_.dynamics.parameter_frame : msg->header.frame_id;
        try_lookup_tf(target_T_com_tf_, child, "COM");
        wrench_queue_.push(msg);
      },
      sensor_options);
  }

  // --- ROS Diagnostics ---
  if (params_.publish_diagnostics) {
    std::string ns = this->get_namespace();
    std::string clean_ns = (ns == "/") ? "" : ns;
    diagnostic_updater_.setHardwareID(clean_ns + "/factor_graph_node");

    std::string prefix = clean_ns.empty() ? "" : "[" + clean_ns + "] ";

    std::string suffix;
    if (params_.experimental.enable_loose_dvl_preintegration) {
      suffix = " (FL-LPI)";
    } else if (params_.solver_type == "ISAM2") {
      suffix = " (iSAM2-B)";
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

FactorGraphNode::FactorGraphNode(const rclcpp::NodeOptions & options)
: Node("factor_graph_node", options),
  diagnostic_updater_(this)
{
  RCLCPP_INFO(get_logger(), "Starting Factor Graph Node...");

  param_listener_ = std::make_shared<factor_graph_node::ParamListener>(
    get_node_parameters_interface());
  params_ = param_listener_->get_params();

  setupRosInterfaces();
  core_ = std::make_unique<FactorGraphCore>(params_);
  state_initializer_ = std::make_unique<utils::StateInitializer>(params_);
  frontend_thread_ = std::thread(&FactorGraphNode::processFrontend, this);
  backend_thread_ = std::thread(&FactorGraphNode::processBackend, this);

  RCLCPP_INFO(get_logger(), "Startup complete! Waiting for sensor messages...");
}

FactorGraphNode::~FactorGraphNode()
{
  is_running_.store(false);
  {
    std::scoped_lock lock(frontend_trigger_mutex_);
    frontend_trigger_ = true;
  }
  frontend_cv_.notify_all();
  {
    std::scoped_lock lock(backend_trigger_mutex_);
    backend_trigger_ = true;
  }
  backend_cv_.notify_all();
  if (frontend_thread_.joinable()) {
    frontend_thread_.join();
  }
  if (backend_thread_.joinable()) {
    backend_thread_.join();
  }
}

void FactorGraphNode::publishGlobalOdom(
  const gtsam::Pose3 & current_pose,
  const gtsam::Matrix & pose_covariance,
  const rclcpp::Time & timestamp)
{
  gtsam::Pose3 target_T_base = toGtsam(target_T_base_tf_.transform);
  gtsam::Pose3 pose_base = current_pose.compose(target_T_base);

  nav_msgs::msg::Odometry odom_msg;
  odom_msg.header.stamp = timestamp;
  odom_msg.header.frame_id = params_.map_frame;
  odom_msg.child_frame_id = params_.base_frame;
  odom_msg.pose.pose = toPoseMsg(pose_base);

  gtsam::Matrix cov_to_pub = pose_covariance;

  if (params_.publish_pose_cov) {
    gtsam::Rot3 R_map_base = pose_base.rotation();
    gtsam::Matrix66 Rot = gtsam::Matrix66::Zero();
    Rot.block<3, 3>(0, 0) = R_map_base.matrix();
    Rot.block<3, 3>(3, 3) = R_map_base.matrix();

    gtsam::Matrix66 warped_covariance = target_T_base.inverse().AdjointMap() * pose_covariance *
      target_T_base.inverse().AdjointMap().transpose();
    cov_to_pub = Rot * warped_covariance * Rot.transpose();
  }

  odom_msg.pose.covariance = toPoseCovarianceMsg(gtsam::Matrix66(cov_to_pub));
  odom_msg.twist.covariance[0] = -1.0;
  global_odom_pub_->publish(odom_msg);
}

void FactorGraphNode::broadcastGlobalTf(
  const gtsam::Pose3 & current_pose,
  const rclcpp::Time & timestamp)
{
  try {
    gtsam::Pose3 target_T_base = toGtsam(target_T_base_tf_.transform);
    gtsam::Pose3 pose_base = current_pose * target_T_base;

    gtsam::Pose3 odom_T_base = toGtsam(
      tf_buffer_->lookupTransform(
        params_.odom_frame, params_.base_frame,
        tf2::TimePointZero).transform);
    gtsam::Pose3 map_T_odom = pose_base * odom_T_base.inverse();

    geometry_msgs::msg::TransformStamped tf_msg;
    tf_msg.header.stamp = timestamp;
    tf_msg.header.frame_id = params_.map_frame;
    tf_msg.child_frame_id = params_.odom_frame;
    tf_msg.transform.translation = toVectorMsg(map_T_odom.translation());
    tf_msg.transform.rotation = toQuatMsg(map_T_odom.rotation());
    tf_broadcaster_->sendTransform(tf_msg);
  } catch (const tf2::TransformException & ex) {
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 5000, "Global TF lookup failed: %s",
      ex.what());
  }
}

void FactorGraphNode::publishSmoothedPath(
  const gtsam::Values & results,
  const rclcpp::Time & timestamp)
{
  nav_msgs::msg::Path path_msg;
  path_msg.header.stamp = timestamp;
  path_msg.header.frame_id = params_.map_frame;

  gtsam::Pose3 target_T_base = toGtsam(target_T_base_tf_.transform);

  std::map<rclcpp::Time, gtsam::Key> keys_snapshot;
  {
    std::scoped_lock lock(core_->buffer_mutex);
    keys_snapshot = core_->time_to_key;
  }

  for (const auto & pair : keys_snapshot) {
    if (results.exists(pair.second)) {
      geometry_msgs::msg::PoseStamped ps;
      ps.header.frame_id = params_.map_frame;
      ps.header.stamp = pair.first;
      ps.pose = toPoseMsg(results.at<gtsam::Pose3>(pair.second) * target_T_base);
      path_msg.poses.push_back(ps);
    }
  }
  smoothed_path_pub_->publish(path_msg);
}

void FactorGraphNode::publishVelocity(
  const gtsam::Vector3 & current_vel,
  const gtsam::Matrix & vel_covariance,
  const rclcpp::Time & timestamp)
{
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

void FactorGraphNode::publishImuBias(
  const gtsam::imuBias::ConstantBias & current_imu_bias,
  const gtsam::Matrix & imu_bias_covariance,
  const rclcpp::Time & timestamp)
{
  geometry_msgs::msg::TwistWithCovarianceStamped imu_bias_msg;
  imu_bias_msg.header.stamp = timestamp;
  imu_bias_msg.header.frame_id = imu_frame_;

  // IMPORTANT! We use 'linear' for accelerometer bias and 'angular' for gyroscope bias.
  imu_bias_msg.twist.twist.linear = toVectorMsg(current_imu_bias.accelerometer());
  imu_bias_msg.twist.twist.angular = toVectorMsg(current_imu_bias.gyroscope());
  imu_bias_msg.twist.covariance = toCovariance36Msg(gtsam::Matrix66(imu_bias_covariance));

  imu_bias_pub_->publish(imu_bias_msg);
}

void FactorGraphNode::publishGraphMetrics(const rclcpp::Time & timestamp)
{
  coug_fgo_msgs::msg::GraphMetrics metrics_msg;
  metrics_msg.header.stamp = timestamp;

  metrics_msg.total_duration = last_total_duration_.load();
  metrics_msg.smoother_duration = last_smoother_duration_.load();
  metrics_msg.cov_duration = last_cov_duration_.load();
  metrics_msg.new_factors = static_cast<uint32_t>(new_factors_.load());
  metrics_msg.total_factors = static_cast<uint32_t>(total_factors_.load());
  metrics_msg.total_variables = static_cast<uint32_t>(total_variables_.load());

  graph_metrics_pub_->publish(metrics_msg);
}

void FactorGraphNode::processFrontend()
{
  while (is_running_.load()) {
    std::unique_lock<std::mutex> lock(frontend_trigger_mutex_);
    frontend_cv_.wait(lock, [this] {return frontend_trigger_ || !is_running_.load();});
    frontend_trigger_ = false;

    if (!is_running_.load()) {
      break;
    }

    lock.unlock();

    if (state_.load() != State::RUNNING) {
      initializeGraph();
    } else {
      bool should_update = true;
      if (params_.max_update_rate > 0.0) {
        rclcpp::Time now = get_clock()->now();
        rclcpp::Duration min_period =
          rclcpp::Duration::from_seconds(1.0 / params_.max_update_rate);
        if (now - last_update_time_ < min_period) {
          should_update = false;
        } else {
          last_update_time_ = now;
        }
      }

      if (should_update) {
        updateGraph();
        {
          std::scoped_lock lock(backend_trigger_mutex_);
          backend_trigger_ = true;
        }
        backend_cv_.notify_one();
      }
    }
  }
}

void FactorGraphNode::processBackend()
{
  while (is_running_.load()) {
    std::unique_lock<std::mutex> lock(backend_trigger_mutex_);
    backend_cv_.wait(lock, [this] {return backend_trigger_ || !is_running_.load();});
    backend_trigger_ = false;

    if (!is_running_.load()) {
      break;
    }

    if (state_.load() == State::RUNNING) {
      lock.unlock();

      bool should_optimize = true;
      if (params_.max_opt_rate > 0.0) {
        rclcpp::Time now = get_clock()->now();
        rclcpp::Duration min_period =
          rclcpp::Duration::from_seconds(1.0 / params_.max_opt_rate);
        if (now - last_opt_time_ < min_period) {
          should_optimize = false;
        } else {
          last_opt_time_ = now;
        }
      }

      if (should_optimize) {
        optimizeGraph();
      }
    }
  }
}

void FactorGraphNode::initializeGraph()
{
  // --- Wait for Sensor TFs ---
  bool imu_ok = !target_T_imu_tf_.header.frame_id.empty();
  bool gps_ok = !(params_.gps.enable_gps || params_.gps.enable_gps_init_only) ||
    !target_T_gps_tf_.header.frame_id.empty();
  bool depth_ok = !target_T_depth_tf_.header.frame_id.empty();
  bool mag_ok = !(params_.mag.enable_mag || params_.mag.enable_mag_init_only) ||
    !target_T_mag_tf_.header.frame_id.empty();
  bool ahrs_ok = !(params_.ahrs.enable_ahrs || params_.ahrs.enable_ahrs_init_only ||
    params_.experimental.enable_loose_dvl_preintegration) ||
    !target_T_ahrs_tf_.header.frame_id.empty();
  bool dvl_ok = !target_T_dvl_tf_.header.frame_id.empty();

  if (!(imu_ok && gps_ok && depth_ok && mag_ok && ahrs_ok && dvl_ok)) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 5000, "Waiting for sensor TFs: %s%s%s%s%s%s",
      !imu_ok ? "[IMU] " : "",
      (!gps_ok && params_.gps.enable_gps) ? "[GPS] " : "",
      (!mag_ok && params_.mag.enable_mag) ? "[Magnetometer] " : "",
      (!ahrs_ok && params_.ahrs.enable_ahrs) ? "[AHRS] " : "",
      !depth_ok ? "[Depth] " : "", !dvl_ok ? "[DVL] " : "");
    return;
  }

  if (target_T_base_tf_.header.frame_id.empty()) {
    try {
      target_T_base_tf_ = tf_buffer_->lookupTransform(
        params_.target_frame, params_.base_frame, tf2::TimePointZero);
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "Failed to lookup base to target transform: %s", ex.what());
      return;
    }
  }

  // --- Compute Initial State ---
  utils::QueueBundle queues;
  queues.imu = imu_queue_.drain();
  queues.gps = gps_queue_.drain();
  queues.depth = depth_queue_.drain();
  queues.mag = mag_queue_.drain();
  queues.ahrs = ahrs_queue_.drain();
  queues.dvl = dvl_queue_.drain();
  queues.wrench = wrench_queue_.drain();
  if (!state_initializer_->update(get_clock()->now(), queues)) {
    return;
  }

  utils::TfBundle tfs{
    toGtsam(target_T_imu_tf_.transform), toGtsam(target_T_gps_tf_.transform),
    toGtsam(target_T_depth_tf_.transform), toGtsam(target_T_mag_tf_.transform),
    toGtsam(target_T_ahrs_tf_.transform), toGtsam(target_T_dvl_tf_.transform),
    toGtsam(target_T_base_tf_.transform), toGtsam(target_T_com_tf_.transform)};
  state_initializer_->compute(tfs);
  core_->initialize(*state_initializer_, tfs);

  state_.store(State::RUNNING);
  RCLCPP_INFO(get_logger(), "Graph initialized successfully!");
}

void FactorGraphNode::updateGraph()
{
  rclcpp::Time target_time{0, 0, RCL_ROS_TIME};
  if (params_.experimental.enable_loose_dvl_preintegration) {
    if (!depth_queue_.empty()) {
      target_time = rclcpp::Time(depth_queue_.back()->header.stamp);
    } else {
      return;
    }
  } else {
    if (!dvl_queue_.empty()) {
      target_time = rclcpp::Time(dvl_queue_.back()->header.stamp);
    } else if (!depth_queue_.empty()) {
      target_time = rclcpp::Time(depth_queue_.back()->header.stamp);
    } else {
      return;
    }
  }

  // --- Update Request ---
  utils::QueueBundle msgs;
  msgs.imu = imu_queue_.drain();
  msgs.gps = gps_queue_.drain();
  msgs.depth = depth_queue_.drain();
  msgs.mag = mag_queue_.drain();
  msgs.ahrs = ahrs_queue_.drain();
  msgs.dvl = dvl_queue_.drain();
  msgs.wrench = wrench_queue_.drain();

  auto result = core_->update(target_time, msgs);

  // Re-queue unused messages
  if (result) {
    if (!result->unused_imu.empty()) {
      imu_queue_.restore(result->unused_imu);
    }
    if (!result->unused_dvl.empty()) {
      dvl_queue_.restore(result->unused_dvl);
    }
  }
}

void FactorGraphNode::optimizeGraph()
{
  // --- Optimization Request ---
  try {
    auto result = core_->optimize();
    if (!result) {return;}

    // --- Update Diagnostics State ---
    last_total_duration_.store(result->total_duration);
    last_smoother_duration_.store(result->smoother_duration);
    last_cov_duration_.store(result->cov_duration);
    new_factors_.store(result->new_factors);
    total_factors_.store(result->total_factors);
    total_variables_.store(result->total_variables);
    processing_overflow_.store(result->processing_overflow);

    if (result->processing_overflow) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "Processing overflow. Batching %zu keyframes.", result->num_keyframes);
    }

    // --- Publish Results ---
    publishGlobalOdom(result->pose, result->pose_cov, result->target_time);

    if (params_.publish_global_tf) {
      broadcastGlobalTf(result->pose, result->target_time);
    }

    if (params_.publish_smoothed_path) {
      publishSmoothedPath(result->all_estimates, result->target_time);
    }

    if (params_.publish_velocity) {
      publishVelocity(result->velocity, result->vel_cov, result->target_time);
    }

    if (params_.publish_imu_bias) {
      publishImuBias(result->imu_bias, result->bias_cov, result->target_time);
    }

    if (params_.publish_graph_metrics) {
      publishGraphMetrics(result->target_time);
    }
  } catch (const std::exception & e) {
    RCLCPP_FATAL(get_logger(), "%s", e.what());
    rclcpp::shutdown();
  }
}

void FactorGraphNode::checkSensorInputs(diagnostic_updater::DiagnosticStatusWrapper & stat)
{
  stat.summary(diagnostic_msgs::msg::DiagnosticStatus::OK, "All requested sensors online.");

  auto check_queue =
    [&](const std::string & name, size_t size, const rclcpp::Time & last_time, bool enabled,
      bool is_critical, double timeout) {
      if (!enabled) {return;}

      double time_since =
        (last_time.nanoseconds() > 0) ? (this->get_clock()->now() - last_time).seconds() : -1.0;

      stat.add(name + " Queue Size", size);
      stat.add(name + " Time Since Last (s)", time_since);

      if (time_since > timeout || (last_time.nanoseconds() == 0 && size == 0)) {
        if (is_critical) {
          stat.mergeSummary(diagnostic_msgs::msg::DiagnosticStatus::ERROR, name + " is offline.");
        } else {
          stat.mergeSummary(diagnostic_msgs::msg::DiagnosticStatus::WARN, name + " is offline.");
        }
      }
    };

  check_queue(
    "IMU", imu_queue_.size(), imu_queue_.getLastTime(), true, true,
    params_.imu.timeout_threshold);
  check_queue(
    "GPS", gps_queue_.size(), gps_queue_.getLastTime(), params_.gps.enable_gps, false,
    params_.gps.timeout_threshold);
  check_queue(
    "Depth", depth_queue_.size(), depth_queue_.getLastTime(), true, true,
    params_.depth.timeout_threshold);
  check_queue(
    "Mag", mag_queue_.size(), mag_queue_.getLastTime(), params_.mag.enable_mag, false,
    params_.mag.timeout_threshold);
  check_queue(
    "AHRS", ahrs_queue_.size(), ahrs_queue_.getLastTime(), params_.ahrs.enable_ahrs, false,
    params_.ahrs.timeout_threshold);
  check_queue(
    "DVL", dvl_queue_.size(), dvl_queue_.getLastTime(), true, true,
    params_.dvl.timeout_threshold);
  check_queue(
    "Wrench", wrench_queue_.size(), wrench_queue_.getLastTime(), params_.dynamics.enable_dynamics,
    false, params_.dynamics.timeout_threshold);
}

void FactorGraphNode::checkGraphState(diagnostic_updater::DiagnosticStatusWrapper & stat)
{
  switch (state_.load()) {
    case State::RUNNING:
      stat.summary(diagnostic_msgs::msg::DiagnosticStatus::OK, "Optimizing factor graph.");
      break;
    case State::WAITING_FOR_SENSORS:
      stat.summary(diagnostic_msgs::msg::DiagnosticStatus::WARN, "Waiting for sensor data.");
      break;
  }
}

void FactorGraphNode::checkProcessingOverflow(diagnostic_updater::DiagnosticStatusWrapper & stat)
{
  if (processing_overflow_.load()) {
    stat.summary(
      diagnostic_msgs::msg::DiagnosticStatus::WARN,
      "Processing overflow detected. Skipping keyframes.");
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

}  // namespace coug_fgo

RCLCPP_COMPONENTS_REGISTER_NODE(coug_fgo::FactorGraphNode)
