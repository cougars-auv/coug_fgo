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
 * @file factor_graph.hpp
 * @brief ROS 2 node for multi-sensor AUV state estimation via factor graph optimization.
 * @author Nelson Durrant
 * @date Jan 2026
 */

#pragma once

#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/navigation/ImuBias.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <diagnostic_updater/diagnostic_updater.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/magnetic_field.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include "coug_fgo/factor_graph_core.hpp"
#include "coug_fgo/factor_graph_parameters.hpp"
#include "coug_fgo/utils/conversions.hpp"
#include "coug_fgo/utils/state_initializer.hpp"
#include "coug_fgo/utils/thread_safe_queue.hpp"
#include <coug_fgo_msgs/msg/graph_metrics.hpp>


namespace coug_fgo
{

/**
 * @class FactorGraphNode
 * @brief ROS 2 node for multi-sensor AUV state estimation via factor graph optimization.
 */
class FactorGraphNode : public rclcpp::Node
{
public:
  /**
   * @brief Constructs the node and launches frontend/backend threads.
   * @param options ROS 2 node options (composable node support).
   */
  explicit FactorGraphNode(const rclcpp::NodeOptions & options);

  /**
   * @brief Joins worker threads and shuts down gracefully.
   */
  ~FactorGraphNode() override;

  enum class State
  {
    WAITING_FOR_SENSORS,
    RUNNING
  };

protected:
  // --- Main Logic ---
  /**
   * @brief Initializes the factor graph using averaged sensor data or parameters.
   */
  void initializeGraph();

  /**
   * @brief Drains queues and delegates factor building to the core.
   */
  void updateGraph();

  /**
   * @brief Delegates optimization to the core and publishes results.
   */
  void optimizeGraph();

  /**
   * @brief The background loop run by the dedicated frontend thread.
   */
  void processFrontend();

  /**
   * @brief The background loop run by the dedicated backend thread.
   */
  void processBackend();

  // --- Setup ---
  /**
   * @brief Creates publishers, subscribers, TF interfaces, and diagnostics.
   */
  void setupRosInterfaces();

  // --- Publishing ---
  /**
   * @brief Publishes the optimized global odometry.
   * @param current_pose The estimated target pose.
   * @param pose_covariance The estimation error covariance.
   * @param timestamp The message timestamp.
   */
  void publishGlobalOdom(
    const gtsam::Pose3 & current_pose,
    const gtsam::Matrix & pose_covariance,
    const rclcpp::Time & timestamp);

  /**
   * @brief Broadcasts the map-to-odom transform.
   * @param current_pose The estimated target pose.
   * @param timestamp The transform timestamp.
   */
  void broadcastGlobalTf(
    const gtsam::Pose3 & current_pose,
    const rclcpp::Time & timestamp);

  /**
   * @brief Publishes the full optimized trajectory path.
   * @param results The final optimized values.
   * @param timestamp The path timestamp.
   */
  void publishSmoothedPath(
    const gtsam::Values & results,
    const rclcpp::Time & timestamp);

  /**
   * @brief Publishes the optimized velocity (at the target frame in the map frame).
   * @param current_vel The estimated velocity.
   * @param vel_covariance The estimation error covariance.
   * @param timestamp The message timestamp.
   */
  void publishVelocity(
    const gtsam::Vector3 & current_vel,
    const gtsam::Matrix & vel_covariance,
    const rclcpp::Time & timestamp);

  /**
   * @brief Publishes the optimized IMU biases.
   * @param current_imu_bias The estimated biases.
   * @param imu_bias_covariance The estimation error covariance.
   * @param timestamp The message timestamp.
   */
  void publishImuBias(
    const gtsam::imuBias::ConstantBias & current_imu_bias,
    const gtsam::Matrix & imu_bias_covariance,
    const rclcpp::Time & timestamp);

  /**
   * @brief Publishes high-frequency timing and graph metadata.
   * @param timestamp The message timestamp.
   */
  void publishGraphMetrics(const rclcpp::Time & timestamp);

  // --- Diagnostics ---
  /**
   * @brief Checks sensor inputs for queue sizes and data freshness.
   * @param stat The diagnostic status wrapper.
   */
  void checkSensorInputs(diagnostic_updater::DiagnosticStatusWrapper & stat);

  /**
   * @brief Checks the overall graph lifecycle state.
   * @param stat The diagnostic status wrapper.
   */
  void checkGraphState(diagnostic_updater::DiagnosticStatusWrapper & stat);

  /**
   * @brief Checks optimization times for processing overflow.
   * @param stat The diagnostic status wrapper.
   */
  void checkProcessingOverflow(diagnostic_updater::DiagnosticStatusWrapper & stat);

  // --- Core ---
  std::unique_ptr<FactorGraphCore> core_;

  // --- Node State ---
  std::atomic<State> state_{State::WAITING_FOR_SENSORS};
  std::unique_ptr<utils::StateInitializer> state_initializer_;
  rclcpp::Time last_update_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_opt_time_{0, 0, RCL_ROS_TIME};

  // --- Diagnostics State ---
  std::atomic<double> last_total_duration_{0.0};
  std::atomic<double> last_smoother_duration_{0.0};
  std::atomic<double> last_cov_duration_{0.0};
  std::atomic<bool> processing_overflow_{false};
  std::atomic<size_t> new_factors_{0};
  std::atomic<size_t> total_factors_{0};
  std::atomic<size_t> total_variables_{0};

  // --- Message Queues ---
  utils::ThreadSafeQueue<sensor_msgs::msg::Imu::SharedPtr> imu_queue_;
  utils::ThreadSafeQueue<nav_msgs::msg::Odometry::SharedPtr> gps_queue_;
  utils::ThreadSafeQueue<nav_msgs::msg::Odometry::SharedPtr> depth_queue_;
  utils::ThreadSafeQueue<sensor_msgs::msg::MagneticField::SharedPtr> mag_queue_;
  utils::ThreadSafeQueue<sensor_msgs::msg::Imu::SharedPtr> ahrs_queue_;
  utils::ThreadSafeQueue<geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr> dvl_queue_;
  utils::ThreadSafeQueue<geometry_msgs::msg::WrenchStamped::SharedPtr> wrench_queue_;

  // --- Multithreading ---
  std::thread frontend_thread_;
  std::thread backend_thread_;
  std::condition_variable frontend_cv_;
  std::condition_variable backend_cv_;
  std::mutex frontend_trigger_mutex_;
  std::mutex backend_trigger_mutex_;
  bool frontend_trigger_{false};
  bool backend_trigger_{false};
  std::atomic<bool> is_running_{true};

  rclcpp::CallbackGroup::SharedPtr sensor_cb_group_;

  // --- Transformations ---
  geometry_msgs::msg::TransformStamped target_T_base_tf_;
  geometry_msgs::msg::TransformStamped target_T_dvl_tf_;
  geometry_msgs::msg::TransformStamped target_T_imu_tf_;
  geometry_msgs::msg::TransformStamped target_T_gps_tf_;
  geometry_msgs::msg::TransformStamped target_T_depth_tf_;
  geometry_msgs::msg::TransformStamped target_T_mag_tf_;
  geometry_msgs::msg::TransformStamped target_T_ahrs_tf_;
  geometry_msgs::msg::TransformStamped target_T_com_tf_;

  std::string imu_frame_;

  // --- ROS Interfaces ---
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr global_odom_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr smoothed_path_pub_;
  rclcpp::Publisher<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr velocity_pub_;
  rclcpp::Publisher<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr imu_bias_pub_;
  rclcpp::Publisher<coug_fgo_msgs::msg::GraphMetrics>::SharedPtr graph_metrics_pub_;

  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr gps_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr depth_sub_;
  rclcpp::Subscription<sensor_msgs::msg::MagneticField>::SharedPtr mag_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr ahrs_sub_;
  rclcpp::Subscription<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr dvl_sub_;
  rclcpp::Subscription<geometry_msgs::msg::WrenchStamped>::SharedPtr wrench_sub_;

  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  diagnostic_updater::Updater diagnostic_updater_;

  // --- Parameters ---
  std::shared_ptr<factor_graph_node::ParamListener> param_listener_;
  factor_graph_node::Params params_;
};

}  // namespace coug_fgo
