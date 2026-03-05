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
 * @file factor_graph_core.hpp
 * @brief ROS-independent GTSAM factor graph logic for AUV state estimation.
 * @author Nelson Durrant
 * @date Mar 2026
 */

#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <optional>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

#include <rclcpp/rclcpp.hpp>

#include <coug_fgo/factor_graph_parameters.hpp>
#include <coug_fgo/utils/dvl_preintegrator.hpp>
#include <coug_fgo/utils/state_initializer.hpp>
#include <coug_fgo/utils/types.hpp>

namespace coug_fgo
{

/**
 * @struct OptimizeResult
 * @brief Output from a successful optimization step.
 */
struct OptimizeResult
{
  gtsam::Pose3 pose;
  gtsam::Vector3 velocity;
  gtsam::imuBias::ConstantBias imu_bias;
  gtsam::Matrix pose_cov;
  gtsam::Matrix vel_cov;
  gtsam::Matrix bias_cov;
  gtsam::Values all_estimates;
  rclcpp::Time target_time{0, 0, RCL_ROS_TIME};
  double opt_duration = 0.0;
  double smoother_duration = 0.0;
  double cov_duration = 0.0;
  bool processing_overflow = false;
  size_t num_keyframes = 0;
  size_t new_factors = 0;
  size_t total_factors = 0;
  size_t total_variables = 0;
};

/**
 * @struct UpdateResult
 * @brief Output from a successful update step (unused messages for re-queueing).
 */
struct UpdateResult
{
  std::deque<sensor_msgs::msg::Imu::SharedPtr> unused_imu;
  std::deque<geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr> unused_dvl;
};

/**
 * @class FactorGraphCore
 * @brief ROS-independent core for AUV state estimation via factor graph optimization.
 */
class FactorGraphCore
{
public:
  /**
   * @brief Constructs the core with node parameters.
   * @param params Node parameters.
   */
  explicit FactorGraphCore(const factor_graph_node::Params & params);

  /**
   * @brief Initializes the graph from computed initial state.
   * @param state_init The computed initial state.
   * @param tfs GTSAM Pose3 sensor transforms.
   */
  void initialize(
    const utils::StateInitializer & state_init,
    const utils::TfBundle & tfs);

  /**
   * @brief Builds factors for one keyframe and writes them to the buffer.
   * @param target_time The keyframe timestamp.
   * @param msgs Drained sensor messages (consumed).
   * @return UpdateResult with unused messages, or nullopt if timestamp was stale.
   */
  std::optional<UpdateResult> update(
    const rclcpp::Time & target_time,
    utils::QueueBundle & msgs);

  /**
   * @brief Consumes the buffer and runs the GTSAM smoother.
   * @return OptimizeResult, or nullopt if no buffer was available.
   */
  std::optional<OptimizeResult> optimize();

  /**
   * @brief Returns the last processed timestamp (for stale check by the node).
   */
  rclcpp::Time prev_time() const { return prev_time_; }

  std::mutex buffer_mutex;
  std::map<rclcpp::Time, gtsam::Key> time_to_key;

private:
  // --- Configuration ---
  /**
   * @brief Configures the GTSAM combined IMU preintegration parameters.
   * @param state_init Provides initial sensor data for covariance estimation.
   * @return Shared pointer to the configured preintegration parameters.
   */
  std::shared_ptr<gtsam::PreintegratedCombinedMeasurements::Params>
  configureImuPreintegration(const utils::StateInitializer & state_init);

  // --- Factor Construction ---
  /**
   * @brief Adds pose, velocity, and IMU bias prior factors to the initial graph.
   * @param state_init Provides computed initial state values.
   * @param graph The factor graph to add priors to.
   * @param values The initial variable estimates.
   */
  void addPriorFactors(
    const utils::StateInitializer & state_init,
    gtsam::NonlinearFactorGraph & graph,
    gtsam::Values & values);

  /**
   * @brief Adds a 2D GPS position factor with lever arm compensation.
   * @param graph The target factor graph.
   * @param gps_msgs Drained GPS odometry messages.
   */
  void addGpsFactor(
    gtsam::NonlinearFactorGraph & graph,
    const std::deque<nav_msgs::msg::Odometry::SharedPtr> & gps_msgs);

  /**
   * @brief Adds a 1D depth factor with lever arm compensation.
   * @param graph The target factor graph.
   * @param depth_msgs Drained depth odometry messages.
   */
  void addDepthFactor(
    gtsam::NonlinearFactorGraph & graph,
    const std::deque<nav_msgs::msg::Odometry::SharedPtr> & depth_msgs);

  /**
   * @brief Adds an AHRS yaw orientation factor with lever arm compensation.
   * @param graph The target factor graph.
   * @param ahrs_msgs Drained AHRS IMU messages.
   */
  void addAhrsFactor(
    gtsam::NonlinearFactorGraph & graph,
    const std::deque<sensor_msgs::msg::Imu::SharedPtr> & ahrs_msgs);

  /**
   * @brief Adds a magnetometer heading factor with lever arm compensation.
   * @param graph The target factor graph.
   * @param mag_msgs Drained magnetometer messages.
   */
  void addMagFactor(
    gtsam::NonlinearFactorGraph & graph,
    const std::deque<sensor_msgs::msg::MagneticField::SharedPtr> & mag_msgs);

  /**
   * @brief Adds a DVL body-frame velocity factor with lever arm compensation.
   * @param graph The target factor graph.
   * @param dvl_msgs Drained DVL twist messages.
   */
  void addDvlFactor(
    gtsam::NonlinearFactorGraph & graph,
    const std::deque<geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr> & dvl_msgs);

  /**
   * @brief Adds a constant-velocity (zero-acceleration) prior between keyframes.
   * @param graph The target factor graph.
   * @param target_time The current keyframe timestamp.
   */
  void addConstantVelocityFactor(
    gtsam::NonlinearFactorGraph & graph,
    const rclcpp::Time & target_time);

  /**
   * @brief Adds a simplified Fossen AUV dynamics factor with lever arm compensation.
   * @param graph The target factor graph.
   * @param wrench_msgs Drained thruster wrench messages (zero-order hold).
   * @param target_time The current keyframe timestamp.
   */
  void addAuvDynamicsFactor(
    gtsam::NonlinearFactorGraph & graph,
    const std::deque<geometry_msgs::msg::WrenchStamped::SharedPtr> & wrench_msgs,
    const rclcpp::Time & target_time);

  /**
   * @brief Integrates IMU measurements and adds a combined IMU factor.
   * @param graph The target factor graph.
   * @param imu_msgs Drained, time-sorted IMU messages.
   * @param target_time Integration endpoint timestamp.
   * @param[out] unused_imu Messages with timestamps beyond target_time for re-queueing.
   */
  void addPreintegratedImuFactor(
    gtsam::NonlinearFactorGraph & graph,
    const std::deque<sensor_msgs::msg::Imu::SharedPtr> & imu_msgs,
    const rclcpp::Time & target_time,
    std::deque<sensor_msgs::msg::Imu::SharedPtr> & unused_imu);

  /**
   * @brief Interpolates IMU-derived orientation at a target timestamp via SLERP.
   * @param imu_msgs Time-sorted IMU messages bracketing the target time.
   * @param target_time The desired interpolation timestamp.
   * @return The interpolated rotation as a GTSAM Rot3.
   */
  gtsam::Rot3 getInterpolatedOrientation(
    const std::deque<sensor_msgs::msg::Imu::SharedPtr> & imu_msgs,
    const rclcpp::Time & target_time);

  /**
   * @brief Integrates DVL measurements (rotated via IMU) and adds a preintegrated DVL factor.
   * @param graph The target factor graph.
   * @param dvl_msgs Drained, time-sorted DVL messages.
   * @param imu_msgs Drained IMU messages for orientation interpolation.
   * @param target_time Integration endpoint timestamp.
   * @param[out] unused_dvl Messages with timestamps beyond target_time for re-queueing.
   */
  void addPreintegratedDvlFactor(
    gtsam::NonlinearFactorGraph & graph,
    const std::deque<geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr> & dvl_msgs,
    const std::deque<sensor_msgs::msg::Imu::SharedPtr> & imu_msgs,
    const rclcpp::Time & target_time,
    std::deque<geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr> & unused_dvl);

  // --- Parameters ---
  const factor_graph_node::Params & params_;

  // --- Sensor Transforms ---
  utils::TfBundle tfs_;

  // --- GTSAM Solver ---
  std::unique_ptr<gtsam::IncrementalFixedLagSmoother> inc_smoother_;
  std::unique_ptr<gtsam::ISAM2> isam_;
  std::unique_ptr<gtsam::PreintegratedCombinedMeasurements> imu_preintegrator_;
  std::unique_ptr<utils::DvlPreintegrator> dvl_preintegrator_;

  // --- State Estimates ---
  size_t prev_step_ = 0;
  size_t current_step_ = 1;
  rclcpp::Time prev_time_{0, 0, RCL_ROS_TIME};

  gtsam::Pose3 prev_pose_;
  gtsam::Vector3 prev_vel_;
  gtsam::imuBias::ConstantBias prev_imu_bias_;

  // --- Cached Sensor Data ---
  gtsam::Vector3 last_dvl_velocity_ = gtsam::Vector3::Zero();
  gtsam::Matrix3 last_dvl_covariance_ = gtsam::Matrix3::Zero();
  gtsam::Vector3 last_imu_acc_ = gtsam::Vector3::Zero();
  gtsam::Vector3 last_imu_gyr_ = gtsam::Vector3::Zero();
  geometry_msgs::msg::WrenchStamped::SharedPtr last_wrench_msg_;

  // --- Double Buffer ---
  gtsam::NonlinearFactorGraph buffer_graph_;
  gtsam::Values buffer_values_;
  gtsam::IncrementalFixedLagSmoother::KeyTimestampMap buffer_timestamps_;
  rclcpp::Time buffer_target_time_{0, 0, RCL_ROS_TIME};
  size_t buffer_last_step_ = 0;
  bool has_buffer_ = false;
};

}  // namespace coug_fgo
