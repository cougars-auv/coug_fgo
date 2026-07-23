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
 * @brief C++ GTSAM factor graph logic for AUV state estimation.
 * @author Nelson Durrant
 * @date May 2026
 */

#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/IncrementalFixedLagSmoother.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "coug_fgo/factor_graph_parameters.hpp"
#include "coug_fgo/utils/data_types.hpp"
#include "coug_fgo/utils/dvl_loose_preintegrator.hpp"
#include "coug_fgo/utils/dvl_tight_preintegrator.hpp"
#include "coug_fgo/utils/logging.hpp"

namespace coug_fgo {

/**
 * @struct OptimizeResult
 * @brief Output from a successful optimization step.
 */
struct OptimizeResult {
  double target_time{0.0};
  gtsam::Pose3 pose;
  gtsam::Vector3 velocity;
  gtsam::imuBias::ConstantBias imu_bias;
  gtsam::Matrix pose_cov;
  gtsam::Matrix vel_cov;
  gtsam::Matrix bias_cov;
  gtsam::Values all_estimates;

  double total_duration = 0.0;
  double smoother_duration = 0.0;
  double cov_duration = 0.0;
  bool processing_overflow = false;

  size_t num_keyframes = 0;
  size_t new_factors = 0;
  size_t total_factors = 0;
  size_t total_variables = 0;
};

/**
 * @class FactorGraphCore
 * @brief C++ GTSAM factor graph logic for AUV state estimation.
 */
class FactorGraphCore {
 public:
  /**
   * @brief Constructs the core with node parameters.
   * @param params Node parameters (copied).
   */
  explicit FactorGraphCore(const factor_graph_node::Params& params);

  /**
   * @brief Sets the sink for core log messages (e.g. rclcpp or Python logging).
   * @param callback Receives a level and message; set before other threads use the core.
   */
  void setLogCallback(utils::LogCallback callback);

  /**
   * @brief Initializes the graph from the computed initial state.
   * @param init_state The computed initial state and averaged initial sensor samples.
   * @param tfs GTSAM Pose3 sensor transforms.
   */
  void initialize(const utils::InitialState& init_state, const utils::TfBundle& tfs);

  /**
   * @brief Builds factors for one keyframe and writes the graph to the buffer.
   * @param target_time The keyframe timestamp.
   * @param queues Drained sensor data structs (consumed).
   * @param tfs Latest sensor transforms to refresh (picks up lazily-resolved ones).
   * @return Structs newer than the keyframe to re-queue, or nullopt if the timestamp was stale.
   */
  std::optional<utils::QueueBundle> update(double target_time, utils::QueueBundle& queues,
                                           const utils::TfBundle& tfs);

  /**
   * @brief Consumes the buffer and runs the GTSAM smoother.
   * @return OptimizeResult, or nullopt if no buffer was available.
   */
  std::optional<OptimizeResult> optimize();

  /**
   * @brief Returns a thread-safe copy of the time-to-key map.
   * @return Snapshot of the current time-to-key map.
   */
  std::map<int64_t, gtsam::Key> snapshotTimeKeys() const;

 private:
  // --- Logging ---
  /**
   * @brief Sends a message to the configured log sink, if any.
   * @param level The message severity.
   * @param msg The log message.
   */
  void logMessage(utils::LogLevel level, const std::string& msg) const;

  /**
   * @brief Builds a warning callback for an unusable sensor message covariance.
   * @param sensor Human-readable sensor name for the warning message.
   * @return Callback for resolveCov/resolveVar to invoke on fallback.
   */
  std::function<void()> covFallbackWarning(const std::string& sensor) const;

  // --- Configuration ---
  /**
   * @brief Configures the GTSAM combined IMU preintegration parameters.
   * @param init_state Provides the averaged initial IMU sample.
   * @return Shared pointer to the configured preintegration parameters.
   */
  std::shared_ptr<gtsam::PreintegratedCombinedMeasurements::Params> configureImuPreintegration(
      const utils::InitialState& init_state) const;

  // --- Factor Construction ---
  /**
   * @brief Adds pose, velocity, and IMU bias prior factors to the initial graph.
   * @param init_state Provides the computed initial state values.
   * @param graph The factor graph to add priors to.
   * @param values The initial variable estimates.
   */
  void addPriorFactors(const utils::InitialState& init_state, gtsam::NonlinearFactorGraph& graph,
                       gtsam::Values& values);

  /**
   * @brief Adds a 2D GPS position factor with lever arm compensation.
   * @param graph The target factor graph.
   * @param gps_msgs Drained GPS odometry structs (only the newest is used).
   */
  void addGpsFactor(gtsam::NonlinearFactorGraph& graph,
                    const std::deque<std::shared_ptr<utils::OdometryData>>& gps_msgs);

  /**
   * @brief Adds a 1D depth factor with lever arm compensation.
   * @param graph The target factor graph.
   * @param depth_msgs Drained depth odometry structs (only the newest is used).
   */
  void addDepthFactor(gtsam::NonlinearFactorGraph& graph,
                      const std::deque<std::shared_ptr<utils::OdometryData>>& depth_msgs);

  /**
   * @brief Adds an AHRS attitude (or yaw) factor with sensor rotation and declination compensation.
   * @param graph The target factor graph.
   * @param ahrs_msgs Drained AHRS IMU structs (only the newest is used).
   */
  void addAhrsFactor(gtsam::NonlinearFactorGraph& graph,
                     const std::deque<std::shared_ptr<utils::AhrsData>>& ahrs_msgs);

  /**
   * @brief Adds a magnetometer field factor with sensor rotation compensation.
   * @param graph The target factor graph.
   * @param mag_msgs Drained magnetometer structs (only the newest is used).
   */
  void addMagFactor(gtsam::NonlinearFactorGraph& graph,
                    const std::deque<std::shared_ptr<utils::MagneticFieldData>>& mag_msgs);

  /**
   * @brief Adds a DVL body-frame velocity factor with lever arm compensation.
   * @param graph The target factor graph.
   * @param dvl_msgs Drained DVL twist structs (only the newest is used).
   * @param held_imu_gyr The held gyro sample (IMU frame) at the keyframe, for the lever arm term.
   */
  void addDvlFactor(gtsam::NonlinearFactorGraph& graph,
                    const std::deque<std::shared_ptr<utils::TwistData>>& dvl_msgs,
                    const gtsam::Vector3& held_imu_gyr);

  /**
   * @brief Adds a constant target-frame velocity factor between keyframes.
   * @param graph The target factor graph.
   * @param target_time The current keyframe timestamp.
   */
  void addConstVelFactor(gtsam::NonlinearFactorGraph& graph, double target_time);

  /**
   * @brief Adds a simplified Fossen AUV dynamics factor with thrust-frame rotation compensation.
   * @param graph The target factor graph.
   * @param wrench_msgs Drained thruster wrench structs (zero-order hold).
   * @param target_time The current keyframe timestamp.
   */
  void addAuvDynamicsFactor(gtsam::NonlinearFactorGraph& graph,
                            const std::deque<std::shared_ptr<utils::WrenchData>>& wrench_msgs,
                            double target_time);

  /**
   * @brief Integrates IMU measurements and adds a combined IMU factor.
   * @param graph The target factor graph.
   * @param imu_msgs Drained, time-sorted IMU structs.
   * @param target_time Integration endpoint timestamp.
   */
  void addImuPreintFactor(gtsam::NonlinearFactorGraph& graph,
                          const std::deque<std::shared_ptr<utils::ImuData>>& imu_msgs,
                          double target_time);

  /**
   * @brief Integrates AHRS-rotated DVL measurements and adds a loosely-coupled DVL factor.
   * @param graph The target factor graph.
   * @param dvl_msgs Drained, time-sorted DVL structs.
   * @param ahrs_msgs Drained AHRS structs for orientation interpolation.
   * @param target_time Integration endpoint timestamp.
   */
  void addDvlLoosePreintFactor(gtsam::NonlinearFactorGraph& graph,
                               const std::deque<std::shared_ptr<utils::TwistData>>& dvl_msgs,
                               const std::deque<std::shared_ptr<utils::AhrsData>>& ahrs_msgs,
                               double target_time);

  /**
   * @brief Integrates IMU-rotated DVL measurements and adds a tightly-coupled DVL factor.
   * @param graph The target factor graph.
   * @param dvl_msgs Drained, time-sorted DVL structs.
   * @param imu_msgs Drained, time-sorted IMU structs for relative rotation calculation.
   * @param target_time Integration endpoint timestamp.
   * @param held_imu_acc The zero-order-held IMU acceleration sample at the window start.
   * @param held_imu_gyr The zero-order-held IMU gyro sample at the window start.
   */
  void addDvlTightPreintFactor(gtsam::NonlinearFactorGraph& graph,
                               const std::deque<std::shared_ptr<utils::TwistData>>& dvl_msgs,
                               const std::deque<std::shared_ptr<utils::ImuData>>& imu_msgs,
                               double target_time, const gtsam::Vector3& held_imu_acc,
                               const gtsam::Vector3& held_imu_gyr);

  /**
   * @brief Adds neighboring-agent odometry, depth, orientation, and range/bearing factors.
   * @param graph The target factor graph.
   * @param values The new variable estimates.
   * @param timestamps The new key timestamps.
   * @param queues Drained per-neighbor status structs, one deque per neighbor.
   * @param target_time The current keyframe timestamp.
   */
  void addMultiAgentFactors(gtsam::NonlinearFactorGraph& graph, gtsam::Values& values,
                            gtsam::IncrementalFixedLagSmoother::KeyTimestampMap& timestamps,
                            utils::QueueBundle& queues, double target_time);

  // --- Parameters ---
  const factor_graph_node::Params params_;
  utils::TfBundle tfs_;

  // --- Logging ---
  utils::LogCallback log_callback_;
  mutable std::mutex log_mutex_;
  mutable std::set<std::string> cov_warned_;

  // --- GTSAM Solver ---
  std::unique_ptr<gtsam::IncrementalFixedLagSmoother> inc_smoother_;
  std::unique_ptr<gtsam::ISAM2> isam_;
  gtsam::NonlinearFactorGraph lm_graph_;
  gtsam::Values lm_values_;
  std::unique_ptr<gtsam::PreintegratedCombinedMeasurements> imu_preintegrator_;
  std::unique_ptr<utils::DvlLoosePreintegrator> dvl_loose_preintegrator_;
  std::unique_ptr<utils::DvlTightPreintegrator> dvl_tight_preintegrator_;

  // --- State Estimates ---
  size_t prev_step_ = 0;
  size_t current_step_ = 1;
  double prev_time_{0.0};

  gtsam::Pose3 prev_pose_;
  gtsam::Vector3 prev_vel_;
  gtsam::imuBias::ConstantBias prev_imu_bias_;

  // --- Sensor Data ---
  gtsam::Vector3 last_dvl_velocity_ = gtsam::Vector3::Zero();
  gtsam::Matrix3 last_dvl_covariance_ = gtsam::Matrix3::Zero();
  gtsam::Vector3 last_imu_acc_ = gtsam::Vector3::Zero();
  gtsam::Vector3 last_imu_gyr_ = gtsam::Vector3::Zero();
  std::shared_ptr<utils::WrenchData> last_wrench_msg_;
  std::vector<std::shared_ptr<utils::AgentStatusData>> last_multiagent_status_;

  // --- Buffer ---
  mutable std::mutex state_mutex_;
  std::map<int64_t, gtsam::Key> time_to_key_;
  gtsam::NonlinearFactorGraph buffer_graph_;
  gtsam::Values buffer_values_;
  gtsam::IncrementalFixedLagSmoother::KeyTimestampMap buffer_timestamps_;
  double buffer_target_time_{0.0};
  size_t buffer_last_step_ = 0;
  bool has_buffer_ = false;
};

}  // namespace coug_fgo
