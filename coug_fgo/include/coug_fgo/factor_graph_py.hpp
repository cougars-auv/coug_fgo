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
 * @file factor_graph_py.hpp
 * @brief Thin Python bindings wrapper for the FactorGraphCore.
 * @author Nelson Durrant
 * @date May 2026
 */

#pragma once

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "coug_fgo/factor_graph_core.hpp"
#include "coug_fgo/factor_graph_parameters.hpp"
#include "coug_fgo/state_initializer.hpp"

namespace coug_fgo {

/**
 * @class FactorGraphPy
 * @brief Thin Python bindings wrapper for the FactorGraphCore.
 */
class FactorGraphPy {
 public:
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  using Matrix6d = Eigen::Matrix<double, 6, 6>;

  using ImuBatch = std::vector<
      std::tuple<double, Eigen::Vector3d, Eigen::Vector3d, Eigen::Matrix3d, Eigen::Matrix3d>>;
  using OdomBatch = std::vector<std::tuple<double, Eigen::Vector3d, Matrix6d>>;
  using DepthBatch = std::vector<std::tuple<double, double, Matrix6d>>;
  using MagBatch = std::vector<std::tuple<double, Eigen::Vector3d, Eigen::Matrix3d>>;
  using AhrsBatch = std::vector<std::tuple<double, Eigen::Vector4d, Eigen::Matrix3d>>;
  using TwistBatch = std::vector<std::tuple<double, Eigen::Vector3d, Matrix6d>>;
  using WrenchBatch = std::vector<std::tuple<double, Vector6d>>;

  /**
   * @brief Constructs the wrapper, loading parameters from ROS 2 YAML config files.
   * @param config_paths Paths to ROS 2 parameter YAML files (later files override earlier ones).
   * @param ns Optional ROS namespace for parameter resolution.
   */
  explicit FactorGraphPy(const std::vector<std::string>& config_paths, const std::string& ns = "");

  /**
   * @brief Returns the loaded parameters needed to drive the offline pipeline.
   * @return Nested dict mirroring the node-level orchestration parameters.
   */
  pybind11::dict get_params() const;

  /**
   * @brief Sets one sensor transform (sensor pose in the target frame).
   * @param name One of "imu", "gps", "depth", "mag", "ahrs", "dvl", "base", "com".
   * @param position Translation [x, y, z] in meters.
   * @param quat_xyzw Orientation quaternion (x, y, z, w).
   * @throws std::invalid_argument If the name is not a known transform.
   */
  void set_tf(const std::string& name, const Eigen::Vector3d& position,
              const Eigen::Vector4d& quat_xyzw);

  /**
   * @brief Feeds measurement batches to the state initializer and initializes when ready.
   * @param current_time Newest timestamp across the batches, in seconds.
   * @return True if the graph is initialized (or was already initialized).
   */
  bool initialize(double current_time, const ImuBatch& imu, const OdomBatch& gps,
                  const DepthBatch& depth, const MagBatch& mag, const AhrsBatch& ahrs,
                  const TwistBatch& dvl, const WrenchBatch& wrench);

  /**
   * @brief Builds factors for one keyframe from the given measurement batches.
   * @param target_time Keyframe timestamp in seconds.
   * @return Dict of leftover batches to re-queue, or None if the keyframe was
   *         rejected (the caller should keep all queued measurements).
   */
  pybind11::object update(double target_time, const ImuBatch& imu, const OdomBatch& gps,
                          const DepthBatch& depth, const MagBatch& mag, const AhrsBatch& ahrs,
                          const TwistBatch& dvl, const WrenchBatch& wrench);

  /**
   * @brief Runs the GTSAM smoother on the buffered keyframes.
   * @return Optimization results, or an empty dict if there was nothing to optimize.
   */
  pybind11::dict optimize();

  /**
   * @brief Resets the estimator to re-initialize from scratch (mirrors the node's reset service).
   */
  void reset();

  /**
   * @brief Returns whether the graph has been initialized.
   */
  bool is_initialized() const { return is_initialized_; }

 private:
  /**
   * @brief Converts Python measurement batches into a core QueueBundle.
   */
  static utils::QueueBundle to_bundle(const ImuBatch& imu, const OdomBatch& gps,
                                      const DepthBatch& depth, const MagBatch& mag,
                                      const AhrsBatch& ahrs, const TwistBatch& dvl,
                                      const WrenchBatch& wrench);

  /**
   * @brief Converts a core QueueBundle back into a dict of Python measurement batches.
   */
  static pybind11::dict from_bundle(const utils::QueueBundle& queues);

  // --- Core ---
  factor_graph_node::Params params_;
  std::unique_ptr<FactorGraphCore> core_;
  std::unique_ptr<StateInitializer> state_init_;

  // --- State ---
  utils::TfBundle tfs_;
  bool is_initialized_{false};
};

}  // namespace coug_fgo
