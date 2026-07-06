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
 * @file factor_graph_core.cpp
 * @brief Implementation of the FactorGraphCore.
 * @author Nelson Durrant
 * @date May 2026
 */

#include "coug_fgo/factor_graph_core.hpp"

#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/slam/PriorFactor.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

#include "coug_fgo/factors/ahrs_factor.hpp"
#include "coug_fgo/factors/auv_dynamics_factor.hpp"
#include "coug_fgo/factors/const_vel_factor.hpp"
#include "coug_fgo/factors/depth_factor.hpp"
#include "coug_fgo/factors/dvl_factor.hpp"
#include "coug_fgo/factors/dvl_loose_preint_factor.hpp"
#include "coug_fgo/factors/dvl_tight_preint_factor.hpp"
#include "coug_fgo/factors/gps_factor.hpp"
#include "coug_fgo/factors/mag_factor.hpp"
#include "coug_fgo/utils/param_enums.hpp"

using coug_fgo::factors::AhrsFactorArm;
using coug_fgo::factors::AuvDynamicsFactorArm;
using coug_fgo::factors::ConstVelFactor;
using coug_fgo::factors::DepthFactorArm;
using coug_fgo::factors::DvlFactorArm;
using coug_fgo::factors::DvlLoosePreintFactorArm;
using coug_fgo::factors::DvlTightPreintFactorArm;
using coug_fgo::factors::Gps2dFactorArm;
using coug_fgo::factors::MagFactorArm;

using gtsam::symbol_shorthand::B;  // Bias (ax,ay,az,gx,gy,gz)
using gtsam::symbol_shorthand::V;  // Velocity (x,y,z)
using gtsam::symbol_shorthand::X;  // Pose3 (x,y,z,r,p,y)

namespace coug_fgo {

using utils::DvlLoosePreintegrator;
using utils::DvlTightPreintegrator;
using utils::parseRobustKernel;
using utils::parseSolverType;
using utils::RobustKernel;
using utils::SolverType;
using utils::StateInitializer;

namespace {

/**
 * @brief Checks a covariance matrix for non-finite or non-positive diagonal entries.
 * @param m The covariance matrix to check.
 * @return True if the covariance is unusable as a noise model.
 */
template <typename Derived>
bool isInvalidCovDiag(const Eigen::MatrixBase<Derived>& m) {
  return !m.allFinite() || (m.diagonal().array() <= 0.0).any();
}

/**
 * @brief Builds a diagonal covariance matrix from a vector of standard deviations.
 * @param sigmas Per-axis standard deviations (at least N entries).
 * @return The N x N diagonal matrix with sigmas^2 on the diagonal.
 */
template <int N = 3>
Eigen::Matrix<double, N, N> sigmasSquaredDiag(const std::vector<double>& sigmas) {
  return Eigen::Matrix<double, N, N>(Eigen::Map<const Eigen::Matrix<double, N, 1>>(sigmas.data())
                                         .array()
                                         .square()
                                         .matrix()
                                         .asDiagonal());
}

/**
 * @brief Resolves a sensor covariance from parameters or a message.
 * @param use_param Prefer the parameter-derived covariance regardless of message validity.
 * @param sigmas Parameter standard deviations (at least N entries).
 * @param scalar Covariance scaling factor (applied to either source).
 * @param msg_cov The covariance reported by the sensor message.
 * @return sigmasSquaredDiag(sigmas)*scalar if use_param or invalid msg_cov, else msg_cov*scalar.
 */
template <int N>
Eigen::Matrix<double, N, N> resolveCov(bool use_param, const std::vector<double>& sigmas,
                                       double scalar, const Eigen::Matrix<double, N, N>& msg_cov) {
  if (use_param || isInvalidCovDiag(msg_cov)) {
    return sigmasSquaredDiag<N>(sigmas) * scalar;
  }
  return msg_cov * scalar;
}

/**
 * @brief Scalar-variance version of resolveCov.
 * @return sigma^2 * scalar if use_param is set or msg_var is unusable, otherwise msg_var * scalar.
 */
double resolveVar(bool use_param, double sigma, double scalar, double msg_var) {
  if (use_param || !std::isfinite(msg_var) || msg_var <= 0.0) {
    return sigma * sigma * scalar;
  }
  return msg_var * scalar;
}

/**
 * @brief Wraps a noise model in the configured robust M-estimator kernel.
 * @param noise The base noise model.
 * @param kernel The robust kernel name ("None", "Huber", or "Tukey").
 * @param k The kernel tuning constant.
 * @return The wrapped noise model, or the base model if no kernel is configured.
 */
gtsam::SharedNoiseModel applyRobustKernel(const gtsam::SharedNoiseModel& noise,
                                          const std::string& kernel, double k) {
  switch (parseRobustKernel(kernel)) {
    case RobustKernel::kHuber:
      return gtsam::noiseModel::Robust::Create(gtsam::noiseModel::mEstimator::Huber::Create(k),
                                               noise);
    case RobustKernel::kTukey:
      return gtsam::noiseModel::Robust::Create(gtsam::noiseModel::mEstimator::Tukey::Create(k),
                                               noise);
    case RobustKernel::kNone:
      break;
  }
  return noise;
}

/**
 * @brief Interpolates AHRS-derived orientation at a target timestamp via SLERP.
 * @param ahrs_msgs Time-sorted AHRS structs bracketing the target time.
 * @param target_time The desired interpolation timestamp.
 * @return The interpolated rotation as a GTSAM Rot3.
 */
gtsam::Rot3 getInterpolatedOrientation(
    const std::deque<std::shared_ptr<utils::AhrsData>>& ahrs_msgs, double target_time) {
  auto it_after = std::lower_bound(ahrs_msgs.begin(), ahrs_msgs.end(), target_time,
                                   [](const auto& msg, double t) { return msg->timestamp < t; });

  if (it_after == ahrs_msgs.begin()) {
    return ahrs_msgs.front()->orientation;
  }

  // If past the last message, extrapolate into the future
  if (it_after == ahrs_msgs.end()) {
    if (ahrs_msgs.size() < 2) {
      return ahrs_msgs.back()->orientation;
    }
    it_after--;
  }

  double t1 = (*(it_after - 1))->timestamp;
  double t2 = (*it_after)->timestamp;
  double denominator = t2 - t1;

  if (std::abs(denominator) < 1e-9) {
    return (*(it_after - 1))->orientation;
  }

  double alpha = (target_time - t1) / denominator;

  // Use Slerp for quaternion interpolation (handles alpha > 1.0 for extrapolation)
  return (*(it_after - 1))->orientation.slerp(alpha, (*it_after)->orientation);
}

}  // namespace

FactorGraphCore::FactorGraphCore(const factor_graph_node::Params& params) : params_(params) {}

void FactorGraphCore::initialize(const StateInitializer& state_init, const utils::TfBundle& tfs) {
  tfs_ = tfs;

  prev_pose_ = state_init.getPose();
  prev_vel_ = state_init.getVelocity();
  prev_imu_bias_ = state_init.getBias();
  prev_time_ = state_init.getTime();

  last_imu_acc_ = state_init.getInitialImu()->linear_acceleration;
  last_imu_gyr_ = state_init.getInitialImu()->angular_velocity;

  // --- Build Initial Graph ---
  gtsam::NonlinearFactorGraph initial_graph;
  gtsam::Values initial_values;
  addPriorFactors(state_init, initial_graph, initial_values);

  if (params_.publish_smoothed_path) {
    time_to_key_[static_cast<int64_t>(prev_time_ * 1e9)] = X(0);
  }

  // --- Initialize Preintegrators ---
  imu_preintegrator_ = std::make_unique<gtsam::PreintegratedCombinedMeasurements>(
      configureImuPreintegration(state_init), prev_imu_bias_);

  if (params_.comparison.enable_loose_dvl_preintegration) {
    dvl_loose_preintegrator_ = std::make_unique<DvlLoosePreintegrator>();
    dvl_loose_preintegrator_->reset(prev_pose_.rotation());
  } else if (params_.comparison.enable_tight_dvl_preintegration) {
    dvl_tight_preintegrator_ = std::make_unique<DvlTightPreintegrator>();
    dvl_tight_preintegrator_->reset();
  }

  if (dvl_loose_preintegrator_ || dvl_tight_preintegrator_) {
    if (params_.dvl.enable_dvl) {
      last_dvl_velocity_ = state_init.getInitialDvl()->linear_velocity;
      last_dvl_covariance_ = resolveCov<3>(
          params_.dvl.use_parameter_covariance,
          params_.dvl.parameter_covariance.velocity_noise_sigmas, params_.dvl.covariance_scalar,
          state_init.getInitialDvl()->twist_covariance.topLeftCorner<3, 3>());
    } else {
      last_dvl_velocity_ = gtsam::Vector3::Zero();
      last_dvl_covariance_ =
          sigmasSquaredDiag(params_.dvl.parameter_covariance.velocity_noise_sigmas) *
          params_.dvl.covariance_scalar;
    }
  }

  // --- Initialize Smoother ---
  gtsam::IncrementalFixedLagSmoother::KeyTimestampMap initial_timestamps;
  initial_timestamps[X(0)] = prev_time_;
  initial_timestamps[V(0)] = prev_time_;
  initial_timestamps[B(0)] = prev_time_;

  gtsam::ISAM2Params isam2_params;
  isam2_params.relinearizeThreshold = params_.relinearize_threshold;
  isam2_params.relinearizeSkip = params_.relinearize_skip;

  switch (parseSolverType(params_.solver_type)) {
    case SolverType::kIsam2:
      isam_ = std::make_unique<gtsam::ISAM2>(isam2_params);
      isam_->update(initial_graph, initial_values);
      break;
    case SolverType::kLevenbergMarquardt:
      lm_graph_ = initial_graph;
      lm_values_ = initial_values;
      break;
    case SolverType::kIncrementalFixedLagSmoother:
      inc_smoother_ =
          std::make_unique<gtsam::IncrementalFixedLagSmoother>(params_.smoother_lag, isam2_params);
      inc_smoother_->update(initial_graph, initial_values, initial_timestamps);
      break;
  }
}

std::shared_ptr<gtsam::PreintegratedCombinedMeasurements::Params>
FactorGraphCore::configureImuPreintegration(const StateInitializer& state_init) const {
  auto imu_params = gtsam::PreintegratedCombinedMeasurements::Params::MakeSharedU();
  imu_params->n_gravity =
      gtsam::Vector3(params_.imu.gravity[0], params_.imu.gravity[1], params_.imu.gravity[2]);
  imu_params->body_P_sensor = tfs_.target_T_imu;

  imu_params->accelerometerCovariance = resolveCov<3>(
      params_.imu.use_parameter_covariance, params_.imu.parameter_covariance.accel_noise_sigmas,
      params_.imu.covariance_scalar, state_init.getInitialImu()->linear_acceleration_covariance);
  imu_params->gyroscopeCovariance = resolveCov<3>(
      params_.imu.use_parameter_covariance, params_.imu.parameter_covariance.gyro_noise_sigmas,
      params_.imu.covariance_scalar, state_init.getInitialImu()->angular_velocity_covariance);
  imu_params->biasAccCovariance = sigmasSquaredDiag(params_.imu.accel_bias_rw_sigmas);
  imu_params->biasOmegaCovariance = sigmasSquaredDiag(params_.imu.gyro_bias_rw_sigmas);
  imu_params->biasAccOmegaInt = gtsam::Matrix66::Zero();
  imu_params->integrationCovariance =
      gtsam::Matrix33::Identity() * params_.imu.integration_covariance;

  return imu_params;
}

void FactorGraphCore::addPriorFactors(const StateInitializer& state_init,
                                      gtsam::NonlinearFactorGraph& graph, gtsam::Values& values) {
  // Add initial pose prior
  gtsam::Vector6 prior_pose_sigmas =
      (gtsam::Vector6() << params_.prior.parameter_priors.initial_orientation_sigmas[0],
       params_.prior.parameter_priors.initial_orientation_sigmas[1],
       params_.prior.parameter_priors.initial_orientation_sigmas[2],
       params_.prior.parameter_priors.initial_position_sigmas[0],
       params_.prior.parameter_priors.initial_position_sigmas[1],
       params_.prior.parameter_priors.initial_position_sigmas[2])
          .finished();

  if (!params_.prior.use_parameter_priors) {
    // Add initial position prior
    if (params_.gps.enable_gps) {
      const gtsam::Matrix2 gps_cov = resolveCov<2>(
          params_.gps.use_parameter_covariance,
          params_.gps.parameter_covariance.position_noise_sigmas, params_.gps.covariance_scalar,
          state_init.getInitialGps()->pose_covariance.topLeftCorner<2, 2>());
      prior_pose_sigmas(3) = std::sqrt(gps_cov(0, 0));
      prior_pose_sigmas(4) = std::sqrt(gps_cov(1, 1));
    } else {
      prior_pose_sigmas(3) = params_.prior.parameter_priors.initial_position_sigmas[0];
      prior_pose_sigmas(4) = params_.prior.parameter_priors.initial_position_sigmas[1];
    }

    if (params_.depth.enable_depth) {
      prior_pose_sigmas(5) = std::sqrt(resolveVar(
          params_.depth.use_parameter_covariance,
          params_.depth.parameter_covariance.position_z_noise_sigma,
          params_.depth.covariance_scalar, state_init.getInitialDepth()->pose_covariance(2, 2)));
    } else {
      prior_pose_sigmas(5) = params_.prior.parameter_priors.initial_position_sigmas[2];
    }

    // Add initial orientation prior
    if (params_.ahrs.enable_ahrs) {
      const gtsam::Matrix3 ahrs_cov = resolveCov<3>(
          params_.ahrs.use_parameter_covariance,
          params_.ahrs.parameter_covariance.orientation_noise_sigmas,
          params_.ahrs.covariance_scalar, state_init.getInitialAhrs()->orientation_covariance);
      prior_pose_sigmas(0) = std::sqrt(ahrs_cov(0, 0));
      prior_pose_sigmas(1) = std::sqrt(ahrs_cov(1, 1));
      prior_pose_sigmas(2) = std::sqrt(ahrs_cov(2, 2));
    } else {
      if (params_.mag.enable_mag) {
        double h_mag = std::sqrt(params_.mag.reference_field[0] * params_.mag.reference_field[0] +
                                 params_.mag.reference_field[1] * params_.mag.reference_field[1]);
        double mag_sigma_norm =
            std::sqrt(resolveVar(params_.mag.use_parameter_covariance,
                                 params_.mag.parameter_covariance.magnetic_field_noise_sigmas[0],
                                 params_.mag.covariance_scalar,
                                 state_init.getInitialMag()->magnetic_field_covariance(0, 0)));
        prior_pose_sigmas(2) = mag_sigma_norm / h_mag;
      }
      const gtsam::Matrix3 accel_cov = resolveCov<3>(
          params_.imu.use_parameter_covariance, params_.imu.parameter_covariance.accel_noise_sigmas,
          params_.imu.covariance_scalar,
          state_init.getInitialImu()->linear_acceleration_covariance);
      const double g_norm =
          std::max(Eigen::Map<const Eigen::Vector3d>(params_.imu.gravity.data()).norm(), 1e-6);
      const auto& bias_sig = params_.prior.initial_accel_bias_sigmas;
      prior_pose_sigmas(0) = std::sqrt(accel_cov(1, 1) + bias_sig[1] * bias_sig[1]) / g_norm;
      prior_pose_sigmas(1) = std::sqrt(accel_cov(0, 0) + bias_sig[0] * bias_sig[0]) / g_norm;
    }
  }

  graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
      X(0), prev_pose_, gtsam::noiseModel::Diagonal::Sigmas(prior_pose_sigmas));
  values.insert(X(0), prev_pose_);

  // Add initial velocity prior
  gtsam::SharedNoiseModel prior_vel_noise;
  if (!params_.prior.use_parameter_priors && params_.dvl.enable_dvl) {
    // Rotate DVL covariance into the map frame
    gtsam::Matrix33 dvl_cov = resolveCov<3>(
        params_.dvl.use_parameter_covariance,
        params_.dvl.parameter_covariance.velocity_noise_sigmas, params_.dvl.covariance_scalar,
        state_init.getInitialDvl()->twist_covariance.topLeftCorner<3, 3>());
    gtsam::Matrix33 map_R_dvl = (prev_pose_.rotation() * tfs_.target_T_dvl.rotation()).matrix();
    prior_vel_noise =
        gtsam::noiseModel::Gaussian::Covariance(map_R_dvl * dvl_cov * map_R_dvl.transpose());
  } else {
    const auto& sigmas = params_.prior.parameter_priors.initial_velocity_sigmas;
    prior_vel_noise =
        gtsam::noiseModel::Diagonal::Sigmas(Eigen::Map<const Eigen::Vector3d>(sigmas.data()));
  }

  graph.emplace_shared<gtsam::PriorFactor<gtsam::Vector3>>(V(0), prev_vel_, prior_vel_noise);
  values.insert(V(0), prev_vel_);

  // Add initial IMU bias prior
  gtsam::Vector6 prior_imu_bias_sigmas =
      (gtsam::Vector6() << Eigen::Map<const Eigen::Vector3d>(
           params_.prior.initial_accel_bias_sigmas.data()),
       Eigen::Map<const Eigen::Vector3d>(params_.prior.initial_gyro_bias_sigmas.data()))
          .finished();

  gtsam::SharedNoiseModel prior_imu_bias_noise =
      gtsam::noiseModel::Diagonal::Sigmas(prior_imu_bias_sigmas);

  graph.emplace_shared<gtsam::PriorFactor<gtsam::imuBias::ConstantBias>>(B(0), prev_imu_bias_,
                                                                         prior_imu_bias_noise);
  values.insert(B(0), prev_imu_bias_);
}

void FactorGraphCore::addGpsFactor(
    gtsam::NonlinearFactorGraph& graph,
    const std::deque<std::shared_ptr<utils::OdometryData>>& gps_msgs) {
  if (gps_msgs.empty()) {
    return;
  }

  const auto& gps_msg = gps_msgs.back();

  gtsam::SharedNoiseModel gps_noise = gtsam::noiseModel::Gaussian::Covariance(resolveCov<2>(
      params_.gps.use_parameter_covariance, params_.gps.parameter_covariance.position_noise_sigmas,
      params_.gps.covariance_scalar, gps_msg->pose_covariance.topLeftCorner<2, 2>()));

  gps_noise = applyRobustKernel(gps_noise, params_.gps.robust_kernel, params_.gps.robust_k);

  graph.emplace_shared<Gps2dFactorArm>(X(current_step_), gps_msg->pose.translation(),
                                       tfs_.target_T_gps, gps_noise);
}

void FactorGraphCore::addDepthFactor(
    gtsam::NonlinearFactorGraph& graph,
    const std::deque<std::shared_ptr<utils::OdometryData>>& depth_msgs) {
  if (depth_msgs.empty()) {
    return;
  }

  const auto& depth_msg = depth_msgs.back();

  const double depth_sigma =
      std::sqrt(resolveVar(params_.depth.use_parameter_covariance,
                           params_.depth.parameter_covariance.position_z_noise_sigma,
                           params_.depth.covariance_scalar, depth_msg->pose_covariance(2, 2)));
  gtsam::SharedNoiseModel depth_noise = gtsam::noiseModel::Isotropic::Sigma(1, depth_sigma);

  depth_noise = applyRobustKernel(depth_noise, params_.depth.robust_kernel, params_.depth.robust_k);

  graph.emplace_shared<DepthFactorArm>(X(current_step_), depth_msg->pose.translation().z(),
                                       tfs_.target_T_depth, depth_noise);
}

void FactorGraphCore::addAhrsFactor(gtsam::NonlinearFactorGraph& graph,
                                    const std::deque<std::shared_ptr<utils::AhrsData>>& ahrs_msgs) {
  if (ahrs_msgs.empty()) {
    return;
  }

  const auto& ahrs_msg = ahrs_msgs.back();

  gtsam::SharedNoiseModel ahrs_noise = gtsam::noiseModel::Gaussian::Covariance(
      resolveCov<3>(params_.ahrs.use_parameter_covariance,
                    params_.ahrs.parameter_covariance.orientation_noise_sigmas,
                    params_.ahrs.covariance_scalar, ahrs_msg->orientation_covariance));

  ahrs_noise = applyRobustKernel(ahrs_noise, params_.ahrs.robust_kernel, params_.ahrs.robust_k);

  graph.emplace_shared<AhrsFactorArm>(X(current_step_), ahrs_msg->orientation, tfs_.target_T_ahrs,
                                      params_.ahrs.mag_declination_radians, ahrs_noise);
}

void FactorGraphCore::addMagFactor(
    gtsam::NonlinearFactorGraph& graph,
    const std::deque<std::shared_ptr<utils::MagneticFieldData>>& mag_msgs) {
  if (mag_msgs.empty()) {
    return;
  }

  const auto& mag_msg = mag_msgs.back();

  // IMPORTANT! The reference field must be in the world frame (ENU), not NED.
  // If getting values from NOAA (NED), convert as: [Y, X, -Z].
  gtsam::Point3 ref_vec(params_.mag.reference_field[0], params_.mag.reference_field[1],
                        params_.mag.reference_field[2]);

  gtsam::SharedNoiseModel mag_noise = gtsam::noiseModel::Gaussian::Covariance(
      resolveCov<3>(params_.mag.use_parameter_covariance,
                    params_.mag.parameter_covariance.magnetic_field_noise_sigmas,
                    params_.mag.covariance_scalar, mag_msg->magnetic_field_covariance));

  mag_noise = applyRobustKernel(mag_noise, params_.mag.robust_kernel, params_.mag.robust_k);

  graph.emplace_shared<MagFactorArm>(X(current_step_), mag_msg->magnetic_field, ref_vec,
                                     tfs_.target_T_mag, mag_noise);
}

void FactorGraphCore::addDvlFactor(gtsam::NonlinearFactorGraph& graph,
                                   const std::deque<std::shared_ptr<utils::TwistData>>& dvl_msgs,
                                   const gtsam::Vector3& held_imu_gyr) {
  if (dvl_msgs.empty()) {
    return;
  }

  const auto& dvl_msg = dvl_msgs.back();

  gtsam::SharedNoiseModel dvl_noise = gtsam::noiseModel::Gaussian::Covariance(resolveCov<3>(
      params_.dvl.use_parameter_covariance, params_.dvl.parameter_covariance.velocity_noise_sigmas,
      params_.dvl.covariance_scalar, dvl_msg->twist_covariance.topLeftCorner<3, 3>()));

  dvl_noise = applyRobustKernel(dvl_noise, params_.dvl.robust_kernel, params_.dvl.robust_k);

  graph.emplace_shared<DvlFactorArm>(X(current_step_), V(current_step_), B(current_step_),
                                     tfs_.target_T_dvl, tfs_.target_T_imu, dvl_msg->linear_velocity,
                                     held_imu_gyr, dvl_noise);
}

void FactorGraphCore::addConstVelFactor(gtsam::NonlinearFactorGraph& graph, double target_time) {
  double dt = target_time - prev_time_;
  Eigen::Vector3d vel_random_walk =
      Eigen::Map<const Eigen::Vector3d>(params_.const_vel.prediction_noise_sigmas.data()) *
      std::sqrt(params_.const_vel.covariance_scalar);
  double sqrt_dt = std::sqrt(std::max(dt, 0.001));
  Eigen::Vector3d scaled_sigma = vel_random_walk * sqrt_dt;

  gtsam::SharedNoiseModel zero_accel_noise = gtsam::noiseModel::Diagonal::Sigmas(scaled_sigma);

  graph.emplace_shared<ConstVelFactor>(X(prev_step_), V(prev_step_), X(current_step_),
                                       V(current_step_), zero_accel_noise);
}

void FactorGraphCore::addAuvDynamicsFactor(
    gtsam::NonlinearFactorGraph& graph,
    const std::deque<std::shared_ptr<utils::WrenchData>>& wrench_msgs, double target_time) {
  // Implements a zero-order hold (ZOH) for wrench commands
  if (!wrench_msgs.empty()) {
    last_wrench_msg_ = wrench_msgs.back();
  }

  if (!last_wrench_msg_) {
    return;
  }

  const auto& wrench_msg = last_wrench_msg_;

  double dt = target_time - prev_time_;
  double sqrt_dt = std::sqrt(std::max(dt, 0.001));
  gtsam::Vector3 dynamics_sigmas =
      Eigen::Map<const Eigen::Vector3d>(params_.dynamics.prediction_noise_sigmas.data()) *
      std::sqrt(params_.dynamics.covariance_scalar) * sqrt_dt;
  gtsam::SharedNoiseModel dynamics_noise = gtsam::noiseModel::Diagonal::Sigmas(dynamics_sigmas);

  dynamics_noise =
      applyRobustKernel(dynamics_noise, params_.dynamics.robust_kernel, params_.dynamics.robust_k);

  graph.emplace_shared<AuvDynamicsFactorArm>(
      X(prev_step_), V(prev_step_), X(current_step_), V(current_step_), dt, wrench_msg->force,
      tfs_.target_T_com,
      gtsam::Matrix33(Eigen::Map<const Eigen::Vector3d>(params_.dynamics.mass.data()).asDiagonal()),
      gtsam::Matrix33(
          Eigen::Map<const Eigen::Vector3d>(params_.dynamics.linear_drag.data()).asDiagonal()),
      gtsam::Matrix33(
          Eigen::Map<const Eigen::Vector3d>(params_.dynamics.quad_drag.data()).asDiagonal()),
      dynamics_noise);
}

void FactorGraphCore::addImuPreintFactor(
    gtsam::NonlinearFactorGraph& graph, const std::deque<std::shared_ptr<utils::ImuData>>& imu_msgs,
    double target_time) {
  // Implements a zero-order hold (ZOH) for IMU measurements
  if (!imu_preintegrator_ || imu_msgs.empty()) {
    return;
  }

  double last_imu_time = prev_time_;

  for (const auto& imu_msg : imu_msgs) {
    double current_imu_time = imu_msg->timestamp;

    if (current_imu_time <= last_imu_time) {
      continue;
    }

    double dt = current_imu_time - last_imu_time;
    if (dt > 1e-9) {
      imu_preintegrator_->integrateMeasurement(last_imu_acc_, last_imu_gyr_, dt);
    }

    last_imu_acc_ = imu_msg->linear_acceleration;
    last_imu_gyr_ = imu_msg->angular_velocity;
    last_imu_time = current_imu_time;
  }

  // Extra measurement to reach exact target time
  if (last_imu_time < target_time) {
    double dt = target_time - last_imu_time;
    if (dt > 1e-6) {
      imu_preintegrator_->integrateMeasurement(last_imu_acc_, last_imu_gyr_, dt);
    }
    last_imu_time = target_time;
  }

  graph.emplace_shared<gtsam::CombinedImuFactor>(X(prev_step_), V(prev_step_), X(current_step_),
                                                 V(current_step_), B(prev_step_), B(current_step_),
                                                 *imu_preintegrator_);
}

void FactorGraphCore::addDvlLoosePreintFactor(
    gtsam::NonlinearFactorGraph& graph,
    const std::deque<std::shared_ptr<utils::TwistData>>& dvl_msgs,
    const std::deque<std::shared_ptr<utils::AhrsData>>& ahrs_msgs, double target_time) {
  // Implements a zero-order hold (ZOH) for DVL velocity measurements
  if (!dvl_loose_preintegrator_ || ahrs_msgs.empty()) {
    return;
  }

  double last_dvl_time = prev_time_;

  gtsam::Rot3 target_R_ahrs = tfs_.target_T_ahrs.rotation();
  gtsam::Rot3 ahrs_R_target = target_R_ahrs.inverse();
  gtsam::Rot3 target_R_dvl = tfs_.target_T_dvl.rotation();

  // Propagate AHRS orientation uncertainty into the preintegrated translation covariance
  gtsam::Matrix3 ahrs_cov =
      resolveCov<3>(params_.ahrs.use_parameter_covariance,
                    params_.ahrs.parameter_covariance.orientation_noise_sigmas,
                    params_.ahrs.covariance_scalar, ahrs_msgs.back()->orientation_covariance);

  gtsam::Rot3 map_R_ahrs_prev = getInterpolatedOrientation(ahrs_msgs, prev_time_);
  gtsam::Rot3 map_R_target_prev = map_R_ahrs_prev * ahrs_R_target;
  dvl_loose_preintegrator_->reset(map_R_target_prev, target_R_ahrs, target_R_dvl, ahrs_cov);

  for (const auto& dvl_msg : dvl_msgs) {
    double current_dvl_time = dvl_msg->timestamp;
    if (current_dvl_time <= last_dvl_time) {
      continue;
    }

    double dt = current_dvl_time - last_dvl_time;
    if (dt > 1e-9) {
      // Integrate DVL measurement alongside the midpoint-interpolated AHRS attitude
      double mid_time = 0.5 * (last_dvl_time + current_dvl_time);
      gtsam::Rot3 map_R_ahrs_mid = getInterpolatedOrientation(ahrs_msgs, mid_time);
      gtsam::Rot3 map_R_target_mid = map_R_ahrs_mid * ahrs_R_target;
      gtsam::Rot3 map_R_dvl_mid = map_R_target_mid * target_R_dvl;

      dvl_loose_preintegrator_->integrateMeasurement(last_dvl_velocity_, map_R_dvl_mid, dt,
                                                     last_dvl_covariance_);

      last_dvl_velocity_ = dvl_msg->linear_velocity;

      last_dvl_covariance_ = resolveCov<3>(params_.dvl.use_parameter_covariance,
                                           params_.dvl.parameter_covariance.velocity_noise_sigmas,
                                           params_.dvl.covariance_scalar,
                                           dvl_msg->twist_covariance.topLeftCorner<3, 3>());
    }
    last_dvl_time = current_dvl_time;
  }

  // Extra measurement to reach exact target time
  if (last_dvl_time < target_time) {
    double dt = target_time - last_dvl_time;
    if (dt > 1e-6) {
      double mid_time = 0.5 * (last_dvl_time + target_time);
      gtsam::Rot3 map_R_ahrs_mid = getInterpolatedOrientation(ahrs_msgs, mid_time);
      gtsam::Rot3 map_R_target_mid = map_R_ahrs_mid * ahrs_R_target;
      gtsam::Rot3 map_R_dvl_mid = map_R_target_mid * target_R_dvl;
      dvl_loose_preintegrator_->integrateMeasurement(last_dvl_velocity_, map_R_dvl_mid, dt,
                                                     last_dvl_covariance_);
    }
    last_dvl_time = target_time;
  }

  graph.emplace_shared<DvlLoosePreintFactorArm>(
      X(prev_step_), X(current_step_), tfs_.target_T_dvl, dvl_loose_preintegrator_->delta(),
      gtsam::noiseModel::Gaussian::Covariance(dvl_loose_preintegrator_->covariance()));
}

void FactorGraphCore::addDvlTightPreintFactor(
    gtsam::NonlinearFactorGraph& graph,
    const std::deque<std::shared_ptr<utils::TwistData>>& dvl_msgs,
    const std::deque<std::shared_ptr<utils::ImuData>>& imu_msgs, double target_time,
    const gtsam::Vector3& held_imu_acc, const gtsam::Vector3& held_imu_gyr) {
  // Implements a zero-order hold (ZOH) for DVL velocity measurements
  if (!dvl_tight_preintegrator_ || imu_msgs.empty()) {
    return;
  }

  double last_dvl_time = prev_time_;
  double last_imu_time = prev_time_;

  gtsam::Rot3 target_R_dvl = tfs_.target_T_dvl.rotation();

  gtsam::PreintegratedCombinedMeasurements temp_imu_preint(*imu_preintegrator_);
  temp_imu_preint.resetIntegrationAndSetBias(prev_imu_bias_);

  gtsam::Vector3 current_imu_acc = held_imu_acc;
  gtsam::Vector3 current_imu_gyr = held_imu_gyr;
  auto imu_it = imu_msgs.begin();

  dvl_tight_preintegrator_->reset();

  auto stepImuPreintegrator = [&](double t_end) {
    while (imu_it != imu_msgs.end()) {
      double imu_time = (*imu_it)->timestamp;
      if (imu_time > t_end) {
        break;
      }
      if (imu_time > last_imu_time) {
        double dt_imu = imu_time - last_imu_time;
        temp_imu_preint.integrateMeasurement(current_imu_acc, current_imu_gyr, dt_imu);
        last_imu_time = imu_time;
      }
      current_imu_acc = (*imu_it)->linear_acceleration;
      current_imu_gyr = (*imu_it)->angular_velocity;
      imu_it++;
    }

    if (last_imu_time < t_end) {
      double dt_rem = t_end - last_imu_time;
      if (dt_rem > 1e-6) {
        temp_imu_preint.integrateMeasurement(current_imu_acc, current_imu_gyr, dt_rem);
      }
      last_imu_time = t_end;
    }
  };

  // Method derived from Forster et al. 2017, IEEE T-RO
  // TODO: Review this paper more in depth
  auto integrateDvlMeasurement = [&](double mid_time, double dt) {
    stepImuPreintegrator(mid_time);

    gtsam::Rot3 delta_R_ik = temp_imu_preint.deltaRij();
    gtsam::Matrix3 Jr = gtsam::Rot3::ExpmapDerivative(gtsam::Vector3(temp_imu_preint.theta()));
    gtsam::Matrix3 rot_cov_k =
        Jr * temp_imu_preint.preintMeasCov().block<3, 3>(0, 0) * Jr.transpose();
    gtsam::Matrix3 J_bg_k = Jr * temp_imu_preint.preintegrated_H_biasOmega().topRows<3>();

    dvl_tight_preintegrator_->integrateMeasurement(last_dvl_velocity_, delta_R_ik, target_R_dvl, dt,
                                                   last_dvl_covariance_, rot_cov_k, J_bg_k);
  };

  for (const auto& dvl_msg : dvl_msgs) {
    double current_dvl_time = dvl_msg->timestamp;
    if (current_dvl_time <= last_dvl_time) {
      continue;
    }

    double dt = current_dvl_time - last_dvl_time;
    if (dt > 1e-9) {
      integrateDvlMeasurement(0.5 * (last_dvl_time + current_dvl_time), dt);

      last_dvl_velocity_ = dvl_msg->linear_velocity;
      last_dvl_covariance_ = resolveCov<3>(params_.dvl.use_parameter_covariance,
                                           params_.dvl.parameter_covariance.velocity_noise_sigmas,
                                           params_.dvl.covariance_scalar,
                                           dvl_msg->twist_covariance.topLeftCorner<3, 3>());
    }
    last_dvl_time = current_dvl_time;
  }

  // Extra measurement to reach exact target time
  if (last_dvl_time < target_time) {
    double dt = target_time - last_dvl_time;
    if (dt > 1e-6) {
      integrateDvlMeasurement(0.5 * (last_dvl_time + target_time), dt);
    }
    last_dvl_time = target_time;
  }

  graph.emplace_shared<DvlTightPreintFactorArm>(
      X(prev_step_), X(current_step_), B(prev_step_), tfs_.target_T_dvl,
      dvl_tight_preintegrator_->delta(), dvl_tight_preintegrator_->preintMeasDerivativeWrtBias(),
      prev_imu_bias_.gyroscope(),
      gtsam::noiseModel::Gaussian::Covariance(dvl_tight_preintegrator_->covariance()));
}

std::optional<utils::QueueBundle> FactorGraphCore::update(double target_time,
                                                          utils::QueueBundle& queues) {
  if (target_time <= prev_time_ + 1e-6) {
    return std::nullopt;
  }

  // Sort sensor messages by timestamp
  auto by_time = [](const auto& a, const auto& b) { return a->timestamp < b->timestamp; };
  std::sort(queues.imu.begin(), queues.imu.end(), by_time);
  std::sort(queues.gps.begin(), queues.gps.end(), by_time);
  std::sort(queues.depth.begin(), queues.depth.end(), by_time);
  std::sort(queues.mag.begin(), queues.mag.end(), by_time);
  std::sort(queues.ahrs.begin(), queues.ahrs.end(), by_time);
  std::sort(queues.dvl.begin(), queues.dvl.end(), by_time);
  std::sort(queues.wrench.begin(), queues.wrench.end(), by_time);

  if (queues.imu.empty() || queues.imu.front()->timestamp > target_time) {
    return std::nullopt;
  }

  // --- Build Factor Graph ---
  gtsam::NonlinearFactorGraph new_graph;
  gtsam::Values new_values;
  gtsam::IncrementalFixedLagSmoother::KeyTimestampMap new_timestamps;

  utils::QueueBundle leftover;

  std::scoped_lock state_lock(state_mutex_);

  const gtsam::Vector3 held_imu_acc = last_imu_acc_;
  const gtsam::Vector3 held_imu_gyr = last_imu_gyr_;

  // Re-queue messages newer than the keyframe for the next update
  auto splitAfterTarget = [target_time](auto& msgs, auto& split) {
    while (!msgs.empty() && msgs.back()->timestamp > target_time) {
      split.push_front(msgs.back());
      msgs.pop_back();
    }
  };
  splitAfterTarget(queues.imu, leftover.imu);
  splitAfterTarget(queues.gps, leftover.gps);
  splitAfterTarget(queues.depth, leftover.depth);
  splitAfterTarget(queues.mag, leftover.mag);
  splitAfterTarget(queues.ahrs, leftover.ahrs);
  splitAfterTarget(queues.dvl, leftover.dvl);
  splitAfterTarget(queues.wrench, leftover.wrench);

  if (params_.comparison.enable_loose_dvl_preintegration && !queues.ahrs.empty()) {
    leftover.ahrs.push_front(queues.ahrs.back());
  }

  addImuPreintFactor(new_graph, queues.imu, target_time);
  if (params_.gps.enable_gps) {
    addGpsFactor(new_graph, queues.gps);
  }
  if (params_.depth.enable_depth) {
    addDepthFactor(new_graph, queues.depth);
  }
  if (params_.mag.enable_mag) {
    addMagFactor(new_graph, queues.mag);
  }
  if (params_.ahrs.enable_ahrs) {
    addAhrsFactor(new_graph, queues.ahrs);
  }

  // DVL dropout handling
  auto addDropoutFactors = [&](gtsam::NonlinearFactorGraph& g) {
    bool use_dynamics =
        params_.dynamics.enable_dynamics || params_.dynamics.enable_dynamics_dropout_only;
    bool use_const_vel =
        params_.const_vel.enable_const_vel || params_.const_vel.enable_const_vel_dropout_only;

    if (use_dynamics) {
      addAuvDynamicsFactor(g, queues.wrench, target_time);
    } else if (use_const_vel) {
      addConstVelFactor(g, target_time);
    }
  };

  if (queues.dvl.empty() || !params_.dvl.enable_dvl) {
    addDropoutFactors(new_graph);
  } else {
    if (params_.comparison.enable_loose_dvl_preintegration) {
      if (queues.ahrs.empty()) {
        addDropoutFactors(new_graph);
        last_dvl_velocity_ = queues.dvl.back()->linear_velocity;
      } else {
        addDvlLoosePreintFactor(new_graph, queues.dvl, queues.ahrs, target_time);
      }
    } else if (params_.comparison.enable_tight_dvl_preintegration) {
      addDvlTightPreintFactor(new_graph, queues.dvl, queues.imu, target_time, held_imu_acc,
                              held_imu_gyr);
    } else {
      addDvlFactor(new_graph, queues.dvl, last_imu_gyr_);

      if (params_.dynamics.enable_dynamics) {
        addAuvDynamicsFactor(new_graph, queues.wrench, target_time);
      } else if (params_.const_vel.enable_const_vel) {
        addConstVelFactor(new_graph, target_time);
      }
    }
  }

  // --- Add State Predictions ---
  auto pred = imu_preintegrator_->predict(gtsam::NavState(prev_pose_, prev_vel_), prev_imu_bias_);
  new_values.insert(X(current_step_), pred.pose());
  new_values.insert(V(current_step_), pred.velocity());
  new_values.insert(B(current_step_), prev_imu_bias_);
  new_timestamps[X(current_step_)] = target_time;
  new_timestamps[V(current_step_)] = target_time;
  new_timestamps[B(current_step_)] = target_time;

  if (!inc_smoother_ && !isam_) {
    prev_pose_ = pred.pose();
    prev_vel_ = pred.velocity();
  }

  // --- Reset Preintegrators ---
  imu_preintegrator_->resetIntegrationAndSetBias(prev_imu_bias_);

  if (params_.publish_smoothed_path) {
    time_to_key_[static_cast<int64_t>(target_time * 1e9)] = X(current_step_);
    if (inc_smoother_) {
      time_to_key_.erase(time_to_key_.begin(), time_to_key_.lower_bound(static_cast<int64_t>(
                                                   (target_time - params_.smoother_lag) * 1e9)));
    }
  }

  prev_time_ = target_time;
  prev_step_ = current_step_;
  current_step_++;

  // --- Add Graph to Buffer ---
  buffer_graph_ += new_graph;
  buffer_values_.insert(new_values);
  buffer_timestamps_.insert(new_timestamps.begin(), new_timestamps.end());
  buffer_target_time_ = target_time;
  buffer_last_step_ = prev_step_;
  has_buffer_ = true;

  return leftover;
}

std::optional<OptimizeResult> FactorGraphCore::optimize() {
  // --- Load Graph from Buffer ---
  gtsam::NonlinearFactorGraph batch_graph;
  gtsam::Values batch_values;
  gtsam::IncrementalFixedLagSmoother::KeyTimestampMap batch_timestamps;
  double batch_target_time{0.0};
  size_t batch_last_step = 0;

  {
    std::scoped_lock state_lock(state_mutex_);
    if (!has_buffer_) {
      return std::nullopt;
    }

    batch_graph = std::move(buffer_graph_);
    batch_values = std::move(buffer_values_);
    batch_timestamps = std::move(buffer_timestamps_);
    batch_target_time = buffer_target_time_;
    batch_last_step = buffer_last_step_;

    buffer_graph_ = gtsam::NonlinearFactorGraph();
    buffer_values_ = gtsam::Values();
    buffer_timestamps_.clear();
    has_buffer_ = false;
  }

  OptimizeResult result;
  result.target_time = batch_target_time;

  // --- Detect Processing Overflow ---
  result.num_keyframes = batch_timestamps.size() / 3;
  if (result.num_keyframes > 1) {
    result.processing_overflow = true;
  }

  // --- Smoother Optimization ---
  auto total_start = std::chrono::steady_clock::now();
  result.new_factors = batch_graph.size();

  if (inc_smoother_) {
    auto smoother_start = std::chrono::steady_clock::now();
    inc_smoother_->update(batch_graph, batch_values, batch_timestamps);
    auto smoother_end = std::chrono::steady_clock::now();
    result.smoother_duration = std::chrono::duration<double>(smoother_end - smoother_start).count();

    {
      std::scoped_lock state_lock(state_mutex_);
      prev_pose_ = inc_smoother_->calculateEstimate<gtsam::Pose3>(X(batch_last_step));
      prev_vel_ = inc_smoother_->calculateEstimate<gtsam::Vector3>(V(batch_last_step));
      prev_imu_bias_ =
          inc_smoother_->calculateEstimate<gtsam::imuBias::ConstantBias>(B(batch_last_step));
    }

    if (params_.publish_diagnostics || params_.publish_graph_metrics) {
      result.total_factors = inc_smoother_->getFactors().nrFactors();
      result.total_variables = inc_smoother_->getLinearizationPoint().size();
    }

  } else if (isam_) {
    auto smoother_start = std::chrono::steady_clock::now();
    isam_->update(batch_graph, batch_values);
    auto smoother_end = std::chrono::steady_clock::now();
    result.smoother_duration = std::chrono::duration<double>(smoother_end - smoother_start).count();

    {
      std::scoped_lock state_lock(state_mutex_);
      prev_pose_ = isam_->calculateEstimate<gtsam::Pose3>(X(batch_last_step));
      prev_vel_ = isam_->calculateEstimate<gtsam::Vector3>(V(batch_last_step));
      prev_imu_bias_ = isam_->calculateEstimate<gtsam::imuBias::ConstantBias>(B(batch_last_step));
    }

    if (params_.publish_diagnostics || params_.publish_graph_metrics) {
      result.total_factors = isam_->getFactorsUnsafe().nrFactors();
      result.total_variables = isam_->getLinearizationPoint().size();
    }
  } else {
    lm_graph_.push_back(batch_graph.begin(), batch_graph.end());
    lm_values_.insert(batch_values);

    auto smoother_start = std::chrono::steady_clock::now();
    gtsam::LevenbergMarquardtParams lm_params;
    gtsam::LevenbergMarquardtOptimizer optimizer(lm_graph_, lm_values_, lm_params);
    lm_values_ = optimizer.optimize();
    auto smoother_end = std::chrono::steady_clock::now();
    result.smoother_duration = std::chrono::duration<double>(smoother_end - smoother_start).count();

    {
      std::scoped_lock state_lock(state_mutex_);
      prev_pose_ = lm_values_.at<gtsam::Pose3>(X(batch_last_step));
      prev_vel_ = lm_values_.at<gtsam::Vector3>(V(batch_last_step));
      prev_imu_bias_ = lm_values_.at<gtsam::imuBias::ConstantBias>(B(batch_last_step));
    }

    if (params_.publish_diagnostics || params_.publish_graph_metrics) {
      result.total_factors = lm_graph_.nrFactors();
      result.total_variables = lm_values_.size();
    }
  }

  {
    std::scoped_lock state_lock(state_mutex_);
    result.pose = prev_pose_;
    result.velocity = prev_vel_;
    result.imu_bias = prev_imu_bias_;
  }

  // --- Calculate Covariances ---
  auto cov_start = std::chrono::steady_clock::now();

  auto marginal_cov = [&](bool enabled, gtsam::Key key, int dim) -> gtsam::Matrix {
    if (enabled) {
      if (inc_smoother_) {
        return inc_smoother_->marginalCovariance(key);
      }
      if (isam_) {
        return isam_->marginalCovariance(key);
      }
    }
    return gtsam::Matrix::Identity(dim, dim) * -1.0;
  };
  result.pose_cov = marginal_cov(params_.publish_pose_cov, X(batch_last_step), 6);
  result.vel_cov =
      marginal_cov(params_.publish_velocity && params_.publish_velocity_cov, V(batch_last_step), 3);
  result.bias_cov =
      marginal_cov(params_.publish_imu_bias && params_.publish_imu_bias_cov, B(batch_last_step), 6);

  auto cov_end = std::chrono::steady_clock::now();
  result.cov_duration = std::chrono::duration<double>(cov_end - cov_start).count();

  // --- Export Smoothed Path ---
  if (params_.publish_smoothed_path) {
    if (inc_smoother_) {
      result.all_estimates = inc_smoother_->calculateEstimate();
    } else if (isam_) {
      result.all_estimates = isam_->calculateEstimate();
    } else {
      result.all_estimates = lm_values_;
    }
  }

  auto total_end = std::chrono::steady_clock::now();
  result.total_duration = std::chrono::duration<double>(total_end - total_start).count();

  return result;
}

std::map<int64_t, gtsam::Key> FactorGraphCore::snapshotTimeKeys() const {
  std::scoped_lock lock(state_mutex_);
  return time_to_key_;
}

}  // namespace coug_fgo
