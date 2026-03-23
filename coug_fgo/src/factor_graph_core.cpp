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
 * @date Jan 2026
 */

#include "coug_fgo/factor_graph_core.hpp"

#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/slam/PriorFactor.h>

#include <algorithm>
#include <chrono>

#include "coug_fgo/factors/ahrs_factor.hpp"
#include "coug_fgo/factors/auv_dynamics_factor.hpp"
#include "coug_fgo/factors/const_vel_factor.hpp"
#include "coug_fgo/factors/depth_factor.hpp"
#include "coug_fgo/factors/dvl_factor.hpp"
#include "coug_fgo/factors/dvl_loose_preint_factor.hpp"
#include "coug_fgo/factors/dvl_tight_preint_factor.hpp"
#include "coug_fgo/factors/gps_factor.hpp"
#include "coug_fgo/factors/mag_factor.hpp"
#include "coug_fgo/utils/conversions.hpp"
#include "coug_fgo/utils/dvl_tight_preintegrator.hpp"

using coug_fgo::factors::AhrsYawFactorArm;
using coug_fgo::factors::AuvDynamicsFactorArm;
using coug_fgo::factors::ConstVelFactor;
using coug_fgo::factors::DepthFactorArm;
using coug_fgo::factors::DvlFactorArm;
using coug_fgo::factors::DvlLoosePreintFactorArm;
using coug_fgo::factors::DvlTightPreintFactorArm;
using coug_fgo::factors::Gps2dFactorArm;
using coug_fgo::factors::MagFactorArm;
using coug_fgo::utils::DvlLoosePreintegrator;
using coug_fgo::utils::DvlTightPreintegrator;
using coug_fgo::utils::toGtsam;
using coug_fgo::utils::toGtsam3x3;
using coug_fgo::utils::toGtsamDiagonal;
using coug_fgo::utils::toGtsamSquaredDiagonal;

using gtsam::symbol_shorthand::B;  // Bias (ax,ay,az,gx,gy,gz)
using gtsam::symbol_shorthand::V;  // Velocity (x,y,z)
using gtsam::symbol_shorthand::X;  // Pose3 (x,y,z,r,p,y)

namespace coug_fgo {

static const auto kLogger = rclcpp::get_logger("factor_graph_core");

FactorGraphCore::FactorGraphCore(const factor_graph_node::Params& params) : params_(params) {}

static gtsam::SharedNoiseModel applyRobustKernel(const gtsam::SharedNoiseModel& noise,
                                                 const std::string& kernel, double k) {
  if (kernel == "Huber") {
    return gtsam::noiseModel::Robust::Create(gtsam::noiseModel::mEstimator::Huber::Create(k),
                                             noise);
  } else if (kernel == "Tukey") {
    return gtsam::noiseModel::Robust::Create(gtsam::noiseModel::mEstimator::Tukey::Create(k),
                                             noise);
  }
  return noise;
}

void FactorGraphCore::initialize(const utils::StateInitializer& state_init,
                                 const utils::TfBundle& tfs) {
  tfs_ = tfs;

  prev_pose_ = state_init.pose_;
  prev_vel_ = state_init.velocity_;
  prev_imu_bias_ = state_init.bias_;
  prev_time_ = state_init.time_;

  last_imu_acc_ = toGtsam(state_init.initial_imu_->linear_acceleration);
  last_imu_gyr_ = toGtsam(state_init.initial_imu_->angular_velocity);

  // --- Build Initial Graph ---
  gtsam::NonlinearFactorGraph initial_graph;
  gtsam::Values initial_values;
  addPriorFactors(state_init, initial_graph, initial_values);

  if (params_.publish_smoothed_path) {
    time_to_key[state_init.time_] = X(0);
  }

  // --- Initialize Preintegrators ---
  imu_preintegrator_ = std::make_unique<gtsam::PreintegratedCombinedMeasurements>(
      configureImuPreintegration(state_init), prev_imu_bias_);

  if (params_.comparison.enable_loose_dvl_preintegration) {
    dvl_loose_preintegrator_ = std::make_unique<utils::DvlLoosePreintegrator>();
    dvl_loose_preintegrator_->reset(prev_pose_.rotation());

    last_dvl_velocity_ = toGtsam(state_init.initial_dvl_->twist.twist.linear);
    if (params_.dvl.use_parameter_covariance) {
      last_dvl_covariance_ =
          toGtsamSquaredDiagonal(params_.dvl.parameter_covariance.velocity_noise_sigmas)
              .block<3, 3>(0, 0);
    } else {
      last_dvl_covariance_ = toGtsam3x3(state_init.initial_dvl_->twist.covariance);
    }
  } else if (params_.comparison.enable_tight_dvl_preintegration) {
    dvl_tight_preintegrator_ = std::make_unique<utils::DvlTightPreintegrator>();
    dvl_tight_preintegrator_->reset();

    last_dvl_velocity_ = toGtsam(state_init.initial_dvl_->twist.twist.linear);
    if (params_.dvl.use_parameter_covariance) {
      last_dvl_covariance_ =
          toGtsamSquaredDiagonal(params_.dvl.parameter_covariance.velocity_noise_sigmas)
              .block<3, 3>(0, 0);
    } else {
      last_dvl_covariance_ = toGtsam3x3(state_init.initial_dvl_->twist.covariance);
    }
  }

  // --- Initialize Smoother ---
  gtsam::IncrementalFixedLagSmoother::KeyTimestampMap initial_timestamps;
  initial_timestamps[X(0)] = prev_time_.seconds();
  initial_timestamps[V(0)] = prev_time_.seconds();
  initial_timestamps[B(0)] = prev_time_.seconds();

  gtsam::ISAM2Params isam2_params;
  isam2_params.relinearizeThreshold = params_.relinearize_threshold;
  isam2_params.relinearizeSkip = params_.relinearize_skip;

  if (params_.solver_type == "ISAM2") {
    isam_ = std::make_unique<gtsam::ISAM2>(isam2_params);
    isam_->update(initial_graph, initial_values);
  } else {
    inc_smoother_ =
        std::make_unique<gtsam::IncrementalFixedLagSmoother>(params_.smoother_lag, isam2_params);
    inc_smoother_->update(initial_graph, initial_values, initial_timestamps);
  }
}

std::shared_ptr<gtsam::PreintegratedCombinedMeasurements::Params>
FactorGraphCore::configureImuPreintegration(const utils::StateInitializer& state_init) {
  auto imu_params = gtsam::PreintegratedCombinedMeasurements::Params::MakeSharedU();
  imu_params->n_gravity =
      gtsam::Vector3(params_.imu.gravity[0], params_.imu.gravity[1], params_.imu.gravity[2]);
  imu_params->body_P_sensor = tfs_.target_T_imu;

  if (params_.imu.use_parameter_covariance) {
    imu_params->accelerometerCovariance =
        toGtsamSquaredDiagonal(params_.imu.parameter_covariance.accel_noise_sigmas)
            .block<3, 3>(0, 0);
  } else {
    imu_params->accelerometerCovariance =
        toGtsam(state_init.initial_imu_->linear_acceleration_covariance);
  }

  if (params_.imu.use_parameter_covariance) {
    imu_params->gyroscopeCovariance =
        toGtsamSquaredDiagonal(params_.imu.parameter_covariance.gyro_noise_sigmas)
            .block<3, 3>(0, 0);
  } else {
    imu_params->gyroscopeCovariance = toGtsam(state_init.initial_imu_->angular_velocity_covariance);
  }
  imu_params->biasAccCovariance = toGtsamSquaredDiagonal(params_.imu.accel_bias_rw_sigmas);
  imu_params->biasOmegaCovariance = toGtsamSquaredDiagonal(params_.imu.gyro_bias_rw_sigmas);
  imu_params->biasAccOmegaInt = gtsam::Matrix66::Zero();
  imu_params->integrationCovariance =
      gtsam::Matrix33::Identity() * params_.imu.integration_covariance;

  return imu_params;
}

void FactorGraphCore::addPriorFactors(const utils::StateInitializer& state_init,
                                      gtsam::NonlinearFactorGraph& graph, gtsam::Values& values) {
  // Add initial pose prior
  gtsam::Vector6 prior_pose_sigmas = gtsam::Vector6::Zero();
  prior_pose_sigmas << params_.prior.parameter_priors.initial_orientation_sigmas[0],
      params_.prior.parameter_priors.initial_orientation_sigmas[1],
      params_.prior.parameter_priors.initial_orientation_sigmas[2],
      params_.prior.parameter_priors.initial_position_sigmas[0],
      params_.prior.parameter_priors.initial_position_sigmas[1],
      params_.prior.parameter_priors.initial_position_sigmas[2];

  if (!params_.prior.use_parameter_priors) {
    // Add initial position prior
    if (params_.gps.enable_gps) {
      prior_pose_sigmas(3) =
          params_.gps.use_parameter_covariance
              ? params_.gps.parameter_covariance.position_noise_sigmas[0]
              : std::sqrt(toGtsam(state_init.initial_gps_->pose.covariance)(0, 0));
      prior_pose_sigmas(4) =
          params_.gps.use_parameter_covariance
              ? params_.gps.parameter_covariance.position_noise_sigmas[1]
              : std::sqrt(toGtsam(state_init.initial_gps_->pose.covariance)(1, 1));
    }
    prior_pose_sigmas(5) = params_.depth.use_parameter_covariance
                               ? params_.depth.parameter_covariance.position_z_noise_sigma
                               : std::sqrt(state_init.initial_depth_->pose.covariance[14]);

    // Add initial orientation prior
    if (params_.ahrs.enable_ahrs) {
      prior_pose_sigmas(2) =
          params_.ahrs.use_parameter_covariance
              ? params_.ahrs.parameter_covariance.yaw_noise_sigma
              : std::sqrt(toGtsam(state_init.initial_ahrs_->orientation_covariance)(2, 2));
    } else if (params_.mag.enable_mag) {
      double h_mag = std::sqrt(params_.mag.reference_field[0] * params_.mag.reference_field[0] +
                               params_.mag.reference_field[1] * params_.mag.reference_field[1]);
      double mag_sigma_norm =
          params_.mag.use_parameter_covariance
              ? params_.mag.parameter_covariance.magnetic_field_noise_sigmas[0]
              : std::sqrt(toGtsam(state_init.initial_mag_->magnetic_field_covariance)(0, 0));
      prior_pose_sigmas(2) = mag_sigma_norm / h_mag;
    }
    prior_pose_sigmas(0) =
        params_.imu.use_parameter_covariance
            ? params_.imu.parameter_covariance.gyro_noise_sigmas[0]
            : std::sqrt(toGtsam(state_init.initial_imu_->angular_velocity_covariance)(0, 0));
    prior_pose_sigmas(1) =
        params_.imu.use_parameter_covariance
            ? params_.imu.parameter_covariance.gyro_noise_sigmas[1]
            : std::sqrt(toGtsam(state_init.initial_imu_->angular_velocity_covariance)(1, 1));
  }

  graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
      X(0), prev_pose_, gtsam::noiseModel::Diagonal::Sigmas(prior_pose_sigmas));
  values.insert(X(0), prev_pose_);

  // Add initial velocity prior
  gtsam::SharedNoiseModel prior_vel_noise;
  if (params_.prior.use_parameter_priors) {
    auto& sigmas = params_.prior.parameter_priors.initial_velocity_sigmas;
    prior_vel_noise = gtsam::noiseModel::Diagonal::Sigmas(toGtsam(sigmas));
  } else {
    if (params_.dvl.use_parameter_covariance) {
      auto& sigmas = params_.dvl.parameter_covariance.velocity_noise_sigmas;
      prior_vel_noise = gtsam::noiseModel::Diagonal::Sigmas(toGtsam(sigmas));
    } else {
      gtsam::Matrix33 dvl_cov = toGtsam3x3(state_init.initial_dvl_->twist.covariance);
      prior_vel_noise = gtsam::noiseModel::Diagonal::Covariance(dvl_cov);
    }
  }

  graph.emplace_shared<gtsam::PriorFactor<gtsam::Vector3>>(V(0), prev_vel_, prior_vel_noise);
  values.insert(V(0), prev_vel_);

  // Add initial IMU bias prior
  gtsam::Vector6 prior_imu_bias_sigmas = gtsam::Vector6::Zero();
  gtsam::Vector3 accel_sigmas = toGtsam(params_.prior.initial_accel_bias_sigmas);
  gtsam::Vector3 gyro_sigmas = toGtsam(params_.prior.initial_gyro_bias_sigmas);
  prior_imu_bias_sigmas << accel_sigmas, gyro_sigmas;

  gtsam::SharedNoiseModel prior_imu_bias_noise =
      gtsam::noiseModel::Diagonal::Sigmas(prior_imu_bias_sigmas);

  graph.emplace_shared<gtsam::PriorFactor<gtsam::imuBias::ConstantBias>>(B(0), prev_imu_bias_,
                                                                         prior_imu_bias_noise);
  values.insert(B(0), prev_imu_bias_);
}

void FactorGraphCore::addGpsFactor(gtsam::NonlinearFactorGraph& graph,
                                   const std::deque<nav_msgs::msg::Odometry::SharedPtr>& gps_msgs) {
  if (gps_msgs.empty()) {
    return;
  }

  const auto& gps_msg = gps_msgs.back();

  gtsam::SharedNoiseModel gps_noise;
  if (params_.gps.use_parameter_covariance) {
    gps_noise = gtsam::noiseModel::Diagonal::Sigmas(
        toGtsam(params_.gps.parameter_covariance.position_noise_sigmas));
  } else {
    gtsam::Matrix22 gps_cov = toGtsam3x3(gps_msg->pose.covariance).block<2, 2>(0, 0);
    gps_noise = gtsam::noiseModel::Gaussian::Covariance(gps_cov);
  }

  gps_noise = applyRobustKernel(gps_noise, params_.gps.robust_kernel, params_.gps.robust_k);

  RCLCPP_DEBUG(kLogger, "Adding GPS factor at step %zu", current_step_);

  graph.emplace_shared<Gps2dFactorArm>(X(current_step_), toGtsam(gps_msg->pose.pose.position),
                                       tfs_.target_T_gps, gps_noise);
}

void FactorGraphCore::addDepthFactor(
    gtsam::NonlinearFactorGraph& graph,
    const std::deque<nav_msgs::msg::Odometry::SharedPtr>& depth_msgs) {
  if (depth_msgs.empty()) {
    return;
  }

  const auto& depth_msg = depth_msgs.back();

  gtsam::SharedNoiseModel depth_noise;
  if (params_.depth.use_parameter_covariance) {
    double depth_sigma = params_.depth.parameter_covariance.position_z_noise_sigma;
    depth_noise = gtsam::noiseModel::Isotropic::Sigma(1, depth_sigma);
  } else {
    gtsam::Matrix11 depth_cov = gtsam::Matrix11::Zero();
    depth_cov << depth_msg->pose.covariance[14];
    depth_noise = gtsam::noiseModel::Gaussian::Covariance(depth_cov);
  }

  depth_noise = applyRobustKernel(depth_noise, params_.depth.robust_kernel, params_.depth.robust_k);

  RCLCPP_DEBUG(kLogger, "Adding depth factor at step %zu", current_step_);

  graph.emplace_shared<DepthFactorArm>(X(current_step_), depth_msg->pose.pose.position.z,
                                       tfs_.target_T_depth, depth_noise);
}

void FactorGraphCore::addAhrsFactor(gtsam::NonlinearFactorGraph& graph,
                                    const std::deque<sensor_msgs::msg::Imu::SharedPtr>& ahrs_msgs) {
  if (ahrs_msgs.empty()) {
    return;
  }

  const auto& ahrs_msg = ahrs_msgs.back();

  gtsam::SharedNoiseModel ahrs_noise;
  if (params_.ahrs.use_parameter_covariance) {
    gtsam::Vector1 ahrs_sigmas;
    ahrs_sigmas << params_.ahrs.parameter_covariance.yaw_noise_sigma;
    ahrs_noise = gtsam::noiseModel::Diagonal::Sigmas(ahrs_sigmas);
  } else {
    gtsam::Matrix11 ahrs_cov = gtsam::Matrix11::Zero();
    ahrs_cov << ahrs_msg->orientation_covariance[8];
    ahrs_noise = gtsam::noiseModel::Gaussian::Covariance(ahrs_cov);
  }

  ahrs_noise = applyRobustKernel(ahrs_noise, params_.ahrs.robust_kernel, params_.ahrs.robust_k);

  RCLCPP_DEBUG(kLogger, "Adding AHRS factor at step %zu", current_step_);

  graph.emplace_shared<AhrsYawFactorArm>(X(current_step_), toGtsam(ahrs_msg->orientation),
                                         tfs_.target_T_ahrs.rotation(),
                                         params_.ahrs.mag_declination_radians, ahrs_noise);
}

void FactorGraphCore::addMagFactor(
    gtsam::NonlinearFactorGraph& graph,
    const std::deque<sensor_msgs::msg::MagneticField::SharedPtr>& mag_msgs) {
  if (mag_msgs.empty()) {
    return;
  }

  const auto& mag_msg = mag_msgs.back();

  // IMPORTANT! The reference field must be in the world frame (ENU), not NED.
  // If getting values from NOAA (NED), convert as: [Y, X, -Z].
  gtsam::Point3 ref_vec(params_.mag.reference_field[0], params_.mag.reference_field[1],
                        params_.mag.reference_field[2]);

  gtsam::SharedNoiseModel mag_noise;
  if (params_.mag.use_parameter_covariance) {
    mag_noise = gtsam::noiseModel::Diagonal::Sigmas(
        toGtsam(params_.mag.parameter_covariance.magnetic_field_noise_sigmas));
  } else {
    gtsam::Matrix33 mag_cov = toGtsam(mag_msg->magnetic_field_covariance);
    mag_noise = gtsam::noiseModel::Gaussian::Covariance(mag_cov);
  }

  mag_noise = applyRobustKernel(mag_noise, params_.mag.robust_kernel, params_.mag.robust_k);

  RCLCPP_DEBUG(kLogger, "Adding mag factor at step %zu", current_step_);

  graph.emplace_shared<MagFactorArm>(X(current_step_), toGtsam(mag_msg->magnetic_field), ref_vec,
                                     tfs_.target_T_mag.rotation(), mag_noise);
}

void FactorGraphCore::addDvlFactor(
    gtsam::NonlinearFactorGraph& graph,
    const std::deque<geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr>& dvl_msgs) {
  if (dvl_msgs.empty()) {
    return;
  }

  const auto& dvl_msg = dvl_msgs.back();

  gtsam::SharedNoiseModel dvl_noise;
  if (params_.dvl.use_parameter_covariance) {
    dvl_noise = gtsam::noiseModel::Diagonal::Sigmas(
        toGtsam(params_.dvl.parameter_covariance.velocity_noise_sigmas));
  } else {
    gtsam::Matrix33 dvl_cov = toGtsam3x3(dvl_msg->twist.covariance);
    dvl_noise = gtsam::noiseModel::Gaussian::Covariance(dvl_cov);
  }

  dvl_noise = applyRobustKernel(dvl_noise, params_.dvl.robust_kernel, params_.dvl.robust_k);

  RCLCPP_DEBUG(kLogger, "Adding DVL factor at step %zu", current_step_);

  graph.emplace_shared<DvlFactorArm>(X(current_step_), V(current_step_), tfs_.target_T_dvl,
                                     toGtsam(dvl_msg->twist.twist.linear), dvl_noise);
}

void FactorGraphCore::addConstVelFactor(gtsam::NonlinearFactorGraph& graph,
                                        const rclcpp::Time& target_time) {
  double dt = (target_time - prev_time_).seconds();
  Eigen::Vector3d vel_random_walk = toGtsam(params_.const_vel.velocity_sigma);
  double sqrt_dt = std::sqrt(std::max(dt, 0.001));
  Eigen::Vector3d scaled_sigma = vel_random_walk * sqrt_dt;

  gtsam::SharedNoiseModel zero_accel_noise = gtsam::noiseModel::Diagonal::Sigmas(scaled_sigma);

  RCLCPP_DEBUG(kLogger, "Adding constant velocity factor at step %zu", current_step_);

  graph.emplace_shared<ConstVelFactor>(X(prev_step_), V(prev_step_), X(current_step_),
                                       V(current_step_), zero_accel_noise);
}

void FactorGraphCore::addAuvDynamicsFactor(
    gtsam::NonlinearFactorGraph& graph,
    const std::deque<geometry_msgs::msg::WrenchStamped::SharedPtr>& wrench_msgs,
    const rclcpp::Time& target_time) {
  // Implements a zero-order hold (ZOH) for wrench commands
  if (!wrench_msgs.empty()) {
    last_wrench_msg_ = wrench_msgs.back();
  }

  if (!last_wrench_msg_) {
    return;
  }

  const auto& wrench_msg = last_wrench_msg_;

  gtsam::Vector3 dynamics_sigmas = toGtsam(params_.dynamics.prediction_noise_sigmas);
  gtsam::SharedNoiseModel dynamics_noise = gtsam::noiseModel::Diagonal::Sigmas(dynamics_sigmas);

  dynamics_noise =
      applyRobustKernel(dynamics_noise, params_.dynamics.robust_kernel, params_.dynamics.robust_k);

  double dt = (target_time - prev_time_).seconds();
  RCLCPP_DEBUG(kLogger, "Adding dynamics factor at step %zu", current_step_);
  graph.emplace_shared<coug_fgo::factors::AuvDynamicsFactorArm>(
      X(prev_step_), V(prev_step_), X(current_step_), V(current_step_), dt,
      toGtsam(wrench_msg->wrench.force), tfs_.target_T_com,
      toGtsamDiagonal(params_.dynamics.mass).block<3, 3>(0, 0),
      toGtsamDiagonal(params_.dynamics.linear_drag).block<3, 3>(0, 0),
      toGtsamDiagonal(params_.dynamics.quad_drag).block<3, 3>(0, 0), dynamics_noise);
}

void FactorGraphCore::addImuPreintFactor(
    gtsam::NonlinearFactorGraph& graph,
    const std::deque<sensor_msgs::msg::Imu::SharedPtr>& imu_msgs, const rclcpp::Time& target_time,
    std::deque<sensor_msgs::msg::Imu::SharedPtr>& unused_imu) {
  if (!imu_preintegrator_ || imu_msgs.empty()) {
    return;
  }

  rclcpp::Time last_imu_time = prev_time_;

  for (const auto& imu_msg : imu_msgs) {
    rclcpp::Time current_imu_time(imu_msg->header.stamp);
    if (current_imu_time > target_time) {
      unused_imu.push_back(imu_msg);
      continue;
    }

    if (current_imu_time <= last_imu_time) {
      RCLCPP_DEBUG(kLogger, "IMU message older than last integrated time. Skipping.");
      continue;
    }

    double dt = (current_imu_time - last_imu_time).seconds();
    if (dt > 1e-9) {
      imu_preintegrator_->integrateMeasurement(last_imu_acc_, last_imu_gyr_, dt);
    }

    last_imu_acc_ = toGtsam(imu_msg->linear_acceleration);
    last_imu_gyr_ = toGtsam(imu_msg->angular_velocity);
    last_imu_time = current_imu_time;
  }

  // Extra measurement to reach exact target time
  if (last_imu_time < target_time) {
    double dt = (target_time - last_imu_time).seconds();
    if (dt > 1e-6) {
      imu_preintegrator_->integrateMeasurement(last_imu_acc_, last_imu_gyr_, dt);
    }
    last_imu_time = target_time;
  }

  graph.emplace_shared<gtsam::CombinedImuFactor>(X(prev_step_), V(prev_step_), X(current_step_),
                                                 V(current_step_), B(prev_step_), B(current_step_),
                                                 *imu_preintegrator_);

  RCLCPP_DEBUG(kLogger, "Adding preintegrated IMU factor at step %zu", current_step_);
}

gtsam::Rot3 FactorGraphCore::getInterpolatedOrientation(
    const std::deque<sensor_msgs::msg::Imu::SharedPtr>& ahrs_msgs,
    const rclcpp::Time& target_time) {
  auto it_after = std::lower_bound(
      ahrs_msgs.begin(), ahrs_msgs.end(), target_time,
      [](const auto& msg, const rclcpp::Time& t) { return rclcpp::Time(msg->header.stamp) < t; });

  if (it_after == ahrs_msgs.begin()) {
    return toGtsam(ahrs_msgs.front()->orientation);
  }

  // If past the last message, extrapolate into the future
  if (it_after == ahrs_msgs.end()) {
    if (ahrs_msgs.size() < 2) {
      return toGtsam(ahrs_msgs.back()->orientation);
    }
    it_after--;
  }

  rclcpp::Time t1((*(it_after - 1))->header.stamp);
  rclcpp::Time t2((*it_after)->header.stamp);
  double denominator = (t2 - t1).seconds();

  if (std::abs(denominator) < 1e-9) {
    return toGtsam((*(it_after - 1))->orientation);
  }

  double alpha = (target_time - t1).seconds() / denominator;

  // Use Slerp for quaternion interpolation (handles alpha > 1.0 for extrapolation)
  return toGtsam((*(it_after - 1))->orientation).slerp(alpha, toGtsam((*it_after)->orientation));
}

void FactorGraphCore::addDvlLoosePreintFactor(
    gtsam::NonlinearFactorGraph& graph,
    const std::deque<geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr>& dvl_msgs,
    const std::deque<sensor_msgs::msg::Imu::SharedPtr>& ahrs_msgs, const rclcpp::Time& target_time,
    std::deque<geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr>& unused_dvl) {
  // Implements a zero-order hold (ZOH) for DVL velocity measurements
  if (!dvl_loose_preintegrator_ || ahrs_msgs.empty()) {
    return;
  }

  rclcpp::Time last_dvl_time = prev_time_;

  gtsam::Rot3 target_R_ahrs = tfs_.target_T_ahrs.rotation();
  gtsam::Rot3 ahrs_R_target = target_R_ahrs.inverse();
  gtsam::Rot3 target_R_dvl = tfs_.target_T_dvl.rotation();

  gtsam::Rot3 map_R_ahrs_prev = getInterpolatedOrientation(ahrs_msgs, prev_time_);
  gtsam::Rot3 map_R_target_prev = map_R_ahrs_prev * ahrs_R_target;
  dvl_loose_preintegrator_->reset(map_R_target_prev);

  for (const auto& dvl_msg : dvl_msgs) {
    rclcpp::Time current_dvl_time(dvl_msg->header.stamp);
    if (current_dvl_time > target_time) {
      unused_dvl.push_back(dvl_msg);
      continue;
    }

    if (current_dvl_time <= last_dvl_time) {
      RCLCPP_DEBUG(kLogger, "DVL message older than last integrated time. Skipping.");
      continue;
    }

    double dt = (current_dvl_time - last_dvl_time).seconds();
    if (dt > 1e-9) {
      // Integrate DVL measurement alongside interpolated AHRS attitude
      gtsam::Rot3 map_R_ahrs_cur = getInterpolatedOrientation(ahrs_msgs, current_dvl_time);
      gtsam::Rot3 map_R_target_cur = map_R_ahrs_cur * ahrs_R_target;
      gtsam::Rot3 map_R_dvl_cur = map_R_target_cur * target_R_dvl;

      dvl_loose_preintegrator_->integrateMeasurement(last_dvl_velocity_, map_R_dvl_cur, dt,
                                                     last_dvl_covariance_);

      last_dvl_velocity_ = toGtsam(dvl_msg->twist.twist.linear);

      if (params_.dvl.use_parameter_covariance) {
        last_dvl_covariance_ =
            toGtsamSquaredDiagonal(params_.dvl.parameter_covariance.velocity_noise_sigmas)
                .block<3, 3>(0, 0);
      } else {
        last_dvl_covariance_ = toGtsam3x3(dvl_msg->twist.covariance);
      }
    }
    last_dvl_time = current_dvl_time;
  }

  if (last_dvl_time < target_time) {
    RCLCPP_DEBUG(kLogger, "No valid DVL measurements found.");
  }

  // Extra measurement to reach exact target time
  if (last_dvl_time < target_time) {
    double dt = (target_time - last_dvl_time).seconds();
    if (dt > 1e-6) {
      gtsam::Rot3 cur_ahrs_att = getInterpolatedOrientation(ahrs_msgs, target_time);
      gtsam::Rot3 cur_target_att = cur_ahrs_att * ahrs_R_target;
      gtsam::Rot3 cur_dvl_att = cur_target_att * target_R_dvl;
      dvl_loose_preintegrator_->integrateMeasurement(last_dvl_velocity_, cur_dvl_att, dt,
                                                     last_dvl_covariance_);
    }
    last_dvl_time = target_time;
  }

  RCLCPP_DEBUG(kLogger, "Adding loose preint DVL factor at step %zu", current_step_);

  graph.emplace_shared<DvlLoosePreintFactorArm>(
      X(prev_step_), X(current_step_), tfs_.target_T_dvl, dvl_loose_preintegrator_->delta(),
      gtsam::noiseModel::Gaussian::Covariance(dvl_loose_preintegrator_->covariance()));
}

void FactorGraphCore::addDvlTightPreintFactor(
    gtsam::NonlinearFactorGraph& graph,
    const std::deque<geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr>& dvl_msgs,
    const std::deque<sensor_msgs::msg::Imu::SharedPtr>& imu_msgs, const rclcpp::Time& target_time,
    std::deque<geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr>& unused_dvl) {
  // Implements a zero-order hold (ZOH) for DVL velocity measurements
  if (!dvl_tight_preintegrator_ || imu_msgs.empty()) {
    return;
  }

  rclcpp::Time last_dvl_time = prev_time_;
  rclcpp::Time last_imu_time = prev_time_;

  gtsam::Rot3 target_R_imu = tfs_.target_T_imu.rotation();
  gtsam::Rot3 target_R_dvl = tfs_.target_T_dvl.rotation();
  gtsam::Rot3 imu_R_dvl = target_R_imu.inverse() * target_R_dvl;

  auto temp_imu_preint =
      std::make_unique<gtsam::PreintegratedCombinedMeasurements>(*imu_preintegrator_);
  temp_imu_preint->resetIntegrationAndSetBias(prev_imu_bias_);

  gtsam::Vector3 current_imu_acc = last_imu_acc_;
  gtsam::Vector3 current_imu_gyr = last_imu_gyr_;
  auto imu_it = imu_msgs.begin();

  dvl_tight_preintegrator_->reset();

  auto stepImuPreintegrator = [&](const rclcpp::Time& t_end) {
    while (imu_it != imu_msgs.end()) {
      rclcpp::Time imu_time((*imu_it)->header.stamp);
      if (imu_time > t_end) {
        break;
      }
      if (imu_time > last_imu_time) {
        double dt_imu = (imu_time - last_imu_time).seconds();
        temp_imu_preint->integrateMeasurement(current_imu_acc, current_imu_gyr, dt_imu);
        last_imu_time = imu_time;
      }
      current_imu_acc = toGtsam((*imu_it)->linear_acceleration);
      current_imu_gyr = toGtsam((*imu_it)->angular_velocity);
      imu_it++;
    }

    if (last_imu_time < t_end) {
      double dt_rem = (t_end - last_imu_time).seconds();
      if (dt_rem > 1e-6) {
        temp_imu_preint->integrateMeasurement(current_imu_acc, current_imu_gyr, dt_rem);
      }
      last_imu_time = t_end;
    }
  };

  for (const auto& dvl_msg : dvl_msgs) {
    rclcpp::Time current_dvl_time(dvl_msg->header.stamp);
    if (current_dvl_time > target_time) {
      unused_dvl.push_back(dvl_msg);
      continue;
    }

    if (current_dvl_time <= last_dvl_time) {
      RCLCPP_DEBUG(kLogger, "DVL message older than last integrated time. Skipping.");
      continue;
    }

    double dt = (current_dvl_time - last_dvl_time).seconds();
    if (dt > 1e-9) {
      stepImuPreintegrator(current_dvl_time);

      // Extract preintegrated IMU relative rotation and Jacobians
      gtsam::Rot3 delta_R_ik = temp_imu_preint->deltaRij();
      gtsam::Matrix3 rot_cov_k = temp_imu_preint->preintMeasCov().block<3, 3>(0, 0);

      gtsam::Matrix96 H_bias;
      temp_imu_preint->biasCorrectedDelta(prev_imu_bias_, H_bias);
      gtsam::Matrix3 J_bg_k = H_bias.block<3, 3>(0, 3);

      dvl_tight_preintegrator_->integrateMeasurement(last_dvl_velocity_, delta_R_ik, imu_R_dvl, dt,
                                                     last_dvl_covariance_, rot_cov_k, J_bg_k);

      last_dvl_velocity_ = toGtsam(dvl_msg->twist.twist.linear);

      if (params_.dvl.use_parameter_covariance) {
        last_dvl_covariance_ =
            toGtsamSquaredDiagonal(params_.dvl.parameter_covariance.velocity_noise_sigmas)
                .block<3, 3>(0, 0);
      } else {
        last_dvl_covariance_ = toGtsam3x3(dvl_msg->twist.covariance);
      }
    }
    last_dvl_time = current_dvl_time;
  }

  if (last_dvl_time < target_time) {
    RCLCPP_DEBUG(kLogger, "No valid DVL measurements found.");
  }

  // Extra measurement to reach exact target time
  if (last_dvl_time < target_time) {
    double dt = (target_time - last_dvl_time).seconds();
    if (dt > 1e-6) {
      stepImuPreintegrator(target_time);

      // Extract preintegrated IMU relative rotation and Jacobians
      gtsam::Rot3 delta_R_ik = temp_imu_preint->deltaRij();
      gtsam::Matrix3 rot_cov_k = temp_imu_preint->preintMeasCov().block<3, 3>(0, 0);

      gtsam::Matrix96 H_bias;
      temp_imu_preint->biasCorrectedDelta(prev_imu_bias_, H_bias);
      gtsam::Matrix3 J_bg_k = H_bias.block<3, 3>(0, 3);

      dvl_tight_preintegrator_->integrateMeasurement(last_dvl_velocity_, delta_R_ik, imu_R_dvl, dt,
                                                     last_dvl_covariance_, rot_cov_k, J_bg_k);
    }
    last_dvl_time = target_time;
  }

  RCLCPP_DEBUG(kLogger, "Adding tight preint DVL factor at step %zu", current_step_);

  graph.emplace_shared<DvlTightPreintFactorArm>(
      X(prev_step_), X(current_step_), B(prev_step_), tfs_.target_T_imu, tfs_.target_T_dvl,
      dvl_tight_preintegrator_->delta(), dvl_tight_preintegrator_->preintMeasDerivativeWrtBias(),
      prev_imu_bias_.gyroscope(),
      gtsam::noiseModel::Gaussian::Covariance(dvl_tight_preintegrator_->covariance()));
}

std::optional<UpdateResult> FactorGraphCore::update(const rclcpp::Time& target_time,
                                                    utils::QueueBundle& msgs) {
  if (target_time <= prev_time_ + rclcpp::Duration::from_seconds(1e-6)) {
    RCLCPP_DEBUG(kLogger, "Duplicate or out-of-order timestamp detected. Skipping.");
    return std::nullopt;
  }

  // Sort IMU/AHRS messages
  auto by_time = [](const auto& a, const auto& b) {
    return rclcpp::Time(a->header.stamp) < rclcpp::Time(b->header.stamp);
  };
  std::sort(msgs.imu.begin(), msgs.imu.end(), by_time);
  if (params_.comparison.enable_loose_dvl_preintegration) {
    std::sort(msgs.ahrs.begin(), msgs.ahrs.end(), by_time);
  }

  // --- Build Factor Graph ---
  gtsam::NonlinearFactorGraph new_graph;
  gtsam::Values new_values;
  gtsam::IncrementalFixedLagSmoother::KeyTimestampMap new_timestamps;

  UpdateResult result;

  addImuPreintFactor(new_graph, msgs.imu, target_time, result.unused_imu);
  if (params_.gps.enable_gps) {
    addGpsFactor(new_graph, msgs.gps);
  }
  addDepthFactor(new_graph, msgs.depth);
  if (params_.mag.enable_mag) {
    addMagFactor(new_graph, msgs.mag);
  }
  if (params_.ahrs.enable_ahrs) {
    addAhrsFactor(new_graph, msgs.ahrs);
  }

  // DVL dropout handling
  auto addDropoutFactors = [&](gtsam::NonlinearFactorGraph& g) {
    bool use_dynamics =
        params_.dynamics.enable_dynamics || params_.dynamics.enable_dynamics_dropout_only;
    bool use_const_vel =
        params_.const_vel.enable_const_vel || params_.const_vel.enable_const_vel_dropout_only;

    if (use_dynamics) {
      addAuvDynamicsFactor(g, msgs.wrench, target_time);
    } else if (use_const_vel) {
      addConstVelFactor(g, target_time);
    }
  };

  if (msgs.dvl.empty()) {
    addDropoutFactors(new_graph);
  } else {
    if (params_.comparison.enable_loose_dvl_preintegration) {
      addDvlLoosePreintFactor(new_graph, msgs.dvl, msgs.ahrs, target_time, result.unused_dvl);
    } else if (params_.comparison.enable_tight_dvl_preintegration) {
      addDvlTightPreintFactor(new_graph, msgs.dvl, msgs.imu, target_time, result.unused_dvl);
    } else {
      addDvlFactor(new_graph, msgs.dvl);

      if (params_.dynamics.enable_dynamics) {
        addAuvDynamicsFactor(new_graph, msgs.wrench, target_time);
      } else if (params_.const_vel.enable_const_vel) {
        addConstVelFactor(new_graph, target_time);
      }
    }
  }

  {
    std::scoped_lock update_lock(buffer_mutex);

    // --- Add State Predictions ---
    auto pred = imu_preintegrator_->predict(gtsam::NavState(prev_pose_, prev_vel_), prev_imu_bias_);
    new_values.insert(X(current_step_), pred.pose());
    new_values.insert(V(current_step_), pred.velocity());
    new_values.insert(B(current_step_), prev_imu_bias_);
    new_timestamps[X(current_step_)] = target_time.seconds();
    new_timestamps[V(current_step_)] = target_time.seconds();
    new_timestamps[B(current_step_)] = target_time.seconds();

    // --- Reset Preintegrators ---
    imu_preintegrator_->resetIntegrationAndSetBias(prev_imu_bias_);

    if (params_.publish_smoothed_path) {
      time_to_key[target_time] = X(current_step_);
      if (!isam_) {
        time_to_key.erase(time_to_key.begin(),
                          time_to_key.lower_bound(
                              target_time - rclcpp::Duration::from_seconds(params_.smoother_lag)));
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
  }

  return result;
}

std::optional<OptimizeResult> FactorGraphCore::optimize() {
  // --- Load Graph from Buffer ---
  gtsam::NonlinearFactorGraph batch_graph;
  gtsam::Values batch_values;
  gtsam::IncrementalFixedLagSmoother::KeyTimestampMap batch_timestamps;
  rclcpp::Time batch_target_time{0, 0, RCL_ROS_TIME};
  size_t batch_last_step = 0;

  {
    std::scoped_lock update_lock(buffer_mutex);
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
  auto total_start = std::chrono::high_resolution_clock::now();
  result.new_factors = batch_graph.size();

  if (inc_smoother_) {
    auto smoother_start = std::chrono::high_resolution_clock::now();
    inc_smoother_->update(batch_graph, batch_values, batch_timestamps);
    auto smoother_end = std::chrono::high_resolution_clock::now();
    result.smoother_duration = std::chrono::duration<double>(smoother_end - smoother_start).count();

    {
      std::scoped_lock update_lock(buffer_mutex);
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
    auto smoother_start = std::chrono::high_resolution_clock::now();
    isam_->update(batch_graph, batch_values);
    auto smoother_end = std::chrono::high_resolution_clock::now();
    result.smoother_duration = std::chrono::duration<double>(smoother_end - smoother_start).count();

    {
      std::scoped_lock update_lock(buffer_mutex);
      prev_pose_ = isam_->calculateEstimate<gtsam::Pose3>(X(batch_last_step));
      prev_vel_ = isam_->calculateEstimate<gtsam::Vector3>(V(batch_last_step));
      prev_imu_bias_ = isam_->calculateEstimate<gtsam::imuBias::ConstantBias>(B(batch_last_step));
    }

    if (params_.publish_diagnostics || params_.publish_graph_metrics) {
      result.total_factors = isam_->getFactorsUnsafe().nrFactors();
      result.total_variables = isam_->getLinearizationPoint().size();
    }
  }

  result.pose = prev_pose_;
  result.velocity = prev_vel_;
  result.imu_bias = prev_imu_bias_;

  // --- Calculate Covariances ---
  auto cov_start = std::chrono::high_resolution_clock::now();
  result.pose_cov = gtsam::Matrix::Identity(6, 6) * -1.0;
  if (params_.publish_pose_cov) {
    if (inc_smoother_) {
      result.pose_cov = inc_smoother_->marginalCovariance(X(batch_last_step));
    } else if (isam_) {
      result.pose_cov = isam_->marginalCovariance(X(batch_last_step));
    }
  }

  result.vel_cov = gtsam::Matrix::Identity(3, 3) * -1.0;
  if (params_.publish_velocity && params_.publish_velocity_cov) {
    if (inc_smoother_) {
      result.vel_cov = inc_smoother_->marginalCovariance(V(batch_last_step));
    } else if (isam_) {
      result.vel_cov = isam_->marginalCovariance(V(batch_last_step));
    }
  }

  result.bias_cov = gtsam::Matrix::Identity(6, 6) * -1.0;
  if (params_.publish_imu_bias && params_.publish_imu_bias_cov) {
    if (inc_smoother_) {
      result.bias_cov = inc_smoother_->marginalCovariance(B(batch_last_step));
    } else if (isam_) {
      result.bias_cov = isam_->marginalCovariance(B(batch_last_step));
    }
  }

  auto cov_end = std::chrono::high_resolution_clock::now();
  result.cov_duration = std::chrono::duration<double>(cov_end - cov_start).count();

  // --- Export Smoothed Path ---
  if (params_.publish_smoothed_path) {
    if (inc_smoother_) {
      result.all_estimates = inc_smoother_->calculateEstimate();
    } else if (isam_) {
      result.all_estimates = isam_->calculateEstimate();
    }
  }

  auto total_end = std::chrono::high_resolution_clock::now();
  result.total_duration = std::chrono::duration<double>(total_end - total_start).count();

  return result;
}

}  // namespace coug_fgo
