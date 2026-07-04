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
 * @file state_initializer.cpp
 * @brief Implementation of the StateInitializer.
 * @author Nelson Durrant
 * @date May 2026
 */

#include "coug_fgo/state_initializer.hpp"

#include <cmath>
#include <memory>
#include <type_traits>

#include "coug_fgo/utils/param_enums.hpp"

namespace coug_fgo {

using utils::AhrsData;
using utils::ImuData;
using utils::KeyframeSource;
using utils::MagneticFieldData;
using utils::OdometryData;
using utils::parseKeyframeSource;
using utils::QueueBundle;
using utils::TfBundle;
using utils::TwistData;

StateInitializer::StateInitializer(const factor_graph_node::Params& params) : params_(params) {}

bool StateInitializer::update(double current_time, QueueBundle& queues) {
  const bool gps_req = params_.gps.enable_gps || params_.gps.enable_gps_init_only;
  const bool depth_req = params_.depth.enable_depth || params_.depth.enable_depth_init_only;
  const bool mag_req = params_.mag.enable_mag || params_.mag.enable_mag_init_only;
  const bool ahrs_req = params_.ahrs.enable_ahrs || params_.ahrs.enable_ahrs_init_only ||
                        params_.comparison.enable_loose_dvl_preintegration;
  const bool dvl_req = params_.dvl.enable_dvl || params_.dvl.enable_dvl_init_only;

  if (params_.prior.use_parameter_priors) {
    if (queues.imu.empty()) {
      return false;
    }
    initial_imu_ = queues.imu.back();

    const KeyframeSource kf = parseKeyframeSource(params_.keyframe_source);
    const KeyframeSource backup_kf = parseKeyframeSource(params_.backup_keyframe_source);

    if (depth_req && (kf == KeyframeSource::kDepth || backup_kf == KeyframeSource::kDepth)) {
      if (queues.depth.empty()) {
        return false;
      }
      initial_depth_ = queues.depth.back();
    }

    if (dvl_req && (params_.comparison.enable_loose_dvl_preintegration ||
                    params_.comparison.enable_tight_dvl_preintegration ||
                    kf == KeyframeSource::kDvl || backup_kf == KeyframeSource::kDvl)) {
      if (queues.dvl.empty()) {
        return false;
      }
      initial_dvl_ = queues.dvl.back();
    }

    return true;
  }

  const bool all_present = (imu_count_ > 0 || !queues.imu.empty()) &&
                           (!gps_req || gps_count_ > 0 || !queues.gps.empty()) &&
                           (!depth_req || depth_count_ > 0 || !queues.depth.empty()) &&
                           (!mag_req || mag_count_ > 0 || !queues.mag.empty()) &&
                           (!ahrs_req || ahrs_count_ > 0 || !queues.ahrs.empty()) &&
                           (!dvl_req || dvl_count_ > 0 || !queues.dvl.empty());
  if (!all_present) {
    return false;
  }

  if (start_avg_time_ == 0.0) {
    start_avg_time_ = current_time;
  }

  incrementAverages(queues);

  return (current_time - start_avg_time_) >= params_.prior.initialization_duration_sec;
}

void StateInitializer::compute(const TfBundle& tfs) {
  gtsam::Rot3 map_R_target = computeInitialOrientation(tfs);
  pose_ = gtsam::Pose3(map_R_target, computeInitialPosition(map_R_target, tfs));
  velocity_ = computeInitialVelocity(map_R_target, tfs);
  bias_ = computeInitialBias();
  switch (parseKeyframeSource(params_.keyframe_source)) {
    case KeyframeSource::kDvl:
      time_ = initial_dvl_->timestamp;
      break;
    case KeyframeSource::kDepth:
      time_ = initial_depth_->timestamp;
      break;
    case KeyframeSource::kTimer:
    case KeyframeSource::kNone:
      time_ = initial_imu_->timestamp;
      break;
  }
}

void StateInitializer::incrementAverages(QueueBundle& queues) {
  auto average = [](auto& initial, size_t& count, const auto& msgs, const auto& blend) {
    for (const auto& msg : msgs) {
      if (count == 0) {
        initial = std::make_shared<std::decay_t<decltype(*msg)>>(*msg);
      } else {
        blend(*initial, *msg, static_cast<double>(count + 1));
        initial->timestamp = msg->timestamp;
      }
      count++;
    }
  };

  average(initial_imu_, imu_count_, queues.imu, [](ImuData& avg, const ImuData& msg, double n) {
    avg.linear_acceleration += (msg.linear_acceleration - avg.linear_acceleration) / n;
    avg.angular_velocity += (msg.angular_velocity - avg.angular_velocity) / n;
  });

  if (params_.gps.enable_gps || params_.gps.enable_gps_init_only) {
    average(initial_gps_, gps_count_, queues.gps,
            [](OdometryData& avg, const OdometryData& msg, double n) {
              gtsam::Point3 p = avg.pose.translation();
              p += (msg.pose.translation() - p) / n;
              avg.pose = gtsam::Pose3(avg.pose.rotation(), p);
            });
  }

  if (params_.depth.enable_depth || params_.depth.enable_depth_init_only) {
    average(initial_depth_, depth_count_, queues.depth,
            [](OdometryData& avg, const OdometryData& msg, double n) {
              gtsam::Point3 p = avg.pose.translation();
              p.z() += (msg.pose.translation().z() - p.z()) / n;
              avg.pose = gtsam::Pose3(avg.pose.rotation(), p);
            });
  }

  if (params_.dvl.enable_dvl || params_.dvl.enable_dvl_init_only) {
    average(initial_dvl_, dvl_count_, queues.dvl,
            [](TwistData& avg, const TwistData& msg, double n) {
              avg.linear_velocity += (msg.linear_velocity - avg.linear_velocity) / n;
            });
  }

  if (params_.mag.enable_mag || params_.mag.enable_mag_init_only) {
    average(initial_mag_, mag_count_, queues.mag,
            [](MagneticFieldData& avg, const MagneticFieldData& msg, double n) {
              avg.magnetic_field += (msg.magnetic_field - avg.magnetic_field) / n;
            });
  }

  if (params_.ahrs.enable_ahrs || params_.ahrs.enable_ahrs_init_only) {
    // Average rotations in the tangent space anchored at the first sample
    if (ahrs_count_ == 0 && !queues.ahrs.empty()) {
      ahrs_ref_ = queues.ahrs.front()->orientation;
      ahrs_log_sum_ = gtsam::Vector3::Zero();
    }
    average(initial_ahrs_, ahrs_count_, queues.ahrs,
            [this](AhrsData& avg, const AhrsData& msg, double n) {
              ahrs_log_sum_ += gtsam::Rot3::Logmap(ahrs_ref_.between(msg.orientation));
              avg.orientation = ahrs_ref_.compose(gtsam::Rot3::Expmap(ahrs_log_sum_ / n));
            });
  }
}

gtsam::Rot3 StateInitializer::computeInitialOrientation(const TfBundle& tfs) {
  double roll = params_.prior.parameter_priors.initial_orientation[0];
  double pitch = params_.prior.parameter_priors.initial_orientation[1];
  double yaw = params_.prior.parameter_priors.initial_orientation[2];

  if (params_.prior.use_parameter_priors) {
    gtsam::Rot3 base_R_target = tfs.target_T_base.rotation().inverse();
    return gtsam::Rot3::Ypr(yaw, pitch, roll) * base_R_target;
  }

  // Account for IMU rotation
  gtsam::Vector3 accel_imu = initial_imu_->linear_acceleration;
  gtsam::Vector3 accel_target = tfs.target_T_imu.rotation().rotate(accel_imu);

  // Use gravity to estimate initial roll and pitch (tilt estimation)
  roll = std::atan2(accel_target.y(), accel_target.z());
  pitch = std::atan2(-accel_target.x(), std::sqrt(accel_target.y() * accel_target.y() +
                                                  accel_target.z() * accel_target.z()));

  // If no heading source is enabled below, yaw keeps the parameter prior value
  if (params_.ahrs.enable_ahrs || params_.ahrs.enable_ahrs_init_only) {
    // Account for AHRS sensor rotation
    gtsam::Rot3 target_R_ahrs = tfs.target_T_ahrs.rotation();
    gtsam::Rot3 map_R_ahrs = initial_ahrs_->orientation;
    gtsam::Rot3 map_R_target_measured = map_R_ahrs * target_R_ahrs.inverse();
    yaw = map_R_target_measured.yaw() - params_.ahrs.mag_declination_radians;
  } else if (params_.mag.enable_mag || params_.mag.enable_mag_init_only) {
    // Account for magnetometer rotation
    gtsam::Rot3 target_R_mag = tfs.target_T_mag.rotation();
    gtsam::Vector3 mag_sensor = initial_mag_->magnetic_field;
    gtsam::Vector3 mag_target = target_R_mag.rotate(mag_sensor);

    // Project the magnetic field vector using the estimated tilt
    gtsam::Rot3 R_rp = gtsam::Rot3::Ypr(0.0, pitch, roll);
    gtsam::Vector3 mag_horizontal = R_rp.rotate(mag_target);

    double measured_yaw = std::atan2(mag_horizontal.y(), mag_horizontal.x());
    double ref_yaw = std::atan2(params_.mag.reference_field[1], params_.mag.reference_field[0]);

    yaw = ref_yaw - measured_yaw;
  }

  return gtsam::Rot3::Ypr(yaw, pitch, roll);
}

gtsam::Point3 StateInitializer::computeInitialPosition(const gtsam::Rot3& map_R_target,
                                                       const TfBundle& tfs) {
  gtsam::Point3 map_p_base(params_.prior.parameter_priors.initial_position[0],
                           params_.prior.parameter_priors.initial_position[1],
                           params_.prior.parameter_priors.initial_position[2]);

  if (params_.prior.use_parameter_priors) {
    gtsam::Point3 target_p_base = tfs.target_T_base.translation();
    return map_p_base - map_R_target.rotate(target_p_base);
  }

  gtsam::Point3 map_p_target = map_p_base;
  if (params_.gps.enable_gps || params_.gps.enable_gps_init_only) {
    // Account for GPS lever arm
    gtsam::Point3 map_p_target_gps = map_R_target.rotate(tfs.target_T_gps.translation());
    map_p_target = initial_gps_->pose.translation() - map_p_target_gps;
  }

  if (params_.depth.enable_depth || params_.depth.enable_depth_init_only) {
    // Account for depth lever arm
    gtsam::Point3 map_p_target_depth = map_R_target.rotate(tfs.target_T_depth.translation());
    map_p_target.z() = initial_depth_->pose.translation().z() - map_p_target_depth.z();
  }

  return map_p_target;
}

gtsam::Vector3 StateInitializer::computeInitialVelocity(const gtsam::Rot3& map_R_target,
                                                        const TfBundle& tfs) {
  if (params_.prior.use_parameter_priors) {
    gtsam::Vector3 base_v_base =
        Eigen::Map<const Eigen::Vector3d>(params_.prior.parameter_priors.initial_velocity.data());
    gtsam::Vector3 target_v_target = tfs.target_T_base.rotation().rotate(base_v_base);
    return map_R_target.rotate(target_v_target);
  }

  if (params_.dvl.enable_dvl || params_.dvl.enable_dvl_init_only) {
    // Account for DVL lever arm
    gtsam::Vector3 target_v_dvl = tfs.target_T_dvl.rotation().rotate(initial_dvl_->linear_velocity);
    return map_R_target.rotate(target_v_dvl);
  }

  return gtsam::Vector3::Zero();
}

gtsam::imuBias::ConstantBias StateInitializer::computeInitialBias() {
  gtsam::Vector3 init_gyro_bias;
  if (params_.prior.use_parameter_priors) {
    init_gyro_bias =
        Eigen::Map<const Eigen::Vector3d>(params_.prior.parameter_priors.initial_gyro_bias.data());
  } else {
    init_gyro_bias = initial_imu_->angular_velocity;
  }
  gtsam::Vector3 init_accel_bias =
      Eigen::Map<const Eigen::Vector3d>(params_.prior.parameter_priors.initial_accel_bias.data());

  return gtsam::imuBias::ConstantBias(init_accel_bias, init_gyro_bias);
}

}  // namespace coug_fgo
