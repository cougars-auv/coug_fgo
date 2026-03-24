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
 * @file state_initializer.hpp
 * @brief Utility for initializing state priors from sensor data.
 * @author Nelson Durrant
 * @date Jan 2026
 */

#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/navigation/ImuBias.h>

#include <memory>

#include "coug_fgo/factor_graph_parameters.hpp"
#include "coug_fgo/utils/data_types.hpp"
#include "coug_fgo/utils/gtsam_conversions.hpp"

namespace coug_fgo::utils {

/**
 * @class StateInitializer
 * @brief Computes initial state priors for factor graph initialization.
 */
class StateInitializer {
 public:
  /**
   * @brief Constructor for StateInitializer.
   * @param params Node parameters.
   */
  explicit StateInitializer(const factor_graph_node::Params& params) : params_(params) {}

  /**
   * @brief Updates the running averages with new data from sensor queues.
   * @param current_time Current time in seconds.
   * @param queues Bundle of sensor message queues.
   * @return True if initialization averaging is complete.
   */
  bool update(double current_time, QueueBundle& queues) {
    if (params_.prior.use_parameter_priors) {
      initial_imu_ = queues.imu.back();
      if (params_.gps.enable_gps || params_.gps.enable_gps_init_only) {
        initial_gps_ = queues.gps.back();
      }
      initial_depth_ = queues.depth.back();
      if (params_.mag.enable_mag || params_.mag.enable_mag_init_only) {
        initial_mag_ = queues.mag.back();
      }
      if (params_.ahrs.enable_ahrs || params_.ahrs.enable_ahrs_init_only ||
          params_.comparison.enable_loose_dvl_preintegration) {
        initial_ahrs_ = queues.ahrs.back();
      }
      initial_dvl_ = queues.dvl.back();
      return true;
    }

    if (start_avg_time_ == 0.0) {
      start_avg_time_ = current_time;
    }

    incrementAverages(queues);

    return (current_time - start_avg_time_) >= params_.prior.initialization_duration;
  }

  /**
   * @brief Computes initial pose, velocity, and bias.
   * @param tfs Bundle of core sensor transformations.
   */
  void compute(const TfBundle& tfs) {
    gtsam::Rot3 map_R_target = computeInitialOrientation(tfs);
    pose_ = gtsam::Pose3(map_R_target, computeInitialPosition(map_R_target, tfs));
    velocity_ = computeInitialVelocity(map_R_target, tfs);
    bias_ = computeInitialBias();
    if (params_.comparison.enable_loose_dvl_preintegration ||
        params_.comparison.enable_tight_dvl_preintegration) {
      time_ = initial_depth_->timestamp;
    } else {
      time_ = initial_dvl_->timestamp;
    }
  }

  gtsam::Pose3 pose_;
  gtsam::Vector3 velocity_;
  gtsam::imuBias::ConstantBias bias_;
  double time_{0.0};

  std::shared_ptr<ImuData> initial_imu_;
  std::shared_ptr<OdometryData> initial_gps_;
  std::shared_ptr<OdometryData> initial_depth_;
  std::shared_ptr<AhrsData> initial_ahrs_;
  std::shared_ptr<MagneticFieldData> initial_mag_;
  std::shared_ptr<TwistData> initial_dvl_;

 private:
  /**
   * @brief Accumulates running averages from drained sensor message deques.
   * @param queues Bundle of drained sensor message deques to process.
   */
  void incrementAverages(QueueBundle& queues) {
    // Average IMU
    for (const auto& msg : queues.imu) {
      if (imu_count_ == 0) {
        initial_imu_ = std::make_shared<ImuData>(*msg);
      } else {
        double n = static_cast<double>(imu_count_ + 1);
        initial_imu_->linear_acceleration += (msg->linear_acceleration - initial_imu_->linear_acceleration) / n;
        initial_imu_->angular_velocity += (msg->angular_velocity - initial_imu_->angular_velocity) / n;
        initial_imu_->timestamp = msg->timestamp;
      }
      imu_count_++;
    }

    // Average GPS
    if (params_.gps.enable_gps || params_.gps.enable_gps_init_only) {
      for (const auto& msg : queues.gps) {
        if (gps_count_ == 0) {
          initial_gps_ = std::make_shared<OdometryData>(*msg);
        } else {
          double n = static_cast<double>(gps_count_ + 1);
          gtsam::Point3 current_p = msg->pose.translation();
          gtsam::Point3 init_p = initial_gps_->pose.translation();
          init_p += (current_p - init_p) / n;
          initial_gps_->pose = gtsam::Pose3(initial_gps_->pose.rotation(), init_p);
          initial_gps_->timestamp = msg->timestamp;
        }
        gps_count_++;
      }
    }

    // Average Depth
    for (const auto& msg : queues.depth) {
      if (depth_count_ == 0) {
        initial_depth_ = std::make_shared<OdometryData>(*msg);
      } else {
        double n = static_cast<double>(depth_count_ + 1);
        gtsam::Point3 current_p = msg->pose.translation();
        gtsam::Point3 init_p = initial_depth_->pose.translation();
        init_p.z() += (current_p.z() - init_p.z()) / n;
        initial_depth_->pose = gtsam::Pose3(initial_depth_->pose.rotation(), init_p);
        initial_depth_->timestamp = msg->timestamp;
      }
      depth_count_++;
    }

    // Average DVL
    for (const auto& msg : queues.dvl) {
      if (dvl_count_ == 0) {
        initial_dvl_ = std::make_shared<TwistData>(*msg);
      } else {
        double n = static_cast<double>(dvl_count_ + 1);
        initial_dvl_->linear_velocity += (msg->linear_velocity - initial_dvl_->linear_velocity) / n;
        initial_dvl_->timestamp = msg->timestamp;
      }
      dvl_count_++;
    }

    // Average Magnetometer
    if (params_.mag.enable_mag || params_.mag.enable_mag_init_only) {
      for (const auto& msg : queues.mag) {
        if (mag_count_ == 0) {
          initial_mag_ = std::make_shared<MagneticFieldData>(*msg);
        } else {
          double n = static_cast<double>(mag_count_ + 1);
          initial_mag_->magnetic_field += (msg->magnetic_field - initial_mag_->magnetic_field) / n;
          initial_mag_->timestamp = msg->timestamp;
        }
        mag_count_++;
      }
    }

    // Average AHRS
    if (params_.ahrs.enable_ahrs || params_.ahrs.enable_ahrs_init_only) {
      for (const auto& msg : queues.ahrs) {
        if (ahrs_count_ == 0) {
          initial_ahrs_ = std::make_shared<AhrsData>(*msg);
          ahrs_ref_ = msg->orientation;
          ahrs_log_sum_ = gtsam::Vector3::Zero();
        } else {
          ahrs_log_sum_ += gtsam::Rot3::Logmap(ahrs_ref_.between(msg->orientation));
          gtsam::Vector3 log_avg = ahrs_log_sum_ / static_cast<double>(ahrs_count_ + 1);
          initial_ahrs_->orientation = ahrs_ref_.compose(gtsam::Rot3::Expmap(log_avg));
          initial_ahrs_->timestamp = msg->timestamp;
        }
        ahrs_count_++;
      }
    }
  }

  /**
   * @brief Computes initial orientation from accelerometer tilt and heading sensors.
   * @param tfs SE(3) sensor transforms for lever arm compensation.
   * @return Initial rotation of the target frame in the map frame.
   */
  gtsam::Rot3 computeInitialOrientation(const TfBundle& tfs) {
    double roll = params_.prior.parameter_priors.initial_orientation[0];
    double pitch = params_.prior.parameter_priors.initial_orientation[1];
    double yaw = params_.prior.parameter_priors.initial_orientation[2];

    if (params_.prior.use_parameter_priors) {
      gtsam::Rot3 base_R_target = tfs.target_T_base.rotation().inverse();
      return gtsam::Rot3::Ypr(yaw, pitch, roll) * base_R_target;
    }

    // Account for IMU rotation
    gtsam::Vector3 accel_imu = initial_imu_->linear_acceleration;
    gtsam::Vector3 accel_target = tfs.target_T_imu.rotation() * accel_imu;

    roll = std::atan2(accel_target.y(), accel_target.z());
    pitch = std::atan2(-accel_target.x(), std::sqrt(accel_target.y() * accel_target.y() +
                                                    accel_target.z() * accel_target.z()));

    if (params_.ahrs.enable_ahrs || params_.ahrs.enable_ahrs_init_only) {
      // Account for AHRS sensor rotation
      gtsam::Rot3 target_R_ahrs = tfs.target_T_ahrs.rotation();
      gtsam::Rot3 map_R_ahrs = initial_ahrs_->orientation;
      gtsam::Rot3 map_R_target_measured = map_R_ahrs * target_R_ahrs.inverse();
      yaw = map_R_target_measured.yaw() + params_.ahrs.mag_declination_radians;
    } else if (params_.mag.enable_mag || params_.mag.enable_mag_init_only) {
      // Account for magnetometer rotation
      gtsam::Rot3 target_R_mag = tfs.target_T_mag.rotation();
      gtsam::Vector3 mag_sensor = initial_mag_->magnetic_field;
      gtsam::Vector3 mag_target = target_R_mag * mag_sensor;

      // Use the tilt-compensated magnetic vector
      gtsam::Rot3 R_rp = gtsam::Rot3::Ypr(0.0, pitch, roll);
      gtsam::Vector3 mag_horizontal = R_rp.unrotate(mag_target);

      double measured_yaw = std::atan2(mag_horizontal.y(), mag_horizontal.x());
      double ref_yaw = std::atan2(params_.mag.reference_field[1], params_.mag.reference_field[0]);

      yaw = ref_yaw - measured_yaw;
    }

    return gtsam::Rot3::Ypr(yaw, pitch, roll);
  }

  /**
   * @brief Computes initial position using GPS and depth with lever arm compensation.
   * @param map_R_target The computed initial rotation.
   * @param tfs SE(3) sensor transforms for lever arm compensation.
   * @return Initial position of the target frame in the map frame.
   */
  gtsam::Point3 computeInitialPosition(const gtsam::Rot3& map_R_target, const TfBundle& tfs) {
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

    // Account for depth lever arm
    gtsam::Point3 map_p_target_depth = map_R_target.rotate(tfs.target_T_depth.translation());
    map_p_target.z() = initial_depth_->pose.translation().z() - map_p_target_depth.z();

    return map_p_target;
  }

  /**
   * @brief Computes initial map-frame velocity from DVL body-frame measurements.
   * @param map_R_target The computed initial rotation.
   * @param tfs SE(3) sensor transforms for lever arm compensation.
   * @return Initial velocity of the target frame in the map frame.
   */
  gtsam::Vector3 computeInitialVelocity(const gtsam::Rot3& map_R_target, const TfBundle& tfs) {
    if (params_.prior.use_parameter_priors) {
      gtsam::Vector3 base_v_base = toGtsam(params_.prior.parameter_priors.initial_velocity);
      gtsam::Vector3 target_v_target = tfs.target_T_base.rotation().rotate(base_v_base);
      return map_R_target.rotate(target_v_target);
    }

    // Account for DVL lever arm
    gtsam::Vector3 target_v_dvl =
        tfs.target_T_dvl.rotation() * initial_dvl_->linear_velocity;

    return map_R_target.rotate(target_v_dvl);
  }

  /**
   * @brief Computes initial IMU bias from averaged gyroscope readings.
   * @return Initial accelerometer and gyroscope bias estimate.
   */
  gtsam::imuBias::ConstantBias computeInitialBias() {
    gtsam::Vector3 init_gyro_bias;
    if (params_.prior.use_parameter_priors) {
      init_gyro_bias = toGtsam(params_.prior.parameter_priors.initial_gyro_bias);
    } else {
      init_gyro_bias = initial_imu_->angular_velocity;
    }
    gtsam::Vector3 init_accel_bias = toGtsam(params_.prior.parameter_priors.initial_accel_bias);

    return gtsam::imuBias::ConstantBias(init_accel_bias, init_gyro_bias);
  }

  const factor_graph_node::Params& params_;
  double start_avg_time_{0.0};
  size_t imu_count_ = 0, gps_count_ = 0, depth_count_ = 0, mag_count_ = 0, ahrs_count_ = 0,
         dvl_count_ = 0;
  gtsam::Rot3 ahrs_ref_;
  gtsam::Vector3 ahrs_log_sum_ = gtsam::Vector3::Zero();
};

}  // namespace coug_fgo::utils
