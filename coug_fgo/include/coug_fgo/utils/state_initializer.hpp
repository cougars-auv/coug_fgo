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
#include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/magnetic_field.hpp>
#include <coug_fgo/utils/thread_safe_queue.hpp>
#include <coug_fgo/factor_graph_parameters.hpp>
#include <coug_fgo/utils/conversion_utils.hpp>

namespace coug_fgo::utils
{

/**
 * @struct TfBundle
 * @brief Container for TF sensor transformations.
 */
struct TfBundle
{
  gtsam::Pose3 imu_to_dvl;
  gtsam::Pose3 gps_to_dvl;
  gtsam::Pose3 depth_to_dvl;
  gtsam::Pose3 mag_to_dvl;
  gtsam::Pose3 ahrs_to_dvl;
  gtsam::Pose3 dvl_to_base;
};

/**
 * @struct QueueBundle
 * @brief Container for sensor message queues.
 */
struct QueueBundle
{
  ThreadSafeQueue<sensor_msgs::msg::Imu::SharedPtr> & imu;
  ThreadSafeQueue<nav_msgs::msg::Odometry::SharedPtr> & gps;
  ThreadSafeQueue<nav_msgs::msg::Odometry::SharedPtr> & depth;
  ThreadSafeQueue<sensor_msgs::msg::MagneticField::SharedPtr> & mag;
  ThreadSafeQueue<sensor_msgs::msg::Imu::SharedPtr> & ahrs;
  ThreadSafeQueue<geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr> & dvl;
};

/**
 * @class StateInitializer
 * @brief Utility for initializing state priors from sensor data.
 */
class StateInitializer
{
public:
  /**
   * @brief Constructor for StateInitializer.
   * @param params Node parameters.
   */
  explicit StateInitializer(const factor_graph_node::Params & params)
  : params_(params) {}

  /**
   * @brief Updates the running averages with new data from sensor queues.
   * @param current_time Current ROS time in seconds.
   * @param queues Bundle of sensor message queues.
   * @return True if initialization averaging is complete.
   */
  bool update(double current_time, QueueBundle & queues)
  {
    if (params_.prior.use_parameter_priors) {
      initial_imu = queues.imu.back();
      if (params_.gps.enable_gps || params_.gps.enable_gps_init_only) {
        initial_gps = queues.gps.back();
      }
      initial_depth = queues.depth.back();
      if (params_.mag.enable_mag || params_.mag.enable_mag_init_only) {
        initial_mag = queues.mag.back();
      }
      if (params_.ahrs.enable_ahrs || params_.ahrs.enable_ahrs_init_only) {
        initial_ahrs = queues.ahrs.back();
      }
      initial_dvl = queues.dvl.back();
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
  void compute(const TfBundle & tfs)
  {
    gtsam::Rot3 initial_orientation_dvl = computeInitialOrientation(tfs);
    pose = gtsam::Pose3(
      initial_orientation_dvl, computeInitialPosition(
        initial_orientation_dvl,
        tfs));
    velocity = computeInitialVelocity(initial_orientation_dvl);
    bias = computeInitialBias();
    if (params_.experimental.enable_dvl_preintegration) {
      time = initial_depth->header.stamp.sec + initial_depth->header.stamp.nanosec * 1e-9;
      stamp = initial_depth->header.stamp;
    } else {
      time = initial_dvl->header.stamp.sec + initial_dvl->header.stamp.nanosec * 1e-9;
      stamp = initial_dvl->header.stamp;
    }
  }

  gtsam::Pose3 pose;
  gtsam::Vector3 velocity;
  gtsam::imuBias::ConstantBias bias;
  double time = 0.0;
  rclcpp::Time stamp;

  sensor_msgs::msg::Imu::SharedPtr initial_imu;
  nav_msgs::msg::Odometry::SharedPtr initial_gps;
  nav_msgs::msg::Odometry::SharedPtr initial_depth;
  sensor_msgs::msg::Imu::SharedPtr initial_ahrs;
  sensor_msgs::msg::MagneticField::SharedPtr initial_mag;
  geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr initial_dvl;

private:
  void incrementAverages(QueueBundle & queues)
  {
    // Average IMU
    auto imu_msgs = queues.imu.drain();
    for (const auto & msg : imu_msgs) {
      if (imu_count_ == 0) {
        initial_imu = msg;
      } else {
        double n = static_cast<double>(imu_count_ + 1);
        initial_imu->linear_acceleration.x +=
          (msg->linear_acceleration.x - initial_imu->linear_acceleration.x) / n;
        initial_imu->linear_acceleration.y +=
          (msg->linear_acceleration.y - initial_imu->linear_acceleration.y) / n;
        initial_imu->linear_acceleration.z +=
          (msg->linear_acceleration.z - initial_imu->linear_acceleration.z) / n;
        initial_imu->angular_velocity.x +=
          (msg->angular_velocity.x - initial_imu->angular_velocity.x) / n;
        initial_imu->angular_velocity.y +=
          (msg->angular_velocity.y - initial_imu->angular_velocity.y) / n;
        initial_imu->angular_velocity.z +=
          (msg->angular_velocity.z - initial_imu->angular_velocity.z) / n;
        initial_imu->header.stamp = msg->header.stamp;
      }
      imu_count_++;
    }

    // Average GPS
    if (params_.gps.enable_gps || params_.gps.enable_gps_init_only) {
      auto gps_msgs = queues.gps.drain();
      for (const auto & msg : gps_msgs) {
        if (gps_count_ == 0) {
          initial_gps = msg;
        } else {
          double n = static_cast<double>(gps_count_ + 1);
          initial_gps->pose.pose.position.x +=
            (msg->pose.pose.position.x - initial_gps->pose.pose.position.x) / n;
          initial_gps->pose.pose.position.y +=
            (msg->pose.pose.position.y - initial_gps->pose.pose.position.y) / n;
          initial_gps->pose.pose.position.z +=
            (msg->pose.pose.position.z - initial_gps->pose.pose.position.z) / n;
          initial_gps->header.stamp = msg->header.stamp;
        }
        gps_count_++;
      }
    }

    // Average Depth
    auto depth_msgs = queues.depth.drain();
    for (const auto & msg : depth_msgs) {
      if (depth_count_ == 0) {
        initial_depth = msg;
      } else {
        double n = static_cast<double>(depth_count_ + 1);
        initial_depth->pose.pose.position.z +=
          (msg->pose.pose.position.z - initial_depth->pose.pose.position.z) / n;
        initial_depth->header.stamp = msg->header.stamp;
      }
      depth_count_++;
    }

    // Average DVL
    auto dvl_msgs = queues.dvl.drain();
    for (const auto & msg : dvl_msgs) {
      if (dvl_count_ == 0) {
        initial_dvl = msg;
      } else {
        double n = static_cast<double>(dvl_count_ + 1);
        initial_dvl->twist.twist.linear.x +=
          (msg->twist.twist.linear.x - initial_dvl->twist.twist.linear.x) / n;
        initial_dvl->twist.twist.linear.y +=
          (msg->twist.twist.linear.y - initial_dvl->twist.twist.linear.y) / n;
        initial_dvl->twist.twist.linear.z +=
          (msg->twist.twist.linear.z - initial_dvl->twist.twist.linear.z) / n;
        initial_dvl->header.stamp = msg->header.stamp;
      }
      dvl_count_++;
    }

    // Average Magnetometer
    if (params_.mag.enable_mag || params_.mag.enable_mag_init_only) {
      auto mag_msgs = queues.mag.drain();
      for (const auto & msg : mag_msgs) {
        if (mag_count_ == 0) {
          initial_mag = msg;
        } else {
          double n = static_cast<double>(mag_count_ + 1);
          initial_mag->magnetic_field.x +=
            (msg->magnetic_field.x - initial_mag->magnetic_field.x) / n;
          initial_mag->magnetic_field.y +=
            (msg->magnetic_field.y - initial_mag->magnetic_field.y) / n;
          initial_mag->magnetic_field.z +=
            (msg->magnetic_field.z - initial_mag->magnetic_field.z) / n;
          initial_mag->header.stamp = msg->header.stamp;
        }
        mag_count_++;
      }
    }

    // Average AHRS
    if (params_.ahrs.enable_ahrs || params_.ahrs.enable_ahrs_init_only) {
      auto ahrs_msgs = queues.ahrs.drain();
      for (const auto & msg : ahrs_msgs) {
        if (ahrs_count_ == 0) {
          initial_ahrs = msg;
          ahrs_ref_ = toGtsam(msg->orientation);
          ahrs_log_sum_ = gtsam::Vector3::Zero();
        } else {
          ahrs_log_sum_ += gtsam::Rot3::Logmap(ahrs_ref_.between(toGtsam(msg->orientation)));
          gtsam::Vector3 log_avg = ahrs_log_sum_ / static_cast<double>(ahrs_count_ + 1);
          initial_ahrs->orientation = toQuatMsg(ahrs_ref_.compose(gtsam::Rot3::Expmap(log_avg)));
          initial_ahrs->header.stamp = msg->header.stamp;
        }
        ahrs_count_++;
      }
    }
  }

  gtsam::Rot3 computeInitialOrientation(const TfBundle & tfs)
  {
    double roll = params_.prior.parameter_priors.initial_orientation[0];
    double pitch = params_.prior.parameter_priors.initial_orientation[1];
    double yaw = params_.prior.parameter_priors.initial_orientation[2];
    gtsam::Rot3 R_base_dvl = tfs.dvl_to_base.rotation();

    if (params_.prior.use_parameter_priors) {
      // Account for DVL rotation
      return gtsam::Rot3::Ypr(yaw, pitch, roll) * R_base_dvl;
    }

    // Account for IMU rotation
    gtsam::Vector3 accel_imu = toGtsam(initial_imu->linear_acceleration);
    gtsam::Vector3 accel_base = tfs.dvl_to_base.rotation() *
      (tfs.imu_to_dvl.rotation() * accel_imu);

    roll = std::atan2(accel_base.y(), accel_base.z());
    pitch = std::atan2(
      -accel_base.x(), std::sqrt(
        accel_base.y() * accel_base.y() +
        accel_base.z() * accel_base.z()));

    if (params_.ahrs.enable_ahrs || params_.ahrs.enable_ahrs_init_only) {
      // Account for AHRS sensor rotation
      gtsam::Rot3 R_base_sensor = R_base_dvl * tfs.ahrs_to_dvl.rotation();
      gtsam::Rot3 R_world_sensor = toGtsam(initial_ahrs->orientation);
      gtsam::Rot3 R_world_base_measured = R_world_sensor * R_base_sensor.inverse();
      yaw = R_world_base_measured.yaw() + params_.ahrs.mag_declination_radians;
    } else if (params_.mag.enable_mag || params_.mag.enable_mag_init_only) {
      // Account for magnetometer rotation
      gtsam::Rot3 R_base_sensor = R_base_dvl * tfs.mag_to_dvl.rotation();
      gtsam::Vector3 mag_sensor = toGtsam(initial_mag->magnetic_field);
      gtsam::Vector3 mag_base = R_base_sensor * mag_sensor;

      // Use the tilt-compensated magnetic vector
      gtsam::Rot3 R_rp = gtsam::Rot3::Ypr(0.0, pitch, roll);
      gtsam::Vector3 mag_horizontal = R_rp.unrotate(mag_base);

      double measured_yaw = std::atan2(mag_horizontal.y(), mag_horizontal.x());
      double ref_yaw = std::atan2(
        params_.mag.reference_field[1],
        params_.mag.reference_field[0]);

      yaw = ref_yaw - measured_yaw;
    }

    return gtsam::Rot3::Ypr(yaw, pitch, roll) * R_base_dvl;
  }

  gtsam::Point3 computeInitialPosition(
    const gtsam::Rot3 & initial_orientation_dvl,
    const TfBundle & tfs)
  {
    gtsam::Point3 P_world_base(
      params_.prior.parameter_priors.initial_position[0],
      params_.prior.parameter_priors.initial_position[1],
      params_.prior.parameter_priors.initial_position[2]);
    gtsam::Rot3 R_world_base = initial_orientation_dvl * tfs.dvl_to_base.rotation().inverse();
    gtsam::Point3 P_world_dvl_param = P_world_base + R_world_base.rotate(
      tfs.dvl_to_base.translation());

    if (params_.prior.use_parameter_priors) {
      // Account for DVL lever arm
      return P_world_dvl_param;
    }

    gtsam::Point3 initial_position_dvl = P_world_dvl_param;
    if (params_.gps.enable_gps || params_.gps.enable_gps_init_only) {
      // Account for GPS lever arm
      gtsam::Point3 world_t_dvl_gps = initial_orientation_dvl.rotate(tfs.gps_to_dvl.translation());
      initial_position_dvl = toGtsam(initial_gps->pose.pose.position) - world_t_dvl_gps;
    }

    // Account for depth lever arm
    gtsam::Point3 world_t_dvl_depth =
      initial_orientation_dvl.rotate(tfs.depth_to_dvl.translation());
    initial_position_dvl.z() = initial_depth->pose.pose.position.z - world_t_dvl_depth.z();

    return initial_position_dvl;
  }

  gtsam::Vector3 computeInitialVelocity(const gtsam::Rot3 & initial_orientation_dvl)
  {
    if (params_.prior.use_parameter_priors) {
      // Account for DVL rotation (in world frame)
      return initial_orientation_dvl.rotate(
        toGtsam(params_.prior.parameter_priors.initial_velocity));
    }

    // Account for DVL rotation (in world frame)
    return initial_orientation_dvl.rotate(toGtsam(initial_dvl->twist.twist.linear));
  }

  gtsam::imuBias::ConstantBias computeInitialBias()
  {
    gtsam::Vector3 init_gyro_bias;
    if (params_.prior.use_parameter_priors) {
      init_gyro_bias = toGtsam(params_.prior.parameter_priors.initial_gyro_bias);
    } else {
      init_gyro_bias = toGtsam(initial_imu->angular_velocity);
    }
    gtsam::Vector3 init_accel_bias = toGtsam(params_.prior.parameter_priors.initial_accel_bias);

    return gtsam::imuBias::ConstantBias(init_accel_bias, init_gyro_bias);
  }

  const factor_graph_node::Params & params_;
  double start_avg_time_ = 0.0;
  size_t imu_count_ = 0, gps_count_ = 0, depth_count_ = 0, mag_count_ = 0, ahrs_count_ = 0,
    dvl_count_ = 0;
  gtsam::Rot3 ahrs_ref_;
  gtsam::Vector3 ahrs_log_sum_ = gtsam::Vector3::Zero();
};

}  // namespace coug_fgo::utils
