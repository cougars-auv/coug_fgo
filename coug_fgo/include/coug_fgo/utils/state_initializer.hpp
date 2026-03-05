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

#include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/magnetic_field.hpp>

#include <coug_fgo/utils/types.hpp>
#include <coug_fgo/factor_graph_parameters.hpp>
#include <coug_fgo/utils/conversions.hpp>

namespace coug_fgo::utils
{

/**
 * @class StateInitializer
 * @brief Computes initial state priors for factor graph initialization.
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
   * @param current_time Current ROS time.
   * @param queues Bundle of sensor message queues.
   * @return True if initialization averaging is complete.
   */
  bool update(const rclcpp::Time & current_time, QueueBundle & queues)
  {
    if (params_.prior.use_parameter_priors) {
      initial_imu_ = queues.imu.back();
      if (params_.gps.enable_gps || params_.gps.enable_gps_init_only) {
        initial_gps_ = queues.gps.back();
      }
      initial_depth_ = queues.depth.back();
      if (params_.mag.enable_mag || params_.mag.enable_mag_init_only) {
        initial_mag_ = queues.mag.back();
      }
      if (params_.ahrs.enable_ahrs || params_.ahrs.enable_ahrs_init_only) {
        initial_ahrs_ = queues.ahrs.back();
      }
      initial_dvl_ = queues.dvl.back();
      return true;
    }

    if (start_avg_time_.nanoseconds() == 0) {
      start_avg_time_ = current_time;
    }

    incrementAverages(queues);

    return (current_time - start_avg_time_).seconds() >= params_.prior.initialization_duration;
  }

  /**
   * @brief Computes initial pose, velocity, and bias.
   * @param tfs Bundle of core sensor transformations.
   */
  void compute(const TfBundle & tfs)
  {
    gtsam::Rot3 initial_orientation_target = computeInitialOrientation(tfs);
    pose_ = gtsam::Pose3(
      initial_orientation_target, computeInitialPosition(
        initial_orientation_target,
        tfs));
    velocity_ = computeInitialVelocity(initial_orientation_target, tfs);
    bias_ = computeInitialBias();
    if (params_.experimental.enable_dvl_preintegration) {
      time_ = rclcpp::Time(initial_depth_->header.stamp);
    } else {
      time_ = rclcpp::Time(initial_dvl_->header.stamp);
    }
  }

  gtsam::Pose3 pose_;
  gtsam::Vector3 velocity_;
  gtsam::imuBias::ConstantBias bias_;
  rclcpp::Time time_{0, 0, RCL_ROS_TIME};

  sensor_msgs::msg::Imu::SharedPtr initial_imu_;
  nav_msgs::msg::Odometry::SharedPtr initial_gps_;
  nav_msgs::msg::Odometry::SharedPtr initial_depth_;
  sensor_msgs::msg::Imu::SharedPtr initial_ahrs_;
  sensor_msgs::msg::MagneticField::SharedPtr initial_mag_;
  geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr initial_dvl_;

private:
  /**
   * @brief Accumulates running averages from drained sensor message deques.
   * @param queues Bundle of drained sensor message deques to process.
   */
  void incrementAverages(QueueBundle & queues)
  {
    // Average IMU
    for (const auto & msg : queues.imu) {
      if (imu_count_ == 0) {
        initial_imu_ = std::make_shared<sensor_msgs::msg::Imu>(*msg);
      } else {
        double n = static_cast<double>(imu_count_ + 1);
        initial_imu_->linear_acceleration.x +=
          (msg->linear_acceleration.x - initial_imu_->linear_acceleration.x) / n;
        initial_imu_->linear_acceleration.y +=
          (msg->linear_acceleration.y - initial_imu_->linear_acceleration.y) / n;
        initial_imu_->linear_acceleration.z +=
          (msg->linear_acceleration.z - initial_imu_->linear_acceleration.z) / n;
        initial_imu_->angular_velocity.x +=
          (msg->angular_velocity.x - initial_imu_->angular_velocity.x) / n;
        initial_imu_->angular_velocity.y +=
          (msg->angular_velocity.y - initial_imu_->angular_velocity.y) / n;
        initial_imu_->angular_velocity.z +=
          (msg->angular_velocity.z - initial_imu_->angular_velocity.z) / n;
        initial_imu_->header.stamp = msg->header.stamp;
      }
      imu_count_++;
    }

    // Average GPS
    if (params_.gps.enable_gps || params_.gps.enable_gps_init_only) {
      for (const auto & msg : queues.gps) {
        if (gps_count_ == 0) {
          initial_gps_ = std::make_shared<nav_msgs::msg::Odometry>(*msg);
        } else {
          double n = static_cast<double>(gps_count_ + 1);
          initial_gps_->pose.pose.position.x +=
            (msg->pose.pose.position.x - initial_gps_->pose.pose.position.x) / n;
          initial_gps_->pose.pose.position.y +=
            (msg->pose.pose.position.y - initial_gps_->pose.pose.position.y) / n;
          initial_gps_->pose.pose.position.z +=
            (msg->pose.pose.position.z - initial_gps_->pose.pose.position.z) / n;
          initial_gps_->header.stamp = msg->header.stamp;
        }
        gps_count_++;
      }
    }

    // Average Depth
    for (const auto & msg : queues.depth) {
      if (depth_count_ == 0) {
        initial_depth_ = std::make_shared<nav_msgs::msg::Odometry>(*msg);
      } else {
        double n = static_cast<double>(depth_count_ + 1);
        initial_depth_->pose.pose.position.z +=
          (msg->pose.pose.position.z - initial_depth_->pose.pose.position.z) / n;
        initial_depth_->header.stamp = msg->header.stamp;
      }
      depth_count_++;
    }

    // Average DVL
    for (const auto & msg : queues.dvl) {
      if (dvl_count_ == 0) {
        initial_dvl_ = std::make_shared<geometry_msgs::msg::TwistWithCovarianceStamped>(*msg);
      } else {
        double n = static_cast<double>(dvl_count_ + 1);
        initial_dvl_->twist.twist.linear.x +=
          (msg->twist.twist.linear.x - initial_dvl_->twist.twist.linear.x) / n;
        initial_dvl_->twist.twist.linear.y +=
          (msg->twist.twist.linear.y - initial_dvl_->twist.twist.linear.y) / n;
        initial_dvl_->twist.twist.linear.z +=
          (msg->twist.twist.linear.z - initial_dvl_->twist.twist.linear.z) / n;
        initial_dvl_->header.stamp = msg->header.stamp;
      }
      dvl_count_++;
    }

    // Average Magnetometer
    if (params_.mag.enable_mag || params_.mag.enable_mag_init_only) {
      for (const auto & msg : queues.mag) {
        if (mag_count_ == 0) {
          initial_mag_ = std::make_shared<sensor_msgs::msg::MagneticField>(*msg);
        } else {
          double n = static_cast<double>(mag_count_ + 1);
          initial_mag_->magnetic_field.x +=
            (msg->magnetic_field.x - initial_mag_->magnetic_field.x) / n;
          initial_mag_->magnetic_field.y +=
            (msg->magnetic_field.y - initial_mag_->magnetic_field.y) / n;
          initial_mag_->magnetic_field.z +=
            (msg->magnetic_field.z - initial_mag_->magnetic_field.z) / n;
          initial_mag_->header.stamp = msg->header.stamp;
        }
        mag_count_++;
      }
    }

    // Average AHRS
    if (params_.ahrs.enable_ahrs || params_.ahrs.enable_ahrs_init_only) {
      for (const auto & msg : queues.ahrs) {
        if (ahrs_count_ == 0) {
          initial_ahrs_ = std::make_shared<sensor_msgs::msg::Imu>(*msg);
          ahrs_ref_ = toGtsam(msg->orientation);
          ahrs_log_sum_ = gtsam::Vector3::Zero();
        } else {
          ahrs_log_sum_ += gtsam::Rot3::Logmap(ahrs_ref_.between(toGtsam(msg->orientation)));
          gtsam::Vector3 log_avg = ahrs_log_sum_ / static_cast<double>(ahrs_count_ + 1);
          initial_ahrs_->orientation = toQuatMsg(ahrs_ref_.compose(gtsam::Rot3::Expmap(log_avg)));
          initial_ahrs_->header.stamp = msg->header.stamp;
        }
        ahrs_count_++;
      }
    }
  }

  /**
   * @brief Computes initial orientation from accelerometer tilt and heading sensors.
   * @param tfs SE(3) sensor transforms for lever arm compensation.
   * @return Initial rotation of the target frame in the world frame.
   */
  gtsam::Rot3 computeInitialOrientation(const TfBundle & tfs)
  {
    double roll = params_.prior.parameter_priors.initial_orientation[0];
    double pitch = params_.prior.parameter_priors.initial_orientation[1];
    double yaw = params_.prior.parameter_priors.initial_orientation[2];

    if (params_.prior.use_parameter_priors) {
      gtsam::Rot3 base_R_target = tfs.target_T_base.rotation().inverse();
      return gtsam::Rot3::Ypr(yaw, pitch, roll) * base_R_target;
    }

    // Account for IMU rotation
    gtsam::Vector3 accel_imu = toGtsam(initial_imu_->linear_acceleration);
    gtsam::Vector3 accel_target = tfs.target_T_imu.rotation() * accel_imu;

    roll = std::atan2(accel_target.y(), accel_target.z());
    pitch = std::atan2(
      -accel_target.x(), std::sqrt(
        accel_target.y() * accel_target.y() +
        accel_target.z() * accel_target.z()));

    if (params_.ahrs.enable_ahrs || params_.ahrs.enable_ahrs_init_only) {
      // Account for AHRS sensor rotation
      gtsam::Rot3 R_target_sensor = tfs.target_T_ahrs.rotation();
      gtsam::Rot3 R_world_sensor = toGtsam(initial_ahrs_->orientation);
      gtsam::Rot3 R_world_target_measured = R_world_sensor * R_target_sensor.inverse();
      yaw = R_world_target_measured.yaw() + params_.ahrs.mag_declination_radians;
    } else if (params_.mag.enable_mag || params_.mag.enable_mag_init_only) {
      // Account for magnetometer rotation
      gtsam::Rot3 R_target_sensor = tfs.target_T_mag.rotation();
      gtsam::Vector3 mag_sensor = toGtsam(initial_mag_->magnetic_field);
      gtsam::Vector3 mag_target = R_target_sensor * mag_sensor;

      // Use the tilt-compensated magnetic vector
      gtsam::Rot3 R_rp = gtsam::Rot3::Ypr(0.0, pitch, roll);
      gtsam::Vector3 mag_horizontal = R_rp.unrotate(mag_target);

      double measured_yaw = std::atan2(mag_horizontal.y(), mag_horizontal.x());
      double ref_yaw = std::atan2(
        params_.mag.reference_field[1],
        params_.mag.reference_field[0]);

      yaw = ref_yaw - measured_yaw;
    }

    return gtsam::Rot3::Ypr(yaw, pitch, roll);
  }

  /**
   * @brief Computes initial position using GPS and depth with lever arm compensation.
   * @param initial_orientation_target The computed initial rotation.
   * @param tfs SE(3) sensor transforms for lever arm compensation.
   * @return Initial position of the target frame in the world frame.
   */
  gtsam::Point3 computeInitialPosition(
    const gtsam::Rot3 & initial_orientation_target,
    const TfBundle & tfs)
  {
    gtsam::Point3 P_world_base(
      params_.prior.parameter_priors.initial_position[0],
      params_.prior.parameter_priors.initial_position[1],
      params_.prior.parameter_priors.initial_position[2]);

    if (params_.prior.use_parameter_priors) {
      gtsam::Point3 target_p_base = tfs.target_T_base.translation();
      return P_world_base - initial_orientation_target.rotate(target_p_base);
    }

    gtsam::Point3 initial_position_target = P_world_base;
    if (params_.gps.enable_gps || params_.gps.enable_gps_init_only) {
      // Account for GPS lever arm
      gtsam::Point3 world_p_target_gps = initial_orientation_target.rotate(
        tfs.target_T_gps.translation());
      initial_position_target = toGtsam(initial_gps_->pose.pose.position) - world_p_target_gps;
    }

    // Account for depth lever arm
    gtsam::Point3 world_t_target_depth =
      initial_orientation_target.rotate(tfs.target_T_depth.translation());
    initial_position_target.z() = initial_depth_->pose.pose.position.z - world_t_target_depth.z();

    return initial_position_target;
  }

  /**
   * @brief Computes initial world-frame velocity from DVL body-frame measurements.
   * @param initial_orientation_target The computed initial rotation.
   * @param tfs SE(3) sensor transforms for lever arm compensation.
   * @return Initial velocity of the target frame in the world frame.
   */
  gtsam::Vector3 computeInitialVelocity(
    const gtsam::Rot3 & initial_orientation_target,
    const TfBundle & tfs)
  {
    if (params_.prior.use_parameter_priors) {
      gtsam::Vector3 v_base = toGtsam(params_.prior.parameter_priors.initial_velocity);
      gtsam::Vector3 v_target = tfs.target_T_base.rotation().rotate(v_base);
      return initial_orientation_target.rotate(v_target);
    }

    // Account for DVL lever arm
    gtsam::Vector3 target_v_dvl = tfs.target_T_dvl.rotation() * toGtsam(
      initial_dvl_->twist.twist.linear);

    return initial_orientation_target.rotate(target_v_dvl);
  }

  /**
   * @brief Computes initial IMU bias from averaged gyroscope readings.
   * @return Initial accelerometer and gyroscope bias estimate.
   */
  gtsam::imuBias::ConstantBias computeInitialBias()
  {
    gtsam::Vector3 init_gyro_bias;
    if (params_.prior.use_parameter_priors) {
      init_gyro_bias = toGtsam(params_.prior.parameter_priors.initial_gyro_bias);
    } else {
      init_gyro_bias = toGtsam(initial_imu_->angular_velocity);
    }
    gtsam::Vector3 init_accel_bias = toGtsam(params_.prior.parameter_priors.initial_accel_bias);

    return gtsam::imuBias::ConstantBias(init_accel_bias, init_gyro_bias);
  }

  const factor_graph_node::Params & params_;
  rclcpp::Time start_avg_time_{0, 0, RCL_ROS_TIME};
  size_t imu_count_ = 0, gps_count_ = 0, depth_count_ = 0, mag_count_ = 0, ahrs_count_ = 0,
    dvl_count_ = 0;
  gtsam::Rot3 ahrs_ref_;
  gtsam::Vector3 ahrs_log_sum_ = gtsam::Vector3::Zero();
};

}  // namespace coug_fgo::utils
