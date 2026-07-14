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
 * @file factor_graph_py.cpp
 * @brief Implementation and pybind11 module for the FactorGraphPy wrapper.
 * @author Nelson Durrant
 * @date May 2026
 */

#include "coug_fgo/factor_graph_py.hpp"

#include <gtsam/inference/Symbol.h>

#include <optional>
#include <rclcpp/rclcpp.hpp>
#include <stdexcept>
#include <unordered_map>

using gtsam::symbol_shorthand::B;  // Bias (ax,ay,az,gx,gy,gz)
using gtsam::symbol_shorthand::V;  // Velocity (x,y,z)

namespace coug_fgo {

using utils::StateInitializer;

namespace {

/**
 * @brief Converts a ROS (x, y, z, w) quaternion and translation to a GTSAM Pose3.
 * @param position Translation [x, y, z].
 * @param quat_xyzw Orientation quaternion (x, y, z, w).
 * @return The equivalent GTSAM Pose3.
 */
gtsam::Pose3 toPose3(const Eigen::Vector3d& position, const Eigen::Vector4d& quat_xyzw) {
  gtsam::Rot3 r = gtsam::Rot3::Quaternion(quat_xyzw(3), quat_xyzw(0), quat_xyzw(1), quat_xyzw(2));
  return gtsam::Pose3(r, gtsam::Point3(position));
}

/**
 * @brief Forwards core log messages to the Python `logging` module.
 * @param level The core log level.
 * @param msg The log message.
 */
void pyLogCallback(utils::LogLevel level, const std::string& msg) {
  int py_level = 30;  // logging.WARNING
  switch (level) {
    case utils::LogLevel::kDebug:
      py_level = 10;
      break;
    case utils::LogLevel::kInfo:
      py_level = 20;
      break;
    case utils::LogLevel::kWarn:
      py_level = 30;
      break;
    case utils::LogLevel::kError:
      py_level = 40;
      break;
  }
  pybind11::gil_scoped_acquire gil;
  pybind11::module_::import("logging")
      .attr("getLogger")("coug_fgo.core")
      .attr("log")(py_level, msg);
}

/**
 * @brief Converts one optimized state into a Python dict of named scalars.
 * @param time State timestamp in seconds.
 * @param pose The optimized pose.
 * @param velocity The optimized velocity, if available.
 * @param bias The optimized IMU bias, if available.
 * @return Dict with time, position, orientation, velocity, and bias entries.
 */
pybind11::dict toStateDict(double time, const gtsam::Pose3& pose,
                           const std::optional<gtsam::Vector3>& velocity,
                           const std::optional<gtsam::imuBias::ConstantBias>& bias) {
  pybind11::dict d;
  gtsam::Quaternion q = pose.rotation().toQuaternion();

  d["time"] = time;
  d["x"] = pose.translation().x();
  d["y"] = pose.translation().y();
  d["z"] = pose.translation().z();
  d["qx"] = q.x();
  d["qy"] = q.y();
  d["qz"] = q.z();
  d["qw"] = q.w();

  if (velocity) {
    d["vx"] = velocity->x();
    d["vy"] = velocity->y();
    d["vz"] = velocity->z();
  }
  if (bias) {
    d["bias_accel_x"] = bias->accelerometer().x();
    d["bias_accel_y"] = bias->accelerometer().y();
    d["bias_accel_z"] = bias->accelerometer().z();
    d["bias_gyro_x"] = bias->gyroscope().x();
    d["bias_gyro_y"] = bias->gyroscope().y();
    d["bias_gyro_z"] = bias->gyroscope().z();
  }
  return d;
}

}  // namespace

FactorGraphPy::FactorGraphPy(const std::vector<std::string>& config_paths, const std::string& ns) {
  if (!rclcpp::ok()) {
    rclcpp::init(0, nullptr);
  }

  std::vector<std::string> args = {"--ros-args"};
  for (const auto& path : config_paths) {
    args.push_back("--params-file");
    args.push_back(path);
  }

  rclcpp::NodeOptions options;
  options.arguments(args);
  auto param_node = std::make_shared<rclcpp::Node>("factor_graph_node", ns, options);

  factor_graph_node::ParamListener param_listener(param_node->get_node_parameters_interface());
  params_ = param_listener.get_params();

  core_ = std::make_unique<FactorGraphCore>(params_);
  core_->setLogCallback(&pyLogCallback);
  state_init_ = std::make_unique<StateInitializer>(params_);
}

pybind11::dict FactorGraphPy::get_params() const {
  pybind11::dict p;

  // --- Node Settings ---
  p["solver_type"] = params_.solver_type;
  p["max_update_rate_hz"] = params_.max_update_rate_hz;
  p["max_opt_rate_hz"] = params_.max_opt_rate_hz;
  p["publish_smoothed_path"] = params_.publish_smoothed_path;
  p["publish_pose_cov"] = params_.publish_pose_cov;
  p["publish_velocity_cov"] = params_.publish_velocity_cov;
  p["publish_imu_bias_cov"] = params_.publish_imu_bias_cov;

  // --- Keyframe Settings ---
  p["keyframe_source"] = params_.keyframe_source;
  p["backup_keyframe_source"] = params_.backup_keyframe_source;
  p["keyframe_timeout_sec"] = params_.keyframe_timeout_sec;
  p["keyframe_timer_hz"] = params_.keyframe_timer_hz;

  // --- ROS Topics and Frames ---
  pybind11::dict topics;
  topics["imu"] = params_.imu_topic;
  topics["gps"] = params_.gps_odom_topic;
  topics["depth"] = params_.depth_odom_topic;
  topics["mag"] = params_.mag_topic;
  topics["ahrs"] = params_.ahrs_topic;
  topics["dvl"] = params_.dvl_topic;
  topics["wrench"] = params_.wrench_topic;
  p["topics"] = topics;

  p["target_frame"] = params_.target_frame;
  p["base_frame"] = params_.base_frame;

  // --- Sensor Settings ---
  auto sensor_dict = [](const auto& s, bool enable, bool enable_extra_only) {
    pybind11::dict d;
    d["enable"] = enable;
    d["enable_extra_only"] = enable_extra_only;  // init_only (sensors) or dropout_only (dynamics)
    d["use_parameter_frame"] = s.use_parameter_frame;
    d["parameter_frame"] = s.parameter_frame;
    d["use_parameter_tf"] = s.use_parameter_tf;
    d["tf_position"] = s.parameter_tf.position;
    d["tf_orientation"] = s.parameter_tf.orientation;
    return d;
  };

  pybind11::dict sensors;
  sensors["imu"] = sensor_dict(params_.imu, true, false);
  sensors["gps"] =
      sensor_dict(params_.gps, params_.gps.enable_gps, params_.gps.enable_gps_init_only);
  sensors["depth"] =
      sensor_dict(params_.depth, params_.depth.enable_depth, params_.depth.enable_depth_init_only);
  sensors["mag"] =
      sensor_dict(params_.mag, params_.mag.enable_mag, params_.mag.enable_mag_init_only);
  sensors["ahrs"] =
      sensor_dict(params_.ahrs, params_.ahrs.enable_ahrs, params_.ahrs.enable_ahrs_init_only);
  sensors["dvl"] =
      sensor_dict(params_.dvl, params_.dvl.enable_dvl, params_.dvl.enable_dvl_init_only);
  sensors["dynamics"] = sensor_dict(params_.dynamics, params_.dynamics.enable_dynamics,
                                    params_.dynamics.enable_dynamics_dropout_only);

  pybind11::dict base;
  base["enable"] = true;
  base["enable_extra_only"] = false;
  base["use_parameter_frame"] = false;
  base["parameter_frame"] = params_.base_frame;
  base["use_parameter_tf"] = params_.base.use_parameter_tf;
  base["tf_position"] = params_.base.parameter_tf.position;
  base["tf_orientation"] = params_.base.parameter_tf.orientation;
  sensors["base"] = base;
  p["sensors"] = sensors;

  // --- Comparison Methods ---
  pybind11::dict comparison;
  comparison["enable_loose_dvl_preintegration"] =
      params_.comparison.enable_loose_dvl_preintegration;
  comparison["enable_tight_dvl_preintegration"] =
      params_.comparison.enable_tight_dvl_preintegration;
  p["comparison"] = comparison;

  return p;
}

void FactorGraphPy::set_tf(const std::string& name, const Eigen::Vector3d& position,
                           const Eigen::Vector4d& quat_xyzw) {
  static const std::unordered_map<std::string, gtsam::Pose3 utils::TfBundle::*> kTfFields = {
      {"imu", &utils::TfBundle::target_T_imu},     {"gps", &utils::TfBundle::target_T_gps},
      {"depth", &utils::TfBundle::target_T_depth}, {"mag", &utils::TfBundle::target_T_mag},
      {"ahrs", &utils::TfBundle::target_T_ahrs},   {"dvl", &utils::TfBundle::target_T_dvl},
      {"base", &utils::TfBundle::target_T_base},   {"com", &utils::TfBundle::target_T_com}};

  auto it = kTfFields.find(name);
  if (it == kTfFields.end()) {
    throw std::invalid_argument("Unknown transform name: " + name);
  }
  tfs_.*(it->second) = toPose3(position, quat_xyzw);
}

utils::QueueBundle FactorGraphPy::to_bundle(const ImuBatch& imu, const OdomBatch& gps,
                                            const DepthBatch& depth, const MagBatch& mag,
                                            const AhrsBatch& ahrs, const TwistBatch& dvl,
                                            const WrenchBatch& wrench) {
  utils::QueueBundle queues;

  for (const auto& [t, accel, gyro, accel_cov, gyro_cov] : imu) {
    auto msg = std::make_shared<utils::ImuData>();
    msg->timestamp = t;
    msg->linear_acceleration = accel;
    msg->angular_velocity = gyro;
    msg->linear_acceleration_covariance = accel_cov;
    msg->angular_velocity_covariance = gyro_cov;
    queues.imu.push_back(msg);
  }

  for (const auto& [t, position, pose_cov] : gps) {
    auto msg = std::make_shared<utils::OdometryData>();
    msg->timestamp = t;
    msg->pose = gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(position));
    msg->pose_covariance = pose_cov;
    queues.gps.push_back(msg);
  }

  for (const auto& [t, depth_z, pose_cov] : depth) {
    auto msg = std::make_shared<utils::OdometryData>();
    msg->timestamp = t;
    msg->pose = gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(0, 0, depth_z));
    msg->pose_covariance = pose_cov;
    queues.depth.push_back(msg);
  }

  for (const auto& [t, field, field_cov] : mag) {
    auto msg = std::make_shared<utils::MagneticFieldData>();
    msg->timestamp = t;
    msg->magnetic_field = field;
    msg->magnetic_field_covariance = field_cov;
    queues.mag.push_back(msg);
  }

  for (const auto& [t, quat_xyzw, orientation_cov] : ahrs) {
    auto msg = std::make_shared<utils::AhrsData>();
    msg->timestamp = t;
    msg->orientation =
        gtsam::Rot3::Quaternion(quat_xyzw(3), quat_xyzw(0), quat_xyzw(1), quat_xyzw(2));
    msg->orientation_covariance = orientation_cov;
    queues.ahrs.push_back(msg);
  }

  for (const auto& [t, velocity, twist_cov] : dvl) {
    auto msg = std::make_shared<utils::TwistData>();
    msg->timestamp = t;
    msg->linear_velocity = velocity;
    msg->twist_covariance = twist_cov;
    queues.dvl.push_back(msg);
  }

  for (const auto& [t, force_torque] : wrench) {
    auto msg = std::make_shared<utils::WrenchData>();
    msg->timestamp = t;
    msg->force = force_torque.head<3>();
    msg->torque = force_torque.tail<3>();
    queues.wrench.push_back(msg);
  }

  return queues;
}

pybind11::dict FactorGraphPy::from_bundle(const utils::QueueBundle& queues) {
  ImuBatch imu;
  for (const auto& m : queues.imu) {
    imu.emplace_back(m->timestamp, m->linear_acceleration, m->angular_velocity,
                     m->linear_acceleration_covariance, m->angular_velocity_covariance);
  }

  OdomBatch gps;
  for (const auto& m : queues.gps) {
    gps.emplace_back(m->timestamp, m->pose.translation(), m->pose_covariance);
  }

  DepthBatch depth;
  for (const auto& m : queues.depth) {
    depth.emplace_back(m->timestamp, m->pose.translation().z(), m->pose_covariance);
  }

  MagBatch mag;
  for (const auto& m : queues.mag) {
    mag.emplace_back(m->timestamp, m->magnetic_field, m->magnetic_field_covariance);
  }

  AhrsBatch ahrs;
  for (const auto& m : queues.ahrs) {
    gtsam::Quaternion q = m->orientation.toQuaternion();
    ahrs.emplace_back(m->timestamp, Eigen::Vector4d(q.x(), q.y(), q.z(), q.w()),
                      m->orientation_covariance);
  }

  TwistBatch dvl;
  for (const auto& m : queues.dvl) {
    dvl.emplace_back(m->timestamp, m->linear_velocity, m->twist_covariance);
  }

  WrenchBatch wrench;
  for (const auto& m : queues.wrench) {
    Vector6d force_torque;
    force_torque << m->force, m->torque;
    wrench.emplace_back(m->timestamp, force_torque);
  }

  pybind11::dict batches;
  batches["imu"] = imu;
  batches["gps"] = gps;
  batches["depth"] = depth;
  batches["mag"] = mag;
  batches["ahrs"] = ahrs;
  batches["dvl"] = dvl;
  batches["wrench"] = wrench;
  return batches;
}

bool FactorGraphPy::initialize(double current_time, const ImuBatch& imu, const OdomBatch& gps,
                               const DepthBatch& depth, const MagBatch& mag, const AhrsBatch& ahrs,
                               const TwistBatch& dvl, const WrenchBatch& wrench) {
  if (is_initialized_) {
    return true;
  }

  utils::QueueBundle queues = to_bundle(imu, gps, depth, mag, ahrs, dvl, wrench);

  if (auto init_state = state_init_->update(current_time, queues, tfs_)) {
    core_->initialize(*init_state, tfs_);
    is_initialized_ = true;
  }
  return is_initialized_;
}

pybind11::object FactorGraphPy::update(double target_time, const ImuBatch& imu,
                                       const OdomBatch& gps, const DepthBatch& depth,
                                       const MagBatch& mag, const AhrsBatch& ahrs,
                                       const TwistBatch& dvl, const WrenchBatch& wrench) {
  if (!is_initialized_) {
    return pybind11::none();
  }

  utils::QueueBundle queues = to_bundle(imu, gps, depth, mag, ahrs, dvl, wrench);
  auto leftover = core_->update(target_time, queues);
  if (!leftover) {
    return pybind11::none();
  }
  return from_bundle(*leftover);
}

pybind11::dict FactorGraphPy::optimize() {
  if (!is_initialized_) {
    return {};
  }

  auto opt_result = core_->optimize();
  if (!opt_result) {
    return {};
  }

  pybind11::dict result = toStateDict(opt_result->target_time, opt_result->pose,
                                      opt_result->velocity, opt_result->imu_bias);

  if (params_.publish_pose_cov) result["pose_cov"] = opt_result->pose_cov;
  if (params_.publish_velocity_cov) result["vel_cov"] = opt_result->vel_cov;
  if (params_.publish_imu_bias_cov) result["bias_cov"] = opt_result->bias_cov;

  if (params_.publish_smoothed_path && !opt_result->all_estimates.empty()) {
    const gtsam::Values& estimates = opt_result->all_estimates;
    pybind11::list smoothed;

    for (const auto& [time_ns, x_key] : core_->snapshotTimeKeys()) {
      if (!estimates.exists(x_key)) {
        continue;
      }
      size_t step = gtsam::Symbol(x_key).index();

      std::optional<gtsam::Vector3> velocity;
      if (estimates.exists(V(step))) {
        velocity = estimates.at<gtsam::Vector3>(V(step));
      }
      std::optional<gtsam::imuBias::ConstantBias> bias;
      if (estimates.exists(B(step))) {
        bias = estimates.at<gtsam::imuBias::ConstantBias>(B(step));
      }
      smoothed.append(toStateDict(static_cast<double>(time_ns) * 1e-9,
                                  estimates.at<gtsam::Pose3>(x_key), velocity, bias));
    }
    result["smoothed_path"] = smoothed;
  }
  return result;
}

void FactorGraphPy::reset() {
  core_ = std::make_unique<FactorGraphCore>(params_);
  core_->setLogCallback(&pyLogCallback);
  state_init_ = std::make_unique<StateInitializer>(params_);
  is_initialized_ = false;
}

}  // namespace coug_fgo

using coug_fgo::FactorGraphPy;

PYBIND11_MODULE(coug_fgo_py, m) {
  m.doc() = "Python bindings for the FactorGraphCore.";

  pybind11::class_<FactorGraphPy>(m, "FactorGraphPy")
      .def(pybind11::init<const std::vector<std::string>&, const std::string&>(),
           pybind11::arg("config_paths"), pybind11::arg("namespace") = "")
      .def("get_params", &FactorGraphPy::get_params)
      .def("set_tf", &FactorGraphPy::set_tf, pybind11::arg("name"), pybind11::arg("position"),
           pybind11::arg("quat_xyzw"))
      .def("initialize", &FactorGraphPy::initialize, pybind11::arg("current_time"),
           pybind11::arg("imu") = FactorGraphPy::ImuBatch(),
           pybind11::arg("gps") = FactorGraphPy::OdomBatch(),
           pybind11::arg("depth") = FactorGraphPy::DepthBatch(),
           pybind11::arg("mag") = FactorGraphPy::MagBatch(),
           pybind11::arg("ahrs") = FactorGraphPy::AhrsBatch(),
           pybind11::arg("dvl") = FactorGraphPy::TwistBatch(),
           pybind11::arg("wrench") = FactorGraphPy::WrenchBatch())
      .def("update", &FactorGraphPy::update, pybind11::arg("target_time"),
           pybind11::arg("imu") = FactorGraphPy::ImuBatch(),
           pybind11::arg("gps") = FactorGraphPy::OdomBatch(),
           pybind11::arg("depth") = FactorGraphPy::DepthBatch(),
           pybind11::arg("mag") = FactorGraphPy::MagBatch(),
           pybind11::arg("ahrs") = FactorGraphPy::AhrsBatch(),
           pybind11::arg("dvl") = FactorGraphPy::TwistBatch(),
           pybind11::arg("wrench") = FactorGraphPy::WrenchBatch())
      .def("optimize", &FactorGraphPy::optimize)
      .def("reset", &FactorGraphPy::reset)
      .def_property_readonly("is_initialized", &FactorGraphPy::is_initialized);
}
