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
 * @date Jan 2026
 */

#include "coug_fgo/factor_graph_py.hpp"

#include <rclcpp/rclcpp.hpp>

FactorGraphPy::FactorGraphPy(const std::string& config_path) {
  if (!rclcpp::ok()) {
    rclcpp::init(0, nullptr);
  }

  rclcpp::NodeOptions options;
  options.arguments({"--ros-args", "--params-file", config_path});
  auto dummy_node = std::make_shared<rclcpp::Node>("factor_graph_node", options);

  auto param_listener = std::make_shared<factor_graph_node::ParamListener>(
      dummy_node->get_node_parameters_interface());

  params_ = param_listener->get_params();
  tfs_ = extract_tfs(params_);

  state_init_ = std::make_unique<coug_fgo::utils::StateInitializer>(params_);
  core_ = std::make_unique<coug_fgo::FactorGraphCore>(params_);
}

bool FactorGraphPy::initialize_graph(double current_time) {
  if (is_initialized_) {
    return true;
  }

  coug_fgo::utils::QueueBundle init_queues = std::move(queues_);
  queues_ = coug_fgo::utils::QueueBundle();

  if (state_init_->update(current_time, init_queues)) {
    state_init_->compute(tfs_);
    core_->initialize(*state_init_, tfs_);
    is_initialized_ = true;
  }
  return is_initialized_;
}

void FactorGraphPy::add_imu(double timestamp, const Eigen::Vector3d& accel,
                            const Eigen::Vector3d& gyro, const Eigen::Matrix3d& accel_cov,
                            const Eigen::Matrix3d& gyro_cov) {
  auto msg = std::make_shared<coug_fgo::utils::ImuData>();
  msg->timestamp = timestamp;
  msg->linear_acceleration = accel;
  msg->angular_velocity = gyro;
  msg->linear_acceleration_covariance = accel_cov;
  msg->angular_velocity_covariance = gyro_cov;
  queues_.imu.push_back(msg);
}

void FactorGraphPy::add_dvl(double timestamp, const Eigen::Vector3d& velocity,
                            const Eigen::Matrix<double, 6, 6>& twist_cov) {
  auto msg = std::make_shared<coug_fgo::utils::TwistData>();
  msg->timestamp = timestamp;
  msg->linear_velocity = velocity;
  msg->twist_covariance = twist_cov;
  queues_.dvl.push_back(msg);
}

void FactorGraphPy::add_ahrs(double timestamp, const Eigen::Vector4d& quat_xyzw,
                             const Eigen::Matrix3d& orientation_cov) {
  auto msg = std::make_shared<coug_fgo::utils::AhrsData>();
  msg->timestamp = timestamp;
  msg->orientation =
      gtsam::Rot3::Quaternion(quat_xyzw(3), quat_xyzw(0), quat_xyzw(1), quat_xyzw(2));
  msg->orientation_covariance = orientation_cov;
  queues_.ahrs.push_back(msg);
}

void FactorGraphPy::add_depth(double timestamp, double depth_z,
                              const Eigen::Matrix<double, 6, 6>& pose_cov) {
  auto msg = std::make_shared<coug_fgo::utils::OdometryData>();
  msg->timestamp = timestamp;
  msg->pose = gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(0, 0, depth_z));
  msg->pose_covariance = pose_cov;
  queues_.depth.push_back(msg);
}

void FactorGraphPy::add_gps(double timestamp, const Eigen::Vector3d& position,
                            const Eigen::Matrix<double, 6, 6>& pose_cov) {
  auto msg = std::make_shared<coug_fgo::utils::OdometryData>();
  msg->timestamp = timestamp;
  msg->pose = gtsam::Pose3(gtsam::Rot3(), position);
  msg->pose_covariance = pose_cov;
  queues_.gps.push_back(msg);
}

void FactorGraphPy::add_mag(double timestamp, const Eigen::Vector3d& mag_field,
                            const Eigen::Matrix3d& mag_cov) {
  auto msg = std::make_shared<coug_fgo::utils::MagneticFieldData>();
  msg->timestamp = timestamp;
  msg->magnetic_field = mag_field;
  msg->magnetic_field_covariance = mag_cov;
  queues_.mag.push_back(msg);
}

void FactorGraphPy::add_wrench(double timestamp, const Eigen::VectorXd& force_torque) {
  auto msg = std::make_shared<coug_fgo::utils::WrenchData>();
  msg->timestamp = timestamp;
  msg->force = gtsam::Vector3(force_torque(0), force_torque(1), force_torque(2));
  msg->torque = gtsam::Vector3(force_torque(3), force_torque(4), force_torque(5));
  queues_.wrench.push_back(msg);
}

bool FactorGraphPy::update_graph(double target_time) {
  if (!is_initialized_) {
    return false;
  }

  coug_fgo::utils::QueueBundle update_queues = std::move(queues_);
  queues_ = coug_fgo::utils::QueueBundle();

  auto update_result = core_->update(target_time, update_queues);

  if (update_result) {
    queues_.imu.insert(queues_.imu.end(), update_result->unused_imu.begin(),
                       update_result->unused_imu.end());
    queues_.dvl.insert(queues_.dvl.end(), update_result->unused_dvl.begin(),
                       update_result->unused_dvl.end());
  } else {
    queues_.imu.insert(queues_.imu.end(), update_queues.imu.begin(), update_queues.imu.end());
    queues_.dvl.insert(queues_.dvl.end(), update_queues.dvl.begin(), update_queues.dvl.end());
    queues_.gps.insert(queues_.gps.end(), update_queues.gps.begin(), update_queues.gps.end());
    queues_.depth.insert(queues_.depth.end(), update_queues.depth.begin(),
                         update_queues.depth.end());
    queues_.ahrs.insert(queues_.ahrs.end(), update_queues.ahrs.begin(), update_queues.ahrs.end());
    queues_.mag.insert(queues_.mag.end(), update_queues.mag.begin(), update_queues.mag.end());
    queues_.wrench.insert(queues_.wrench.end(), update_queues.wrench.begin(),
                          update_queues.wrench.end());
    return false;
  }
  return true;
}

pybind11::dict FactorGraphPy::optimize_graph() {
  pybind11::dict result;
  if (!is_initialized_) {
    return result;
  }

  auto opt_result = core_->optimize();
  if (opt_result) {
    gtsam::Pose3 p = opt_result->pose;
    gtsam::Vector3 v = opt_result->velocity;
    gtsam::Vector3 b_a = opt_result->imu_bias.accelerometer();
    gtsam::Vector3 b_g = opt_result->imu_bias.gyroscope();
    gtsam::Quaternion q = p.rotation().toQuaternion();

    result["time"] = opt_result->target_time;

    result["x"] = p.translation().x();
    result["y"] = p.translation().y();
    result["z"] = p.translation().z();
    result["qx"] = q.x();
    result["qy"] = q.y();
    result["qz"] = q.z();
    result["qw"] = q.w();

    result["vx"] = v.x();
    result["vy"] = v.y();
    result["vz"] = v.z();

    result["bias_accel_x"] = b_a.x();
    result["bias_accel_y"] = b_a.y();
    result["bias_accel_z"] = b_a.z();
    result["bias_gyro_x"] = b_g.x();
    result["bias_gyro_y"] = b_g.y();
    result["bias_gyro_z"] = b_g.z();

    if (params_.publish_pose_cov) result["pose_cov"] = opt_result->pose_cov;
    if (params_.publish_velocity_cov) result["vel_cov"] = opt_result->vel_cov;
    if (params_.publish_imu_bias_cov) result["bias_cov"] = opt_result->bias_cov;
  }
  return result;
}

coug_fgo::utils::TfBundle FactorGraphPy::extract_tfs(const factor_graph_node::Params& p) {
  coug_fgo::utils::TfBundle tfs;

  auto make_tf = [](const std::vector<double>& pos,
                    const std::vector<double>& quat) -> gtsam::Pose3 {
    // Convert ROS (x, y, z, w) to GTSAM (w, x, y, z) quaternion conventions
    gtsam::Rot3 r = gtsam::Rot3::Quaternion(quat[3], quat[0], quat[1], quat[2]);
    gtsam::Point3 t(pos[0], pos[1], pos[2]);
    return gtsam::Pose3(r, t);
  };

  tfs.target_T_imu = make_tf(p.imu.parameter_tf.position, p.imu.parameter_tf.orientation);
  tfs.target_T_dvl = make_tf(p.dvl.parameter_tf.position, p.dvl.parameter_tf.orientation);
  tfs.target_T_depth = make_tf(p.depth.parameter_tf.position, p.depth.parameter_tf.orientation);
  tfs.target_T_gps = make_tf(p.gps.parameter_tf.position, p.gps.parameter_tf.orientation);
  tfs.target_T_ahrs = make_tf(p.ahrs.parameter_tf.position, p.ahrs.parameter_tf.orientation);
  tfs.target_T_mag = make_tf(p.mag.parameter_tf.position, p.mag.parameter_tf.orientation);
  tfs.target_T_com = make_tf(p.dynamics.parameter_tf.position, p.dynamics.parameter_tf.orientation);
  tfs.target_T_base = make_tf(p.base.parameter_tf.position, p.base.parameter_tf.orientation);

  return tfs;
}

PYBIND11_MODULE(pybind11fgo, m) {
  m.doc() = "Python bindings for the FactorGraphCore.";

  pybind11::class_<FactorGraphPy>(m, "FactorGraphPy")
      .def(pybind11::init<const std::string&>(), pybind11::arg("config_path"))
      .def("add_imu", &FactorGraphPy::add_imu, pybind11::arg("timestamp"), pybind11::arg("accel"),
           pybind11::arg("gyro"), pybind11::arg("accel_cov"), pybind11::arg("gyro_cov"))
      .def("add_dvl", &FactorGraphPy::add_dvl, pybind11::arg("timestamp"),
           pybind11::arg("velocity"), pybind11::arg("twist_cov"))
      .def("add_ahrs", &FactorGraphPy::add_ahrs, pybind11::arg("timestamp"),
           pybind11::arg("quat_xyzw"), pybind11::arg("orientation_cov"))
      .def("add_depth", &FactorGraphPy::add_depth, pybind11::arg("timestamp"),
           pybind11::arg("depth_z"), pybind11::arg("pose_cov"))
      .def("add_gps", &FactorGraphPy::add_gps, pybind11::arg("timestamp"),
           pybind11::arg("position"), pybind11::arg("pose_cov"))
      .def("add_mag", &FactorGraphPy::add_mag, pybind11::arg("timestamp"),
           pybind11::arg("mag_field"), pybind11::arg("mag_cov"))
      .def("add_wrench", &FactorGraphPy::add_wrench, pybind11::arg("timestamp"),
           pybind11::arg("force_torque"))
      .def("initialize_graph", &FactorGraphPy::initialize_graph, pybind11::arg("current_time"))
      .def("update_graph", &FactorGraphPy::update_graph, pybind11::arg("target_time"))
      .def("optimize_graph", &FactorGraphPy::optimize_graph);
}
