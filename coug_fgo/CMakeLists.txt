cmake_minimum_required(VERSION 3.22)
project(coug_fgo)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -flto=auto)
  add_link_options(-flto=auto)
endif()

# --- Dependencies ---
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()
find_package(GTSAM REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GeographicLib REQUIRED IMPORTED_TARGET geographiclib)

# --- GTSAM & Eigen Setup ---
get_target_property(GTSAM_TARGET_INC gtsam INTERFACE_INCLUDE_DIRECTORIES)
find_path(GTSAM_EIGEN_INCLUDE_DIR Eigen/Core
  PATHS ${GTSAM_TARGET_INC}
  PATH_SUFFIXES gtsam/3rdparty/Eigen
  NO_DEFAULT_PATH
)

# --- Parameters ---
generate_parameter_library(
  factor_graph_parameters
  "params/factor_graph_params.yaml"
)

generate_parameter_library(
  dvl_a50_twist_parameters
  "params/dvl_a50_twist_params.yaml"
)

generate_parameter_library(
  dvl_a50_odom_parameters
  "params/dvl_a50_odom_params.yaml"
)

generate_parameter_library(
  fluid_pressure_odom_parameters
  "params/fluid_pressure_odom_params.yaml"
)

generate_parameter_library(
  imu_ned_to_enu_parameters
  "params/imu_ned_to_enu_params.yaml"
)

generate_parameter_library(
  odom_ned_to_enu_parameters
  "params/odom_ned_to_enu_params.yaml"
)

generate_parameter_library(
  navsat_odom_parameters
  "params/navsat_odom_params.yaml"
)

generate_parameter_library(
  seatrac_x150_imu_parameters
  "params/seatrac_x150_imu_params.yaml"
)

# --- Component Library ---
ament_auto_add_library(factor_graph_component SHARED
  src/factor_graph_core.cpp
  src/factor_graph.cpp
  src/state_initializer.cpp
  src/dvl_a50_twist.cpp
  src/dvl_a50_odom.cpp
  src/fluid_pressure_odom.cpp
  src/imu_ned_to_enu.cpp
  src/odom_ned_to_enu.cpp
  src/navsat_odom.cpp
  src/seatrac_x150_imu.cpp
)

target_include_directories(factor_graph_component PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${GTSAM_EIGEN_INCLUDE_DIR}
)

target_link_libraries(factor_graph_component
  gtsam
  gtsam_unstable
  factor_graph_parameters
  dvl_a50_twist_parameters
  dvl_a50_odom_parameters
  fluid_pressure_odom_parameters
  imu_ned_to_enu_parameters
  odom_ned_to_enu_parameters
  navsat_odom_parameters
  seatrac_x150_imu_parameters
  PkgConfig::GeographicLib
)

rclcpp_components_register_nodes(factor_graph_component
  "coug_fgo::FactorGraphNode"
  "coug_fgo::DvlA50TwistNode"
  "coug_fgo::DvlA50OdomNode"
  "coug_fgo::FluidPressureOdomNode"
  "coug_fgo::ImuNedToEnuNode"
  "coug_fgo::OdomNedToEnuNode"
  "coug_fgo::NavsatOdomNode"
  "coug_fgo::SeatracX150ImuNode"
)

# --- Executables ---
ament_auto_add_executable(factor_graph src/factor_graph_node.cpp)
target_link_libraries(factor_graph factor_graph_component)

ament_auto_add_executable(dvl_a50_twist src/dvl_a50_twist_node.cpp)
target_link_libraries(dvl_a50_twist factor_graph_component)

ament_auto_add_executable(dvl_a50_odom src/dvl_a50_odom_node.cpp)
target_link_libraries(dvl_a50_odom factor_graph_component)

ament_auto_add_executable(fluid_pressure_odom src/fluid_pressure_odom_node.cpp)
target_link_libraries(fluid_pressure_odom factor_graph_component)

ament_auto_add_executable(imu_ned_to_enu src/imu_ned_to_enu_node.cpp)
target_link_libraries(imu_ned_to_enu factor_graph_component)

ament_auto_add_executable(odom_ned_to_enu src/odom_ned_to_enu_node.cpp)
target_link_libraries(odom_ned_to_enu factor_graph_component)

ament_auto_add_executable(navsat_odom src/navsat_odom_node.cpp)
target_link_libraries(navsat_odom factor_graph_component)

ament_auto_add_executable(seatrac_x150_imu src/seatrac_x150_imu_node.cpp)
target_link_libraries(seatrac_x150_imu factor_graph_component)

# --- Python Bindings ---
pybind11_add_module(pybind11fgo src/factor_graph_py.cpp)
target_include_directories(pybind11fgo PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${GTSAM_EIGEN_INCLUDE_DIR}
)
target_link_libraries(pybind11fgo PRIVATE
  factor_graph_component
)
install(FILES $<TARGET_FILE:pybind11fgo>
  DESTINATION "lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
)

# --- Testing ---
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_ahrs_factor test/test_ahrs_factor.cpp)
  target_include_directories(test_ahrs_factor PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${GTSAM_EIGEN_INCLUDE_DIR}
  )
  target_link_libraries(test_ahrs_factor gtsam)

  ament_add_gtest(test_auv_dynamics_factor test/test_auv_dynamics_factor.cpp)
  target_include_directories(test_auv_dynamics_factor PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${GTSAM_EIGEN_INCLUDE_DIR}
  )
  target_link_libraries(test_auv_dynamics_factor gtsam)

  ament_add_gtest(test_const_vel_factor test/test_const_vel_factor.cpp)
  target_include_directories(test_const_vel_factor PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${GTSAM_EIGEN_INCLUDE_DIR}
  )
  target_link_libraries(test_const_vel_factor gtsam)

  ament_add_gtest(test_depth_factor test/test_depth_factor.cpp)
  target_include_directories(test_depth_factor PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${GTSAM_EIGEN_INCLUDE_DIR}
  )
  target_link_libraries(test_depth_factor gtsam)

  ament_add_gtest(test_dvl_factor test/test_dvl_factor.cpp)
  target_include_directories(test_dvl_factor PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${GTSAM_EIGEN_INCLUDE_DIR}
  )
  target_link_libraries(test_dvl_factor gtsam)

  ament_add_gtest(test_dvl_loose_preint_factor test/test_dvl_loose_preint_factor.cpp)
  target_include_directories(test_dvl_loose_preint_factor PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${GTSAM_EIGEN_INCLUDE_DIR}
  )
  target_link_libraries(test_dvl_loose_preint_factor gtsam)

  ament_add_gtest(test_dvl_tight_preint_factor test/test_dvl_tight_preint_factor.cpp)
  target_include_directories(test_dvl_tight_preint_factor PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${GTSAM_EIGEN_INCLUDE_DIR}
  )
  target_link_libraries(test_dvl_tight_preint_factor gtsam)

  ament_add_gtest(test_gps_factor test/test_gps_factor.cpp)
  target_include_directories(test_gps_factor PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${GTSAM_EIGEN_INCLUDE_DIR}
  )
  target_link_libraries(test_gps_factor gtsam)

  ament_add_gtest(test_mag_factor test/test_mag_factor.cpp)
  target_include_directories(test_mag_factor PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${GTSAM_EIGEN_INCLUDE_DIR}
  )
  target_link_libraries(test_mag_factor gtsam)
endif()

ament_auto_package(
  INSTALL_TO_SHARE
  launch
  config
  params
  rviz
  plotjuggler
  USE_SCOPED_HEADER_INSTALL_DIR
)
