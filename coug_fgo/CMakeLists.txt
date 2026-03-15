cmake_minimum_required(VERSION 3.8)
project(coug_fgo)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# --- Dependencies ---
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()
find_package(GTSAM REQUIRED)

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
  origin_manager_parameters
  "params/origin_manager_params.yaml"
)

generate_parameter_library(
  dvl_a50_twist_parameters
  "params/dvl_a50_twist_params.yaml"
)

generate_parameter_library(
  dvl_a50_odom_parameters
  "params/dvl_a50_odom_params.yaml"
)

# --- Component Library ---
ament_auto_add_library(factor_graph_component SHARED
  src/factor_graph_core.cpp
  src/factor_graph.cpp
  src/origin_manager.cpp
  src/dvl_a50_twist.cpp
  src/dvl_a50_odom.cpp
)

target_include_directories(factor_graph_component PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${GTSAM_EIGEN_INCLUDE_DIR}
)

target_link_libraries(factor_graph_component
  gtsam
  gtsam_unstable
  factor_graph_parameters
  origin_manager_parameters
  dvl_a50_twist_parameters
  dvl_a50_odom_parameters
)

rclcpp_components_register_nodes(factor_graph_component
  "coug_fgo::FactorGraphNode"
  "coug_fgo::OriginManagerNode"
  "coug_fgo::DvlA50TwistNode"
  "coug_fgo::DvlA50OdomNode"
)

# --- Executables ---
ament_auto_add_executable(factor_graph src/factor_graph_node.cpp)
target_link_libraries(factor_graph factor_graph_component)

ament_auto_add_executable(origin_manager src/origin_manager_node.cpp)
target_link_libraries(origin_manager factor_graph_component)

ament_auto_add_executable(dvl_a50_twist src/dvl_a50_twist_node.cpp)
target_link_libraries(dvl_a50_twist factor_graph_component)

ament_auto_add_executable(dvl_a50_odom src/dvl_a50_odom_node.cpp)
target_link_libraries(dvl_a50_odom factor_graph_component)

# --- Testing ---
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

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
  USE_SCOPED_HEADER_INSTALL_DIR
)
