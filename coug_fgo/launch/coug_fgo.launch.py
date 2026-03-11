# Copyright (c) 2026 BYU FROST Lab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import (
    LaunchConfiguration,
    PythonExpression,
    PathJoinSubstitution,
    EnvironmentVariable,
)


def generate_launch_description():

    use_sim_time = LaunchConfiguration("use_sim_time")
    auv_ns = LaunchConfiguration("auv_ns")
    set_origin = LaunchConfiguration("set_origin")

    fleet_params = PathJoinSubstitution(
        [
            EnvironmentVariable("CONFIG_FOLDER"),
            "fleet",
            "coug_fgo_params.yaml",
        ]
    )
    auv_params = PathJoinSubstitution(
        [
            EnvironmentVariable("CONFIG_FOLDER"),
            PythonExpression(["'", auv_ns, "' + '_params.yaml'"]),
        ]
    )

    odom_frame = PythonExpression(
        ["'", auv_ns, "/odom' if '", auv_ns, "' != '' else 'odom'"]
    )

    dvl_odom_frame = PythonExpression(
        [
            "'",
            auv_ns,
            "/dvl_odom' if '",
            auv_ns,
            "' != '' else 'dvl_odom'",
        ]
    )

    base_link_frame = PythonExpression(
        [
            "'",
            auv_ns,
            "/base_link' if '",
            auv_ns,
            "' != '' else 'base_link'",
        ]
    )

    gps_link_frame = PythonExpression(
        [
            "'",
            auv_ns,
            "/gps_link' if '",
            auv_ns,
            "' != '' else 'gps_link'",
        ]
    )

    imu_link_frame = PythonExpression(
        [
            "'",
            auv_ns,
            "/imu_link' if '",
            auv_ns,
            "' != '' else 'imu_link'",
        ]
    )

    dvl_link_frame = PythonExpression(
        [
            "'",
            auv_ns,
            "/dvl_link' if '",
            auv_ns,
            "' != '' else 'dvl_link'",
        ]
    )

    depth_link_frame = PythonExpression(
        [
            "'",
            auv_ns,
            "/depth_link' if '",
            auv_ns,
            "' != '' else 'depth_link'",
        ]
    )

    com_link_frame = PythonExpression(
        [
            "'",
            auv_ns,
            "/com_link' if '",
            auv_ns,
            "' != '' else 'com_link'",
        ]
    )

    # modem_link_frame = PythonExpression(
    #     [
    #         "'",
    #         auv_ns,
    #         "/modem_link' if '",
    #         auv_ns,
    #         "' != '' else 'modem_link'",
    #     ]
    # )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="false",
                description="Use simulation/rosbag clock if true",
            ),
            DeclareLaunchArgument(
                "auv_ns",
                default_value="auv0",
                description="Namespace for the AUV (e.g. auv0)",
            ),
            DeclareLaunchArgument(
                "set_origin",
                default_value="true",
                description="Whether to set the origin (true) or subscribe to it (false)",
            ),
            DeclareLaunchArgument(
                "compare",
                default_value="false",
                description="Launch additional localization nodes if true",
            ),
            Node(
                package="coug_fgo",
                executable="factor_graph",
                name="factor_graph_node",
                parameters=[
                    fleet_params,
                    auv_params,
                    {
                        "use_sim_time": use_sim_time,
                        "map_frame": "map",
                        "odom_frame": odom_frame,
                        "base_frame": base_link_frame,
                        "target_frame": dvl_link_frame,
                        "imu.parameter_frame": imu_link_frame,
                        "dvl.parameter_frame": dvl_link_frame,
                        "depth.parameter_frame": depth_link_frame,
                        "gps.parameter_frame": gps_link_frame,
                        "mag.parameter_frame": imu_link_frame,
                        "ahrs.parameter_frame": imu_link_frame,
                        "dynamics.parameter_frame": com_link_frame,
                    },
                ],
            ),
            # ISAM2 (comparison)
            Node(
                package="coug_fgo",
                executable="factor_graph",
                name="factor_graph_node_isam2",
                condition=IfCondition(LaunchConfiguration("compare")),
                parameters=[
                    fleet_params,
                    auv_params,
                    {
                        "use_sim_time": use_sim_time,
                        "map_frame": "map",
                        "odom_frame": odom_frame,
                        "base_frame": base_link_frame,
                        "target_frame": dvl_link_frame,
                        "imu.parameter_frame": imu_link_frame,
                        "dvl.parameter_frame": dvl_link_frame,
                        "depth.parameter_frame": depth_link_frame,
                        "gps.parameter_frame": gps_link_frame,
                        "mag.parameter_frame": imu_link_frame,
                        "ahrs.parameter_frame": imu_link_frame,
                        "dynamics.parameter_frame": com_link_frame,
                        "global_odom_topic": "odometry/global_isam2",
                        "smoothed_path_topic": "smoothed_path_isam2",
                        "publish_global_tf": False,
                        "publish_smoothed_path": False,
                        "solver_type": "ISAM2",
                    },
                ],
            ),
            # TURTLMap (comparison)
            Node(
                package="coug_fgo",
                executable="factor_graph",
                name="factor_graph_node_lpi",
                condition=IfCondition(LaunchConfiguration("compare")),
                parameters=[
                    fleet_params,
                    auv_params,
                    {
                        "use_sim_time": use_sim_time,
                        "map_frame": "map",
                        "odom_frame": odom_frame,
                        "base_frame": base_link_frame,
                        "target_frame": dvl_link_frame,
                        "imu.parameter_frame": imu_link_frame,
                        "dvl.parameter_frame": dvl_link_frame,
                        "depth.parameter_frame": depth_link_frame,
                        "gps.parameter_frame": gps_link_frame,
                        "mag.parameter_frame": imu_link_frame,
                        "ahrs.parameter_frame": imu_link_frame,
                        "dynamics.parameter_frame": com_link_frame,
                        "max_update_rate": 4.0,
                        "global_odom_topic": "odometry/global_lpi",
                        "smoothed_path_topic": "smoothed_path_lpi",
                        "publish_global_tf": False,
                        "comparison.enable_loose_dvl_preintegration": True,
                        # "comparison.enable_pseudo_dvl_w_imu": True,
                    },
                ],
            ),
            # AQUA-SLAM (comparison)
            # Node(
            #     package="coug_fgo",
            #     executable="factor_graph",
            #     name="factor_graph_node_tpi",
            #     condition=IfCondition(LaunchConfiguration("compare")),
            #     parameters=[
            #         fleet_params,
            #         auv_params,
            #         {
            #             "use_sim_time": use_sim_time,
            #             "map_frame": "map",
            #             "odom_frame": odom_frame,
            #             "base_frame": base_link_frame,
            #             "target_frame": dvl_link_frame,
            #             "imu.parameter_frame": imu_link_frame,
            #             "dvl.parameter_frame": dvl_link_frame,
            #             "depth.parameter_frame": depth_link_frame,
            #             "gps.parameter_frame": gps_link_frame,
            #             "mag.parameter_frame": imu_link_frame,
            #             "ahrs.parameter_frame": imu_link_frame,
            #             "dynamics.parameter_frame": com_link_frame,
            #             "global_odom_topic": "odometry/global_tpi",
            #             "smoothed_path_topic": "smoothed_path_tpi",
            #             "publish_global_tf": False,
            #             "comparison.enable_tight_dvl_preintegration": True,
            #         },
            #     ],
            # ),
            Node(
                package="coug_fgo",
                executable="origin_manager",
                name="origin_manager_node",
                parameters=[
                    fleet_params,
                    auv_params,
                    {
                        "use_sim_time": use_sim_time,
                        "map_frame": "map",
                        "set_origin": set_origin,
                        "parameter_child_frame": gps_link_frame,
                    },
                ],
            ),
            Node(
                package="coug_fgo",
                executable="dvl_a50_twist",
                name="dvl_a50_twist_node",
                parameters=[
                    fleet_params,
                    auv_params,
                    {
                        "use_sim_time": use_sim_time,
                        "parameter_frame": dvl_link_frame,
                    },
                ],
            ),
            Node(
                package="coug_fgo",
                executable="dvl_a50_odom",
                name="dvl_a50_odom_node",
                parameters=[
                    fleet_params,
                    auv_params,
                    {
                        "use_sim_time": use_sim_time,
                        "dvl_odom_frame": dvl_odom_frame,
                        "base_frame": base_link_frame,
                        "parameter_frame": dvl_link_frame,
                    },
                ],
            ),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="map_to_dvl_odom_transform",
                arguments=[
                    "--x",
                    "0",
                    "--y",
                    "0",
                    "--z",
                    "0",
                    "--yaw",
                    "1.57079632679",
                    "--pitch",
                    "0",
                    "--roll",
                    "3.14159265359",
                    "--frame-id",
                    "map",
                    "--child-frame-id",
                    dvl_odom_frame,
                ],
                parameters=[{"use_sim_time": use_sim_time}],
            ),
            # --- Robot Localization Pipeline ---
            # https://docs.ros.org/en/melodic/api/robot_localization/html/state_estimation_nodes.html
            Node(
                package="robot_localization",
                executable="ekf_node",
                name="ekf_filter_node_odom",
                parameters=[
                    fleet_params,
                    auv_params,
                    {
                        "use_sim_time": use_sim_time,
                        "odom_frame": odom_frame,
                        "base_link_frame": base_link_frame,
                        "world_frame": odom_frame,
                    },
                ],
                remappings=[("odometry/filtered", "odometry/local")],
            ),
            # https://docs.ros.org/en/melodic/api/robot_localization/html/state_estimation_nodes.html
            Node(
                package="robot_localization",
                executable="ekf_node",
                name="ekf_filter_node_map",
                condition=IfCondition(LaunchConfiguration("compare")),
                parameters=[
                    fleet_params,
                    auv_params,
                    {
                        "use_sim_time": use_sim_time,
                        "map_frame": "map",
                        "odom_frame": odom_frame,
                        "base_link_frame": base_link_frame,
                        "world_frame": "map",
                    },
                ],
                remappings=[("odometry/filtered", "odometry/global_ekf")],
            ),
            # https://docs.ros.org/en/melodic/api/robot_localization/html/state_estimation_nodes.html
            Node(
                package="robot_localization",
                executable="ukf_node",
                name="ukf_filter_node_map",
                condition=IfCondition(LaunchConfiguration("compare")),
                parameters=[
                    fleet_params,
                    auv_params,
                    {
                        "use_sim_time": use_sim_time,
                        "map_frame": "map",
                        "odom_frame": odom_frame,
                        "base_link_frame": base_link_frame,
                        "world_frame": "map",
                    },
                ],
                remappings=[("odometry/filtered", "odometry/global_ukf")],
            ),
        ]
    )
