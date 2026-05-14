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
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)


def generate_launch_description() -> LaunchDescription:

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

    # imu_link_frame = PythonExpression(
    #     [
    #         "'",
    #         auv_ns,
    #         "/imu_link' if '",
    #         auv_ns,
    #         "' != '' else 'imu_link'",
    #     ]
    # )

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

    # com_link_frame = PythonExpression(
    #     [
    #         "'",
    #         auv_ns,
    #         "/com_link' if '",
    #         auv_ns,
    #         "' != '' else 'com_link'",
    #     ]
    # )

    modem_link_frame = PythonExpression(
        [
            "'",
            auv_ns,
            "/modem_link' if '",
            auv_ns,
            "' != '' else 'modem_link'",
        ]
    )

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
                description="Whether navsat_odom_node owns and publishes the ENU origin",
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
                        "odom_frame": odom_frame,
                        "base_frame": base_link_frame,
                        "parameter_frame": dvl_link_frame,
                    },
                ],
            ),
            Node(
                package="coug_fgo",
                executable="fluid_pressure_odom",
                name="fluid_pressure_odom_node",
                parameters=[
                    fleet_params,
                    auv_params,
                    {
                        "use_sim_time": use_sim_time,
                        "map_frame": "map",
                        "parameter_child_frame": depth_link_frame,
                    },
                ],
            ),
            Node(
                package="coug_fgo",
                executable="navsat_odom",
                name="navsat_odom_node",
                parameters=[
                    fleet_params,
                    auv_params,
                    {
                        "use_sim_time": use_sim_time,
                        "map_frame": "map",
                        "parameter_child_frame": gps_link_frame,
                        "set_origin": set_origin,
                    },
                ],
            ),
            Node(
                package="coug_fgo",
                executable="seatrac_x150_imu",
                name="seatrac_x150_imu_node",
                parameters=[
                    fleet_params,
                    auv_params,
                    {
                        "use_sim_time": use_sim_time,
                        "parameter_frame": modem_link_frame,
                    },
                ],
            ),
            Node(
                package="coug_fgo",
                executable="imu_ned_to_enu",
                name="seatrac_imu_ned_to_enu_node",
                parameters=[
                    fleet_params,
                    auv_params,
                    {
                        "use_sim_time": use_sim_time,
                    },
                ],
            ),
            Node(
                package="coug_fgo",
                executable="odom_ned_to_enu",
                name="odom_ned_to_enu_node",
                parameters=[
                    fleet_params,
                    auv_params,
                    {
                        "use_sim_time": use_sim_time,
                    },
                ],
            ),
            Node(
                package="topic_tools",
                executable="relay",
                name="gps_to_truth_relay",
                parameters=[
                    fleet_params,
                    auv_params,
                    {
                        "use_sim_time": use_sim_time,
                    },
                ],
            ),
            # https://docs.ros.org/en/melodic/api/robot_localization/html/state_estimation_nodes.html
            Node(
                package="robot_localization",
                executable="ekf_node",
                name="ekf_filter_node_map",
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
                remappings=[("odometry/filtered", "odometry/global")],
            ),
        ]
    )
