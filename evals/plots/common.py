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

ALGORITHMS = [
    "FL-B",
    "iS2-B",
    "FL-LPI",
    "FL-TPI",
    "IEKF",
    # "UKF",
    # "EKF",
    "TM",
    "SBG",
    "DVL",
]
COLORS = {
    "FL-B": "#55A868",
    "iS2-B": "#DD8452",
    "FL-LPI": "#4C72B0",
    "FL-TPI": "#C44E52",
    "IEKF": "#8172B2",
    "UKF": "#937860",
    "EKF": "#DA8BC3",
    "TM": "#8C8C8C",
    "SBG": "#CCB974",
    "DVL": "#64B5CD",
    "GT": "#000000",
}
NAME_MAPPING = {
    "global": "FL-B",
    "global_isam2": "iS2-B",
    "global_lpi": "FL-LPI",
    "global_tpi": "FL-TPI",
    "global_iekf": "IEKF",
    "global_ukf": "UKF",
    "global_ekf": "EKF",
    "global_tm": "TM",
    "imu": "SBG",
    "dvl": "DVL",
}
