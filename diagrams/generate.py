#!/usr/bin/env python3
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

import daft
import copy
from pathlib import Path
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).parent

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"

COLOR_VAR = "#B5CBE6"
COLOR_PRIOR = "#4C72B0"
COLOR_FACTOR_DEPTH = "#DD8452"
COLOR_FACTOR_HEADING = "#C44E52"
COLOR_FACTOR_GPS = "#55A868"
COLOR_FACTOR_DVL = "#8172B2"
COLOR_FACTOR_IMU = "#000000"
COLOR_FACTOR_DYNAMICS = "#DA8BC3"

style_var = {"facecolor": COLOR_VAR, "edgecolor": "black"}
style_prior = {"facecolor": COLOR_PRIOR, "edgecolor": "black"}
style_factor_depth = {"facecolor": COLOR_FACTOR_DEPTH, "edgecolor": "black"}
style_factor_heading = {"facecolor": COLOR_FACTOR_HEADING, "edgecolor": "black"}
style_factor_gps = {"facecolor": COLOR_FACTOR_GPS, "edgecolor": "black"}
style_factor_dvl = {"facecolor": COLOR_FACTOR_DVL, "edgecolor": "black"}
style_factor_imu = {"facecolor": COLOR_FACTOR_IMU, "edgecolor": "black"}
style_factor_dynamics = {"facecolor": COLOR_FACTOR_DYNAMICS, "edgecolor": "black"}

col_spacing = 1.5
start_x = 2.0
prior_dist = col_spacing / 2

win_pad = col_spacing / 2
win_left = (start_x + col_spacing) - win_pad
win_width = (3 * col_spacing) + (2 * win_pad)
win_bottom = 1 - win_pad

# =============================================================================
# BASE FACTOR GRAPH
# =============================================================================

pgm = daft.PGM(directed=False)

for sym, row in [("x", 3), ("v", 2), ("b", 1)]:
    pgm.add_node(
        f"p{sym}",
        f"$p_\\mathbf{{{sym}}}$",
        start_x - prior_dist,
        row,
        fixed=True,
        offset=[0, 3],
        plot_params=style_prior,
    )

for i in range(5):
    col_x = start_x + (i * col_spacing)
    pgm.add_node(f"x{i}", f"$\\mathbf{{x}}_{{{i}}}$", col_x, 3, plot_params=style_var)
    pgm.add_node(f"v{i}", f"$\\mathbf{{v}}_{{{i}}}$", col_x, 2, plot_params=style_var)
    pgm.add_node(f"b{i}", f"$\\mathbf{{b}}_{{{i}}}$", col_x, 1, plot_params=style_var)

for sym in ["x", "v", "b"]:
    pgm.add_edge(f"p{sym}", f"{sym}0")

for i in range(5):
    col_x = start_x + (i * col_spacing)

    pgm.add_node(
        f"depth{i}",
        f"$z_{i}$",
        col_x - 0.4,
        3.4,
        fixed=True,
        plot_params=style_factor_depth,
        offset=[0, 3],
    )
    pgm.add_edge(f"x{i}", f"depth{i}")

    pgm.add_node(
        f"heading{i}",
        f"$\\psi_{i}$",
        col_x + 0.4,
        3.4,
        fixed=True,
        plot_params=style_factor_heading,
        offset=[0, 3],
    )
    pgm.add_edge(f"x{i}", f"heading{i}")

for i in range(4):
    col_x = start_x + (i * col_spacing)
    mid_x = col_x + (col_spacing / 2)

    pgm.add_node(
        f"imu{i}",
        f"$\\mathcal{{I}}_{{{i}{i + 1}}}$",
        mid_x,
        2,
        fixed=True,
        offset=[0, -25],
        plot_params=style_factor_imu,
    )
    pgm.add_edge(f"x{i}", f"imu{i}")
    pgm.add_edge(f"v{i}", f"imu{i}")
    pgm.add_edge(f"b{i}", f"imu{i}")
    pgm.add_edge(f"imu{i}", f"x{i + 1}")
    pgm.add_edge(f"imu{i}", f"v{i + 1}")
    pgm.add_edge(f"imu{i}", f"b{i + 1}")

gps_x = start_x + (3 * col_spacing)
pgm.add_node(
    "gps",
    "$xy_3$",
    gps_x,
    3.6,
    fixed=True,
    offset=[0, 3],
    plot_params=style_factor_gps,
)
pgm.add_edge("x3", "gps")

# Shade the sliding window
win_top = 3.6 + win_pad - 0.2
pgm.add_plate(
    [win_left, win_bottom, win_width, win_top - win_bottom],
    label="sliding window",
    position="bottom right",
    rect_params={"fc": "#E8E8E8", "ec": "none"},
    fontsize=10,
)

pgm_preint_loose = copy.deepcopy(pgm)
pgm_preint_tight = copy.deepcopy(pgm)
pgm_dynamics = copy.deepcopy(pgm)
pgm_const_vel = copy.deepcopy(pgm)

# =============================================================================
# BINARY DVL GRAPH
# =============================================================================

for i in range(5):
    col_x = start_x + (i * col_spacing)
    pgm.add_node(
        f"dvl{i}",
        f"${{v}}_{{{i}}}$",
        col_x,
        2.5,
        fixed=True,
        offset=[-10, -8],
        plot_params=style_factor_dvl,
    )
    pgm.add_edge(f"x{i}", f"dvl{i}")
    pgm.add_edge(f"v{i}", f"dvl{i}")

pgm.render()
pgm.figure.savefig(OUTPUT_DIR / "fgo_dvl_binary.pdf", bbox_inches="tight")
pgm.figure.savefig(OUTPUT_DIR / "fgo_dvl_binary.png", bbox_inches="tight", dpi=300)

# =============================================================================
# LOOSE PREINTEGRATED DVL GRAPH
# =============================================================================

for i in range(4):
    col_x = start_x + (i * col_spacing)
    mid_x = col_x + (col_spacing / 2)

    pgm_preint_loose.add_node(
        f"dvl_preint{i}",
        f"$\\mathcal{{D}}_{{{i}{i + 1}}}$",
        mid_x,
        3,
        fixed=True,
        plot_params=style_factor_dvl,
        offset=[0, 3],
    )
    pgm_preint_loose.add_edge(f"x{i}", f"dvl_preint{i}")
    pgm_preint_loose.add_edge(f"dvl_preint{i}", f"x{i + 1}")

pgm_preint_loose.render()
pgm_preint_loose.figure.savefig(
    OUTPUT_DIR / "fgo_dvl_preint_loose.pdf", bbox_inches="tight"
)
pgm_preint_loose.figure.savefig(
    OUTPUT_DIR / "fgo_dvl_preint_loose.png", bbox_inches="tight", dpi=300
)

# =============================================================================
# TIGHT PREINTEGRATED DVL GRAPH
# =============================================================================

for i in range(4):
    col_x = start_x + (i * col_spacing)
    mid_x = col_x + (col_spacing / 2)

    pgm_preint_tight.add_node(
        f"dvl_preint{i}",
        f"$\\mathcal{{D}}_{{{i}{i + 1}}}$",
        mid_x,
        2.5,
        fixed=True,
        plot_params=style_factor_dvl,
        offset=[0, 5],
    )
    pgm_preint_tight.add_edge(f"x{i}", f"dvl_preint{i}")
    pgm_preint_tight.add_edge(f"dvl_preint{i}", f"x{i + 1}")
    pgm_preint_tight.add_edge(f"b{i}", f"dvl_preint{i}")

pgm_preint_tight.render()
pgm_preint_tight.figure.savefig(
    OUTPUT_DIR / "fgo_dvl_preint_tight.pdf", bbox_inches="tight"
)
pgm_preint_tight.figure.savefig(
    OUTPUT_DIR / "fgo_dvl_preint_tight.png", bbox_inches="tight", dpi=300
)

# =============================================================================
# DYNAMICS GRAPH
# =============================================================================

for i in [0, 1, 4]:
    col_x = start_x + (i * col_spacing)
    pgm_dynamics.add_node(
        f"dvl{i}",
        f"${{v}}_{{{i}}}$",
        col_x,
        2.5,
        fixed=True,
        offset=[-10, -8],
        plot_params=style_factor_dvl,
    )
    pgm_dynamics.add_edge(f"x{i}", f"dvl{i}")
    pgm_dynamics.add_edge(f"v{i}", f"dvl{i}")

for i in range(4):
    col_x = start_x + (i * col_spacing)
    mid_x = col_x + (col_spacing / 2)

    pgm_dynamics.add_node(
        f"dynamics{i}",
        f"$\\mathcal{{M}}_{{{i}{i + 1}}}$",
        mid_x,
        2.5,
        fixed=True,
        plot_params=style_factor_dynamics,
        offset=[0, 5],
    )
    pgm_dynamics.add_edge(f"x{i}", f"dynamics{i}")
    pgm_dynamics.add_edge(f"dynamics{i}", f"x{i + 1}")
    pgm_dynamics.add_edge(f"v{i}", f"dynamics{i}")
    pgm_dynamics.add_edge(f"v{i + 1}", f"dynamics{i}")

pgm_dynamics.render()
pgm_dynamics.figure.savefig(OUTPUT_DIR / "fgo_dynamics.pdf", bbox_inches="tight")
pgm_dynamics.figure.savefig(
    OUTPUT_DIR / "fgo_dynamics.png", bbox_inches="tight", dpi=300
)

# =============================================================================
# CONSTANT VELOCITY GRAPH
# =============================================================================

for i in [0, 1, 4]:
    col_x = start_x + (i * col_spacing)
    pgm_const_vel.add_node(
        f"dvl{i}",
        f"${{v}}_{{{i}}}$",
        col_x,
        2.5,
        fixed=True,
        offset=[-10, -8],
        plot_params=style_factor_dvl,
    )
    pgm_const_vel.add_edge(f"x{i}", f"dvl{i}")
    pgm_const_vel.add_edge(f"v{i}", f"dvl{i}")

for i in range(1, 3):
    col_x = start_x + (i * col_spacing)
    mid_x = col_x + (col_spacing / 2)

    pgm_const_vel.add_node(
        f"const_vel{i}",
        f"${{\\dot{{v}}}}_{{{i}{i + 1}}}$",
        mid_x,
        2.5,
        fixed=True,
        plot_params=style_factor_dynamics,
        offset=[0, 3],
    )
    pgm_const_vel.add_edge(f"v{i}", f"const_vel{i}")
    pgm_const_vel.add_edge(f"v{i + 1}", f"const_vel{i}")

pgm_const_vel.render()
pgm_const_vel.figure.savefig(OUTPUT_DIR / "fgo_const_vel.pdf", bbox_inches="tight")
pgm_const_vel.figure.savefig(
    OUTPUT_DIR / "fgo_const_vel.png", bbox_inches="tight", dpi=300
)

# =============================================================================
# MULTIAGENT GRAPH
# =============================================================================

# Lead agent factor graph
pgm_multiagent = daft.PGM(directed=False)

for sym, row in [("x", 3), ("v", 2), ("b", 1)]:
    pgm_multiagent.add_node(
        f"p{sym}",
        f"$p^0_\\mathbf{{{sym}}}$",
        start_x - prior_dist,
        row,
        fixed=True,
        offset=[0, 3],
        plot_params=style_prior,
    )

for i in range(5):
    col_x = start_x + (i * col_spacing)

    pgm_multiagent.add_node(
        f"x{i}", f"$\\mathbf{{x}}^0_{{{i}}}$", col_x, 3, plot_params=style_var
    )
    pgm_multiagent.add_node(
        f"v{i}", f"$\\mathbf{{v}}^0_{{{i}}}$", col_x, 2, plot_params=style_var
    )
    pgm_multiagent.add_node(
        f"b{i}", f"$\\mathbf{{b}}^0_{{{i}}}$", col_x, 1, plot_params=style_var
    )

for sym in ["x", "v", "b"]:
    pgm_multiagent.add_edge(f"p{sym}", f"{sym}0")

for i in range(5):
    col_x = start_x + (i * col_spacing)

    pgm_multiagent.add_node(
        f"depth{i}",
        f"$z^0_{i}$",
        col_x - 0.4,
        3.4,
        fixed=True,
        plot_params=style_factor_depth,
        offset=[0, 3],
    )
    pgm_multiagent.add_edge(f"x{i}", f"depth{i}")

    pgm_multiagent.add_node(
        f"heading{i}",
        f"$\\psi^0_{i}$",
        col_x + 0.4,
        3.4,
        fixed=True,
        plot_params=style_factor_heading,
        offset=[0, 3],
    )
    pgm_multiagent.add_edge(f"x{i}", f"heading{i}")

for i in range(4):
    col_x = start_x + (i * col_spacing)
    mid_x = col_x + (col_spacing / 2)

    pgm_multiagent.add_node(
        f"imu{i}",
        f"$\\mathcal{{I}}^0_{{{i}{i + 1}}}$",
        mid_x,
        2,
        fixed=True,
        offset=[0, -25],
        plot_params=style_factor_imu,
    )
    pgm_multiagent.add_edge(f"x{i}", f"imu{i}")
    pgm_multiagent.add_edge(f"v{i}", f"imu{i}")
    pgm_multiagent.add_edge(f"b{i}", f"imu{i}")
    pgm_multiagent.add_edge(f"imu{i}", f"x{i + 1}")
    pgm_multiagent.add_edge(f"imu{i}", f"v{i + 1}")
    pgm_multiagent.add_edge(f"imu{i}", f"b{i + 1}")

gps_x = start_x + (3 * col_spacing)
pgm_multiagent.add_node(
    "gps",
    "$xy^0_3$",
    gps_x,
    3.6,
    fixed=True,
    offset=[0, 3],
    plot_params=style_factor_gps,
)
pgm_multiagent.add_edge("x3", "gps")

for i in range(5):
    col_x = start_x + (i * col_spacing)
    pgm_multiagent.add_node(
        f"dvl{i}",
        f"${{v}}^0_{{{i}}}$",
        col_x,
        2.5,
        fixed=True,
        offset=[-10, -8],
        plot_params=style_factor_dvl,
    )
    pgm_multiagent.add_edge(f"x{i}", f"dvl{i}")
    pgm_multiagent.add_edge(f"v{i}", f"dvl{i}")

# Second agent factor graph
pgm_multiagent.add_node(
    "px1",
    "$p^1_\\mathbf{{x}}$",
    start_x - prior_dist,
    5.7,
    fixed=True,
    offset=[0, 3],
    plot_params=style_prior,
)

for i in [0, 2, 4]:
    col_x = start_x + (i * col_spacing)
    pgm_multiagent.add_node(
        f"x1_{i}", f"$\\mathbf{{x}}^1_{{{i}}}$", col_x, 5.7, plot_params=style_var
    )

pgm_multiagent.add_edge("px1", "x1_0")

for i in [0, 2, 4]:
    col_x = start_x + (i * col_spacing)

    pgm_multiagent.add_node(
        f"depth1_{i}",
        f"$z^1_{i}$",
        col_x - 0.4,
        6.1,
        fixed=True,
        plot_params=style_factor_depth,
        offset=[0, 3],
    )
    pgm_multiagent.add_edge(f"x1_{i}", f"depth1_{i}")

    pgm_multiagent.add_node(
        f"heading1_{i}",
        f"$\\psi^1_{i}$",
        col_x + 0.4,
        6.1,
        fixed=True,
        plot_params=style_factor_heading,
        offset=[0, 3],
    )
    pgm_multiagent.add_edge(f"x1_{i}", f"heading1_{i}")

for a, b in [(0, 2), (2, 4)]:
    mid_x = start_x + ((a + b) / 2) * col_spacing
    pgm_multiagent.add_node(
        f"odom1_{a}{b}",
        f"$u^1_{{{a}{b}}}$",
        mid_x,
        5.7,
        fixed=True,
        offset=[0, -20],
        plot_params=style_factor_imu,
    )
    pgm_multiagent.add_edge(f"x1_{a}", f"odom1_{a}{b}")
    pgm_multiagent.add_edge(f"odom1_{a}{b}", f"x1_{b}")

# Anchor nodes
for agent, row in [(1, 5), (0, 4)]:
    pgm_multiagent.add_node(
        f"panchor{agent}",
        f"$p^{agent}_\\mathbf{{\\Delta}}$",
        start_x - prior_dist,
        row,
        fixed=True,
        offset=[0, 3],
        plot_params=style_prior,
    )
    pgm_multiagent.add_node(
        f"anchor{agent}",
        f"$\\mathbf{{\\Delta}}^{agent}$",
        start_x,
        row,
        plot_params=style_var,
    )
    pgm_multiagent.add_edge(f"panchor{agent}", f"anchor{agent}")

# Inter-agent constraint factors
for i in [2, 4]:
    pgm_multiagent.add_node(
        f"constraint{i}",
        f"$c_{i}$",
        start_x + i * col_spacing,
        4.5,
        fixed=True,
        offset=[13, -8],
        plot_params=style_factor_dynamics,
    )
    pgm_multiagent.add_edge("anchor0", f"constraint{i}")
    pgm_multiagent.add_edge("anchor1", f"constraint{i}")
    pgm_multiagent.add_edge(f"x{i}", f"constraint{i}")
    pgm_multiagent.add_edge(f"x1_{i}", f"constraint{i}")

pgm_multiagent.render()
pgm_multiagent.figure.savefig(OUTPUT_DIR / "fgo_multiagent.pdf", bbox_inches="tight")
pgm_multiagent.figure.savefig(
    OUTPUT_DIR / "fgo_multiagent.png", bbox_inches="tight", dpi=300
)
