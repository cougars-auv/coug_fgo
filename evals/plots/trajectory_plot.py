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

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
import seaborn as sns
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface, plot

from utils import estimators, evo_tools, plotting

logger = logging.getLogger(__name__)

sns.reset_orig()
plt.style.use(["science", "ieee"])

COLORS = estimators.color_map()
ALGORITHMS = estimators.labels()[::-1]


def add_start_end_markers(
    ax: plt.Axes,
    traj: PoseTrajectory3D,
    color: str,
    start_symbol: str = "o",
    end_symbol: str = "x",
    size: int = 15,
) -> None:
    """
    Mark the start and end points of a trajectory on the given axes.

    :param ax: Axes to draw on.
    :param traj: Trajectory whose endpoints are marked.
    :param color: Marker color.
    :param start_symbol: Marker symbol for the start point.
    :param end_symbol: Marker symbol for the end point.
    :param size: Marker size.
    """
    if traj.num_poses > 0:
        start, end = traj.positions_xyz[0], traj.positions_xyz[-1]
        ax.scatter(
            start[0], start[1], marker=start_symbol, color=color, zorder=10, s=size
        )
        ax.scatter(end[0], end[1], marker=end_symbol, color=color, zorder=10, s=size)


def load_trajectories(
    evo_agent_dir: Path,
) -> tuple[dict[str, PoseTrajectory3D], PoseTrajectory3D | None]:
    """
    Load the estimated and ground truth trajectories for one agent.

    :param evo_agent_dir: Agent directory inside a bag's evo folder.
    :return: Estimated trajectories keyed by algorithm, and the ground truth.
    """
    gt_file = evo_tools.latest_tum(evo_agent_dir)
    gt_traj = file_interface.read_tum_trajectory_file(str(gt_file)) if gt_file else None

    est_trajs = {}
    for sub in sorted(evo_agent_dir.iterdir()):
        algo = estimators.label_for_folder(sub.name)
        if sub.is_dir() and algo in ALGORITHMS and (tum := evo_tools.latest_tum(sub)):
            est_trajs[algo] = file_interface.read_tum_trajectory_file(str(tum))

    return est_trajs, gt_traj


def positions_with_gaps(traj: PoseTrajectory3D) -> np.ndarray:
    """
    Return positions with NaN breaks inserted at large timestamp gaps.

    :param traj: Trajectory to extract positions from.
    :return: Positions with NaN rows inserted where the timestamps jump.
    """
    pos = traj.positions_xyz
    gap_idx = plotting.gap_indices(traj.timestamps)
    return np.insert(pos, gap_idx, np.nan, axis=0) if len(gap_idx) else pos


def render(target_dir: Path, do_align: bool = False) -> None:
    """
    Save a top-down trajectory plot for every evaluated agent under a directory.

    :param target_dir: A bag or directory of bags that has been evaluated.
    :param do_align: Whether to Umeyama-align estimates to the ground truth.
    """
    for bag_dir, agent_dir in evo_tools.iter_evaluated_agents(target_dir):
        est_trajs, gt_traj = load_trajectories(agent_dir)

        if not est_trajs and gt_traj is None:
            continue

        if gt_traj is not None and do_align:
            for algo, traj in est_trajs.items():
                try:
                    evo_tools.umeyama_align(traj, gt_traj)
                except Exception as e:
                    logger.error(f"Could not align {algo}: {e}")

        fig, ax = plt.subplots()
        ax.set_aspect("equal", adjustable="datalim")
        ax.set(xlabel="$x$ (m)", ylabel="$y$ (m)", title="")

        for algo in ALGORITHMS:
            if algo in est_trajs:
                plot.traj(
                    ax,
                    plot.PlotMode.xy,
                    est_trajs[algo],
                    style="-",
                    color=COLORS[algo],
                    label=algo,
                )
                add_start_end_markers(ax, est_trajs[algo], COLORS[algo])

        if gt_traj:
            gt_pos = positions_with_gaps(gt_traj)
            ax.plot(gt_pos[:, 0], gt_pos[:, 1], "--", color=COLORS["GT"], label="GT")
            add_start_end_markers(ax, gt_traj, COLORS["GT"])

        legend = plt.legend(frameon=True)
        legend.set_zorder(100)

        fig.savefig(
            bag_dir / f"{agent_dir.name}_trajectories.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)
