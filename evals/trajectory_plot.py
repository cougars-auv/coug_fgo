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

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
import seaborn as sns
from evo.core import lie_algebra as lie
from evo.core import sync
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface, plot

sns.reset_orig()
plt.style.use(["science", "ieee"])

ALGORITHMS = [
    "DVL",
    "SBG",
    "TM",
    # "EKF",
    # "UKF",
    "IEKF",
    "FL-TPI",
    "FL-LPI",
    "iS2-B",
    "FL-B",
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


def load_data(
    evo_agent_dir: Path,
) -> tuple[dict[str, PoseTrajectory3D], PoseTrajectory3D | None]:
    """
    Load the estimated and ground truth trajectories for one agent.

    :param evo_agent_dir: Agent directory inside a bag's evo folder.
    :return: Estimated trajectories keyed by algorithm, and the ground truth.
    """
    gt_files = sorted(evo_agent_dir.glob("*.tum"))
    gt_traj = (
        file_interface.read_tum_trajectory_file(str(gt_files[0])) if gt_files else None
    )

    est_trajs = {}
    for sub in sorted(evo_agent_dir.iterdir()):
        algo = NAME_MAPPING.get(sub.name)
        if (
            sub.is_dir()
            and algo in ALGORITHMS
            and (tum_files := sorted(sub.glob("*.tum")))
        ):
            est_trajs[algo] = file_interface.read_tum_trajectory_file(str(tum_files[0]))

    return est_trajs, gt_traj


def align_to_ref(
    est: PoseTrajectory3D,
    ref: PoseTrajectory3D,
    do_align: bool,
    do_align_origin: bool,
) -> None:
    """
    Align an estimated trajectory to the reference in place.

    :param est: Estimated trajectory to modify.
    :param ref: Reference (ground truth) trajectory.
    :param do_align: Whether to apply an Umeyama alignment.
    :param do_align_origin: Whether to align the first pose to the reference.
    """
    if do_align_origin:
        est.align_origin(ref)
    if not do_align:
        return
    ref_sync, est_sync = sync.associate_trajectories(ref, est, max_diff=0.05)
    if est_sync.num_poses < 2:
        return
    r, t, s = est_sync.align(ref_sync, correct_scale=False)
    est.scale(s)
    est.transform(lie.se3(r, t))


def positions_with_gaps(traj: PoseTrajectory3D) -> np.ndarray:
    """
    Return positions with NaN breaks inserted at large timestamp gaps.

    :param traj: Trajectory to extract positions from.
    :return: Positions with NaN rows inserted where the timestamps jump.
    """
    pos = traj.positions_xyz
    ts = traj.timestamps
    if len(ts) < 2:
        return pos
    dts = np.diff(ts)
    threshold = max(float(np.median(dts)) * 5.0, 0.5)
    gap_idx = np.where(dts > threshold)[0] + 1
    return np.insert(pos, gap_idx, np.nan, axis=0) if len(gap_idx) else pos


def generate_plots(
    evo_agent_dir: Path,
    output_dir: Path,
    auv_name: str,
    do_align: bool = False,
    do_align_origin: bool = False,
) -> None:
    """
    Save a top-down trajectory plot for one agent.

    :param evo_agent_dir: Agent directory inside a bag's evo folder.
    :param output_dir: Directory to save the figure in.
    :param auv_name: Agent name used in the output file name.
    :param do_align: Whether to apply an Umeyama alignment to the estimates.
    :param do_align_origin: Whether to align estimate origins to the truth.
    """
    est_trajs, gt_traj = load_data(evo_agent_dir)

    if not est_trajs and gt_traj is None:
        return

    if gt_traj is not None and (do_align or do_align_origin):
        for algo, traj in est_trajs.items():
            try:
                align_to_ref(traj, gt_traj, do_align, do_align_origin)
            except Exception as e:
                print(f"Could not align {algo}: {e}")

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

    save_path = output_dir / f"{auv_name}_trajectories.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: trajectory_plot.py <target_dir> [--align] [--align_origin]")
        return

    target_dir = Path(sys.argv[1])
    if not target_dir.exists():
        print(f"Error: {target_dir} does not exist.")
        return

    do_align = "--align" in sys.argv[2:]
    do_align_origin = "--align_origin" in sys.argv[2:]

    for evo_dir in target_dir.rglob("evo"):
        bag_dir = evo_dir.parent
        if not (bag_dir / "metadata.yaml").exists():
            continue
        for agent_dir in filter(Path.is_dir, evo_dir.iterdir()):
            generate_plots(
                agent_dir,
                bag_dir,
                agent_dir.name,
                do_align=do_align,
                do_align_origin=do_align_origin,
            )

    print(f"Plots saved to bag directories in {target_dir}")


if __name__ == "__main__":
    main()
