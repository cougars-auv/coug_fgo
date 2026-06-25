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

import numpy as np
from evo.core import lie_algebra as lie
from evo.core import sync
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface, plot
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots  # noqa: F401

sns.reset_orig()
plt.style.use(["science", "ieee"])

ALGORITHMS = [
    "DVL",
    "AQS",
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
    "AQS": "#CCB974",
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
    "global_aqs": "AQS",
    "dvl": "DVL",
}


def add_start_end_markers(
    ax: plt.Axes,
    traj: PoseTrajectory3D,
    color: str,
    start_symbol: str = "o",
    end_symbol: str = "x",
    size: int = 15,
    y_idx: int = 1,
) -> None:
    if traj.num_poses == 0:
        return
    start, end = traj.positions_xyz[0], traj.positions_xyz[-1]
    ax.scatter(
        start[0], start[y_idx], marker=start_symbol, color=color, zorder=10, s=size
    )
    ax.scatter(end[0], end[y_idx], marker=end_symbol, color=color, zorder=10, s=size)


def load_trajectories(
    evo_agent_dir: str,
) -> tuple[dict[str, PoseTrajectory3D], PoseTrajectory3D | None]:
    est_trajs, gt_traj = {}, None
    odom_dir = Path(evo_agent_dir) / "odometry"
    if not odom_dir.is_dir():
        return est_trajs, gt_traj

    gt_files = sorted(odom_dir.glob("*.tum"))
    if gt_files:
        gt_traj = file_interface.read_tum_trajectory_file(str(gt_files[0]))

    for sub in sorted(odom_dir.iterdir()):
        algo_label = NAME_MAPPING.get(sub.name)
        if not sub.is_dir() or algo_label not in ALGORITHMS:
            continue
        tum_files = sorted(sub.glob("*.tum"))
        if tum_files:
            est_trajs[algo_label] = file_interface.read_tum_trajectory_file(
                str(tum_files[0])
            )

    return est_trajs, gt_traj


def align_to_ref(
    est: PoseTrajectory3D,
    ref: PoseTrajectory3D,
    do_align: bool,
    do_align_origin: bool,
) -> None:
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
    pos = traj.positions_xyz
    ts = traj.timestamps
    if len(ts) < 2:
        return pos
    dts = np.diff(ts)
    threshold = max(float(np.median(dts)) * 5.0, 0.5)
    gap_idx = np.where(dts > threshold)[0] + 1
    return np.insert(pos, gap_idx, np.nan, axis=0) if len(gap_idx) else pos


def plot_auv(
    evo_agent_dir: str,
    output_dir: str,
    auv_name: str,
    do_align: bool = False,
    do_align_origin: bool = False,
) -> None:
    est_trajs, gt_traj = load_trajectories(evo_agent_dir)

    if not est_trajs and gt_traj is None:
        return

    if gt_traj is not None and (do_align or do_align_origin):
        for algo, traj in est_trajs.items():
            try:
                align_to_ref(traj, gt_traj, do_align, do_align_origin)
            except Exception as e:
                print(f"Could not align {algo}: {e}")

    is_xz = auv_name in ("aquaslam", "aquaslam_wt")
    plot_mode = plot.PlotMode.xz if is_xz else plot.PlotMode.xy
    y_idx = 2 if is_xz else 1
    y_label = "$z$ (m)" if is_xz else "$y$ (m)"

    fig, ax = plt.subplots()
    ax.set_aspect("equal", adjustable="datalim")
    ax.set(xlabel="$x$ (m)", ylabel=y_label, title="")

    for algo in ALGORITHMS:
        if algo in est_trajs:
            plot.traj(
                ax,
                plot_mode,
                est_trajs[algo],
                style="-",
                color=COLORS[algo],
                label=algo,
            )
            add_start_end_markers(ax, est_trajs[algo], COLORS[algo], y_idx=y_idx)

    if gt_traj:
        gt_pos = positions_with_gaps(gt_traj)
        ax.plot(gt_pos[:, 0], gt_pos[:, y_idx], "--", color=COLORS["GT"], label="GT")
        add_start_end_markers(ax, gt_traj, COLORS["GT"], y_idx=y_idx)

    legend = plt.legend(frameon=True)
    legend.set_zorder(100)

    save_path = Path(output_dir) / f"{auv_name}_trajectories.png"
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
        for agent_dir in evo_dir.iterdir():
            if agent_dir.is_dir():
                plot_auv(
                    str(agent_dir),
                    str(bag_dir),
                    agent_dir.name,
                    do_align=do_align,
                    do_align_origin=do_align_origin,
                )

    print(f"Plots saved to bag directories in {target_dir}")


if __name__ == "__main__":
    main()
