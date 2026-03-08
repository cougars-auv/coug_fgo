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

import glob
import os
from pathlib import Path

from evo.tools import file_interface, plot
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots  # noqa: F401

sns.reset_orig()
plt.style.use(["science", "ieee"])

ALGORITHMS = ["FL-B", "iSAM2-B", "FL-PI", "EKF", "UKF", "IEKF", "DVL"]
COLORS = {
    "FL-B": "#55A868",
    "iSAM2-B": "#DD8452",
    "FL-PI": "#4C72B0",
    "IEKF": "#C44E52",
    "UKF": "#8172B2",
    "EKF": "#937860",
    "DVL": "#DA8BC3",
}
NAME_MAPPING = {
    "global": "FL-B",
    "global_isam2": "iSAM2-B",
    "global_pi": "FL-PI",
    "global_iekf": "IEKF",
    "global_ukf": "UKF",
    "global_ekf": "EKF",
    "dvl": "DVL",
}


def add_start_end_markers(ax, traj, color, start_symbol="o", end_symbol="x", size=15):
    if traj.num_poses == 0:
        return
    start, end = traj.positions_xyz[0], traj.positions_xyz[-1]
    ax.scatter(start[0], start[1], marker=start_symbol, color=color, zorder=10, s=size)
    ax.scatter(end[0], end[1], marker=end_symbol, color=color, zorder=10, s=size)


def load_trajectories(evo_agent_dir):
    est_trajs, gt_traj = {}, None
    zips = glob.glob(
        os.path.join(evo_agent_dir, "**", "*ape_trans*.zip"), recursive=True
    )

    for z in zips:
        algo_label = NAME_MAPPING.get(Path(z).parent.name)
        if algo_label not in ALGORITHMS:
            continue

        try:
            res = file_interface.load_res_file(z, load_trajectories=True)

            ref_key = (
                "reference"
                if "reference" in res.trajectories
                else res.info.get("ref_name", "").split("/")[-1]
            )
            est_key = (
                "estimate"
                if "estimate" in res.trajectories
                else res.info.get("est_name", "").split("/")[-1]
            )

            if gt_traj is None and ref_key in res.trajectories:
                gt_traj = res.trajectories[ref_key]

            if est_key in res.trajectories:
                est_trajs[algo_label] = res.trajectories[est_key]

        except Exception as e:
            print(f"Error loading {z}: {e}")

    return est_trajs, gt_traj


def plot_auv(evo_agent_dir, output_dir, auv_name):
    est_trajs, gt_traj = load_trajectories(evo_agent_dir)

    if not est_trajs and gt_traj is None:
        return

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
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
        plot.traj(
            ax, plot.PlotMode.xy, gt_traj, style="--", color=COLORS["GT"], label="GT"
        )
        add_start_end_markers(ax, gt_traj, COLORS["GT"])

    plt.legend(frameon=True)

    output_path = Path(output_dir) / f"{auv_name}_trajectories.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    bags_root = (
        Path(os.environ.get("HOME", os.path.expanduser("~"))) / "cougars-dev" / "bags"
    )

    if not bags_root.exists():
        print(f"Error: {bags_root} does not exist.")
        return

    for evo_dir in bags_root.rglob("evo"):
        bag_dir = evo_dir.parent
        for agent_dir in evo_dir.iterdir():
            if agent_dir.is_dir():
                plot_auv(str(agent_dir), str(bag_dir), agent_dir.name)

    print(f"Plots saved to 'evo' directories in {bags_root}")


if __name__ == "__main__":
    main()
