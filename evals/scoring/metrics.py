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

import numpy as np
from evo.core import lie_algebra as lie
from evo.core import metrics, sync
from evo.core.trajectory import PoseTrajectory3D
from scipy.spatial.transform import Rotation

from scoring.tum import run_logged

logger = logging.getLogger(__name__)


def run_evo_evaluations(
    gt_file: str | Path, est_file: str | Path, evo_dir: Path, evo_flags: list[str]
) -> None:
    """
    Run the evo APE and RPE evaluations and save the result archives.

    :param gt_file: Ground truth trajectory in TUM format.
    :param est_file: Estimated trajectory in TUM format.
    :param evo_dir: Directory to save the evo result archives in.
    :param evo_flags: Extra evo flags forwarded to every APE and RPE run.
    """
    base_flags = ["--t_max_diff", "0.05", "--no_warnings"]

    for metric, cmd in [("APE", "evo_ape"), ("RPE", "evo_rpe")]:
        for pose_relation, suffix in [("trans_part", "trans"), ("angle_deg", "rot")]:
            args = [cmd, "tum", str(gt_file), str(est_file), "-r", pose_relation]
            args += base_flags + evo_flags
            args += ["--save_results", str(evo_dir / f"{metric.lower()}_{suffix}.zip")]
            if metric == "RPE":
                args += ["--delta", "1", "--delta_unit", "m", "--all_pairs"]

            run_logged(args)


def compute_ape_rmse(
    gt: dict | None, est: dict | None, crashed: bool = False, max_diff: float = 0.05
) -> float:
    """
    Compute the aligned translational APE RMSE between two trajectories.

    :param gt: Ground truth arrays keyed by state name.
    :param est: Estimated arrays keyed by state name.
    :param crashed: Whether the factor graph crashed while producing the estimate.
    :param max_diff: Maximum timestamp difference for pose association.
    :return: RMSE in meters, or infinity if the inputs are unusable.
    """
    if not gt or not est or crashed:
        return float("inf")

    try:
        gt_sync, est_sync = sync.associate_trajectories(
            dict_to_trajectory(gt), dict_to_trajectory(est), max_diff=max_diff
        )
        est_sync.align(gt_sync)

        ape = metrics.APE(metrics.PoseRelation.translation_part)
        ape.process_data((gt_sync, est_sync))
        return ape.get_statistic(metrics.StatisticsType.rmse)
    except Exception as e:
        logger.error(f"Could not compute APE RMSE: {e}")
        return float("inf")


def dict_to_trajectory(pose: dict) -> PoseTrajectory3D:
    """
    Convert pose arrays into an evo PoseTrajectory3D.

    :param pose: Dictionary of pose arrays keyed by state name.
    :return: An evo PoseTrajectory3D object.
    """
    return PoseTrajectory3D(
        positions_xyz=np.column_stack([pose["x"], pose["y"], pose["z"]]),
        orientations_quat_wxyz=np.column_stack(
            [pose["qw"], pose["qx"], pose["qy"], pose["qz"]]
        ),
        timestamps=pose["time"],
    )


def umeyama_align(est: PoseTrajectory3D, ref: PoseTrajectory3D) -> None:
    """
    Umeyama-align an estimated trajectory to the reference in place.

    :param est: Estimated trajectory to modify.
    :param ref: Reference (ground truth) trajectory.
    """
    ref_sync, est_sync = sync.associate_trajectories(ref, est, max_diff=0.05)
    if est_sync.num_poses < 2:
        return
    r, t, s = est_sync.align(ref_sync, correct_scale=False)
    est.scale(s)
    est.transform(lie.se3(r, t))


def align_dicts(est: dict, ref: dict) -> None:
    """
    Umeyama-align the estimated state dictionary to the reference in place.

    :param est: Estimated arrays keyed by state name.
    :param ref: Reference ground truth arrays keyed by state name.
    """
    est_traj = dict_to_trajectory(est)
    umeyama_align(est_traj, dict_to_trajectory(ref))

    for i, key in enumerate(("x", "y", "z")):
        est[key] = est_traj.positions_xyz[:, i]
    for i, key in enumerate(("qw", "qx", "qy", "qz")):
        est[key] = est_traj.orientations_quat_wxyz[:, i]

    quats_xyzw = est_traj.orientations_quat_wxyz[:, [1, 2, 3, 0]]
    est["roll"], est["pitch"], est["yaw"] = (
        Rotation.from_quat(quats_xyzw).as_euler("xyz").T
    )


def build_benchmark_tables(agent_dir: Path, metrics_names: tuple[str, ...]) -> None:
    """
    Aggregate an agent's evo result archives into per-metric benchmark tables.

    :param agent_dir: The agent's evo output directory holding the result zips.
    :param metrics_names: Metric names to tabulate (e.g. ``ape_trans``).
    """
    if not any(agent_dir.glob("*/*.zip")):
        return

    for old_table in agent_dir.glob("benchmark_*.csv"):
        old_table.unlink()

    for metric in metrics_names:
        metric_zips = sorted(agent_dir.glob(f"*/{metric}.zip"))
        if not metric_zips:
            continue
        args = ["evo_res", *map(str, metric_zips)]
        args += ["--save_table", str(agent_dir / f"benchmark_{metric}.csv")]
        run_logged(args)
