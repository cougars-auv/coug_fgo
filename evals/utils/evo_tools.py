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
import subprocess
from pathlib import Path

import numpy as np
from evo.core import lie_algebra as lie
from evo.core import metrics, sync
from evo.core.trajectory import PoseTrajectory3D
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

TUM_KEYS = ("time", "x", "y", "z", "qx", "qy", "qz", "qw")


def _run_logged(
    args: list[str], cwd: Path | None = None, log_stdout: bool = True
) -> subprocess.CompletedProcess:
    """
    Run a subprocess, logging stdout at info and stderr at error on failure (warning otherwise).

    :param args: Command and arguments to execute.
    :param cwd: Working directory to run the command in, if any.
    :param log_stdout: Whether to log captured stdout.
    :return: The completed process.
    """
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    if log_stdout and result.stdout:
        logger.info(result.stdout.strip())
    if result.stderr:
        log = logger.error if result.returncode != 0 else logger.warning
        log(result.stderr.strip())
    return result


def evo_agent_dir(bag_path: str, namespace: str) -> Path:
    """
    Return the bag's evo output directory for one agent namespace.

    :param bag_path: Path to the ROS 2 bag directory.
    :param namespace: AUV namespace whose outputs are stored.
    :return: The agent's evo output directory.
    """
    return Path(bag_path) / "evo" / namespace


def find_ground_truth(bag_path: str, namespace: str) -> Path | None:
    """
    Return the ground truth TUM file exported at the agent's evo root.

    :param bag_path: Path to the ROS 2 bag directory.
    :param namespace: AUV namespace the ground truth was exported under.
    :return: Path to the ground truth TUM file, or None if not found.
    """
    tum_files = sorted(evo_agent_dir(bag_path, namespace).glob("*.tum"))
    return tum_files[0] if tum_files else None


def load_ground_truth(bag_path: str, namespace: str) -> dict:
    """
    Load the ground truth into state arrays, exporting from the bag's truth topic first if needed.

    :param bag_path: Path to the ROS 2 bag directory.
    :param namespace: AUV namespace the ground truth was exported under.
    :return: Arrays keyed by state name, or an empty dict if unavailable.
    """
    tum_path = find_ground_truth(bag_path, namespace)
    if tum_path is None:
        agent_dir = evo_agent_dir(bag_path, namespace)
        logger.info(f"No ground truth TUM found in {agent_dir}; attempting export...")
        truth_topic = f"/{namespace}/odometry/truth"
        tum_path = export_bag_tum(bag_path, truth_topic, agent_dir)
    if tum_path is None:
        logger.warning(f"Could not find or export ground truth for {namespace}")
        return {}
    try:
        data = np.loadtxt(tum_path, comments="#", ndmin=2)
    except (OSError, ValueError):
        logger.warning(f"Could not load ground truth from {tum_path}")
        return {}
    if data.size == 0:
        logger.warning(f"Ground truth file is empty: {tum_path}")
        return {}
    if data.shape[1] < len(TUM_KEYS):
        logger.warning(f"Ground truth file has too few columns: {tum_path}")
        return {}

    pose = {k: data[:, i] for i, k in enumerate(TUM_KEYS)}
    roll, pitch, yaw = Rotation.from_quat(data[:, 4:8]).as_euler("xyz").T
    pose.update({"roll": roll, "pitch": pitch, "yaw": yaw})

    logger.info(f"Loaded ground truth: {tum_path}")
    return pose


def save_tum(path: Path, results: dict) -> None:
    """
    Write trajectory arrays to a TUM-format text file.

    :param path: Destination file path.
    :param results: Arrays keyed by state name (must contain TUM_KEYS).
    """
    np.savetxt(
        path,
        np.column_stack([results[k] for k in TUM_KEYS]),
        fmt="%.9f",
    )
    logger.info(f"Saved TUM trajectory: {path}")


def export_bag_tum(bag_path: str, topic: str, out_dir: Path) -> Path | None:
    """
    Export a recorded trajectory topic from a bag to a TUM file with evo.

    :param bag_path: Path to the ROS 2 bag directory.
    :param topic: Trajectory topic to export (e.g. an odometry topic).
    :param out_dir: Directory to write the exported TUM file into.
    :return: Path to the exported TUM file, or None if the export failed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    args = ["evo_traj", "bag2", str(Path(bag_path).resolve()), topic, "--save_as_tum"]
    if _run_logged(args, cwd=out_dir).returncode != 0:
        return None

    tum_files = sorted(out_dir.glob("*.tum"), key=lambda p: p.stat().st_mtime)
    return tum_files[-1] if tum_files else None


def run_evo_evaluations(
    gt_file: str, est_file: str, evo_dir: Path, evo_flags: list[str]
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
            args = [cmd, "tum", gt_file, est_file, "-r", pose_relation]
            args += base_flags + evo_flags
            args += ["--save_results", str(evo_dir / f"{metric.lower()}_{suffix}.zip")]
            if metric == "RPE":
                args += ["--delta", "1", "--delta_unit", "m", "--all_pairs"]

            _run_logged(args)


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

    def to_trajectory(pose: dict) -> PoseTrajectory3D:
        """Convert pose arrays into an evo PoseTrajectory3D."""
        return PoseTrajectory3D(
            positions_xyz=np.column_stack([pose["x"], pose["y"], pose["z"]]),
            orientations_quat_wxyz=np.column_stack(
                [pose["qw"], pose["qx"], pose["qy"], pose["qz"]]
            ),
            timestamps=pose["time"],
        )

    try:
        gt_sync, est_sync = sync.associate_trajectories(
            to_trajectory(gt), to_trajectory(est), max_diff=max_diff
        )
        est_sync.align(gt_sync)

        ape = metrics.APE(metrics.PoseRelation.translation_part)
        ape.process_data((gt_sync, est_sync))
        return ape.get_statistic(metrics.StatisticsType.rmse)
    except Exception as e:
        logger.warning(f"Could not compute APE RMSE: {e}")
        return float("inf")


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
        _run_logged(args, log_stdout=False)
