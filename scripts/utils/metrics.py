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

import matplotlib.pyplot as plt
import numpy as np
from evo.core import metrics, sync
from evo.core.trajectory import PoseTrajectory3D
from scipy.spatial.transform import Rotation

from .pipeline import process_bag_offline

logger = logging.getLogger(__name__)

TUM_KEYS = ("time", "x", "y", "z", "qx", "qy", "qz", "qw")


def _evo_agent_dir(bag_path: str, namespace: str) -> Path:
    """
    Return the bag's evo output directory for one agent namespace.

    :param bag_path: Path to the ROS 2 bag directory.
    :param namespace: AUV namespace whose outputs are stored.
    :return: The agent's evo output directory.
    """
    return Path(bag_path) / "evo" / namespace


def _find_ground_truth(bag_path: str, namespace: str) -> Path | None:
    """
    Return the ground truth TUM file exported at the agent's evo root.

    :param bag_path: Path to the ROS 2 bag directory.
    :param namespace: AUV namespace the ground truth was exported under.
    :return: Path to the ground truth TUM file, or None if not found.
    """
    tum_files = sorted(_evo_agent_dir(bag_path, namespace).glob("*.tum"))
    return tum_files[0] if tum_files else None


def _to_trajectory(pose: dict) -> PoseTrajectory3D:
    """
    Convert pose arrays keyed by state name into an evo trajectory.

    :param pose: Arrays keyed by state name, with time and xyzw quaternion.
    :return: The poses as an evo PoseTrajectory3D.
    """
    return PoseTrajectory3D(
        positions_xyz=np.column_stack([pose["x"], pose["y"], pose["z"]]),
        orientations_quat_wxyz=np.column_stack(
            [pose["qw"], pose["qx"], pose["qy"], pose["qz"]]
        ),
        timestamps=pose["time"],
    )


def load_ground_truth(bag_path: str, namespace: str) -> dict:
    """
    Load the exported ground truth trajectory into state arrays.

    :param bag_path: Path to the ROS 2 bag directory.
    :param namespace: AUV namespace the ground truth was exported under.
    :return: Arrays keyed by state name, or an empty dict if unavailable.
    """
    tum_path = _find_ground_truth(bag_path, namespace)
    if tum_path is None:
        agent_dir = _evo_agent_dir(bag_path, namespace)
        logger.warning(f"No ground truth TUM file found in {agent_dir}")
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


def compute_ape_rmse(
    gt: dict | None, est: dict | None, crashed: bool = False, max_diff: float = 0.05
) -> float:
    """
    Compute the aligned translational APE RMSE between two trajectories.

    :param gt: Ground truth arrays keyed by state name.
    :param est: Estimated arrays keyed by state name.
    :param crashed: Whether the pipeline crashed while producing the estimate.
    :param max_diff: Maximum timestamp difference for pose association.
    :return: RMSE in meters, or infinity if the inputs are unusable.
    """
    if not gt or not est or crashed:
        return float("inf")

    try:
        gt_sync, est_sync = sync.associate_trajectories(
            _to_trajectory(gt), _to_trajectory(est), max_diff=max_diff
        )
        est_sync.align(gt_sync)

        ape = metrics.APE(metrics.PoseRelation.translation_part)
        ape.process_data((gt_sync, est_sync))
        return ape.get_statistic(metrics.StatisticsType.rmse)
    except Exception as e:
        logger.warning(f"Could not compute APE RMSE: {e}")
        return float("inf")


def run_evo_evaluations(
    gt_file: str, est_file: str, evo_dir: Path, evo_flags: list[str]
) -> None:
    """
    Run the evo APE and RPE evaluations and save the result archives.

    :param gt_file: Ground truth trajectory in TUM format.
    :param est_file: Estimated trajectory in TUM format.
    :param evo_dir: Directory to save the evo result archives in.
    :param evo_flags: Extra evo flags; plane projections skip rotation runs.
    """
    base_flags = ["--t_max_diff", "0.05", "--no_warnings"]

    plane_flags = [
        f for f in evo_flags if f.startswith("--project") or f in ("xy", "xz", "yz")
    ]
    rotation_flags = [f for f in evo_flags if f not in plane_flags]

    for metric, cmd in [("APE", "evo_ape"), ("RPE", "evo_rpe")]:
        for mode, pose_relation, suffix in [
            ("Translation", "trans_part", "trans"),
            ("Rotation", "angle_deg", "rot"),
        ]:
            flags = evo_flags if suffix == "trans" else rotation_flags
            args = [cmd, "tum", gt_file, est_file, "-r", pose_relation]
            args += base_flags + flags
            args += ["--save_results", str(evo_dir / f"{metric.lower()}_{suffix}.zip")]
            if metric == "RPE":
                args += ["--delta", "1", "--delta_unit", "m", "--all_pairs"]

            result = subprocess.run(args, capture_output=True, text=True)
            if result.stdout:
                logger.info(f"{metric} ({mode}):\n{result.stdout.strip()}")
            if result.stderr:
                logger.error(f"{metric} ({mode}):\n{result.stderr.strip()}")


def process_and_evaluate(
    bag_path: str,
    config_paths: list[str],
    namespace: str,
    evo_dir_name: str,
    evo_flags: list[str],
    urdf_path: str | None = None,
    verbose: bool = True,
) -> tuple[dict | None, dict]:
    """
    Run the offline pipeline on a bag, then save and evaluate results.

    :param bag_path: Path to the ROS 2 bag directory.
    :param config_paths: Parameter YAML files, in increasing priority.
    :param namespace: AUV namespace used for topics and parameters.
    :param evo_dir_name: Subdirectory name for this run's evo outputs.
    :param evo_flags: Extra evo flags passed to the APE and RPE evaluations.
    :param urdf_path: Optional URDF path, resolved from configs if omitted.
    :param verbose: Whether to log progress and show a progress bar.
    :return: Result arrays (or None on failure) and the ground truth arrays.
    """
    logger.info(f"Processing bag: {bag_path}")
    pose_gt = load_ground_truth(bag_path, namespace)
    results, _ = process_bag_offline(
        bag_path, config_paths, namespace, urdf_path, verbose
    )
    if not results:
        return None, pose_gt

    evo_dir = _evo_agent_dir(bag_path, namespace) / evo_dir_name
    evo_dir.mkdir(parents=True, exist_ok=True)

    est_path = evo_dir / f"{namespace}_{evo_dir_name}.tum"
    np.savetxt(est_path, np.column_stack([results[k] for k in TUM_KEYS]), fmt="%.9f")
    logger.info(f"Saved TUM trajectory: {est_path}")

    gt_path = _find_ground_truth(bag_path, namespace)
    if pose_gt and gt_path is not None:
        run_evo_evaluations(str(gt_path), str(est_path), evo_dir, evo_flags)

    return results, pose_gt


def _mask_gaps(t: np.ndarray, vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Insert NaN breaks at large timestamp gaps to avoid bridging in plots.

    :param t: Sample timestamps.
    :param vals: Sample values aligned with the timestamps.
    :return: Timestamps and values with NaN entries inserted at the gaps.
    """
    if len(t) < 2:
        return t, vals
    dts = np.diff(t)
    threshold = max(float(np.median(dts)) * 5.0, 0.5)
    gap_idx = np.where(dts > threshold)[0] + 1
    if len(gap_idx):
        t = np.insert(t, gap_idx, np.nan)
        vals = np.insert(vals, gap_idx, np.nan)
    return t, vals


def plot_results(results: dict, pose_gt: dict, label: str = "") -> None:
    """
    Plot the estimated states against ground truth where available.

    :param results: Estimated arrays keyed by state name.
    :param pose_gt: Ground truth arrays keyed by state name (may be empty).
    :param label: Figure window title, typically the bag name.
    """
    t0 = results["time"][0]
    t_fgo = results["time"] - t0

    layout = [
        (["x", "y", "z"], ["X (m)", "Y (m)", "Z (m)"], pose_gt),
        (["roll", "pitch", "yaw"], ["Roll (rad)", "Pitch (rad)", "Yaw (rad)"], pose_gt),
        (["vx", "vy", "vz"], ["Vx (m/s)", "Vy (m/s)", "Vz (m/s)"], None),
        (
            ["bias_accel_x", "bias_accel_y", "bias_accel_z"],
            ["Accel Bias X", "Accel Bias Y", "Accel Bias Z"],
            None,
        ),
        (
            ["bias_gyro_x", "bias_gyro_y", "bias_gyro_z"],
            ["Gyro Bias X", "Gyro Bias Y", "Gyro Bias Z"],
            None,
        ),
    ]

    _, axes = plt.subplots(len(layout), 3, figsize=(12, 10), num=label or None)
    for row, (keys, labels, gt_data) in enumerate(layout):
        for col, (key, axis_label) in enumerate(zip(keys, labels)):
            ax = axes[row, col]
            if gt_data:
                gt_t, gt_vals = _mask_gaps(gt_data["time"] - t0, gt_data[key])
                ax.plot(gt_t, gt_vals, "-k", label="GT")
            if key in results:
                ax.plot(t_fgo, results[key], "-r", label="FGO")
            ax.set_ylabel(axis_label)
            if row == len(layout) - 1:
                ax.set_xlabel("Time (s)")

    plt.tight_layout()
