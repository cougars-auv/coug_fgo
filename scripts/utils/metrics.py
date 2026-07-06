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


def _odometry_dir(bag_path: str, namespace: str) -> Path:
    return Path(bag_path) / "evo" / namespace / "odometry"


def _to_trajectory(pose: dict) -> PoseTrajectory3D:
    return PoseTrajectory3D(
        positions_xyz=np.column_stack([pose["x"], pose["y"], pose["z"]]),
        orientations_quat_wxyz=np.column_stack(
            [pose["qw"], pose["qx"], pose["qy"], pose["qz"]]
        ),
        timestamps=pose["time"],
    )


def load_ground_truth(bag_path: str, namespace: str) -> dict:
    tum_path = _odometry_dir(bag_path, namespace) / f"{namespace}_odometry_truth.tum"
    try:
        data = np.loadtxt(tum_path, comments="#", ndmin=2)
    except (OSError, ValueError):
        logger.warning(f"Could not load GT from {tum_path}")
        return {}
    if data.size == 0:
        logger.warning(f"GT file is empty: {tum_path}")
        return {}

    pose = {k: data[:, i] for i, k in enumerate(TUM_KEYS)}
    roll, pitch, yaw = Rotation.from_quat(data[:, 4:8]).as_euler("xyz").T
    pose.update({"roll": roll, "pitch": pitch, "yaw": yaw})

    logger.info(f"Loaded GT: {tum_path}")
    return pose


def compute_ape_rmse(
    gt: dict | None, est: dict | None, crashed: bool = False, max_diff: float = 0.05
) -> float:
    if not gt or not est or crashed:
        return float("inf")

    gt_sync, est_sync = sync.associate_trajectories(
        _to_trajectory(gt), _to_trajectory(est), max_diff=max_diff
    )
    est_sync.align(gt_sync)

    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data((gt_sync, est_sync))
    return ape.get_statistic(metrics.StatisticsType.rmse)


def run_evo_evaluations(
    gt_file: str, est_file: str, evo_dir: Path, evo_flags: list[str]
) -> None:
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


def evaluate_and_save(
    bag_path: str,
    config_paths: list[str],
    namespace: str,
    evo_dir_name: str,
    evo_flags: list[str],
    urdf_path: str | None = None,
    verbose: bool = True,
) -> tuple[dict | None, dict]:
    logger.info(f"Processing bag: {bag_path}")
    pose_gt = load_ground_truth(bag_path, namespace)
    results, _ = process_bag_offline(
        bag_path, config_paths, namespace, urdf_path, verbose
    )
    if not results:
        return None, pose_gt

    odom_dir = _odometry_dir(bag_path, namespace)
    evo_dir = odom_dir / evo_dir_name
    evo_dir.mkdir(parents=True, exist_ok=True)

    est_path = evo_dir / f"{namespace}_odometry_{evo_dir_name}.tum"
    np.savetxt(est_path, np.column_stack([results[k] for k in TUM_KEYS]), fmt="%.9f")
    logger.info(f"Saved TUM: {est_path}")

    if pose_gt:
        gt_path = odom_dir / f"{namespace}_odometry_truth.tum"
        run_evo_evaluations(str(gt_path), str(est_path), evo_dir, evo_flags)

    return results, pose_gt


def _mask_gaps(t: np.ndarray, vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
