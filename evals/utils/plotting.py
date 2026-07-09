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

import matplotlib.pyplot as plt
import numpy as np


def gap_indices(t: np.ndarray) -> np.ndarray:
    """
    Find the indices where NaN breaks belong to split large timestamp gaps.

    :param t: Sample timestamps.
    :return: Insertion indices where the timestamps jump (empty if none).
    """
    if len(t) < 2:
        return np.array([], dtype=int)
    dts = np.diff(t)
    threshold = max(float(np.median(dts)) * 5.0, 0.5)
    return np.where(dts > threshold)[0] + 1


def _mask_gaps(t: np.ndarray, vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Insert NaN breaks at large timestamp gaps to avoid bridging in plots.

    :param t: Sample timestamps.
    :param vals: Sample values aligned with the timestamps.
    :return: Timestamps and values with NaN entries inserted at the gaps.
    """
    gap_idx = gap_indices(t)
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
