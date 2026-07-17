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
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from scoring import estimators

logger = logging.getLogger(__name__)

TUM_KEYS = ("time", "x", "y", "z", "qx", "qy", "qz", "qw")


def run_logged(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    """
    Run a subprocess. It will write directly to the terminal.

    :param args: Command and arguments to execute.
    :param cwd: Working directory to run the command in, if any.
    :return: The completed process.
    """
    return subprocess.run(args, cwd=cwd)


def evo_agent_dir(bag_path: str | Path, namespace: str) -> Path:
    """
    Return the bag's evo output directory for one agent namespace.

    :param bag_path: Path to the ROS 2 bag directory.
    :param namespace: AUV namespace whose outputs are stored.
    :return: The agent's evo output directory.
    """
    return Path(bag_path) / "evo" / namespace


def iter_evaluated_agents(target_dir: Path) -> Iterator[tuple[Path, Path]]:
    """
    Yield each evaluated agent directory under a target directory.

    :param target_dir: A bag or directory of bags that has been evaluated.
    :return: ``(bag_dir, agent_dir)`` pairs found under every ``evo`` folder.
    """
    for evo_dir in target_dir.rglob("evo"):
        bag_dir = evo_dir.parent
        if not (bag_dir / "metadata.yaml").exists():
            continue
        for agent_dir in filter(Path.is_dir, evo_dir.iterdir()):
            yield bag_dir, agent_dir


def latest_tum(directory: Path) -> Path | None:
    """
    Return the most recently modified TUM file in a directory, if any.

    :param directory: Directory to search for ``*.tum`` files.
    :return: The newest TUM file by modification time, or None if none exist.
    """
    tum_files = sorted(directory.glob("*.tum"), key=lambda p: p.stat().st_mtime)
    return tum_files[-1] if tum_files else None


def ensure_ground_truth(bag_path: str | Path, namespace: str) -> Path | None:
    """
    Return the agent's ground truth TUM file, exporting it from the bag if needed.

    :param bag_path: Path to the ROS 2 bag directory.
    :param namespace: AUV namespace the ground truth belongs to.
    :return: Path to the ground truth TUM file, or None if it could not be produced.
    """
    agent_dir = evo_agent_dir(bag_path, namespace)
    tum_path = latest_tum(agent_dir)
    if tum_path is None:
        logger.warning(
            f"No ground truth TUM found in {agent_dir}; attempting export..."
        )
        truth_topic = f"/{namespace}/{estimators.TRUTH_TOPIC}"
        tum_path = export_bag_tum(bag_path, truth_topic, agent_dir)
    return tum_path


def load_ground_truth(bag_path: str | Path, namespace: str) -> tuple[dict, Path | None]:
    """
    Load the ground truth into state arrays, exporting it from the bag first if needed.

    :param bag_path: Path to the ROS 2 bag directory.
    :param namespace: AUV namespace the ground truth was exported under.
    :return: Tuple of arrays keyed by state name, and the path to the TUM file.
    """
    tum_path = ensure_ground_truth(bag_path, namespace)
    if tum_path is None:
        logger.error(f"Could not find or export ground truth for {namespace}.")
        return {}, None
    try:
        data = np.loadtxt(tum_path, comments="#", ndmin=2)
    except (OSError, ValueError):
        logger.error(f"Could not load ground truth from {tum_path}")
        return {}, None
    if data.size == 0:
        logger.error(f"Ground truth file is empty: {tum_path}")
        return {}, None
    if data.shape[1] < len(TUM_KEYS):
        logger.error(f"Ground truth file has too few columns: {tum_path}")
        return {}, None

    pose = {k: data[:, i] for i, k in enumerate(TUM_KEYS)}
    roll, pitch, yaw = Rotation.from_quat(data[:, 4:8]).as_euler("xyz").T
    pose.update({"roll": roll, "pitch": pitch, "yaw": yaw})

    logger.info(f"Loaded ground truth: {tum_path}")
    return pose, tum_path


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


def export_bag_tum(bag_path: str | Path, topic: str, out_dir: Path) -> Path | None:
    """
    Export a recorded trajectory topic from a bag to a TUM file with evo.

    :param bag_path: Path to the ROS 2 bag directory.
    :param topic: Trajectory topic to export (e.g. an odometry topic).
    :param out_dir: Directory to write the exported TUM file into.
    :return: Path to the exported TUM file, or None if the export failed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    args = ["evo_traj", "bag2", str(Path(bag_path).resolve()), topic, "--save_as_tum"]
    if run_logged(args, cwd=out_dir).returncode != 0:
        return None

    return latest_tum(out_dir)
