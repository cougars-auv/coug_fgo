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

import argparse
import logging
import subprocess
from pathlib import Path

import colorlog
import yaml

from utils import evo_tools

logger = logging.getLogger(__name__)

# Estimator topics to look for under each discovered agent namespace.
TRUTH_SUFFIX = "odometry/truth"
SUFFIXES = (
    "odometry/global",
    "odometry/global_isam2",
    "odometry/global_lpi",
    "odometry/global_tpi",
    "odometry/global_tm",
    "dvl/odometry",
    "imu/odometry",
)
BENCHMARK_METRICS = ("ape_trans", "ape_rot", "rpe_trans", "rpe_rot")
PLOTS_DIR = Path(__file__).resolve().parent / "plots"


def setup_logging() -> None:
    """Configure colored console logging for the script."""
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s[%(levelname)s] %(message)s",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "white",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )
    )
    logging.basicConfig(level=logging.INFO, handlers=[handler])


def find_bags(target_dir: Path) -> list[Path]:
    """
    Return every bag directory at or beneath a target directory.

    :param target_dir: A bag directory or a directory containing bags.
    :return: The bag directories, identified by their metadata.yaml files.
    """
    return sorted(meta.parent for meta in target_dir.rglob("metadata.yaml"))


def bag_message_counts(bag_path: Path) -> dict[str, int]:
    """
    Map each recorded topic in a bag to its message count.

    :param bag_path: Path to the ROS 2 bag directory.
    :return: Message count keyed by topic name.
    """
    meta = yaml.safe_load((bag_path / "metadata.yaml").read_text())
    info = meta.get("rosbag2_bagfile_information", {})
    counts: dict[str, int] = {}
    for entry in info.get("topics_with_message_count", []):
        name = entry.get("topic_metadata", {}).get("name")
        if name is not None:
            counts[name] = entry.get("message_count", 0)
    return counts


def discover_agents(counts: dict[str, int]) -> list[str]:
    """
    Find agent namespaces in a bag, auto-detected from ``/{agent}/odometry/truth`` topics.

    :param counts: Message count keyed by topic name for the bag.
    :return: Sorted, de-duplicated agent namespaces.
    """
    agents = set()
    for topic in counts:
        parts = topic.split("/")
        # Match "/{agent}/odometry/truth" -> ["", agent, "odometry", "truth"].
        if topic.endswith(f"/{TRUTH_SUFFIX}") and len(parts) == 4:
            agents.add(parts[1])
    return sorted(agents)


def _existing_tum(out_dir: Path) -> Path | None:
    """Return the first exported TUM file in a directory, if any."""
    tum_files = sorted(out_dir.glob("*.tum"))
    return tum_files[0] if tum_files else None


def evaluate_agent(
    bag_path: Path, agent: str, counts: dict[str, int], evo_flags: list[str]
) -> None:
    """
    Export, evaluate, and benchmark every estimator topic for one agent.

    :param bag_path: Path to the ROS 2 bag directory.
    :param agent: AUV namespace to evaluate.
    :param counts: Message count keyed by topic name for this bag.
    :param evo_flags: Extra evo flags passed to the APE and RPE evaluations.
    """
    agent_dir = bag_path / "evo" / agent
    truth_topic = f"/{agent}/{TRUTH_SUFFIX}"

    # Export the ground truth once, reusing it across every estimator topic.
    gt_tum = _existing_tum(agent_dir)
    if gt_tum is None:
        logger.info(f"Exporting ground truth for {agent}...")
        gt_tum = evo_tools.export_bag_tum(str(bag_path), truth_topic, agent_dir)
    if gt_tum is None:
        logger.warning(f"No ground truth found for {agent}; skipping.")
        return

    for suffix in SUFFIXES:
        topic = f"/{agent}/{suffix}"
        # Name the output folder after the non-odometry topic segment.
        name = suffix.removeprefix("odometry/").removesuffix("/odometry")
        out_dir = agent_dir / name

        est_tum = _existing_tum(out_dir)
        # Skip topics that were never recorded (and not already exported).
        if est_tum is None and counts.get(topic, 0) == 0:
            continue

        if est_tum is not None and all(
            (out_dir / f"{m}.zip").exists() for m in BENCHMARK_METRICS
        ):
            logger.info(f"Skipping {topic} (results already exist)")
            continue

        logger.info(f"Evaluating {topic}...")
        if est_tum is None:
            est_tum = evo_tools.export_bag_tum(str(bag_path), topic, out_dir)
        if est_tum is None:
            logger.warning(f"Could not export {topic}; skipping.")
            continue

        evo_tools.run_evo_evaluations(str(gt_tum), str(est_tum), out_dir, evo_flags)

    evo_tools.build_benchmark_tables(agent_dir, BENCHMARK_METRICS)


def render_plots(target_dir: Path, evo_flags: list[str]) -> None:
    """
    Render the trajectory, timing, benchmark, and lag summary plots.

    :param target_dir: The bag or directory of bags that was evaluated.
    :param evo_flags: Evo flags; alignment flags are forwarded to the plotter.
    """
    align_flags = [f for f in evo_flags if f in ("--align", "--align_origin")]
    subprocess.run(
        [
            "python3",
            str(PLOTS_DIR / "trajectory_plot.py"),
            str(target_dir),
            *align_flags,
        ]
    )
    for script in ("timing_plot.py", "benchmark_plot.py", "lag_plot.py"):
        subprocess.run(["python3", str(PLOTS_DIR / script), str(target_dir)])


def main() -> None:
    """Evaluate the selected bags and render the summary plots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target_dir", help="A bag directory or a directory of bags.")
    parser.add_argument("--align", action="store_true", help="Umeyama alignment.")
    parser.add_argument("--align_origin", action="store_true", help="Align origins.")
    parser.add_argument(
        "--project_to_plane", action="store_true", help="Project to the xy plane."
    )
    args = parser.parse_args()

    setup_logging()

    evo_flags: list[str] = []
    if args.align:
        evo_flags.append("--align")
    if args.align_origin:
        evo_flags.append("--align_origin")
    if args.project_to_plane:
        evo_flags += ["--project_to_plane", "xy"]

    target_dir = Path(args.target_dir)
    bags = find_bags(target_dir)
    if not bags:
        logger.error(f"No bags found in {target_dir}")
        return

    for bag_path in bags:
        logger.info(f"Processing {bag_path}...")
        counts = bag_message_counts(bag_path)
        for agent in discover_agents(counts):
            evaluate_agent(bag_path, agent, counts, evo_flags)

    render_plots(target_dir, evo_flags)


if __name__ == "__main__":
    main()
