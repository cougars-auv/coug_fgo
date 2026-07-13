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
from pathlib import Path

import yaml

from utils import estimators, evo_tools
from utils.logging import setup_logging

logger = logging.getLogger(__name__)

CONFIG_SUFFIX = "_params.yaml"
BENCHMARK_METRICS = ("ape_trans", "ape_rot", "rpe_trans", "rpe_rot")


def find_bags(target_dir: Path) -> list[Path]:
    """
    Return every bag directory at or beneath a target directory.

    :param target_dir: A bag directory or a directory containing bags.
    :return: The bag directories, identified by their ``metadata.yaml`` files.
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


def discover_agents(bag_path: Path) -> list[str]:
    """
    List agent namespaces from the config snapshot copied into the bag.

    Namespaces come from the per-agent ``<namespace>_params.yaml`` files at the top
    level of the bag's ``config`` directory, the same files the launch scripts load
    to resolve each agent's parameters. The snapshot is a full copy of the repo
    config, so it may name agents that were not actually recorded; those are skipped
    later when no ground truth is found.

    :param bag_path: Path to the ROS 2 bag directory.
    :return: Sorted agent namespaces.
    """
    config_dir = bag_path / "config"
    return sorted(
        p.name.removesuffix(CONFIG_SUFFIX) for p in config_dir.glob(f"*{CONFIG_SUFFIX}")
    )


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
    agent_dir = evo_tools.evo_agent_dir(bag_path, agent)
    truth_topic = f"/{agent}/{estimators.TRUTH_TOPIC}"

    # Config lists every possible agent; skip those absent from this bag before
    # attempting an export that we already know would fail.
    if evo_tools.latest_tum(agent_dir) is None and counts.get(truth_topic, 0) == 0:
        return

    # Resolve the ground truth once, reusing it across every estimator topic.
    gt_tum = evo_tools.ensure_ground_truth(bag_path, agent)
    if gt_tum is None:
        logger.warning(f"No ground truth found for {agent}; skipping.")
        return

    for est in estimators.exported_estimators():
        topic = f"/{agent}/{est.topic}"
        # Each estimator's outputs live in a folder named after its registry key.
        out_dir = agent_dir / est.key

        est_tum = evo_tools.latest_tum(out_dir)
        # Skip topics that were never recorded (and not already exported).
        if est_tum is None and counts.get(topic, 0) == 0:
            continue

        if est_tum is not None and all(
            (out_dir / f"{m}.zip").exists() for m in BENCHMARK_METRICS
        ):
            logger.info(f"Skipping {topic} (results already exist)")
            continue

        logger.info(f"Evaluating {topic}...")
        est_tum = est_tum or evo_tools.export_bag_tum(str(bag_path), topic, out_dir)
        if est_tum is None:
            logger.warning(f"Could not export {topic}; skipping.")
            continue

        evo_tools.run_evo_evaluations(str(gt_tum), str(est_tum), out_dir, evo_flags)

    evo_tools.build_benchmark_tables(agent_dir, BENCHMARK_METRICS)


def render_plots(target_dir: Path, evo_flags: list[str]) -> None:
    """
    Render the trajectory, timing, benchmark, and lag summary plots.

    :param target_dir: The bag or directory of bags that was evaluated.
    :param evo_flags: Evo flags; alignment is forwarded to the trajectory plot.
    """
    # Imported here so the plotting stack is only loaded once evaluation succeeds.
    from plots import benchmark_plot, lag_plot, timing_plot, trajectory_plot

    do_align = "--align" in evo_flags
    plotters = [
        ("trajectory", lambda: trajectory_plot.render(target_dir, do_align=do_align)),
        ("timing", lambda: timing_plot.render(target_dir)),
        ("benchmark", lambda: benchmark_plot.render(target_dir)),
        ("lag", lambda: lag_plot.render(target_dir)),
    ]
    for name, render in plotters:
        try:
            render()
        except Exception:
            logger.exception(f"{name} plot failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target_dir", help="A bag directory or a directory of bags.")
    parser.add_argument("--align", action="store_true", help="Umeyama alignment.")
    parser.add_argument(
        "--project_to_plane", action="store_true", help="Project to the xy plane."
    )
    args = parser.parse_args()

    setup_logging()

    evo_flags: list[str] = []
    if args.align:
        evo_flags.append("--align")
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
        for agent in discover_agents(bag_path):
            evaluate_agent(bag_path, agent, counts, evo_flags)

    render_plots(target_dir, evo_flags)


if __name__ == "__main__":
    main()
