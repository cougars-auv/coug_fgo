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

import yaml
from plots import benchmark_plots, lag_plots, timing_plots, trajectory_plots

from utils import estimators, evo_tools

logger = logging.getLogger(__name__)

BENCHMARK_METRICS = ("ape_trans", "ape_rot", "rpe_trans", "rpe_rot")


def _find_bags(target_dir: Path) -> list[Path]:
    """
    Return every bag directory at or beneath a target directory.

    :param target_dir: A bag directory or a directory containing bags.
    :return: Bag directories, identified by their ``metadata.yaml`` files.
    """
    return sorted(meta.parent for meta in target_dir.rglob("metadata.yaml"))


def _bag_message_counts(bag_path: Path) -> dict[str, int]:
    """
    Map each recorded topic in a bag to its message count.

    :param bag_path: Path to the ROS 2 bag directory.
    :return: Message count keyed by topic name.
    """
    meta = yaml.safe_load((bag_path / "metadata.yaml").read_text())
    info = meta.get("rosbag2_bagfile_information", {})
    return {
        entry["topic_metadata"]["name"]: entry.get("message_count", 0)
        for entry in info.get("topics_with_message_count", [])
        if "topic_metadata" in entry and "name" in entry.get("topic_metadata", {})
    }


def _evaluate_estimator(
    bag_path: Path,
    est: estimators.Estimator,
    agent: str,
    agent_dir: Path,
    gt_tum: Path | None,
    counts: dict[str, int],
    evo_flags: list[str],
) -> None:
    """
    Export and benchmark a single estimator topic against ground truth.

    :param bag_path: Path to the ROS 2 bag directory.
    :param est: Estimator registry entry to evaluate.
    :param agent: AUV namespace being evaluated.
    :param agent_dir: The agent's evo output directory.
    :param gt_tum: Ground truth TUM path, or None if unavailable.
    :param counts: Message counts keyed by topic name for this bag.
    :param evo_flags: Extra evo flags forwarded to APE and RPE runs.
    """
    topic = f"/{agent}/{est.topic}"
    out_dir = agent_dir / est.key
    est_tum = evo_tools.latest_tum(out_dir)

    if est_tum is None and counts.get(topic, 0) == 0:
        return

    if est_tum is not None:
        if gt_tum is None:
            logger.warning(f"Skipping {topic}, no ground truth available.")
            return
        if all((out_dir / f"{m}.zip").exists() for m in BENCHMARK_METRICS):
            logger.info(f"Skipping {topic}, results already exist.")
            return

    logger.info(f"Evaluating {topic}...")
    est_tum = est_tum or evo_tools.export_bag_tum(bag_path, topic, out_dir)
    if est_tum is None:
        logger.error(f"Could not export {topic}.")
        return

    if gt_tum is not None:
        evo_tools.run_evo_evaluations(gt_tum, est_tum, out_dir, evo_flags)


def _evaluate_agent(
    bag_path: Path, agent: str, counts: dict[str, int], evo_flags: list[str]
) -> None:
    """
    Export, evaluate, and benchmark every estimator topic for one agent.

    :param bag_path: Path to the ROS 2 bag directory.
    :param agent: AUV namespace to evaluate.
    :param counts: Message count keyed by topic name for this bag.
    :param evo_flags: Extra evo flags forwarded to APE and RPE runs.
    """
    agent_dir = evo_tools.evo_agent_dir(bag_path, agent)
    truth_topic = f"/{agent}/{estimators.TRUTH_TOPIC}"

    has_gt = (
        evo_tools.latest_tum(agent_dir) is not None or counts.get(truth_topic, 0) > 0
    )
    has_est = any(
        evo_tools.latest_tum(agent_dir / est.key) is not None
        or counts.get(f"/{agent}/{est.topic}", 0) > 0
        for est in estimators.exported_estimators()
    )
    if not has_gt and not has_est:
        return

    gt_tum = evo_tools.ensure_ground_truth(bag_path, agent) if has_gt else None
    if gt_tum is None:
        logger.warning(f"No ground truth found for {agent}.")

    for est in estimators.exported_estimators():
        _evaluate_estimator(bag_path, est, agent, agent_dir, gt_tum, counts, evo_flags)

    evo_tools.build_benchmark_tables(agent_dir, BENCHMARK_METRICS)


def evaluate_bags(target_dir: Path, agents: list[str], evo_flags: list[str]) -> None:
    """
    Evaluate every bag at or beneath a target directory and render summary plots.

    :param target_dir: A bag directory or a directory of bags to evaluate.
    :param agents: AUV namespaces to evaluate; absent agents are skipped.
    :param evo_flags: Extra evo flags forwarded to APE and RPE runs.
    """
    bags = _find_bags(target_dir)
    if not bags:
        logger.error(f"No bags found in {target_dir}")
        return

    for bag_path in bags:
        logger.info(f"Processing {bag_path}...")
        counts = _bag_message_counts(bag_path)
        for agent in agents:
            _evaluate_agent(bag_path, agent, counts, evo_flags)

    do_align = "--align" in evo_flags
    trajectory_plots.render(target_dir, do_align=do_align)
    timing_plots.render(target_dir)
    benchmark_plots.render(target_dir)
    lag_plots.render(target_dir)
