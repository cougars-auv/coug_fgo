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

from utils import estimators, evo_tools

logger = logging.getLogger(__name__)

BENCHMARK_METRICS = ("ape_trans", "ape_rot", "rpe_trans", "rpe_rot")


def _find_bags(target_dir: Path) -> list[Path]:
    """
    Return every bag directory at or beneath a target directory.

    :param target_dir: A bag directory or a directory containing bags.
    :return: The bag directories, identified by their ``metadata.yaml`` files.
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
    counts: dict[str, int] = {}
    for entry in info.get("topics_with_message_count", []):
        name = entry.get("topic_metadata", {}).get("name")
        if name is not None:
            counts[name] = entry.get("message_count", 0)
    return counts


def _evaluate_agent(
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

    # The agent list may name agents that were not recorded in this bag; skip them
    # before attempting an export that we already know would fail.
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
        est_tum = est_tum or evo_tools.export_bag_tum(bag_path, topic, out_dir)
        if est_tum is None:
            logger.warning(f"Could not export {topic}; skipping.")
            continue

        evo_tools.run_evo_evaluations(gt_tum, est_tum, out_dir, evo_flags)

    evo_tools.build_benchmark_tables(agent_dir, BENCHMARK_METRICS)


def _render_plots(target_dir: Path, evo_flags: list[str]) -> None:
    """
    Render the trajectory, timing, benchmark, and lag summary plots.

    :param target_dir: The bag or directory of bags that was evaluated.
    :param evo_flags: Evo flags; alignment is forwarded to the trajectory plot.
    """
    # Imported here so the plotting stack is only loaded once evaluation succeeds.
    from plots import benchmark_plots, lag_plots, timing_plots, trajectory_plots

    do_align = "--align" in evo_flags
    plotters = [
        ("trajectory", lambda: trajectory_plots.render(target_dir, do_align=do_align)),
        ("timing", lambda: timing_plots.render(target_dir)),
        ("benchmark", lambda: benchmark_plots.render(target_dir)),
        ("lag", lambda: lag_plots.render(target_dir)),
    ]
    failed = []
    for name, render in plotters:
        logger.info(f"Rendering {name} plot...")
        try:
            render()
        except Exception as e:
            # Keep rendering the remaining plots; report the traceback and collect
            # the failure for a summary once every plot has been attempted.
            failed.append(name)
            logger.exception(f"Failed to render {name} plot for {target_dir}: {e}")

    if failed:
        logger.error(f"{len(failed)}/{len(plotters)} plots failed: {', '.join(failed)}")


def evaluate_bags(target_dir: Path, agents: list[str], evo_flags: list[str]) -> None:
    """
    Evaluate every bag at or beneath a target directory and render summary plots.

    For each bag, exports and benchmarks every recorded estimator topic against
    ground truth, then renders the trajectory, timing, benchmark, and lag plots
    across the whole target.

    :param target_dir: A bag directory or a directory of bags to evaluate.
    :param agents: AUV namespaces to evaluate; those absent from a given bag (no
        recorded ground truth) are skipped.
    :param evo_flags: Extra evo flags passed to the APE and RPE evaluations.
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

    _render_plots(target_dir, evo_flags)
