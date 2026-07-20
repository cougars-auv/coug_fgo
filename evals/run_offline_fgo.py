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
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from tqdm.contrib.logging import logging_redirect_tqdm

from config import BAG_PATHS, EVO_FLAGS, NAMESPACE, config_paths
from logs import setup_logging
from offline import pipeline
from plots import state
from scoring import metrics, tum

logger = logging.getLogger(__name__)


def save_config(dest_dir: Path) -> None:
    """
    Copy the active config directory into a run's output directory.

    :param dest_dir: Output directory to copy the config folder into.
    """
    config_dir = os.environ.get("CONFIG_DIR", "")
    if not config_dir or not os.path.isdir(config_dir):
        return

    dest = dest_dir / "config"
    shutil.copytree(config_dir, dest, dirs_exist_ok=True)
    logger.info(f"Config saved: {dest}")


def process_and_evaluate(
    bag_path: str,
    config_paths: list[str],
    namespace: str,
    tag: str,
    evo_flags: list[str],
    **kwargs,
) -> tuple[dict, dict, str] | None:
    """
    Run the full offline pipeline for one bag: load truth, process, save, evaluate.

    :param bag_path: Path to the ROS 2 bag directory.
    :param config_paths: Parameter YAML files, in increasing priority.
    :param namespace: AUV namespace used for topics and parameters.
    :param tag: Subdirectory and file suffix for this run (e.g. ``offline``).
    :param evo_flags: Extra evo flags forwarded to the APE and RPE runs.
    :param kwargs: Extra keyword arguments forwarded to ``process_bag_offline``.
    :return: ``(results, pose_gt, label)`` tuple, or None if no results.
    """
    logger.info(f"Processing bag: {bag_path}")
    pose_gt, gt_path = tum.load_ground_truth(bag_path, namespace)
    results, _ = pipeline.process_bag_offline(
        bag_path, config_paths, namespace, **kwargs
    )
    if not results:
        return None

    evo_dir = tum.evo_agent_dir(bag_path, namespace) / tag
    evo_dir.mkdir(parents=True, exist_ok=True)
    save_config(evo_dir)
    est_path = evo_dir / f"{namespace}_{tag}.tum"
    tum.save_tum(est_path, results)

    if pose_gt and gt_path is not None:
        metrics.run_evo_evaluations(gt_path, est_path, evo_dir, evo_flags)
        if "--align" in evo_flags:
            metrics.align_dicts(results, pose_gt)

    return results, pose_gt, Path(bag_path).name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--namespace",
        default=NAMESPACE,
        help="AUV namespace to process the bags under",
    )
    parser.add_argument(
        "--bags",
        nargs="+",
        default=BAG_PATHS,
        help="Bag directories to process offline",
    )
    parser.add_argument(
        "--evo-flags",
        default=" ".join(EVO_FLAGS),
        help="Extra evo flags forwarded to APE and RPE runs, e.g. "
        "--evo-flags='--align --project_to_plane xy'",
    )
    args = parser.parse_args()

    setup_logging()

    plot_args = []
    with logging_redirect_tqdm():
        for bag in args.bags:
            result = process_and_evaluate(
                bag,
                config_paths(args.namespace),
                args.namespace,
                "offline",
                args.evo_flags.split(),
            )
            if result is not None:
                plot_args.append(result)

    for results, pose_gt, label in plot_args:
        state.plot_results(results, pose_gt, label)
    plt.show()


if __name__ == "__main__":
    main()
