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

import math
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import optuna

import logging
import colorlog
from tqdm.contrib.logging import logging_redirect_tqdm

from utils import metrics
from utils import pipeline as fgo_pipeline

NAMESPACE = "turtlmap"
BAG_PATHS = [
    str(Path.home() / "cougars-dev/bags/turtlmap_batch/log1_batch_2026-05-05-11-16-18"),
    str(Path.home() / "cougars-dev/bags/turtlmap_batch/log2_batch_2026-05-05-11-22-42"),
]
CONFIG_PATHS = [
    str(Path.home() / "cougars-dev/config/fleet/coug_fgo_params.yaml"),
    str(Path.home() / f"cougars-dev/config/{NAMESPACE}_params.yaml"),
]
EVO_FLAGS = ["--align"]  # , "--project_to_plane", "xy"]

DB_URL = f"sqlite:///{Path(__file__).parent.resolve()}/scalar_tuning.db"
STUDY_NAME = f"{NAMESPACE}_scalar_sweep_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

SCALARS_TO_TUNE = ["ahrs", "depth", "dvl"]
N_OPTUNA_TRIALS = 1
MIN_SCALAR = 0.01
MAX_SCALAR = 100


def objective(trial: optuna.Trial, ground_truths: list[dict]) -> float:
    scalars = {
        s: trial.suggest_float(s, MIN_SCALAR, MAX_SCALAR, log=True)
        for s in SCALARS_TO_TUNE
    }
    rmses = []
    with fgo_pipeline.covariance_override_file(scalars) as override_path:
        for bag, pose_gt in zip(BAG_PATHS, ground_truths):
            results, crashed = fgo_pipeline.process_bag_offline(
                bag, CONFIG_PATHS + [override_path], NAMESPACE, verbose=False
            )
            rmses.append(metrics.compute_ape_rmse(pose_gt, results, crashed))

    if not rmses:
        return float("inf")
    return math.sqrt(sum(r**2 for r in rmses) / len(rmses))


def main() -> None:
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

    ground_truths = [metrics.load_ground_truth(bag, NAMESPACE) for bag in BAG_PATHS]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=STUDY_NAME, storage=DB_URL, direction="minimize"
    )
    with logging_redirect_tqdm():
        study.optimize(
            lambda trial: objective(trial, ground_truths),
            n_trials=N_OPTUNA_TRIALS,
            show_progress_bar=True,
        )
    print(f"Best scalars: {study.best_params}")

    plot_args = []
    with logging_redirect_tqdm():
        with fgo_pipeline.covariance_override_file(
            study.best_params
        ) as best_override_path:
            for bag in BAG_PATHS:
                results, pose_gt = metrics.evaluate_and_save(
                    bag,
                    CONFIG_PATHS + [best_override_path],
                    NAMESPACE,
                    "tuned",
                    EVO_FLAGS,
                )
                if results:
                    plot_args.append((results, pose_gt, Path(bag).name))

    for results, pose_gt, label in plot_args:
        metrics.plot_results(results, pose_gt, label)
    plt.show()


if __name__ == "__main__":
    main()
