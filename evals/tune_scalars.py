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

import logging
import math
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import optuna
from tqdm.contrib.logging import logging_redirect_tqdm

from plots import state_plots
from utils import evo_tools, pipeline
from utils.log_setup import setup_logging

logger = logging.getLogger(__name__)

NAMESPACE = "turtlmap"
BAG_PATHS = [
    str(
        Path.home()
        / "cougars-dev/bags/turtlmap_offline/log1_offline_2026-07-06-16-08-28"
    ),
    str(
        Path.home()
        / "cougars-dev/bags/turtlmap_offline/log2_offline_2026-07-06-16-15-19"
    ),
]
CONFIG_PATHS = [
    str(Path.home() / "cougars-dev/config/fleet/coug_fgo_params.yaml"),
    str(Path.home() / f"cougars-dev/config/{NAMESPACE}_params.yaml"),
]
EVO_FLAGS = ["--align"]  # , "--project_to_plane", "xy"]

DB_URL = f"sqlite:///{Path(__file__).parent.resolve()}/scalar_tuning.db"
STUDY_NAME = f"{NAMESPACE}_scalar_sweep_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

SCALARS_TO_TUNE = ["const_vel"]
N_OPTUNA_TRIALS = 100
MIN_SCALAR = 0.01
MAX_SCALAR = 100

QUIET_LOGGERS = (
    "utils.pipeline",
    "utils.factor_graph",
    "utils.urdf",
    # "utils.evo_tools",
    "coug_fgo.core",
)


@contextmanager
def quiet_loggers(level: int = logging.ERROR):
    """
    Temporarily raise the level of the noisy per-run loggers.

    :param level: Level to hold the loggers at while the context is active.
    """
    loggers = [logging.getLogger(name) for name in QUIET_LOGGERS]
    previous = [lg.level for lg in loggers]
    for lg in loggers:
        lg.setLevel(level)
    try:
        yield
    finally:
        for lg, lvl in zip(loggers, previous):
            lg.setLevel(lvl)


def objective(trial: optuna.Trial, ground_truths: list[dict]) -> float:
    """
    Score one set of covariance scalars across all configured bags.

    :param trial: Optuna trial used to suggest the scalar values.
    :param ground_truths: Ground truth arrays for each bag in BAG_PATHS.
    :return: Root mean square of the per-bag APE RMSE values.
    """
    scalars = {
        s: trial.suggest_float(s, MIN_SCALAR, MAX_SCALAR, log=True)
        for s in SCALARS_TO_TUNE
    }
    rmses = []
    with pipeline.covariance_override_file(scalars) as override_path:
        for bag, pose_gt in zip(BAG_PATHS, ground_truths):
            results, crashed = pipeline.process_bag_offline(
                bag, CONFIG_PATHS + [override_path], NAMESPACE
            )
            rmses.append(evo_tools.compute_ape_rmse(pose_gt, results, crashed))

    if not rmses:
        return float("inf")
    return math.sqrt(sum(r**2 for r in rmses) / len(rmses))


def main() -> None:
    setup_logging()

    ground_truths = [evo_tools.load_ground_truth(bag, NAMESPACE) for bag in BAG_PATHS]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=STUDY_NAME, storage=DB_URL, direction="minimize"
    )
    with logging_redirect_tqdm(), quiet_loggers():
        study.optimize(
            lambda trial: objective(trial, ground_truths),
            n_trials=N_OPTUNA_TRIALS,
            show_progress_bar=True,
        )
    print(f"Best scalars: {study.best_params}")

    plot_args = []
    with logging_redirect_tqdm():
        with pipeline.covariance_override_file(study.best_params) as best_override_path:
            for bag in BAG_PATHS:
                result = pipeline.process_and_evaluate(
                    bag,
                    CONFIG_PATHS + [best_override_path],
                    NAMESPACE,
                    "tuned",
                    EVO_FLAGS,
                )
                if result is not None:
                    plot_args.append(result)

    for results, pose_gt, label in plot_args:
        state_plots.plot_results(results, pose_gt, label)
    plt.show()


if __name__ == "__main__":
    main()
