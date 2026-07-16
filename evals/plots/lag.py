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

import matplotlib.pyplot as plt
import pandas as pd
import scienceplots  # noqa: F401
import seaborn as sns

from scoring import bags, estimators, tum

logger = logging.getLogger(__name__)

plt.style.use(["science", "ieee"])

FGO_COLOR = estimators.timed_estimators()[0].color


def _collect_durations_by_lag(target_dir: Path) -> pd.DataFrame:
    """
    Collect solver durations and smoother lags from all evaluated bags.

    :param target_dir: Directory tree containing evaluated bags.
    :return: DataFrame pairing each duration with its configured lag.
    """
    timing_data = []

    for bag_dir, agent_dir in tum.iter_evaluated_agents(target_dir):
        lag = bags.get_smoother_lag(bag_dir, agent_dir.name)
        if lag is None:
            logger.warning(
                f"No smoother_lag found in {bag_dir} for {agent_dir.name}, skipping."
            )
            continue

        for duration in bags.read_bag_durations(bag_dir, agent_dir.name):
            timing_data.append(
                {
                    "Smoother Lag (s)": lag,
                    "Duration (s)": duration,
                }
            )

    return pd.DataFrame(timing_data)


def render(target_dir: Path) -> None:
    """
    Save violin and box plots of solver duration by smoother lag.

    :param target_dir: A bag or directory of bags that has been evaluated.
    """
    logger.info("Rendering lag plots...")
    df = _collect_durations_by_lag(target_dir)
    if df.empty:
        logger.error(f"No solver durations with smoother lags found for {target_dir}.")
        return

    lag_values = sorted(df["Smoother Lag (s)"].unique())
    df["Smoother Lag (s)"] = pd.Categorical(
        df["Smoother Lag (s)"], categories=lag_values, ordered=True
    )
    df = df.sort_values("Smoother Lag (s)")

    for plot_type in ["violin", "box"]:
        fig, ax = plt.subplots()

        plot_func = sns.violinplot if plot_type == "violin" else sns.boxplot
        kwargs = (
            {"inner": "box", "linewidth": 0.5, "cut": 0}
            if plot_type == "violin"
            else {}
        )

        plot_func(
            x="Smoother Lag (s)",
            y="Duration (s)",
            data=df,
            color=FGO_COLOR,
            ax=ax,
            **kwargs,
        )

        ax.set(title="", xlabel="Smoother Lag (s)", ylabel="Duration (s)")
        out_path = target_dir / f"lag_{plot_type}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved plot: {out_path}")
