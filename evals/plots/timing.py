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

COLORS = estimators.color_map()
ALGORITHMS = [e.label for e in estimators.timed_estimators()]


def render(target_dir: Path) -> None:
    """
    Save solver timing plots for every evaluated agent under a target directory.

    :param target_dir: A bag or directory of bags that has been evaluated.
    """
    logger.info("Rendering timing plots...")
    for bag_dir, agent_dir in tum.iter_evaluated_agents(target_dir):
        df = bags.read_timing_metrics(bag_dir, agent_dir.name)
        if df.empty:
            logger.warning(
                f"No timing metrics found for {agent_dir.name} in {bag_dir.name}."
            )
            continue

        present_algos = [
            algo for algo in ALGORITHMS if algo in df["Algorithm"].unique()
        ]
        df["Algorithm"] = pd.Categorical(
            df["Algorithm"], categories=present_algos, ordered=True
        )
        df = df.sort_values("Algorithm")

        df_melted = df.melt(
            id_vars=["Algorithm"],
            value_vars=["Total", "Smoother", "Covariance"],
            var_name="Timing Type",
            value_name="Duration (s)",
        )

        for plot_type in ["violin", "box"]:
            fig, ax = plt.subplots()

            plot_func = sns.violinplot if plot_type == "violin" else sns.boxplot
            kwargs = (
                {"inner": "box", "linewidth": 0.5, "cut": 0}
                if plot_type == "violin"
                else {}
            )

            plot_func(
                x="Timing Type",
                y="Duration (s)",
                hue="Algorithm",
                data=df_melted,
                palette=COLORS,
                ax=ax,
                **kwargs,
            )

            ax.set(title="", xlabel="", ylabel="Duration (s)")
            ax.legend(title=None)
            out_path = bag_dir / f"{agent_dir.name}_timing_{plot_type}.png"
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Saved plot: {out_path}")
