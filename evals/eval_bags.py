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
from pathlib import Path

from utils import evaluation
from utils.log_setup import setup_logging

logger = logging.getLogger(__name__)

TARGET_DIR = Path.home() / "cougars-dev/bags/turtlmap_offline"
AGENTS = ["coug1sim", "coug2sim", "coug3sim", "blue1sim", "bluerov2", "turtlmap"]
EVO_FLAGS = ["--align"]  # , "--project_to_plane", "xy"]


def main() -> None:
    setup_logging()
    evaluation.evaluate_bags(TARGET_DIR, AGENTS, EVO_FLAGS)


if __name__ == "__main__":
    main()
