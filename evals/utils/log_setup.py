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

import colorlog


def setup_logging() -> None:
    """Configure colored console logging for the script."""
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s[%(levelname)s]%(reset)s %(message)s",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "white",
                "WARNING": "light_yellow",
                "ERROR": "light_red",
                "CRITICAL": "light_red,bg_white",
            },
        )
    )
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    logging.getLogger("numexpr").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.category").setLevel(logging.WARNING)
