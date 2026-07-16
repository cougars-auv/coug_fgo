#!/bin/bash
optuna-dashboard "sqlite:///$(dirname "$0")/tuning.db" --host 0.0.0.0 --port 9000
