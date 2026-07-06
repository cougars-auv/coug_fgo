#!/bin/bash
optuna-dashboard "sqlite:///$(dirname "$0")/scalar_tuning.db" --host 0.0.0.0 --port 9000
