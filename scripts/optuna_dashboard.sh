#!/bin/bash
optuna-dashboard "sqlite:///$(dirname "$0")/optuna_fgo.db" --host 0.0.0.0 --port 9000
