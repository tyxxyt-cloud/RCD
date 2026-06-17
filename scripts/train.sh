#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/default.yaml}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

python3 -m trdmu.data.build_dataset --config "${CONFIG_PATH}"
python3 -m trdmu.training.train --config "${CONFIG_PATH}"
