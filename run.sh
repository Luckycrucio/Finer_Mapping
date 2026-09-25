#!/usr/bin/env bash
# Convenience wrapper: activate the pipeline's venv and run the full build.
# Any extra arguments are passed through to build_finer_map.py, e.g.:
#   ./run.sh coverage1 --voxel-size 0.02 --skip-frame-cache
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source .venv/bin/activate
python3 build_finer_map.py "$@"
