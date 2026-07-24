#!/usr/bin/env bash
# Run the pick-and-place proof of concept on the robot.
set -euo pipefail
# .env and venv live at the repo root, one level up from this script
cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
    echo "Missing .env file (needs VIAM_ADDRESS, VIAM_API_KEY, VIAM_API_KEY_ID)" >&2
    exit 1
fi

set -a
source .env
set +a

exec venv/bin/python pickplace/demo.py
