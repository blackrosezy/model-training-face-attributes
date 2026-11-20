#!/usr/bin/env bash
# Launch the labeling GUI. Usage: ./run_gui.sh
set -e
cd "$(dirname "$(readlink -f "$0")")"
source /home/cribpc/py-env/bin/activate
exec python tools/gui.py "$@"
