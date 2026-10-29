#!/usr/bin/env bash
# The four runs and the video. Panoramas must exist (see README, step 2).
# Usage: ./run_all.sh [--agent scripted] [--model MODEL] [--max-fix N]
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
EXTRA=("$@")
for q in board chairs; do
  pano=$([ "$q" = board ] && echo panos/near_board || echo panos/back_of_room)
  for m in 2d 3d; do
    "$PY" run_agent.py --pano "$pano" --question "$q" --mode "$m" --out "runs/${q}_${m}" "${EXTRA[@]}"
  done
done
"$PY" make_video.py --runs runs/board_2d runs/board_3d runs/chairs_2d runs/chairs_3d --still stills
"$PY" make_video.py --runs runs/board_2d runs/board_3d runs/chairs_2d runs/chairs_3d --out yarbus.mp4
