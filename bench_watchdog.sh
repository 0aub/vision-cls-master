#!/bin/bash
# Keeps the GPU fed. Unlike v1 this NEVER exits on completion - when a queue
# finishes it starts the next one, and if none is left it re-runs the last one
# (every step self-skips, so a no-op pass is cheap) and reports idleness loudly.
set -uo pipefail
cd "$(dirname "$0")"
QUEUES=(bench_queue5.sh bench_queue4.sh bench_queue2.sh bench_queue3.sh bench_queue.sh)
while true; do
  if pgrep -f "[b]ench_queue[0-9]*\.sh" >/dev/null; then sleep 60; continue; fi
  started=""
  for q in "${QUEUES[@]}"; do
    [ -x "$q" ] || continue
    tag="$(basename "$q" .sh | tr 'a-z' 'A-Z') COMPLETE"
    if ! grep -q "$tag" log/bench-progress.txt 2>/dev/null; then
      echo "[watchdog] starting $q  $(date -Iseconds)"
      nohup "./$q" >> "log/$(basename "$q" .sh).log" 2>&1 &
      started="$q"; sleep 120; break
    fi
  done
  if [ -z "$started" ]; then
    echo "[watchdog] ALL QUEUES COMPLETE - GPU IDLE  $(date -Iseconds)"
    echo "IDLE: all queued work finished $(date -Iseconds)" >> log/bench-progress.txt
    sleep 300
  fi
done
