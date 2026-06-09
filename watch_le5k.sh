#!/usr/bin/env bash
# Live colored tail of run_le5k.py progress.
#   green  = YES        red  = NO     yellow = NEEDS_REVIEW
#   gray   = no domain found / blank verdict
#
# Usage:  watch_le5k.sh [logfile]
#         (defaults to console/le5k.log next to this script)

LOG="${1:-$(dirname "$0")/le5k.log}"

if [[ ! -f "$LOG" ]]; then
    echo "Log not found: $LOG" >&2
    exit 1
fi

ARROW=$'\xe2\x86\x92'   # " → "

# stream: backfill all existing lines, then follow new ones
# stdbuf -oL forces line-buffering through every stage so colored output
# appears immediately even when stdout is piped to a terminal under mawk.
stdbuf -oL tail -n +1 -F "$LOG" \
  | stdbuf -oL grep -E "verdict=" \
  | stdbuf -oL awk -v arrow="$ARROW" '
      {
          line = $0
          sub(/^\[[0-9]+\/[0-9]+\] +/, "", line)
          ap = index(line, " " arrow " ")
          if (ap == 0) next
          le   = substr(line, 1, ap - 1);              sub(/ +$/, "", le)
          rest = substr(line, ap + length(arrow) + 2)
          vp   = index(rest, "verdict=")
          if (vp == 0) next
          dom  = substr(rest, 1, vp - 1);              sub(/ +$/, "", dom)
          tail = substr(rest, vp + 8)
          split(tail, p, " "); v = p[1]

          if      (v == "YES")           c = "\033[1;32m"
          else if (v == "NO")            c = "\033[1;31m"
          else if (v == "NEEDS_REVIEW")  c = "\033[1;33m"
          else if (v == "ERROR")         c = "\033[1;91m"
          else                           c = "\033[90m"

          printf "%-50s  %-32s  %s%s\033[0m\n", le, dom, c, v
          fflush()
      }'
