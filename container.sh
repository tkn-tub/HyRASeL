#!/bin/bash
# Runs `make <arguments>` in the Docker container started by the top-level Makefile.
# The command runs in the tmux session "main" (make attach) and its output is
# written to the log file (make logs). The container ends with the session.
#
# Usage: container.sh <log file> <make arguments...>

if [ "$1" = "--in-tmux" ]; then
    shift
    log=$1
    shift
    make "$@" 2>&1 | tee "$log"
    status=${PIPESTATUS[0]}
    echo "Finished: make $* (exit code $status)" | tee -a "$log"
    # Keep the container on failure, so that its tmux windows can be inspected
    if [ "$status" != 0 ]; then
        read -rp "Press Enter to close the container." _
    fi
    exit "$status"
fi

tmux new-session -d -s main -x 200 -y 50 "$0" --in-tmux "$@"
while tmux has-session -t main 2>/dev/null; do
    sleep 5
done
