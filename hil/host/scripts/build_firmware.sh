#!/bin/bash
set -euo pipefail
ROOT='/mnt/cUSER_HOME/Desktop/HIL of PSOand NN'
MOUNT='/tmp/psonn_hil_mount'
mkdir -p "$MOUNT"
if ! mountpoint -q "$MOUNT"; then mount --bind "$ROOT" "$MOUNT"; fi
export PATH=/usr/bin:/usr/local/bin:$PATH
cd "$MOUNT/firmware/PX4-Autopilot"
make px4_fmu-v6c_default -j4