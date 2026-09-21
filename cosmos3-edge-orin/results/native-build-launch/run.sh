#!/bin/bash
cd /home/jetson/cosmos-edge
bash scripts/build_backend.sh fp16 > results/native-build-launch/output.log 2>&1
rc=$?
printf '%s\n' "$rc" > results/native-build-launch/exit-code
exit "$rc"
