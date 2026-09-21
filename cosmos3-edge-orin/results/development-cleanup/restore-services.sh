#!/usr/bin/env bash
set -euo pipefail
sudo systemctl start iperf3.service
sudo systemctl start fwupd-refresh.timer
sudo systemctl start fwupd.service
sudo systemctl start kerneloops.service
