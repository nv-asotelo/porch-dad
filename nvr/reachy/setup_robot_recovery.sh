#!/usr/bin/env bash
# Let porch-dad restart the Reachy Mini's daemon, and nothing else, when the daemon wedges.
#
# Run ON THE ORIN, once per robot (and again after the robot's OS is reinstalled):
#
#   nvr/reachy/setup_robot_recovery.sh [pollen@192.168.6.162]
#
# Why this exists. Daemon 1.11 leaks sockets on every WebRTC session (TURN refreshes libnice never
# closes) and runs under the default soft limit of 1024 descriptors. Once it reaches the limit
# ("Too many open files") every new viewer fails, each failed setup stops the robot's camera
# pipeline, and neither the daemon's restart API nor a media release helps: both run inside the
# same process. Measured 2026-09-24: 1014 of 1024 descriptors in use, 778 of them sockets. Only a
# new process recovers, and a person was needed to power-cycle the robot.
#
# What it changes:
#   Orin   /home/orin/.ssh/reachy_recover_ed25519 (created if missing), the robot's host key in
#          known_hosts.
#   Robot  one line in ~pollen/.ssh/authorized_keys for that key, with a forced command: whatever the
#          client asks for, the key can only run `sudo -n systemctl restart reachy-mini-daemon`, with
#          no pty and no forwarding. pollen already has passwordless sudo on the stock image.
#          /etc/systemd/system/reachy-mini-daemon.service.d/porch-dad-nofile.conf raising the
#          daemon's descriptor limit to 16384, so the leak takes 16x longer to wedge it. Takes effect
#          at the daemon's next restart.
#
# It prompts for pollen's password once (the stock image uses "root") and never stores it.
set -euo pipefail

ROBOT="${1:-pollen@192.168.6.162}"
KEY=/home/orin/.ssh/reachy_recover_ed25519
UNIT=reachy-mini-daemon.service
NOFILE=16384

[ -f "$KEY" ] || ssh-keygen -q -t ed25519 -N "" -C "porch-dad-reachy-recover" -f "$KEY"
PUB=$(cat "$KEY.pub")
ENTRY="command=\"sudo -n /usr/bin/systemctl restart $UNIT\",no-pty,no-port-forwarding,no-X11-forwarding,no-agent-forwarding $PUB"

read -rsp "password for $ROBOT: " SSHPASS; echo
export SSHPASS
sshpass -e ssh -o StrictHostKeyChecking=accept-new -o PubkeyAuthentication=no "$ROBOT" bash -s <<REMOTE
set -euo pipefail
install -d -m 700 ~/.ssh
touch ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
grep -qF "$PUB" ~/.ssh/authorized_keys || echo '$ENTRY' >> ~/.ssh/authorized_keys
sudo -n install -d /etc/systemd/system/$UNIT.d
printf '[Service]\n# porch-dad: the daemon leaks sockets per WebRTC session; 1024 wedged it (see\n# nvr/reachy/setup_robot_recovery.sh in porch-dad).\nLimitNOFILE=$NOFILE\n' \
  | sudo -n tee /etc/systemd/system/$UNIT.d/porch-dad-nofile.conf >/dev/null
sudo -n systemctl daemon-reload
echo "robot: key installed, LimitNOFILE=$NOFILE staged (applies at the next daemon restart)"
REMOTE
unset SSHPASS
echo "orin: key $KEY. Test (restarts the robot daemon):  ssh -i $KEY $ROBOT"
