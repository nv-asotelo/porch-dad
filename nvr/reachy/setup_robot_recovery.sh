#!/usr/bin/env bash
# Let porch-dad restart the Reachy Mini's daemon, and nothing else, when the daemon wedges.
#
# Run ON THE ORIN, once per robot (and again after the robot's OS is reinstalled):
#
#   nvr/reachy/setup_robot_recovery.sh [pollen@192.168.6.162] [orin-address-the-robot-sees]
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
#   Robot  one line in ~pollen/.ssh/authorized_keys for that key: `restrict` (no pty, forwarding,
#          agent, X11 or user rc), `from=` the Orin's address only, and a forced command - whatever
#          the client asks for, the key runs `sudo -n systemctl restart --no-block
#          reachy-mini-daemon` and nothing else. --no-block returns once systemd has queued it; the
#          bridge then watches for the new daemon pid. pollen has passwordless sudo on the stock
#          image. Re-running replaces this line (matched by its comment), so it also repairs it.
#          /etc/systemd/system/reachy-mini-daemon.service.d/porch-dad-nofile.conf raising the
#          daemon's descriptor limit to 16384, so the leak takes 16x longer to wedge it. Takes effect
#          at the daemon's next restart.
#
# It prompts for pollen's password once (the stock image uses "root") and never stores it.
#
# After reinstalling the robot's OS its host key changes, and both this script (accept-new) and the
# bridge (StrictHostKeyChecking=yes) will refuse it - as they should. Read the new fingerprint at
# the robot itself (`ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub`), then on the Orin run
# `ssh-keygen -R 192.168.6.162`, re-run this script and compare the fingerprint it shows first.
set -euo pipefail

ROBOT="${1:-pollen@192.168.6.162}"
# The address the robot sees this Orin connect from, for the key's from= restriction.
FROM="${2:-$(ip -4 route get "${ROBOT#*@}" | sed -n 's/.* src \([0-9.]*\).*/\1/p')}"
[ -n "$FROM" ] || { echo "cannot tell which address reaches ${ROBOT#*@}; pass it as argument 2" >&2; exit 1; }
TAG=porch-dad-reachy-recover
KEY=/home/orin/.ssh/reachy_recover_ed25519
UNIT=reachy-mini-daemon.service
NOFILE=16384

[ -f "$KEY" ] || ssh-keygen -q -t ed25519 -N "" -C "$TAG" -f "$KEY"
PUB=$(cat "$KEY.pub")
ENTRY="restrict,from=\"$FROM\",command=\"sudo -n /usr/bin/systemctl restart --no-block $UNIT\" $PUB"

read -rsp "password for $ROBOT: " SSHPASS; echo
export SSHPASS
sshpass -e ssh -o StrictHostKeyChecking=accept-new -o PubkeyAuthentication=no "$ROBOT" bash -s <<REMOTE
set -euo pipefail
install -d -m 700 ~/.ssh
ak=~/.ssh/authorized_keys
touch "\$ak"
# A file without a final newline would glue the new entry onto the last key, where sshd reads it as
# that key's comment and never authorizes it. Terminate it first.
[ -s "\$ak" ] && [ -n "\$(tail -c1 "\$ak")" ] && echo >> "\$ak"
# Rewrite rather than append: replaces an earlier porch-dad line (old options or an old key), so a
# re-run repairs it. Only lines that START with porch-dad's own options and end in its tag go: a
# line where our entry was once glued onto someone else's key starts with that key, and stays.
tmp=\$(mktemp ~/.ssh/ak.XXXXXX)
grep -vE "^(restrict,|command=).* $TAG\$" "\$ak" > "\$tmp" || true
printf '%s\n' '$ENTRY' >> "\$tmp"
chmod 600 "\$tmp" && mv "\$tmp" "\$ak"
sudo -n install -d /etc/systemd/system/$UNIT.d
printf '[Service]\n# porch-dad: the daemon leaks sockets per WebRTC session; 1024 wedged it (see\n# nvr/reachy/setup_robot_recovery.sh in porch-dad).\nLimitNOFILE=$NOFILE\n' \
  | sudo -n tee /etc/systemd/system/$UNIT.d/porch-dad-nofile.conf >/dev/null
sudo -n systemctl daemon-reload
echo "robot: key installed (from=$FROM), LimitNOFILE=$NOFILE staged (applies at the next daemon restart)"
REMOTE
unset SSHPASS
echo "orin: key $KEY. Test (restarts the robot daemon):  ssh -i $KEY $ROBOT"
