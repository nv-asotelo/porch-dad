#!/bin/bash
# Point-in-time snapshot of the porch-dad Orin onto an SD card in the Orin's own slot.
#
# Run this from the workstation. The workstation only ever sends commands: every byte
# copied moves NVMe -> SD inside the Orin, never across the LAN. That is deliberate.
# The Orin currently hangs off an Eero mesh node whose backhaul is wireless, so it
# sustains roughly 1 MB/s to anywhere; pulling 147 GB through it would take over a day,
# while the same copy on-device runs at ~49 MB/s. When the Orin eventually moves to the
# garage switch this script does not change - the slow link simply stops being a factor.
#
# Snapshots are hardlinked against the previous one (rsync --link-dest), so the first
# capsule costs ~147 GB and each later one costs only what actually changed.
#
#   ./time-capsule.sh                 # take a capsule
#   ./time-capsule.sh --init          # format a fresh card as a capsule target first
#   ./time-capsule.sh --keep 5        # prune to the newest 5 capsules when done
#   ./time-capsule.sh --no-quiesce    # do not stop services (faster, less consistent)
#
set -euo pipefail

JETSON="${JETSON_HOST:-orin@jetson.local}"
LABEL="porchdad-capsule"      # the card is found by filesystem label, not by /dev path,
                              # because mmcblk numbering is not guaranteed across boots.
MOUNT=/mnt/capsule
KEEP=0
QUIESCE=1
INIT=0

while [ $# -gt 0 ]; do
  case "$1" in
    --init)       INIT=1 ;;
    --keep)       KEEP="${2:?--keep needs a number}"; shift ;;
    --no-quiesce) QUIESCE=0 ;;
    -h|--help)    sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

say() { printf '\n== %s\n' "$*"; }

say "target: $JETSON"
ssh -o BatchMode=yes "$JETSON" true || { echo "cannot reach $JETSON"; exit 1; }

# ---------------------------------------------------------------- init
if [ "$INIT" = 1 ]; then
  say "INIT: formatting a card as '$LABEL'"
  cat <<'WARN'
  This ERASES the SD card in the Orin. It refuses to run if the card holds a
  Jetson boot chain (an APP partition), so it cannot eat a bootable clone by
  accident. Type the card size in GB shown below to confirm.
WARN
  ssh -o BatchMode=yes "$JETSON" '
    set -e
    [ -b /dev/mmcblk0 ] || { echo "no SD card present"; exit 1; }
    if lsblk -no PARTLABEL /dev/mmcblk0 | grep -qx "APP"; then
      echo "REFUSING: this card carries a Jetson APP partition (bootable clone)."
      exit 1
    fi
    lsblk -dno SIZE /dev/mmcblk0
  '
  read -rp "  card size to confirm: " _c
  ssh -o BatchMode=yes "$JETSON" "
    set -e
    sudo umount ${MOUNT} 2>/dev/null || true
    sudo wipefs -a /dev/mmcblk0
    sudo parted -s /dev/mmcblk0 mklabel gpt
    sudo parted -s /dev/mmcblk0 mkpart primary ext4 1MiB 100%
    sudo partprobe /dev/mmcblk0 || true; sleep 3
    sudo mkfs.ext4 -F -q -L ${LABEL} -m 0 /dev/mmcblk0p1
    echo '  formatted'
  "
fi

# ---------------------------------------------------------------- mount
say "mounting the capsule card"
ssh -o BatchMode=yes "$JETSON" "
  set -e
  DEV=\$(blkid -L ${LABEL} || true)
  [ -n \"\$DEV\" ] || { echo 'no card labelled ${LABEL} found - run with --init'; exit 1; }
  sudo mkdir -p ${MOUNT}
  mountpoint -q ${MOUNT} || sudo mount \"\$DEV\" ${MOUNT}
  sudo mkdir -p ${MOUNT}/snapshots
  df -h ${MOUNT} | tail -1
"

STAMP="$(date +%Y%m%dT%H%M%S)"
say "capsule $STAMP"

# ---------------------------------------------------------------- copy
# The remote side runs as one script so the service restore trap is armed on the
# device itself: if the ssh connection drops mid-run, services still come back.
ssh -o BatchMode=yes "$JETSON" "MOUNT=${MOUNT} STAMP=${STAMP} QUIESCE=${QUIESCE} bash -s" <<'REMOTE'
set -u
DEST="$MOUNT/snapshots/$STAMP"
PREV="$(ls -1d "$MOUNT"/snapshots/*/ 2>/dev/null | sort | tail -1 || true)"

# Exactly what was running before we touched anything. Restarting a fixed list
# would silently "fix" deliberate state - porch-dad.service is installed but
# intentionally not enabled, and some containers are intentionally stopped.
RUNNING_UNITS="$(systemctl list-units --type=service --state=running --no-legend \
                 | awk '{print $1}' | grep -E 'cosmos|vlm|frigate-notify|reachy|porch' || true)"
RUNNING_CTRS="$(docker ps --format '{{.Names}}' || true)"

restore() {
  [ "$QUIESCE" = 1 ] || return 0
  echo "== restoring services"
  for c in mosquitto; do echo "$RUNNING_CTRS" | grep -qx "$c" && docker start "$c" >/dev/null 2>&1; done
  sleep 3
  for c in $RUNNING_CTRS; do [ "$c" = mosquitto ] || docker start "$c" >/dev/null 2>&1; done
  echo "$RUNNING_UNITS" | grep -q cosmos3-edge-shim && { sudo systemctl start cosmos3-edge-shim.service; sleep 5; }
  for u in $RUNNING_UNITS; do [ "$u" = cosmos3-edge-shim.service ] || sudo systemctl start "$u"; done
  echo "   restored $(echo "$RUNNING_UNITS" | wc -w) units, $(echo "$RUNNING_CTRS" | wc -w) containers"
}
trap restore EXIT

if [ "$QUIESCE" = 1 ]; then
  echo "== quiescing (sqlite databases must not be written mid-copy)"
  for u in $RUNNING_UNITS; do [ "$u" = cosmos3-edge-shim.service ] || sudo systemctl stop "$u"; done
  sudo systemctl stop cosmos3-edge-shim.service 2>/dev/null || true
  [ -n "$RUNNING_CTRS" ] && docker stop -t 30 $RUNNING_CTRS >/dev/null
fi

echo "== copying${PREV:+ (hardlinked against $(basename "$PREV"))}"
sudo mkdir -p "$DEST"
sudo rsync -aHAXx --numeric-ids --stats \
  ${PREV:+--link-dest="$PREV/rootfs"} \
  --exclude=/lost+found \
  / "$DEST/rootfs/" | tail -6

sudo mkdir -p "$DEST/meta"
{ echo "captured: $(date -Is)"
  echo "platform: $(head -1 /etc/nv_tegra_release)"
  echo "kernel:   $(uname -r)"
  echo "units:    $RUNNING_UNITS"
  echo "ctrs:     $RUNNING_CTRS"
} | sudo tee "$DEST/meta/MANIFEST.txt" >/dev/null
dpkg -l            | sudo tee "$DEST/meta/dpkg-full.txt"  >/dev/null
apt-mark showmanual| sudo tee "$DEST/meta/apt-manual.txt" >/dev/null

sudo rm -f "$MOUNT/latest"
sudo ln -s "snapshots/$STAMP" "$MOUNT/latest"
sync
echo "== capsule written: $DEST"
REMOTE

# ---------------------------------------------------------------- prune
if [ "$KEEP" -gt 0 ]; then
  say "pruning to newest $KEEP"
  ssh -o BatchMode=yes "$JETSON" "
    cd ${MOUNT}/snapshots || exit 0
    ls -1d */ | sort | head -n -${KEEP} | while read -r d; do
      echo \"  removing \$d\"; sudo rm -rf \"\$d\"
    done
  "
fi

say "done"
ssh -o BatchMode=yes "$JETSON" "df -h ${MOUNT} | tail -1; ls -1 ${MOUNT}/snapshots | tail -5"
