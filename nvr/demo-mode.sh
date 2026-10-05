#!/usr/bin/env bash
# Quiesce the board down to just Live Vision, or put the full stack back.
#
# Why this exists: the Orin Nano has 8 GB and 6 cores, and the NVR stack (Frigate CPU detection,
# ring-mqtt, Home Assistant) uses most of both. With everything running, memory sat at 6.73 GB used
# with 0.80 GB available and load average 7.56 on 6 cores. Demoing a Live UI in that state measures
# the NVR, not the model - and a model that does not fit beside the NVR cannot be shown at all.
#
#   ./demo-mode.sh on     stop everything Live Vision does not need
#   ./demo-mode.sh off    bring back what `on` stopped
#   ./demo-mode.sh status what is up right now
#   ./demo-mode.sh json   the same, for Live Vision's Demo mode button
#
# It keeps four units: cosmos3-edge-shim (holds the engine), cosmos-edge-ui (Live Vision),
# reachy-mjpeg-bridge (its Reachy source) and pokemon-classifier (its classifiers). `on` records
# what it stopped in STATE, and `off` restores exactly that, so a container that was already down
# (Home Assistant is not in the boot set) is not started by leaving demo mode.
#
# One systemctl call per unit, so a sudoers file can allow exactly these command lines.
set -uo pipefail

CONTAINERS=(frigate ring-mqtt homeassistant mosquitto scout-bridge scout-bridge-first-floor scout-bridge-1f-wheeled)
SERVICES=(frigate-notify porch-feed porch-dad)
# Desktop/remote-access units. Only worth stopping when the recording is driven from another
# machine's browser - if you are recording ON the Orin's own desktop, stopping these kills it.
DESKTOP=(x11vnc gnome-remote-desktop jetson-oled)
# Not required by anything here; they just add noise to a CPU measurement.
NOISE=(iperf3 kerneloops fwupd)
KEEP=(cosmos3-edge-shim cosmos-edge-ui reachy-mjpeg-bridge pokemon-classifier)
STATE=${DEMO_MODE_STATE:-/home/orin/nvr/.demo-mode}

usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }

mem() { free -m | awk '/^Mem:/ {printf "%.2f GB used of %.2f GB, %.2f GB available\n", $3/1024, $2/1024, $7/1024}'; }
running() { [ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null)" = true ]; }

case "${1:-}" in
on)
  echo "== stopping non-essential =="
  stopped=()
  for s in "${SERVICES[@]}"; do
    if systemctl is-active -q "$s"; then
      sudo -n systemctl stop "$s" && stopped+=("service:$s")
    fi
    printf '  service  %-26s %s\n' "$s" "$(systemctl is-active "$s")"
  done
  for c in "${CONTAINERS[@]}"; do
    if running "$c"; then
      docker stop -t 15 "$c" >/dev/null 2>&1 && stopped+=("container:$c") && printf '  container %-25s stopped\n' "$c"
    fi
  done
  # Keep the desktop stack unless the caller says the recording is remote.
  if [[ "${2:-}" == "--remote-recording" ]]; then
    for s in "${DESKTOP[@]}"; do
      systemctl is-active -q "$s" && sudo -n systemctl stop "$s" && stopped+=("service:$s")
    done
    echo "  stopped desktop/VNC units (recording is remote)"
  else
    echo "  KEPT desktop/VNC units - pass --remote-recording to stop them too"
  fi
  for s in "${NOISE[@]}"; do
    systemctl is-active -q "$s" && sudo -n systemctl stop "$s" && stopped+=("service:$s")
  done
  # Appended, not overwritten: a second `on` must not forget what the first one stopped.
  printf '%s\n' "${stopped[@]}" >> "$STATE"
  echo "== required, still up =="
  for s in "${KEEP[@]}"; do printf '  %-26s %s\n' "$s" "$(systemctl is-active "$s")"; done
  echo "== memory =="; mem
  echo "Give the board ~45 s to settle before trusting a measurement."
  ;;

off)
  echo "== restoring =="
  if [ -s "$STATE" ]; then
    mapfile -t back < <(sort -u "$STATE")
  else
    # No record (demo mode entered by hand before STATE existed): bring back the whole stack.
    back=()
    for c in "${CONTAINERS[@]}"; do back+=("container:$c"); done
    for s in "${SERVICES[@]}" "${DESKTOP[@]}"; do back+=("service:$s"); done
  fi
  # mosquitto first: frigate and the notifier both publish to it.
  for c in mosquitto ring-mqtt frigate homeassistant scout-bridge scout-bridge-first-floor scout-bridge-1f-wheeled; do
    if printf '%s\n' "${back[@]}" | grep -qx "container:$c"; then
      docker start "$c" >/dev/null 2>&1 && printf '  container %-25s started\n' "$c"
    fi
  done
  # KEEP is started too, not just restored-alongside. A reboot between `on` and `off` leaves a KEEP
  # unit down if it is not enabled, and `on` never stopped it so `off` would never think to start
  # it. Measured that exact gap when KEEP was the WebUI: the board rebooted, every enabled unit came
  # back, and live-vlm-webui did not because it was `disabled`.
  for s in "${KEEP[@]}"; do sudo -n systemctl enable --now "$s" 2>/dev/null; done
  for b in "${back[@]}"; do
    [[ "$b" == service:* ]] && sudo -n systemctl start "${b#service:}" 2>/dev/null
  done
  rm -f "$STATE"
  for s in "${KEEP[@]}" "${SERVICES[@]}"; do
    printf '  service  %-26s %-9s (boot: %s)\n' "$s" "$(systemctl is-active "$s")" \
      "$(systemctl is-enabled "$s" 2>/dev/null)"
  done
  echo "== memory =="; mem
  echo "Frigate takes ~60 s to reconnect every camera."
  ;;

status)
  echo "== demo mode: $([ -e "$STATE" ] && echo on || echo off) =="
  echo "== required for Live Vision =="
  for s in "${KEEP[@]}"; do
    printf '  %-26s %-9s (boot: %s)\n' "$s" "$(systemctl is-active "$s")" \
      "$(systemctl is-enabled "$s" 2>/dev/null)"
  done
  echo "== optional =="
  for s in "${SERVICES[@]}" "${DESKTOP[@]}"; do printf '  %-26s %s\n' "$s" "$(systemctl is-active "$s")"; done
  echo "== containers =="
  for c in "${CONTAINERS[@]}"; do printf '  %-26s %s\n' "$c" "$(running "$c" && echo running || echo stopped)"; done
  echo "== memory =="; mem
  echo "== load ==";  uptime
  ;;

json)
  up=() down=()
  for c in "${CONTAINERS[@]}"; do if running "$c"; then up+=("$c"); else down+=("$c"); fi; done
  for s in "${SERVICES[@]}"; do if systemctl is-active -q "$s"; then up+=("$s"); else down+=("$s"); fi; done
  list() { local IFS=,; local out=(); for x in "$@"; do out+=("\"$x\""); done; echo "[${out[*]}]"; }
  avail=$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo)
  printf '{"on": %s, "running": %s, "stopped": %s, "mem_available_mb": %s}\n' \
    "$([ -e "$STATE" ] && echo true || echo false)" "$(list "${up[@]}")" "$(list "${down[@]}")" "$avail"
  ;;

*) usage ;;
esac
