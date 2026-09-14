#!/usr/bin/env bash
# Quiesce the board down to just the Live VLM WebUI, or put the full stack back.
#
# Why this exists: the Orin Nano has 8 GB and 6 cores, and the NVR stack (Frigate CPU detection,
# ring-mqtt, Home Assistant) uses most of both. With everything running, memory sat at 6.73 GB used
# with 0.80 GB available and load average 7.56 on 6 cores. Demoing the WebUI in that state measures
# the NVR, not the model.
#
#   ./demo-mode.sh on     stop everything the WebUI does not need
#   ./demo-mode.sh off    bring the full stack back
#   ./demo-mode.sh status what is up right now
#
# The WebUI needs exactly two units: cosmos3-edge-shim.service (holds the engine) and
# live-vlm-webui.service. Everything below is stopped in `on` and restored in `off`.
set -uo pipefail

CONTAINERS=(frigate ring-mqtt homeassistant mosquitto)
SERVICES=(frigate-notify porch-feed)
# Desktop/remote-access units. Only worth stopping when the recording is driven from another
# machine's browser - if you are recording ON the Orin's own desktop, stopping these kills it.
DESKTOP=(x11vnc gnome-remote-desktop jetson-oled)
# Not required by anything here; they just add noise to a CPU measurement.
NOISE=(iperf3 kerneloops fwupd)
KEEP=(cosmos3-edge-shim live-vlm-webui)

usage() { sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }

mem() { free -m | awk '/^Mem:/ {printf "%.2f GB used of %.2f GB, %.2f GB available\n", $3/1024, $2/1024, $7/1024}'; }

case "${1:-}" in
on)
  echo "== stopping non-essential =="
  sudo -n systemctl stop "${SERVICES[@]}" 2>/dev/null
  for s in "${SERVICES[@]}"; do printf '  service  %-22s %s\n' "$s" "$(systemctl is-active "$s")"; done

  for c in "${CONTAINERS[@]}"; do
    docker stop -t 15 "$c" >/dev/null 2>&1 && printf '  container %-21s stopped\n' "$c"
  done

  # Keep the desktop stack unless the caller says the recording is remote.
  if [[ "${2:-}" == "--remote-recording" ]]; then
    sudo -n systemctl stop "${DESKTOP[@]}" 2>/dev/null
    echo "  stopped desktop/VNC units (recording is remote)"
  else
    echo "  KEPT desktop/VNC units - pass --remote-recording to stop them too"
  fi
  sudo -n systemctl stop "${NOISE[@]}" 2>/dev/null

  echo "== required, still up =="
  for s in "${KEEP[@]}"; do printf '  %-22s %s\n' "$s" "$(systemctl is-active "$s")"; done
  echo "== memory =="; mem
  echo "Give the board ~45 s to settle before trusting a measurement."
  ;;

off)
  echo "== restoring full stack =="
  # mosquitto first: frigate and the notifier both publish to it.
  for c in mosquitto ring-mqtt frigate homeassistant; do
    docker start "$c" >/dev/null 2>&1 && printf '  container %-21s started\n' "$c"
  done
  sudo -n systemctl start "${DESKTOP[@]}" "${SERVICES[@]}" 2>/dev/null
  for s in "${SERVICES[@]}" "${DESKTOP[@]}"; do
    printf '  service  %-22s %s\n' "$s" "$(systemctl is-active "$s")"
  done
  echo "== memory =="; mem
  echo "Frigate takes ~60 s to reconnect every camera."
  ;;

status)
  echo "== required for the WebUI =="
  for s in "${KEEP[@]}"; do printf '  %-22s %s\n' "$s" "$(systemctl is-active "$s")"; done
  echo "== optional =="
  for s in "${SERVICES[@]}" "${DESKTOP[@]}"; do printf '  %-22s %s\n' "$s" "$(systemctl is-active "$s")"; done
  echo "== containers =="
  docker ps -a --filter "name=frigate" --filter "name=ring-mqtt" --filter "name=mosquitto" \
    --filter "name=homeassistant" --format '  {{.Names}}  {{.Status}}'
  echo "== memory =="; mem
  echo "== load ==";  uptime
  ;;

*) usage ;;
esac
