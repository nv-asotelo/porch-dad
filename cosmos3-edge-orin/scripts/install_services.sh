#!/usr/bin/env bash
# Install the selected Orin deployment, with explicitly selected static clocks.
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == --help ]]; then
  printf '%s\n' 'Usage: sudo bash scripts/install_services.sh' \
    'Requires deployment/selected.env; backend validates its selected cache on startup.' \
    'Installs cosmos-edge-backend and cosmos-edge-ui systemd services.' \
    'Only exact COSMOS_STATIC_CLOCKS=1 adds a root oneshot jetson_clocks service.' \
    'Unset or COSMOS_STATIC_CLOCKS=0 leaves clocks unchanged; no services are started.' \
    'To restore dynamic clocks: set COSMOS_STATIC_CLOCKS=0 and rerun this installer,' \
    'then sudo systemctl disable --now cosmos-edge-clocks.service and run:' \
    'sudo /usr/bin/jetson_clocks --restore /home/jetson/cosmos-edge/data/power/stock25w.before.conf'
  exit 0
fi
[[ $EUID == 0 && $(uname -m) == aarch64 ]] || { echo 'Run as root on the Orin.' >&2; exit 2; }
[[ "$project_dir" == /home/jetson/cosmos-edge ]] || { echo 'Unexpected deployment path.' >&2; exit 2; }
[[ -f "$project_dir/deployment/selected.env" ]] || { echo 'Select and record a validated engine profile first.' >&2; exit 2; }
# Read only this literal option. Never execute the selected environment as root.
static_clocks="$(/usr/bin/python3 - "$project_dir/deployment/selected.env" <<'PY'
from pathlib import Path
import re
import sys
values = []
for line in Path(sys.argv[1]).read_text().splitlines():
    stripped = line.strip()
    if not stripped or stripped.startswith(('#', ';')):
        continue
    if re.match(r'(?:export\s+)?COSMOS_STATIC_CLOCKS(?:\s|=|$)', stripped):
        if line not in ('COSMOS_STATIC_CLOCKS=0', 'COSMOS_STATIC_CLOCKS=1'):
            raise SystemExit('Use the exact unquoted line COSMOS_STATIC_CLOCKS=0 or COSMOS_STATIC_CLOCKS=1')
        values.append(line[-1])
if len(values) > 1:
    raise SystemExit('Duplicate COSMOS_STATIC_CLOCKS assignments are not supported')
print(values[0] if values else '0')
PY
)"
for script in run_selected_backend.sh run_backend.sh rtn_backend.py serve_backend.py serve_ui.py \
  preflight_cosmos_artifacts.py repair_cosmos_runtime_config.py repair_cosmos_chat_template.py build_model_cache.py; do
  [[ -f "$project_dir/scripts/$script" ]] || { printf 'Missing deployment script: %s\n' "$script" >&2; exit 2; }
done
[[ -x "$project_dir/external/TensorRT-Edge-LLM/.venv/bin/python" ]] || { echo 'Missing backend Python environment.' >&2; exit 2; }
services=(cosmos-edge-backend cosmos-edge-ui)
clock_after=""
clock_requires=""
if [[ "$static_clocks" == 1 ]]; then
  [[ -x /usr/bin/jetson_clocks ]] || { echo 'Missing /usr/bin/jetson_clocks.' >&2; exit 2; }
  services+=(cosmos-edge-clocks)
  clock_after=" cosmos-edge-clocks.service"
  clock_requires="Requires=cosmos-edge-clocks.service"
fi
install -d -m 755 "$project_dir/results/service-install"
for service in "${services[@]}"; do
  if [[ -f "/etc/systemd/system/$service.service" ]]; then
    cp -a "/etc/systemd/system/$service.service" "$project_dir/results/service-install/$service.before.$(date -u +%Y%m%dT%H%M%SZ)"
  fi
done
if [[ "$static_clocks" == 1 ]]; then
  cat > /etc/systemd/system/cosmos-edge-clocks.service <<EOF
[Unit]
Description=Task-selected static clocks within the existing Orin power mode
After=nvpmodel.service
Before=cosmos-edge-backend.service

[Service]
Type=oneshot
User=root
ExecStart=/usr/bin/jetson_clocks
RemainAfterExit=yes
TimeoutStartSec=30

[Install]
WantedBy=multi-user.target
EOF
fi
cat > /etc/systemd/system/cosmos-edge-backend.service <<EOF
[Unit]
Description=Cosmos3-Edge TensorRT backend on Orin
After=network.target$clock_after
$clock_requires
StartLimitIntervalSec=300
StartLimitBurst=3

[Service]
Type=simple
User=jetson
Group=jetson
WorkingDirectory=$project_dir
EnvironmentFile=$project_dir/deployment/selected.env
ExecStart=/usr/bin/bash $project_dir/scripts/run_selected_backend.sh
Restart=on-failure
RestartSec=10
TimeoutStopSec=30
KillMode=control-group

[Install]
WantedBy=multi-user.target
EOF
cat > /etc/systemd/system/cosmos-edge-ui.service <<EOF
[Unit]
Description=Cosmos3-Edge streaming browser interface
After=network.target cosmos-edge-backend.service
Wants=cosmos-edge-backend.service
StartLimitIntervalSec=300
StartLimitBurst=3

[Service]
Type=simple
User=jetson
Group=jetson
WorkingDirectory=$project_dir
ExecStart=/usr/bin/python3 $project_dir/scripts/serve_ui.py --host 127.0.0.1 --port 8090
Restart=on-failure
RestartSec=5
TimeoutStopSec=10

[Install]
WantedBy=multi-user.target
EOF
unit_paths=()
unit_names=()
for service in "${services[@]}"; do
  unit_paths+=("/etc/systemd/system/$service.service")
  unit_names+=("$service.service")
done
systemd-analyze verify "${unit_paths[@]}"
systemctl daemon-reload
if [[ "$static_clocks" == 0 && -f /etc/systemd/system/cosmos-edge-clocks.service ]]; then
  # Remove an earlier task-owned boot activation without stopping services or
  # silently changing the current clocks. Restore the saved state separately.
  systemctl disable cosmos-edge-clocks.service
fi
systemctl enable "${unit_names[@]}"
printf '%s\n' 'Installed and enabled. Stop the owned foreground servers before starting these units.'
if [[ "$static_clocks" == 1 ]]; then
  printf '%s\n' 'Selected static clocks will apply when the clocks/backend service is started; the current power mode and fan settings are unchanged by this installer.'
fi
printf '%s\n' 'Dynamic-clock restoration requires COSMOS_STATIC_CLOCKS=0 and reinstalling units, then disabling the clocks unit:' \
  'sudo systemctl disable --now cosmos-edge-clocks.service' \
  'sudo /usr/bin/jetson_clocks --restore /home/jetson/cosmos-edge/data/power/stock25w.before.conf'
