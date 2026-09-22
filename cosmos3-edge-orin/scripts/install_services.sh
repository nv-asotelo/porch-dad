#!/usr/bin/env bash
# Install units for a configured device. Never start services or change running clocks/power.
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
service_user="${SUDO_USER:-}"
env_file=""
dry_run=0
usage() {
  printf '%s\n' 'Usage: sudo bash scripts/install_services.sh [--user ACCOUNT] [--env-file ABSOLUTE_PATH] [--dry-run]' \
    'Account defaults to SUDO_USER and must be an existing non-root account.' \
    'Uses deployment/local.env when present, otherwise frozen deployment/selected.env.' \
    'Installs/enables backend and loopback UI units; no services are started.' \
    'Only literal COSMOS_STATIC_CLOCKS=1 installs a future static-clock dependency.' \
    '--dry-run validates and prints units without writing systemd files; works without sudo.' \
    'An existing LAN drop-in is preserved: reinstalling does not remove HTTP/HTTPS exposure.'
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --user|--env-file)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      if [[ "$1" == --user ]]; then service_user="$2"; else env_file="$2"; fi
      shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done
if [[ "$dry_run" == 0 ]]; then
  [[ $EUID == 0 && $(uname -m) == aarch64 ]] || { echo 'Run as root on the Orin.' >&2; exit 2; }
fi
validation_args=(--project-dir "$project_dir" --user "$service_user")
if [[ -n "$env_file" ]]; then validation_args+=(--env-file "$env_file"); fi
validated="$(/usr/bin/python3 "$project_dir/scripts/configure_deployment.py" _validate "${validation_args[@]}")"
IFS=$'\t' read -r service_user service_group env_file static_clocks ignored_ip <<< "$validated"
for script in run_selected_backend.sh run_backend.sh rtn_backend.py serve_backend.py cosmos_runtime.py serve_ui.py \
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
emit_unit() {
  case "$1" in
    cosmos-edge-clocks)
      cat <<EOF
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
      ;;
    cosmos-edge-backend)
      cat <<EOF
[Unit]
Description=Cosmos3-Edge TensorRT backend on Orin
After=network.target$clock_after
$clock_requires
StartLimitIntervalSec=300
StartLimitBurst=3

[Service]
Type=simple
User=$service_user
Group=$service_group
WorkingDirectory=$project_dir
EnvironmentFile=$env_file
ExecStart=/usr/bin/bash $project_dir/scripts/run_selected_backend.sh
Restart=on-failure
RestartSec=10
TimeoutStopSec=30
KillMode=control-group

[Install]
WantedBy=multi-user.target
EOF
      ;;
    cosmos-edge-ui)
      cat <<EOF
[Unit]
Description=Cosmos3-Edge streaming browser interface
After=network.target cosmos-edge-backend.service
Wants=cosmos-edge-backend.service
StartLimitIntervalSec=300
StartLimitBurst=3

[Service]
Type=simple
User=$service_user
Group=$service_group
WorkingDirectory=$project_dir
ExecStart=/usr/bin/python3 $project_dir/scripts/serve_ui.py --host 127.0.0.1 --port 8090
Restart=on-failure
RestartSec=5
TimeoutStopSec=10

[Install]
WantedBy=multi-user.target
EOF
      ;;
  esac
}
if [[ -f /etc/systemd/system/cosmos-edge-ui.service.d/lan.conf ]]; then
  printf '%s\n' 'Existing LAN drop-in retained; HTTP/HTTPS exposure remains configured.' >&2
fi
if [[ "$dry_run" == 1 ]]; then
  for service in "${services[@]}"; do
    printf '# %s.service\n' "$service"
    emit_unit "$service"
  done
  exit 0
fi
# Validate all staged units before changing any installed unit.
staging_dir="$(mktemp -d)"
trap 'rm -rf "$staging_dir"' EXIT
unit_paths=()
unit_names=()
for service in "${services[@]}"; do
  emit_unit "$service" > "$staging_dir/$service.service"
  unit_paths+=("$staging_dir/$service.service")
  unit_names+=("$service.service")
done
systemd-analyze verify "${unit_paths[@]}"
install -d -m 755 "$project_dir/results/service-install"
for service in "${services[@]}"; do
  if [[ -f "/etc/systemd/system/$service.service" ]]; then
    cp -a "/etc/systemd/system/$service.service" "$project_dir/results/service-install/$service.before.$(date -u +%Y%m%dT%H%M%SZ)"
  fi
  install -m 644 "$staging_dir/$service.service" "/etc/systemd/system/$service.service"
done
systemctl daemon-reload
if [[ "$static_clocks" == 0 && -f /etc/systemd/system/cosmos-edge-clocks.service ]]; then
  systemctl disable cosmos-edge-clocks.service
fi
systemctl enable "${unit_names[@]}"
printf '%s\n' 'Installed and enabled. No services were started and current clocks/power were not changed.' \
  'Fresh UI access uses an SSH tunnel to 127.0.0.1:8090. Existing LAN drop-ins remain in effect.'
if [[ "$static_clocks" == 1 ]]; then
  printf '%s\n' 'Selected static clocks apply only when the clocks/backend service is explicitly started.'
fi
