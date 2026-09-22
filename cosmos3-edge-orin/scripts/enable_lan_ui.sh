#!/usr/bin/env bash
# Explicit LAN opt-in. Inference stays on loopback; no running service is restarted here.
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
service_user="${SUDO_USER:-}"
env_file=""
lan_ip=""
dry_run=0
usage() {
  printf '%s\n' 'Usage: sudo bash scripts/enable_lan_ui.sh [--user ACCOUNT] [--env-file ABSOLUTE_PATH] [--dry-run] DEVICE_IPV4' \
    'Account defaults to SUDO_USER; uses deployment/local.env, otherwise selected.env.' \
    'IPv4 must actually belong to this device. Requires previously installed UI unit.' \
    'Explicitly configures HTTP :8090 and HTTPS :8443 on all interfaces.' \
    'Always creates a new local certificate/key; never reuses another device certificate.' \
    'Does not start/restart services. Use an SSH tunnel for default loopback-only access.'
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --user|--env-file)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      if [[ "$1" == --user ]]; then service_user="$2"; else env_file="$2"; fi
      shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    --help|-h) usage; exit 0 ;;
    --*) usage >&2; exit 2 ;;
    *) [[ -z "$lan_ip" ]] || { usage >&2; exit 2; }; lan_ip="$1"; shift ;;
  esac
done
[[ -n "$lan_ip" ]] || { usage >&2; exit 2; }
if [[ "$dry_run" == 0 ]]; then
  [[ $EUID == 0 && $(uname -m) == aarch64 ]] || { echo 'Run as root on the Orin.' >&2; exit 2; }
fi
validation_args=(--project-dir "$project_dir" --user "$service_user" --lan-ip "$lan_ip")
if [[ -n "$env_file" ]]; then validation_args+=(--env-file "$env_file"); fi
validated="$(/usr/bin/python3 "$project_dir/scripts/configure_deployment.py" _validate "${validation_args[@]}")"
IFS=$'\t' read -r service_user service_group env_file static_clocks lan_ip <<< "$validated"
[[ -f /etc/systemd/system/cosmos-edge-ui.service ]] || { echo 'Install cosmos-edge-ui.service first.' >&2; exit 2; }
# The drop-in inherits User/Group from the base service; prevent a mismatched certificate owner.
/usr/bin/python3 - /etc/systemd/system/cosmos-edge-ui.service "$service_user" "$service_group" <<'PY'
import configparser
import sys
unit = configparser.ConfigParser(interpolation=None, strict=False)
unit.read(sys.argv[1])
if unit.get('Service', 'User', fallback='root') != sys.argv[2] or unit.get('Service', 'Group', fallback='') != sys.argv[3]:
    raise SystemExit('Installed UI account differs; rerun install_services.sh with the same --user first')
PY
tls_dir="$project_dir/deployment/tls"
if [[ -L "$tls_dir" || -L "$tls_dir/orin.key" || -L "$tls_dir/orin.crt" ]]; then
  echo 'TLS directory and certificate files must not be symbolic links.' >&2; exit 2
fi
emit_dropin() {
  cat <<EOF
[Service]
ExecStart=
ExecStart=/usr/bin/python3 $project_dir/scripts/serve_ui.py --host 0.0.0.0 --port 8090 --allow-insecure-lan --https-port 8443 --cert $tls_dir/orin.crt --key $tls_dir/orin.key
EOF
}
if [[ "$dry_run" == 1 ]]; then
  emit_dropin
  printf '# A fresh certificate will be generated for IP:%s; nothing was changed.\n' "$lan_ip"
  exit 0
fi
# Generate and verify the fresh identity before replacing any current certificate or unit drop-in.
staging_dir="$(mktemp -d)"
trap 'rm -rf "$staging_dir"' EXIT
openssl req -x509 -newkey rsa:2048 -sha256 -nodes -days 365 \
  -keyout "$staging_dir/orin.key" -out "$staging_dir/orin.crt" \
  -subj "/CN=$lan_ip" \
  -addext "subjectAltName=IP:$lan_ip,DNS:localhost,IP:127.0.0.1"
openssl x509 -in "$staging_dir/orin.crt" -noout -checkip "$lan_ip"
openssl x509 -in "$staging_dir/orin.crt" -noout -checkend 86400
cp /etc/systemd/system/cosmos-edge-ui.service "$staging_dir/cosmos-edge-ui.service"
mkdir "$staging_dir/cosmos-edge-ui.service.d"
if [[ -d /etc/systemd/system/cosmos-edge-ui.service.d ]]; then
  for existing_dropin in /etc/systemd/system/cosmos-edge-ui.service.d/*.conf; do
    [[ -f "$existing_dropin" ]] || continue
    cp "$existing_dropin" "$staging_dir/cosmos-edge-ui.service.d/"
  done
fi
emit_dropin > "$staging_dir/cosmos-edge-ui.service.d/lan.conf"
systemd-analyze verify "$staging_dir/cosmos-edge-ui.service"
install -d -o "$service_user" -g "$service_group" -m 700 "$tls_dir"
install -d -m 755 /etc/systemd/system/cosmos-edge-ui.service.d "$project_dir/results/service-install"
dropin=/etc/systemd/system/cosmos-edge-ui.service.d/lan.conf
if [[ -f "$dropin" ]]; then
  cp -a "$dropin" "$project_dir/results/service-install/lan.before.$(date -u +%Y%m%dT%H%M%SZ)"
fi
install -o "$service_user" -g "$service_group" -m 600 "$staging_dir/orin.key" "$tls_dir/orin.key"
install -o "$service_user" -g "$service_group" -m 644 "$staging_dir/orin.crt" "$tls_dir/orin.crt"
emit_dropin > "$dropin"
systemctl daemon-reload
printf 'Configured HTTP http://%s:8090 and HTTPS https://%s:8443 with a fresh certificate.\n' "$lan_ip" "$lan_ip"
printf '%s\n' 'No services were restarted. Restart cosmos-edge-ui only when ready to apply LAN exposure.' \
  'The local certificate must be accepted/trusted again; the prior certificate was not reused.'
