#!/usr/bin/env bash
# Task-owned LAN listeners. Keeps inference on loopback and the selected engine resident.
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Path guard. Upstream this was pinned to /home/jetson/cosmos-edge; porch-dad installs the
# same tree at /home/orin/nvr/ui, so both are accepted. The point of the check is that we are
# root, on the Orin, and pointed at a real copy of this UI - not the literal path.
case "$project_dir" in
  /home/jetson/cosmos-edge|/home/orin/nvr/ui) : ;;
  *) echo "Run with sudo on the Orin in /home/jetson/cosmos-edge or /home/orin/nvr/ui." >&2; exit 2 ;;
esac
[[ $EUID == 0 && $(uname -m) == aarch64 ]] || { echo 'Run with sudo on the Orin (aarch64).' >&2; exit 2; }
lan_ip="$(python3 - "${1:?Usage: sudo bash scripts/enable_lan_ui.sh ORIN_LAN_IPV4}" <<'PY'
import ipaddress, sys
address = ipaddress.IPv4Address(sys.argv[1])
if address.is_loopback or address.is_unspecified or address.is_multicast:
    raise SystemExit('Provide the Orin LAN IPv4 address')
print(address)
PY
)"
tls_dir="$project_dir/deployment/tls"
install -d -o jetson -g jetson -m 700 "$tls_dir"
if [[ ! -f "$tls_dir/orin.crt" || ! -f "$tls_dir/orin.key" ]]; then
  openssl req -x509 -newkey rsa:2048 -sha256 -nodes -days 365 \
    -keyout "$tls_dir/orin.key" -out "$tls_dir/orin.crt" \
    -subj '/CN=Cosmos3 Edge Orin' \
    -addext "subjectAltName=IP:$lan_ip,DNS:jetson.local,DNS:localhost,IP:127.0.0.1"
  chown jetson:jetson "$tls_dir/orin.key" "$tls_dir/orin.crt"
  chmod 600 "$tls_dir/orin.key"
  chmod 644 "$tls_dir/orin.crt"
fi
openssl x509 -in "$tls_dir/orin.crt" -noout -checkip "$lan_ip"
openssl x509 -in "$tls_dir/orin.crt" -noout -checkend 86400
install -d -m 755 /etc/systemd/system/cosmos-edge-ui.service.d "$project_dir/results/service-install"
dropin=/etc/systemd/system/cosmos-edge-ui.service.d/lan.conf
if [[ -f "$dropin" ]]; then
  cp -a "$dropin" "$project_dir/results/service-install/lan.before.$(date -u +%Y%m%dT%H%M%SZ)"
fi
cat > "$dropin" <<EOF
[Service]
ExecStart=
ExecStart=/usr/bin/python3 $project_dir/scripts/serve_ui.py --host 0.0.0.0 --port 8090 --allow-insecure-lan --https-port 8443 --cert $tls_dir/orin.crt --key $tls_dir/orin.key
EOF
systemd-analyze verify /etc/systemd/system/cosmos-edge-ui.service
systemctl daemon-reload
printf 'Configured HTTP http://%s:8090 and HTTPS https://%s:8443. Restart cosmos-edge-ui at an idle interval to apply.\n' "$lan_ip" "$lan_ip"
