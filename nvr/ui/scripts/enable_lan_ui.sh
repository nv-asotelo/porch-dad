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

# Ports and interpreter differ per install. Upstream cosmos-edge owns 8090, but on porch-dad
# 8090 is already the older Live VLM WebUI and binding it here would collide with a running
# service - so this UI uses 8092 there, matching its unit. Override with argv 2 and 3.
case "$project_dir" in
  /home/orin/nvr/ui)
    http_port="${2:-8092}"; https_port="${3:-8443}"
    python_bin=/home/orin/TensorRT-Edge-LLM/.venv/bin/python ;;
  *)
    http_port="${2:-8090}"; https_port="${3:-8443}"
    python_bin=/usr/bin/python3 ;;
esac
for p in "$http_port" "$https_port"; do
  [[ "$p" =~ ^[0-9]+$ ]] || { echo "Bad port: $p" >&2; exit 2; }
done
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
ExecStart=$python_bin $project_dir/scripts/serve_ui.py --host 0.0.0.0 --port $http_port --allow-insecure-lan --https-port $https_port --cert $tls_dir/orin.crt --key $tls_dir/orin.key
EOF
systemd-analyze verify /etc/systemd/system/cosmos-edge-ui.service
systemctl daemon-reload
printf 'Configured HTTP http://%s:%s and HTTPS https://%s:%s. Restart cosmos-edge-ui at an idle interval to apply.\n' "$lan_ip" "$http_port" "$lan_ip" "$https_port"
