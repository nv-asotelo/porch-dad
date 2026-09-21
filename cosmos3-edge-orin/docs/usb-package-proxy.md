# Package downloads over the existing USB connection

Prepared 2026-09-20 from public upstream and Ubuntu documentation. This recipe was not installed or exercised by the research subtask. It uses the Linux guest's existing internet connection and normal SSH forwarding; it does not change macOS routing, Internet Sharing, USB driver ownership or global proxy settings.

## Guest: executable-only Tinyproxy

Install **`tinyproxy-bin`**, not the service package `tinyproxy`, in the Ubuntu 22.04 guest:

```bash
sudo apt-get install --no-install-recommends tinyproxy-bin
/usr/bin/tinyproxy -v
```

Ubuntu Jammy publishes `tinyproxy-bin` 1.11.0-1 separately, depending only on libc6. Its file list contains the executable, documentation and HTML resources, with no service unit or global configuration. Use the repository's available package revision and record it. [Ubuntu package](https://packages.ubuntu.com/jammy/tinyproxy-bin), [package files](https://packages.ubuntu.com/jammy/amd64/tinyproxy-bin/filelist)

As the ordinary guest user `flash`, create a task-specific directory `/home/flash/cosmos-egress` and save this as `tinyproxy.conf` there:

```text
Port 8888
Listen 127.0.0.1
Allow 127.0.0.1
Timeout 300
MaxClients 8
ConnectPort 443
LogLevel Warning
PidFile "/home/flash/cosmos-egress/tinyproxy.pid"
```

Start it as `flash`, without sudo:

```bash
/usr/bin/tinyproxy -d -c /home/flash/cosmos-egress/tinyproxy.conf
```

`-d` keeps it in the foreground; `-c` selects this file. With no LogFile/Syslog directive it logs to standard output, which the task supervisor can retain. User/Group directives are unnecessary when already running unprivileged. Port 8888 needs no privileged binding. [Official quickstart](https://tinyproxy.github.io/), [1.11.0 sample configuration](https://github.com/tinyproxy/tinyproxy/blob/1.11.0/etc/tinyproxy.conf.in)

This caps simultaneous clients at eight and idle connections at 300 seconds. The timeout is **inactivity**, not a whole-download deadline. `ConnectPort 443` permits HTTPS CONNECT only to port 443; omitting it would allow all CONNECT ports. Ordinary HTTP repository requests still work. This is not a destination-domain allowlist or a firewall for ordinary HTTP ports. Only trusted task package commands should use it. Tinyproxy 1.11 uses threads; obsolete `StartServers`/`MinSpareServers` directives are unnecessary. [Pinned directive parser](https://github.com/tinyproxy/tinyproxy/blob/1.11.0/src/conf.c), [configuration manual](https://github.com/tinyproxy/tinyproxy/blob/1.11.0/docs/man5/tinyproxy.conf.txt.in)

## Mac: two loopback-only forwards

Keep these SSH clients alive on the Mac while downloading. The first uses the existing guest SSH configuration:

```bash
ssh -F data/flash-host/ssh_config -NT \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
  -L 127.0.0.1:22888:127.0.0.1:8888 flash-host
```

The second connects to the **verified postboot Jetson SSH endpoint**. Set `JETSON_SSH_TARGET` to its actual user/address before running; do not assume the recovery address survives normal boot:

```bash
ssh -NT \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
  -R 127.0.0.1:8888:127.0.0.1:22888 "$JETSON_SSH_TARGET"
```

Use the project's existing device key and verified host-key handling where required. Both ports exceed 1023; the normal Jetson user can request this remote forward. Server-side TCP forwarding must be enabled, with loopback binding permitted. The chain is Jetson localhost:8888 → Mac localhost:22888 → guest localhost:8888. No listener is exposed on the LAN. [OpenSSH forwarding](https://man.openbsd.org/ssh), [server forwarding settings](https://man.openbsd.org/sshd_config)

## Jetson: command-scoped apt and pip

Both HTTP and HTTPS clients use an **HTTP proxy URL**. Tinyproxy tunnels HTTPS bytes with CONNECT; it does not terminate TLS. Keep normal CA validation, package signatures and hashes. DNS for public destinations is performed by the guest proxy, so the Jetson does not need a separate internet DNS route for these proxied requests.

For apt, explicit per-command options avoid dependence on sudo preserving environment variables:

```bash
sudo apt-get \
  -o Acquire::http::Proxy=http://127.0.0.1:8888 \
  -o Acquire::https::Proxy=http://127.0.0.1:8888 \
  -o Acquire::Connect::AddrConfig=false \
  -o Acquire::Retries=2 -o APT::Update::Error-Mode=any update
```

Repeat those options on each required `apt-get install` command. They do not configure arbitrary network downloads run by package scripts; if a documented installer subprocess also needs the proxy, supply `sudo env http_proxy=... https_proxy=... no_proxy=localhost,127.0.0.1,::1` for that single command. Do not write `/etc/apt/apt.conf.d` or `/etc/environment`. [APT HTTP proxy options](https://manpages.debian.org/bookworm/apt/apt-transport-http.1.en.html), [APT HTTPS behavior](https://manpages.debian.org/bookworm/apt/apt-transport-https.1.en.html)

For the active backend virtual environment, prefix the already reviewed installation command:

```bash
env http_proxy=http://127.0.0.1:8888 \
    https_proxy=http://127.0.0.1:8888 \
    no_proxy=localhost,127.0.0.1,::1 \
    python -m pip install --retries 2 --timeout 60 -e '.[server,server-tools]'
```

Run this example from the pinned backend checkout with its virtual environment active. Pip officially accepts these variables or `--proxy`; command-scoped variables also reach compliant build subprocesses. A subprocess using a different network stack still needs verification. Do not disable TLS verification or add trusted-host exceptions to make a failing connection pass. [Pip proxy documentation](https://pip.pypa.io/en/stable/user_guide/#using-a-proxy-server)

## Bounded verification and cleanup

Check guest/Mac/Jetson listeners bind only to `127.0.0.1`. From the Jetson, verify one public HTTPS request with a 30-second deadline:

```bash
curl --fail --head --connect-timeout 10 --max-time 30 \
  --proxy http://127.0.0.1:8888 https://pypi.org/simple/pip/
```

Then verify the needed Ubuntu/NVIDIA repository metadata and proceed with the scoped package commands. Record package versions and the first actual transfer success; tunnel creation alone is insufficient. After two failed attempts, inspect the task proxy/SSH log instead of looping. Run one package-manager operation at a time; eight clients is a connection ceiling, not a memory or bandwidth guarantee.

When setup finishes, stop the two task-owned tunnel sessions and this Tinyproxy process. Its PID file identifies the intended process; verify its command line before signaling it. No global proxy service or network setting was enabled, so no global rollback is needed. Internet access remains limited to applications explicitly configured to use this proxy; it is not general IP connectivity.

## Observed RAM-recovery provisioning

After flashing, a native ARM chroot of the NVMe rootfs successfully fetched public HTTPS through the same forwarding helper using port 8888 at both ends. The rootfs shares the RAM recovery network namespace. APT initially rejected `127.0.0.1` because recovery had only IPv6 on non-loopback interfaces, and APT enables `AI_ADDRCONFIG`. The same failure reproduced with Python's `getaddrinfo(..., AI_ADDRCONFIG)` despite ordinary HTTPS already working. The installed APT HTTP method contains the `Acquire::Connect::AddrConfig` option; the provisioning helper disables this address-filtering hint only for proxied commands. This changes address resolution behavior, not TLS or signature checks. [APT's public connection implementation](https://github.com/Debian/apt/blob/main/methods/connect.cc).

Chroot provisioning temporarily suppresses service starts with `policy-rc.d`, uses private bind mounts for `/proc`, `/sys` and `/dev` (the first two read-only), and does not bind recovery `/run`. It must restore the temporary policy, sync and unmount before normal reboot. Compute-package installation in this environment is not a driver or GPU execution test.
