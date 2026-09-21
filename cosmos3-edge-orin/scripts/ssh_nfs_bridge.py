#!/usr/bin/env python3
"""Bridge target loopback NFS to guest loopback NFS using two pinned SSH sessions.

Original project integration; SSH implementation is Paramiko (LGPL-2.1).
No NFS service is exposed on the Mac. This foreground helper does not mount,
export, flash, reconnect, or write remote files. See docs/usb-nfs-bridge.md.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import select
import stat
import sys
import threading


LOOPBACK = "127.0.0.1"


def port_number(value):
    number = int(value)
    if not 1 <= number <= 65535:
        raise argparse.ArgumentTypeError("Port must be from 1 through 65535")
    return number


def host_name(value):
    if not value or re.search(r"[\s/@\\\x00-\x1f]", value) or value.startswith(("-", "[")):
        raise argparse.ArgumentTypeError("Use a hostname or unbracketed IP address, without username or URI")
    return value


def checked_file(value, private=False):
    path = Path(value).expanduser()
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"Expected a regular, non-symlink file: {path}")
    if info.st_uid != os.getuid():
        raise ValueError(f"File must be owned by the current user: {path}")
    if info.st_mode & (0o077 if private else 0o022):
        rule = "owner-only permissions (0600 or 0400)" if private else "no group/other write permissions"
        raise ValueError(f"File requires {rule}: {path}")
    return path


def read_password(args):
    if args.target_password_env:
        name = args.target_password_env
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError("Password environment variable name is invalid")
        value = os.environ.get(name, "")
    else:
        path = checked_file(args.target_password_file, private=True)
        value = path.read_text(encoding="utf-8").removesuffix("\n").removesuffix("\r")
    if not value or "\n" in value or "\r" in value or "\x00" in value:
        raise ValueError("Password source must contain one nonempty line; its contents are never printed")
    return value


def pinned_client(paramiko, filename, host, port):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.load_host_keys(str(checked_file(filename)))
    name = host if port == 22 else f"[{host}]:{port}"
    if not client.get_host_keys().lookup(name):
        client.close()
        raise ValueError(f"Task known-hosts file has no pinned key for {name}")
    return client


def verify_loopback_listener(text, port):
    """Reject an SSH server which broadens the requested IPv4 loopback bind."""
    listeners = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 4 and fields[3] == "0A":
            address, _, encoded_port = fields[1].rpartition(":")
            if encoded_port and int(encoded_port, 16) == port:
                listeners.append(address.upper())
    if not listeners or any(address != "0100007F" for address in listeners):
        raise ValueError("Target NFS listener is absent or not exclusively 127.0.0.1; refusing the bridge")


def remote_read(client, command):
    _, stdout, stderr = client.exec_command(command, timeout=10)
    output, errors = stdout.read(), stderr.read()
    if stdout.channel.recv_exit_status():
        raise RuntimeError("Remote setup/check failed: " + errors.decode(errors="replace")[:300])
    return output.decode(errors="replace")


def relay(incoming, guest_transport, target_transport, stopping, guest_nfs_port):
    onward = None
    try:
        onward = guest_transport.open_channel("direct-tcpip", (LOOPBACK, guest_nfs_port),
                                              (LOOPBACK, 0), timeout=10)
        incoming.settimeout(30)
        onward.settimeout(30)
        while not stopping.is_set() and target_transport.is_active() and guest_transport.is_active():
            readable, _, _ = select.select([incoming, onward], [], [], 1)
            for source in readable:
                data = source.recv(65536)
                if not data:
                    return
                (onward if source is incoming else incoming).sendall(data)
    except Exception as error:
        print(f"NFS relay connection ended: {type(error).__name__}", file=sys.stderr, flush=True)
    finally:
        incoming.close()
        if onward is not None:
            onward.close()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--guest-host", type=host_name, default=LOOPBACK)
    result.add_argument("--guest-port", type=port_number, default=2222)
    result.add_argument("--guest-user", default="flash")
    result.add_argument("--guest-known-hosts", required=True)
    result.add_argument("--guest-key", required=True)
    result.add_argument("--guest-nfs-port", type=port_number, default=2049)
    result.add_argument("--target-host", type=host_name, required=True, help="For USB, e.g. fe80::1%%en8")
    result.add_argument("--target-port", type=port_number, default=22)
    result.add_argument("--target-user", default="root")
    result.add_argument("--target-known-hosts", required=True)
    result.add_argument("--target-nfs-port", type=port_number, default=2049)
    password = result.add_mutually_exclusive_group(required=True)
    password.add_argument("--target-password-env", help="Environment variable NAME, never a password value")
    password.add_argument("--target-password-file", help="Existing owner-only file containing one password line")
    result.add_argument("--raise-target-loopback", action="store_true",
                        help="Explicitly run 'ip link set lo up' in the target's temporary RAM system")
    result.add_argument("--check", action="store_true", help="Validate local files, credentials and key pins without network access")
    return result


def main(argv=None):
    options = parser()
    args = options.parse_args(argv)
    guest = target = None
    stopping = threading.Event()
    try:
        try:
            import paramiko
        except ImportError:
            raise ValueError("Paramiko is required; use the task SSH environment or install paramiko==5.0.0") from None
        guest_key = checked_file(args.guest_key, private=True)
        password = read_password(args)
        guest = pinned_client(paramiko, args.guest_known_hosts, args.guest_host, args.guest_port)
        target = pinned_client(paramiko, args.target_known_hosts, args.target_host, args.target_port)
        if args.check:
            print("Local checks passed: credential permissions and exact host-key pins; no network access performed.")
            return 0
        settings = {"look_for_keys": False, "allow_agent": False, "timeout": 10, "auth_timeout": 10, "banner_timeout": 10}
        guest.connect(args.guest_host, port=args.guest_port, username=args.guest_user,
                      key_filename=str(guest_key), **settings)
        target.connect(args.target_host, port=args.target_port, username=args.target_user,
                       password=password, **settings)
        password = None
        gt, tt = guest.get_transport(), target.get_transport()
        gt.set_keepalive(15)
        tt.set_keepalive(15)
        if args.raise_target_loopback:
            remote_read(target, "ip link set lo up")
        tt.request_port_forward(LOOPBACK, args.target_nfs_port)
        verify_loopback_listener(remote_read(target, "cat /proc/net/tcp /proc/net/tcp6"), args.target_nfs_port)
        print(f"NFS bridge ready: target {LOOPBACK}:{args.target_nfs_port} -> two pinned SSH sessions "
              f"-> guest {LOOPBACK}:{args.guest_nfs_port}. No mount or flash performed.", flush=True)
        while tt.is_active() and gt.is_active():
            channel = tt.accept(1)
            if channel is not None:
                threading.Thread(target=relay, args=(channel, gt, tt, stopping, args.guest_nfs_port), daemon=True).start()
        print("An SSH session ended; bridge stopped without reconnecting.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        # Paramiko does not include authentication secrets in normal exceptions.
        # Only locally controlled validation diagnostics are printed in full.
        message = str(error) if isinstance(error, (ValueError, OSError)) else type(error).__name__
        print("Bridge failed: " + message, file=sys.stderr)
        return 1
    finally:
        stopping.set()
        if target is not None:
            target.close()
        if guest is not None:
            guest.close()


if __name__ == "__main__":
    raise SystemExit(main())
