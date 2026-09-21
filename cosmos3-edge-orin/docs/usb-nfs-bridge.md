# USB recovery NFS bridge

During this task, both the original temporary helper and the reusable `scripts/ssh_nfs_bridge.py` successfully carried an **NFSv4.1 read-only mount from the Jetson's RAM recovery system to the Linux VM**, using two SSH sessions with pinned host keys. The Mac reached the VM at `127.0.0.1:2222` and the Jetson over native USB link-local IPv6 at `fe80::1%en8:22`. The interface name is specific to that connection and must be verified again after reconnecting. This proved the mount route; it did not prove a final flash, installed JetPack, or model inference.

```text
Jetson RAM system: NFS client -> 127.0.0.1:2049
  -> target SSH remote forwarding over native USB IPv6
  -> Python helper on the Mac
  -> guest SSH direct-tcpip forwarding
  -> Linux VM: 127.0.0.1:2049 NFS server
```

The reusable `scripts/ssh_nfs_bridge.py` preserves that design. It avoids macOS privilege changes for USB NCM passthrough and creates no Mac NFS listener. Both NFS endpoint addresses are fixed to `127.0.0.1`; ports and SSH endpoints are configurable. After requesting the target listener, it reads the target's `/proc/net/tcp` and `/proc/net/tcp6` and refuses a listener exposed beyond IPv4 loopback. Both SSH clients load only the specified task known-hosts files and reject missing or changed host keys. Obtain/verify the host keys independently; the helper never enrolls or replaces them.

Use the task environment, which currently contains Paramiko 5.0.0, or install that public dependency in a separate environment. This example contains no password and assumes an existing owner-only credential file:

```sh
.qa/ssh-client-venv/bin/python scripts/ssh_nfs_bridge.py \
  --guest-host 127.0.0.1 --guest-port 2222 --guest-user flash \
  --guest-known-hosts data/flash-host/known_hosts \
  --guest-key data/flash-host/id_ed25519 \
  --target-host 'fe80::1%en8' --target-port 22 --target-user root \
  --target-known-hosts .qa/jetson-ram-known_hosts \
  --target-password-file /absolute/path/to/owner-only-password-file \
  --check
```

`--check` validates local credential permissions and exact host-key entries without opening connections; remove it to start the foreground bridge. For a nondefault SSH port, the known-hosts entry must use `[host]:port`. The password file and guest private key must be owned by the current user, have no group/other permissions, and be regular files rather than symlinks. Known-hosts files must be owned by the current user and not writable by group/others. Alternatively, pass `--target-password-env VARIABLE_NAME` to read an existing environment variable; no command-line password argument exists. Keep credentials out of shell history, logs and committed files.

The target loopback interface must already be up. To authorize the temporary network-state change, add `--raise-target-loopback`, which runs only `ip link set lo up`. Without that flag the helper runs only the read-only listener check, opens the requested SSH channels, and does not modify remote files or network-interface state. It does not configure exports, create mountpoints, mount storage, write images, or flash the device. The VM's existing NFS server/export and SSH forwarding permissions are prerequisites. Use NFSv4.1 over TCP with a read-only export/mount for the initial check; forwarding only port 2049 does not supply the additional services expected by many NFSv3 configurations.

The bridge is an opaque TCP relay: **read-only access comes from the NFS export and mount configuration**, not from the relay. Keep it running while a verified client depends on the mount; stop it with Ctrl-C when finished. It closes both sessions on exit and does not reconnect automatically. Do not start a second bridge on a target port already in use.

At approximately **2026-09-20 02:03 UTC**, the reusable helper passed a live check on the separate target port **20490**, leaving the original port-2049 bridge untouched. The Jetson mounted `127.0.0.1:/` at `/mnt/cosmos-bridge-check` with `ro,nosuid,nodev,nosharecache,vers=4.1,proto=tcp,port=20490`. Reading `internal/flash.idx` through that mount produced SHA-256 `7da3dfcfd6835028f247d254d7780dd59b7abbd6ff29616f3c9e966859a18496`, exactly matching the VM source. Unmount completed with exit code 0, then Ctrl-C stopped the test helper with exit code 130. Pinned host keys, the guest private key and a protected target-password file were used. [Recorded validation evidence](../results/ssh-nfs-bridge-validation.json) distinguishes this successful live mount/integrity check from the earlier 23 offline guard checks. No flash or model-performance conclusion follows from this test.

The forwarding integration is original project work. SSH transport and authentication come from [Paramiko](https://github.com/paramiko/paramiko), whose installed 5.0.0 package identifies **LGPL-2.1** and includes its [license](https://github.com/paramiko/paramiko/blob/main/LICENSE). Paramiko is used as an external, unmodified dependency; no Paramiko implementation is copied or relicensed here. Preserve its license and dependency notices when distributing a bundled environment. See also `THIRD_PARTY_NOTICES.md`.
