# Audit: native macOS USB networking with tunneled NFS

2026-09-20 UTC. Static inspection of official R39.2.1 scripts, public OpenSSH/NFS references, and read-only queries of the prepared Linux guest. This subtask made no device, VM, export, SSH configuration, mount, or storage changes. Commands below are prospective actions for the primary agent, not executed results.

The primary agent reports successful native Mac SSH to `root@fe80::1%en8`, identifying a partitionless WD_BLACK SN7100 1TB NVMe and no microSD. QEMU stderr now explicitly reports `libusb_detach_kernel_driver: -3 [ACCESS]`. That confirms the driver-capture denial previously only suspected. The earlier QEMU trace's `request_emulated status -3` by itself meant `USB_RET_STALL`, not a libusb permission result. libusb's Darwin source requires device-capture entitlement plus authorization, or effective root, to detach a bound driver; it captures the entire device. Native NCM networking avoids that operation. [libusb source](https://github.com/libusb/libusb/blob/v1.0.29/libusb/os/darwin_usb.c#L2821), [QEMU claim path](https://github.com/qemu/qemu/blob/v11.1.1/hw/usb/host-libusb.c#L1341), [QEMU status definitions](https://github.com/qemu/qemu/blob/v11.1.1/include/hw/usb/usb.h)

## Result and transport

**The proposed two SSH forwards are compatible with an explicit NFSv4.1 mount and the official target-side flasher.** This is a project-specific transport adaptation, not a claim that NVIDIA documents this macOS/VM workflow. Mount and package validation must succeed before invoking the destructive script.

```text
Jetson NFS client -> Jetson 127.0.0.1:2049
  --SSH remote forward--> Mac 127.0.0.1:22049
  --SSH local forward--> Ubuntu guest 127.0.0.1:2049 (nfsd)
```

Run both persistent SSH clients on the **Mac**, keeping the current USB network and Jetson RAM boot alive:

```bash
ssh -F data/flash-host/ssh_config -NT \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
  -L 127.0.0.1:22049:127.0.0.1:2049 flash-host

ssh -6 -NT \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
  -R 127.0.0.1:2049:127.0.0.1:22049 root@fe80::1%en8
```

Use the existing verified recovery SSH authentication/host-key handling. A successful forward allocation does not prove NFS connectivity; verify the mount and actual image reads. Each `127.0.0.1` is relative to the endpoint shown above. No Mac SSH server, LAN listener, guest USB ownership, or privileged Mac networking is needed. [OpenSSH forwarding behavior](https://man.openbsd.org/ssh)

## Server export and mount path

The complete package directory is `/home/flash/Linux_for_Tegra/tools/kernel_flash/images`, containing `l4t_flash_from_kernel.sh`, `internal/`, `external/`, configurations, checksums and any packaged helpers. Wait for successful `--no-flash` generation and retain the package unchanged during use. Export **only this directory**, to guest loopback only. With no existing conflicting NFSv4 root, the precise proposed export is:

```text
/home/flash/Linux_for_Tegra/tools/kernel_flash/images 127.0.0.1(ro,sync,insecure,no_subtree_check,no_root_squash,fsid=0)
```

`fsid=0` makes this directory the NFSv4 export root, so the target mounts **`127.0.0.1:/`**, not the guest's full filesystem path. `insecure` permits the nonprivileged source port used by the guest SSH forwarding process; the client allowlist remains loopback. `no_root_squash` permits the target root process to read root-owned image files; the export remains read-only. If an NFSv4 root already exists, reconcile the export namespace instead of assigning a second `fsid=0`. [Linux exports reference](https://man7.org/linux/man-pages/man5/exports.5.html)

Read-only guest evidence: `/proc/fs/nfsd/versions` reports `-2 +3 +4 +4.1 +4.2`. Pin 4.1 and TCP port 2049 to avoid rpcbind/mountd and NFSv4.0's separate callback connection. NFSv4.1 uses the existing connection for callbacks. A prospective mount on the Jetson is:

```bash
mount -t nfs -o ro,vers=4.1,proto=tcp,port=2049,hard 127.0.0.1:/ /mnt
```

Keep the normal hard mount behavior for image reads. Both tunnels must remain healthy through the full flash; a lost tunnel can stall I/O. Confirm `/proc/mounts` shows the intended NFS version/source and verify readable files through `/mnt`. [NFS mount options](https://man7.org/linux/man-pages/man5/nfs.5.html), [NFSv4.1 callback protocol](https://www.rfc-editor.org/info/rfc8881/)

## Exact official flasher invocation

After target identity, image provenance, full package validation and mount checks, the prospective destructive invocation **on the Jetson** is:

```bash
USER=root /mnt/l4t_flash_from_kernel.sh --no-reboot
```

The script resolves `COMMON_IMAGES_DIR` from its own location and sources `/mnt/internal/flash.cfg` followed by `/mnt/external/flash.cfg`. The external configuration must specify the confirmed NVMe (`nvme0n1p1`); internal `flash.idx` must contain QSPI only, external `flash.idx` only the intended NVMe layout. Ensure no inherited `EXTDEV_ON_HOST`, `EXTDEV_ON_TARGET`, or other device override changes these selections. Do not add `--host-mode`, `--direct`, `--external-only`, `--qspi-only` or `-k` for this full QSPI+NVMe install. `--no-reboot` keeps the current RAM session alive to inspect the result. **This script has no `--no-flash` preview mode.** It creates GPTs, then writes QSPI and storage; only run after preparation is complete. [Initialization and argument parsing](/Users/asotelo/Documents/ChatGPT/jetson-orin-streaming-codes/.qa/bsp-inspect/Linux_for_Tegra/tools/kernel_flash/l4t_flash_from_kernel.sh:1400)

No `NFS_ROOTFS_DIR`, `NFS_IMAGES_DIR`, server-IP environment variable or target `initrd_flash.cfg` edit is required for this direct invocation. The generated APP artifact is already packaged in `images/external`; ordinary flashing reads the package and writes target storage, not the NFS image directory. This supports a read-only export. Resolve any absolute symlink escaping the exported package before execution. [Image packaging](/Users/asotelo/Documents/ChatGPT/jetson-orin-streaming-codes/.qa/bsp-inspect/Linux_for_Tegra/tools/kernel_flash/l4t_create_images_for_kernel_flash.sh:369)

The convenience wrapper `/bin/nv_flash_from_network.sh` is less suitable here: it clears `hostip`, sources `/initrd_flash.cfg`, defaults to `fc00:1:1:${instance}::1`, mounts with only `nolock`, and invokes the same flasher. Passing `hostip=127.0.0.1` in its environment does not reliably override that path. An explicit mount avoids negotiation and configuration ambiguity while retaining NVIDIA's actual writer unchanged. [Wrapper](/Users/asotelo/Documents/ChatGPT/jetson-orin-streaming-codes/.qa/bsp-inspect/Linux_for_Tegra/tools/kernel_flash/initrd_flash/nv_flash_from_network.sh:37)

**Independently verify every packaged artifact before starting**, including the APP file against its `.sha1sum` sidecar through the mounted path. `do_write_APP` checks that a sidecar exists but its checksum comparison is commented out. QSPI writing does check image hashes, but QSPI and external writes then run concurrently, so a late read/checksum error is too late to establish that nothing changed. [APP writer](/Users/asotelo/Documents/ChatGPT/jetson-orin-streaming-codes/.qa/bsp-inspect/Linux_for_Tegra/tools/kernel_flash/l4t_flash_from_kernel.sh:1013), [parallel flash entry](/Users/asotelo/Documents/ChatGPT/jetson-orin-streaming-codes/.qa/bsp-inspect/Linux_for_Tegra/tools/kernel_flash/l4t_flash_from_kernel.sh:1535)

## SSH forwarding checks

The guest's effective `sshd -T` reports `allowtcpforwarding yes`, `disableforwarding no`, `permitopen any`, `permitlisten any`, `gatewayports no`, compatible with the proposed local forward. NVIDIA's `ota_make_recovery_img_dtb.sh:89–112` adds root password access for initrd but adds no TCP-forwarding restriction. The copied rootfs SSH config leaves `AllowTcpForwarding` at its normal enabled default, and its original `sshd_config.d` was empty. This supports forwarding in the generated initrd, but the primary should confirm the **actual RAM system's** `sshd -T` output and successful remote-listener setup. `GatewayPorts no` is correct for loopback-only `-R`. [OpenSSH server settings](https://man.openbsd.org/sshd_config)

This audit confirms code-path compatibility and the guest's server capabilities. It does not assert that tunnels were created, the NFS package mounted, storage flashed or JetPack installed.
