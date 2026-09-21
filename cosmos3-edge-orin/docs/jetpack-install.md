# JetPack installation over USB-C

**Observed result:** NVIDIA's R39.2.1 Super QSPI/NVMe flash completed with exit code 0 and `Flashing success` on the connected Orin Nano. The written Ubuntu 24.04.4 / Jetson Linux 39.2.1 filesystem and GPT passed read-only checks. Normal NVMe boot, CUDA execution, compute package installation and rootfs expansion subsequently passed. See [flash evidence](../results/jetpack-flash-status.json), [payload verification](../results/install-app-payload-verification.json), [firmware verification](../results/install-firmware-payload-verification.json), [normal boot](../results/normal-boot.json), and [CUDA preflight](../results/cuda-preflight.json).

## Identified hardware and selected software

Official recovery EEPROM reads identified P3767-0005, FAB 300, revision T.1, chip SKU D5 and RAMCODE 2: an Orin Nano 8 GB developer-kit module. RAM recovery identified one WD_BLACK SN7100 1 TB NVMe, 1,000,204,886,016 bytes, with no partitions or recognized filesystem signature and no microSD. The flash targeted only this NVMe and the module's 64 MiB QSPI. [Board evidence](../results/jetson-board.json), [storage evidence](../results/jetson-storage.json).

The selected platform is JetPack 7.2.1 / Jetson Linux 39.2.1, Ubuntu 24.04, CUDA 13.2 and TensorRT 10.16.2. NVIDIA's signed repository metadata resolves the CUDA toolkit metapackage to 13.2.2-1 and compiler/runtime components to 13.2.86-1; these precise package versions supersede the download page's broader CUDA 13.2.1 wording for this installation. The component-selection recipe avoids unnecessary JetPack development/sample metapackages. Patch-version runtime compatibility still needs device validation. [Official downloads](https://developer.nvidia.com/embedded/jetpack/downloads), [compute recipe and signed metadata](../research/jetpack-compute-install.md), [backend support matrix](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/support-matrix.html).

## The route that worked on this Mac

NVIDIA documents Ubuntu x86_64 for its direct USB recovery-flash tools. This host is Apple Silicon macOS. We used the official NVIDIA tools in an isolated Ubuntu 22.04.5 amd64 guest under QEMU 11.1.1, with a small original transport bridge for the second stage. This is an observed integration on this board and host, not a claim that NVIDIA officially supports this Mac/QEMU route. [Official BSP setup guide](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/setup_bsp.html).

1. A user-performed cold recovery reset with FC REC connected to GND resolved an initial USB transfer stall. The Orin remained separately powered through its DC supply; USB-C carried data.
2. The guest used NVIDIA's recovery tools to read the board identity and load the official kernel/initrd into RAM. This stage did not write persistent target storage.
3. After RAM boot, macOS owned the USB NCM network interface and unprivileged QEMU could not detach it. Native Mac SSH to the target's USB link-local interface worked. No macOS security-policy or driver change was necessary.
4. Through native SSH we inventoried the target disk and memory. The guest generated the official Super QSPI plus explicit NVMe installation package offline, using the read board identity.
5. A pinned-host-key SSH bridge exposed the guest's read-only NFS export at the target's loopback interface. The Jetson mounted this export, and independently verified the 2,774,391,691-byte APP payload and all 36 referenced firmware files before flashing. [Transport documentation](usb-nfs-bridge.md).
6. On the target, `USER=root /mnt/cosmos-flash/l4t_flash_from_kernel.sh --no-reboot` ran NVIDIA's unmodified flasher. QSPI and NVMe completed successfully. No `--erase-all`, host-drive override or SD layout was used.
7. We ran `sgdisk --verify` and unmounted-filesystem `e2fsck -fn`, inspected the installed release and deployment public key, then cleanly unmounted the NFS export and stopped the bridge. The raw flash log remains in the private task logs because it includes the board serial number.

The APP archive's own sidecar hash was used: this package's APP index row has empty image/hash fields and selects `system.img` through `APP_ext`. The official archive-extraction path does not enforce its commented-out checksum comparison, so independent payload verification mattered. [Package review](../research/prepared-super-nvme-package-review.md).

The generated APP was initially 55 GiB even though the SSD is larger. It is partition 1 but physically last. After normal boot, guarded `growpart` and `resize2fs` expanded it to a 998,601,301,504-byte partition while preserving its start/GUID and every auxiliary partition. [Expansion procedure](expand-nvme-rootfs.md), [completed expansion](../results/storage-expansion.json).

## Completed normal boot and compute validation

The user confirmed that the recovery jumper was disconnected. Normal boot then exposed the device over LAN/mDNS, and the task-specific SSH key authenticated. The normal-boot receipt verifies an aarch64 NVMe ext4 root. This confirmation establishes reported jumper state, not the exact time or physical removal procedure.

The compute installer completed its plan and apply phases in a native ARM64 chroot on the freshly written NVMe rootfs before normal boot, using the [command-scoped USB package proxy](usb-package-proxy.md) with TLS and apt signature verification. Bind mounts were cleaned up before boot. [Compute installation](../results/chroot-compute-provisioning.json), [cleanup](../results/chroot-cleanup.json).

After normal boot, a CUDA array calculation passed on SM87 and TensorRT 10.16.2.10 imported successfully. The later native build, real image inference and browser validation are recorded in [the completed goal](../GOAL.md). The UI and model now run as enabled services on the Orin. A physical reboot after final service installation was not tested.

## Alternative official Mac media route

NVIDIA also documents preparing a bootable installer USB stick from macOS and moving it to the Jetson. The official installer ISO was downloaded and locally hashed during preparation, but it was not the route used for this flash. [Orin Nano quick start](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/quick_start.html), [ISO record](../results/installer-download.json).
