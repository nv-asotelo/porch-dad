# Recovery flashing from the connected Apple Silicon Mac

Research snapshot: 2026-09-19. Public sources only. The primary agent now observes NVIDIA APX `0955:7523` in Force Recovery on the Mac and reports that unprivileged libusb open/claim-interface0/release succeeded (`results/apx-libusb-access.json`). That confirms initial APX capture, not recovery boot or flashing. This research did not touch the device, install host software, or download BSP/rootfs payloads. The procedures below have not yet been validated on this hardware.

## Conclusion and bounded route

An x86_64 Ubuntu VM under QEMU TCG with libusb passthrough is a practical candidate. The architecture can execute NVIDIA's documented host tools, and QEMU has automatic USB rediscovery. It is **not yet a proven flash path**: the decisive gate is retaining the Jetson after recovery firmware boots a temporary Linux system and changes USB identity. Test board reads, then RAM-only recovery boot and storage inspection, before a storage-writing command.

Prefer Ubuntu 22.04 amd64 for this attempt if it is the quickest available cloud image. R39.2.1 release notes explicitly support both Ubuntu 22.04 and 24.04 flashing hosts. The Orin Nano setup guide explicitly specifies Ubuntu **x86_64** for SDK Manager/flash scripts. Its `aarch64` BSP filename describes the target architecture; it is not evidence of an Arm host toolchain. The Quick Start's older host wording differs, so use the release-specific release notes for supported Ubuntu versions. [Release notes, p2](https://docs.nvidia.com/jetson/archives/r39.2.1/ReleaseNotes/Jetson_Linux_Release_Notes_r39.2.1.pdf), [Orin Nano BSP setup](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/setup_bsp.html)

Apple Silicon can accelerate an Arm guest with HVF; an x86 guest needs CPU emulation. An Arm VM plus x86 user emulation introduces another compatibility layer and is not the shortest initial diagnostic route. Use `qemu-system-x86_64`, `-accel tcg`, and a Linux ext4 guest disk. Avoid placing the working BSP/rootfs on an ordinary macOS shared directory: Linux ownership, permissions, symlinks, loop mounts and filesystem images need faithful Linux semantics. [QEMU accelerators](https://www.qemu.org/docs/master/system/introduction.html), [Apple Silicon HVF announcement](https://www.qemu.org/2021/12/14/qemu-6-2-0/)

## Official release artifacts

The NVIDIA download page lists JetPack 7.2.1, L4T R39.2.1, Ubuntu 24.04 on target, CUDA 13.2.1, and TensorRT 10.16.2. The following URLs were taken from that page. HTTPS HEAD requests on 2026-09-19 returned 200 and the sizes below. [JetPack downloads](https://developer.nvidia.com/embedded/jetpack/downloads)

| Artifact | Official URL | HTTP Content-Length |
|---|---|---:|
| BSP / driver package | https://developer.nvidia.com/downloads/embedded/L4T/r39_Release_v2.1/release/Jetson_Linux_R39.2.1_aarch64.tbz2 | 1,287,223,769 bytes |
| Sample rootfs | https://developer.nvidia.com/downloads/embedded/L4T/r39_Release_v2.1/release/Tegra_Linux_Sample-Root-Filesystem_R39.2.1_aarch64.tbz2 | 2,022,864,741 bytes |
| ISO, already acquired by primary agent | https://developer.nvidia.com/downloads/embedded/l4t/r39_release_v2.1/iso/jetsoninstaller-r39.2.1-2026-08-07-18-30-47-arm64.iso | Use the primary agent's recorded local size/hash |

The BSP and rootfs redirect to the same path at `developer.download.nvidia.com`. Both returned `Last-Modified: Tue, 11 Aug 2026`.

No publisher tarball SHA256 was located on the official downloads page or in the bounded checksum search. The page's SHA hash links describe Debian packages, not these tarballs. The HTTP ETags are multipart S3 values, **not file MD5/SHA256 checksums**. Record each local SHA256 after download, validate the archive, and label the digest as locally calculated rather than publisher verified. Do not invent a matching checksum.

## USB passthrough and reenumeration

First verify the installed build advertises `usb-host` and inspect its available properties:

```sh
qemu-system-x86_64 -device usb-host,help
```

Initial VM USB fragment, when exactly one NVIDIA USB device is connected:

```text
-device qemu-xhci,id=xhci
-device usb-host,id=jetson,bus=xhci.0,vendorid=0x0955
```

This is deliberately a vendor filter without a fixed APX product ID. A fixed `productid=0x7523` will not match a different PID after boot. If other NVIDIA USB devices are attached, use the particular `hostbus`/`hostport` reported by QEMU instead, after checking whether that physical port keeps the same bus identity across speed changes. Avoid `hostaddr`: it changes on reconnect. Start with the default guest reset behavior. Only test `guest-resets-all=true` or `guest-reset=false` after a documented reset failure, before writing storage. [QEMU USB properties](https://www.qemu.org/docs/master/system/devices/usb.html)

Source verification: QEMU v10.1.0 `usb_host_auto_check()` rescans every two seconds, treats absent vendor/product/port selectors as wildcards, closes disappeared devices, and retries matching devices. It stops trying to open a continuously present device after three failures until it disappears or the QEMU USB device is recreated. Therefore automatic reconnection is implemented, but successful macOS capture still needs a real test. The installed version may differ; record its version. [QEMU tagged libusb source](https://github.com/qemu/qemu/blob/v10.1.0/hw/usb/host-libusb.c#L1834)

macOS libusb supports device capture. Its Darwin backend requires root privileges or the `com.apple.vm.device-access` entitlement to detach an existing kernel driver. APX might be claimable without elevation, while the later USB network gadget may already have a macOS driver. If logs show `LIBUSB_ERROR_ACCESS` or failure to detach, an appropriately privileged QEMU launch is a bounded next step. This is not grounds for disabling SIP or changing unrelated system security settings. [libusb v1.0.29 Darwin backend](https://github.com/libusb/libusb/blob/v1.0.29/libusb/os/darwin_usb.c#L2820)

Keep a QEMU monitor/QMP socket and USB logs. At each stage inspect `info usbhost`, `info usb`, guest `lsusb`, guest `dmesg`, and guest `ip link`. A QMP `device_del`/`device_add` can recreate a stalled passthrough instance during diagnostics. Do not improvise these resets while an actual flash is writing.

## Before any flash: board read, RAM boot, storage read

After extracting the BSP **inside the Linux VM**, inspect `file bootloader/tegrarcm_v2 bootloader/tegrahost_v2 bootloader/chkbdinfo` and the shipped script help. That establishes the actual binary architecture instead of assuming it from the archive name. Install the BSP's host prerequisites inside the guest. Assemble/apply the rootfs only as needed for the RAM boot; preparing host files does not program the Jetson.

`flash.sh --read-info` is NVIDIA's board/chip/fuse/EEPROM query. Adding `--no-flash` generates `bootloader/readinfocmd.txt` rather than executing the query immediately. The following two-stage workflow permits inspection of the exact command first. It should not contain fuse burning, erase, format, or write operations. [Flashing support: read-info](https://docs.nvidia.com/jetson/archives/r39.2.1/DeveloperGuide/SD/FlashingSupport.html#other-examples-of-using-flash-sh)

```sh
# Working directory: the guest's freshly extracted Linux_for_Tegra.
# Candidate family alias for an Orin Nano/NX on the reference carrier.
sudo ./flash.sh --read-info --no-flash jetson-orin-nano-devkit internal
cat bootloader/readinfocmd.txt
# After inspecting the generated read-only command:
cd bootloader
sudo bash readinfocmd.txt
```

Record the module, SKU, FAB/revision, RAM code and carrier identification. `0955:7523` alone does not establish storage or exact SKU. Official EEPROM mappings: P3767-0003 and P3767-0005 are Orin Nano 8GB; P3767-0004 is 4GB; P3767-0000/0001 are Orin NX. The reference carrier is P3768. A custom carrier requires its own compatible BSP configuration. [EEPROM layout](https://docs.nvidia.com/jetson/archives/r39.2.1/DeveloperGuide/HR/JetsonEepromLayout.html), [Orin Nano/NX carrier requirements](https://docs.nvidia.com/jetson/archives/r39.2.1/DeveloperGuide/HR/JetsonModuleAdaptationAndBringUp/JetsonOrinNxNanoSeries.html)

EEPROM does not enumerate installed NVMe/microSD capacities. NVIDIA documents a recovery `--initrd` boot for entering a shell. **Inspect the shipped `--initrd` control path first**, including invoked scripts and initrd startup, to ensure it exits before automatic flash or filesystem mounts/writes. Then the candidate command is:

```sh
sudo ./tools/kernel_flash/l4t_initrd_flash.sh \
  --initrd --showlogs jetson-orin-nano-devkit internal
```

This uploads/runs firmware and Linux in RAM, so it changes transient device state even though the intended operation writes no persistent storage. It is the appropriate reenumeration test. Do not copy the cloning documentation's `/etc/fstab` edit or `dd`/mount steps: those are unnecessary for inventory. [Flashing support: initrd clone shell](https://docs.nvidia.com/jetson/archives/r39.2.1/DeveloperGuide/SD/FlashingSupport.html#cloning-rootfs-with-initrd)

Once the recovery shell is reachable, run only inventory commands:

```sh
uname -a
tr '\0' '\n' < /proc/device-tree/model
cat /proc/meminfo
lsblk -b -o NAME,SIZE,MODEL,SERIAL,TYPE,FSTYPE,MOUNTPOINTS
blkid
cat /proc/partitions
cat /proc/mounts
```

Use available minimal-shell equivalents if `lsblk` is absent. Do not mount storage to discover its size. Capture real device names, capacities, partition signatures, and mounted state before selecting a flash target.

Initrd flashing subsequently depends on guest NFS/SSH over USB IPv6 `fc00:1:1::/48`. The USB NIC must appear **inside the Linux guest**, with its firewall permitting that path. A Mac USB NIC alone is not sufficient. Use addresses/interface names printed by the actual R39.2.1 tool rather than assuming the ordinary installed-OS address `192.168.55.1`. [Initrd requirements](https://docs.nvidia.com/jetson/archives/r39.2.1/DeveloperGuide/SD/FlashingSupport.html#requirements)

## Proceed / stop gates

1. **Guest gate:** Ubuntu amd64 boots, networking works, and the NVIDIA host binaries run. Allocate a growable Linux disk with enough physical free space for rootfs plus generated images; 100GB maximum virtual size is a reasonable engineering starting point, not a documented minimum.
2. **Recovery gate:** the guest sees exactly the intended APX device and the read-info transaction completes. If unsupported host binaries, denied USB capture, or repeated transport errors occur, resolve those before RAM boot.
3. **Reenumeration gate:** RAM-only recovery boot returns as a guest USB network device, shell/NFS connectivity works, and storage inspection is repeatable. One success should be followed by one repeat of this non-writing transition if practical. No successful USB transition means no storage-writing attempt.
4. **Target gate:** identify the actual module/carrier and intended storage from evidence. Use `jetson-orin-nano-devkit-super` for the final reference-devkit configuration if appropriate. Derive the complete flash command from the **shipped R39.2.1 script and README**, with actual storage capacity. The generic docs default to media at least 64GB; smaller layouts need explicit sizing.
5. **Bounded recovery effort:** allow one default passthrough attempt and up to two changes justified by specific logs (privilege/driver capture, matching, or reset policy). Stop repeating the same failed transport. If the VM cannot hold the RAM-boot transition, switch to a native Ubuntu x86_64 host connected to this USB device, or the official USB-stick ISO route if media/display access is available. This is a transport fallback, not a claim that Apple Silicon flashing is impossible.

The ISO route uses a separate USB installer and the Jetson's own UEFI; it is not a file sent to APX. JetPack 7.2+ no longer provides the old SD image workflow. It still needs an identified target drive and working firmware/capsule update path. [Official Orin Nano quick start](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/quick_start.html)

No command in this research has flashed the board, and no VM/USB success is being claimed. The next useful result is an actual board-information log, followed by a RAM-boot storage inventory.
