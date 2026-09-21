# JetPack 7.2.1 ISO and USB recovery assets

Checked 2026-09-19. Inspection was read-only: archive listing and selected file extraction from `downloads/jetsoninstaller-r39.2.1-arm64.iso` into `.qa/iso-inspect`. No target commands, mounts, flash operations or fuse operations were executed.

## Decision

Use the separate official **Jetson Linux R39.2.1 BSP plus sample root filesystem** for the host-driven USB RCM workflow. The existing ISO is a UEFI-bootable ARM64 installer with target OS layers and firmware-update packages. It is not a ready `Linux_for_Tegra` host recovery bundle. NVIDIA documents these as distinct installation paths. [Official Orin Nano BSP setup](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/setup_bsp.html).

The official download page links directly to:

- [Jetson_Linux_R39.2.1_aarch64.tbz2 — BSP](https://developer.nvidia.com/downloads/embedded/L4T/r39_Release_v2.1/release/Jetson_Linux_R39.2.1_aarch64.tbz2)
- [Tegra_Linux_Sample-Root-Filesystem_R39.2.1_aarch64.tbz2 — sample rootfs](https://developer.nvidia.com/downloads/embedded/L4T/r39_Release_v2.1/release/Tegra_Linux_Sample-Root-Filesystem_R39.2.1_aarch64.tbz2)

Links were extracted from [NVIDIA's current JetPack download page](https://developer.nvidia.com/embedded/jetpack/downloads); neither tarball was downloaded by this research subtask.

## What is actually in the downloaded ISO

`file` identifies an ISO 9660 bootable CD image labeled `jetsoninstaller-r39.2.1`. The outer archive has 3,790 entries, including:

- `boot/grub/grub.cfg`, ARM64 EFI files, `casper/Image`, and `casper/initrd`.
- Ubuntu minimal/server Squashfs layers and `ubuntu-server-minimal.ubuntu-server.installer.kernel.nvidia.squashfs`.
- A `pool/` apt repository with ARM64 packages at version `39.2.1-20260806224157`.
- `nvidia-l4t-bsp` is a 20,286-byte metapackage; its data archive contains only documentation. It is not the driver-package tarball despite its name.
- `nvidia-l4t-bootloader` contains target firmware capsules such as `TEGRA_BL_3767.Cap`, `TEGRA_BL_3767_super.Cap` and `TEGRA_BL_3767_nanoe8gb_super.Cap`, plus ARM64 boot EFI files.
- `nvidia-l4t-bootloader-utils` contains target utilities including `/usr/sbin/nvbootctrl` and `/usr/sbin/nv_bootloader_capsule_updater.sh`.

No `flash.sh`, `tegrarcm_v2`, or complete `Linux_for_Tegra` tree appears in the outer ISO listing or the inspected BSP/bootloader/tools packages. Squashfs interiors were not fully expanded; this is not a claim that every byte of every layer was searched. The official ISO architecture confirms its contents are installer and target rootfs layers, packages and boot assets. Target capsules are not substitutes for host TegraFlash tools, signed RCM applets, BCTs and board configurations. [Official ISO customization guide](https://docs.nvidia.com/jetson/archives/r39.2.1/DeveloperGuide/SD/JetsonISOCustomization.html).

It might be possible to reconstruct a rootfs from the layered ISO plus apt packages, but this is additional custom image work, not the documented sample-rootfs recovery path. Reuse the ISO for installation media or its rescue shell when existing UEFI works.

## Safest recovery preflight sequence

These are reference commands for a prepared Ubuntu x86_64 host, **not commands executed here**. Confirm board configuration from the package and EEPROM; do not hardcode an assumed SKU from the USB ID alone.

1. Query USB enumeration with `lsusb -d 0955:7523`. APX proves recovery enumeration, not available NVMe/microSD storage or installed JetPack.

2. The official Orin ECID query is:

   ```sh
   sudo ./bootloader/tegrarcm_v2 --new_session --chip 0x23 --uid
   ```

   This requests the chip UID; it does not flash storage or program fuses. The example is documented in the official R36.2 Orin debugging guide; check the R39.2.1 binary's help before use. [NVIDIA ECID example](https://docs.nvidia.com/jetson/archives/r36.2/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/DebuggingOnJetsonPlatforms.html).

3. R39.2.1 documents `flash.sh --read-info` for board/chip/EEPROM and fuse **reading**. To prepare an inspectable read command first:

   ```sh
   sudo ./flash.sh --read-info --no-flash jetson-orin-nano-devkit internal
   # Review bootloader/readinfocmd.txt, then execute that read script in RCM.
   ```

   This combination generates `readinfocmd.txt`. The separate `--no-flash --no-systemimg` workflow retrieves `cvm.bin` and `chip_info.bin_bak`; `chkbdinfo` decodes board/SKU/FAB/RAMCODE locally. **Do not run the generated `flashcmd.txt` for inspection:** that is a flash script. [R39.2.1 flashing support](https://docs.nvidia.com/jetson/archives/r39.2.1/DeveloperGuide/SD/FlashingSupport.html#other-examples-of-using-flash-sh).

4. Storage discovery requires a running kernel/initrd with NVMe/SD drivers. Use the BSP's documented `l4t_initrd_flash.sh --initrd <board> <rootdev>` boot-only workflow after reviewing its bundled README. It permits a shell without performing the normal flash workflow. Then run read-only queries such as `lsblk -b -o NAME,SIZE,TYPE,MODEL,SERIAL,FSTYPE,MOUNTPOINTS`, `cat /proc/partitions`, and `cat /proc/mtd`. Confirm actual device names and capacities before choosing a target layout. No partitioning, formatting, cloning, or mount changes are needed for these queries. [Official initrd shell entry](https://docs.nvidia.com/jetson/archives/r39.2.1/DeveloperGuide/SD/FlashingSupport.html#cloning-rootfs-with-initrd).

5. On an ordinary boot of the installed firmware, `sudo nvbootctrl dump-slots-info` reports current firmware version, slot state and capsule status. UART cold-boot output is another source. Record whether any version came from original cold boot or freshly RCM-loaded binaries; loading a new boot chain into RAM is not proof of the preexisting QSPI version. [Boot slot information](https://docs.nvidia.com/jetson/archives/r39.2.1/DeveloperGuide/SD/Bootloader/UpdateAndRedundancy.html#to-show-the-slot-status).

## Rescue and host constraints

The ISO's inspected GRUB menu has **Boot Into Rescue Shell**, using `boot-live-env systemd.unit=rescue.target`; it also has installer targets for NVMe and microSD. This route needs working JetPack 6.x-generation or newer UEFI/QSPI and bootable installation media. It cannot be sent directly to bare APX as an ISO. Selecting an install target invokes autoinstall; use the rescue entry for inspection.

NVIDIA specifies an Ubuntu x86_64 host for host flashing. An x86 Linux VM on Apple Silicon is a compatibility experiment, not the documented host setup. USB passthrough must survive recovery-device re-enumeration, and initrd flash requires USB device networking with IPv6 `fc00:1:1::/48`, NFS and SSH. Verify these before an actual flash.

Keep default USB recovery behavior. NVIDIA's separate “enable USB3 recovery” procedure programs fuses and is not part of this inspection or a required recovery prerequisite. No FSKP, fuseburn, EEPROM override, capsule update or flash operation is needed to obtain the above preflight evidence.
