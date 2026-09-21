# Confirmed board: recovery flash choices

Static audit dated 2026-09-19, using the completed public NVIDIA R39.2.1 BSP and its matching public documentation. This subtask executed no hardware command and changed no BSP file. The primary agent reports a successful recovery read: `BOARDID=3767`, `FAB=300`, `BOARDSKU=0005`, `CHIP_SKU=00:00:00:D5`, `RAMCODE=2`. Actual storage remains unconfirmed at the time of this note.

**Use `jetson-orin-nano-devkit-super` with an explicitly identified target device.** NVIDIA lists P3767-0005 as the development Orin Nano 8GB module, supporting QSPI plus microSD, USB or NVMe through initrd flash on the reference P3768 carrier. The module read alone does not identify a custom carrier; these choices assume the reported developer kit/reference carrier. The Super configuration makes higher power and clock modes available; it does not require immediately selecting the highest power mode. [NVIDIA R39.2.1 Quick Start](https://docs.nvidia.com/jetson/archives/r39.2.1/DeveloperGuide/IN/QuickStart.html)

## Exact command choices

These are **prospective full flashes**, to run only after the primary agent identifies the intended media and confirms its contents can be replaced. Each command replaces QSPI firmware and repartitions/formats the selected drive. Removing `--erase-all` does **not** preserve data on that selected drive. Run in the prepared Linux host's `/home/flash/Linux_for_Tegra`, with the actual board in recovery and the matching populated rootfs/NVIDIA binaries already prepared.

NVMe, when inventory identifies the intended SSD as `nvme0n1`:

```bash
sudo ./tools/kernel_flash/l4t_initrd_flash.sh \
  --external-device nvme0n1p1 \
  -c tools/kernel_flash/flash_l4t_t234_nvme.xml \
  -p "-c bootloader/generic/cfg/flash_t234_qspi.xml" \
  -S 55GiB --showlogs --network usb0 \
  jetson-orin-nano-devkit-super internal
```

microSD, when inventory identifies the intended card as `mmcblk0`:

```bash
sudo ./tools/kernel_flash/l4t_initrd_flash.sh \
  --external-device mmcblk0p1 \
  -c tools/kernel_flash/flash_l4t_t234_nvme.xml \
  -p "-c bootloader/generic/cfg/flash_t234_qspi.xml" \
  -S 55GiB --showlogs --network usb0 \
  jetson-orin-nano-devkit-super internal
```

The explicit XML and `-S` arguments spell out the matching defaults. The XML's name contains `nvme`, but its device is `type="external"`; the script supports `mmcblk` and routes that layout to `--external-device`. NVIDIA's R39.2.1 Quick Start explicitly uses `--external-device mmcblk0p1 ... jetson-orin-nano-devkit-super internal` for microSD. Its `internal` final argument is a rootfs/boot-argument choice, **not a storage-selection safeguard**. Bare initrd flash inherits `EXTERNAL_DEVICE="nvme0n1p1"`, even with final argument `internal`. [Quick Start](https://docs.nvidia.com/jetson/archives/r39.2.1/DeveloperGuide/IN/QuickStart.html), [device validation](/Users/asotelo/Documents/ChatGPT/jetson-orin-streaming-codes/.qa/bsp-inspect/Linux_for_Tegra/tools/kernel_flash/l4t_create_images_for_kernel_flash.sh:478), [defaults](/Users/asotelo/Documents/ChatGPT/jetson-orin-streaming-codes/.qa/bsp-inspect/Linux_for_Tegra/p3768-0000-p3767-0000-a0.conf:88)

With **two NVMe devices**, choose the inventoried SSD's node explicitly and use final argument `external` so the generated root argument uses PARTUUID. NVIDIA documents C4 as `nvme0n1` and C7 as `nvme1n1` when both are populated; a single SSD in either slot normally becomes `nvme0n1`. Do not substitute `nvme1n1p1` based only on a slot assumption. [NVIDIA Orin initrd instructions](https://docs.nvidia.com/jetson/archives/r39.2.1/DeveloperGuide/SD/FlashingSupport.html#using-initrd-flash-with-orin-nx-and-nano)

For a reviewable preparation stage, add `--no-flash` to the chosen complete command. Inspect the newly generated `tools/kernel_flash/images/internal/flash.idx`, `images/external/flash.idx`, accompanying `flash.cfg`/XML and `initrdflashparam.txt` before the actual flash. Internal entries should address QSPI only; external entries must address exactly the chosen external layout, with the expected APP image/size. Confirm the stored external-device argument too. `--no-flash` still prepares host files and may query the connected board; it is not a pure dry-run. Do not reuse a previous `--flash-only` package after switching media or board options; regenerate and inspect the package. The original complete command without `--no-flash` can regenerate and then flash the reviewed choice.

## Sizing and expansion

The actual R39.2.1 common configuration sets **55GiB APP** and **57GiB nominal device layout**. The XML uses `EXT_NUM_SECTORS`; `flash.sh` defaults this to `57 * 1024^3 / 512 = 119537664` sectors. Therefore the physical disk must have at least **61,203,283,968 bytes**, with sufficient host free space for image generation. A nominal 64GB card normally clears that size, but inventory must establish actual bytes. Both public instructions and this default target media of at least 64GB. [BSP constants](/Users/asotelo/Documents/ChatGPT/jetson-orin-streaming-codes/.qa/bsp-inspect/Linux_for_Tegra/p3767.conf.common:165), [sector computation](/Users/asotelo/Documents/ChatGPT/jetson-orin-streaming-codes/.qa/bsp-inspect/Linux_for_Tegra/flash.sh:2156), [NVIDIA flashing sizing](https://docs.nvidia.com/jetson/archives/r39.2.1/DeveloperGuide/SD/FlashingSupport.html)

`-S` sets the initial APP size, not the total drive capacity. Do not set `-S` to all available bytes: kernel, recovery, EFI, UDA, GPT and reserved partitions also occupy the disk. For this bounded deployment, retain 55GiB initially on supported media rather than introducing a custom layout. A smaller card requires coordinated APP and sector-count changes, not only `-S`, and is outside this prepared default.

APP has partition ID 1, is physically the last data partition, and has allocation attribute `0x808`. The target `create_gpt` routine corrects the secondary GPT for a larger physical disk and expands the partition when the generated flash-index attribute marks it `expand`. Its raw-image writer runs `resize2fs`; tar-rootfs mode formats the full destination partition. Sparse/zstd-image paths differ, so verify `lsblk` and `df` after boot before claiming all SSD capacity is usable. Retaining the default initial APP size also avoids unnecessarily generating a drive-sized host image. [External XML](/Users/asotelo/Documents/ChatGPT/jetson-orin-streaming-codes/.qa/bsp-inspect/Linux_for_Tegra/tools/kernel_flash/flash_l4t_t234_nvme.xml:157), [GPT expansion](/Users/asotelo/Documents/ChatGPT/jetson-orin-streaming-codes/.qa/bsp-inspect/Linux_for_Tegra/tools/kernel_flash/l4t_flash_from_kernel.sh:1183), [rootfs writers](/Users/asotelo/Documents/ChatGPT/jetson-orin-streaming-codes/.qa/bsp-inspect/Linux_for_Tegra/tools/kernel_flash/l4t_flash_from_kernel.sh:1037)

## Media scope and Super selection

- **Keep `--erase-all` absent.** Although the Quick Start describes erasing the flashed boot media, the supplied Orin gadget setup contains unconditional discard attempts for every present node among `mmcblk0`, `nvme0n1`, `sda`, `mmcblk0boot0`, and `mmcblk0boot1` when this flag is set. A separate branch also discards an explicitly selected external device. This is broader than the chosen layout. Normal flashing still replaces the selected drive's partition table and filesystems. [Exact gadget path](/Users/asotelo/Documents/ChatGPT/jetson-orin-streaming-codes/.qa/bsp-inspect/Linux_for_Tegra/tools/kernel_flash/initrd_flash/nv_enable_remote.sh:188)
- **Use the QSPI-only internal XML.** `flash_t234_qspi.xml` contains one SPI device (64MiB) and no SD/eMMC/NVMe user device. The external XML contains one external device. This is the intended two-layout initrd arrangement. The base `EMMC_CFG` of `flash_t234_qspi_sd.xml` also contains SD storage; do not use that as the internal layout while flashing an unrelated NVMe. [QSPI layout](/Users/asotelo/Documents/ChatGPT/jetson-orin-streaming-codes/.qa/bsp-inspect/Linux_for_Tegra/bootloader/generic/cfg/flash_t234_qspi.xml:2), [internal-layout propagation](/Users/asotelo/Documents/ChatGPT/jetson-orin-streaming-codes/.qa/bsp-inspect/Linux_for_Tegra/tools/kernel_flash/l4t_initrd_flash_internal.sh:183)
- **Keep `--direct`, `--external-only`, `--qspi-only`, `--append`, `-k`, `--erase-all`, `--boot-rootfs` and stale image-reuse flags out of this initial full-install command.** They change which media/partitions are affected or skip a required portion. `--direct` targets host-attached storage; `--external-only` skips the required coordinated QSPI update. Do not inherit manual board identity, ROOTFS_AB, ROOTFS_ENC, ERASE_QSPI, UPHYLANE or layout overrides from another operation.
- **Match identity and capacity immediately before flashing.** Use the RAM-boot inventory's storage type, model/serial where available, size and device node. Minimize attached storage. If the target node or contents are uncertain, resolve that specific uncertainty first. The command cannot distinguish two same-named drives across changed attachments.
- **Use the plain Super alias for initrd.** It resolves to `p3768-0000-p3767-0000-super.conf`, which selects SKU0005's `tegra234-p3768-0000+p3767-0005-nv-super.dtb` and the Nano 8GB Super BPMP DTB. The `-super-nvme` alias changes legacy `EMMC_CFG`; it is unnecessary with the explicit initrd layouts above. Do not select an emulation alias or replace the confirmed SKU with 0003. [Super selection](/Users/asotelo/Documents/ChatGPT/jetson-orin-streaming-codes/.qa/bsp-inspect/Linux_for_Tegra/p3768-0000-p3767-0000-super.conf:34)

After successful boot, confirm the release, actual RAM and root filesystem device; ensure UEFI selected the freshly flashed media. Then install/verify the matching JetPack compute components and measure available memory before model work. No flash, capacity, firmware installation or performance success is asserted by this static audit.
