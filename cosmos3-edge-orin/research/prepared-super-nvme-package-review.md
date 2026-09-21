# Super NVMe package review

Status: **completed package passes this structural review; APP is fixed at 55 GiB, and payload checksum validation remains with the primary agent before flashing.** The preparation log reached `Finish generating flash package.` before the final read-only snapshot at `2026-09-20T02:19:15Z`. Review scope was `/home/flash/Linux_for_Tegra/tools/kernel_flash/images` in the Ubuntu guest, using NVIDIA Jetson Linux R39.2.1. No target commands, guest writes, flash operations, or package payload checksum computations were performed by this reviewer.

## Scope and observed identity

The preparation log starts with `--no-flash --external-device nvme0n1p1 -c tools/kernel_flash/flash_l4t_t234_nvme.xml -p '-c bootloader/generic/cfg/flash_t234_qspi.xml' -S 55GiB ... jetson-orin-nano-devkit-super internal`. Both internal and external child `flash.sh` invocations include `--no-flash --sign`. Generated `secureflash` command text in that log is an output artifact of preparation, not evidence of executing a storage write.

The identity chain is explicit:

1. `.qa/jetson-read-info-cold.log:349–361` records actual chip information and CVM EEPROM reads succeeding after the cold recovery reset; lines 382–387 report `BOARDID=3767`, `FAB=300`, `BOARDSKU=0005`, `CHIP_SKU=00:00:00:D5`, `RAMCODE_ID=2`. `results/jetson-board.json` records that successful read.
2. During preparation, read-only inspection of only the allowed identity variables in the active guest processes confirmed the inherited environment: the same five values plus `BOARDREV=T.1`. `SKIP_EEPROM_CHECK` was not present in the selected environment fields.
3. NVIDIA `l4t_create_images_for_kernel_flash.sh:599–604` exports these board variables to `flash.sh`. `flash.sh:3620–3656` accepts supplied identity for offline generation and assigns `RAMCODE=RAMCODE_ID`. The live preparation log independently prints the same board, RAMCODE and Super specification for both internal and external generation.
4. `bootloader/ecid.bin` contains matching values, with the normalized chip SKU `D5`. This file is **generated from the supplied preparation values** (`flash.sh:3711–3722`); it is corroboration of configuration, not an independent hardware read.

The original read-info script derives board ID/FAB/SKU/revision with `chkbdinfo` from `cvm.bin`, and RAMCODE from the last byte of `chkbdinfo -R chip_info.bin_bak`, modulo 16 (`flash.sh:1369–1375`). Read-only identity-asset fingerprints in the guest were `cvm.bin` SHA-256 `42448a117c84d05707f1df79f0e0afbc5269906ff36a4b6927b0f71eed18d245` and `chip_info.bin_bak` SHA-256 `7f8a94319855063f6c67f01380a368eb3443b227a6fca02e56657d100c774b87`. These are local fingerprints, not publisher signatures.

## Internal package: structurally consistent

`internal/flash.idx` has 61 rows (indices 0–60), and every row has device type and instance `3:0`, NVIDIA's SPI identifier. It has no APP partition and no eMMC, SD or external-storage device entries. Its final `secondary_gpt` starts at byte 67,091,968 with size 16,896: the layout ends at exactly 67,108,864 bytes (64 MiB). Firmware A/B entries, QSPI variables/reserved regions and QSPI GPT records are expected here.

All nonempty internal image references exist, have the byte lengths recorded in the index, fit their declared partition, and resolve inside the `images` tree. Every nonempty index image hash is syntactically a 40-digit hexadecimal SHA-1. The completed package tree has no symlinks. Reserved entries with empty image filenames are intentional index entries, not missing files. The final read-only metadata snapshot is `.qa/package-review-metadata.json`, produced by sending `.qa/review_flash_package_metadata.py` to guest Python on stdin; it reports no structural issues and does not compute payload hashes.

`internal/flash.cfg` is exactly:

```sh
external_device=nvme0n1p1
CHIPID=0x23
```

The external-device setting in this shared configuration does not turn the SPI-only index into an SD or NVMe partition list. The official target flasher dispatches by each index row's type; `SPI_DEVICE=3`, and the reviewed internal rows only exercise that path. The external package has been separately reviewed below.

The alias `jetson-orin-nano-devkit-super.conf` resolves to `p3768-0000-p3767-0000-super.conf`. That configuration explicitly maps SKU `0005` to:

- Kernel/boot DTB: `tegra234-p3768-0000+p3767-0005-nv-super.dtb`.
- BPMP DTB: `tegra234-bpmp-3767-0003-3768-super.dtb`.

Thus the internal A/B BPMP entries containing `3767-0003-3768-super` are correct for this `0005` board, not a SKU mismatch. A/B DCE images contain the `p3767-0005-nv-super` DTB name. The preparation log also records copying that exact Super DTB into the root filesystem.

## External package: structurally consistent, fixed 55 GiB APP

`external/flash.idx` contains 18 rows, all type/instance `9:0` (external storage), with no SPI, SD or eMMC entries. The final configuration is exactly:

```sh
APP_ext=system.img
external_device=nvme0n1p1
CHIPID=0x23
```

The selected whole disk is therefore `/dev/nvme0n1`, with APP partition `/dev/nvme0n1p1`. Both A/B kernel-DTB entries use `kernel_tegra234-p3768-0000+p3767-0005-nv-super.dtb`; the recovery DTB is the corresponding `.rec` file. All nonempty image references exist, have the index byte lengths, fit their allocated partitions and stay within `images`. No absolute or parent-traversing image paths or package symlinks were found.

| Item | Generated value |
| --- | --- |
| Nominal NVMe layout | 61,203,283,968 bytes (57 GiB), 119,537,664 sectors of 512 bytes |
| APP partition number | 1 |
| APP start | 1,603,567,616 bytes; LBA 3,131,968 |
| APP size | 59,055,800,320 bytes (55 GiB) |
| APP last LBA | 118,475,327 |
| APP index attribute | `fixed-<reserved>-1` |
| APP payload selected by configuration | `external/system.img`, 2,774,391,691 bytes |
| APP payload magic | `28b52ffd` (zstd), filename has no `.zst` suffix |
| APP sidecar | `system.img.sha1sum`, 41 bytes; digest `1c4d9c03d0fc840ed2fd5348af6554f6618e65bc` |

The APP size/start were independently decoded from the generated primary GPT binary and agree with the index. The index has an intentionally empty APP filename/size/hash; `APP_ext` selects the archive. The selected payload and SHA-1 sidecar are regular files contained within the export. The sidecar has valid digest-only syntax, but this reviewer did not calculate the archive hash.

**The final index marks APP fixed, despite `0x808` in the source XML template.** The target `create_gpt` code only grows a partition when an index row starts with `expand`. Consequently this reviewed package will relocate the backup GPT to the actual larger NVMe but will not grow APP during flashing. This conclusion uses the generated index and binary GPT; it supersedes the earlier template-based expectation of expansion. Inspect the partition after boot before claiming additional usable space; any deliberate later expansion belongs to the deployment stage.

The primary agent reports the actual target contains only one WD_BLACK SN7100 NVMe device, 1,000,204,886,016 bytes, with no partitions and no SD card. This reviewer has not independently queried the target. That reported capacity comfortably exceeds the generated 57 GiB layout. The primary agent owns actual payload checksum verification and a final target identity/selection check before flashing.

The official APP extraction path does not enforce its commented-out SHA-1 comparison; a sidecar's existence alone is not content verification. This remains a required primary-agent pre-flash check. Check the APP payload named by `APP_ext` against its own `.sha1sum`. The generator copies `flash.idx` before preparing APP's final payload (`l4t_create_images_for_kernel_flash.sh:309,352–420`); in this actual package, APP's index image fields are empty. Other non-APP image bytes should match their index size/hash.

The target dispatches APP using the configuration variable, not the index filename. This zstd payload without a `.zst` suffix selects archive extraction, including `mkfs.ext4` on the selected 55 GiB APP partition. A `.zst` suffix would select compressed disk-image writing. The archive path formats the partition after the GPT step; `create_gpt`'s optional expansion is gated by the index `expand` attribute (`l4t_flash_from_kernel.sh:1195–1209`), which this actual APP row does not have.

No `--erase-all`, `--direct`, host-device override or SD layout is part of the reviewed preparation command.

Sources: generated guest `images/{internal,external}/{flash.idx,flash.cfg}`, external `system.img.sha1sum`, selected payload file metadata/magic, decoded `external/gpt_primary_9_0.bin`, board alias/configuration, filtered preparation process environment; local `.qa/prepare-super-nvme.log`, `.qa/jetson-read-info-cold.log`, `results/jetson-board.json`; immutable public BSP scripts extracted under `.qa/bsp-inspect/Linux_for_Tegra/`, particularly `flash.sh`, `p3768-0000-p3767-0000-super.conf`, `tools/kernel_flash/l4t_create_images_for_kernel_flash.sh` and `tools/kernel_flash/l4t_flash_from_kernel.sh`.
