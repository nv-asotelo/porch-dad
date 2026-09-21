# Expand APP after the first successful NVMe boot

**Recommendation:** use the installed `growpart` followed by online `resize2fs`, after verifying the live disk against the generated layout. This is a procedure, not an executed resize. This review accessed only guest files and public documentation; it did not access the Jetson or change the VM/device.

The generated primary GPT proves that APP is **partition 1 and physically last**. Partitions 2–15 occupy the earlier LBAs; partition 15 (`reserved`) ends at LBA 3,131,967, immediately before APP begins. There is no auxiliary partition after APP to move. The current APP covers LBAs 3,131,968–118,475,327, exactly 55 GiB. Its index attribute is `fixed`, so the NVIDIA flasher repairs the secondary GPT's location on the larger drive but does not automatically enlarge APP. See [the package review](/Users/asotelo/Documents/ChatGPT/jetson-orin-streaming-codes/research/prepared-super-nvme-package-review.md) and `.qa/package-review-metadata.json`.

Growing APP into the available tail space is appropriate for this single-rootfs deployment. On the observed 1,000,204,886,016-byte, 512-byte-sector NVMe, the installed growpart algorithm predicts approximately **930.02 GiB** for APP (998,601,301,504 bytes), leaving the existing auxiliary partitions and secondary-GPT space intact. Use the actual dry-run output as authoritative. Filesystem capacity and available space will be smaller because of metadata and reserved blocks.

The prepared rootfs already contains `cloud-guest-utils 0.33-1`, `fdisk/util-linux 2.39.3`, `gdisk 1.0.10` and `e2fsprogs 1.47.0`; no installation is expected. The standard `/usr/lib/nvidia/resizefs/nvresizefs.sh` path is absent. The installed growpart script finds the next partition by its start sector, changes the selected size while keeping its start, and requests a kernel partition update. Its GPT path reserves the secondary header. [Canonical growpart documentation](https://github.com/canonical/cloud-utils/blob/main/man/growpart.1), [public 0.33 source](https://github.com/canonical/cloud-utils/blob/0.33/bin/growpart).

## Procedure on the booted Jetson

Run these in the **installed Jetson OS**, once the first NVMe boot and SSH are working, before model deployment. Keep the official flash package available. Confirm `/` resolves to `/dev/nvme0n1p1`, is mounted as writable ext4, and the disk is the inventoried WD_BLACK SN7100. If those facts or the partition ordering differ, reassess the actual layout before writing.

```bash
findmnt -n -o SOURCE,FSTYPE,OPTIONS /
lsblk -b -o NAME,TYPE,SIZE,START,FSTYPE,PARTLABEL,PARTUUID,MOUNTPOINTS,MODEL,SERIAL /dev/nvme0n1
sudo blockdev --getss /dev/nvme0n1
sudo blockdev --getsize64 /dev/nvme0n1
sudo sgdisk --verify /dev/nvme0n1
```

Expect 512-byte sectors, 1,000,204,886,016 bytes and a valid GPT. Verify APP remains last in LBA order, not partition-number order. A different APP size may mean first-boot code already expanded it. If GPT verification reports corruption or an unmoved backup header, resolve that specific condition before the resize; do not add an unconditional GPT-repair operation.

Save partition metadata and copy these two files back to the task workspace before changing the table. The GPT backup preserves partition metadata; it is not a backup of filesystem contents.

```bash
resize_record=$(mktemp -d "$HOME/cosmos-storage-resize.XXXXXX")
sudo sgdisk --backup="$resize_record/nvme-before.gpt" /dev/nvme0n1
sudo sfdisk --json /dev/nvme0n1 > "$resize_record/before.json"
sudo growpart --dry-run --verbose /dev/nvme0n1 1
```

The dry run must keep APP start **3,131,968**, partition number, UUID, type and label unchanged, increase only its size/end, and preserve every entry for partitions 2–15. Expected existing APP size is 115,343,360 sectors; predicted new size is 1,950,393,167 sectors with this disk and installed growpart version. Exit 0 means a proposed change; exit 1 can mean no additional space. An already expanded partition needs no repeated table change.

After reviewing the proposed change, perform it once and capture the resulting table:

```bash
sudo growpart --update=on /dev/nvme0n1 1
sudo udevadm settle
sudo sfdisk --json /dev/nvme0n1 > "$resize_record/after.json"
sudo sgdisk --verify /dev/nvme0n1
```

Compare the actual tables before filesystem resizing:

```bash
python3 - "$resize_record/before.json" "$resize_record/after.json" <<'PY'
import json, sys
before, after = [json.load(open(p))['partitiontable'] for p in sys.argv[1:]]
assert before['id'] == after['id'], 'disk GUID changed'
b = {p['node']: p for p in before['partitions']}
a = {p['node']: p for p in after['partitions']}
assert a.keys() == b.keys(), 'partition set changed'
root = '/dev/nvme0n1p1'
assert b[root]['start'] == 3131968
for node in b:
    if node != root:
        assert a[node] == b[node], f'auxiliary partition changed: {node}'
    else:
        assert a[node]['size'] >= b[node]['size']
        assert {k:v for k,v in a[node].items() if k != 'size'} == {
            k:v for k,v in b[node].items() if k != 'size'}, 'APP metadata changed'
print('Partition metadata preserved. Expected APP bytes:', a[root]['size'] * 512)
PY
sudo blockdev --getsize64 /dev/nvme0n1p1
```

The last command must report the enlarged byte count printed by the comparison. If the on-disk table grew but the kernel still reports 55 GiB, do not repeat growpart or start filesystem resizing: update partition 1 with `sudo partx --update --nr 1 /dev/nvme0n1`, recheck, and use a controlled reboot if the kernel still cannot refresh it. A command failure can occur after a table write, so inspect the table before deciding how to continue.

Once the kernel sees the larger partition:

```bash
sudo resize2fs /dev/nvme0n1p1
findmnt -n -o SOURCE,FSTYPE,OPTIONS /
df -hT /
sudo sgdisk --verify /dev/nvme0n1
```

`resize2fs` without a size uses the partition capacity; ext4 supports online growth, so `/` can remain mounted. If it reports a filesystem error, stop for a maintenance diagnosis rather than forcing resize or running `e2fsck` on the mounted root. Retain the before/after JSON, command output and final capacity as deployment evidence. [Upstream resize2fs manual](https://man7.org/linux/man-pages/man8/resize2fs.8.html).

NVIDIA documents the same partition-then-filesystem sequence for its SD image first-boot resizing. For this NVMe, suitability is established by the actual decoded GPT and R39.2.1 `l4t_flash_from_kernel.sh:1182–1209`, whose expansion branch depends on an `expand` index attribute. No auxiliary partition relocation, formatting, reflashing, QSPI change or manual partition recreation is needed. [NVIDIA R39.2.1 resizing explanation](https://docs.nvidia.com/jetson/archives/r39.2.1/DeveloperGuide/SD/FlashingSupport.html#resizing-the-root-partition-to-fill-the-available-sd-card-space).
