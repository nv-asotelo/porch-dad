# NVMe completion timeouts: investigation, no fix yet

Recorded 2026-09-20 UTC. Normal Jetson boot is working; the official QSPI/NVMe flash and online filesystem growth completed. This note does not claim that inference or storage reliability is validated.

## Observed platform evidence

The Orin Nano runs L4T R39.2.1, kernel `6.8.12-1021-tegra`, with a WD_BLACK SN7100 1 TB root NVMe, firmware `7615M0WD`, PCI device Sandisk `5045`. A cold TensorRT import eventually succeeded after roughly two minutes. The initial log excerpt contained three NVMe warnings near uptime 518, 550, and 580 seconds: `I/O tag ... QID 1/3 timeout, completion polled`. The later full capture includes two additional boot-stage warnings, for five total; their exact times and the subsequent measured window are recorded below.

The captured `.qa/nvme-diagnostics.log` shows all six CPUs online, stock 25 W power mode, PCIe Gen3 x4, MSI-X enabled, endpoint ASPM disabled, and all L1 substates disabled. Endpoint AER status bits are clear in that snapshot. NVMe IRQ counters accrue only on CPU0 despite per-queue configured affinity masks spanning CPU0–5; effective-affinity lists are empty. The NVMe module's default maximum power-state latency is 100000 microseconds. That parameter alone does not establish the controller's active APST table. The endpoint is under PCIe controller `14160000.pcie`.

## Public evidence and interpretation

- [Linux v6.8 `nvme_timeout`, lines 1304–1317](https://github.com/torvalds/linux/blob/v6.8/drivers/nvme/host/pci.c#L1304-L1317) polls for a missed interrupt and emits this warning when the request is no longer in flight. This is a recovered completion at timeout, not proof of a failed SSD power-state exit. Lost or delayed interrupt handling is the first hypothesis to investigate; the warning alone does not identify its cause.
- [NVIDIA forum, August 2026: JetPack 7.2, Orin, SN7100 and KC3000](https://forums.developer.nvidia.com/t/nvme-timeout-completion-polled/380388) contains a closely matching first-hand report. The reporter found Gen3 limiting ineffective. NVIDIA staff suggested checking an older MSI/GICv2m patch, but the reporter's attempted JP7.2 adaptation then disabled MSI/MSI-X and reduced throughput. There is no verified R39.2.1 fix in that thread.
- [Older R36.3 GICv2m investigation](https://forums.developer.nvidia.com/t/r36-3-patch-to-re-enable-gicv2m-for-pcie-msi-interrupts-and-restore-i-o-performance/297495) documents CPU0 interrupt concentration with DesignWare MSI routing. [Upstream v6.8 DesignWare code](https://github.com/torvalds/linux/blob/v6.8/drivers/pci/controller/dwc/pcie-designware-host.c) rejects individual MSI affinity changes in `dw_pci_msi_set_affinity`. These support examining actual counters and the parent interrupt; configured masks are insufficient. They do not justify transplanting an older kernel/device-tree patch.
- [Linux NVMe developer guidance on per-device APST control](https://lists.infradead.org/pipermail/linux-nvme/2024-January/044719.html) describes a reversible controller-local latency-tolerance setting. APST concerns the NVMe controller's internal power states; ASPM concerns the PCIe link. No APST fault has been established here, and ASPM is already disabled.

## Next observation

Run the sampler during an already-planned native/model build, without adding disk stress or changing settings:

```sh
python3 /home/jetson/cosmos-edge/scripts/sample_storage_irqs.py \
  /home/jetson/cosmos-edge/results/storage-irqs-build-01.jsonl \
  --interval 1 --duration 1800
```

The output path must be new and its parent must exist. No root access or extra packages are required. The sampler reads only `/proc/interrupts` entries containing `nvme` or `14160000.pcie`, `/proc/stat` CPU0, `/proc/uptime`, and `/proc/diskstats` for `nvme0n1`. It records UTC, monotonic time, read duration, missing/read errors, cumulative counters, and an end record. SIGINT/SIGTERM flush and stop; exit codes are 130/143. Ordinary completion exits zero. Flushes occur after the first sample reaching each five-second threshold; a longer interval or blocked filesystem operation can delay timing or flushing. There is no `fsync`, daemon, root operation, or device-setting write. JSONL output itself creates a small amount of disk activity. Missed sampling slots are skipped, not replayed in a burst.

Retain contemporaneous kernel messages separately, using kernel uptime timestamps for correlation. The sampler does not capture dmesg. Compare deltas around the pauses: parent/queue interrupt progress, CPU0 IRQ/softirq ticks, completed disk I/O, and in-flight I/O. Counters alone cannot prove a lost interrupt. CPU counters are in the recorded `SC_CLK_TCK` units; guest ticks overlap user/nice, and disk sectors are 512 bytes. Do not add overlapping counters or treat `io_in_progress` as cumulative. Snapshots are sequential rather than atomic.

If available, a read-only `nvme get-feature /dev/nvme0 -f 0x0c -H` can capture the actual APST table. Only a reproducible idle-related failure with otherwise healthy IRQ service would justify a later isolated APST comparison, saving and restoring the per-controller setting. No kernel patch, boot flag, affinity change, APST change, or hardware operation was performed for this investigation. **No fix is established or applied.**

## Measured observation — 2026-09-20, 04:50:35–05:20:35 UTC

The sampler completed its requested 1,800-second duration during the native backend build. Its **1,800 samples had no recorded read errors**, with first/last samples at uptime **1208.77–3007.77 seconds**. Counter differences therefore span 1,798.999934 seconds; the duration-complete end record follows the last sample by one second. The largest sample gap was 1.0021 seconds and the longest `/proc` read sequence was 2.54 ms. Sources and exact arithmetic are retained in [storage-observation.json](../results/storage-observation.json) and [the raw JSONL](../results/storage-irqs-build-01.jsonl). The earlier 436-sample summary matches the corresponding prefix of this completed recording.

The final separately captured kernel log contains **five** `timeout, completion polled` messages:

| Observed stage | Kernel uptime, seconds | Queue |
| --- | --- | --- |
| Boot | 36.593037; 68.593005 | QID 4; QID 4 |
| First TensorRT import | 518.407859; 550.375268; 580.582783 | QID 1; QID 3; QID 3 |

**All five precede the sampler. No additional completion-polled timeout appears in the captured log during the sampled window.** The JSONL does not capture kernel messages itself; this comparison uses the later `.qa/final-storage-observation.log`. That absence does not resolve the earlier faults or establish storage reliability.

| Counter observation over the sampled window | Measured change |
| --- | --- |
| Completed reads / writes | 5,906 / 11,067 |
| Read / written bytes, using 512-byte diskstats sectors | 326,746,112 / 1,001,964,032 bytes, or **311.61 / 955.55 MiB** |
| Mean read / write traffic over the whole window | 0.173 / 0.531 MiB/s; ordinary build/OS activity, not an SSD bandwidth benchmark |
| Read / write accumulated request time | 834 / 13,777 ms; cumulative diskstats counters, not wall-clock stall duration |
| Diskstats I/O time / weighted I/O time | 5,144 / 18,407 ms |
| Flushes | 400 completed; 85 ms accumulated flush time |
| Discards | 8,436 completed; logical ranges totaling **882.70 GiB**. This is discard/TRIM range coverage, not read/write traffic, physical NAND erase volume, or measured throughput. |
| In-flight I/O snapshots | First and last: 0; sampled maximum: 1, in 2 of 1,800 samples. Activity between samples remains unobserved. |
| NVMe queue interrupt increments | **23,809, all on CPU0**; q0–q6: 0, 2,878, 6,994, 8,617, 4,138, 750, 432. CPU1–5 increments were zero. |
| CPU0 IRQ / softirq / iowait time | 7.51 / 1.26 / 0.26 seconds at the recorded 100 ticks/s |
| CPU0 time distribution | User 30.452%, system 0.855%, idle 68.190%, IRQ 0.418%, softirq 0.070%, iowait 0.014%; denominator is non-overlapping CPU0 tick deltas. |

The selected disk and queue IRQ counters did not regress, and the seven captured queue identities stayed stable. No separate `14160000.pcie` parent-controller IRQ row was present in these samples, so parent IRQ service cannot be compared from this recording. Disk counters cover the whole NVMe device, including other OS processes, sampler output, and background activity; they are not attributable exclusively to compiler work.

This interval demonstrates continued disk and queue-interrupt activity without a new captured timeout during the build. It does not reproduce the earlier cold-stage events, identify their cause, or prove that CPU0 interrupt concentration causes them. **No kernel, boot, affinity, or power-state setting was changed, and no fix is claimed.** These storage observations are separate from model inference and latency/memory measurements.

## Later cold-reload recurrence

A new `nvme0 ... timeout, completion polled` event occurred at kernel uptime 9550.525721 during a later cold FP16 backend reload. The process subsequently reached ready and completed all six image requests with outputs identical to the corrected FP16 baseline. This precedes the static-clock benchmark. The earlier 30-minute observation is a limited window and does not establish a fix. No kernel or NVMe power-management setting was changed. Startup can be delayed by this unresolved storage observation; steady-state request timing and the final soak are reported separately. See [the recurrence receipt](../results/nvme-cold-reload-observation.json).

## Final service audit

The final [service audit](../results/final-service-audit.json) captures nine completion-polled NVMe timeout warnings across this boot, through monotonic kernel time 10025.646445. Both resident services remained active with zero restarts, and the subsequent 600.7-second inference soak passed. The audit preserves all kernel warnings rather than treating a quiet observation window as a fix. No NVMe/APST/ASPM remediation was applied, and no storage fix is claimed.
