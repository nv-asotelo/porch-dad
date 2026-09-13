# 01 — Hardware and flashing JetPack 7.2.1

This document covers getting a Jetson Orin Nano Super 8 GB from a box to a booted JetPack 7.2.1
(L4T R39.2.1) system running from an NVMe SSD, with CUDA, cuDNN, TensorRT and DeepStream actually
present on the device.

Everything downstream in this repo — building TensorRT engines, running the resident shim, the INT4
quantization work described in [../docs/report.md](../docs/report.md) — assumes the verification
checklist at the end of this page passes. If it does not, stop here; a half-provisioned image fails
later in ways that look like runtime bugs.

Three bring-up failures are documented below with symptom, cause and fix. All three were hit on this
deployment. They are ordered by when they bite you, not by severity.

---

## 1. Hardware and host prerequisites

| Item | What was used here | Notes |
|---|---|---|
| Compute module + carrier | Jetson Orin Nano Super, 8 GB LPDDR5 | Ampere sm_87. No FP8 or NVFP4 hardware — this constrains the whole optimization story later. |
| Boot/root storage | NVMe SSD, 915 GB usable as reported by the booted system | M.2 Key-M slot on the carrier. Capacity is not critical; engine builds and model weights want tens of GB, not hundreds. |
| Flashing cable | USB-C, data-capable, host → carrier board USB-C port | Many USB-C cables are charge-only. If `lsusb` never shows the board, suspect the cable before suspecting the board. |
| Power | The dev kit's barrel-jack supply | Do not try to flash while powering the board from the host's USB port. |
| Jumper | One female-to-female jumper wire (or a 2.54 mm shunt) | Required for Force Recovery Mode — see §2. |
| Host machine | x86_64 Ubuntu desktop running NVIDIA SDK Manager | SDK Manager needs a real desktop session; it is not a headless-friendly tool. The exact SDK Manager build used here was not recorded — use whichever version offers JetPack 7.2.1 in its release list. |
| Host disk | Tens of GB free under the SDK Manager download/target directories | The host stages the full BSP and all target debs before pushing anything. |

Target software stack for this deployment:

| Component | Version |
|---|---|
| JetPack | 7.2.1 |
| L4T | R39.2.1 |
| OS | Ubuntu 24.04 |
| CUDA | 13.2 |
| TensorRT | 10.16.2.10 |
| cuDNN | 9.20 |
| DeepStream | 9.1 |

A note that matters later: TensorRT 10.16 is a hard ceiling on this image. At least one
TensorRT-Edge-LLM optimization path (`USE_TRT_NATIVE_ATTN=1`, fused ViT attention) requires
TensorRT ≥ 11 and fails on JetPack 7.2.1 with a misleading "Plugin not found" error. See
[../docs/report.md](../docs/report.md) for the full account. Do not expect a JetPack update to fix
that within the 7.2.x line.

---

## 2. Entering Force Recovery Mode

The host cannot flash a board that is running normally. The Tegra has to be brought up with its boot
ROM in Force Recovery (APX) mode, which is done by holding the `FC REC` (force recovery) pin low
**while power is applied**.

Two practical points that cost real time on this build:

- **A case-mounted recovery button does nothing on a fresh board.** The Elecrow-style cases expose a
  recovery button wired to the carrier's button header, but the button path depends on firmware that
  is not present until the board has been flashed at least once. On an out-of-the-box board the
  button is inert. **Use a physical jumper.**
- The jumper must be in place *before* power is applied and can be removed once the board is up in
  recovery mode.

Procedure:

1. Disconnect power from the carrier board.
2. On the carrier's button header, short the pin labelled `FC REC` to an adjacent `GND` pin with a
   jumper wire or shunt. Check the pin positions against NVIDIA's carrier board specification for
   your exact carrier revision — the specific pin numbers were not recorded for this deployment, and
   guessing on a powered board is a bad trade.
3. Connect the USB-C cable from the carrier to the Ubuntu host.
4. Apply power.
5. Verify with `lsusb` on the host (§3), then remove the jumper.

The board in recovery mode does nothing visible: no display output, no network, no SSH. `lsusb` on
the host is the only confirmation you get.

---

## 3. Verifying recovery mode with `lsusb`

Run this on the **host**, not the Jetson:

```bash
lsusb | grep -i 0955
```

Interpretation of the USB ID, which is the single most useful diagnostic during bring-up:

| USB ID | Meaning |
|---|---|
| `0955:7523` | NVIDIA APX device — the board is in Force Recovery Mode and is flashable. |
| `0955:7020` | The board enumerated as a normally-booted L4T device. Not flashable; the jumper was not applied, or was applied after power-on. |
| *(nothing)* | The host does not see the board at all. Check the USB-C cable (charge-only cables are the usual culprit), the port, and that the carrier actually has power. |

Only `0955:7523` means you can proceed. If you see `0955:7020`, power down, re-apply the jumper, and
power up again.

---

## 4. Flashing to NVMe with SDK Manager

With the board enumerated as `0955:7523`:

1. Launch SDK Manager on the Ubuntu host and log in with your NVIDIA developer account.
2. Select the target hardware (Jetson Orin Nano) and **JetPack 7.2.1** as the release.
3. Accept the download of both the *Jetson Linux* (BSP/OS) and *Jetson SDK Components*
   (CUDA, cuDNN, TensorRT, DeepStream, and friends) groups.
4. When SDK Manager reaches the flash configuration dialog, set:
   - **OEM configuration**: pre-config or runtime, your choice. Pre-config lets you set the username
     and password up front, which avoids needing a monitor and keyboard for first boot. The device
     user for this deployment is `orin`, which is why every later command in this repo uses
     `orin@jetson.local`.
   - **Storage device**: **NVMe**. This is not the default — see §5.2.
5. Flash. The board reboots into the new image when the OS flash completes.
6. SDK Manager then asks for the device's IP address and credentials so it can install the SDK
   components *over the network*. This step is where the third failure below happens (§5.3).

After the OS flash, the device is reachable over SSH. Set the host-side environment variable that the
rest of this repo's tooling uses:

```bash
export JETSON_HOST=orin@jetson.local   # or orin@<your-device-hostname>
ssh "$JETSON_HOST" 'uname -a'
```

The two host-side scripts ([../scripts/collect_perf.py](../scripts/collect_perf.py) and
[../scripts/compare_perf.py](../scripts/compare_perf.py)) read `JETSON_HOST` and default to
`orin@jetson.local`; `profile_fixed.py`, `tok_vs_res.py` and `quality.py` run on the device itself
against `http://127.0.0.1:8000`. Never hardcode an address.

---

## 5. The three failures hit on this deployment

### 5.1 SDK Manager fails when IPv6 is disabled on the host

**Symptom.** SDK Manager fails during download/setup or fails to establish its connection to the
device. The error text does not mention IPv6.

**Cause.** SDK Manager requires IPv6 to be enabled on the host. Many hardened or corporate-managed
Ubuntu installs disable IPv6 via `sysctl`, which silently breaks it.

**Fix.** Re-enable IPv6 on the host before starting SDK Manager:

```bash
sudo sysctl -w net.ipv6.conf.all.disable_ipv6=0
sudo sysctl -w net.ipv6.conf.default.disable_ipv6=0
```

To confirm, both should read `0`:

```bash
sysctl net.ipv6.conf.all.disable_ipv6 net.ipv6.conf.default.disable_ipv6
```

These `sysctl -w` settings do not survive a reboot. If your host's IPv6 is disabled by a file in
`/etc/sysctl.d/`, you will need to edit or override that file to make the change persistent, and you
may want to revert it afterwards if it was set deliberately by your IT policy. Restart SDK Manager
after making the change — it does not re-probe.

### 5.2 SDK Manager defaults the storage target to the SD slot, not NVMe

**Symptom.** The flash "succeeds" but the system boots from (or tries to boot from) the SD card slot,
`/dev/mmcblk0`. On a board with no SD card the flash fails outright; with an SD card present you get
a working but wrong installation, on slow storage with little space.

**Cause.** The storage target dropdown in SDK Manager's flash configuration dialog defaults to the SD
slot on Orin Nano. Having an NVMe drive installed does not change the default.

**Fix.** Explicitly select **NVMe** as the storage device in the flash configuration dialog before
starting the flash. There is no way to migrate afterwards that is faster than reflashing correctly.

**Verify after boot** — the root filesystem should be on an `nvme0n1` partition, not `mmcblk0`:

```bash
ssh "$JETSON_HOST" 'findmnt -n -o SOURCE,FSTYPE,SIZE /'
ssh "$JETSON_HOST" 'lsblk -o NAME,SIZE,TYPE,MOUNTPOINT'
```

This deployment runs root on NVMe with 915 GB of capacity reported by the booted system.

### 5.3 SDK Manager flashes the OS but never pushes the target components

**Symptom.** The board boots into a working JetPack 7.2.1 system, but CUDA, cuDNN, TensorRT and
DeepStream are simply not installed. `nvcc` is not found; `import tensorrt` fails; there is no
`/usr/local/cuda`. SDK Manager may report success, or may fail at the post-flash "install SDK
components" step after the OS flash has already completed.

**Cause.** The component-install phase is a *separate* network-based step performed after the flash,
over SSH to the freshly booted device. It is fragile: it depends on the device being reachable, on
the credentials entered matching, and on the host-side networking (including §5.1) still being sane
after the board reboots and re-enumerates. When it does not run, you are left with a bare L4T image.

**Fix used here.** Install the target components manually on the device from the debs SDK Manager
already downloaded on the host. SDK Manager stages them under its download directory on the host
(by default under `~/Downloads/nvidia/sdkm_downloads`; confirm the actual path in SDK Manager's
settings rather than assuming it).

```bash
# On the host: copy the staged debs to the device.
# Adjust the source path to your SDK Manager download directory.
# Create the destination directory first: scp will not create it, and with a single
# matching .deb it would silently write a *file* named jetpack-debs instead.
ssh "$JETSON_HOST" 'mkdir -p /tmp/jetpack-debs'
scp ~/Downloads/nvidia/sdkm_downloads/*.deb "$JETSON_HOST":/tmp/jetpack-debs/
```

```bash
# On the device: install, letting apt resolve inter-package dependencies.
sudo apt-get update
sudo apt-get install -y /tmp/jetpack-debs/*.deb
```

`apt-get install` on local `.deb` paths is preferable to `dpkg -i` because it resolves dependencies
from the repository instead of leaving half-configured packages. If ordering problems persist,
install in dependency order — CUDA first, then cuDNN, then TensorRT, then DeepStream — and re-run
`sudo apt-get -f install` between groups.

The exact deb filenames and the exact install ordering are not recorded for this deployment, so treat
the commands above as the shape of the fix rather than a transcript. What *is* verified is the
outcome: the four component stacks in the checklist below were present and working on the device
after a manual install, and were not installed by SDK Manager.

---

## 6. Verification checklist

Run these on the device. Each one should report the version in the "expect" column. The literal
output text of these commands was not captured during this deployment, so the checklist specifies the
version string to look for, not a full expected transcript.

```bash
export JETSON_HOST=orin@jetson.local
ssh "$JETSON_HOST"
```

| # | Check | Command (on device) | Expect |
|---|---|---|---|
| 1 | L4T release | `cat /etc/nv_tegra_release` | R39, revision 2.1 |
| 2 | JetPack meta-package | `apt-cache show nvidia-jetpack \| grep -m1 Version` | 7.2.1 |
| 3 | L4T core package | `dpkg-query --show nvidia-l4t-core` | R39.2.1 |
| 4 | OS | `lsb_release -d` | Ubuntu 24.04 |
| 5 | Root on NVMe | `findmnt -n -o SOURCE /` | an `nvme0n1` partition, **not** `mmcblk0` |
| 6 | CUDA toolkit | `nvcc --version` | release 13.2 |
| 7 | CUDA packages | `dpkg -l \| grep -i cuda-toolkit` | 13.2 |
| 8 | cuDNN | `dpkg -l \| grep -i cudnn` | 9.20 |
| 9 | TensorRT | `dpkg -l \| grep -i tensorrt` | 10.16.2.10 |
| 10 | TensorRT from Python | `python3 -c 'import tensorrt; print(tensorrt.__version__)'` | 10.16.2.10 |
| 11 | DeepStream | `deepstream-app --version` | 9.1 |
| 12 | GPU visible | `nvidia-smi` or `tegrastats` | the iGPU is present and reporting |
| 13 | Memory | `free -h` | ~8 GB total (7.x GB usable), plus the 2 GB swapfile used here |

If `nvcc` is not on `PATH`, it is at `/usr/local/cuda/bin/nvcc`; add `/usr/local/cuda/bin` to `PATH`
and `/usr/local/cuda/lib64` to `LD_LIBRARY_PATH` for the service user (this repo's shim unit,
[../systemd/cosmos3-edge-shim.service](../systemd/cosmos3-edge-shim.service), puts
`/usr/local/cuda/bin` on the unit's `PATH` explicitly).

Failure of checks 6–11 almost always means §5.3 — the OS flashed, the components did not.

### Image quirks worth knowing before you script against this device

- **`/usr/bin/time` does not exist** on this JetPack image. Any benchmarking script that shells out to
  `/usr/bin/time` fails with rc=127 and, depending on how you capture it, produces an empty
  measurement rather than an obvious error. Use in-process timing instead; the measurement scripts in
  [../scripts](../scripts) do.
- Power mode and clocks (`nvpmodel -m 2` for MAXN_SUPER, plus `jetson_clocks`) are **not** part of
  flashing and are not set by the checklist above. They are a significant part of the measured
  performance on this deployment and are covered with the rest of the runtime setup — see
  [../docs/report.md](../docs/report.md).

---

## 7. What this page deliberately does not cover

- Installing TensorRT-Edge-LLM, building engines, or the INT4 quantization recipe.
- Power-mode and clock configuration.
- The shim, the systemd units, or the Live VLM WebUI front end.

Those follow in the rest of the `deploy/` series and in [../docs/report.md](../docs/report.md).
