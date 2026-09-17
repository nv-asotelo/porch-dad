# 09 — Moorebot Scout: surviving the vendor's end of life

Moorebot appears to be defunct — the company still runs a "zombie" cloud (up, unmaintained) but is
not developing or supporting the Scout. When that cloud finally goes dark, the phone app stops
working. The question this document answers is: **what actually breaks, and what have we secured
against it while access was still possible.** Everything below was verified against the live robot
at `192.168.7.6` on 2026-09-17.

## The good news: the robot does not need the cloud to run

This was the central fear (repeated by an LLM advisor): that the robot "won't boot past its
cloud-handshake phase" once the servers die. **That is not true of this firmware**, established
three ways:

1. **The whole ROS graph runs right now with no app and no cloud connected.** We have been
   streaming its camera and driving it over `/cmd_vel` for a day with nothing logged into the
   vendor cloud. Camera, motors, ToF, the ROS master — all local.
2. **No boot script contacts the cloud.** The boot chain is a Rockchip-style init that loads the
   motor kernel modules, starts pulseaudio, then runs `/usr/local/bin/start.sh` (the roller_eye
   launcher) and `/bin/ota_daemon.sh`. Grepping the boot scripts for `amazonaws|http|curl|wget|
   server|cloud` finds nothing. The cloud connection is made *inside* the ROS nodes (`/CloudNode`,
   `/AppNode`) as an outbound connection; if it cannot reach home, those nodes simply fail to
   connect and everything else carries on.
3. **OTA is reactive, not a poller.** `ota_daemon.sh` watches for local flag files in `/userdata/`
   (`ota_needed_start`, `ota_installing`, `scout_resetting`) that the cloud would drop after
   sending an update command. It never contacts a server on its own. A dead company sends no update
   command, so there is no OTA and therefore no risk of a firmware update locking the device.

So the realistic EOL outcome is mild: the **app** stops working, the robot keeps working locally.
Our integration (`nvr/scout/`) already replaces the app for camera and drive, and talks only to the
robot's local ROS master — never the cloud.

## What the robot is

| | |
|---|---|
| Hostname | `linaro-alip` (stock Linaro ALIP image) |
| OS | Debian 9 (stretch), kernel 4.4.189 aarch64 (Rockchip BSP) |
| ROS | Melodic, package `roller_eye` at `/opt/ros/melodic/{share,lib}/roller_eye` |
| Storage | 14.6 GB eMMC; `/` is `mmcblk0p8` (2.5 GB), `/userdata` is `mmcblk0p9` (8.9 GB) |
| Config | `/var/roller_eye/config/` |

Note it is **Debian 9 + ROS Melodic**, not the "Ubuntu 18.04" some guides claim, and the Jetson
side needs **no `ros1_bridge`** — our bridge is already a native ROS1 Melodic client in a container,
which is why it talks to the robot directly.

## Access (secured while we could)

* **SSH works with the stock user:** `ssh linaro@192.168.7.6`, password `linaro`. This is a
  publicly documented default, not a secret, and a defunct vendor will not be changing it. It is
  the durable local back door.
* **`root/plt` does NOT work**, and **`sudo` is deliberately crippled** — the setuid bit is
  stripped from `/usr/bin/sudo`, so `linaro` cannot escalate on its own.
* **Root IS available** through the phone app's "elevated rights" feature, which sets a root SSH
  password. With that one-time root window, a **durable SSH key** (the Jetson's
  `~/.ssh/scout_ed25519`) was installed into both `/root/.ssh/authorized_keys` and
  `/home/linaro/.ssh/authorized_keys` (whose directory was root-owned and had to be chowned to
  `linaro` first). **Passwordless key login now works for both accounts and no longer depends on
  any password** — this is the permanent back door, and it survives a password change or the app
  revoking its grant. The root password itself is deliberately **not recorded here**; the key
  supersedes it.
* `linaro` was added to the **`audio` group** (`usermod -aG audio linaro`) so audio capture and
  playback work unprivileged — see the audio section below.

## Preservation backup

A read-only backup was streamed off the robot to the Jetson (nothing written to the robot):

```
/home/orin/backups/scout/run-<timestamp>/
  scout-rootfs-partial.tar.gz   ~31 MB, 257 files: the full roller_eye stack
                                (27 compiled node binaries, launch, msg/srv, meshes),
                                /var/roller_eye config, /usr/local/bin, boot scripts,
                                sshd_config and NetworkManager config
  boot-scripts.txt              start.sh, ota_daemon.sh, ota_install.sh verbatim
  roller_eye-config.txt         /var/roller_eye/config dumped
  dpkg.txt, systemd-units.txt, os.txt, ros.txt, launch-and-services.txt
```

This is **not committed to this repo** — it is Moorebot's firmware, not ours, and stays on the
Jetson only. It exists so the local ROS stack can be understood or reconstructed if the device is
ever wiped or replaced.

## Audio (two-way, local)

The Scout has a mic and speaker but exposes **no ROS audio and no RTSP audio** — its audio path is
the closed cloud P2P stack. Local audio is nonetheless possible through ALSA now that `linaro` has
audio-group access, and the codec is **not** held exclusively by the cloud stack (verified: nothing
in `fuser /dev/snd/*`, and a capture succeeded while the vendor services ran).

Card 0 is the Rockchip rk809 codec with two devices:

| ALSA device | Role | Working params (measured) |
|---|---|---|
| `hw:0,1` | microphone (PDM voice) | `S16_LE`, **2 channels** (mono is rejected), 16 kHz — captured a live signal at peak 28006/32767 |
| `hw:0,0` | speaker (i2s hifi) | `S16_LE`, 2 channels, 8–96 kHz |

So the local audio path is: `ssh linaro@scout` (key auth) → `arecord -D hw:0,1 -f S16_LE -c 2 -r 16000`
for the mic, and pipe PCM to `aplay -D hw:0,0` for the speaker, relayed to the browser by porch-dad
over a WebSocket. Both capture channels carry the same voice; downmix to mono for transport.
Half-duplex push-to-talk is the honest design — there is no on-robot echo cancellation, so a live
mic and live speaker would feed back.

## What was deliberately NOT done, and why

An LLM advisor recommended several hardening steps. These were declined:

* **Disabling the OTA / phone-home boot scripts.** Requires root (unavailable), and — more to the
  point — is unnecessary: OTA is cloud-command-driven and a dead cloud sends no commands. Editing
  boot scripts on a headless embedded device you cannot easily recover is the definition of a
  hard-to-reverse, brick-risk change, taken here for no benefit.
* **`chmod 4755 /usr/bin/sudo`** to restore escalation. Requires root to perform (circular), and
  modifying a setuid security binary on the robot is not worth it when password login already
  gives the access we need.
* **A `ros1_bridge` container / ROS 2 on the Jetson.** The Jetson does not run ROS 2 for this; the
  bridge is already a native ROS1 client. No bridge-of-a-bridge needed.

## The one thing worth doing that we cannot do from here

**Isolate the Scout from the internet at the router.** Give it a static DHCP reservation and a
firewall rule dropping all WAN traffic to/from its MAC. It has no need to reach the internet — all
our use is LAN-local — and cutting it off means a compromised or misbehaving zombie-cloud endpoint
can never reach it, and the robot never wastes boot time waiting on servers that are gone. This is
a router change, so it is the user's to make.
