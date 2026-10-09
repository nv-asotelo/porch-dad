# VITURE Luma Ultra on the Jetson Orin Nano

The VITURE Luma Ultra XR glasses plug into the Orin Nano Developer Kit's USB-C port, and their
camera becomes a Live Vision source (**Use VITURE Luma Ultra**, [`../nvr/README.md`](../nvr/README.md),
"VITURE Luma Ultra"). This page is what the glasses present to the Orin, what works without any extra
hardware, and what each remaining function - display, head tracking - would need.

Measured 2026-10-09 on this Orin: Jetson Orin Nano Engineering Reference Developer Kit Super,
L4T R39.2.1, kernel 6.8.12-1021-tegra.

## What works out of the box: the camera, the microphone

Plugged straight into the dev kit's USB-C port, the glasses come up as a small USB 2.0 tree behind
their own hub - nothing to install, no driver to build:

| USB ID | Name | What the Orin makes of it |
|---|---|---|
| `1a86:8091` | USB HUB (WCH) | the glasses' internal 4-port USB 2.0 hub |
| `0c45:636b` | USB 2.0 Camera (Sonix) | a standard UVC webcam, `uvcvideo`: `/dev/video0` (picture) and `/dev/video1` (metadata) |
| `35ca:1102` | VITURE Microphone | USB audio capture (ALSA card "Microphone"), and three HID interfaces: two 64-byte in/out pairs as `hidraw0`, `hidraw1`; the third, an 8-byte one, is refused by `hid-generic` ("unbalanced collection at end of report description") |
| `35ca:1104` | VITURE Luma Ultra XR GLASSES | a vendor-specific interface (bulk and interrupt endpoints), for VITURE's own software |

The camera offers MJPEG at every size up to 1920x1080, at 5-30 fps, and uncompressed YUY2 at up to
1920x1080 but only 5 fps there. Its MJPEG frames carry their own Huffman tables, so browsers decode
them as they come. The L4T kernel already has everything this needs: `uvcvideo` (a module, loaded
when the camera appears), USB HID with `hidraw`, and USB audio. The `orin` user is in `video` and
`plugdev`.

Check it from a shell:

```bash
lsusb | grep -E "0c45:636b|35ca:"                 # the camera and both VITURE devices
ls -l /dev/v4l/by-id/                              # usb-Sonix_..._USB_2.0_Camera_...-video-index0
gst-device-monitor-1.0 Video/Source | grep -m3 image/jpeg
gst-launch-1.0 -q v4l2src device=/dev/video0 num-buffers=30 \
  ! image/jpeg,width=1920,height=1080,framerate=30/1 ! multifilesink location=/tmp/luma-%02d.jpg
```

(JetPack's `ffmpeg` is built without V4L2 input, so use GStreamer; `v4l2-ctl` is in `v4l-utils`,
not installed by default.)

For Live Vision, install the camera bridge and give Live Vision `--luma-url`
([`../nvr/luma/README.md`](../nvr/luma/README.md)).

### When nothing appears

The first time the glasses were plugged in on 2026-10-09, the port's Type-C controller (`fusb301`)
saw them and took the host role, and nothing else happened: no hub, no camera, no error. Unplugged and
plugged in again, they enumerated within a second. The controller logged exactly the same attach both
times (same orientation, same advertised current), so the cause is not known. If `lsusb` shows no
`35ca` device a few seconds after plugging in:

```bash
sudo journalctl -k -n 30 | grep -E "fusb301|usb 1-1"   # "fusb_update_state: 7" = attached as host
```

and replug the glasses. `fusb_update_state: 7` with no `usb 1-1: new high-speed USB device` after it
is this case. NVIDIA's forums report the same symptom on this dev kit's Type-C port with other
devices - only `fusb301` lines, nothing enumerating - and replugging recovered it there too. Seat the
glasses' magnetic connector fully: on another Linux host a partly seated one brought up the hub
without its devices, or `device descriptor read/64, error -71`, once the cameras drew current.

## What the Orin's USB-C port can and cannot give

The dev kit's USB-C port is the Orin's USB 2.0 OTG port (`usb2-0`) with a USB 3 lane (`usb3-1`),
switched between host and device by the `fusb301`: a Type-C controller (attach and detach, host or
device role, plug orientation, the advertised current) with no USB Power Delivery. As host it
supplies 5 V from the board's always-on `VDD_5V0_SYS` rail and advertises the default USB current
(`fusb301_set_dfp_power: host current(1)`). There is no DisplayPort Alt Mode on it: the board's
display output is its DisplayPort connector (`DP-1`). That settles what the glasses can do on this
port:

| Function | On the Orin's USB-C port | Why |
|---|---|---|
| Camera | **Yes** | plain UVC over USB 2.0 |
| Microphone | **Yes** | plain USB audio |
| Display | **No** | needs DisplayPort Alt Mode, which this port does not carry |
| Head tracking (IMU) | **Not without VITURE's software** | the data is on the HID and vendor interfaces above, in VITURE's own protocol |

The port's 5 V goes through an AP22811 load switch (which trips at 2.2-3.2 A), but NVIDIA's carrier
spec rates it, and each USB-A port, at 0.5 A. The `fusb301` driver hard-codes the Default
advertisement; its `fhostcur` sysfs file can raise it until the next attach, but that goes past
NVIDIA's rating, so don't. VITURE's own docks give the glasses 5 V at 1 A, and its HDMI adapter
0.6 A. With no display to drive, the glasses' camera and microphone ran from this port without a
fault.

This port is also its own USB 2.0 root port: the four USB-A ports share one Realtek hub. The camera's
highest isochronous setting reserves about 196 Mbit/s of USB 2.0's periodic bandwidth, so on a USB-A
port it would share that hub's budget with any other camera or audio device there. If this port ever
misbehaves, a USB-A port through a USB-C-to-A adapter is the same USB 2.0 connection (untried with
these glasses). Don't put a pass-through charging adapter between this port and the glasses: with a
charger attached it offers power to the Orin, and the port's controller, which tries to sink power
first, could turn the Orin into the USB device (untested, from the driver's policy).

## Display: from the DisplayPort connector, through a DP-to-USB-C adapter

The Orin cannot send video over USB-C - not this port, not any Orin. NVIDIA's dev kit guide says so
outright: "The USB-C port does not output a display signal. HDMI or DisplayPort over USB-C are not
supported." The hardware agrees: the Orin Nano's one display output goes only to the DisplayPort
connector, the USB-C connector's SBU pins (DP AUX in Alt Mode) are unconnected, and the `fusb301` has
no USB Power Delivery to enter an Alt Mode with. (The carrier spec's block diagram labels the port's
SuperSpeed pair "DP0_x (USBSS1)", after the module pins' names, but on the Orin those pins carry USB
3.2 only.) Asked about DP over USB-C on a custom carrier board, NVIDIA's engineers answer "There is no
support for this on any Orin platform." No device tree, driver or JetPack patch adds it.

What VITURE and Linux users do instead, on hosts without DP over USB-C:

- **A one-way DP-to-USB-C (or HDMI-to-USB-C) adapter with a USB-A lead.** VITURE's own desktop guide
  prescribes it - "a DP + USB-A to USB-C adapter or an HDMI + USB-A to USB-C adapter ... the DP/HDMI
  plug provides the video, and the USB-A plug provides the data and power" - though it cannot vouch
  for adapters it doesn't make. The adapter presents a DP Alt Mode USB-C socket to the glasses and
  returns their USB 2.0 data to the host over the USB-A lead. The author of XRLinuxDriver and Breezy
  Desktop uses fairikabe's HDMI version ("you can get both video and USB data") and, asked about the
  Luma Ultra, answered "I believe I used it when testing 6DoF", adding that it has never worked in
  side-by-side 3D. An XREAL on a Raspberry Pi 5 got video, audio and head tracking from an HDMI+USB
  cable; a Windows user ran a Luma Ultra from a PC's DisplayPort through a "USB-C combiner" with
  SpaceWalker's 3DoF tracking.
- **Not** VITURE's HDMI XR Adapter or Pro Mobile Dock: both carry video and power only, with no USB
  path back (the HDMI adapter talks to VITURE's iPhone app over Bluetooth), so no camera, head
  tracking or brightness control reaches the host, and VITURE's desktop app does not support the HDMI
  adapter. VITURE's Mobile Dock Mini and USB-C XR Charging Adapter Ultra need a source that already
  has DP over USB-C. VirtualLink adapters are gone from the market.

For this Orin, the DisplayPort connector (`DP-1`) takes a DP-source adapter. Candidates sold for XR
glasses, none yet reported with a Luma Ultra or on a Jetson:

| Adapter | Why |
|---|---|
| fairikabe DisplayPort-to-USB-C cable, Amazon B0BW8QKFHK | DP source to USB-C display; a USB-A lead for data and a USB-C input for a charger (the listing asks for PD, 60 W); the DP sibling of the HDMI cable XRLinuxDriver's author uses |
| NEXHYPE DP-to-USB-C female adapter B0H1M2CQ3V | DP source to a USB-C socket; a USB-A lead, plus a "PD100W" USB-C input for a charger; listed for VITURE Luma |
| NEXHYPE DP-to-USB-C cable B0GCDNN5WT | DP source to USB-C; one USB-A lead carries both power and data (the listing warns its power is limited), so on the Orin it needs a powered hub, below; listed for Viture |
| A USB-C female-to-female coupler (XRLinuxDriver's author uses Amazon B08BFS89RG) | joins a converter's USB-C plug to the glasses' own USB-C-to-magnetic cable |

Avoid adapters that say they cannot "transmit data" (some only power the screen): without the USB
lead's data, the glasses' camera and head tracking never reach the Orin. NVIDIA's staff have said of
this board's DP output that "only native DP or active DP-HDMI adapter are working"; if a DP-to-USB-C
converter will not light the glasses, an active DP-to-HDMI adapter into fairikabe's HDMI-to-USB-C
cable (B0B5XBYQSM, the one XRLinuxDriver's author recalls using for 6DoF) is the fallback. That cable
draws its power, 5 V at 1 A, through its USB-A lead.

Wiring: Orin `DP-1` -> adapter -> the glasses' cable; the adapter's USB-A lead -> an Orin USB-A port
(the camera, microphone and head tracking come back this way, so the Luma camera bridge still finds the
camera, by its neighbours, on whichever port); the adapter's power input, if it has one -> a USB-C
charger, not the Orin. An adapter without one powers the glasses through its USB-A lead, and the
Orin's USB-A ports are rated for 0.5 A, half the 1 A VITURE's own docks give the glasses: put a
mains-powered USB hub, with ports rated 1 A or more, between that lead and the Orin. To check it,
`cat /sys/class/drm/card*-DP-1/status` should read `connected`, and the glasses' EDID is in
`/sys/class/drm/card*-DP-1/edid` (`edid-decode`).
The Orin Nano drives one 4K30 display over DP 1.2: 1920x1080 or 1920x1200 at 60 Hz in 2D is
comfortable, while the 3840-wide side-by-side 3D and 120 Hz modes need about the same 297 MHz pixel
clock as 4K30, the board's ceiling, and are untried. This Orin also boots headless
(`multi-user.target`), so a picture on the glasses needs a desktop session
(`sudo systemctl isolate graphical.target`) or an application drawing to DRM/KMS directly.

## Head tracking and glasses control: VITURE's SDK

The IMU, the two grayscale tracking cameras and the glasses' controls (display mode, brightness,
volume, the electrochromic film) speak VITURE's own protocol, on the `35ca:1104` vendor interface
and the `35ca:1102` HID interfaces. Reading them takes VITURE's **XR Glasses SDK** (`libglasses`, a
C library; v2.4.0 of 2026-08-06). It supports the Luma Ultra (its "Carina" family) on Linux aarch64
and gives:

- 3DoF or 6DoF pose by polling (`xr_device_provider_get_gl_pose_carina`), and the raw IMU;
- the two tracking cameras: 8-bit grayscale, 640x480 per eye, about 25 Hz;
- display mode (including 3840-wide side-by-side 3D), brightness, volume and film mode;
- the RGB camera too, but over its own libusb UVC client, which needs the kernel's `uvcvideo` detached
  first. Keep the camera on V4L2 and the bridge, and use the SDK for pose and controls only; one Linux
  project (aarch64 and x86_64) does exactly that.

Getting it: viture.com/developer, **Download** on the XR Glasses SDK opens a request form (target
platform "Linux (arm64)"), and the link arrives by email. The licence is personal and non-transferable,
object code only, and asks business users to contact VITURE first (bd@viture.com). XRLinuxDriver
(GPL-3.0) ships the v2.4.0 aarch64 libraries in its repo, built for Debian bookworm, for its own
driver.

On this Orin (Ubuntu 24.04 userland, glibc 2.39): `libglasses.so` needs only `libudev.so.1` and glibc
2.34 or later; `libcarina_vio.so` (6DoF, loaded at run time since v2.4.0) needs glibc 2.35 and OpenCV
4.6 (`.406` sonames, as Ubuntu 24.04 ships them; check `ldd` against JetPack's own OpenCV). Without
`libcarina_vio` the SDK still loads - in VITURE's words "6DoF simply becomes unavailable" - though
whether 3DoF then works is unconfirmed. Nobody has reported the SDK on a Jetson yet.

Permissions: VITURE's rule uses `TAG+="uaccess"`, which only grants access to a user logged in at the
seat, and this Orin runs headless. Use a group instead (`orin` is in `plugdev`):

```bash
sudo tee /etc/udev/rules.d/70-viture.rules >/dev/null <<'RULES'
SUBSYSTEM=="usb", ATTRS{idVendor}=="35ca", MODE="0660", GROUP="plugdev"
SUBSYSTEM=="hidraw", KERNEL=="hidraw[0-9]*", ATTRS{idVendor}=="35ca", MODE="0660", GROUP="plugdev"
RULES
sudo udevadm control --reload-rules && sudo udevadm trigger
```

Known issues from Linux users: the SDK can segfault in `libcarina_vio.so`
(`carina_a1088_read_custom_data`) when the glasses are plugged in at boot - unplug, restart the
program, replug; 6DoF runs on the host and takes about one full core on an x86 PC (more is likely on
the Orin's Cortex-A78AE); and it needs a well-lit, textured view.

## Firmware

Update the glasses' firmware first, from another machine: VITURE's updater runs only in Chrome on
Windows (with SpaceWalker installed) or macOS, or from VITURE's Android app, which VITURE says works
even on a phone without DP Alt Mode; there is no Linux tool.
This unit is the `35ca:1104` board revision of the Luma Ultra (from mid-2026; the first one was
`35ca:1101`).

The front camera ships behind a privacy sticker, and an indicator light on the glasses shows while it
runs - with the bridge, while Live Vision is watching it and for 30 s after.

## In short

| You want | You need |
|---|---|
| The camera in Live Vision | Nothing more: the glasses on the Orin's USB-C port, the bridge, `--luma-url` |
| The microphone | Nothing more: ALSA card "Microphone" |
| A picture in the glasses | A DP-to-USB-C adapter with a USB-A data lead, on `DP-1`, and power for it: a USB-C charger in its power input, or a powered USB hub for its USB-A lead |
| Head tracking, display modes | VITURE's XR Glasses SDK (request form), the udev rule above |

Sources: VITURE's XR Glasses SDK docs and release notes (viture.com/developer), its product and
Academy pages for the adapters, the Luma series and SpaceWalker for desktop; NVIDIA's Orin Nano
developer kit user guide and carrier board spec (SP-11324-001 v1.3) and its developer forums; the
adapters' Amazon listings; wheaney/XRLinuxDriver issues #78, #105, #122 and wheaney/breezy-desktop
issues #104, #120, #127, #172, #181; rohitsangwan01/Verto_XR #6;
brianhasquestions/Viture_AR_Playground; atacolak/xr; AchromaAssist/viture-ultra-ar-color-helper;
SalvatoreMastrangelo/Spatial-Screens - all read on 2026-10-09.
