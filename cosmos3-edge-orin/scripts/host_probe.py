#!/usr/bin/env python3
"""Read-only macOS probe for a USB-connected Jetson and installation media."""
import datetime
import glob
import json
import platform
import plistlib
import subprocess


def plist_command(argv):
    return plistlib.loads(subprocess.check_output(argv, timeout=20))


def walk(nodes):
    if isinstance(nodes, dict):
        nodes = [nodes]
    for node in nodes:
        yield node
        yield from walk(node.get('IORegistryEntryChildren', []))


def main():
    if platform.system() != 'Darwin':
        raise SystemExit('Run this host probe on macOS. Use device_inventory.sh on Jetson.')
    usb = plist_command(['ioreg', '-p', 'IOUSB', '-a', '-l'])
    nodes = list(walk(usb))
    nvidia = []
    for node in nodes:
        if node.get('idVendor') == 0x0955:
            nvidia.append({
                'name': node.get('USB Product Name', node.get('IORegistryEntryName')),
                'vendor_id': '0955',
                'product_id': f"{node.get('idProduct', 0):04x}",
                'location_id': node.get('locationID'),
            })
    disks = plist_command(['diskutil', 'list', '-plist', 'external', 'physical'])
    report = {
        'observed_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'host_os': platform.system(), 'host_arch': platform.machine(),
        'enumerated_usb_devices': sum(n.get('IOObjectClass') == 'IOUSBHostDevice' for n in nodes),
        'nvidia_usb_devices': nvidia,
        'usb_serial_ports': glob.glob('/dev/cu.usb*'),
        'external_physical_disks': [
            {'identifier': d['DeviceIdentifier'], 'size_bytes': d.get('Size')}
            for d in disks.get('AllDisksAndPartitions', [])
        ],
        'device_flash_performed': False,
    }
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
