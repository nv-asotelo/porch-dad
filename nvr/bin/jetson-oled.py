#!/usr/bin/env python3
import psutil
import time
import smbus
import os
from pathlib import Path

class JetsonSystemMonitor:
    def __init__(self):
        # Initialize I2C bus
        self.bus = smbus.SMBus(7)  # Use I2C-7
        self.i2c_address = 0x2D    # I2C slave address

    def get_cpu_usage(self):
        """Get CPU usage percentage"""
        return psutil.cpu_percent(interval=1)

    def get_memory_info(self):
        """Get memory usage information"""
        mem = psutil.virtual_memory()
        total_gb = round(mem.total / (1024**3), 1)  # Convert to GB
        return mem.percent, total_gb

    def get_temperature(self):
        """Get Jetson's temperature"""
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                temp = float(f.read().strip()) / 1000
            return temp
        except:
            return 0

    def get_nvme_size(self):
        """Get NVME physical size in GB"""
        try:
            if not os.path.exists("/dev/nvme0n1"):
                return None

            # Read disk size from /sys/block/nvme0n1/size
            # This gives number of sectors, each sector is 512 bytes
            with open("/sys/block/nvme0n1/size", "r") as f:
                sectors = int(f.read().strip())
                size_gb = round((sectors * 512) / (1024**3), 1)
            return size_gb
        except Exception as e:
            print(f"Get NVME size error: {e}")
            return None

    def get_root_device(self):
        """Get the device where root (/) is mounted"""
        try:
            with open('/proc/mounts', 'r') as f:
                for line in f:
                    device, mount_point, *_ = line.split()
                    if mount_point == '/':
                        return device
            return None
        except Exception as e:
            print(f"Get root device error: {e}")
            return None

    def get_storage_info(self):
        """Get storage information for root directory and NVME"""
        try:
            # Get root directory information
            root_disk = psutil.disk_usage('/')
            root_total_gb = round(root_disk.total / (1024**3), 1)
            root_used_percent = root_disk.percent

            # Check if root is on NVME
            root_device = self.get_root_device()

            is_root_nvme = root_device is not None and 'nvme' in root_device.lower()

            # Get NVME physical size if exists
            nvme_size = self.get_nvme_size()
            nvme_info = (True, nvme_size) if nvme_size is not None else None

            return {
                'is_root_nvme': is_root_nvme,
                'root_used_percent': root_used_percent,
                'root_total_gb': root_total_gb,
                'nvme_info': nvme_info
            }
        except Exception as e:
            print(f"Storage info error: {e}")
            return None

    def send_to_oled(self, x, y, message):
        """Send data to OLED display"""
        try:
            # Build data packet: x coordinate, y coordinate, string content
            data = [x, y] + list(message.encode('ascii'))
            self.bus.write_i2c_block_data(self.i2c_address, 0x00, data)
            time.sleep(0.01)  # Wait for data processing
        except Exception as e:
            print(f"Send data error: {e}")

    def send_big_to_oled(self, x, y, message):
        """Send data to OLED display"""
        try:
            # Build data packet: x coordinate, y coordinate, string content
            data = [x, y] + list(message.encode('ascii'))
            self.bus.write_i2c_block_data(self.i2c_address, 0x01, data)
            time.sleep(0.01)  # Wait for data processing
        except Exception as e:
            print(f"Send data error: {e}")

    def send_progress_to_oled(self, y, progress):
        """Send data to OLED display"""
        try:
            data = [0xFF, 0xF0, y, progress]
            self.bus.write_i2c_block_data(self.i2c_address, 0x00, data)
            time.sleep(0.01)  # Wait for data processing
        except Exception as e:
            print(f"Send progress error: {e}")

    def clear_screen(self):
        """Clear OLED screen"""
        try:
            self.bus.write_i2c_block_data(self.i2c_address, 0x00, [0xFF, 0xFF])
            time.sleep(0.01)
        except Exception as e:
            print(f"Clear screen error: {e}")

    def run(self):
        """Main running loop"""
        display_cycle = 0  # Track current display content
        cycle_start_time = time.time()  # Record current cycle start time
        empty_line = " ".ljust(19)  # Empty line for clearing unused lines

        while True:
            try:
                current_time = time.time()
                # Switch display content based on cycle
                if display_cycle < 2:
                    # First two cycles (CPU/Memory and Storage) show for 8 seconds
                    if current_time - cycle_start_time >= 8:
                        cycle_start_time = current_time
                        display_cycle = (display_cycle + 1) % 4
                        if display_cycle != 0:
                            self.clear_screen()
                else:
                    # Last two cycles (Temperature and Time) show for 5 seconds
                    if current_time - cycle_start_time >= 5:
                        cycle_start_time = current_time
                        display_cycle = (display_cycle + 1) % 4
                        if display_cycle != 0:
                            self.clear_screen()

                if display_cycle == 0:
                    # Display CPU and memory information
                    cpu_usage = self.get_cpu_usage()
                    mem_usage, mem_total = self.get_memory_info()

                    cpu_msg = f"CPU:{cpu_usage:.1f}%".ljust(19)
                    mem_msg = f"MEM:{mem_usage:.1f}% => {mem_total}G".ljust(19)

                    self.send_to_oled(0, 0, cpu_msg)
                    self.send_progress_to_oled(1, int(cpu_usage))
                    self.send_to_oled(0, 2, empty_line)
                    self.send_to_oled(0, 3, mem_msg)
                    self.send_progress_to_oled(4, int(mem_usage))

                elif display_cycle == 1:
                    # Display storage information
                    storage_info = self.get_storage_info()
                    if storage_info:
                        if storage_info['is_root_nvme']:
                            nvme_msg = f"NVME:{storage_info['root_used_percent']:.1f}% => {storage_info['root_total_gb']}G".ljust(19)
                            self.send_to_oled(0, 1, nvme_msg)
                            self.send_progress_to_oled(2, int(storage_info['root_used_percent']))
                        else:
                            sd_msg = f"SD:{storage_info['root_used_percent']:.1f}% => {storage_info['root_total_gb']}G".ljust(19)
                            self.send_to_oled(0, 1, sd_msg)
                            self.send_progress_to_oled(2, int(storage_info['root_used_percent']))

                            if storage_info['nvme_info']:
                                exists, total_gb = storage_info['nvme_info']
                                nvme_msg = f"NVME => {total_gb}G".ljust(19)
                            else:
                                nvme_msg = "NVME not installed".ljust(19)
                            self.send_to_oled(0, 4, nvme_msg)

                elif display_cycle == 2:
                    # Display temperature information
                    temperature = self.get_temperature()
                    temp_msg = f"TEMP: {temperature:.1f}C".center(14)  # Center align for big font display

                    self.send_big_to_oled(0, 2, temp_msg)  # Display in the middle of the screen
                    self.send_big_to_oled(0, 2, temp_msg)  # 调整到屏幕中间位置显示

                else:  # display_cycle == 3
                    # Display system time
                    current_datetime = time.localtime()
                    date_msg = time.strftime("  %Y/%m/%d", current_datetime).ljust(14)
                    time_msg = time.strftime("     %H:%M", current_datetime).ljust(14)

                    self.send_big_to_oled(0, 1, date_msg)
                    self.send_big_to_oled(0, 3, time_msg)

                # Update interval
                time.sleep(1)  # Reduce update interval for smoother display

            except Exception as e:
                print(f"Runtime error: {e}")
                time.sleep(5)

if __name__ == "__main__":
    monitor = JetsonSystemMonitor()
    monitor.run()
