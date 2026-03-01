#!/usr/bin/env python3
"""wall_power.py - Read power meter serial data from Raspberry Pi
Run on: raspi (10.10.1.203)
Usage: python3 wall_power.py -i 1 -t 120 -o wall_power.csv
"""

import serial
import re
import time
import csv
import argparse


class AdPower:
    def __init__(self, port='/dev/ttyUSB0', baudrate=9600):
        self.ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1
        )

    def start(self):
        self.ser.write(b'S#')

    def stop(self):
        self.ser.write(b'E#')

    @staticmethod
    def _get_watt(watt, unit):
        divisors = {
            0: 0, 1: 10000000, 2: 1000000, 3: 100000,
            4: 10000, 5: 1000, 6: 100, 7: 10, 8: 1, 9: 0.1
        }
        if unit == 0:
            return 0.0
        d = divisors.get(unit, 1)
        return watt / d if d >= 1 else watt * 10.0

    def probe(self):
        while True:
            data = self.ser.readline()
            if not data:
                return None
            pattern = (
                r'^S:\s*(\d{1,5})(\d)Vo:\s*(\d{1,5})(\d)Am:\s*(\d{1,5})(\d)Wa:'
                r'\s*(\d{1,5})(\d)Wh:\s*(\d{1,5})(\d)Pf:\s*(\d{1,5})(\d)Hz\s*E'
            )
            matches = re.match(pattern, data.decode(errors='ignore').strip())
            if not matches:
                continue
            watt = self._get_watt(int(matches.group(5)), int(matches.group(6)))
            volt = self._get_watt(int(matches.group(1)), int(matches.group(2)))
            amp = self._get_watt(int(matches.group(3)), int(matches.group(4)))
            return watt, volt, amp


def main():
    parser = argparse.ArgumentParser(description='Wall power measurement (raspi)')
    parser.add_argument('-i', '--interval', type=float, default=1.0,
                        help='Sampling interval in seconds (default: 1)')
    parser.add_argument('-t', '--timeout', type=int, default=120,
                        help='Total measurement duration in seconds (default: 120)')
    parser.add_argument('-o', '--output', type=str, default='wall_power.csv',
                        help='Output CSV file name')
    args = parser.parse_args()

    adp = AdPower()

    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'elapsed_s', 'wall_W', 'volt_V', 'amp_A'])

        print(f"=== Wall Power Measurement ===")
        print(f"Interval: {args.interval}s, Duration: {args.timeout}s")
        print(f"Output: {args.output}")
        print(f"Starting... (Ctrl+C to stop)")

        adp.start()
        start_time = time.time()

        try:
            next_sample = start_time + args.interval
            while True:
                elapsed = time.time() - start_time
                if elapsed > args.timeout:
                    break

                res = adp.probe()
                if res is None:
                    print("WARNING: No data from power meter")
                    time.sleep(0.1)
                    continue

                now = time.time()
                if now >= next_sample:
                    watt, volt, amp = res
                    row = [
                        f"{now:.3f}",
                        f"{now - start_time:.2f}",
                        f"{watt:.2f}",
                        f"{volt:.1f}",
                        f"{amp:.3f}"
                    ]
                    writer.writerow(row)
                    f.flush()
                    ts = time.strftime('%H:%M:%S')
                    print(f"[{ts}] {elapsed:.1f}s | Wall: {watt:.2f}W | {volt:.1f}V {amp:.3f}A")
                    next_sample += args.interval

                time.sleep(0.05)

        except KeyboardInterrupt:
            print("\nStopped by user")
        finally:
            adp.stop()
            print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
