#!/usr/bin/env python3
"""component_power.py - CPU/DRAM/GPU power measurement on the gpu server
Run on: gpu server (203.255.176.80)
Usage: sudo python3 component_power.py -i 1 -t 120 -o component_power.csv
"""

import time
import subprocess
import csv
import os
import glob
import argparse


# ── CPU utilization ──
_prev_cpu_times = None

def read_cpu_utilization():
    """Return overall CPU utilization (%) based on /proc/stat."""
    global _prev_cpu_times
    try:
        with open('/proc/stat') as f:
            line = f.readline()  # first line: cpu  user nice system idle ...
        parts = line.split()
        times = [int(x) for x in parts[1:]]  # user, nice, system, idle, iowait, irq, softirq, steal
        idle = times[3] + times[4]  # idle + iowait
        total = sum(times)

        if _prev_cpu_times is None:
            _prev_cpu_times = (idle, total)
            return 0.0

        prev_idle, prev_total = _prev_cpu_times
        _prev_cpu_times = (idle, total)

        d_idle = idle - prev_idle
        d_total = total - prev_total
        if d_total == 0:
            return 0.0
        return (1.0 - d_idle / d_total) * 100.0
    except Exception:
        return 0.0


# ── GPU utilization ──
def read_gpu_utilization():
    """Return a list of GPU utilization (%) for each GPU."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            text=True, timeout=2
        )
        utils = []
        for line in out.strip().split("\n"):
            line = line.strip()
            if line:
                utils.append(float(line))
        return utils
    except Exception:
        return []


# ── RAPL ──
def find_rapl_domains():
    """Auto-detect RAPL domains (package, core, dram, etc.)."""
    domains = {}
    base = "/sys/class/powercap"

    for pkg_path in sorted(glob.glob(f"{base}/intel-rapl:*")):
        # Package level
        name_file = os.path.join(pkg_path, "name")
        if os.path.exists(name_file):
            with open(name_file) as f:
                name = f.read().strip()
            energy_file = os.path.join(pkg_path, "energy_uj")
            if os.path.exists(energy_file):
                domains[name] = energy_file

        # Sub-domains (core, dram, uncore, etc.)
        for sub_path in sorted(glob.glob(f"{pkg_path}/intel-rapl:*")):
            sub_name_file = os.path.join(sub_path, "name")
            if os.path.exists(sub_name_file):
                with open(sub_name_file) as f:
                    sub_name = f.read().strip()
                sub_energy_file = os.path.join(sub_path, "energy_uj")
                if os.path.exists(sub_energy_file):
                    domains[sub_name] = sub_energy_file

    return domains


def read_energy_uj(path):
    with open(path) as f:
        return int(f.read().strip())


# ── GPU (nvidia-smi) ──
def read_gpu_power():
    """Return a list of GPU power draw (W) for each GPU."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.draw",
             "--format=csv,noheader,nounits"],
            text=True, timeout=2
        )
        powers = []
        for line in out.strip().split("\n"):
            line = line.strip()
            if line:
                powers.append(float(line))
        return powers
    except Exception as e:
        print(f"GPU read error: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description='Component power measurement (gpu server)')
    parser.add_argument('-i', '--interval', type=float, default=1.0,
                        help='Sampling interval in seconds (default: 1)')
    parser.add_argument('-t', '--timeout', type=int, default=120,
                        help='Total measurement duration in seconds (default: 120)')
    parser.add_argument('-o', '--output', type=str, default='component_power.csv',
                        help='Output CSV file name')
    args = parser.parse_args()

    # Discover RAPL domains
    domains = find_rapl_domains()
    print(f"RAPL domains found: {list(domains.keys())}")

    if not domains:
        print("ERROR: No RAPL domains found. Run with sudo?")
        return

    # Detect GPUs
    gpu_powers = read_gpu_power()
    gpu_count = len(gpu_powers)
    print(f"GPUs found: {gpu_count}")

    # Build CSV header
    rapl_names = sorted(domains.keys())  # e.g., ['core', 'dram', 'package-0']
    gpu_names = [f"gpu{i}_W" for i in range(gpu_count)]

    fieldnames = ['timestamp', 'elapsed_s', 'cpu_util_pct']
    for name in rapl_names:
        fieldnames.append(f"rapl_{name}_W")
    fieldnames.extend(gpu_names)
    fieldnames.append('gpu_total_W')
    gpu_util_names = [f"gpu{i}_util_pct" for i in range(gpu_count)]
    fieldnames.extend(gpu_util_names)

    # Initial values
    prev_energy = {name: read_energy_uj(path) for name, path in domains.items()}
    prev_time = time.time()

    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        print(f"\n=== Component Power Measurement ===")
        print(f"Interval: {args.interval}s, Duration: {args.timeout}s")
        print(f"Output: {args.output}")
        print(f"Columns: {fieldnames}")
        print(f"Starting... (Ctrl+C to stop)\n")

        start_time = time.time()

        try:
            while True:
                time.sleep(args.interval)
                now = time.time()
                dt = now - prev_time
                elapsed = now - start_time

                if elapsed > args.timeout:
                    break

                cpu_util = read_cpu_utilization()
                row = {
                    'timestamp': f"{now:.3f}",
                    'elapsed_s': f"{elapsed:.2f}",
                    'cpu_util_pct': f"{cpu_util:.1f}",
                }

                # Read RAPL
                for name in rapl_names:
                    path = domains[name]
                    curr = read_energy_uj(path)
                    power_w = (curr - prev_energy[name]) / (dt * 1e6)
                    row[f"rapl_{name}_W"] = f"{power_w:.2f}"
                    prev_energy[name] = curr

                # Read GPU (power + utilization)
                gpu_powers = read_gpu_power()
                gpu_utils = read_gpu_utilization()
                gpu_total = 0.0
                for i, pw in enumerate(gpu_powers):
                    row[f"gpu{i}_W"] = f"{pw:.2f}"
                    gpu_total += pw
                for i, ut in enumerate(gpu_utils):
                    row[f"gpu{i}_util_pct"] = f"{ut:.0f}"
                row['gpu_total_W'] = f"{gpu_total:.2f}"

                prev_time = now
                writer.writerow(row)
                f.flush()

                # Console output
                ts = time.strftime('%H:%M:%S')
                pkg = row.get('rapl_package-0_W', '?')
                dram = row.get('rapl_dram_W', '?')
                gpu_ut = row.get('gpu0_util_pct', '?')
                print(f"[{ts}] {elapsed:.1f}s | CPU_pkg: {pkg}W ({cpu_util:.0f}%) | DRAM: {dram}W | GPU: {gpu_total:.1f}W ({gpu_ut}%)")

        except KeyboardInterrupt:
            print("\nStopped by user")
        finally:
            print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
