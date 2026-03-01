#!/usr/bin/env python3
"""concurrent_power.py - System + per-PID power/utilization measurement for concurrent workloads

Includes all system metrics from component_power.py, plus
per-workload (PID) CPU/GPU utilization and GPU memory collection.
Use instead of component_power.py for concurrent execution experiments.

Run on: gpu server (203.255.176.80)
Usage:
  sudo python3 concurrent_power.py \
    --workloads vm_a:workload_resnet vm_b:workload_gemm \
    -i 1 -t 180 -o logs/conc_resnet_gemm_0211.csv
"""

import time
import subprocess
import csv
import os
import glob
import argparse
import threading
import re


# ── CPU utilization (system-wide) ──
_prev_cpu_times = None


def read_cpu_utilization():
    """Return overall CPU utilization (%) based on /proc/stat."""
    global _prev_cpu_times
    try:
        with open('/proc/stat') as f:
            line = f.readline()
        parts = line.split()
        times = [int(x) for x in parts[1:]]
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


# ── System-wide CPU ticks (also used for per-PID CPU% calculation) ──
_prev_total_ticks = None


def read_total_cpu_ticks():
    """Return total system CPU ticks (for per-PID CPU% calculation)."""
    global _prev_total_ticks
    try:
        with open('/proc/stat') as f:
            line = f.readline()
        parts = line.split()
        total = sum(int(x) for x in parts[1:])

        prev = _prev_total_ticks
        _prev_total_ticks = total

        if prev is None:
            return None, total
        return prev, total
    except Exception:
        return None, 0


# ── GPU utilization (system-wide) ──
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
        name_file = os.path.join(pkg_path, "name")
        if os.path.exists(name_file):
            with open(name_file) as f:
                name = f.read().strip()
            energy_file = os.path.join(pkg_path, "energy_uj")
            if os.path.exists(energy_file):
                domains[name] = energy_file

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


# ── GPU power (nvidia-smi) ──
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


# ── Per-PID CPU utilization ──
_prev_pid_ticks = {}  # {pid: utime+stime}
_cpu_debug_count = {}  # {pid: count} — limit debug output count


def read_pid_cpu_pct(pid, prev_total, curr_total):
    """Return CPU utilization (%) for a specific PID, summing all threads."""
    global _prev_pid_ticks, _cpu_debug_count
    try:
        # Sum utime+stime from all threads in /proc/[pid]/task/*/stat
        ticks = 0
        task_dir = f'/proc/{pid}/task'
        thread_count = 0
        if os.path.isdir(task_dir):
            for tid in os.listdir(task_dir):
                try:
                    with open(f'{task_dir}/{tid}/stat') as f:
                        stat = f.read()
                    idx = stat.rfind(')')
                    fields = stat[idx + 2:].split()
                    ticks += int(fields[11]) + int(fields[12])
                    thread_count += 1
                except:
                    continue
        else:
            # fallback: /proc/[pid]/stat
            with open(f'/proc/{pid}/stat') as f:
                stat = f.read()
            idx = stat.rfind(')')
            fields = stat[idx + 2:].split()
            ticks = int(fields[11]) + int(fields[12])
            thread_count = 1

        prev = _prev_pid_ticks.get(pid)
        _prev_pid_ticks[pid] = ticks

        # Debug: print only the first 5 samples
        dbg_n = _cpu_debug_count.get(pid, 0)
        if dbg_n < 5:
            _cpu_debug_count[pid] = dbg_n + 1
            d_p = (ticks - prev) if prev is not None else -1
            d_t = (curr_total - prev_total) if prev_total is not None else -1
            print(f"  [CPU-DBG] PID {pid}: threads={thread_count} ticks={ticks} "
                  f"prev={prev} d_pid={d_p} d_total={d_t}")

        if prev is None or prev_total is None:
            return 0.0

        d_pid = ticks - prev
        d_total = curr_total - prev_total
        if d_total == 0:
            return 0.0
        return (d_pid / d_total) * 100.0
    except Exception as e:
        print(f"  [CPU-DBG] PID {pid} error: {e}")
        return 0.0


def is_pid_alive(pid):
    """Check if a process is alive."""
    try:
        return os.path.exists(f'/proc/{pid}')
    except Exception:
        return False


# ── Per-PID memory usage (RSS) ──
def read_pid_rss_mib(pid):
    """Return RSS (Resident Set Size) in MiB for a PID. Based on /proc/[pid]/status."""
    try:
        with open(f'/proc/{pid}/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    # VmRSS:    123456 kB
                    kb = int(line.split()[1])
                    return kb / 1024.0
        return 0.0
    except Exception:
        return 0.0


# ── Per-PID disk I/O ──
_prev_pid_io = {}  # {pid: (read_bytes, write_bytes)}


def read_pid_io(pid, dt):
    """Return disk I/O rate (MB/s) for a PID. Based on /proc/[pid]/io."""
    global _prev_pid_io
    try:
        read_bytes = 0
        write_bytes = 0
        with open(f'/proc/{pid}/io') as f:
            for line in f:
                if line.startswith('read_bytes:'):
                    read_bytes = int(line.split()[1])
                elif line.startswith('write_bytes:'):
                    write_bytes = int(line.split()[1])

        prev = _prev_pid_io.get(pid)
        _prev_pid_io[pid] = (read_bytes, write_bytes)

        if prev is None or dt <= 0:
            return 0.0, 0.0

        r_mbs = (read_bytes - prev[0]) / (dt * 1024 * 1024)
        w_mbs = (write_bytes - prev[1]) / (dt * 1024 * 1024)
        return max(r_mbs, 0.0), max(w_mbs, 0.0)
    except Exception:
        return 0.0, 0.0


# ── Per-PID GPU metrics (nvidia-smi pmon) ──
class GpuPmon:
    """Run nvidia-smi pmon in background to collect per-PID GPU metrics."""

    def __init__(self):
        self._data = {}  # {pid: {'sm': float, 'mem': float, 'fb_mib': float}}
        self._lock = threading.Lock()
        self._proc = None
        self._thread = None
        self._stop = False

    def start(self):
        try:
            # stdbuf -oL: force line-buffered output for pipe
            self._proc = subprocess.Popen(
                ["stdbuf", "-oL", "nvidia-smi", "pmon", "-d", "1", "-s", "um"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1  # line-buffered
            )
            self._thread = threading.Thread(target=self._reader, daemon=True)
            self._thread.start()
        except Exception as e:
            print(f"pmon start error: {e}")

    def _reader(self):
        """Continuously read stdout and update _data."""
        # Dynamically parse column indices from header
        sm_idx, mem_idx, fb_idx = 3, 4, 7  # defaults
        line_count = 0
        while not self._stop and self._proc and self._proc.poll() is None:
            line = self._proc.stdout.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            line_count += 1
            # Debug: print the first 5 lines
            if line_count <= 5:
                print(f"  [PMON-DBG] line {line_count}: {line}")
            # Determine column positions from header line
            if line.startswith('#'):
                cols = line.replace('#', '').split()
                for i, col in enumerate(cols):
                    if col == 'sm':
                        sm_idx = i
                    elif col == 'fb':
                        fb_idx = i
                if line_count <= 5:
                    print(f"  [PMON-DBG] detected: sm_idx={sm_idx}, fb_idx={fb_idx}")
                continue
            parts = line.split()
            if len(parts) < fb_idx + 1:
                continue
            try:
                pid = int(parts[1])
                if pid <= 0:
                    continue
                sm = self._parse_val(parts[sm_idx])
                fb_mib = self._parse_val(parts[fb_idx])
                with self._lock:
                    self._data[pid] = {
                        'sm': sm,
                        'mem': 0.0,
                        'fb_mib': fb_mib
                    }
            except (ValueError, IndexError):
                continue

    @staticmethod
    def _parse_val(s):
        """Parse numeric value; treat '-' as 0."""
        try:
            return float(s)
        except (ValueError, TypeError):
            return 0.0

    def get(self, pid):
        """Return the latest GPU metrics for a PID."""
        with self._lock:
            return self._data.get(pid, {'sm': 0.0, 'mem': 0.0, 'fb_mib': 0.0})

    def stop(self):
        self._stop = True
        if self._proc:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=3)
            except Exception:
                pass


# ── PID auto-detection ──
def find_pid_by_pattern(pattern, exclude_pid=None):
    """Find PID whose cmdline contains pattern (excluding self and sudo parent)."""
    my_pid = os.getpid()
    my_ppid = os.getppid()  # exclude sudo parent process
    try:
        for entry in os.listdir('/proc'):
            if not entry.isdigit():
                continue
            pid = int(entry)
            if pid == my_pid or pid == my_ppid or pid == exclude_pid:
                continue
            try:
                with open(f'/proc/{pid}/cmdline', 'rb') as f:
                    cmdline = f.read().replace(b'\x00', b' ').decode('utf-8', errors='replace')
                # exclude concurrent_power's own process
                if 'concurrent_power' in cmdline:
                    continue
                if pattern in cmdline:
                    return pid
            except (PermissionError, FileNotFoundError, ProcessLookupError):
                continue
    except Exception:
        pass
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Concurrent workload power measurement (system + per-PID)')
    parser.add_argument('--workloads', nargs='+', metavar='NAME:PATTERN',
                        default=[],
                        help='Workload specs as name:pattern pairs '
                             '(e.g. vm_a:workload_resnet vm_b:workload_gemm)')
    parser.add_argument('-i', '--interval', type=float, default=1.0,
                        help='Sampling interval in seconds (default: 1)')
    parser.add_argument('-t', '--timeout', type=int, default=120,
                        help='Total measurement duration in seconds (default: 120)')
    parser.add_argument('-o', '--output', type=str, default='concurrent_power.csv',
                        help='Output CSV file name')
    args = parser.parse_args()

    # Parse workload specs
    workloads = []  # [(name, pattern)]
    for spec in args.workloads:
        if ':' not in spec:
            print(f"ERROR: Invalid workload spec '{spec}'. Use name:pattern format.")
            return
        name, pattern = spec.split(':', 1)
        workloads.append((name, pattern))

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

    # Build CSV header -- system metrics
    rapl_names = sorted(domains.keys())
    gpu_names = [f"gpu{i}_W" for i in range(gpu_count)]

    fieldnames = ['timestamp', 'elapsed_s', 'cpu_util_pct']
    for name in rapl_names:
        fieldnames.append(f"rapl_{name}_W")
    fieldnames.extend(gpu_names)
    fieldnames.append('gpu_total_W')
    gpu_util_names = [f"gpu{i}_util_pct" for i in range(gpu_count)]
    fieldnames.extend(gpu_util_names)

    # Build CSV header -- per-PID metrics
    for wname, _ in workloads:
        fieldnames.extend([
            f"{wname}_pid",
            f"{wname}_cpu_pct",
            f"{wname}_gpu_sm_pct",
            f"{wname}_gpu_mem_mib",
            f"{wname}_rss_mib",
            f"{wname}_io_read_mbs",
            f"{wname}_io_write_mbs",
        ])

    # Initial values
    prev_energy = {name: read_energy_uj(path) for name, path in domains.items()}
    prev_time = time.time()

    # Initial CPU tick read (system-wide + shared for per-PID)
    read_cpu_utilization()     # initialize _prev_cpu_times
    read_total_cpu_ticks()     # initialize _prev_total_ticks

    # Start GPU pmon
    pmon = GpuPmon()
    if workloads:
        pmon.start()

    # Workload PID tracking state
    wl_pids = {}       # {name: pid or None}
    wl_found = {}      # {name: bool} -- locked once found
    for wname, _ in workloads:
        wl_pids[wname] = None
        wl_found[wname] = False

    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        print(f"\n=== Concurrent Power Measurement ===")
        print(f"Interval: {args.interval}s, Duration: {args.timeout}s")
        print(f"Output: {args.output}")
        print(f"Workloads: {workloads if workloads else '(none)'}")
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

                # ── System metrics ──
                cpu_util = read_cpu_utilization()
                prev_total, curr_total = read_total_cpu_ticks()

                row = {
                    'timestamp': f"{now:.3f}",
                    'elapsed_s': f"{elapsed:.2f}",
                    'cpu_util_pct': f"{cpu_util:.1f}",
                }

                # RAPL
                for name in rapl_names:
                    path = domains[name]
                    curr = read_energy_uj(path)
                    power_w = (curr - prev_energy[name]) / (dt * 1e6)
                    row[f"rapl_{name}_W"] = f"{power_w:.2f}"
                    prev_energy[name] = curr

                # GPU power + utilization
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

                # ── Per-PID metrics ──
                for wname, pattern in workloads:
                    pid = wl_pids[wname]

                    # Detect PID (only if not yet found)
                    if not wl_found[wname]:
                        found = find_pid_by_pattern(pattern)
                        if found is not None:
                            wl_pids[wname] = found
                            wl_found[wname] = True
                            pid = found
                            print(f"  >> Detected {wname} (pattern='{pattern}') → PID {pid}")

                    if pid is not None and is_pid_alive(pid):
                        cpu_pct = read_pid_cpu_pct(pid, prev_total, curr_total)
                        gpu_info = pmon.get(pid)
                        rss = read_pid_rss_mib(pid)
                        io_r, io_w = read_pid_io(pid, dt)
                        row[f"{wname}_pid"] = pid
                        row[f"{wname}_cpu_pct"] = f"{cpu_pct:.1f}"
                        row[f"{wname}_gpu_sm_pct"] = f"{gpu_info['sm']:.0f}"
                        row[f"{wname}_gpu_mem_mib"] = f"{gpu_info['fb_mib']:.0f}"
                        row[f"{wname}_rss_mib"] = f"{rss:.1f}"
                        row[f"{wname}_io_read_mbs"] = f"{io_r:.2f}"
                        row[f"{wname}_io_write_mbs"] = f"{io_w:.2f}"
                    else:
                        # Not found or terminated
                        row[f"{wname}_pid"] = pid if pid is not None else 0
                        row[f"{wname}_cpu_pct"] = "0.0"
                        row[f"{wname}_gpu_sm_pct"] = "0"
                        row[f"{wname}_gpu_mem_mib"] = "0"
                        row[f"{wname}_rss_mib"] = "0.0"
                        row[f"{wname}_io_read_mbs"] = "0.00"
                        row[f"{wname}_io_write_mbs"] = "0.00"

                writer.writerow(row)
                f.flush()

                # Console output
                ts = time.strftime('%H:%M:%S')
                pkg = row.get('rapl_package-0_W', '?')
                dram = row.get('rapl_dram_W', '?')
                gpu_ut = row.get('gpu0_util_pct', '?')
                wl_info = ""
                for wname, _ in workloads:
                    pid = wl_pids[wname]
                    cpct = row.get(f"{wname}_cpu_pct", "?")
                    gsm = row.get(f"{wname}_gpu_sm_pct", "?")
                    pid_str = str(pid) if pid else "-"
                    wl_info += f" | {wname}[{pid_str}]: CPU {cpct}% GPU {gsm}%"
                print(f"[{ts}] {elapsed:.1f}s | CPU_pkg: {pkg}W ({cpu_util:.0f}%) "
                      f"| DRAM: {dram}W | GPU: {gpu_total:.1f}W ({gpu_ut}%){wl_info}")

        except KeyboardInterrupt:
            print("\nStopped by user")
        finally:
            pmon.stop()
            print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
