#!/usr/bin/env python3
"""run_perf_stat.py - Per-workload perf stat measurement (IPC, LLC cache misses)

Measures computational characteristics of 6 workloads (cgroup 10 cores / 14G, same as Solo baseline)

Usage:
  sudo python3 -u run_perf_stat.py 2>&1 | tee logs/perf_stat_0225.log
  sudo python3 -u run_perf_stat.py --workloads resnet llm nodejs  # select workloads
"""

import subprocess
import time
import os
import sys
import signal
import argparse
import re
import csv
from datetime import datetime

# ==========================================
# Configuration
# ==========================================
HOME = "/home/optimus"
SCRIPTS_DIR = f"{HOME}/jiyoon_energy/scripts"
LOGS_DIR = os.path.join(SCRIPTS_DIR, "logs")
VENV_PYTHON = f"{HOME}/yolo/venv/bin/python3"

# cgroup
CGROUP_BASE = "/sys/fs/cgroup/optimus"
CGROUP_VM = os.path.join(CGROUP_BASE, "vm_a")
SOLO_CPUS = "0-9"
SOLO_MEM = "14G"

# Measurement protocol
WARMUP = 15          # Workload stabilization wait (sec)
PERF_DURATION = 60   # perf stat measurement time (sec)
COOLDOWN = 5         # Cooldown between workloads (sec)

WORKLOAD_DUR = WARMUP + PERF_DURATION + 15  # Workload execution time (with margin)

# perf events
PERF_EVENTS = "instructions,cycles,LLC-load-misses,LLC-loads,cache-references,cache-misses"

# ==========================================
# Workload definitions (same as run_solo_experiments.py)
# ==========================================
WORKLOADS = {
    "resnet": {
        "cmd": f"{VENV_PYTHON} {SCRIPTS_DIR}/workload_resnet.py -t {WORKLOAD_DUR}",
        "pattern": "workload_resnet.py",
        "gpu": False,
    },
    "resnet_gpu": {
        "cmd": f"{VENV_PYTHON} {SCRIPTS_DIR}/workload_resnet_gpu.py -t {WORKLOAD_DUR}",
        "pattern": "workload_resnet_gpu",
        "gpu": True,
    },
    "gemm": {
        "cmd": f"{VENV_PYTHON} {SCRIPTS_DIR}/workload_gemm.py -t {WORKLOAD_DUR}",
        "pattern": "workload_gemm",
        "gpu": True,
    },
    "yolo": {
        "cmd": f"{VENV_PYTHON} {SCRIPTS_DIR}/workload_yolo.py -t {WORKLOAD_DUR} --source {HOME}/yolo/test_video.mp4",
        "pattern": "workload_yolo",
        "gpu": True,
    },
    "llm": {
        "cmd": f"{VENV_PYTHON} {SCRIPTS_DIR}/workload_llm.py -t {WORKLOAD_DUR}",
        "pattern": "workload_llm",
        "gpu": False,
    },
    "nodejs": {
        "cmd_server": f"node {HOME}/node/server.js",
        "cmd_load": f"autocannon -c 100 -d {WORKLOAD_DUR} http://localhost:3000",
        "pattern": "server.js",
        "two_phase": True,
        "gpu": False,
    },
}

# CPU-dominant first (primary), GPU-dominant for reference
DEFAULT_ORDER = ["resnet", "llm", "nodejs", "resnet_gpu", "gemm", "yolo"]

DATE_STR = datetime.now().strftime("%m%d")


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def find_pid(pattern):
    """Find workload PID by pattern matching"""
    my_pid = os.getpid()
    for entry in os.listdir('/proc'):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid == my_pid:
            continue
        try:
            with open(f'/proc/{pid}/cmdline', 'rb') as f:
                cmdline = f.read().replace(b'\x00', b' ').decode('utf-8', errors='replace')
            if pattern in cmdline and 'perf' not in cmdline and 'run_perf_stat' not in cmdline:
                return pid
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue
    return None


def setup_cgroup():
    try:
        with open(os.path.join(CGROUP_VM, "cpuset.cpus"), "w") as f:
            f.write(SOLO_CPUS)
        with open(os.path.join(CGROUP_VM, "memory.max"), "w") as f:
            f.write(SOLO_MEM)
        return True
    except Exception as e:
        log(f"ERROR: cgroup setup failed: {e}")
        return False


def reset_cgroup():
    try:
        with open(os.path.join(CGROUP_VM, "cpuset.cpus"), "w") as f:
            f.write("0-19")
        with open(os.path.join(CGROUP_VM, "memory.max"), "w") as f:
            f.write("max")
    except Exception:
        pass


def kill_procs(procs):
    for proc in procs:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass
    time.sleep(2)
    for proc in procs:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass


def cleanup_residual():
    """Kill residual processes from previous experiments"""
    patterns = ["server.js", "workload_yolo", "workload_gemm",
                "workload_resnet", "workload_llm", "autocannon"]
    my_pid = os.getpid()
    killed = []
    for entry in os.listdir('/proc'):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid == my_pid:
            continue
        try:
            with open(f'/proc/{pid}/cmdline', 'rb') as f:
                cmdline = f.read().replace(b'\x00', b' ').decode('utf-8', errors='replace')
            for pat in patterns:
                if pat in cmdline:
                    os.kill(pid, signal.SIGKILL)
                    killed.append((pid, pat))
                    break
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue
    if killed:
        log(f"  Residual processes killed: {killed}")


def run_perf_stat(workload_name):
    """Run workload and measure with perf stat"""
    wl = WORKLOADS[workload_name]
    log(f"{'='*55}")
    log(f"  Workload: {workload_name}")
    log(f"{'='*55}")

    # Cleanup residuals
    cleanup_residual()

    # cgroup setup
    setup_cgroup()
    log(f"  cgroup: cpus={SOLO_CPUS}, mem={SOLO_MEM}")

    # Start workload
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)

    procs = []
    if wl.get("two_phase"):
        # NodeJS: server + load generator
        server_proc = subprocess.Popen(
            ["bash", "-c", wl["cmd_server"]],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid, env=env,
        )
        with open(os.path.join(CGROUP_VM, "cgroup.procs"), "w") as f:
            f.write(str(server_proc.pid))
        log(f"  Server started (PID={server_proc.pid})")
        time.sleep(2)

        load_proc = subprocess.Popen(
            ["bash", "-c", wl["cmd_load"]],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid, env=env,
        )
        with open(os.path.join(CGROUP_VM, "cgroup.procs"), "w") as f:
            f.write(str(load_proc.pid))
        log(f"  autocannon started (PID={load_proc.pid})")
        procs = [server_proc, load_proc]
        target_pid = server_proc.pid
    else:
        cmd = wl["cmd"]
        if wl.get("gpu"):
            cmd += " --device cuda:0"

        proc = subprocess.Popen(
            ["bash", "-c", cmd],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid, env=env,
        )
        with open(os.path.join(CGROUP_VM, "cgroup.procs"), "w") as f:
            f.write(str(proc.pid))
        log(f"  Workload started (PID={proc.pid})")
        procs = [proc]
        target_pid = proc.pid

    # Warmup wait
    log(f"  Stabilization wait ({WARMUP}s)...")
    time.sleep(WARMUP)

    # Verify actual PID (may be a child process)
    actual_pid = find_pid(wl["pattern"])
    if actual_pid and actual_pid != target_pid:
        log(f"  Actual workload PID: {actual_pid} (child process)")
        target_pid = actual_pid
    elif actual_pid is None:
        log(f"  WARNING: Pattern '{wl['pattern']}' PID not found, using original PID: {target_pid}")

    # Run perf stat
    perf_output = os.path.join(LOGS_DIR, f"perf_stat_{workload_name}_{DATE_STR}.txt")
    perf_cmd = [
        "perf", "stat",
        "-p", str(target_pid),
        "-e", PERF_EVENTS,
        "--", "sleep", str(PERF_DURATION),
    ]

    log(f"  perf stat started (PID={target_pid}, {PERF_DURATION}s)...")
    try:
        with open(perf_output, "w") as f:
            subprocess.run(
                perf_cmd,
                stdout=f, stderr=f,
                timeout=PERF_DURATION + 15,
            )
        log(f"  perf stat done -> {perf_output}")
    except subprocess.TimeoutExpired:
        log(f"  WARNING: perf stat timeout")
    except Exception as e:
        log(f"  ERROR: perf stat failed: {e}")

    # Print results
    try:
        with open(perf_output, "r") as f:
            content = f.read()
        log(f"  --- perf stat results ---")
        for line in content.strip().split("\n"):
            log(f"    {line}")
    except Exception:
        pass

    # Stop workload
    kill_procs(procs)
    reset_cgroup()
    log(f"  Cooldown ({COOLDOWN}s)...")
    time.sleep(COOLDOWN)

    return perf_output


def parse_perf_output(filepath):
    """Parse perf stat output"""
    results = {}
    try:
        with open(filepath, "r") as f:
            content = f.read()
    except Exception:
        return results

    # instructions
    m = re.search(r'([\d,]+)\s+instructions', content)
    if m:
        results['instructions'] = int(m.group(1).replace(',', ''))

    # cycles
    m = re.search(r'([\d,]+)\s+cycles', content)
    if m:
        results['cycles'] = int(m.group(1).replace(',', ''))

    # IPC (computed directly by perf)
    m = re.search(r'([\d.]+)\s+insn per cycle', content)
    if m:
        results['ipc'] = float(m.group(1))
    elif 'instructions' in results and 'cycles' in results and results['cycles'] > 0:
        results['ipc'] = results['instructions'] / results['cycles']

    # LLC-load-misses
    m = re.search(r'([\d,]+)\s+LLC-load-misses', content)
    if m:
        results['llc_load_misses'] = int(m.group(1).replace(',', ''))

    # LLC-loads
    m = re.search(r'([\d,]+)\s+LLC-loads', content)
    if m:
        results['llc_loads'] = int(m.group(1).replace(',', ''))

    # cache-references
    m = re.search(r'([\d,]+)\s+cache-references', content)
    if m:
        results['cache_references'] = int(m.group(1).replace(',', ''))

    # cache-misses
    m = re.search(r'([\d,]+)\s+cache-misses', content)
    if m:
        results['cache_misses'] = int(m.group(1).replace(',', ''))

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Solo workload perf stat measurement (IPC, LLC cache misses)")
    parser.add_argument("--workloads", nargs="+", default=None,
                        help="Select workloads (e.g., resnet llm nodejs)")
    args = parser.parse_args()

    if os.geteuid() != 0:
        print("ERROR: sudo is required.")
        print(f"  sudo python3 -u run_perf_stat.py 2>&1 | tee logs/perf_stat_{DATE_STR}.log")
        sys.exit(1)

    os.makedirs(LOGS_DIR, exist_ok=True)

    if not os.path.isdir(CGROUP_VM):
        print(f"ERROR: cgroup directory not found: {CGROUP_VM}")
        print(f"  sudo mkdir -p {CGROUP_VM}")
        sys.exit(1)

    order = args.workloads if args.workloads else DEFAULT_ORDER[:]
    for wl in order:
        if wl not in WORKLOADS:
            print(f"ERROR: Unknown workload: {wl}")
            print(f"  Available: {', '.join(WORKLOADS.keys())}")
            sys.exit(1)

    est_min = len(order) * (WARMUP + PERF_DURATION + COOLDOWN + 5) / 60
    log(f"perf stat measurement — {len(order)} workloads")
    log(f"Environment: cpus={SOLO_CPUS}, mem={SOLO_MEM} (10 cores)")
    log(f"Protocol: {WARMUP}s warmup -> {PERF_DURATION}s perf stat -> {COOLDOWN}s cooldown")
    log(f"Estimated time: ~{est_min:.0f} min")
    log(f"Order: {', '.join(order)}")
    log("")

    # Run measurements
    all_results = {}
    for i, wl_name in enumerate(order):
        log(f"\n[{i+1}/{len(order)}] {wl_name}")
        output_file = run_perf_stat(wl_name)
        results = parse_perf_output(output_file)
        all_results[wl_name] = results

        if 'ipc' in results:
            log(f"  ★ IPC: {results['ipc']:.2f}")
        if 'llc_load_misses' in results:
            log(f"  ★ LLC misses: {results['llc_load_misses']:,}")

    # Print summary
    log(f"\n{'='*70}")
    log("Summary")
    log(f"{'='*70}")
    header = f"{'Workload':<15} {'IPC':>6} {'Instructions':>18} {'Cycles':>18} {'LLC Misses':>15}"
    log(header)
    log("-" * 70)
    for wl_name in order:
        r = all_results.get(wl_name, {})
        ipc = f"{r['ipc']:.2f}" if 'ipc' in r else "-"
        inst = f"{r['instructions']:,}" if 'instructions' in r else "-"
        cyc = f"{r['cycles']:,}" if 'cycles' in r else "-"
        llc = f"{r['llc_load_misses']:,}" if 'llc_load_misses' in r else "-"
        log(f"{wl_name:<15} {ipc:>6} {inst:>18} {cyc:>18} {llc:>15}")

    # Save CSV
    csv_path = os.path.join(LOGS_DIR, f"perf_stat_summary_{DATE_STR}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "workload", "instructions", "cycles", "ipc",
            "llc_load_misses", "llc_loads",
            "cache_references", "cache_misses",
        ])
        for wl_name in order:
            r = all_results.get(wl_name, {})
            writer.writerow([
                wl_name,
                r.get('instructions', ''),
                r.get('cycles', ''),
                f"{r['ipc']:.4f}" if 'ipc' in r else '',
                r.get('llc_load_misses', ''),
                r.get('llc_loads', ''),
                r.get('cache_references', ''),
                r.get('cache_misses', ''),
            ])
    log(f"\nCSV: {csv_path}")

    log(f"\nDone! Individual perf stat results: logs/perf_stat_<workload>_{DATE_STR}.txt")
    log(f"Transfer to Mac:")
    log(f"  scp -P 4247 optimus@203.255.176.80:{csv_path} ~/capstone2/data/")


if __name__ == "__main__":
    main()
