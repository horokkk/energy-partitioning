#!/usr/bin/env python3
"""run_solo_experiments.py - Automated Solo baseline experiments

7 experiments: idle + 6 workloads (ResNet-CPU, ResNet-GPU, LLM, GEMM, YOLO, NodeJS)
All under cgroup with 10 cores / 14G (matching 1:1 ratio)

Run location: gpu server (203.255.176.80)
Prerequisites:
  1. cgroup group created (/sys/fs/cgroup/optimus/vm_a)
  2. wall_power.py running on raspi (long duration)
  3. sudo required

Usage:
  sudo python3 -u run_solo_experiments.py 2>&1 | tee logs/solo_run_MMDD.log
  sudo python3 -u run_solo_experiments.py --workloads resnet_gpu  # select workloads
  sudo python3 -u run_solo_experiments.py --skip-idle             # skip idle baseline
  sudo python3 -u run_solo_experiments.py --dry-run               # test mode
"""

import subprocess
import time
import os
import sys
import signal
import argparse
import csv
from datetime import datetime
from pathlib import Path

# ==========================================
# Configuration
# ==========================================
HOME = "/home/optimus"
SCRIPTS_DIR = f"{HOME}/jiyoon_energy/scripts"
LOGS_DIR = os.path.join(SCRIPTS_DIR, "logs")
RESULTS_DIR = os.path.join(LOGS_DIR, "results")
CONC_POWER_PY = os.path.join(SCRIPTS_DIR, "concurrent_power.py")
VENV_PYTHON = f"{HOME}/yolo/venv/bin/python3"

# Experiment protocol
IDLE_WAIT = 30
WORKLOAD_DUR = 120
COOLDOWN = 30
TOTAL_MEASURE = IDLE_WAIT + WORKLOAD_DUR + COOLDOWN  # 180 sec

# Idle detection settings
IDLE_CPU_TH = 5.0
IDLE_GPU_TH = 5.0
IDLE_STABLE_SEC = 5
IDLE_TIMEOUT = 120

# cgroup path (solo mode: only vm_a used)
CGROUP_BASE = "/sys/fs/cgroup/optimus"
CGROUP_VM = os.path.join(CGROUP_BASE, "vm_a")

# Solo cgroup settings (matching 1:1 ratio)
SOLO_CPUS = "0-9"
SOLO_MEM = "14G"
SOLO_CORES = 10

# ==========================================
# Workload definitions
# ==========================================
WORKLOADS = {
    "resnet": {
        "cmd": f"{VENV_PYTHON} {SCRIPTS_DIR}/workload_resnet.py -t {WORKLOAD_DUR}",
        "pattern": "workload_resnet.py",  # .py included to prevent matching resnet_gpu
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
        "cmd_load": "autocannon -c 100 -d {dur} http://localhost:3000",
        "pattern": "server.js",
        "two_phase": True,
        "gpu": False,
    },
    # Phase 2 workloads
    "training": {
        "cmd": f"{VENV_PYTHON} {SCRIPTS_DIR}/workload_training.py -t {WORKLOAD_DUR}",
        "pattern": "workload_training",
        "gpu": True,
    },
    "ffmpeg": {
        "cmd": f"{VENV_PYTHON} {SCRIPTS_DIR}/workload_ffmpeg.py -t {WORKLOAD_DUR}",
        "pattern": "libx264",  # Match actual ffmpeg process (not the Python wrapper)
        "gpu": False,
    },
    "gpu_llm": {
        "cmd": f"{VENV_PYTHON} {SCRIPTS_DIR}/workload_gpu_llm.py -t {WORKLOAD_DUR}",
        "pattern": "workload_gpu_llm",
        "gpu": True,
    },
}

# Solo experiment order (idle first, then GPU workloads, then CPU workloads)
SOLO_ORDER = ["idle", "resnet_gpu", "gemm", "yolo", "training", "gpu_llm",
              "resnet", "llm", "ffmpeg", "nodejs"]

DATE_STR = datetime.now().strftime("%m%d")

# ==========================================
# Logging
# ==========================================
_log_file = None

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    if _log_file:
        _log_file.write(line + "\n")
        _log_file.flush()

def log_sep():
    log("=" * 60)

# ==========================================
# Idle detection
# ==========================================
def get_cpu_util():
    try:
        with open("/proc/stat") as f:
            line = f.readline()
        parts = line.split()
        times = [int(x) for x in parts[1:]]
        idle = times[3] + times[4]
        total = sum(times)
        return idle, total
    except Exception:
        return None, None

def get_gpu_util():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            text=True, timeout=2
        ).strip()
        return float(out.split("\n")[0].strip())
    except Exception:
        return 0.0

def wait_for_idle():
    log(f"Idle detection (CPU<{IDLE_CPU_TH}%, GPU<{IDLE_GPU_TH}%, "
        f"{IDLE_STABLE_SEC}s stable)")

    stable = 0
    prev_idle, prev_total = get_cpu_util()
    time.sleep(1)
    start = time.time()

    while stable < IDLE_STABLE_SEC:
        if time.time() - start > IDLE_TIMEOUT:
            log(f"WARNING: Idle wait exceeded {IDLE_TIMEOUT}s — forcing start")
            return False

        curr_idle, curr_total = get_cpu_util()
        if curr_idle is None or prev_total is None:
            time.sleep(1)
            continue

        d_idle = curr_idle - prev_idle
        d_total = curr_total - prev_total
        cpu_util = (1.0 - d_idle / d_total) * 100.0 if d_total > 0 else 0.0
        gpu_util = get_gpu_util()

        prev_idle, prev_total = curr_idle, curr_total

        if cpu_util < IDLE_CPU_TH and gpu_util < IDLE_GPU_TH:
            stable += 1
            log(f"  CPU={cpu_util:.1f}%, GPU={gpu_util:.0f}% "
                f"(stable {stable}/{IDLE_STABLE_SEC})")
        else:
            if stable > 0:
                log(f"  Reset: CPU={cpu_util:.1f}%, GPU={gpu_util:.0f}%")
            stable = 0

        time.sleep(1)

    log("Idle confirmed")
    return True

# ==========================================
# cgroup setup
# ==========================================
def setup_cgroup(cpus, mem):
    try:
        with open(os.path.join(CGROUP_VM, "cpuset.cpus"), "w") as f:
            f.write(cpus)
        with open(os.path.join(CGROUP_VM, "memory.max"), "w") as f:
            f.write(mem)
        log(f"  cgroup: cpus={cpus}, mem={mem}")
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

# ==========================================
# Workload execution
# ==========================================
def start_workload_in_cgroup(workload_name, output_dir=None):
    wl = WORKLOADS[workload_name]
    err_log = os.path.join(LOGS_DIR, f"wl_solo_{workload_name}_{DATE_STR}.err")
    err_f = open(err_log, "w")

    env = os.environ.copy()
    # Do not use CUDA_VISIBLE_DEVICES (bug during concurrent execution after GeForce 210 removal)
    env.pop("CUDA_VISIBLE_DEVICES", None)

    if wl.get("two_phase"):
        # NodeJS
        if output_dir:
            env["LOG_DIR"] = output_dir
            log(f"  LOG_DIR={output_dir}")

        server_proc = subprocess.Popen(
            ["bash", "-c", wl["cmd_server"]],
            stdout=subprocess.DEVNULL, stderr=err_f,
            preexec_fn=os.setsid, env=env,
        )
        with open(os.path.join(CGROUP_VM, "cgroup.procs"), "w") as f:
            f.write(str(server_proc.pid))
        log(f"  NodeJS server started (PID={server_proc.pid})")
        time.sleep(2)

        load_cmd = wl["cmd_load"].format(dur=WORKLOAD_DUR)
        load_proc = subprocess.Popen(
            ["bash", "-c", load_cmd],
            stdout=subprocess.DEVNULL, stderr=err_f,
            preexec_fn=os.setsid, env=env,
        )
        with open(os.path.join(CGROUP_VM, "cgroup.procs"), "w") as f:
            f.write(str(load_proc.pid))
        log(f"  autocannon started (PID={load_proc.pid})")
        return [server_proc, load_proc]
    else:
        cmd = wl["cmd"]
        # GPU workload: --device cuda:0 (Solo always uses GPU0)
        if wl.get("gpu"):
            cmd += " --device cuda:0"
            log(f"  --device cuda:0")
        if output_dir:
            cmd += f" --output-dir {output_dir}"
            log(f"  --output-dir {output_dir}")

        proc = subprocess.Popen(
            ["bash", "-c", cmd],
            stdout=subprocess.DEVNULL, stderr=err_f,
            preexec_fn=os.setsid, env=env,
        )
        with open(os.path.join(CGROUP_VM, "cgroup.procs"), "w") as f:
            f.write(str(proc.pid))
        log(f"  {workload_name} started (PID={proc.pid})")
        return [proc]

def stop_processes(procs):
    for proc in procs:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass
    deadline = time.time() + 5
    for proc in procs:
        remaining = max(0, deadline - time.time())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass

def cleanup_residual_processes():
    patterns = ["server.js", "workload_yolo", "workload_gemm",
                "workload_resnet", "workload_llm", "autocannon",
                "workload_training", "libx264", "workload_gpu_llm", "ffmpeg"]
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
            if 'concurrent_power' in cmdline:
                continue
            for pat in patterns:
                if pat in cmdline:
                    os.kill(pid, signal.SIGKILL)
                    killed.append((pid, pat))
                    break
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue
    if killed:
        log(f"  Residual processes killed: {killed}")
    return killed

# ==========================================
# Experiment execution
# ==========================================
def run_solo_experiment(workload_name, dry_run=False):
    exp_id = f"S-{workload_name}-r1" if workload_name != "idle" else "S-idle-r1"
    log_sep()
    log(f"Experiment {exp_id}: {workload_name} solo (cpus={SOLO_CPUS}, mem={SOLO_MEM})")
    log_sep()

    # 0. Clean up residual processes
    cleanup_residual_processes()

    # 0.5. Flush page cache
    try:
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3")
        log("  Page cache flushed (drop_caches=3)")
    except Exception as e:
        log(f"  WARNING: drop_caches failed: {e}")

    # 1. Idle detection
    wait_for_idle()

    # 2. cgroup setup
    log("Setting up cgroup...")
    if not setup_cgroup(SOLO_CPUS, SOLO_MEM):
        return False, "", ""

    # 3. Start concurrent_power.py
    csv_name = f"solo_{exp_id.lower()}_{DATE_STR}.csv"
    csv_path = os.path.join(LOGS_DIR, csv_name)

    if workload_name != "idle":
        pattern = WORKLOADS[workload_name]["pattern"]
        conc_cmd = [
            "python3", CONC_POWER_PY,
            "--workloads", f"vm_a:{pattern}",
            "-i", "1", "-t", str(TOTAL_MEASURE),
            "-o", csv_path,
        ]
    else:
        conc_cmd = [
            "python3", CONC_POWER_PY,
            "--workloads", "vm_a:__idle_placeholder__",
            "-i", "1", "-t", str(TOTAL_MEASURE),
            "-o", csv_path,
        ]

    log(f"Measurement: {csv_name}")
    ts_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log(f">>> Start: {ts_start}")

    if dry_run:
        log(f"[DRY-RUN] {workload_name}(cpus={SOLO_CPUS}, mem={SOLO_MEM})")
        reset_cgroup()
        return True, ts_start, ts_start

    conc_proc = subprocess.Popen(
        conc_cmd,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    # 4. Idle period
    log(f"Idle period ({IDLE_WAIT}s)...")
    time.sleep(IDLE_WAIT)

    # 5. Start workload (skip if idle)
    procs = []
    if workload_name != "idle":
        output_dir = os.path.join(RESULTS_DIR, f"solo_{exp_id.lower()}_{DATE_STR}")
        os.makedirs(output_dir, exist_ok=True)

        log("Starting workload...")
        procs = start_workload_in_cgroup(workload_name, output_dir=output_dir)

        log(f"Workload running ({WORKLOAD_DUR}s)...")
        time.sleep(WORKLOAD_DUR)

        log("Stopping workload...")
        stop_processes(procs)
    else:
        log(f"Idle baseline ({WORKLOAD_DUR}s)...")
        time.sleep(WORKLOAD_DUR)

    # 6. Cooldown
    log(f"Cooldown ({COOLDOWN}s)...")
    time.sleep(COOLDOWN)

    # 7. Stop measurement
    try:
        conc_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        conc_proc.terminate()
        conc_proc.wait(timeout=5)

    ts_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log(f">>> End: {ts_end}")
    log(f"CSV: {csv_path}")

    # 8. Reset cgroup
    reset_cgroup()

    return True, ts_start, ts_end

# ==========================================
# Main
# ==========================================
def main():
    parser = argparse.ArgumentParser(
        description="Solo baseline experiments (7: idle + 6 workloads, 10 cores / 14G)")
    parser.add_argument("--workloads", nargs="+", default=None,
                        help="Select workloads (e.g., resnet_gpu gemm)")
    parser.add_argument("--skip-idle", action="store_true",
                        help="Skip idle baseline")
    parser.add_argument("--dry-run", action="store_true",
                        help="Test mode")
    parser.add_argument("--mem", default=None,
                        help="Memory allocation override (e.g., 12G)")
    args = parser.parse_args()

    # --mem override
    global SOLO_MEM
    if args.mem:
        SOLO_MEM = args.mem

    # Build workload list
    if args.workloads:
        order = args.workloads
        if not args.skip_idle and "idle" not in order:
            order = ["idle"] + order
    else:
        order = SOLO_ORDER[:]

    if args.skip_idle:
        order = [w for w in order if w != "idle"]

    if not order:
        print("ERROR: No experiments to run.")
        sys.exit(1)

    # Check sudo
    if os.geteuid() != 0 and not args.dry_run:
        print("ERROR: sudo is required.")
        print("  sudo python3 -u run_solo_experiments.py 2>&1 | tee logs/solo_run.log")
        sys.exit(1)

    os.makedirs(LOGS_DIR, exist_ok=True)

    # Verify cgroup directory
    if not os.path.isdir(CGROUP_VM):
        print(f"ERROR: cgroup directory not found: {CGROUP_VM}")
        print(f"  sudo mkdir -p {CGROUP_VM}")
        print(f"  sudo chown -R optimus:optimus {CGROUP_BASE}")
        sys.exit(1)

    # Log file
    global _log_file
    log_path = os.path.join(LOGS_DIR, f"solo_log_{DATE_STR}.txt")
    _log_file = open(log_path, "a")

    # Print experiment list
    log_sep()
    log(f"Solo baseline — {len(order)} total")
    log(f"Protocol: {IDLE_WAIT}s idle + {WORKLOAD_DUR}s workload + {COOLDOWN}s cooldown")
    log(f"cgroup: cpus={SOLO_CPUS}, mem={SOLO_MEM} ({SOLO_CORES} cores)")
    if args.dry_run:
        log("[DRY-RUN mode]")
    log(f"\nExperiment order: {', '.join(order)}")
    est_min = len(order) * (TOTAL_MEASURE + 60) / 60
    log(f"Estimated time: ~{est_min:.0f} min")
    log_sep()

    # Timestamp log
    ts_log_path = os.path.join(LOGS_DIR, f"solo_timestamps_{DATE_STR}.csv")
    ts_log = open(ts_log_path, "w", newline="")
    ts_writer = csv.writer(ts_log)
    ts_writer.writerow(["exp_id", "workload", "cpus", "mem", "cores",
                         "start_time", "end_time"])

    # Run experiments
    success_count = 0
    for i, wl_name in enumerate(order):
        log(f"\n[{i+1}/{len(order)}] {wl_name} preparing...")

        ok, ts_start, ts_end = run_solo_experiment(wl_name, dry_run=args.dry_run)

        if ok:
            success_count += 1
            exp_id = f"S-{wl_name}-r1" if wl_name != "idle" else "S-idle-r1"
            ts_writer.writerow([
                exp_id, wl_name, SOLO_CPUS, SOLO_MEM, SOLO_CORES,
                ts_start, ts_end,
            ])
            ts_log.flush()

        log(f"{wl_name} {'SUCCESS' if ok else 'FAILED'}")

    # Done
    log_sep()
    log(f"All done: {success_count}/{len(order)} succeeded")
    log(f"Timestamps: {ts_log_path}")
    log(f"Log: {log_path}")
    log_sep()

    ts_log.close()
    _log_file.close()

    print(f"\nNext steps:")
    print(f"  1. Stop raspi wall_power.py (Ctrl+C)")
    print(f"  2. Transfer wall CSV + solo CSV to Mac")
    print(f"  3. Match wall data with merge_data.py")


if __name__ == "__main__":
    main()
