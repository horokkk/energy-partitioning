#!/usr/bin/env python3
"""run_cgroup_experiments.py - Automated cgroup experiment runner

10 combos x 7 ratios = 70 cgroup experiments, executed sequentially.
- Waits for system idle before each experiment
- Automatically starts/stops concurrent_power.py
- Runs workloads inside cgroups
- Logs per-experiment timestamps (for raspi wall CSV matching)

Run location: gpu server (203.255.176.80)
Prerequisites:
  1. cgroup groups created (/sys/fs/cgroup/optimus/vm_a, vm_b)
  2. wall_power.py running on raspi (long duration)
  3. sudo required

Usage:
  sudo python3 run_cgroup_experiments.py                           # all 24
  sudo python3 run_cgroup_experiments.py --experiments CG-A1 CG-A2 # select by ID
  sudo python3 run_cgroup_experiments.py --combo A B               # select by combo
  sudo python3 run_cgroup_experiments.py --ratio 1 2               # select by ratio
  sudo python3 run_cgroup_experiments.py --dry-run                 # test mode
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
# sudo secure_path overrides venv PATH, so use absolute path to venv python
VENV_PYTHON = f"{HOME}/yolo/venv/bin/python3"

# Experiment protocol (same as 0208/0209/0211)
IDLE_WAIT = 30       # Idle period after measurement start (sec)
WORKLOAD_DUR = 120   # Workload execution time (sec)
COOLDOWN = 30        # Cooldown after workload stop (sec)
TOTAL_MEASURE = IDLE_WAIT + WORKLOAD_DUR + COOLDOWN  # 180 sec

# Idle detection settings
IDLE_CPU_TH = 5.0     # CPU utilization threshold (%)
IDLE_GPU_TH = 5.0     # GPU utilization threshold (%)
IDLE_STABLE_SEC = 5   # Required consecutive stable seconds
IDLE_TIMEOUT = 120    # Max wait time (sec)

# cgroup paths (cgroup v2)
CGROUP_BASE = "/sys/fs/cgroup/optimus"
CGROUP_VM_A = os.path.join(CGROUP_BASE, "vm_a")
CGROUP_VM_B = os.path.join(CGROUP_BASE, "vm_b")

# ==========================================
# Workload definitions
# ==========================================
# sudo secure_path overrides venv PATH, so use absolute path to venv python
WORKLOADS = {
    "resnet": {
        "cmd": f"{VENV_PYTHON} {SCRIPTS_DIR}/workload_resnet.py -t {WORKLOAD_DUR}",
        "pattern": "workload_resnet.py",
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
        "pattern": "workload_llm.py",  # .py included to distinguish from workload_gpu_llm
    },
    "nodejs": {
        "cmd_server": f"node {HOME}/node/server.js",
        "cmd_load": "autocannon -c 100 -d {dur} http://localhost:3000",
        "pattern": "server.js",
        "two_phase": True,
    },
    "resnet_gpu": {
        "cmd": f"{VENV_PYTHON} {SCRIPTS_DIR}/workload_resnet_gpu.py -t {WORKLOAD_DUR}",
        "pattern": "workload_resnet_gpu",
        "gpu": True,
    },
    # ── Phase 2 workloads (0225~) ──
    "training": {
        "cmd": f"{VENV_PYTHON} {SCRIPTS_DIR}/workload_training.py -t {WORKLOAD_DUR}",
        "pattern": "workload_training",
        "gpu": True,
    },
    "ffmpeg": {
        "cmd": f"{VENV_PYTHON} {SCRIPTS_DIR}/workload_ffmpeg.py -t {WORKLOAD_DUR}",
        "pattern": "libx264",  # Match actual ffmpeg child process (not the Python wrapper)
    },
    "gpu_llm": {
        "cmd": f"{VENV_PYTHON} {SCRIPTS_DIR}/workload_gpu_llm.py -t {WORKLOAD_DUR}",
        "pattern": "workload_gpu_llm",
        "gpu": True,
    },
}

# ==========================================
# Combo x Ratio = Experiment matrix
# ==========================================
# 6 combos (by interference type)
COMBINATIONS = {
    "A": {
        "vm_a_wl": "resnet", "vm_b_wl": "gemm",
        "name": "ResNet+GEMM",
        "type": "CPU↔GPU isolated",
        "desc": "CPU AI vs GPU compute (no-interference baseline)",
    },
    "B": {
        "vm_a_wl": "resnet", "vm_b_wl": "llm",
        "name": "ResNet+LLM",
        "type": "CPU↔CPU contention",
        "desc": "CPU AI vs CPU+Mem AI (core isolation effect)",
    },
    "C": {
        "vm_a_wl": "gemm", "vm_b_wl": "yolo",
        "name": "GEMM+YOLO",
        "type": "GPU↔GPU contention",
        "desc": "GPU sharing vGPU scenario (SM% partitioning)",
    },
    "D": {
        "vm_a_wl": "yolo", "vm_b_wl": "nodejs",
        "name": "YOLO+NodeJS",
        "type": "AI↔Non-AI",
        "desc": "AI vs Non-AI energy fairness",
    },
    "E": {
        "vm_a_wl": "gemm", "vm_b_wl": "nodejs",
        "name": "GEMM+NodeJS",
        "type": "Non-AI↔GPU full-load",
        "desc": "Non-AI vs GPU 98% (extreme energy asymmetry)",
    },
    "D": {
        "vm_a_wl": "llm", "vm_b_wl": "nodejs",
        "name": "LLM+NodeJS",
        "type": "CPU AI↔Non-AI",
        "desc": "CPU+memory fairness without GPU",
    },
    "E": {
        "vm_a_wl": "resnet_gpu", "vm_b_wl": "resnet",
        "name": "ResNetGPU+ResNetCPU",
        "type": "Same model GPU↔CPU",
        "desc": "Same model GPU vs CPU deployment energy comparison",
    },
    "H": {
        "vm_a_wl": "resnet_gpu", "vm_b_wl": "nodejs",
        "name": "ResNetGPU+NodeJS",
        "type": "GPU AI↔Non-AI",
        "desc": "GPU inference vs Non-AI (cf. Combo D: YOLO vs ResNetGPU)",
    },
    "F": {
        "vm_a_wl": "resnet_gpu", "vm_b_wl": "llm",
        "name": "ResNetGPU+LLM",
        "type": "GPU AI↔CPU AI",
        "desc": "GPU inference vs CPU AI (cf. Combo A: GEMM vs ResNetGPU)",
    },
    # ── Phase 2 combos (0225~) ──
    "G": {
        "vm_a_wl": "ffmpeg", "vm_b_wl": "llm",
        "name": "ffmpeg+LLM",
        "type": "CPU↔CPU (compute vs memory)",
        "desc": "CPU compute-bound vs memory-bound — IPC difference reproducibility (cf. Combo B)",
    },
    "H": {
        "vm_a_wl": "training", "vm_b_wl": "ffmpeg",
        "name": "Training+ffmpeg",
        "type": "GPU Training↔CPU Compute",
        "desc": "Training GPU independence + CPU contention test",
    },
    "I": {
        "vm_a_wl": "gpu_llm", "vm_b_wl": "gemm",
        "name": "GPU-LLM+GEMM",
        "type": "GPU↔GPU (memory vs compute)",
        "desc": "GPU memory-heavy vs SM-heavy spectrum",
    },
    "J": {
        "vm_a_wl": "training", "vm_b_wl": "nodejs",
        "name": "Training+NodeJS",
        "type": "AI Training↔Non-AI",
        "desc": "Training vs Non-AI energy fairness (cf. Combo D)",
    },
}

# 7 ratios (proportional allocation) — 20 total cores (E5-2630v4: 10Cx2T), 24GB allocated (8GB reserved for system)
# 24 = LCM(2,3,4,6) — all ratios divide evenly into integers
RATIOS = {
    "1": {
        "label": "1:5",
        "vm_a_cpus": "0-2",   "vm_b_cpus": "3-19",
        "vm_a_mem": "4G",     "vm_b_mem": "20G",
        "vm_a_cores": 3,      "vm_b_cores": 17,
    },
    "2": {
        "label": "1:3",
        "vm_a_cpus": "0-4",   "vm_b_cpus": "5-19",
        "vm_a_mem": "6G",     "vm_b_mem": "18G",
        "vm_a_cores": 5,      "vm_b_cores": 15,
    },
    "3": {
        "label": "1:2",
        "vm_a_cpus": "0-6",   "vm_b_cpus": "7-19",
        "vm_a_mem": "8G",     "vm_b_mem": "16G",
        "vm_a_cores": 7,      "vm_b_cores": 13,
    },
    "4": {
        "label": "1:1",
        "vm_a_cpus": "0-9",   "vm_b_cpus": "10-19",
        "vm_a_mem": "12G",    "vm_b_mem": "12G",
        "vm_a_cores": 10,     "vm_b_cores": 10,
    },
    "5": {
        "label": "2:1",
        "vm_a_cpus": "0-12",  "vm_b_cpus": "13-19",
        "vm_a_mem": "16G",    "vm_b_mem": "8G",
        "vm_a_cores": 13,     "vm_b_cores": 7,
    },
    "6": {
        "label": "3:1",
        "vm_a_cpus": "0-14",  "vm_b_cpus": "15-19",
        "vm_a_mem": "18G",    "vm_b_mem": "6G",
        "vm_a_cores": 15,     "vm_b_cores": 5,
    },
    "7": {
        "label": "5:1",
        "vm_a_cpus": "0-16",  "vm_b_cpus": "17-19",
        "vm_a_mem": "20G",    "vm_b_mem": "4G",
        "vm_a_cores": 17,     "vm_b_cores": 3,
    },
}

DATE_STR = datetime.now().strftime("%m%d")
TAG = ""  # Set via --tag, appended to filenames (to distinguish Run 1/2)

def get_experiment_id(combo_key, ratio_key):
    return f"CG-{combo_key}{ratio_key}"

def build_experiment(combo_key, ratio_key, mem_fixed=None):
    """Build experiment config from combo+ratio. mem_fixed: fix both VMs' memory (e.g., "12G")"""
    combo = COMBINATIONS[combo_key]
    ratio = RATIOS[ratio_key]
    exp_id = get_experiment_id(combo_key, ratio_key)
    vm_a_mem = mem_fixed if mem_fixed else ratio["vm_a_mem"]
    vm_b_mem = mem_fixed if mem_fixed else ratio["vm_b_mem"]
    desc = f"{combo['name']} ({ratio['label']}"
    if mem_fixed:
        desc += f", mem={mem_fixed}"
    desc += f") — {combo['type']}"
    return {
        "id": exp_id,
        "combo": combo_key,
        "ratio": ratio_key,
        "desc": desc,
        "vm_a": {
            "workload": combo["vm_a_wl"],
            "cpus": ratio["vm_a_cpus"],
            "mem": vm_a_mem,
        },
        "vm_b": {
            "workload": combo["vm_b_wl"],
            "cpus": ratio["vm_b_cpus"],
            "mem": vm_b_mem,
        },
    }

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
_prev_cpu = None

def get_cpu_util():
    """Get current CPU idle/total ticks"""
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
    """Get current GPU utilization (%)"""
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
    """Wait until system reaches idle state"""
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
def setup_cgroup(cgroup_path, cpus, mem):
    """Assign CPU cores and memory to cgroup"""
    try:
        with open(os.path.join(cgroup_path, "cpuset.cpus"), "w") as f:
            f.write(cpus)
        with open(os.path.join(cgroup_path, "memory.max"), "w") as f:
            f.write(mem)
        log(f"  {os.path.basename(cgroup_path)}: cpus={cpus}, mem={mem}")
        return True
    except Exception as e:
        log(f"ERROR: cgroup setup failed {cgroup_path}: {e}")
        return False

def reset_cgroup(cgroup_path):
    """Remove cgroup resource limits"""
    try:
        with open(os.path.join(cgroup_path, "cpuset.cpus"), "w") as f:
            f.write("0-19")
        with open(os.path.join(cgroup_path, "memory.max"), "w") as f:
            f.write("max")
    except Exception:
        pass

# ==========================================
# Workload execution
# ==========================================
def start_workload_in_cgroup(cgroup_path, workload_name, cuda_device=None, output_dir=None):
    """Start workload inside cgroup. cuda_device: GPU index (e.g., "0", "1")"""
    wl = WORKLOADS[workload_name]

    env = os.environ.copy()
    # GPU workloads: use --device cuda:N directly instead of CUDA_VISIBLE_DEVICES
    # (GPU assignment bug when using CUDA_VISIBLE_DEVICES after GeForce 210 removal)
    env.pop("CUDA_VISIBLE_DEVICES", None)

    # Workload error log path
    err_log = os.path.join(LOGS_DIR, f"wl_{workload_name}_{DATE_STR}.err")
    err_f = open(err_log, "w")

    if wl.get("two_phase"):
        # NodeJS: pass access log path via LOG_DIR env var
        if output_dir:
            env["LOG_DIR"] = output_dir
            log(f"  LOG_DIR={output_dir}")

        # NodeJS: start server -> move to cgroup -> start autocannon -> move to cgroup
        server_proc = subprocess.Popen(
            ["bash", "-c", wl["cmd_server"]],
            stdout=subprocess.DEVNULL, stderr=err_f,
            preexec_fn=os.setsid,
            env=env,
        )
        # Move PID to cgroup
        with open(os.path.join(cgroup_path, "cgroup.procs"), "w") as f:
            f.write(str(server_proc.pid))
        log(f"  NodeJS server started (PID={server_proc.pid})")
        time.sleep(2)

        load_cmd = wl["cmd_load"].format(dur=WORKLOAD_DUR)
        load_proc = subprocess.Popen(
            ["bash", "-c", load_cmd],
            stdout=subprocess.DEVNULL, stderr=err_f,
            preexec_fn=os.setsid,
            env=env,
        )
        with open(os.path.join(cgroup_path, "cgroup.procs"), "w") as f:
            f.write(str(load_proc.pid))
        log(f"  autocannon started (PID={load_proc.pid})")
        return [server_proc, load_proc]
    else:
        # Start process then move to cgroup (workaround for exec source issue)
        cmd = wl["cmd"]
        # GPU workload: specify GPU directly via --device cuda:N
        if cuda_device is not None and wl.get("gpu"):
            cmd += f" --device cuda:{cuda_device}"
            log(f"  --device cuda:{cuda_device}")
        # Pass output directory
        if output_dir:
            cmd += f" --output-dir {output_dir}"
            log(f"  --output-dir {output_dir}")

        proc = subprocess.Popen(
            ["bash", "-c", cmd],
            stdout=subprocess.DEVNULL, stderr=err_f,
            preexec_fn=os.setsid,
            env=env,
        )
        with open(os.path.join(cgroup_path, "cgroup.procs"), "w") as f:
            f.write(str(proc.pid))
        log(f"  {workload_name} started (PID={proc.pid})")
        return [proc]

def stop_processes(procs):
    """Terminate a list of processes"""
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
    """Kill residual processes from previous experiments (server.js, workload_*, etc.)"""
    patterns = ["server.js", "workload_yolo", "workload_gemm",
                "workload_resnet", "workload_llm", "workload_training",
                "workload_gpu_llm", "ffmpeg", "autocannon"]
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
def run_experiment(exp, dry_run=False):
    """Run a single experiment"""
    exp_id = exp["id"]
    log_sep()
    log(f"Experiment {exp_id}: {exp['desc']}")
    log_sep()

    # 0. Clean up residual processes
    cleanup_residual_processes()

    # 0.5. Flush page cache (ensure I/O consistency between Solo/Conc)
    try:
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3")
        log("  Page cache flushed (drop_caches=3)")
    except Exception as e:
        log(f"  WARNING: drop_caches failed: {e}")

    vm_a = exp["vm_a"]
    vm_b = exp["vm_b"]

    # 1. Idle detection
    wait_for_idle()

    # 2. cgroup setup
    log("Setting up cgroups...")
    if not setup_cgroup(CGROUP_VM_A, vm_a["cpus"], vm_a["mem"]):
        return False
    if not setup_cgroup(CGROUP_VM_B, vm_b["cpus"], vm_b["mem"]):
        return False

    # 3. Start concurrent_power.py
    tag_s = f"_{TAG}" if TAG else ""
    csv_name = f"conc_{exp_id.lower()}{tag_s}_{DATE_STR}.csv"
    csv_path = os.path.join(LOGS_DIR, csv_name)
    pattern_a = WORKLOADS[vm_a["workload"]]["pattern"]
    pattern_b = WORKLOADS[vm_b["workload"]]["pattern"]

    conc_cmd = [
        "python3", CONC_POWER_PY,
        "--workloads", f"vm_a:{pattern_a}", f"vm_b:{pattern_b}",
        "-i", "1", "-t", str(TOTAL_MEASURE),
        "-o", csv_path,
    ]
    log(f"Measurement: {csv_name}")

    ts_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log(f">>> Start: {ts_start}")

    if dry_run:
        log(f"[DRY-RUN] vm_a: {vm_a['workload']}(cpus={vm_a['cpus']}, mem={vm_a['mem']})")
        log(f"[DRY-RUN] vm_b: {vm_b['workload']}(cpus={vm_b['cpus']}, mem={vm_b['mem']})")
        reset_cgroup(CGROUP_VM_A)
        reset_cgroup(CGROUP_VM_B)
        return True

    conc_proc = subprocess.Popen(
        conc_cmd,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    # 4. Idle period (30s)
    log(f"Idle period ({IDLE_WAIT}s)...")
    time.sleep(IDLE_WAIT)

    # 5. Start workloads (vm_a -> GPU 0, vm_b -> GPU 1)
    # Create output directories
    out_base = os.path.join(RESULTS_DIR, f"conc_{exp_id.lower()}{tag_s}_{DATE_STR}")
    out_a = os.path.join(out_base, "vm_a")
    out_b = os.path.join(out_base, "vm_b")
    os.makedirs(out_a, exist_ok=True)
    os.makedirs(out_b, exist_ok=True)

    log("Starting workloads...")
    procs_a = start_workload_in_cgroup(CGROUP_VM_A, vm_a["workload"], cuda_device=0, output_dir=out_a)
    procs_b = start_workload_in_cgroup(CGROUP_VM_B, vm_b["workload"], cuda_device=1, output_dir=out_b)

    # 6. Wait for workload (120s)
    log(f"Workload running ({WORKLOAD_DUR}s)...")
    time.sleep(WORKLOAD_DUR)

    # 7. Stop workloads
    log("Stopping workloads...")
    stop_processes(procs_a + procs_b)

    # 8. Cooldown (30s)
    log(f"Cooldown ({COOLDOWN}s)...")
    time.sleep(COOLDOWN)

    # 9. Stop concurrent_power.py
    try:
        conc_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        conc_proc.terminate()
        conc_proc.wait(timeout=5)

    ts_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log(f">>> End: {ts_end}")
    log(f"CSV: {csv_path}")

    # 10. Reset cgroup
    reset_cgroup(CGROUP_VM_A)
    reset_cgroup(CGROUP_VM_B)

    return True, ts_start, ts_end

# ==========================================
# Main
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Automated cgroup experiments (10 combos x 7 ratios = 70 experiments)")
    parser.add_argument("--experiments", nargs="+", default=None,
                        help="Experiment IDs to run (e.g., CG-A1 CG-B2)")
    parser.add_argument("--combo", nargs="+", default=None,
                        help="Combos to run (e.g., A B C)")
    parser.add_argument("--ratio", nargs="+", default=None,
                        help="Ratios to run (e.g., 1 2)")
    parser.add_argument("--mem-fixed", default=None,
                        help="Fix both VMs' memory (e.g., 12G). For CPU-only sweep")
    parser.add_argument("--tag", default="",
                        help="Tag appended to filenames (e.g., mf). For Run 1/2 distinction")
    parser.add_argument("--dry-run", action="store_true",
                        help="Test mode (no workload execution)")
    args = parser.parse_args()

    # Determine list of experiments to run
    combo_keys = list(COMBINATIONS.keys())  # A~F
    ratio_keys = list(RATIOS.keys())        # 1~4

    if args.combo:
        combo_keys = [c.upper() for c in args.combo]
    if args.ratio:
        ratio_keys = [r for r in args.ratio]

    # TAG setting (for filename distinction)
    global TAG
    TAG = args.tag

    experiments = []
    if args.experiments:
        # Directly specified
        for eid in args.experiments:
            eid = eid.upper().replace("CG-", "")
            if len(eid) == 2:
                c, r = eid[0], eid[1]
                if c in COMBINATIONS and r in RATIOS:
                    experiments.append(build_experiment(c, r, mem_fixed=args.mem_fixed))
                else:
                    print(f"ERROR: Unknown experiment ID 'CG-{eid}'")
                    print(f"  Combos: {', '.join(COMBINATIONS.keys())}")
                    print(f"  Ratios: {', '.join(RATIOS.keys())}")
                    sys.exit(1)
    else:
        # Generate combo x ratio combinations
        for c in combo_keys:
            for r in ratio_keys:
                experiments.append(build_experiment(c, r, mem_fixed=args.mem_fixed))

    if not experiments:
        print("ERROR: No experiments to run.")
        sys.exit(1)

    # Check sudo
    if os.geteuid() != 0 and not args.dry_run:
        print("ERROR: sudo is required.")
        print("  sudo python3 run_cgroup_experiments.py")
        sys.exit(1)

    # Verify directories
    os.makedirs(LOGS_DIR, exist_ok=True)

    # Verify cgroup directories
    for cg in [CGROUP_BASE, CGROUP_VM_A, CGROUP_VM_B]:
        if not os.path.isdir(cg):
            print(f"ERROR: cgroup directory not found: {cg}")
            print(f"  sudo mkdir -p {CGROUP_VM_A} {CGROUP_VM_B}")
            sys.exit(1)

    # Log file
    global _log_file
    tag_s = f"_{TAG}" if TAG else ""
    log_path = os.path.join(LOGS_DIR, f"experiment_log{tag_s}_{DATE_STR}.txt")
    _log_file = open(log_path, "a")

    # Print experiment matrix
    log_sep()
    log(f"Automated cgroup experiments — {len(experiments)} total")
    log(f"Protocol: {IDLE_WAIT}s idle + {WORKLOAD_DUR}s workload + {COOLDOWN}s cooldown")
    if args.dry_run:
        log("[DRY-RUN mode]")
    if args.mem_fixed:
        log(f"[MEM-FIXED: both VMs = {args.mem_fixed}] — CPU-only sweep")
    if TAG:
        log(f"[TAG: {TAG}] — filename suffix")
    log("")
    log("Experiment matrix:")
    log(f"  {'ID':<8} {'Combo':<16} {'Ratio':<6} {'vm_a cores':<10} {'vm_b cores':<10} "
        f"{'vm_a mem':<10} {'vm_b mem':<10}")
    log(f"  {'-'*8} {'-'*16} {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for exp in experiments:
        log(f"  {exp['id']:<8} "
            f"{COMBINATIONS[exp['combo']]['name']:<16} "
            f"{RATIOS[exp['ratio']]['label']:<6} "
            f"{exp['vm_a']['cpus']:<10} {exp['vm_b']['cpus']:<10} "
            f"{exp['vm_a']['mem']:<10} {exp['vm_b']['mem']:<10}")
    log_sep()

    # Timestamp log
    ts_log_path = os.path.join(LOGS_DIR, f"timestamps{tag_s}_{DATE_STR}.csv")
    ts_log = open(ts_log_path, "w", newline="")
    ts_writer = csv.writer(ts_log)
    ts_writer.writerow(["exp_id", "combo", "ratio", "desc",
                         "start_time", "end_time",
                         "vm_a_workload", "vm_a_cpus", "vm_a_mem",
                         "vm_b_workload", "vm_b_cpus", "vm_b_mem"])

    # Run experiments
    success_count = 0
    for i, exp in enumerate(experiments):
        log(f"\n[{i+1}/{len(experiments)}] {exp['id']} preparing...")

        result = run_experiment(exp, dry_run=args.dry_run)

        if isinstance(result, tuple):
            ok, ts_start, ts_end = result
        else:
            ok = result
            ts_start = ts_end = ""

        if ok:
            success_count += 1
            ts_writer.writerow([
                exp["id"], exp["combo"], exp["ratio"], exp["desc"],
                ts_start, ts_end,
                exp["vm_a"]["workload"], exp["vm_a"]["cpus"], exp["vm_a"]["mem"],
                exp["vm_b"]["workload"], exp["vm_b"]["cpus"], exp["vm_b"]["mem"],
            ])
            ts_log.flush()

        log(f"{exp['id']} {'SUCCESS' if ok else 'FAILED'}")

    # Done
    log_sep()
    log(f"All done: {success_count}/{len(experiments)} succeeded")
    log(f"Timestamps: {ts_log_path}")
    log(f"Log: {log_path}")
    log_sep()

    ts_log.close()
    _log_file.close()

    print(f"\nNext steps:")
    print(f"  1. Stop raspi wall_power.py")
    print(f"  2. Transfer wall CSV + experiment CSV to Mac")
    print(f"  3. Use timestamps_{DATE_STR}.csv to segment wall CSV")


if __name__ == "__main__":
    main()
