# Energy Behavior of AI Workloads under Resource Partitioning



This repository contains the experiment scripts, measurement tools, and analysis code for studying the energy characteristics of 9 AI/non-AI workloads under various CPU/memory resource partitioning. We use cgroup v2 to simulate VMs on a dual-GPU server and measure energy at the component level (CPU via RAPL, GPU via nvidia-smi, DRAM via RAPL, wall power via a serial power meter).

## Quick Start

**Analyze pre-collected data** (no server required):

```bash
git clone <this-repo> && cd energy-partitioning

# Install Python dependencies
pip install pandas numpy matplotlib

# Place your data (see "Data Layout" below), then:
python analysis/merge_data.py        # Merge CSVs + generate summary
python analysis/plot_report.py       # Generate 14 figures → graphs/
python analysis/export_gsheet.py     # Export 7 summary CSVs → project root
```

**Run experiments from scratch** (requires the measurement server):

```bash
# 1. On Raspberry Pi: start wall power logging
python measurement/wall_power.py -t 3600 -o wall_cgroup.csv

# 2. On GPU server: run solo baselines (10 workloads)
sudo python experiments/run_solo_experiments.py

# 3. On GPU server: run concurrent experiments (10 combos x 4 ratios)
sudo python experiments/run_cgroup_experiments.py --combo A B C D E F G H I J

# 4. Transfer CSVs to local machine, then run analysis (see above)
```

## Installation

**Python 3.8+** required.

```bash
pip install -r requirements.txt
```

<details>
<summary>System dependencies (for running experiments)</summary>

| Package | Purpose | Install |
|---------|---------|---------|
| ffmpeg + libx264 | ffmpeg workload | `sudo apt install ffmpeg` |
| Node.js + npm | NodeJS workload | `sudo apt install nodejs npm` |
| autocannon | NodeJS load generator | `npm install -g autocannon` |
| perf | IPC/cache measurement | `sudo apt install linux-tools-$(uname -r)` |

</details>

## Data Layout

The analysis scripts expect the following directory structure **relative to the project root**:

```
energy-partitioning/           # project root
├── data/                      # raw experiment CSVs (NOT included in repo)
│   ├── conc_cg-a1_0223.csv   # concurrent experiment: combo A, ratio 1
│   ├── conc_cg-a2_0223.csv
│   ├── ...                    # 40 concurrent CSVs total
│   ├── solo_s-resnet-r1_0224.csv
│   ├── ...                    # 10 solo CSVs
│   ├── wall_cgroup_0223.csv   # wall power from Raspberry Pi
│   ├── wall_solo_0224.csv
│   ├── timestamps_0223.csv
│   └── perf_stat_summary.csv  # perf stat results (IPC, LLC)
├── data_merged/               # OUTPUT of merge_data.py
│   ├── summary.csv            # 50-row summary (10 solo + 40 concurrent)
│   ├── conc_cg-a1.csv         # merged CSV with wall_W column
│   └── ...
├── graphs/                    # OUTPUT of plot_report.py (14 PNGs)
├── analysis/
├── experiments/
├── measurement/
└── workloads/
```

**File naming convention:**
- `conc_cg-{combo}{ratio}_{date}.csv` — concurrent experiment (e.g., `conc_cg-a1_0223.csv` = combo A, ratio 1, collected 2025-02-23)
- `solo_s-{workload}-r1_{date}.csv` — solo baseline
- `wall_*.csv` — wall power meter readings (timestamp + wall_W)

## Repository Structure

```
energy-partitioning/
├── README.md
├── requirements.txt
│
├── workloads/                      # 9 workload generators (run on GPU server)
│   ├── workload_resnet.py          # CPU ResNet-18 inference
│   ├── workload_resnet_gpu.py      # GPU ResNet-18 inference
│   ├── workload_gemm.py            # GPU GEMM (upper-bound anchor)
│   ├── workload_yolo.py            # GPU YOLOv8 inference
│   ├── workload_llm.py             # CPU LLM text generation (GPT-2)
│   ├── workload_training.py        # GPU ResNet-18 CIFAR-10 fine-tuning
│   ├── workload_ffmpeg.py          # CPU ffmpeg x264 encoding
│   ├── workload_gpu_llm.py         # GPU LLM OPT-1.3B inference (fp16)
│   └── server.js                   # NodeJS Express server (lower-bound anchor)
│
├── measurement/                    # Power/resource measurement (run on server/raspi)
│   ├── concurrent_power.py         # System + per-PID metrics (RAPL, nvidia-smi, /proc)
│   ├── component_power.py          # System-level power measurement
│   └── wall_power.py               # Raspberry Pi serial power meter reader
│
├── experiments/                    # Experiment automation (run on GPU server with sudo)
│   ├── run_cgroup_experiments.py   # Concurrent: 10 combos x 4 ratios = 40 experiments
│   ├── run_solo_experiments.py     # Solo baseline: 10 workloads
│   └── run_perf_stat.py            # perf stat: IPC/LLC for 9 workloads
│
└── analysis/                       # Data processing (run on any machine with Python)
    ├── merge_data.py               # Merge component CSVs with wall power → data_merged/
    ├── plot_report.py              # Generate 14 figures → graphs/
    └── export_gsheet.py            # Export 7 summary CSVs → project root
```

## Experiment Design

### 9 Workloads

| Category | Workload | Role | CPU% | GPU SM% | IPC |
|:--------:|----------|------|:----:|:-------:|:---:|
| GPU-dominant | GEMM | Upper-bound anchor | ~5% | 95% | 2.29 |
| GPU-dominant | ResNet-GPU | GPU inference | ~5% | 83% | 1.01 |
| GPU-dominant | YOLO | Lightweight GPU | ~6% | 15% | 1.39 |
| GPU-dominant | Training | Fine-tuning (backward) | ~8% | 89% | 1.68 |
| GPU-dominant | GPU-LLM | GPU memory-heavy | ~3% | 25% | 0.66 |
| CPU-dominant | ResNet-CPU | Compute-bound anchor | ~46% | - | 1.75 |
| CPU-dominant | ffmpeg | Compute-bound (realistic) | ~34% | - | 1.32 |
| CPU-dominant | LLM | Memory-bound | ~45% | - | 0.48 |
| Lightweight | NodeJS | Lower-bound anchor | ~5% | - | 1.24 |

### 10 Workload Combinations

| Combo | VM A (GPU0) | VM B (GPU1/CPU) | Tests |
|:-----:|-------------|-----------------|-------|
| A | ResNet-CPU | GEMM | CPU vs GPU orthogonality |
| B | ResNet-CPU | LLM | Compute-bound vs memory-bound |
| C | GEMM | YOLO | GPU vs GPU (passthrough) |
| D | LLM | NodeJS | Memory-bound vs lightweight |
| E | ResNet-GPU | ResNet-CPU | Same model, different deployment |
| F | ResNet-GPU | LLM | GPU-dominant vs CPU-dominant |
| G | ffmpeg | LLM | CPU compute vs memory (IPC replication) |
| H | Training | ffmpeg | GPU training + CPU contention |
| I | GPU-LLM | GEMM | GPU memory-heavy vs SM-heavy |
| J | Training | NodeJS | AI training vs non-AI |

### 4 Allocation Ratios (total 20 logical cores)

| Ratio | VM A cores | VM B cores | VM A mem | VM B mem |
|:-----:|:---------:|:---------:|:-------:|:-------:|
| 1:1 | 10 | 10 | 14 GB | 14 GB |
| 2:1 | 14 | 6 | 20 GB | 8 GB |
| 3:1 | 15 | 5 | 24 GB | 4 GB |
| 5:1 | 17 | 3 | 24 GB | 4 GB |

**Total: 10 solo baselines + 40 concurrent experiments = 50 experiments**

## Hardware

| Component | Specification |
|-----------|--------------|
| CPU | Intel Xeon E5-2630 v4 (10C x 2HT, 1 socket) @ 2.20 GHz |
| GPU | NVIDIA Titan V (GV100) x 2 |
| RAM | DDR4 2133 MHz, 32 GiB (1 DIMM of 8 slots) |
| Storage | Samsung 860 1 TB SSD (OS) + HGST 4 TB HDD (data) |
| Motherboard | ASUS X99-E WS (Intel C610/X99 chipset) |

### Measurement Infrastructure

```
[Power Meter] --USB serial--> [Raspberry Pi] --> wall_power.py (wall power, W)

[GPU Server]
  ├── RAPL (/sys/class/powercap/)       --> CPU package + DRAM power (W)
  ├── nvidia-smi                        --> per-GPU power (W) + utilization (%)
  ├── /proc/stat, /proc/[pid]/stat      --> CPU utilization (%)
  └── concurrent_power.py               --> unified per-PID CSV
```

### System Configuration

Applied via systemd service before every experiment:

| Setting | Value |
|---------|-------|
| CPU governor | `ondemand` |
| Intel Turbo Boost | **OFF** (`no_turbo=1`) |
| GPU persistence mode | **ON** |
| I/O scheduler | `mq-deadline` |

## Experiment Protocol

Each experiment follows this protocol:

```
  30s idle      120s workload      30s cooldown
  |------------|---------------------|------------|
  ^            ^                     ^            ^
  start        PHASE_START(30s)      PHASE_END(150s)  end

  - 1-second sampling interval
  - drop_caches=3 before each experiment
  - cgroup v2: cpuset (core isolation) + memory.max (cap)
  - GPU assigned via --device cuda:N (not CUDA_VISIBLE_DEVICES)
```

## Analysis Outputs

After running `merge_data.py`, `plot_report.py`, and `export_gsheet.py`:

| Output | Description |
|--------|-------------|
| `data_merged/summary.csv` | 50-row table: per-experiment averages of wall power, RAPL, GPU, DRAM, per-PID CPU%, SM%, RSS |
| `graphs/*.png` (14 files) | Solo energy profile, CPU scaling, GPU independence, cross-combo verification, IPC comparison, anchor-spectrum, etc. |
| `gsheet_*.csv` (7 files) | Formatted CSVs for spreadsheet review: system power, resource usage, verification, scaling, cross-combo, IPC/cache, anchor-spectrum |

## Citation

If you use this code, please cite:

```bibtex
@article{kim2026energy,
  title={Energy Behavior of AI Workloads under Resource Partitioning in Cloud Systems},
  author={Kim, Jiyun and Shin, Woorim and Kang, Siyeon and Cho, Kyungwoon and Bahn, Hyokyung},
  year={2026}
}
```

## License

This project is for academic research purposes.
