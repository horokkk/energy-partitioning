#!/usr/bin/env python3
"""merge_data.py — capstone2 wall power matching + summary generation

Data source mapping (0223/0224/0225 Phase 1+2):
  Phase 1:
    A,B,F (0223) → wall_cgroup_0223.csv
    C (0224) → wall_cgroup_0224_C.csv
    E,F (0224) → wall_cgroup_0224_GI.csv
  Phase 2:
    G,H,I,J (0225) → wall_conc_0225.csv
  Solo:
    Phase 1 (0224) → wall_solo_0224.csv
    Phase 2 (0225) → wall_solo_0225.csv

Ratios: 4 (1:1, 2:1, 3:1, 5:1)

Output:
  data_merged/conc_cg-{x}{n}.csv  — merged CSV with wall_W column added
  data_merged/solo_s-*.csv        — solo merged CSV with wall_W column added
  data_merged/summary.csv         — 40 concurrent + 10 solo workload phase averages
"""

import pandas as pd
import numpy as np
import os
import re
from pathlib import Path

BASE = Path(__file__).parent.parent
DATA = BASE / "data"
MERGED = BASE / "data_merged"

PHASE_START, PHASE_END = 30, 150  # workload steady-state interval (seconds)

# ── combo definitions ──
COMBOS = {
    # Phase 1
    "A": {"vm_a": "ResNet-CPU", "vm_b": "GEMM",       "type": "CPU AI vs GPU compute"},
    "B": {"vm_a": "ResNet-CPU", "vm_b": "LLM",        "type": "Saturating vs Linear"},
    "C": {"vm_a": "GEMM",      "vm_b": "YOLO",        "type": "GPU vs GPU"},
    "D": {"vm_a": "LLM",       "vm_b": "NodeJS",      "type": "CPU AI vs Non-AI"},
    "E": {"vm_a": "ResNet-GPU", "vm_b": "ResNet-CPU",  "type": "Same model GPU vs CPU"},
    "F": {"vm_a": "ResNet-GPU", "vm_b": "LLM",         "type": "GPU AI vs CPU AI"},
    # Phase 2
    "G": {"vm_a": "ffmpeg",    "vm_b": "LLM",         "type": "CPU compute vs memory"},
    "H": {"vm_a": "Training",  "vm_b": "ffmpeg",      "type": "GPU Training vs CPU compute"},
    "I": {"vm_a": "GPU-LLM",   "vm_b": "GEMM",        "type": "GPU memory-heavy vs SM-heavy"},
    "J": {"vm_a": "Training",  "vm_b": "NodeJS",      "type": "AI Training vs Non-AI"},
}

# OLD 4 ratios
RATIOS = {1: "1:1", 2: "2:1", 3: "3:1", 4: "5:1"}
RATIO_CORES = {
    1: (10, 10, "14G", "14G"),
    2: (14, 6,  "20G", "8G"),
    3: (15, 5,  "24G", "4G"),
    4: (17, 3,  "24G", "4G"),
}

# ── file mapping (OLD: 0223/0224/0225) ──
FILE_MAP = {
    "A": ("0223", "wall_cgroup_0223.csv"),
    "B": ("0223", "wall_cgroup_0223.csv"),
    "C": ("0224", "wall_cgroup_0224_C.csv"),
    "D": ("0223", "wall_cgroup_0223.csv"),
    "E": ("0224", "wall_cgroup_0224_GI.csv"),
    "F": ("0224", "wall_cgroup_0224_GI.csv"),
    "G": ("0225", "wall_conc_0225.csv"),
    "H": ("0225", "wall_conc_0225.csv"),
    "I": ("0225", "wall_conc_0225.csv"),
    "J": ("0225", "wall_conc_0225.csv"),
}


def load_wall(wall_fname):
    """Load wall CSV"""
    fpath = DATA / wall_fname
    if not fpath.exists():
        print(f"  WARNING: {wall_fname} not found")
        return None
    df = pd.read_csv(fpath)
    df["timestamp"] = df["timestamp"].astype(float)
    print(f"  wall loaded: {wall_fname} ({len(df)} rows)")
    return df


def merge_wall(comp_df, wall_df):
    """Add wall_W column to component CSV (nearest timestamp, tolerance 2s)"""
    comp = comp_df.copy()
    comp["timestamp"] = comp["timestamp"].astype(float)
    wall = wall_df[["timestamp", "wall_W"]].copy()
    wall["timestamp"] = wall["timestamp"].astype(float)
    comp = comp.sort_values("timestamp")
    wall = wall.sort_values("timestamp")
    merged = pd.merge_asof(comp, wall, on="timestamp", direction="nearest", tolerance=2.0)
    matched = merged["wall_W"].notna().sum()
    total = len(merged)
    print(f"    wall matched {matched}/{total} rows")
    return merged


def phase_avg(df, col):
    """Workload phase (30~150s) average"""
    phase = df[(df["elapsed_s"] >= PHASE_START) & (df["elapsed_s"] <= PHASE_END)]
    if col not in phase.columns:
        return np.nan
    vals = phase[col].dropna()
    if col.startswith("rapl"):
        vals = vals[(vals >= 0) & (vals < 500)]
    if "cpu_pct" in col:
        vals = vals[(vals >= 0) & (vals < 200)]
    return vals.mean() if len(vals) > 0 else np.nan


def main():
    MERGED.mkdir(exist_ok=True)

    # wall CSV cache (avoid redundant loading of same file)
    wall_cache = {}

    summary_rows = []

    print("=" * 70)
    print("capstone2 data merge (0225 version — 4 ratios, 0223/0224/0225 dates)")
    print("=" * 70)

    for combo_letter, combo_info in COMBOS.items():
        date_suffix, wall_fname = FILE_MAP[combo_letter]

        # load wall (cached)
        if wall_fname not in wall_cache:
            wall_cache[wall_fname] = load_wall(wall_fname)
        wall_df = wall_cache[wall_fname]

        print(f"\n--- Combo {combo_letter}: {combo_info['vm_a']} + {combo_info['vm_b']} ---")

        for ratio_idx in range(1, 5):  # 4 ratios
            fname = f"conc_cg-{combo_letter.lower()}{ratio_idx}_{date_suffix}.csv"
            fpath = DATA / fname

            if not fpath.exists():
                print(f"  {fname}: file not found")
                continue

            print(f"  {fname}")
            comp_df = pd.read_csv(fpath)

            # wall matching
            if wall_df is not None:
                merged = merge_wall(comp_df, wall_df)
            else:
                merged = comp_df.copy()
                merged["wall_W"] = np.nan

            # save
            out_fname = f"conc_cg-{combo_letter.lower()}{ratio_idx}.csv"
            merged.to_csv(MERGED / out_fname, index=False)

            # summary statistics (workload phase)
            a_cores, b_cores, a_mem, b_mem = RATIO_CORES[ratio_idx]
            row = {
                "exp_id": f"CG-{combo_letter}{ratio_idx}",
                "combo": combo_letter,
                "ratio": RATIOS[ratio_idx],
                "vm_a": combo_info["vm_a"],
                "vm_b": combo_info["vm_b"],
                "type": combo_info["type"],
                "vm_a_cores": a_cores,
                "vm_b_cores": b_cores,
                "vm_a_mem": a_mem,
                "vm_b_mem": b_mem,
                "wall_W": phase_avg(merged, "wall_W"),
                "rapl_W": phase_avg(merged, "rapl_package-0_W"),
                "gpu0_W": phase_avg(merged, "gpu0_W"),
                "gpu1_W": phase_avg(merged, "gpu1_W"),
                "dram_W": phase_avg(merged, "rapl_dram_W"),
                "sys_cpu_pct": phase_avg(merged, "cpu_util_pct"),
                "a_cpu_pct": phase_avg(merged, "vm_a_cpu_pct"),
                "b_cpu_pct": phase_avg(merged, "vm_b_cpu_pct"),
                "a_sm_pct": phase_avg(merged, "vm_a_gpu_sm_pct"),
                "b_sm_pct": phase_avg(merged, "vm_b_gpu_sm_pct"),
                "a_fb_mib": phase_avg(merged, "vm_a_gpu_mem_mib"),
                "b_fb_mib": phase_avg(merged, "vm_b_gpu_mem_mib"),
                "a_rss_mib": phase_avg(merged, "vm_a_rss_mib"),
                "b_rss_mib": phase_avg(merged, "vm_b_rss_mib"),
                "a_io_r_mbs": phase_avg(merged, "vm_a_io_read_mbs"),
                "a_io_w_mbs": phase_avg(merged, "vm_a_io_write_mbs"),
                "b_io_r_mbs": phase_avg(merged, "vm_b_io_read_mbs"),
                "b_io_w_mbs": phase_avg(merged, "vm_b_io_write_mbs"),
            }
            # Others = Wall - RAPL - GPU0 - GPU1 - DRAM
            if not np.isnan(row["wall_W"]):
                row["others_W"] = (row["wall_W"] - row["rapl_W"]
                                   - row["gpu0_W"] - row["gpu1_W"] - row["dram_W"])
            else:
                row["others_W"] = np.nan

            summary_rows.append(row)

    # ── Solo baseline merge ──
    print(f"\n{'=' * 70}")
    print("Solo baseline merge")
    print(f"{'=' * 70}")

    SOLO_FILES = [
        # Phase 1 (0224) → wall_solo_0224.csv
        ("idle",       "solo_s-idle-r1_0224.csv",        "wall_solo_0224.csv"),
        ("ResNet-GPU", "solo_s-resnet_gpu-r1_0224.csv",  "wall_solo_0224.csv"),
        ("GEMM",       "solo_s-gemm-r1_0224.csv",        "wall_solo_0224.csv"),
        ("YOLO",       "solo_s-yolo-r1_0224.csv",        "wall_solo_0224.csv"),
        ("ResNet-CPU", "solo_s-resnet-r1_0224.csv",      "wall_solo_0224.csv"),
        ("LLM",        "solo_s-llm-r1_0224.csv",         "wall_solo_0224.csv"),
        ("NodeJS",     "solo_s-nodejs-r1_0224.csv",       "wall_solo_0224.csv"),
        # Phase 2 (0225) → wall_solo_0225.csv
        ("Training",   "solo_s-training-r1_0225.csv",    "wall_solo_0225.csv"),
        ("ffmpeg",     "solo_s-ffmpeg-r1_0225.csv",      "wall_solo_0225.csv"),
        ("GPU-LLM",    "solo_s-gpu_llm-r1_0225.csv",    "wall_solo_0225.csv"),
    ]

    solo_rows = []
    for name, fname, wall_fname in SOLO_FILES:
        fpath = DATA / fname
        if not fpath.exists():
            print(f"  {fname}: file not found")
            continue

        print(f"  {fname}")
        comp_df = pd.read_csv(fpath)

        # load wall (cached)
        if wall_fname not in wall_cache:
            wall_cache[wall_fname] = load_wall(wall_fname)
        solo_wall_df = wall_cache[wall_fname]

        if solo_wall_df is not None:
            merged = merge_wall(comp_df, solo_wall_df)
        else:
            merged = comp_df.copy()
            merged["wall_W"] = np.nan

        # save: remove date suffix
        out_fname = re.sub(r'_\d{4}\.csv$', '.csv', fname)
        merged.to_csv(MERGED / out_fname, index=False)

        # summary statistics
        row = {
            "exp_id": f"S-{name}",
            "combo": "-",
            "ratio": "-",
            "vm_a": name,
            "vm_b": "-",
            "type": "solo" if name != "idle" else "idle",
            "vm_a_cores": 10,
            "vm_b_cores": "-",
            "vm_a_mem": "14G",
            "vm_b_mem": "-",
            "wall_W": phase_avg(merged, "wall_W"),
            "rapl_W": phase_avg(merged, "rapl_package-0_W"),
            "gpu0_W": phase_avg(merged, "gpu0_W"),
            "gpu1_W": phase_avg(merged, "gpu1_W"),
            "dram_W": phase_avg(merged, "rapl_dram_W"),
            "sys_cpu_pct": phase_avg(merged, "cpu_util_pct"),
            "a_cpu_pct": phase_avg(merged, "vm_a_cpu_pct"),
            "b_cpu_pct": np.nan,
            "a_sm_pct": phase_avg(merged, "vm_a_gpu_sm_pct"),
            "b_sm_pct": np.nan,
            "a_fb_mib": phase_avg(merged, "vm_a_gpu_mem_mib"),
            "b_fb_mib": np.nan,
            "a_rss_mib": phase_avg(merged, "vm_a_rss_mib"),
            "b_rss_mib": np.nan,
            "a_io_r_mbs": phase_avg(merged, "vm_a_io_read_mbs"),
            "a_io_w_mbs": phase_avg(merged, "vm_a_io_write_mbs"),
            "b_io_r_mbs": np.nan,
            "b_io_w_mbs": np.nan,
        }
        if not np.isnan(row["wall_W"]):
            row["others_W"] = (row["wall_W"] - row["rapl_W"]
                               - row["gpu0_W"] - row["gpu1_W"] - row["dram_W"])
        else:
            row["others_W"] = np.nan

        solo_rows.append(row)

    # Save summary CSV (solo + concurrent)
    all_rows = solo_rows + summary_rows
    summary_df = pd.DataFrame(all_rows)
    summary_path = MERGED / "summary.csv"
    summary_df.to_csv(summary_path, index=False, float_format="%.2f")
    print(f"\n{'=' * 70}")
    print(f"Summary saved: {summary_path}")
    print(f"Merged CSVs: {MERGED}/")
    print(f"Total experiments: {len(solo_rows)} solo + {len(summary_rows)} concurrent = {len(all_rows)}")

    # ── Solo summary output ──
    print(f"\n{'=' * 70}")
    print("SOLO SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n{'Exp':<14} {'Wall':>6} {'RAPL':>6} {'GPU0':>6} {'GPU1':>6} {'DRAM':>6} {'Others':>7} {'CPU%':>6} {'SM%':>5} {'FB':>6}")
    print(f"{'---'*14} {'---'*6} {'---'*6} {'---'*6} {'---'*6} {'---'*6} {'---'*7} {'---'*6} {'---'*5} {'---'*6}")

    for r in solo_rows:
        wall = f"{r['wall_W']:.1f}" if not np.isnan(r['wall_W']) else "N/A"
        others = f"{r['others_W']:.1f}" if not np.isnan(r['others_W']) else "N/A"
        cpu_pct = f"{r['a_cpu_pct']:.1f}" if not np.isnan(r['a_cpu_pct']) else "0.0"
        sm_pct = f"{r['a_sm_pct']:.0f}" if not np.isnan(r['a_sm_pct']) else "0"
        fb = f"{r['a_fb_mib']:.0f}M" if not np.isnan(r['a_fb_mib']) else "0M"
        print(f"{r['exp_id']:<14} {wall:>6} {r['rapl_W']:>6.1f} {r['gpu0_W']:>6.1f} "
              f"{r['gpu1_W']:>6.1f} {r['dram_W']:>6.2f} {others:>7} {cpu_pct:>6} {sm_pct:>5} {fb:>6}")

    # ── Concurrent summary output ──
    print(f"\n{'=' * 70}")
    print("CONCURRENT SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n{'Exp':<8} {'vm_a':<12} {'vm_b':<12} {'Ratio':<6} "
          f"{'Wall':>6} {'RAPL':>6} {'GPU0':>6} {'GPU1':>6} {'DRAM':>6} {'Others':>7}")
    print(f"{'---'*8} {'---'*12} {'---'*12} {'---'*6} "
          f"{'---'*6} {'---'*6} {'---'*6} {'---'*6} {'---'*6} {'---'*7}")

    for r in summary_rows:
        wall = f"{r['wall_W']:.1f}" if not np.isnan(r['wall_W']) else "N/A"
        others = f"{r['others_W']:.1f}" if not np.isnan(r['others_W']) else "N/A"
        print(f"{r['exp_id']:<8} {r['vm_a']:<12} {r['vm_b']:<12} {r['ratio']:<6} "
              f"{wall:>6} {r['rapl_W']:>6.1f} {r['gpu0_W']:>6.1f} "
              f"{r['gpu1_W']:>6.1f} {r['dram_W']:>6.2f} {others:>7}")

    # Per-PID summary
    print(f"\n{'Exp':<8} {'a CPU%':>7} {'a SM%':>6} {'a FB':>6} "
          f"{'b CPU%':>7} {'b SM%':>6} {'b FB':>6}")
    print(f"{'---'*8} {'---'*7} {'---'*6} {'---'*6} "
          f"{'---'*7} {'---'*6} {'---'*6}")

    for r in summary_rows:
        print(f"{r['exp_id']:<8} {r['a_cpu_pct']:>7.1f} {r['a_sm_pct']:>6.0f} "
              f"{r['a_fb_mib']:>5.0f}M "
              f"{r['b_cpu_pct']:>7.1f} {r['b_sm_pct']:>6.0f} "
              f"{r['b_fb_mib']:>5.0f}M")


if __name__ == "__main__":
    main()
