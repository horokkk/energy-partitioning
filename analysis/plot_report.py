#!/usr/bin/env python3
"""Capstone2 report graph generation script (14 graphs).

Data source: data_merged/summary.csv (relative to project root)
Output: graphs/

4 ratios (1:1, 2:1, 3:1, 5:1)

Usage:
    cd <project-root>   # directory containing data_merged/ and data/
    python analysis/plot_report.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── paths (relative to project root = parent of analysis/) ──
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY = os.path.join(BASE, 'data_merged', 'summary.csv')
OUT_DIR = os.path.join(BASE, 'graphs')
os.makedirs(OUT_DIR, exist_ok=True)

# ── style ──
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 9,
    'figure.facecolor': 'white',
})

# ── color palette ──
COLORS = {
    'CPU': '#2196D3',
    'GPU0': '#D44336',
    'GPU1': '#FF9800',
    'DRAM': '#4CAD50',
    'Others': '#9E9E9E',
    'idle_base': '#E0E0E0',
}
# v5 two-level classification: GPU-dominant = warm, CPU-dominant = cool
WL_COLORS = {
    # GPU-dominant (warm)
    'GEMM': '#D32D2F',        # deep red
    'ResNet-GPU': '#D44336',  # red
    'YOLO': '#FD7043',        # orange-red
    # GPU-dominant Phase 2
    'Training': '#E91E63',    # pink (GPU mixed - Training)
    'GPU-LLM': '#FD5722',     # deep orange (GPU memory-heavy)
    # CPU-dominant (cool)
    'ResNet-CPU': '#1976D2',  # blue
    'LLM': '#0288D1',         # light blue
    'NodeJS': '#7B1FA2',      # purple
    # CPU-dominant Phase 2
    'ffmpeg': '#3D51B5',      # indigo (CPU compute)
    # misc
    'idle': '#9E9E9E',
}

# Phase 1 workload order: GPU-dominant -> CPU-dominant
WL_ORDER_GPU_CPU = ['GEMM', 'ResNet-GPU', 'YOLO', 'ResNet-CPU', 'LLM', 'NodeJS']

# Phase 1+2 full workload order: GPU-dominant -> CPU-dominant
WL_ORDER_ALL = ['GEMM', 'GPU-LLM', 'Training', 'ResNet-GPU', 'YOLO',
                'ResNet-CPU', 'ffmpeg', 'LLM', 'NodeJS']

PERF_CSV = os.path.join(BASE, 'data', 'perf_stat_summary.csv')

# ── load data ──
def load_all():
    return pd.read_csv(SUMMARY)

def load_solo():
    df = load_all()
    return df[df['combo'] == '-'].copy()

def load_conc():
    df = load_all()
    return df[df['combo'] != '-'].copy()


# =====================================================================
# Graph 1: Solo Energy Profile (Stacked Bar)
# =====================================================================
def plot_solo_energy_profile():
    solo = load_solo()
    idle = solo[solo['vm_a'] == 'idle'].iloc[0]

    idle_cpu = idle['rapl_W']
    idle_gpu0 = idle['gpu0_W']
    idle_gpu1 = idle['gpu1_W']
    idle_dram = idle['dram_W']
    idle_base = idle_cpu + idle_gpu0 + idle_gpu1 + idle_dram

    # v7: idle -> GPU-dominant(Phase1+2) -> CPU-dominant(Phase1+2)
    workloads = ['idle'] + WL_ORDER_ALL

    cpu_delta, gpu0_delta, gpu1_delta, dram_delta = [], [], [], []
    idle_vals = []
    skipped = []

    for wl in workloads:
        rows = solo[solo['vm_a'] == wl]
        if len(rows) == 0:
            skipped.append(wl)
            continue
        row = rows.iloc[0]
        cpu_delta.append(max(0, row['rapl_W'] - idle_cpu))
        gpu0_delta.append(max(0, row['gpu0_W'] - idle_gpu0))
        gpu1_delta.append(max(0, row['gpu1_W'] - idle_gpu1))
        dram_delta.append(max(0, row['dram_W'] - idle_dram))
        idle_vals.append(idle_base)

    actual_wl = [w for w in workloads if w not in skipped]

    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(len(actual_wl))
    w = 0.55

    bottom = np.array(idle_vals, dtype=float)
    ax.bar(x, idle_vals, w, label='Idle Baseline', color=COLORS['idle_base'], edgecolor='white', linewidth=0.5)
    ax.bar(x, cpu_delta, w, bottom=bottom, label='dCPU (RAPL)', color=COLORS['CPU'], edgecolor='white', linewidth=0.5)
    bottom = bottom + np.array(cpu_delta)
    ax.bar(x, gpu0_delta, w, bottom=bottom, label='dGPU0', color=COLORS['GPU0'], edgecolor='white', linewidth=0.5)
    bottom = bottom + np.array(gpu0_delta)
    ax.bar(x, gpu1_delta, w, bottom=bottom, label='dGPU1', color=COLORS['GPU1'], edgecolor='white', linewidth=0.5)
    bottom = bottom + np.array(gpu1_delta)
    ax.bar(x, dram_delta, w, bottom=bottom, label='dDRAM', color=COLORS['DRAM'], edgecolor='white', linewidth=0.5)

    comp_total = np.array(idle_vals) + np.array(cpu_delta) + np.array(gpu0_delta) + np.array(gpu1_delta) + np.array(dram_delta)
    for i, ct in enumerate(comp_total):
        ax.annotate(f'{ct:.0f}W', (x[i], ct + 2), ha='center', va='bottom',
                    fontsize=8, fontweight='bold')

    # GPU-dominant / CPU-dominant separator line
    gpu_count = sum(1 for w in actual_wl if w in ['GEMM', 'GPU-LLM', 'Training', 'ResNet-GPU', 'YOLO'])
    sep_x = gpu_count + 0.5  # after idle(0) + GPU-dominant
    ax.axvline(x=sep_x, color='black', linestyle='--', alpha=0.3, linewidth=1)
    ax.text(sep_x / 2, 370, 'GPU-dominant', ha='center', fontsize=10, fontstyle='italic', color='#D32D2F')
    ax.text((sep_x + len(actual_wl) - 1) / 2, 370, 'CPU-dominant', ha='center', fontsize=10, fontstyle='italic', color='#1976D2')

    ax.set_xticks(x)
    ax.set_xticklabels(actual_wl, fontsize=8, rotation=15, ha='right')
    ax.set_ylabel('Power (W)')
    ax.set_title('Solo Workload Energy Profile (Component Power, 9 Workloads)')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_ylim(0, 400)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'solo_energy_profile.png'))
    plt.close()
    print('[1/14] solo_energy_profile.png')


# =====================================================================
# Graph 2: CPU Scaling Pattern (Line Chart)
# =====================================================================
def plot_cpu_scaling():
    conc = load_conc()

    # Convert cores to int
    for col in ['vm_a_cores', 'vm_b_cores']:
        conc[col] = pd.to_numeric(conc[col], errors='coerce').astype('Int64')

    # LLM: B combo vm_b (3,5,6,10) + F combo vm_a (10,14,15,17) -- full range
    b = conc[conc['combo'] == 'B']
    f = conc[conc['combo'] == 'D']
    llm_low = b[['vm_b_cores', 'b_cpu_pct']].rename(columns={'vm_b_cores': 'cores', 'b_cpu_pct': 'cpu'})
    llm_high = f[['vm_a_cores', 'a_cpu_pct']].rename(columns={'vm_a_cores': 'cores', 'a_cpu_pct': 'cpu'})
    llm_data = pd.concat([llm_low, llm_high]).sort_values('cores')
    llm_data = llm_data.groupby('cores')['cpu'].mean().reset_index()

    # ResNet-CPU: G combo vm_b (3,5,6,10) + B combo vm_a (10,14,15,17) -- full range
    g = conc[conc['combo'] == 'E']
    resnet_low = g[['vm_b_cores', 'b_cpu_pct']].rename(columns={'vm_b_cores': 'cores', 'b_cpu_pct': 'cpu'})
    resnet_high = b[['vm_a_cores', 'a_cpu_pct']].rename(columns={'vm_a_cores': 'cores', 'a_cpu_pct': 'cpu'})
    resnet_data = pd.concat([resnet_low, resnet_high]).sort_values('cores')
    resnet_data = resnet_data.groupby('cores')['cpu'].mean().reset_index()

    # GEMM: A combo vm_b (3,5,6,10) + C combo vm_a (10,14,15,17) -- Flat
    a = conc[conc['combo'] == 'A']
    c = conc[conc['combo'] == 'C']
    gemm_low = a[['vm_b_cores', 'b_cpu_pct']].rename(columns={'vm_b_cores': 'cores', 'b_cpu_pct': 'cpu'})
    gemm_high = c[['vm_a_cores', 'a_cpu_pct']].rename(columns={'vm_a_cores': 'cores', 'a_cpu_pct': 'cpu'})
    gemm_data = pd.concat([gemm_low, gemm_high]).sort_values('cores')
    gemm_data = gemm_data.groupby('cores')['cpu'].mean().reset_index()

    # NodeJS: F combo vm_b (3,5,6,10 cores) -- Flat
    nodejs_data = f[['vm_b_cores', 'b_cpu_pct']].rename(columns={'vm_b_cores': 'cores', 'b_cpu_pct': 'cpu'})
    nodejs_data = nodejs_data.sort_values('cores')

    # ffmpeg: G combo vm_a (10,14,15,17) + H combo vm_b (3,5,6,10) -- Phase 2
    j = conc[conc['combo'] == 'G']
    k = conc[conc['combo'] == 'H']
    ffmpeg_high = j[['vm_a_cores', 'a_cpu_pct']].rename(columns={'vm_a_cores': 'cores', 'a_cpu_pct': 'cpu'})
    ffmpeg_low = k[['vm_b_cores', 'b_cpu_pct']].rename(columns={'vm_b_cores': 'cores', 'b_cpu_pct': 'cpu'})
    ffmpeg_data = pd.concat([ffmpeg_low, ffmpeg_high]).sort_values('cores')
    ffmpeg_data = ffmpeg_data.groupby('cores')['cpu'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(9, 6))

    # v6: GPU-dominant = gray dashed (all Flat, background)
    ax.plot(gemm_data['cores'], gemm_data['cpu'], '^--', color='#BDBDBD',
            linewidth=1.5, markersize=7, label='GEMM (GPU-dominant, Flat)', zorder=3)
    ax.plot(nodejs_data['cores'], nodejs_data['cpu'], 'D--', color='#9E9E9E',
            linewidth=1.5, markersize=7, label='NodeJS (Light)', zorder=3)

    # CPU-dominant = solid, highlighted
    ax.plot(llm_data['cores'], llm_data['cpu'], 'o-', color=WL_COLORS['LLM'],
            linewidth=2.5, markersize=9, label='LLM (Memory-bound)', zorder=5)
    ax.plot(resnet_data['cores'], resnet_data['cpu'], 's-', color=WL_COLORS['ResNet-CPU'],
            linewidth=2.5, markersize=9, label='ResNet-CPU (Compute-bound)', zorder=5)
    ax.plot(ffmpeg_data['cores'], ffmpeg_data['cpu'], 'P-', color=WL_COLORS['ffmpeg'],
            linewidth=2.5, markersize=9, label='ffmpeg (Compute-bound, no saturation)', zorder=5)

    # GPU-dominant Flat zone shade
    ax.axhspan(0, 8, color='#FFEBEE', alpha=0.4, zorder=1)
    ax.text(16.5, 6.5, 'GPU-dominant\n= all Flat', fontsize=8, color='#B71C1C',
            ha='center', fontstyle='italic', alpha=0.7)

    # Saturation zone shade
    ax.axhspan(42, 48, color='#E3D2FD', alpha=0.3, zorder=1)

    ax.set_xlabel('Allocated CPU Cores')
    ax.set_ylabel('CPU Utilization (%)')
    ax.set_title('CPU Scaling: LLM/ResNet-CPU Saturate at ~10 Cores, ffmpeg Keeps Scaling')
    ax.legend(loc='upper left', framealpha=0.9, fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 62)
    ax.set_xlim(2, 18)
    ax.set_xticks([3, 5, 6, 10, 14, 15, 17])

    # Annotations
    ax.annotate('LLM/ResNet-CPU saturate\nat ~45% (~10 cores)',
                xy=(13.5, 45), xytext=(11, 52),
                fontsize=8, color='#333333', fontstyle='italic',
                ha='center',
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))
    ax.annotate('ffmpeg: 56% at 17 cores\n(no saturation)',
                xy=(17, ffmpeg_data[ffmpeg_data['cores']==17]['cpu'].values[0] if 17 in ffmpeg_data['cores'].values else 56),
                xytext=(14.5, 58),
                fontsize=8, color=WL_COLORS['ffmpeg'], fontweight='bold',
                ha='center',
                arrowprops=dict(arrowstyle='->', color=WL_COLORS['ffmpeg'], lw=1.2))

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'cpu_scaling_pattern.png'))
    plt.close()
    print('[2/14] cpu_scaling_pattern.png')


# =====================================================================
# Graph 3: ResNet CPU vs GPU Deployment (Combo E data!)
# =====================================================================
def plot_resnet_cpu_vs_gpu():
    """Combo E: ResNet-GPU(vm_a) vs ResNet-CPU(vm_b) -- same model, different deployment"""
    solo = load_solo()

    # Solo baselines
    s_gpu = solo[solo['vm_a'] == 'ResNet-GPU'].iloc[0]
    s_cpu = solo[solo['vm_a'] == 'ResNet-CPU'].iloc[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left: CPU% comparison
    labels = ['ResNet-GPU\n(Solo)', 'ResNet-CPU\n(Solo)']
    cpu_vals = [s_gpu['a_cpu_pct'], s_cpu['a_cpu_pct']]
    colors = [WL_COLORS['ResNet-GPU'], WL_COLORS['ResNet-CPU']]
    bars1 = ax1.bar(labels, cpu_vals, color=colors, width=0.5, edgecolor='white')
    ax1.set_ylabel('CPU Utilization (%)')
    ax1.set_title('CPU Usage')
    for bar, val in zip(bars1, cpu_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax1.set_ylim(0, 55)
    ax1.grid(axis='y', alpha=0.3)

    # Right: GPU Power comparison
    gpu_vals = [s_gpu['gpu0_W'], s_cpu['gpu0_W']]
    bars2 = ax2.bar(labels, gpu_vals, color=colors, width=0.5, edgecolor='white')
    ax2.set_ylabel('GPU0 Power (W)')
    ax2.set_title('GPU Power')
    for bar, val in zip(bars2, gpu_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val:.1f}W', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 200)
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(y=24.7, color='gray', linestyle='--', alpha=0.5, label='GPU idle (24.7W)')
    ax2.legend(fontsize=8)

    fig.suptitle('ResNet-18: CPU Inference vs GPU Inference', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'resnet_cpu_vs_gpu.png'))
    plt.close()
    print('[3/14] resnet_cpu_vs_gpu.png')


# =====================================================================
# Graph 4: System Power Breakdown (Stacked Bar)
# =====================================================================
def plot_system_power_breakdown():
    solo = load_solo()
    conc = load_conc()
    conc_1_1 = conc[conc['ratio'] == '1:1'].copy()

    labels = []
    cpu_vals, gpu0_vals, gpu1_vals, dram_vals, others_vals = [], [], [], [], []

    # Solo: v7 order (idle -> GPU-dominant(Phase1+2) -> CPU-dominant(Phase1+2))
    solo_order = ['idle'] + WL_ORDER_ALL
    n_solo = 0
    for wl in solo_order:
        rows = solo[solo['vm_a'] == wl]
        if len(rows) == 0:
            continue
        row = rows.iloc[0]
        labels.append(f"S:{row['vm_a']}")
        cpu_vals.append(row['rapl_W'])
        gpu0_vals.append(row['gpu0_W'])
        gpu1_vals.append(row['gpu1_W'])
        dram_vals.append(row['dram_W'])
        others_vals.append(row['others_W'])
        n_solo += 1

    # Concurrent 1:1
    for _, row in conc_1_1.sort_values('combo').iterrows():
        labels.append(f"C:{row['vm_a']}+{row['vm_b']}")
        cpu_vals.append(row['rapl_W'])
        gpu0_vals.append(row['gpu0_W'])
        gpu1_vals.append(row['gpu1_W'])
        dram_vals.append(row['dram_W'])
        others_vals.append(row['others_W'])

    fig, ax = plt.subplots(figsize=(20, 5.5))
    x = np.arange(len(labels))
    w = 0.65

    bottom = np.zeros(len(labels))
    ax.bar(x, cpu_vals, w, bottom=bottom, label='CPU (RAPL)', color=COLORS['CPU'], edgecolor='white', linewidth=0.5)
    bottom += np.array(cpu_vals)
    ax.bar(x, gpu0_vals, w, bottom=bottom, label='GPU0', color=COLORS['GPU0'], edgecolor='white', linewidth=0.5)
    bottom += np.array(gpu0_vals)
    ax.bar(x, gpu1_vals, w, bottom=bottom, label='GPU1', color=COLORS['GPU1'], edgecolor='white', linewidth=0.5)
    bottom += np.array(gpu1_vals)
    ax.bar(x, dram_vals, w, bottom=bottom, label='DRAM', color=COLORS['DRAM'], edgecolor='white', linewidth=0.5)
    bottom += np.array(dram_vals)
    ax.bar(x, others_vals, w, bottom=bottom, label='Others', color=COLORS['Others'], edgecolor='white', linewidth=0.5)

    total = np.array(cpu_vals) + np.array(gpu0_vals) + np.array(gpu1_vals) + np.array(dram_vals) + np.array(others_vals)
    for i, t in enumerate(total):
        ax.text(x[i], t + 2, f'{t:.0f}', ha='center', va='bottom', fontsize=6, fontweight='bold')

    sep_x = n_solo - 0.5
    ax.axvline(x=sep_x, color='black', linestyle='--', alpha=0.3)
    ax.text(sep_x - 0.3, ax.get_ylim()[1]*0.95, 'Solo', ha='right', fontsize=9, fontstyle='italic')
    ax.text(sep_x + 0.3, ax.get_ylim()[1]*0.95, 'Concurrent 1:1', ha='left', fontsize=9, fontstyle='italic')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=6)
    ax.set_ylabel('Power (W)')
    ax.set_title('System Power Breakdown: Solo vs Concurrent (1:1, Phase 1+2)')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'system_power_breakdown.png'))
    plt.close()
    print('[4/14] system_power_breakdown.png')


# =====================================================================
# Graph 5: GPU Power Independence (Line Chart) -- OLD 4-ratio version
# =====================================================================
def plot_gpu_independence():
    conc = load_conc()
    solo = load_solo()

    for col in ['vm_a_cores', 'vm_b_cores']:
        conc[col] = pd.to_numeric(conc[col], errors='coerce').astype('Int64')

    # OLD: 4 ratios
    n_ratios = 4
    ratio_labels = ['1:1\n(10:10)', '2:1\n(14:6)', '3:1\n(15:5)', '5:1\n(17:3)']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── Left: Phase 1 (A, C combos) ──
    c = conc[conc['combo'] == 'C'].sort_values('vm_a_cores')
    a = conc[conc['combo'] == 'A'].sort_values('vm_a_cores')

    gemm_solo = solo[solo['vm_a'] == 'GEMM'].iloc[0]['gpu0_W']
    yolo_solo = solo[solo['vm_a'] == 'YOLO'].iloc[0]['gpu0_W']

    ax1.plot(range(n_ratios), c['gpu0_W'].values, 'o-', color=WL_COLORS['GEMM'], linewidth=2, markersize=8,
             label='GEMM GPU0 (C)')
    ax1.plot(range(n_ratios), c['gpu1_W'].values, 's-', color=WL_COLORS['YOLO'], linewidth=2, markersize=8,
             label='YOLO GPU1 (C)')
    ax1.plot(range(n_ratios), a['gpu1_W'].values, '^-', color='#E91E63', linewidth=2, markersize=8,
             label='GEMM GPU1 (A)')

    ax1.axhline(y=gemm_solo, color=WL_COLORS['GEMM'], linestyle='--', alpha=0.4, linewidth=1)
    ax1.text(n_ratios - 0.9, gemm_solo, f'Solo: {gemm_solo:.0f}W', fontsize=8, color=WL_COLORS['GEMM'], va='center')
    ax1.axhline(y=yolo_solo, color=WL_COLORS['YOLO'], linestyle='--', alpha=0.4, linewidth=1)
    ax1.text(n_ratios - 0.9, yolo_solo, f'Solo: {yolo_solo:.0f}W', fontsize=8, color=WL_COLORS['YOLO'], va='center')

    ax1.set_xticks(range(n_ratios))
    ax1.set_xticklabels(ratio_labels, fontsize=8)
    ax1.set_xlabel('Allocation Ratio (vm_a : vm_b cores)')
    ax1.set_ylabel('GPU Power (W)')
    ax1.set_title('Phase 1: GEMM, YOLO (A/C combos)')
    ax1.legend(framealpha=0.9, fontsize=8)
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 260)

    # ── Right: Phase 2 (I, J, H combos) ──
    l = conc[conc['combo'] == 'I'].sort_values('vm_a_cores')
    m = conc[conc['combo'] == 'J'].sort_values('vm_a_cores')
    k = conc[conc['combo'] == 'H'].sort_values('vm_a_cores')

    gpu_llm_solo = solo[solo['vm_a'] == 'GPU-LLM'].iloc[0]['gpu0_W']
    training_solo = solo[solo['vm_a'] == 'Training'].iloc[0]['gpu0_W']

    ax2.plot(range(n_ratios), l['gpu0_W'].values, 'o-', color=WL_COLORS['GPU-LLM'], linewidth=2, markersize=8,
             label='GPU-LLM GPU0 (I)')
    ax2.plot(range(n_ratios), l['gpu1_W'].values, 's-', color=WL_COLORS['GEMM'], linewidth=2, markersize=8,
             label='GEMM GPU1 (I)')
    ax2.plot(range(n_ratios), m['gpu0_W'].values, '^-', color=WL_COLORS['Training'], linewidth=2, markersize=8,
             label='Training GPU0 (J)')
    ax2.plot(range(n_ratios), k['gpu0_W'].values, 'v--', color=WL_COLORS['Training'], linewidth=2, markersize=8,
             alpha=0.6, label='Training GPU0 (H)')

    ax2.axhline(y=training_solo, color=WL_COLORS['Training'], linestyle='--', alpha=0.4, linewidth=1)
    ax2.text(n_ratios - 0.9, training_solo, f'Solo: {training_solo:.0f}W', fontsize=8, color=WL_COLORS['Training'], va='center')
    ax2.axhline(y=gpu_llm_solo, color=WL_COLORS['GPU-LLM'], linestyle='--', alpha=0.4, linewidth=1)
    ax2.text(n_ratios - 0.9, gpu_llm_solo, f'Solo: {gpu_llm_solo:.0f}W', fontsize=8, color=WL_COLORS['GPU-LLM'], va='center')

    # H1 annotation (1:1 = ratio_idx 1, but Training gets 10 cores at 1:1)
    # In 0225, H1 = 1:1 (10:10), H4 = 5:1 (17:3) — H combo vm_a=Training
    # At 5:1 Training gets 17 cores, at 1:1 gets 10. Check if any ratio starves Training
    # Actually in the old 4-ratio setup: ratio 1=1:1(10:10), ratio 4=5:1(17:3)
    # H combo: vm_a=Training, vm_b=ffmpeg. At ratio 4 (5:1), Training=17, ffmpeg=3
    # The H1 boundary was at the most extreme ratio where Training had fewest cores
    # In 0225 with 4 ratios, H1 is at 1:1 (Training=10 cores) which should be fine
    # Let's annotate H4 if Training gets starved at the extreme
    if len(k) > 0:
        # Check H1 (1:1, index 0) — Training has 10 cores, should be OK
        # Check if any K value is notably lower
        k_min_idx = k['gpu0_W'].values.argmin()
        k_min_val = k['gpu0_W'].values[k_min_idx]
        if k_min_val < training_solo * 0.85:  # >15% drop
            ax2.annotate(f'CPU starvation\n({RATIO_CORES_LABELS[k_min_idx]})',
                         xy=(k_min_idx, k_min_val),
                         xytext=(k_min_idx + 1.2, k_min_val + 35),
                         fontsize=7, color='red', fontweight='bold',
                         arrowprops=dict(arrowstyle='->', color='red', lw=1.2))

    ax2.set_xticks(range(n_ratios))
    ax2.set_xticklabels(ratio_labels, fontsize=8)
    ax2.set_xlabel('Allocation Ratio (vm_a : vm_b cores)')
    ax2.set_ylabel('GPU Power (W)')
    ax2.set_title('Phase 2: GPU-LLM, Training, GEMM (I/J/H combos)')
    ax2.legend(framealpha=0.9, fontsize=8)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 260)

    fig.suptitle('GPU Power Independence from CPU/Memory Allocation (Phase 1 + 2)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'gpu_independence.png'))
    plt.close()
    print('[5/14] gpu_independence.png')


# Helper for K boundary annotation
RATIO_CORES_LABELS = {0: '10 cores', 1: '14 cores', 2: '15 cores', 3: '17 cores'}


# =====================================================================
# Graph 6: Solo vs Concurrent Verification
# =====================================================================
def plot_solo_vs_concurrent():
    """Phase 1+2 per-combo Solo vs Concurrent CPU% + GPU Power comparison."""
    solo = load_solo()
    conc = load_conc()
    conc_1_1 = conc[conc['ratio'] == '1:1'].sort_values('combo')

    # Phase 1 + Phase 2 combos
    combos_order = [c for c in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
                    if c in conc_1_1['combo'].values]
    combos = {
        'A': ('ResNet-CPU', 'GEMM'), 'B': ('ResNet-CPU', 'LLM'),
        'C': ('GEMM', 'YOLO'), 'D': ('LLM', 'NodeJS'),
        'E': ('ResNet-GPU', 'ResNet-CPU'), 'F': ('ResNet-GPU', 'LLM'),
        'G': ('ffmpeg', 'LLM'), 'H': ('Training', 'ffmpeg'),
        'I': ('GPU-LLM', 'GEMM'), 'J': ('Training', 'NodeJS'),
    }
    gpu_map = {
        'A': [('b', 'gpu1_W', 'GEMM')],
        'C': [('a', 'gpu0_W', 'GEMM'), ('b', 'gpu1_W', 'YOLO')],
        'E': [('a', 'gpu0_W', 'ResNet-GPU')],
        'F': [('a', 'gpu0_W', 'ResNet-GPU')],
        'H': [('a', 'gpu0_W', 'Training')],
        'I': [('a', 'gpu0_W', 'GPU-LLM'), ('b', 'gpu1_W', 'GEMM')],
        'J': [('a', 'gpu0_W', 'Training')],
    }

    n_combos = len(combos_order)
    ncols = min(4, n_combos)
    nrows = (n_combos + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 3.5*nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    for idx, combo_id in enumerate(combos_order):
        ax = axes[idx // ncols, idx % ncols]
        vm_a, vm_b = combos[combo_id]
        c_rows = conc_1_1[conc_1_1['combo'] == combo_id]
        if len(c_rows) == 0:
            ax.set_visible(False)
            continue
        c_row = c_rows.iloc[0]

        s_a = solo[solo['vm_a'] == vm_a]
        s_b = solo[solo['vm_a'] == vm_b]
        if len(s_a) == 0 or len(s_b) == 0:
            ax.set_visible(False)
            continue

        s_a_cpu = s_a.iloc[0]['a_cpu_pct']
        s_b_cpu = s_b.iloc[0]['a_cpu_pct']

        w = 0.8
        c_a = WL_COLORS.get(vm_a, '#888')
        c_b = WL_COLORS.get(vm_b, '#888')

        ax.bar([0], [s_a_cpu], w, color=c_a, alpha=0.45, edgecolor='white')
        ax.bar([2.5], [s_b_cpu], w, color=c_b, alpha=0.45, edgecolor='white')
        ax.bar([1], [c_row['a_cpu_pct']], w, color=c_a, alpha=1.0, edgecolor='white')
        ax.bar([3.5], [c_row['b_cpu_pct']], w, color=c_b, alpha=1.0, edgecolor='white')

        for pos, val in [(0, s_a_cpu), (1, c_row['a_cpu_pct']), (2.5, s_b_cpu), (3.5, c_row['b_cpu_pct'])]:
            if val > 2:
                ax.text(pos, val + 1, f'{val:.1f}', ha='center', fontsize=6, fontweight='bold')

        ax.set_xticks([0.5, 3.0])
        ax.set_xticklabels([vm_a, vm_b], fontsize=7)
        ax.set_title(f'{combo_id}: {vm_a}+{vm_b}', fontsize=9, fontweight='bold')
        ax.set_ylabel('CPU%' if idx % ncols == 0 else '')
        ax.grid(axis='y', alpha=0.3)
        max_val = max(s_a_cpu, s_b_cpu, c_row['a_cpu_pct'], c_row['b_cpu_pct'])
        ax.set_ylim(0, max(max_val * 1.3, 10))

        gpu_entries = gpu_map.get(combo_id, [])
        for vm_side, gpu_col, wl_name in gpu_entries:
            s_wl = solo[solo['vm_a'] == wl_name]
            if len(s_wl) == 0:
                continue
            s_gpu = s_wl.iloc[0]['gpu0_W']
            c_gpu = c_row[gpu_col]
            delta_pct = (c_gpu - s_gpu) / s_gpu * 100
            sign = '+' if delta_pct >= 0 else ''
            ax.text(0.97, 0.97, f'GPU:{s_gpu:.0f}->{c_gpu:.0f}W({sign}{delta_pct:.1f}%)',
                    transform=ax.transAxes, ha='right', va='top', fontsize=6,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.8))

        if idx == 0:
            from matplotlib.patches import Patch
            ax.legend(handles=[
                Patch(facecolor='gray', alpha=0.45, label='Solo'),
                Patch(facecolor='gray', alpha=1.0, label='Concurrent'),
            ], fontsize=6, loc='upper left')

    # hide empty subplots
    for idx in range(n_combos, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle('Solo vs Concurrent Verification (CPU% + GPU Power, 1:1, All Combos)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'solo_vs_concurrent.png'))
    plt.close()
    print('[6/14] solo_vs_concurrent.png')


# =====================================================================
# Graph 7: Energy Asymmetry (Horizontal Bar)
# =====================================================================
def plot_energy_asymmetry():
    solo = load_solo()
    idle = solo[solo['vm_a'] == 'idle'].iloc[0]

    idle_total = idle['rapl_W'] + idle['gpu0_W'] + idle['gpu1_W'] + idle['dram_W']

    # v7: all 9 workloads (Phase 1+2)
    workloads = [w for w in WL_ORDER_ALL if len(solo[solo['vm_a'] == w]) > 0]
    active_energy = []

    for wl in workloads:
        row = solo[solo['vm_a'] == wl].iloc[0]
        wl_total = row['rapl_W'] + row['gpu0_W'] + row['gpu1_W'] + row['dram_W']
        active_energy.append(wl_total - idle_total)

    min_e = min(e for e in active_energy if e > 0)

    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(workloads))
    colors = [WL_COLORS.get(w, '#888') for w in workloads]

    bars = ax.barh(y, active_energy, color=colors, height=0.5, edgecolor='white')

    for i, (bar, val) in enumerate(zip(bars, active_energy)):
        ratio = val / min_e if min_e > 0 else 0
        label = f'  {val:.1f}W'
        if i == 0:
            label += f'  ({ratio:.0f}:1 vs {workloads[-1]})'
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                label, va='center', fontsize=9, fontweight='bold')

    # GPU-dominant / CPU-dominant separator line
    gpu_wl = ['GEMM', 'GPU-LLM', 'Training', 'ResNet-GPU', 'YOLO']
    gpu_count = sum(1 for w in workloads if w in gpu_wl)
    sep_y = gpu_count - 0.5
    ax.axhline(y=sep_y, color='black', linestyle='--', alpha=0.3, linewidth=1)
    ax.text(active_energy[0] * 1.3, gpu_count / 2 - 0.5, 'GPU-\ndominant', ha='center', fontsize=9,
            color='#D32D2F', fontstyle='italic', fontweight='bold')
    ax.text(active_energy[0] * 1.3, (gpu_count + len(workloads) - 1) / 2, 'CPU-\ndominant', ha='center', fontsize=9,
            color='#1976D2', fontstyle='italic', fontweight='bold')

    ax.set_yticks(y)
    ax.set_yticklabels(workloads, fontsize=10)
    ax.set_xlabel('Active Energy (W) = Component Total - idle')
    ax.set_title('Energy Asymmetry: Active Power by Workload (9 Workloads)')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, active_energy[0] * 1.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'energy_asymmetry.png'))
    plt.close()
    print('[7/14] energy_asymmetry.png')


# =====================================================================
# Graph 8: GPU-dominant vs CPU-dominant Contrast
# =====================================================================
def plot_gpu_vs_cpu_contrast():
    """v5: Level 1 classification contrast graph. Energy composition + CPU utilization."""
    solo = load_solo()
    idle = solo[solo['vm_a'] == 'idle'].iloc[0]

    idle_cpu = idle['rapl_W']
    idle_gpu0 = idle['gpu0_W']
    idle_dram = idle['dram_W']

    workloads = [w for w in WL_ORDER_ALL if len(solo[solo['vm_a'] == w]) > 0]
    cpu_active, gpu_active, dram_active = [], [], []

    for wl in workloads:
        row = solo[solo['vm_a'] == wl].iloc[0]
        cpu_active.append(max(0, row['rapl_W'] - idle_cpu))
        gpu_active.append(max(0, row['gpu0_W'] - idle_gpu0))
        dram_active.append(max(0, row['dram_W'] - idle_dram))

    cpu_pct = []
    for wl in workloads:
        row = solo[solo['vm_a'] == wl].iloc[0]
        cpu_pct.append(row['a_cpu_pct'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5),
                                    gridspec_kw={'width_ratios': [1.3, 1]})

    x = np.arange(len(workloads))
    w = 0.55

    # Left: Active Energy Composition (stacked: GPU + CPU + DRAM)
    ax1.bar(x, gpu_active, w, label='GPU (active)', color=COLORS['GPU0'], edgecolor='white', linewidth=0.5)
    ax1.bar(x, cpu_active, w, bottom=gpu_active, label='CPU (active)', color=COLORS['CPU'], edgecolor='white', linewidth=0.5)
    bottom2 = np.array(gpu_active) + np.array(cpu_active)
    ax1.bar(x, dram_active, w, bottom=bottom2, label='DRAM (active)', color=COLORS['DRAM'], edgecolor='white', linewidth=0.5)

    totals = np.array(gpu_active) + np.array(cpu_active) + np.array(dram_active)
    for i, t in enumerate(totals):
        ax1.text(x[i], t + 2, f'{t:.0f}W', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Separator: GPU-dominant count
    gpu_wl = ['GEMM', 'GPU-LLM', 'Training', 'ResNet-GPU', 'YOLO']
    gpu_count = sum(1 for w in workloads if w in gpu_wl)
    sep = gpu_count - 0.5
    ax1.axvline(x=sep, color='black', linestyle='--', alpha=0.3, linewidth=1)
    ax1.text(sep / 2, 228, 'GPU-dominant', ha='center', fontsize=10, color='#D32D2F', fontweight='bold')
    ax1.text((sep + len(workloads) - 1) / 2, 228, 'CPU-dominant', ha='center', fontsize=10, color='#1976D2', fontweight='bold')

    ax1.set_xticks(x)
    ax1.set_xticklabels(workloads, fontsize=8, rotation=20, ha='right')
    ax1.set_ylabel('Active Power (W)')
    ax1.set_title('Active Energy Composition (excl. idle, 9 Workloads)')
    ax1.legend(loc='center right', framealpha=0.9, fontsize=9)
    ax1.set_ylim(0, 240)
    ax1.grid(axis='y', alpha=0.3)

    # Right: CPU Utilization
    bar_colors = [WL_COLORS.get(wl, '#888') for wl in workloads]
    ax2.bar(x, cpu_pct, w, color=bar_colors, edgecolor='white', linewidth=0.5)

    for i, val in enumerate(cpu_pct):
        ax2.text(x[i], val + 0.8, f'{val:.0f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax2.axvline(x=sep, color='black', linestyle='--', alpha=0.3, linewidth=1)
    ax2.text(sep / 2, 52, 'all ~5%\n(Flat)', ha='center', fontsize=9, color='#D32D2F', fontstyle='italic')
    ax2.text((sep + len(workloads) - 1) / 2, 52, 'diverge\n(Level 2)', ha='center', fontsize=9, color='#1976D2', fontstyle='italic')

    ax2.set_xticks(x)
    ax2.set_xticklabels(workloads, fontsize=8, rotation=20, ha='right')
    ax2.set_ylabel('CPU Utilization (%)')
    ax2.set_title('CPU Usage (Solo, 10 cores)')
    ax2.set_ylim(0, 60)
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle('Level 1: GPU-dominant vs CPU-dominant (9 Workloads)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'gpu_vs_cpu_contrast.png'))
    plt.close()
    print('[8/14] gpu_vs_cpu_contrast.png')


# =====================================================================
# Graph 9: Cross-Combo Verification (all ratios)
# =====================================================================
def plot_cross_combo_verification():
    """Same workload, same cores, different partners -> profile preserved?"""
    df = load_all()
    solo = df[df['combo'] == '-'].copy()
    conc = df[df['combo'] != '-'].copy()

    for col in ['vm_a_cores', 'vm_b_cores']:
        conc[col] = pd.to_numeric(conc[col], errors='coerce')

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    combo_style = {
        'A': ('o', '#E91E63'), 'B': ('s', '#9C27B0'), 'C': ('^', '#FD5722'),
        'D': ('D', '#009688'), 'E': ('v', '#3D51B5'), 'F': ('P', '#795548'),
        'G': ('X', '#FF9800'), 'H': ('p', '#607D8B'), 'I': ('h', '#4CAD50'),
        'J': ('d', '#00BCD4'),
    }

    def _plot(ax, cores, vals, combo, partner):
        m, c = combo_style[combo]
        ax.plot(cores, vals, f'{m}-', color=c, markersize=7, linewidth=1.5,
                label=f'Combo {combo} (vs {partner})')

    def _solo(ax, x, y):
        ax.scatter([x], [y], marker='*', s=200, color='gold',
                   edgecolors='black', linewidth=1, zorder=10, label='Solo')

    def _delta_box(ax, text, loc='lower right'):
        props = dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9)
        pos = {'lower right': (0.97, 0.05, 'right', 'bottom'),
               'upper right': (0.97, 0.95, 'right', 'top')}
        px, py, ha, va = pos[loc]
        ax.text(px, py, text, transform=ax.transAxes, fontsize=8,
                ha=ha, va=va, bbox=props)

    # ═══ Top-left: LLM CPU% ═══
    ax = axes[0, 0]
    s = solo[solo['vm_a'] == 'LLM'].iloc[0]
    _solo(ax, 10, s['a_cpu_pct'])

    b = conc[conc['combo'] == 'B'].sort_values('vm_b_cores')
    _plot(ax, b['vm_b_cores'], b['b_cpu_pct'], 'B', 'ResNet-CPU')

    f = conc[conc['combo'] == 'D'].sort_values('vm_a_cores')
    _plot(ax, f['vm_a_cores'], f['a_cpu_pct'], 'D', 'NodeJS')

    i_data = conc[conc['combo'] == 'F'].sort_values('vm_b_cores')
    _plot(ax, i_data['vm_b_cores'], i_data['b_cpu_pct'], 'F', 'ResNet-GPU')

    j = conc[conc['combo'] == 'G'].sort_values('vm_b_cores')
    _plot(ax, j['vm_b_cores'], j['b_cpu_pct'], 'G', 'ffmpeg')

    _delta_box(ax, 'max cross-combo\n@ same cores: 1.0%p')
    ax.set_title('LLM (Memory-bound)', fontweight='bold', color=WL_COLORS['LLM'])
    ax.set_xlabel('Allocated Cores')
    ax.set_ylabel('CPU Utilization (%)')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(alpha=0.3)
    ax.set_xlim(2, 18)
    ax.set_ylim(0, 55)

    # ═══ Top-middle: ResNet-CPU CPU% ═══
    ax = axes[0, 1]
    s = solo[solo['vm_a'] == 'ResNet-CPU'].iloc[0]
    _solo(ax, 10, s['a_cpu_pct'])

    a = conc[conc['combo'] == 'A'].sort_values('vm_a_cores')
    _plot(ax, a['vm_a_cores'], a['a_cpu_pct'], 'A', 'GEMM')

    b = conc[conc['combo'] == 'B'].sort_values('vm_a_cores')
    _plot(ax, b['vm_a_cores'], b['a_cpu_pct'], 'B', 'LLM')

    g = conc[conc['combo'] == 'E'].sort_values('vm_b_cores')
    _plot(ax, g['vm_b_cores'], g['b_cpu_pct'], 'E', 'ResNet-GPU')

    _delta_box(ax, 'max cross-combo\n@ same cores: 2.6%p')
    ax.set_title('ResNet-CPU (Compute-bound)', fontweight='bold', color=WL_COLORS['ResNet-CPU'])
    ax.set_xlabel('Allocated Cores')
    ax.set_ylabel('CPU Utilization (%)')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(alpha=0.3)
    ax.set_xlim(2, 18)
    ax.set_ylim(0, 55)

    # ═══ Top-right: ffmpeg CPU% (Phase 2) ═══
    ax = axes[0, 2]
    s = solo[solo['vm_a'] == 'ffmpeg'].iloc[0]
    _solo(ax, 10, s['a_cpu_pct'])

    j = conc[conc['combo'] == 'G'].sort_values('vm_a_cores')
    _plot(ax, j['vm_a_cores'], j['a_cpu_pct'], 'G', 'LLM')

    k = conc[conc['combo'] == 'H'].sort_values('vm_b_cores')
    _plot(ax, k['vm_b_cores'], k['b_cpu_pct'], 'H', 'Training')

    _delta_box(ax, 'max cross-combo\n@ 10 cores: 1.7%p')
    ax.set_title('ffmpeg (Compute-bound)', fontweight='bold', color=WL_COLORS['ffmpeg'])
    ax.set_xlabel('Allocated Cores')
    ax.set_ylabel('CPU Utilization (%)')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(alpha=0.3)
    ax.set_xlim(2, 18)
    ax.set_ylim(0, 62)

    # ═══ Bottom-left: GEMM GPU Power ═══
    ax = axes[1, 0]
    s = solo[solo['vm_a'] == 'GEMM'].iloc[0]
    _solo(ax, 10, s['gpu0_W'])

    a = conc[conc['combo'] == 'A'].sort_values('vm_b_cores')
    _plot(ax, a['vm_b_cores'], a['gpu1_W'], 'A', 'ResNet-CPU')

    c = conc[conc['combo'] == 'C'].sort_values('vm_a_cores')
    _plot(ax, c['vm_a_cores'], c['gpu0_W'], 'C', 'YOLO')

    l = conc[conc['combo'] == 'I'].sort_values('vm_b_cores')
    _plot(ax, l['vm_b_cores'], l['gpu1_W'], 'I', 'GPU-LLM')

    _delta_box(ax, 'max cross-combo\n@ 10 cores: 5.5W (2.4%)')
    ax.set_title('GEMM (GPU SM-heavy)', fontweight='bold', color=WL_COLORS['GEMM'])
    ax.set_xlabel('Allocated Cores')
    ax.set_ylabel('GPU Power (W)')
    ax.legend(fontsize=7, loc='lower right', bbox_to_anchor=(0.98, 0.25))
    ax.grid(alpha=0.3)
    ax.set_xlim(2, 18)
    ax.set_ylim(205, 235)

    # ═══ Bottom-middle: ResNet-GPU GPU Power ═══
    ax = axes[1, 1]
    s = solo[solo['vm_a'] == 'ResNet-GPU'].iloc[0]
    _solo(ax, 10, s['gpu0_W'])

    g = conc[conc['combo'] == 'E'].sort_values('vm_a_cores')
    _plot(ax, g['vm_a_cores'], g['gpu0_W'], 'E', 'ResNet-CPU')

    i_data = conc[conc['combo'] == 'F'].sort_values('vm_a_cores')
    _plot(ax, i_data['vm_a_cores'], i_data['gpu0_W'], 'F', 'LLM')

    _delta_box(ax, 'max cross-combo\n@ same cores: 6.6W (4.5%)')
    ax.set_title('ResNet-GPU (GPU-dominant)', fontweight='bold', color=WL_COLORS['ResNet-GPU'])
    ax.set_xlabel('Allocated Cores')
    ax.set_ylabel('GPU Power (W)')
    ax.legend(fontsize=7, loc='lower right', bbox_to_anchor=(0.98, 0.25))
    ax.grid(alpha=0.3)
    ax.set_xlim(9, 18)
    ax.set_ylim(135, 175)

    # ═══ Bottom-right: NodeJS CPU% (Phase 2) ═══
    ax = axes[1, 2]
    s = solo[solo['vm_a'] == 'NodeJS'].iloc[0]
    _solo(ax, 10, s['a_cpu_pct'])

    f = conc[conc['combo'] == 'D'].sort_values('vm_b_cores')
    _plot(ax, f['vm_b_cores'], f['b_cpu_pct'], 'D', 'LLM')

    m = conc[conc['combo'] == 'J'].sort_values('vm_b_cores')
    _plot(ax, m['vm_b_cores'], m['b_cpu_pct'], 'J', 'Training')

    _delta_box(ax, 'max cross-combo\n@ same cores: 0.05%p')
    ax.set_title('NodeJS (Lightweight)', fontweight='bold', color=WL_COLORS['NodeJS'])
    ax.set_xlabel('Allocated Cores')
    ax.set_ylabel('CPU Utilization (%)')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(alpha=0.3)
    ax.set_xlim(2, 18)
    ax.set_ylim(0, 10)

    fig.suptitle('Cross-Combo Verification: Same Workload, Different Partners',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'cross_combo_verification.png'))
    plt.close()
    print('[9/14] cross_combo_verification.png')


# =====================================================================
# Graph 10: IPC & LLC Cache Misses (perf stat 0225)
# =====================================================================
def plot_ipc_comparison():
    """v7: 9 workloads IPC + LLC cache misses."""
    if not os.path.exists(PERF_CSV):
        print('[10/14] SKIP: perf_stat_summary.csv not found')
        return

    perf = pd.read_csv(PERF_CSV)

    # Rename to display names
    name_map = {
        'resnet': 'ResNet-CPU', 'llm': 'LLM', 'nodejs': 'NodeJS',
        'resnet_gpu': 'ResNet-GPU', 'gemm': 'GEMM', 'yolo': 'YOLO',
        'training': 'Training', 'ffmpeg': 'ffmpeg', 'gpu_llm': 'GPU-LLM',
    }
    perf['display'] = perf['workload'].map(name_map)

    # v7: all 9 workloads
    order = [w for w in WL_ORDER_ALL if w in perf['display'].values]
    perf = perf.set_index('display').loc[order].reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5),
                                    gridspec_kw={'width_ratios': [1, 1.2]})

    x = np.arange(len(order))
    w = 0.55
    colors = [WL_COLORS.get(wl, '#888') for wl in order]

    # Left: IPC
    bars1 = ax1.bar(x, perf['ipc'], w, color=colors, edgecolor='white', linewidth=0.5)
    for i, (bar, val) in enumerate(zip(bars1, perf['ipc'])):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    gpu_wl = ['GEMM', 'GPU-LLM', 'Training', 'ResNet-GPU', 'YOLO']
    gpu_count = sum(1 for w in order if w in gpu_wl)
    sep_x = gpu_count - 0.5
    ax1.axvline(x=sep_x, color='black', linestyle='--', alpha=0.3, linewidth=1)
    ax1.text(sep_x / 2, 2.6, 'GPU-dominant', ha='center', fontsize=9, color='#D32D2F', fontstyle='italic')
    ax1.text((sep_x + len(order) - 1) / 2, 2.6, 'CPU-dominant', ha='center', fontsize=9, color='#1976D2', fontstyle='italic')

    ax1.set_xticks(x)
    ax1.set_xticklabels(order, fontsize=8, rotation=25, ha='right')
    ax1.set_ylabel('IPC (Instructions Per Cycle)')
    ax1.set_title('IPC by Workload (9 Workloads)')
    ax1.set_ylim(0, 2.8)
    ax1.grid(axis='y', alpha=0.3)

    # Right: LLC Cache Misses (millions)
    llc_misses_M = perf['llc_load_misses'] / 1e6
    bars2 = ax2.bar(x, llc_misses_M, w, color=colors, edgecolor='white', linewidth=0.5)
    for i, (bar, val) in enumerate(zip(bars2, llc_misses_M)):
        if val > 10:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                     f'{val:.0f}M', ha='center', va='bottom', fontsize=8, fontweight='bold')
        elif val > 0.1:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                     f'{val:.0f}M', ha='center', va='bottom', fontsize=7)
        else:
            ax2.text(bar.get_x() + bar.get_width()/2, 100,
                     f'{val:.1f}M', ha='center', va='bottom', fontsize=7)

    ax2.axvline(x=sep_x, color='black', linestyle='--', alpha=0.3, linewidth=1)

    ax2.set_xticks(x)
    ax2.set_xticklabels(order, fontsize=8, rotation=25, ha='right')
    ax2.set_ylabel('LLC Load Misses (millions)')
    ax2.set_title('LLC Cache Misses by Workload')
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle('IPC and Cache Behavior (perf stat, 60s, 9 Workloads)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'ipc_comparison.png'))
    plt.close()
    print('[10/14] ipc_comparison.png')


# =====================================================================
# Graph 11: Anchor-Spectrum Framing
# =====================================================================
def plot_anchor_spectrum():
    """Anchor (GEMM, ResNet-CPU, NodeJS) + spectrum (others) placement visualization."""
    solo = load_solo()
    idle = solo[solo['vm_a'] == 'idle'].iloc[0]

    idle_total = idle['rapl_W'] + idle['gpu0_W'] + idle['gpu1_W'] + idle['dram_W']

    workloads = [w for w in WL_ORDER_ALL if len(solo[solo['vm_a'] == w]) > 0]
    data = []
    for wl in workloads:
        row = solo[solo['vm_a'] == wl].iloc[0]
        active = (row['rapl_W'] + row['gpu0_W'] + row['gpu1_W'] + row['dram_W']) - idle_total
        gpu_frac = max(0, row['gpu0_W'] - idle['gpu0_W']) / max(active, 1) * 100
        data.append({'wl': wl, 'active': active, 'gpu_frac': gpu_frac, 'cpu_pct': row['a_cpu_pct']})

    anchors = ['GEMM', 'ResNet-CPU', 'NodeJS']

    fig, ax = plt.subplots(figsize=(10, 6))

    for d in data:
        color = WL_COLORS.get(d['wl'], '#888')
        marker = '*' if d['wl'] in anchors else 'o'
        size = 200 if d['wl'] in anchors else 100
        edge = 'black' if d['wl'] in anchors else 'white'
        ax.scatter(d['gpu_frac'], d['active'], c=color, s=size, marker=marker,
                   edgecolors=edge, linewidth=1.5, zorder=5)
        offset_x, offset_y = 2, 3
        if d['wl'] == 'NodeJS':
            offset_y = -8
        ax.annotate(d['wl'], (d['gpu_frac'], d['active']),
                    xytext=(offset_x, offset_y), textcoords='offset points',
                    fontsize=9, fontweight='bold' if d['wl'] in anchors else 'normal')

    ax.axvline(x=50, color='gray', linestyle='--', alpha=0.3)
    ax.text(75, max(d['active'] for d in data) * 0.9, 'GPU-dominant\nzone', fontsize=9,
            color='#D32D2F', fontstyle='italic', ha='center')
    ax.text(25, max(d['active'] for d in data) * 0.9, 'CPU-dominant\nzone', fontsize=9,
            color='#1976D2', fontstyle='italic', ha='center')

    ax.set_xlabel('GPU Fraction of Active Energy (%)')
    ax.set_ylabel('Active Energy (W)')
    ax.set_title('Anchor-Spectrum Framing: 9 Workloads')
    ax.grid(alpha=0.3)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gray', markersize=12,
               markeredgecolor='black', label='Anchor (calibration)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8,
               markeredgecolor='white', label='Spectrum (realistic)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'anchor_spectrum.png'))
    plt.close()
    print('[11/14] anchor_spectrum.png')


# =====================================================================
# Graph 12: GPU-dominant Internal Spectrum
# =====================================================================
def plot_gpu_spectrum():
    """GPU-dominant internal: SM-heavy(GEMM) -> Memory-heavy(GPU-LLM) -> Lightweight(YOLO) spectrum."""
    solo = load_solo()

    gpu_wl = ['GEMM', 'GPU-LLM', 'Training', 'ResNet-GPU', 'YOLO']
    gpu_wl = [w for w in gpu_wl if len(solo[solo['vm_a'] == w]) > 0]

    if len(gpu_wl) < 3:
        print('[12/14] SKIP: insufficient GPU workloads')
        return

    data = []
    for wl in gpu_wl:
        row = solo[solo['vm_a'] == wl].iloc[0]
        data.append({
            'wl': wl,
            'gpu0_W': row['gpu0_W'],
            'sm_pct': row['a_sm_pct'],
            'fb_mib': row['a_fb_mib'],
        })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(data))
    labels = [d['wl'] for d in data]
    colors = [WL_COLORS.get(d['wl'], '#888') for d in data]

    # Left: GPU Power + SM%
    bars = ax1.bar(x, [d['gpu0_W'] for d in data], 0.55, color=colors, edgecolor='white')
    for i, d in enumerate(data):
        ax1.text(x[i], d['gpu0_W'] + 2, f"{d['gpu0_W']:.0f}W\n({d['sm_pct']:.0f}% SM)",
                 ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel('GPU0 Power (W)')
    ax1.set_title('GPU Power & SM Utilization')
    ax1.grid(axis='y', alpha=0.3)

    # Right: GPU FB Memory
    bars2 = ax2.bar(x, [d['fb_mib'] for d in data], 0.55, color=colors, edgecolor='white')
    for i, d in enumerate(data):
        ax2.text(x[i], d['fb_mib'] + 10, f"{d['fb_mib']:.0f}MiB",
                 ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel('GPU Memory (MiB)')
    ax2.set_title('GPU Framebuffer Memory')
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle('GPU-dominant Internal Spectrum: SM-heavy to Lightweight', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'gpu_spectrum.png'))
    plt.close()
    print('[12/14] gpu_spectrum.png')


# =====================================================================
# Graph 13: CPU IPC Replication (ffmpeg confirms compute-bound)
# =====================================================================
def plot_cpu_ipc_replication():
    """CPU-dominant IPC comparison: ffmpeg(1.32) vs ResNet-CPU(1.75) vs LLM(0.48) -- n=3 reproducibility."""
    if not os.path.exists(PERF_CSV):
        print('[13/14] SKIP: perf_stat_summary.csv not found')
        return

    perf = pd.read_csv(PERF_CSV)
    name_map = {
        'resnet': 'ResNet-CPU', 'llm': 'LLM', 'nodejs': 'NodeJS',
        'ffmpeg': 'ffmpeg',
    }
    perf['display'] = perf['workload'].map(name_map)

    cpu_wl = ['ResNet-CPU', 'ffmpeg', 'LLM', 'NodeJS']
    perf_cpu = perf[perf['display'].isin(cpu_wl)].set_index('display').loc[cpu_wl].reset_index()

    solo = load_solo()
    rapl_data = {}
    for wl in cpu_wl:
        rows = solo[solo['vm_a'] == wl]
        if len(rows) > 0:
            rapl_data[wl] = rows.iloc[0]['rapl_W']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    x = np.arange(len(cpu_wl))
    colors = [WL_COLORS.get(wl, '#888') for wl in cpu_wl]
    w = 0.55

    # Left: IPC
    bars1 = ax1.bar(x, perf_cpu['ipc'], w, color=colors, edgecolor='white')
    for i, (bar, val) in enumerate(zip(bars1, perf_cpu['ipc'])):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Compute-bound zone
    ax1.axhspan(1.0, 2.0, color='#BBDEFB', alpha=0.2)
    ax1.text(1.5, 1.85, 'Compute-bound\n(IPC > 1.0)', fontsize=8, ha='center',
             color='#1565C0', fontstyle='italic')
    # Memory-bound zone
    ax1.axhspan(0, 0.7, color='#FFECB3', alpha=0.2)
    ax1.text(1.5, 0.15, 'Memory-bound\n(IPC < 0.7)', fontsize=8, ha='center',
             color='#E65100', fontstyle='italic')

    ax1.set_xticks(x)
    ax1.set_xticklabels(cpu_wl, fontsize=10)
    ax1.set_ylabel('IPC')
    ax1.set_title('IPC: CPU-dominant Workloads')
    ax1.set_ylim(0, 2.2)
    ax1.grid(axis='y', alpha=0.3)

    # Right: RAPL
    rapl_vals = [rapl_data.get(wl, 0) for wl in cpu_wl]
    bars2 = ax2.bar(x, rapl_vals, w, color=colors, edgecolor='white')
    for i, (bar, val) in enumerate(zip(bars2, rapl_vals)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.1f}W', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2.set_xticks(x)
    ax2.set_xticklabels(cpu_wl, fontsize=10)
    ax2.set_ylabel('RAPL Power (W)')
    ax2.set_title('CPU Power: Same CPU% -> Different RAPL')
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle('CPU IPC Replication: ffmpeg Confirms Compute-bound Pattern (n=3)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'cpu_ipc_replication.png'))
    plt.close()
    print('[13/14] cpu_ipc_replication.png')


# =====================================================================
# Graph 14: Training vs Inference
# =====================================================================
def plot_training_vs_inference():
    """Training(backward) vs ResNet-GPU(inference): GPU memory, energy, SM% comparison."""
    solo = load_solo()

    wl_list = ['Training', 'ResNet-GPU']
    rows = {}
    for wl in wl_list:
        r = solo[solo['vm_a'] == wl]
        if len(r) == 0:
            print('[14/14] SKIP: Training or ResNet-GPU solo data not found')
            return
        rows[wl] = r.iloc[0]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    colors = [WL_COLORS['Training'], WL_COLORS['ResNet-GPU']]

    # GPU Power
    ax = axes[0]
    vals = [rows['Training']['gpu0_W'], rows['ResNet-GPU']['gpu0_W']]
    bars = ax.bar(wl_list, vals, color=colors, width=0.5, edgecolor='white')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}W', ha='center', fontsize=11, fontweight='bold')
    ax.set_ylabel('GPU0 Power (W)')
    ax.set_title('GPU Power')
    ax.grid(axis='y', alpha=0.3)

    # GPU SM%
    ax = axes[1]
    vals = [rows['Training']['a_sm_pct'], rows['ResNet-GPU']['a_sm_pct']]
    bars = ax.bar(wl_list, vals, color=colors, width=0.5, edgecolor='white')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')
    ax.set_ylabel('GPU SM Utilization (%)')
    ax.set_title('SM Utilization')
    ax.grid(axis='y', alpha=0.3)

    # GPU FB Memory
    ax = axes[2]
    vals = [rows['Training']['a_fb_mib'], rows['ResNet-GPU']['a_fb_mib']]
    bars = ax.bar(wl_list, vals, color=colors, width=0.5, edgecolor='white')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{val:.0f}MiB', ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel('GPU FB Memory (MiB)')
    ax.set_title('GPU Memory')
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Training (backward) vs Inference (ResNet-GPU): Energy Structure',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'training_vs_inference.png'))
    plt.close()
    print('[14/14] training_vs_inference.png')


# =====================================================================
if __name__ == '__main__':
    print(f'Data: {SUMMARY}')
    print(f'Output: {OUT_DIR}')
    print()

    plot_solo_energy_profile()       # 1
    plot_cpu_scaling()               # 2
    plot_resnet_cpu_vs_gpu()         # 3
    plot_system_power_breakdown()    # 4
    plot_gpu_independence()          # 5
    plot_solo_vs_concurrent()        # 6
    plot_energy_asymmetry()          # 7
    plot_gpu_vs_cpu_contrast()       # 8
    plot_cross_combo_verification()  # 9
    plot_ipc_comparison()            # 10
    plot_anchor_spectrum()           # 11 (new)
    plot_gpu_spectrum()              # 12 (new)
    plot_cpu_ipc_replication()       # 13 (new)
    plot_training_vs_inference()     # 14 (new)

    print()
    print('Done! All 14 graphs saved.')
